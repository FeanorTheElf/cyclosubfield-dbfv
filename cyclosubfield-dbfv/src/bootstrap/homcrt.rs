//! The homomorphic CRT transform for `O_K/(2^e)`: the linear transform moving slot values to
//! period-basis coefficients (and back), used during bootstrapping.
//!
//! Since `(2)` doesn't split completely in `L`, the transform is significantly more
//! complicated than the mod-`p` CRT transforms of [`crate::ring`]: it decomposes into one
//! factor `u_p` per prime `p | m_K`, where each factor is a `J_p`-cyclic linear transform
//! ```text
//!   x  ->  Σ_i u_p[i] σ^i(x),    σ = generator of J_p = I_p / H_p
//! ```
//! whose coefficients `u_p[i]` are elements of `Lʼ` modulo `2^e`. The coefficients are found
//! by computing the transform in a Galois ring `GR(2^e, D)` and expressing the results w.r.t.
//! the period basis of `Lʼ` (a linear solve modulo `2^e`); see [`CRTTransformFactor::new()`].
//!
//! Homomorphically, each factor is evaluated with the baby-step-giant-step technique on dBFV
//! ciphertexts; see [`slots_to_coeffs_dbfv()`].

use feanor_math::integer::IntegerRingStore;
use tracing::instrument;

use crate::bfv::{KeySwitchKey, bfv_hom_gal};
use crate::bootstrap::galoisring::{GaloisRing, solve_mod_2e};
use crate::dbfv::{
    DBFVCiphertext, DBFVModulus, dbfv_acc_add, dbfv_acc_inner_product_element_lprime, dbfv_allocate_accumulator,
    dbfv_digit, dbfv_digit_count, dbfv_empty, dbfv_finish_accumulator, dbfv_hom_fma_element_lprime,
};
use crate::field::{FieldDescriptor, inv_mod};
use crate::hal::{Backend, TensorMut, TensorOwned};
use crate::ring::{Basis, RingContext, RingEls, ZZbig, ZZi128, cmod};

/// One factor `u_p` of the CRT transform for `O_K/(2^e)`; see the module documentation.
pub struct CRTTransformFactor {
    p: u64,
    t: i128,
    /// coefficients of the slots-to-coeffs factor, as elements of `Lʼ` given w.r.t. the
    /// period basis of `Lʼ` induced by `M` (i.e. already sign-corrected)
    u_p: Vec<Vec<i128>>,
    /// coefficients of the inverse (coeffs-to-slots) factor
    inv_u_p: Vec<Vec<i128>>,
    /// the generator of `J_p` (as an element of `(Z/mZ)*`) that the transform is cyclic over
    gen_jp: u64,
}

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize { if b == 0 { a } else { gcd(b, a % b) } }
    a / gcd(a, b) * b
}

impl CRTTransformFactor {
    /// Computes the factor `u_p` (and its inverse) for the prime `p | m_K`.
    ///
    /// # Expressing the coefficients in `Lʼ`
    ///
    /// The coefficients are Gauss sums, so they naturally live in the Galois ring
    /// `GR(2^e, D) = O_(Q(ζ_(p mʼ)))/(P^e)` for a prime `P | 2`, with
    /// `D = lcm(f_p, ord_mʼ(2))`. To use them as plaintext constants they have to be pulled
    /// back to `O_Lʼ/(2^e)`, which is done by solving for their coordinates w.r.t. the period
    /// basis of `Lʼ`.
    ///
    /// Reducing `O_Lʼ` modulo `P` is only one of the `[Lʼ : Q] / f_Lʼ` embeddings
    /// `O_Lʼ/(2^e) -> GR(2^e, D)`, one per prime of `Lʼ` above `2`. Solving against that one
    /// embedding alone would leave the coefficient's image at the other primes unconstrained —
    /// and those matter, because the plaintext ring `O_L/(2^e)` has slots above every prime of
    /// `Lʼ`. Since `Gal(L|Q)` is abelian and `σ` acts trivially on `Lʼ`, conjugating the
    /// transform identity by `Gal(L|Lʼ)` reproduces it at all slots that share a prime of `Lʼ`,
    /// but at no others; each prime of `Lʼ` must therefore see the same coefficient value. So
    /// the system is set up with one block of `D` rows per embedding and the same right-hand
    /// side in every block, which makes it square (`[Lʼ : Q]` unknowns, as many independent
    /// equations) and uniquely solvable. If `(2)` is totally inert in `Lʼ`, there is exactly
    /// one embedding and this degenerates to the single-block system.
    pub fn new(t: i128, p: u64, field: &FieldDescriptor) -> Self {
        assert!(t >= 2 && (t & (t - 1)) == 0, "t must be a power of two");
        let e = ZZi128.abs_log2_ceil(&t).unwrap() as usize;
        let m = field.m();
        let info = field.prime_info(p);
        let deg_lprime = field.deg_lprime();

        let d = lcm(info.f_p(), field.mprime_frobenius_order());
        let gr = GaloisRing::new(e, d);

        // the period basis of Lʼ (induced by M) under each embedding `O_Lʼ/(2^e) -> GR`, as
        // one block of `d` rows per prime of `Lʼ` above 2 (see above)
        let rou_mprime = gr.root_of_unity(field.m_prime() as usize);
        let scale = field.scale_mprime_to_m_period_basis() as i128;
        let embeddings = field.lprime_embedding_transversal();
        let rows = embeddings.len() * d;
        let mut matrix = vec![0i128; rows * deg_lprime];
        for (block, gamma) in embeddings.iter().enumerate() {
            for (j, g) in field.lprime_q_transversal().into_iter().enumerate() {
                let mut period = gr.zero();
                for h in field.gal_mprime_lprime_elements() {
                    let conjugate = (*gamma as u128 * g as u128 % m as u128) as u64;
                    let power = (conjugate as u128 * *h as u128 % m as u128) as u64 % field.m_prime();
                    period = gr.add(&period, &gr.pow(&rou_mprime, power as u128));
                }
                period = gr.mul_scalar(&period, scale);
                for i in 0..d {
                    matrix[(block * d + i) * deg_lprime + j] = period[i];
                }
            }
        }

        // the transform coefficients, as elements of GR
        let rou_p = gr.root_of_unity(p as usize);
        let gen_jp = info.i_p_gen();
        let gen_jp_inv = inv_mod(gen_jp, m);
        let jp_ord = info.j_p_order();
        let h_p_ord = info.h_p().len();
        let orbit_sum = |base: u64, i: usize, conjugate: bool| {
            let mut power_of_gen = 1u64;
            for _ in 0..i {
                power_of_gen = (power_of_gen as u128 * base as u128 % m as u128) as u64;
            }
            let step = if conjugate { m - 1 } else { 1 };
            let mut sum = gr.zero();
            for g in info.h_p() {
                let mut el = (*g as u128 * power_of_gen as u128 % m as u128) as u64;
                el = (el as u128 * step as u128 % m as u128) as u64;
                sum = gr.add(&sum, &gr.pow(&rou_p, (el % p) as u128));
            }
            sum
        };
        let p_inv = gr.inv(&gr.from_scalar(p as i128));
        // the same target value in every block, i.e. under every embedding of `Lʼ`
        let mut rhs = vec![0i128; rows * 2 * jp_ord];
        let mut set_rhs = |col: usize, coeff: &[i128]| {
            for block in 0..embeddings.len() {
                for row in 0..d {
                    rhs[(block * d + row) * 2 * jp_ord + col] = coeff[row];
                }
            }
        };
        for i in 0..jp_ord {
            // u_p[i] = Σ_(g in H_p) ζ_p^(g gen^-i)
            set_rhs(i, &orbit_sum(gen_jp_inv, i, false));
            // inv_u_p[i] = p^-1 (Σ_(g in H_p) ζ_p^(-g gen^i) - #H_p)
            let mut coeff = orbit_sum(gen_jp, i, true);
            coeff = gr.sub(&coeff, &gr.from_scalar(h_p_ord as i128));
            coeff = gr.mul(&coeff, &p_inv);
            set_rhs(jp_ord + i, &coeff);
        }

        // express the coefficients w.r.t. the period basis of Lʼ
        let solution = solve_mod_2e(e, rows, deg_lprime, 2 * jp_ord, &mut matrix, &mut rhs);
        let extract =
            |col: usize| (0..deg_lprime).map(|c| gr.smallest_lift(solution[c * 2 * jp_ord + col])).collect::<Vec<_>>();
        return Self {
            p,
            t,
            u_p: (0..jp_ord).map(extract).collect(),
            inv_u_p: (jp_ord..2 * jp_ord).map(extract).collect(),
            gen_jp,
        };
    }

    pub fn t(&self) -> i128 { self.t }

    pub fn u_p(&self) -> &[Vec<i128>] { &self.u_p }

    pub fn inv_u_p(&self) -> &[Vec<i128>] { &self.inv_u_p }

    pub fn gen_jp(&self) -> u64 { self.gen_jp }

    pub fn reduce_t(&self, new_t: i128) -> Self {
        assert!(self.t % new_t == 0);
        let reduce =
            |coeffs: &[Vec<i128>]| coeffs.iter().map(|coeff| coeff.iter().map(|c| cmod(*c, new_t)).collect()).collect();
        return Self {
            p: self.p,
            t: new_t,
            u_p: reduce(&self.u_p),
            inv_u_p: reduce(&self.inv_u_p),
            gen_jp: self.gen_jp,
        };
    }

    /// Computes the cost (in homomorphic Galois automorphisms) of evaluating this factor
    /// homomorphically. Concretely, the result is `(h, u)` for the **h**oisted and
    /// **u**nhoisted automorphism counts, respectively. Note that this will count any
    /// computation of l non-identity automorphisms on a single value as 1 unhoisted automorphism
    /// and l - 1 hoisted automorphisms.
    pub fn hom_cost(&self) -> (usize, usize) {
        let len = self.u_p.len();
        let (b, g) = bsgs_split(len);
        return (b.saturating_sub(2), g.saturating_sub(1) + if b > 1 { 1 } else { 0 });
    }
}

/// Applies the cyclic transform `x -> Σ_i coeffs[i] σ^i(x)` to a plaintext ring element batch
/// in CRT basis.
#[instrument(skip_all)]
fn apply_cyclic_transform_plain<B: Backend>(
    ring: &RingContext<B>,
    data: &RingEls<B>,
    automorphism: u64,
    coeffs: &[Vec<i128>],
) -> RingEls<B> {
    assert_eq!(Basis::CRTBasis, data.basis);
    let base = &data.rns_base;
    let count = data.count();
    let mut acc = ring.allocate_wide(count, base);
    let mut current = ring.clone_els(data);
    for (i, coeff) in coeffs.iter().enumerate() {
        let tensor = ring.lprime_element_tensor(base, coeff);
        ring.fma_tensor(&mut acc, &current, tensor.as_ref());
        if i + 1 < coeffs.len() {
            let mut next = ring.allocate_in_basis(count, base, Basis::CRTBasis);
            ring.apply_galois(&mut next, &current, automorphism);
            current = next;
        }
    }
    return ring.reduce_wide(&acc, Basis::CRTBasis);
}

/// Represents the full CRT transform for `O_K/(2^e)`, as the composition of the factors `u_p`
/// for all `p | m_K`.
pub struct UnsplitCRTTransform {
    t: i128,
    factors: Vec<CRTTransformFactor>,
}

impl UnsplitCRTTransform {
    pub fn new(t: i128, field: &FieldDescriptor) -> Self {
        let factors = field.m_k_factors().iter().map(|p| CRTTransformFactor::new(t, *p, field)).collect();
        return Self { t, factors };
    }

    pub fn t(&self) -> i128 { self.t }

    pub fn factors(&self) -> &[CRTTransformFactor] { &self.factors }

    pub fn reduce_t(&self, new_t: i128) -> Self {
        assert!(self.t % new_t == 0);
        Self { t: new_t, factors: self.factors.iter().map(|f| f.reduce_t(new_t)).collect() }
    }

    fn assert_no_overflow<B: Backend>(&self, ring: &RingContext<B>) {
        assert!(
            self.factors.len()
                * (2 * ZZi128.abs_log2_ceil(&self.t).unwrap() as usize + ring.log2_pb_expansion_factor_bound())
                < ZZbig.abs_log2_floor(&ring.modulus_of(&ring.initial_rns_base())).unwrap(),
            "RNS modulus too small to apply the CRT transform on plaintexts"
        );
    }

    /// Applies the (plaintext) slots-to-coeffs transform and reads off the slot values of a
    /// plaintext given by its period-basis coefficients (which must lie in `K`, i.e. have
    /// `deg_Lʼ` equal blocks).
    pub fn extract_slot_values<B: Backend>(&self, ring: &RingContext<B>, ptxt_pb: &[i128]) -> Vec<i128> {
        assert_eq!(ring.n(), ptxt_pb.len());
        let deg_k = ring.field().deg_k();
        for i in 1..ring.field().deg_lprime() {
            assert_eq!(&ptxt_pb[..deg_k], &ptxt_pb[i * deg_k..(i + 1) * deg_k], "plaintext is not contained in K");
        }
        self.assert_no_overflow(ring);
        let base = ring.initial_rns_base();
        let mut data = crate::bfv::encode_pb_unscaled(ring, &base, self.t, ptxt_pb);
        ring.pb_to_crt(&mut data);
        for factor in &self.factors {
            data = apply_cyclic_transform_plain(ring, &data, factor.gen_jp, &factor.u_p);
        }
        ring.crt_to_pb(&mut data);
        let mut result = crate::bfv::decode_pb_unscaled(ring, self.t, &data, 0);
        result.truncate(deg_k);
        let scale = ring.field().scale_m_k_to_m_period_basis() as i128;
        for x in &mut result {
            *x = cmod(*x * scale, self.t);
        }
        return result;
    }

    /// Inverse of [`UnsplitCRTTransform::extract_slot_values()`]: packs slot values into a
    /// plaintext in `K`, given by its period-basis coefficients.
    pub fn pack_slot_values<B: Backend>(&self, ring: &RingContext<B>, slot_values: &[i128]) -> Vec<i128> {
        let deg_k = ring.field().deg_k();
        assert_eq!(deg_k, slot_values.len());
        self.assert_no_overflow(ring);
        let base = ring.initial_rns_base();
        let repeated = (0..ring.field().deg_lprime()).flat_map(|_| slot_values.iter().copied()).collect::<Vec<_>>();
        let mut data = crate::bfv::encode_pb_unscaled(ring, &base, self.t, &repeated);
        ring.pb_to_crt(&mut data);
        for factor in &self.factors {
            data = apply_cyclic_transform_plain(ring, &data, factor.gen_jp, &factor.inv_u_p);
        }
        ring.crt_to_pb(&mut data);
        let scale = ring.field().scale_m_k_to_m_period_basis() as i128;
        return crate::bfv::decode_pb_unscaled(ring, self.t, &data, 0)
            .into_iter()
            .map(|x| cmod(x * scale, self.t))
            .collect();
    }

    /// Computes the cost (in homomorphic Galois automorphisms) of evaluating this transform
    /// homomorphically. Concretely, the result is `(h, u)` for the **h**oisted and
    /// **u**nhoisted automorphism counts, respectively. Note that this will count any
    /// computation of l non-identity automorphisms on a single value as 1 unhoisted automorphism
    /// and l - 1 hoisted automorphisms.
    pub fn hom_cost(&self) -> (usize, usize) {
        let (mut h, mut u) = (0, 0);
        for factor in self.factors() {
            h += factor.hom_cost().0;
            u += factor.hom_cost().1;
        }
        return (h, u);
    }
}

/// Cost of an unhoisted galois application, relative to a baby step, for choosing the BSGS
/// split.
///
/// This is deliberately *below* the measured cost ratio, which is ~9 on the full-size
/// parameter sets (an unhoisted key switch takes ~186 ms against ~20 ms for a hoisted one; the
/// difference is the `gadget_decompose` that all baby steps share). The split does not only
/// fix the running time, it also fixes how much memory the transform needs: both the number of
/// Galois keys and the `babys` buffer of [`hom_cyclic_transform_dbfv()`] grow linearly in the
/// baby step count, so the time-optimal split would need roughly three times the memory of the
/// value chosen here. The inner product over `babys` that dominates the dBFV transforms is
/// invariant under the split, which makes the trade cheap exactly where memory is scarce:
/// against the time-optimal choice this costs ~15% of the transform time on the large dBFV
/// parameter sets, and roughly halves the key material and the peak working set.
const UNHOISTED_OVERHEAD: usize = 3;

fn bsgs_split(len: usize) -> (usize, usize) {
    let baby_steps = (1..=len).min_by_key(|count| count + UNHOISTED_OVERHEAD * ((len - 1) / count + 1)).unwrap();
    let giant_steps = (len - 1) / baby_steps + 1;
    return (baby_steps, giant_steps);
}

fn pow_mod_m(g: u64, e: usize, m: u64) -> u64 {
    let mut result = 1u64;
    for _ in 0..e {
        result = (result as u128 * g as u128 % m as u128) as u64;
    }
    return result;
}

/// The Galois elements needed to homomorphically evaluate one transform factor.
fn factor_galois_elements(field: &FieldDescriptor, factor: &CRTTransformFactor) -> Vec<u64> {
    let len = factor.u_p.len();
    if len <= 1 {
        return Vec::new();
    }
    let (baby_steps, giant_steps) = bsgs_split(len);
    let mut result = (1..baby_steps).map(|j| pow_mod_m(factor.gen_jp, j, field.m())).collect::<Vec<_>>();
    if giant_steps > 1 {
        result.push(pow_mod_m(factor.gen_jp, baby_steps, field.m()));
    }
    return result;
}

/// All Galois elements needed to homomorphically evaluate the CRT transform (in either
/// direction); generate Galois keys for exactly these.
pub fn required_galois_elements(field: &FieldDescriptor, transform: &UnsplitCRTTransform) -> Vec<u64> {
    let mut result =
        transform.factors.iter().flat_map(|factor| factor_galois_elements(field, factor)).collect::<Vec<_>>();
    result.sort_unstable();
    result.dedup();
    return result;
}

/// Homomorphically applies the cyclic transform `Σ_i coeffs[i] σ^i` to a dBFV ciphertext (in
/// CRT basis) with baby-step-giant-step: the baby steps `σ^j(ct)` share the gadget
/// decomposition of `ct`, and each giant step applies one unhoisted automorphism to the
/// accumulated result.
#[instrument(skip_all)]
fn hom_cyclic_transform_dbfv<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    automorphism: u64,
    coeffs: &[Vec<i128>],
    ct: DBFVCiphertext<B>,
    gks: &[(u64, KeySwitchKey<B>)],
) -> DBFVCiphertext<B> {
    let len = coeffs.len();
    let base = ct.rns_base.to_vec();
    if len == 0 {
        return dbfv_empty(ring, &base, Basis::CRTBasis);
    } else if len == 1 {
        let summand = dbfv_empty(ring, &base, Basis::CRTBasis);
        return dbfv_hom_fma_element_lprime(ring, modulus, &ct, &coeffs[0], summand);
    }
    let (baby_steps, giant_steps) = bsgs_split(len);
    let m = ring.field().m();
    let d = dbfv_digit_count(&ct);

    // baby steps: babys[j] = σ^j(ct), sharing the gadget decomposition per digit
    let baby_exponents = (1..baby_steps).map(|j| pow_mod_m(automorphism, j, m)).collect::<Vec<_>>();
    let mut babys: Vec<DBFVCiphertext<B>> = Vec::with_capacity(baby_steps);
    babys.push(ct);
    for _ in 1..baby_steps {
        babys.push(ring.allocate_in_basis(2 * d, &base, Basis::CRTBasis));
    }
    for digit_idx in 0..d {
        let rotated = bfv_hom_gal(ring, dbfv_digit(&babys[0], digit_idx), &baby_exponents, gks);
        for (j, rot) in rotated.into_iter().enumerate() {
            ring.assign(
                babys[j + 1].data.as_mut().slice(2 * digit_idx..2 * digit_idx + 2, .., .., ..),
                rot.data.as_ref(),
            );
        }
    }

    // giant steps, last block first; all baby-step fmas of one block share a single
    // accumulator, so the modular reduction happens once per giant step
    let giant_exponent = pow_mod_m(automorphism, baby_steps, m);
    let mut out: DBFVCiphertext<B> = dbfv_empty(ring, &base, Basis::CRTBasis);
    for giant in (0..giant_steps).rev() {
        if giant != giant_steps - 1 {
            let out_digits = dbfv_digit_count(&out);
            let mut rotated = ring.allocate_in_basis(2 * out_digits, &base, Basis::CRTBasis);
            for digit_idx in 0..out_digits {
                let [r] =
                    bfv_hom_gal(ring, dbfv_digit(&out, digit_idx), &[giant_exponent], gks).try_into().ok().unwrap();
                ring.assign(rotated.data.as_mut().slice(2 * digit_idx..2 * digit_idx + 2, .., .., ..), r.data.as_ref());
            }
            out = rotated;
        }
        let mut acc = dbfv_allocate_accumulator(ring, modulus, &base, Basis::CRTBasis);
        // the baby-step inner product `Σ_j σ^j(ct) * coeffs_j` is one cache-tiled backend call
        // per output digit; it runs first (on the still-fresh accumulator, which the inner
        // product overwrites), then the carried-over `out` is added on top
        let terms = (0..usize::min(baby_steps, len - giant * baby_steps))
            .map(|j| (&babys[j], &coeffs[giant * baby_steps + j][..]))
            .collect::<Vec<_>>();
        dbfv_acc_inner_product_element_lprime(ring, &mut acc, &terms);
        dbfv_acc_add(ring, &mut acc, &out);
        out = dbfv_finish_accumulator(ring, &acc);
    }
    return out;
}

/// Homomorphically moves the slot values of a dBFV ciphertext (CRT basis) to the period-basis
/// coefficients of `K` (up to the sign
/// [`FieldDescriptor::scale_m_k_to_m_period_basis()`]).
#[instrument(skip_all)]
pub fn slots_to_coeffs_dbfv<B: Backend>(
    ring: &RingContext<B>,
    transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    mut ct: DBFVCiphertext<B>,
    gks: &[(u64, KeySwitchKey<B>)],
) -> DBFVCiphertext<B> {
    assert_eq!(modulus.t(), transform.t());
    for factor in &transform.factors {
        ct = hom_cyclic_transform_dbfv(ring, modulus, factor.gen_jp, &factor.u_p, ct, gks);
    }
    return ct;
}

/// Inverse of [`slots_to_coeffs_dbfv()`].
#[instrument(skip_all)]
pub fn coeffs_to_slots_dbfv<B: Backend>(
    ring: &RingContext<B>,
    transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    mut ct: DBFVCiphertext<B>,
    gks: &[(u64, KeySwitchKey<B>)],
) -> DBFVCiphertext<B> {
    assert_eq!(modulus.t(), transform.t());
    for factor in &transform.factors {
        ct = hom_cyclic_transform_dbfv(ring, modulus, factor.gen_jp, &factor.inv_u_p, ct, gks);
    }
    return ct;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfv::mul_plain_pb;
    use crate::field::{field_2, field_16, field_18, field_32, field_192, field_32768_ex};
    use crate::hal::cpu::CpuBackend;

    #[test]
    fn test_crt_transform_factor_field_2() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_2(), 100..120, 1);
        let factor = CRTTransformFactor::new(2, 7, ring.field());

        // since J_7 is cyclic, there is only one ordering of the period basis
        let base = ring.initial_rns_base();
        let mut data = crate::bfv::encode_pb_unscaled(&ring, &base, 2, &[0, 1]);
        ring.pb_to_crt(&mut data);
        let mut result = apply_cyclic_transform_plain(&ring, &data, factor.gen_jp, &factor.u_p);
        ring.crt_to_pb(&mut result);
        let result = crate::bfv::decode_pb_unscaled(&ring, 2, &result, 0);
        let nonzero = result.iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
        assert_eq!(nonzero, [1]);
    }

    #[test]
    fn test_crt_transform_factor_field_16() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_16(), 100..120, 1);
        let factor = CRTTransformFactor::new(2, 257, ring.field());

        let slot_unit_vector = vec![1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0];
        assert_eq!(slot_unit_vector, mul_plain_pb(&ring, 2, &slot_unit_vector, &slot_unit_vector));
        let base = ring.initial_rns_base();
        let mut data = crate::bfv::encode_pb_unscaled(&ring, &base, 2, &slot_unit_vector);
        ring.pb_to_crt(&mut data);
        let mut result = apply_cyclic_transform_plain(&ring, &data, factor.gen_jp, &factor.u_p);
        ring.crt_to_pb(&mut result);
        let result = crate::bfv::decode_pb_unscaled(&ring, 2, &result, 0);
        let nonzero = result.iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
        assert_eq!(nonzero, [1]);
    }

    #[test]
    fn test_crt_transform_factor_field_18() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let factor_7 = CRTTransformFactor::new(16, 7, ring.field());
        let factor_13 = CRTTransformFactor::new(16, 13, ring.field());

        let slot_unit_vector_in_k = [6, -3, -2, -2, -7, 7];
        let slot_unit_vector = (0..3).flat_map(|_| slot_unit_vector_in_k.iter().copied()).collect::<Vec<_>>();
        assert_eq!(slot_unit_vector, mul_plain_pb(&ring, 16, &slot_unit_vector, &slot_unit_vector));
        let base = ring.initial_rns_base();
        let mut data = crate::bfv::encode_pb_unscaled(&ring, &base, 16, &slot_unit_vector);
        ring.pb_to_crt(&mut data);
        let step_1 = apply_cyclic_transform_plain(&ring, &data, factor_7.gen_jp, &factor_7.u_p);
        let mut result = apply_cyclic_transform_plain(&ring, &step_1, factor_13.gen_jp, &factor_13.u_p);
        ring.crt_to_pb(&mut result);
        let result = crate::bfv::decode_pb_unscaled(&ring, 16, &result, 0);
        assert_eq!(&result[..6], &result[6..12]);
        assert_eq!(&result[..6], &result[12..]);
        let nonzero = result[..6].iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
        assert_eq!(vec![-1], nonzero);
    }

    #[test]
    fn test_crt_transform_field_18() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 200..220, 4);
        let transform = UnsplitCRTTransform::new(16, ring.field());

        let ptxt_slots = (0..6).map(|i| cmod(i, 16)).collect::<Vec<_>>();
        let ptxt_pb = transform.pack_slot_values(&ring, &ptxt_slots);
        let res_pb = mul_plain_pb(&ring, 16, &ptxt_pb, &ptxt_pb);
        let res_slots = transform.extract_slot_values(&ring, &res_pb);
        let expected = (0..6).map(|i| cmod(i * i, 16)).collect::<Vec<_>>();
        assert_eq!(expected, res_slots);
    }

    /// The round trip over a field in which `(2)` is not totally inert in `Lʼ`, so that the
    /// transform coefficients have to be pinned down at both primes of `Lʼ` above `2`.
    #[test]
    fn test_crt_transform_field_32() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_32(), 300..320, 4);
        assert_eq!(2, ring.field().lprime_embedding_transversal().len());
        let transform = UnsplitCRTTransform::new(16, ring.field());

        let ptxt1_slots = vec![cmod(5, 16), cmod(-3, 16)];
        let ptxt1_pb = transform.pack_slot_values(&ring, &ptxt1_slots);
        let ptxt2_slots = vec![cmod(7, 16), cmod(6, 16)];
        let ptxt2_pb = transform.pack_slot_values(&ring, &ptxt2_slots);
        let res_pb = mul_plain_pb(&ring, 16, &ptxt1_pb, &ptxt2_pb);
        let res_slots = transform.extract_slot_values(&ring, &res_pb);
        assert_eq!(vec![cmod(35, 16), cmod(-18, 16)], res_slots);
    }

    /// A factor and its inverse must compose to the identity on all of `O_L/(2^e)` — not just
    /// on the slots above one prime of `Lʼ`. On this field `Lʼ` has two primes above `2`, so
    /// coefficients that were only pinned down at one of them would fail here. Uses the cheap
    /// factor (`p = 3`, `#J_3 = 2`); the expensive `p = 65537` one is left to the bootstrapping
    /// tests.
    #[test]
    fn test_crt_transform_factor_field_32768_ex() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_32768_ex(), 300..320, 4);
        assert_eq!(2, ring.field().lprime_embedding_transversal().len());
        let factor = CRTTransformFactor::new(16, 3, ring.field());
        assert_eq!(2, factor.u_p().len());

        let ptxt = (0..ring.n()).map(|i| cmod(i as i128 * 7 + 3, 16)).collect::<Vec<_>>();
        let base = ring.initial_rns_base();
        let mut data = crate::bfv::encode_pb_unscaled(&ring, &base, 16, &ptxt);
        ring.pb_to_crt(&mut data);
        let forward = apply_cyclic_transform_plain(&ring, &data, factor.gen_jp, &factor.u_p);
        let mut back = apply_cyclic_transform_plain(&ring, &forward, factor.gen_jp, &factor.inv_u_p);
        ring.crt_to_pb(&mut back);
        assert_eq!(ptxt, crate::bfv::decode_pb_unscaled(&ring, 16, &back, 0));
    }

    #[test]
    fn test_crt_transform_field_192() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_192(), 300..320, 4);
        let transform = UnsplitCRTTransform::new(16, ring.field());

        let ptxt1_slots = (0..96).map(|i| cmod(i + 5, 16)).collect::<Vec<_>>();
        let ptxt1_pb = transform.pack_slot_values(&ring, &ptxt1_slots);
        let ptxt2_slots = (0..96).map(|i| cmod(i % 7 + 2, 16)).collect::<Vec<_>>();
        let ptxt2_pb = transform.pack_slot_values(&ring, &ptxt2_slots);
        let res_pb = mul_plain_pb(&ring, 16, &ptxt1_pb, &ptxt2_pb);
        let res_slots = transform.extract_slot_values(&ring, &res_pb);
        let expected = (0..96).map(|i| cmod((i + 5) * (i % 7 + 2), 16)).collect::<Vec<_>>();
        assert_eq!(expected, res_slots);
    }
}
