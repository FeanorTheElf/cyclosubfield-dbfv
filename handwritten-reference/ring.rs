use std::iter::repeat_n;
use std::ops::Range;

use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::prelude::*;
use feanor_math::ring_impls::float_complex::Complex64;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use feanor_math::ring_impls::zn::zn_rns::ZnRNS;
use feanor_math::ring_properties::integer::{BigIntRing, IntegerRingStore, ZZbig, ZZi64};
use feanor_math::ring_properties::pid::PrincipalIdealRingStore;
use feanor_math::seq::VectorView;
use tracing::instrument;

use crate::fft::ComplexCRTTransform;
use crate::fielddesc::*;
use crate::ntt::SplitCRTTransform;
use crate::{ZZi128, cmod, pmod};

/// the extended modulus is chosen to have at least `2 * log2(q) + log2(t) + EXTENDED_Q_EXTRA_BITS`;
/// therefore, for the moment, these extra bits have to account for the period-basis expansion
/// factor and also any sums/inner products performed in lifted crt form; in the future, at least
/// the period-basis expansion factor should be added explicitly, but for now, it is as it is.
const EXTENDED_Q_EXTRA_BITS: usize = 60;

fn largest_prime_leq_congruent_to_one(leq_than: i64, congruent_to_one_mod: i64) -> Option<i64> {
    assert!(leq_than > congruent_to_one_mod);
    let mut current = leq_than - (leq_than - 1) % congruent_to_one_mod;
    while !is_prime(&StaticRing::<i64>::RING, &current, 10) {
        current -= congruent_to_one_mod;
        if current <= 0 {
            return None;
        }
    }
    return Some(current);
}

fn extend_sampled_primes<F>(
    begin_with: &[i64],
    min_bits: usize,
    max_bits: usize,
    max_bits_each_modulus: usize,
    mut largest_prime_leq: F,
) -> Option<Vec<i64>>
where
    F: FnMut(i64) -> Option<i64>,
{
    assert!(max_bits > min_bits);

    let mut result = begin_with.iter().copied().collect::<Vec<_>>();
    let mut current_bits = result.iter().map(|i| (*i as f64).log2()).sum::<f64>();
    assert!((current_bits.floor() as usize) < max_bits);
    let mut current_upper_bound = ZZi64.power_of_two(max_bits_each_modulus);

    let min = |x, y| if x > y { y } else { x };

    while current_bits < min_bits as f64 {
        if min_bits as f64 - current_bits < max_bits_each_modulus as f64 {
            current_upper_bound = min(
                current_upper_bound,
                ZZi64.power_of_two(
                    f64::min(max_bits as f64 - current_bits, max_bits_each_modulus as f64).floor() as usize,
                ),
            );
        } else {
            let required_number_of_primes =
                ((min_bits as f64 - current_bits) / max_bits_each_modulus as f64).ceil() as usize;
            current_upper_bound = min(
                current_upper_bound,
                ZZi64.power_of_two(
                    f64::min(
                        (max_bits as f64 - current_bits) / required_number_of_primes as f64,
                        max_bits_each_modulus as f64,
                    )
                    .floor() as usize,
                ),
            );
        }

        let mut prime = largest_prime_leq(current_upper_bound)?;
        current_upper_bound = prime - 1;
        while begin_with.iter().any(|p| *p == prime) {
            prime = largest_prime_leq(current_upper_bound)?;
            current_upper_bound = prime - 1;
        }
        let bits = (prime as f64).log2();
        current_bits += bits;
        result.push(prime);
    }
    return Some(result);
}

pub struct RingContext {
    field_descriptor: FieldDescriptor,
    crt_transforms: Vec<SplitCRTTransform>,
    complex_crt_transform: ComplexCRTTransform,
    sparse_key_encaps_rns_base: ZnRNS<Zn64B, BigIntRing>,
    rns_base: ZnRNS<Zn64B, BigIntRing>,
    extended_rns_basis: ZnRNS<Zn64B, BigIntRing>,
}

/// Stores the data underlying a ring element. Intentionally a pure data
/// holder, without restricting the current representation of the data
/// (i.e. whether coefficients are w.r.t. double-CRT or period-basis)
#[derive(Clone)]
pub enum RingElementData<'a> {
    Owned(Vec<El<Zn64B>>),
    Borrowed(&'a [El<Zn64B>]),
}

impl RingContext {
    #[instrument(skip_all)]
    pub fn new(field_descriptor: FieldDescriptor, log2_q: Range<usize>, t_bits: usize) -> Self {
        let rou_order = field_descriptor.m() as i64
            * field_descriptor
                .GalLQ_cyclic_decomposition()
                .into_iter()
                .map(|(_, order)| order as i64)
                .fold(1, |x, y| ZZi64.lcm(&x, &y));
        let rns_base = extend_sampled_primes(&[], log2_q.start, log2_q.end, 57, |B| {
            largest_prime_leq_congruent_to_one(B, rou_order)
        })
        .unwrap();
        let extended_rns_base = extend_sampled_primes(
            &rns_base,
            2 * log2_q.end + EXTENDED_Q_EXTRA_BITS + t_bits,
            2 * log2_q.end + EXTENDED_Q_EXTRA_BITS + t_bits + 20,
            57,
            |B| largest_prime_leq_congruent_to_one(B, rou_order),
        )
        .unwrap();
        let rns_base = ZnRNS::new(rns_base.iter().map(|p| Zn64B::new(*p as u64)).collect(), ZZbig);
        let extended_rns_basis = ZnRNS::new(extended_rns_base.iter().map(|p| Zn64B::new(*p as u64)).collect(), ZZbig);
        let sparse_key_encaps_rns_base = ZnRNS::new(rns_base.as_iter().take(3).cloned().collect(), ZZbig);
        let crt_transforms = extended_rns_basis
            .as_iter()
            .map(|Zp| SplitCRTTransform::new(*Zp, &field_descriptor))
            .collect();
        let complex_crt_transform = ComplexCRTTransform::new(&field_descriptor);
        return Self {
            crt_transforms,
            field_descriptor,
            rns_base,
            extended_rns_basis,
            complex_crt_transform,
            sparse_key_encaps_rns_base,
        };
    }

    pub fn field_descriptor(&self) -> &FieldDescriptor { &self.field_descriptor }

    pub fn n(&self) -> usize { self.field_descriptor.deg_L() }

    pub fn slot_count(&self) -> usize { self.field_descriptor.deg_K() }

    pub fn crt_transforms(&self) -> &[SplitCRTTransform] { &self.crt_transforms }

    pub fn complex_crt_transform(&self) -> &ComplexCRTTransform { &self.complex_crt_transform }

    pub fn sparse_key_encaps_rns_base(&self) -> &ZnRNS<Zn64B, BigIntRing> { &self.sparse_key_encaps_rns_base }

    pub fn rns_base(&self) -> &ZnRNS<Zn64B, BigIntRing> { &self.rns_base }

    pub fn extended_rns_base(&self) -> &ZnRNS<Zn64B, BigIntRing> { &self.extended_rns_basis }

    /// Returns a bound on
    /// ```text
    ///   max_(x, y != 0) log2 |xy|_pb∞ - log2 |x|_pb∞ - log2 |y|_pb∞
    /// ```
    /// where `|x|_pb∞` is the "period-basis l_∞-norm", i.e. the largest magnitude among
    /// coefficients in the period-basis representation of x.
    pub fn log2_pb_expansion_factor_bound(&self) -> usize {
        ZZi64.abs_log2_ceil(&(self.field_descriptor.m() as i64)).unwrap() + self.field_descriptor.m_factors().len() + 1
    }

    /// since both period-basis and CRT-basis are normal bases, this function
    /// works on both representations!
    pub fn apply_galois(&self, dst: &mut [El<Zn64B>], src: &[El<Zn64B>], shift_gal: &[i64]) {
        let n = self.n();
        let used_rns_factors = dst.len() / n;
        assert_eq!(n * used_rns_factors, dst.len());
        assert_eq!(n * used_rns_factors, src.len());

        let cyclic_components = self.field_descriptor.GalLQ_cyclic_decomposition();
        let head_l = &cyclic_components[..(cyclic_components.len() - 1)];
        let indices_to_index = |indices: &[usize], shift: &[i64]| {
            indices
                .iter()
                .zip(shift.iter())
                .zip(head_l.iter())
                .fold(0, |current, ((idx, shift), (_, l))| {
                    pmod(*idx as i64 + shift, *l) + current * l
                })
        };

        assert_eq!(cyclic_components.len(), shift_gal.len());
        let last_l = cyclic_components.last().unwrap().1;
        let head_shifts = &shift_gal[..(cyclic_components.len() - 1)];
        let zero_shift = (0..(cyclic_components.len() - 1)).map(|_| 0).collect::<Vec<_>>();
        let mut perm_table = (0..(n / last_l)).map(|_| 0).collect::<Vec<_>>();
        for (src, dst) in multi_cartesian_product(
            head_l.iter().map(|(_, l)| 0..*l),
            |indices| {
                (
                    indices_to_index(indices, &zero_shift),
                    indices_to_index(indices, head_shifts),
                )
            },
            |_, x| *x,
        ) {
            perm_table[src] = dst;
        }

        let block_count = n / last_l;
        let last_s = pmod(*shift_gal.last().unwrap(), last_l);
        for i in 0..used_rns_factors {
            for j in 0..block_count {
                dst[(i * n + perm_table[j] * last_l)..(i * n + (perm_table[j] + 1) * last_l)]
                    .copy_from_slice(&src[(i * n + j * last_l)..(i * n + (j + 1) * last_l)]);
                dst[(i * n + perm_table[j] * last_l)..(i * n + (perm_table[j] + 1) * last_l)].rotate_right(last_s);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn pb_to_crt(&self, data: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = data.len() / n;
        assert_eq!(n * used_rns_factors, data.len());
        for i in 0..used_rns_factors {
            self.crt_transforms[i].crt_transform(&mut data[(i * n)..((i + 1) * n)]);
        }
    }

    #[instrument(skip_all)]
    pub fn dual_pb_to_crt(&self, data: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = data.len() / n;
        assert_eq!(n * used_rns_factors, data.len());
        for i in 0..used_rns_factors {
            self.crt_transforms[i].dual_crt_transform(&mut data[(i * n)..((i + 1) * n)]);
        }
    }

    #[instrument(skip_all)]
    pub fn crt_to_pb(&self, data: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = data.len() / n;
        assert_eq!(n * used_rns_factors, data.len());
        for i in 0..used_rns_factors {
            self.crt_transforms[i].inv_crt_transform(&mut data[(i * n)..((i + 1) * n)]);
        }
    }

    #[instrument(skip_all)]
    pub fn fma_crt(&self, lhs_crt: &[El<Zn64B>], rhs_crt: &[El<Zn64B>], dst_crt: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = rhs_crt.len() / n;
        assert_eq!(n * used_rns_factors, rhs_crt.len());
        assert_eq!(n * used_rns_factors, lhs_crt.len());
        assert_eq!(n * used_rns_factors, dst_crt.len());
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            for j in 0..n {
                dst_crt[i * n + j] = Zp.fma(&lhs_crt[i * n + j], &rhs_crt[i * n + j], dst_crt[i * n + j]);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn fma_scalar_large(&self, lhs: &[El<Zn64B>], scalar: &El<ZnRNS<Zn64B, BigIntRing>>, dst: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = lhs.len() / n;
        let rns_base = if used_rns_factors == self.sparse_key_encaps_rns_base.len() {
            self.sparse_key_encaps_rns_base()
        } else {
            self.rns_base()
        };
        assert_eq!(n * rns_base.len(), lhs.len());
        assert_eq!(n * rns_base.len(), dst.len());
        let scalar_wrt_congruence = self.rns_base.get_congruence(scalar);
        for (i, Zp) in rns_base.as_iter().enumerate() {
            let coeff = scalar_wrt_congruence.at(i);
            for j in 0..n {
                dst[i * n + j] = Zp.fma(&lhs[i * n + j], &coeff, dst[i * n + j]);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn fma_scalar(&self, lhs: &[El<Zn64B>], scalar: i128, dst: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = lhs.len() / n;
        assert_eq!(n * used_rns_factors, lhs.len());
        assert_eq!(n * used_rns_factors, dst.len());
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            let coeff = Zp.coerce(&ZZi128, scalar);
            for j in 0..n {
                dst[i * n + j] = Zp.fma(&lhs[i * n + j], &coeff, dst[i * n + j]);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn fma_element_Lʼ(&self, lhs_crt: &[El<Zn64B>], el: &[i128], dst_crt: &mut [El<Zn64B>]) {
        let deg_Lʼ = self.field_descriptor.deg_Lʼ();
        assert_eq!(deg_Lʼ, el.len());
        let n = self.n();
        let used_rns_factors = lhs_crt.len() / n;
        assert_eq!(n * used_rns_factors, lhs_crt.len());
        assert_eq!(n * used_rns_factors, dst_crt.len());
        let mut el_embedded = Vec::with_capacity(deg_Lʼ * used_rns_factors);
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            let hom = Zp.can_hom(&ZZi128).unwrap();
            el_embedded.extend(el.iter().map(|x| hom.map(*x)));
            self.crt_transforms[i].crt_transform_Lʼ(&mut el_embedded[(i * deg_Lʼ)..((i + 1) * deg_Lʼ)]);
        }
        let step = n / deg_Lʼ;
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            for k in 0..deg_Lʼ {
                let coeff = el_embedded[i * deg_Lʼ + k];
                for j in (k * step)..((k + 1) * step) {
                    dst_crt[i * n + j] = Zp.fma(&lhs_crt[i * n + j], &coeff, dst_crt[i * n + j]);
                }
            }
        }
    }

    #[instrument(skip_all)]
    pub fn mul_scalar(&self, dst: &mut [El<Zn64B>], scalar: i128) {
        let n = self.n();
        let used_rns_factors = dst.len() / n;
        assert_eq!(n * used_rns_factors, dst.len());
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            let coeff = Zp.coerce(&ZZi128, scalar);
            for j in 0..n {
                Zp.mul_assign(&mut dst[i * n + j], coeff);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn mul_scalar_large(&self, dst: &mut [El<Zn64B>], scalar: &El<ZnRNS<Zn64B, BigIntRing>>) {
        let n = self.n();
        assert_eq!(n * self.rns_base().len(), dst.len());
        let scalar_wrt_congruence = self.rns_base.get_congruence(scalar);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            let coeff = scalar_wrt_congruence.at(i);
            for j in 0..n {
                Zp.mul_assign(&mut dst[i * n + j], *coeff);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn add(&self, src: &[El<Zn64B>], dst: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = dst.len() / n;
        assert_eq!(n * used_rns_factors, dst.len());
        assert_eq!(n * used_rns_factors, src.len());
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            for j in 0..n {
                Zp.add_assign(&mut dst[i * n + j], src[i * n + j]);
            }
        }
    }

    /// Cdds a scalar component-wise to the given ring element.
    ///
    /// If the given coefficients are w.r.t. CRT basis, this just adds the scalar as a constant to
    /// the ring element. If the the given coefficients are w.r.t. period basis, this also adds the
    /// scalar as constant to the ring element, except that it is scaled by a factor of
    /// [`FieldDescriptor::scale_Q_to_M_period_basis()`].
    #[instrument(skip_all)]
    pub fn add_scalar_large(&self, scalar: &El<ZnRNS<Zn64B, BigIntRing>>, dst: &mut [El<Zn64B>]) {
        let n = self.n();
        assert_eq!(n * self.rns_base().len(), dst.len());
        let scalar_wrt_congruence = self.rns_base.get_congruence(scalar);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            let coeff = *scalar_wrt_congruence.at(i);
            for j in 0..n {
                Zp.add_assign(&mut dst[i * n + j], coeff);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn sub(&self, src: &[El<Zn64B>], dst: &mut [El<Zn64B>]) {
        let n = self.n();
        let used_rns_factors = dst.len() / n;
        assert_eq!(n * used_rns_factors, dst.len());
        assert_eq!(n * used_rns_factors, src.len());
        for (i, Zp) in self.extended_rns_base().as_iter().enumerate().take(used_rns_factors) {
            for j in 0..n {
                Zp.sub_assign(&mut dst[i * n + j], src[i * n + j]);
            }
        }
    }

    #[instrument(skip_all)]
    pub fn plaintext_into_ciphertext_ring_unscaled_pb(&self, t: i128, ptxt_pb: &[i128]) -> RingElementData<'static> {
        let n = self.n();
        assert_eq!(n, ptxt_pb.len());
        let mut res = self.init_zero();
        let res_data = res.borrow_mut();
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            let hom = Zp.can_hom(&ZZi128).unwrap();
            for j in 0..n {
                res_data[i * n + j] = hom.map(cmod(ptxt_pb[j], t));
            }
        }
        return res;
    }

    #[instrument(skip_all)]
    pub fn plaintext_from_ciphertext_ring_unscaled_pb(&self, t: i128, value_pb: RingElementData) -> Vec<i128> {
        let n = self.n();
        let used_rns_factors = value_pb.borrow().len() / n;
        let rns_base = if used_rns_factors == self.sparse_key_encaps_rns_base.len() {
            self.sparse_key_encaps_rns_base()
        } else {
            self.rns_base()
        };
        let t_bigint = int_cast(t, ZZbig, ZZi128);
        let extract_coeff = |j| {
            ZZbig.euclidean_rem(
                rns_base
                    .smallest_lift(rns_base.from_congruence((0..rns_base.len()).map(|i| value_pb.borrow()[i * n + j]))),
                &t_bigint,
            )
        };
        return (0..n)
            .map(|j| cmod(int_cast(extract_coeff(j), ZZi128, ZZbig), t))
            .collect();
    }

    #[instrument(skip_all)]
    pub fn plaintext_into_ciphertext_ring_scaled_pb(&self, t: i128, ptxt_pb: &[i128]) -> RingElementData<'static> {
        let mut embedded = self.plaintext_into_ciphertext_ring_unscaled_pb(t, ptxt_pb);
        let Delta = self.rns_base().coerce(
            &ZZbig,
            ZZbig.rounded_div(self.rns_base().modulus().clone(), &int_cast(t, ZZbig, ZZi128)),
        );
        self.mul_scalar_large(embedded.borrow_mut(), &Delta);
        return embedded;
    }

    #[instrument(skip_all)]
    pub fn plaintext_from_ciphertext_ring_scaled_pb(&self, t: i128, value_pb: RingElementData) -> Vec<i128> {
        let n = self.n();
        let used_rns_factors = value_pb.borrow().len() / n;
        let rns_base = if used_rns_factors == self.sparse_key_encaps_rns_base.len() {
            self.sparse_key_encaps_rns_base()
        } else {
            self.rns_base()
        };
        let t_bigint = int_cast(t, ZZbig, ZZi128);
        let extract_coeff = |j| {
            ZZbig.rounded_div(
                ZZbig.mul_ref_snd(
                    rns_base.smallest_lift(
                        rns_base.from_congruence((0..rns_base.len()).map(|i| value_pb.borrow()[i * n + j])),
                    ),
                    &t_bigint,
                ),
                rns_base.modulus(),
            )
        };
        return (0..n)
            .map(|j| cmod(int_cast(extract_coeff(j), ZZi128, ZZbig), t))
            .collect();
    }

    #[instrument(skip_all)]
    pub fn mul_plain_pb(&self, t: i128, lhs: &[i128], rhs: &[i128]) -> Vec<i128> {
        assert!(
            ZZbig.abs_log2_floor(self.rns_base().modulus()).unwrap()
                >= ZZi128.abs_log2_ceil(&t).unwrap() * 2 + self.log2_pb_expansion_factor_bound()
        );
        let mut lhs_q = self.plaintext_into_ciphertext_ring_unscaled_pb(t, lhs);
        let mut rhs_q = self.plaintext_into_ciphertext_ring_unscaled_pb(t, rhs);
        self.pb_to_crt(lhs_q.borrow_mut());
        self.pb_to_crt(rhs_q.borrow_mut());
        let mut res_q = self.init_zero();
        self.fma_crt(lhs_q.borrow_mut(), rhs_q.borrow_mut(), res_q.borrow_mut());
        self.crt_to_pb(res_q.borrow_mut());
        return self.plaintext_from_ciphertext_ring_unscaled_pb(t, res_q);
    }

    #[instrument(skip_all)]
    pub fn init_zero_with_len(&self, used_rns_factors: usize) -> RingElementData<'static> {
        let mut result = Vec::with_capacity(used_rns_factors * self.field_descriptor.deg_L());
        result.extend(
            self.extended_rns_base()
                .as_iter()
                .take(used_rns_factors)
                .flat_map(|Zp| repeat_n(Zp.zero(), self.field_descriptor.deg_L())),
        );
        RingElementData::Owned(result)
    }

    #[instrument(skip_all)]
    pub fn log2_pbinf_norm(&self, value_pb: &[El<Zn64B>]) -> f64 {
        let n = self.n();
        assert_eq!(n * self.rns_base.len(), value_pb.len());
        let mut current = 0.;
        for j in 0..n {
            let coeff = self.rns_base().smallest_lift(
                self.rns_base()
                    .from_congruence((0..self.rns_base().len()).map(|i| value_pb[i * n + j])),
            );
            current = f64::max(current, ZZbig.abs_highest_set_bit(&coeff).unwrap_or(0) as f64);
        }
        return current;
    }

    #[instrument(skip_all)]
    pub fn log2_caninf_norm(&self, value_pb: &[El<Zn64B>]) -> f64 {
        let n = self.n();
        assert_eq!(n * self.rns_base.len(), value_pb.len());
        let mut data = Vec::with_capacity(n);
        for j in 0..n {
            let coeff = self.rns_base().smallest_lift(
                self.rns_base()
                    .from_congruence((0..self.rns_base().len()).map(|i| value_pb[i * n + j])),
            );
            data.push(Complex64::RING.from_f64(ZZbig.to_float_approx(&coeff)));
        }
        self.complex_crt_transform.crt_transform(&mut data);
        return data
            .iter()
            .map(|x| Complex64::RING.abs(*x).log2())
            .max_by(f64::total_cmp)
            .unwrap();
    }

    #[instrument(skip_all)]
    pub fn log2_caninf_norm_small(&self, value_pb: &[El<Zn64B>]) -> f64 {
        let n = self.n();
        assert_eq!(n * self.rns_base.len(), value_pb.len());
        let mut data = Vec::with_capacity(n);
        for j in 0..n {
            let coeff = self.rns_base().at(0).smallest_lift(value_pb[j]);
            data.push(Complex64::RING.from_f64(coeff as f64));
        }
        self.complex_crt_transform.crt_transform(&mut data);
        return data
            .iter()
            .map(|x| Complex64::RING.abs(*x).log2())
            .max_by(f64::total_cmp)
            .unwrap();
    }

    pub fn init_zero(&self) -> RingElementData<'static> { self.init_zero_with_len(self.rns_base().len()) }

    pub fn init_zero_extended(&self) -> RingElementData<'static> {
        self.init_zero_with_len(self.extended_rns_base().len())
    }
}

impl<'a> RingElementData<'a> {
    pub fn borrow_mut<'b>(&'b mut self) -> &'b mut [El<Zn64B>] {
        if let RingElementData::Borrowed(data) = self {
            *self = RingElementData::Owned(data.to_vec());
        }
        match self {
            RingElementData::Borrowed(_) => unreachable!(),
            RingElementData::Owned(data) => data,
        }
    }

    pub fn borrow<'b>(&'b self) -> &'b [El<Zn64B>] {
        match self {
            RingElementData::Borrowed(data) => *data,
            RingElementData::Owned(data) => data,
        }
    }

    pub fn into_owned(self) -> Vec<El<Zn64B>> {
        match self {
            RingElementData::Borrowed(data) => data.to_vec(),
            RingElementData::Owned(data) => data,
        }
    }

    pub fn to_static(self) -> RingElementData<'static> { RingElementData::Owned(self.into_owned()) }
}

#[test]
fn test_apply_galois() {
    feanor_tracing::DelayedLogger::init_test();
    // group structure is 2 x 2 x 3 x 16
    let ring = RingContext::new(field_192(), 100..120, 4);
    assert_eq!(
        vec![2, 2, 3, 16],
        ring.field_descriptor()
            .GalLQ_cyclic_decomposition()
            .iter()
            .map(|(_, l)| *l)
            .collect::<Vec<_>>()
    );
    let Zp = ring.rns_base().at(0);

    let idx_to_value = |x: &[i64], shift: &[i64; 4]| {
        pmod(x[0] + shift[0], 2) * 96
            + pmod(x[1] + shift[1], 2) * 48
            + pmod(x[2] + shift[2], 3) * 16
            + pmod(x[3] + shift[3], 16)
    };
    let data = (0..ring.n()).map(|i| Zp.int_hom().map(i as i32)).collect::<Vec<_>>();
    let mut actual = (0..ring.n()).map(|_| Zp.zero()).collect::<Vec<_>>();

    ring.apply_galois(&mut actual, &data, &[0, 0, 0, 11]);
    let expected = multi_cartesian_product(
        [0..2, 0..2, 0..3, 0..16].into_iter(),
        |x| idx_to_value(x, &[0, 0, 0, -11]),
        |_, x| *x,
    )
    .map(|x| Zp.coerce(&ZZi64, x as i64))
    .collect::<Vec<_>>();
    for i in 0..ring.n() {
        assert_el_eq!(Zp, expected[i], actual[i]);
    }

    ring.apply_galois(&mut actual, &data, &[0, 0, 1, 0]);
    let expected = multi_cartesian_product(
        [0..2, 0..2, 0..3, 0..16].into_iter(),
        |x| idx_to_value(x, &[0, 0, -1, 0]),
        |_, x| *x,
    )
    .map(|x| Zp.coerce(&ZZi64, x as i64))
    .collect::<Vec<_>>();
    for i in 0..ring.n() {
        assert_el_eq!(Zp, expected[i], actual[i]);
    }

    ring.apply_galois(&mut actual, &data, &[0, 0, 1, -5]);
    let expected = multi_cartesian_product(
        [0..2, 0..2, 0..3, 0..16].into_iter(),
        |x| idx_to_value(x, &[0, 0, -1, 5]),
        |_, x| *x,
    )
    .map(|x| Zp.coerce(&ZZi64, x as i64))
    .collect::<Vec<_>>();
    for i in 0..ring.n() {
        assert_el_eq!(Zp, expected[i], actual[i]);
    }

    ring.apply_galois(&mut actual, &data, &[1, 1, 0, -2]);
    let expected = multi_cartesian_product(
        [0..2, 0..2, 0..3, 0..16].into_iter(),
        |x| idx_to_value(x, &[-1, -1, 0, 2]),
        |_, x| *x,
    )
    .map(|x| Zp.coerce(&ZZi64, x as i64))
    .collect::<Vec<_>>();
    for i in 0..ring.n() {
        assert_el_eq!(Zp, expected[i], actual[i]);
    }
}
