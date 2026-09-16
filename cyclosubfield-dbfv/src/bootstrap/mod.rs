//! Bootstrapping for BFV/dBFV, following the sparse-key-encapsulation approach of the
//! reference implementation:
//!
//! 1. Move the slot values of the input ciphertext to the period-basis coefficients (homomorphic
//!    slots-to-coeffs transform, see [`homcrt`]), and recombine the dBFV digits into a single BFV
//!    ciphertext.
//! 2. Mod-switch it down to the small sparse-key-encapsulation RNS base and key-switch to an
//!    encapsulated sparse key; the ciphertext is now small enough that its decryption `c0 + c1
//!    s_sparse` can be evaluated homomorphically at the larger intermediate plaintext modulus,
//!    using the dBFV encryption of the sparse key.
//! 3. Trace down to `K`, decode `(c0, c1)` into plaintexts, and evaluate the noisy decryption
//!    homomorphically; add a rounding constant.
//! 4. Move the coefficients back into the slots (coeffs-to-slots), and remove the noise (the low
//!    bits) with homomorphic floor division (digit extraction, see [`digitex`]).

pub mod digitex;
pub mod galoisring;
pub mod homcrt;

use feanor_math::integer::{IntegerRingStore, int_cast};
use feanor_math::ring::RingStore;
use tracing::{debug, instrument};

use self::homcrt::{UnsplitCRTTransform, coeffs_to_slots_dbfv, slots_to_coeffs_dbfv};
use crate::bfv::*;
use crate::bootstrap::digitex::hom_floordiv_dbfv;
use crate::dbfv::*;
use crate::hal::{Backend, TensorMut, TensorOwned, TensorRef};
use crate::ring::{Basis, RingContext, RingEls, RingElsRef, ZZbig, ZZi128};

pub(crate) const RNS_FACTOR_SIZE_LOG2: f64 = 57.;

/// The keys needed to switch to (and homomorphically decrypt under) an encapsulated sparse
/// key.
pub struct SparseKeyEncaps<B: Backend> {
    /// key switching key from the main secret key to the sparse key, over the small
    /// sparse-key-encapsulation RNS base
    pub ksk: KeySwitchKey<B>,
    /// dBFV encryption of the sparse key under the main key, at the intermediate plaintext
    /// modulus, over the initial RNS base
    pub enc: DBFVCiphertext<B>,
}

/// Samples a sparse ternary key with hamming weight *exactly* `hwt` in the period basis of `K`,
/// and returns its period-basis coefficients over `L` (i.e. `deg_Lʼ` identical blocks, with the
/// sign corrected from the `M_K`-induced to the `M`-induced period basis).
///
/// The support is drawn as `hwt` *distinct* positions (colliding draws are retried) and every
/// position gets a nonzero value, so the result is uniform on the ternary vectors of weight
/// exactly `hwt`. That the weight cannot come out lower is what makes the rejection sampling of
/// [`create_sparse_key_encaps_with_bound()`] safe; see there.
#[instrument(skip_all)]
fn gen_sparse_sk_in_k<B: Backend, R: FnMut() -> u64>(ring: &RingContext<B>, hwt: usize, rng: &mut R) -> Vec<i128> {
    let deg_k = ring.field().deg_k();
    assert!(hwt <= deg_k);
    let mut sparse_sk = vec![0i128; deg_k];
    for _ in 0..hwt {
        loop {
            let j = rng() as usize % deg_k;
            if sparse_sk[j] == 0 {
                sparse_sk[j] = (rng() % 2) as i128 * 2 - 1;
                break;
            }
        }
    }
    let scale = ring.field().scale_m_k_to_m_period_basis() as i128;
    return (0..ring.field().deg_lprime()).flat_map(|_| sparse_sk.iter().map(|x| x * scale)).collect();
}

/// Creates the sparse key encapsulation: samples a sparse key in `K` (no rejection sampling),
/// encrypts it as a dBFV ciphertext at the intermediate dBFV modulus (the digit base of the
/// main modulus, at the intermediate plaintext modulus; see [`crate::dbfv::dbfv_enc()`]), and
/// generates the key switching key from the main key to it over the
/// sparse-key-encapsulation RNS base.
#[instrument(skip_all)]
pub fn create_sparse_key_encaps<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    intermediate_modulus: DBFVModulus,
    sk: &SecretKey<B>,
    hwt: usize,
    rng: &mut R,
    log2_alpha: f64,
) -> SparseKeyEncaps<B> {
    create_sparse_key_encaps_with_key(ring, intermediate_modulus, sk, hwt, rng, log2_alpha).0
}

/// The variant of [`create_sparse_key_encaps()`] that additionally returns the sampled sparse
/// key (in CRT basis over the initial RNS base), for debugging/noise measurements in tests.
#[instrument(skip_all)]
pub fn create_sparse_key_encaps_with_key<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    intermediate_modulus: DBFVModulus,
    sk: &SecretKey<B>,
    hwt: usize,
    rng: &mut R,
    log2_alpha: f64,
) -> (SparseKeyEncaps<B>, SecretKey<B>) {
    let sparse_sk_coeffs = gen_sparse_sk_in_k(ring, hwt, rng);
    return finish_sparse_key_encaps(ring, intermediate_modulus, sk, &sparse_sk_coeffs, rng, log2_alpha);
}

/// Creates the sparse key encapsulation as [`create_sparse_key_encaps_with_key()`], but rejects
/// sparse keys whose plaintext-noise amplification (see [`log2_sparse_key_amplification()`])
/// exceeds `max_log2_amplification`. Since that amplification is the plaintext noise `ν` which the
/// digit extraction of a bootstrap has to remove, this lets the bootstrap extract fewer digits (one
/// less, in practice; see the measurements in [`log2_sparse_key_amplification()`]).
///
/// Panics if no key passes within `max_attempts` attempts.
///
/// # Why this rejection sampling is safe
///
/// The hamming weight of a sampled key is *exactly* `hwt` — the support consists of `hwt` distinct
/// positions and every one of them gets a value in `{-1, 1}` (see [`gen_sparse_sk_in_k()`]) — so
/// the accepted key is uniform on `{ keys of weight exactly hwt : amplification <= bound }`, and
/// the conditioning cannot trade weight for a smaller norm.
///
/// This is the pitfall of the reference implementation, which draws the value of each of the `hwt`
/// chosen coefficients uniformly from `{-1, 0, 1}`: a third of them vanish, so the effective weight
/// is usually well below `hwt` and, in the tail, far below it. As the amplification grows with the
/// weight, rejection sampling then selects precisely those low-weight (i.e. weak) keys, which
/// undermines the sparse-key security assumption that the weight was chosen for.
///
/// What remains is the standard assumption of this kind of bootstrapping: that conditioning a
/// fixed-weight sparse key on a small norm does not weaken it. The conditioning is mild — at the
/// bound the tests use (one bit above the median amplification) most keys are accepted, and the
/// achievable gain is at most about one digit anyway.
#[instrument(skip_all)]
pub fn create_sparse_key_encaps_with_bound<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    intermediate_modulus: DBFVModulus,
    sk: &SecretKey<B>,
    hwt: usize,
    max_log2_amplification: f64,
    max_attempts: usize,
    rng: &mut R,
    log2_alpha: f64,
) -> (SparseKeyEncaps<B>, SecretKey<B>) {
    for attempt in 0..max_attempts {
        let sparse_sk_coeffs = gen_sparse_sk_in_k(ring, hwt, rng);
        let log2_amplification = log2_sparse_key_amplification(ring, &sparse_sk_coeffs, AMPLIFICATION_SAMPLES, rng);
        debug!("(sample_log2_amplification={})", log2_amplification);
        if log2_amplification <= max_log2_amplification {
            debug!("(total_attempts={})", attempt + 1);
            return finish_sparse_key_encaps(ring, intermediate_modulus, sk, &sparse_sk_coeffs, rng, log2_alpha);
        }
    }
    panic!(
        "no sparse key of hamming weight {} with log2 amplification <= {:.2} found in {} attempts",
        hwt, max_log2_amplification, max_attempts
    );
}

/// Encrypts the given sparse key (period-basis coefficients over `L`) under `sk` and generates the
/// key switching key to it, i.e. the shared part of the `create_sparse_key_encaps*` functions.
fn finish_sparse_key_encaps<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    intermediate_modulus: DBFVModulus,
    sk: &SecretKey<B>,
    sparse_sk_coeffs: &[i128],
    rng: &mut R,
    log2_alpha: f64,
) -> (SparseKeyEncaps<B>, SecretKey<B>) {
    // a proper dBFV encryption (a fresh zero-encryption per digit, followed by the dBFV
    // reduction), so that security follows from RLWE hardness at the improved
    // modulus-to-noise ratio; see [`dbfv_enc()`]
    let sparse_enc =
        dbfv_enc(ring, intermediate_modulus, &ring.initial_rns_base(), sparse_sk_coeffs, sk, rng, log2_alpha);

    let ske_base = ring.sparse_key_encaps_rns_base();
    let mut sparse_sk = ring.element_from_pb_coeffs(&ske_base, sparse_sk_coeffs);
    ring.pb_to_crt(&mut sparse_sk);
    let digits = (0..ske_base.len()).map(|i| i..i + 1).collect::<Vec<_>>();
    // over the (small) encapsulation base the requested ratio would demand a `σ` far below 1,
    // so this key gets the floor `MIN_SIGMA` — which is what sparse-key encapsulation needs
    // anyway, since its own ratio is much smaller than that of the main base
    let ksk = gen_ksk(ring, &ske_base, sk, &sparse_sk, &digits, rng, log2_alpha);

    let mut sparse_sk_full = ring.element_from_pb_coeffs(&ring.initial_rns_base(), sparse_sk_coeffs);
    ring.pb_to_crt(&mut sparse_sk_full);
    return (SparseKeyEncaps { ksk, enc: sparse_enc }, sparse_sk_full);
}

/// Replaces every element of the (CRT-basis) batch by its trace to `K`, i.e. the sum of its
/// `Gal(L|K)`-conjugates.
#[instrument(skip_all)]
pub fn trace_to_k<B: Backend>(ring: &RingContext<B>, els: &mut RingEls<B>) {
    assert_eq!(Basis::CRTBasis, els.basis);
    let mut rotated = ring.allocate_in_basis(els.count(), &els.rns_base, els.basis);
    let mut acc = ring.allocate_wide(els.count(), &els.rns_base);
    ring.add(&mut acc, &*els);
    for g in ring.field().lprime_q_transversal().into_iter().skip(1) {
        ring.apply_galois(&mut rotated, &*els, g);
        ring.add(&mut acc, &rotated);
    }
    ring.reduce_wide_into(els.as_mut(), &acc);
}

/// The number of rounding-error samples that [`log2_sparse_key_amplification()`] maximizes over
/// when it is used as a rejection-sampling criterion.
const AMPLIFICATION_SAMPLES: usize = 8;

/// An estimate of `log2 |r s_sparse|_pbinf` for a rounding-error element `r` with
/// `|r|_pbinf <= 1/2`, i.e. of the plaintext noise `ν` that a bootstrap using this sparse key has
/// to remove (see [`report_plaintext_noise()`]). A bootstrap must extract
/// `v >= log2_sparse_key_amplification(..) + 1` digits.
///
/// The estimate is computed on the host (the sparse key is known at key-generation time) by
/// multiplying the key with `samples` random rounding errors; the average over the samples (and
/// over the maximum of all coefficients) is returned. It is thus a sample average, not a proven
/// bound.
#[instrument(skip_all)]
pub fn log2_sparse_key_amplification<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    sparse_sk_coeffs: &[i128],
    samples: usize,
    rng: &mut R,
) -> f64 {
    // `r` is modelled scaled by 2^SCALE, so that its coefficients (uniform in [-1/2, 1/2]) can be
    // integers; the plaintext modulus must be large enough for the product to not wrap
    const SCALE: usize = 20;
    let log2_t = 2 * SCALE + 20;
    let t = 1i128 << log2_t;
    // the shortest prefix of the initial RNS base that can hold the (unreduced) product; using the
    // whole initial base would make the big-integer decoding needlessly expensive, which matters
    // when this is called in a rejection-sampling loop
    let required_bits = 2 * log2_t + ring.log2_pb_expansion_factor_bound();
    let mut base = Vec::new();
    while ZZbig.abs_log2_floor(&ring.modulus_of(&base)).unwrap_or(0) < required_bits {
        assert!(base.len() < ring.initial_rns_base_len(), "RNS modulus too small to measure the amplification");
        base.push(base.len());
    }

    let mut rhs = encode_pb_unscaled(ring, &base, t, sparse_sk_coeffs);
    ring.pb_to_crt(&mut rhs);

    let deg_k = ring.field().deg_k();
    let mut sum = 0i128;
    for _ in 0..samples {
        let r_k = (0..deg_k).map(|_| (rng() % (1 << SCALE)) as i128 - (1 << (SCALE - 1))).collect::<Vec<_>>();
        // after the trace the rounding error is K-valued, i.e. has deg_Lʼ equal blocks
        let r = (0..ring.field().deg_lprime()).flat_map(|_| r_k.iter().copied()).collect::<Vec<_>>();
        let mut lhs = encode_pb_unscaled(ring, &base, t, &r);
        ring.pb_to_crt(&mut lhs);
        let mut acc = ring.allocate_wide(1, &base);
        ring.fma(&mut acc, &lhs, &rhs);
        let mut prod = ring.reduce_wide(&acc, Basis::CRTBasis);
        ring.crt_to_pb(&mut prod);
        let values = decode_pb_unscaled(ring, t, &prod, 0);
        sum += values.iter().map(|x| x.abs()).max().unwrap();
    }
    return (sum as f64 / samples as f64).log2() - SCALE as f64;
}

/// Bootstraps a dBFV ciphertext (digits in CRT basis over the initial RNS base): returns an
/// encryption of the plaintext, with noise independent of the input noise.
///
/// - `crt_transform` must be created at the intermediate plaintext modulus, which must exceed
///   `modulus.t()` by enough bits to accommodate the decryption noise of a ciphertext over the
///   sparse-key-encapsulation base under the sparse key.
/// - `sparse_key_encaps` must be created at the intermediate dBFV modulus
///   `DBFVModulus::new_incomplete(modulus.log2_b(), log2_intermediate_t)`.
/// - `gks` must contain Galois keys for [`homcrt::required_galois_elements()`].
///
/// # Removing the factor `deg_Lʼ`
///
/// The trace step projects the decryption noise into `K`, so that the subsequent digit
/// extraction sees honest integers in the slots. On the `K`-valued message the `L|K`-trace acts
/// as `Tr_(Lʼ|Q)` and would multiply it by `[L : K] = [Lʼ : Q] = deg_Lʼ`. To avoid that factor,
/// both `(c0, c1)` components are multiplied by an `Lʼ`-element `α` with `Tr_(Lʼ|Q)(α) = 1`
/// before the trace (see [`FieldDescriptor::lprime_trace_one_coeffs()`]): since the message is
/// `K`-valued, `Tr_(Lʼ|Q)(α · message) = message`, and since `α ∈ Lʼ` while the sparse key lives
/// in `K`, `α` commutes with both the trace and the homomorphic decryption. The result is the
/// message unscaled.
#[instrument(skip_all)]
pub fn bootstrap_bfv_dbfv<B: Backend>(
    ring: &RingContext<B>,
    crt_transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    mut ct: DBFVCiphertext<B>,
    rk: &KeySwitchKey<B>,
    gks: &[(u64, KeySwitchKey<B>)],
    sparse_key_encaps: &SparseKeyEncaps<B>,
    debug_sk: Option<&SecretKey<B>>,
) -> DBFVCiphertext<B> {
    let base = ring.initial_rns_base();
    let intermediate_t = crt_transform.t();
    let log2_intermediate_t = ZZi128.abs_log2_ceil(&intermediate_t).unwrap() as usize;
    assert_eq!(1i128 << log2_intermediate_t, intermediate_t);

    // move slots to coefficients and recombine the digits into a single BFV ciphertext
    let reduced_transform = crt_transform.reduce_t(modulus.t());
    if ct.basis == Basis::CRTBasis {
        ring.crt_to_pb(&mut ct);
    }
    // the noise the Galois key switches inside the transform contribute, derived from the keys
    // themselves rather than assumed to be that of a single 57-bit gadget digit at the minimal
    // error: both the digit size and the key error vary (the latter follows the ciphertext
    // modulus, see the [`crate::bfv`] module documentation), and underestimating it here picks
    // too small a base and makes the bootstrap fail
    let gk_log2_noise = gks.iter().map(|(_, gk)| gk.log2_key_switch_noise(ring, &ct.rns_base)).fold(0., f64::max);
    // the field functions give a low estimate, so multiply by 2
    let slots_to_coeffs_noise =
        crt_transform.factors().len() as f64 * ring.field().noise_growth_hom_mul_plain_lower_estimate(modulus.b()) * 2.
            + gk_log2_noise
            + (modulus.t() as f64).log2();
    let slots_to_coeffs_rns_base_len = usize::min(
        ct.rns_base.len(),
        usize::max(
            (((modulus.t() as f64).log2() + slots_to_coeffs_noise) / RNS_FACTOR_SIZE_LOG2).ceil() as usize,
            ring.sparse_key_encaps_rns_base().len(),
        ),
    );
    debug!(
        "(slots_to_coeffs_rns_base_len={}, gk_log2_noise={:.1}, slots_to_coeffs_noise={:.1})",
        slots_to_coeffs_rns_base_len, gk_log2_noise, slots_to_coeffs_noise
    );
    let mut downscaled_ct = mod_switch_down(ring, &ct, slots_to_coeffs_rns_base_len);
    let slots_to_coeffs_rns_base = downscaled_ct.rns_base.to_vec();
    ring.pb_to_crt(&mut downscaled_ct);
    let in_coeffs = slots_to_coeffs_dbfv(ring, &reduced_transform, modulus, downscaled_ct, gks);
    let mut acc = ring.allocate_wide(2, &slots_to_coeffs_rns_base);
    for i in 0..dbfv_digit_count(&in_coeffs) {
        let factor = ring.scalar_tensor(&slots_to_coeffs_rns_base, 1i128 << (i * modulus.log2_b()));
        acc.fma_all(
            ring.backend(),
            in_coeffs.data.as_ref().slice(2 * i..2 * i + 2, .., .., ..),
            in_coeffs.bound,
            factor.as_ref(),
            1,
        );
    }
    let mut in_coeffs_bfv = ring.reduce_wide(&acc, Basis::CRTBasis);

    // mod-switch down to the sparse-key-encapsulation base and switch to the sparse key;
    // c0 is transformed to the CRT basis in place (the c1 row stays in period basis)
    ring.crt_to_pb(&mut in_coeffs_bfv);
    let mut switched = mod_switch_down(ring, &in_coeffs_bfv, ring.sparse_key_encaps_rns_base().len());
    ring.pb_to_crt_tensor(switched.data.as_mut().slice(0..1, .., .., ..), &switched.rns_base.clone());
    let c0 = RingElsRef {
        rns_base: &switched.rns_base,
        basis: Basis::CRTBasis,
        bound: switched.bound,
        data: switched.data.as_ref().slice(0..1, .., .., ..),
    };
    let c1 = RingElsRef {
        rns_base: &switched.rns_base,
        basis: Basis::PeriodBasis,
        bound: switched.bound,
        data: switched.data.as_ref().slice(1..2, .., .., ..),
    };
    let mut keyswitched = key_switch(ring, c0, c1, &sparse_key_encaps.ksk, None);

    // multiply both components by the Lʼ-correction α (trace one), so that the subsequent
    // L|K-trace recovers the K-valued message instead of scaling it by [Lʼ : Q] (see the doc)
    let alpha = ring.field().lprime_trace_one_coeffs();
    ring.mul_lprime_element(&mut keyswitched, &alpha);

    // trace down to K and decode the two components into plaintexts mod intermediate_t
    trace_to_k(ring, &mut keyswitched);
    ring.crt_to_pb(&mut keyswitched);
    let as_plain_0 = decode_pb_scaled(ring, intermediate_t, &keyswitched, 0);
    let as_plain_1 = decode_pb_scaled(ring, intermediate_t, &keyswitched, 1);

    // homomorphically evaluate the noisy decryption c0 + c1 * s_sparse, plus a constant that
    // turns the subsequent floor division into rounding
    let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), log2_intermediate_t);
    let empty = dbfv_empty(ring, &base, Basis::CRTBasis);
    let mut noisy_message = dbfv_hom_fma_plain(ring, intermediate_modulus, &sparse_key_encaps.enc, &as_plain_1, empty);
    let delta = ZZbig.rounded_div(ring.modulus_of(&base), &int_cast(intermediate_t, ZZbig, ZZi128));
    let mut constant = encode_pb_unscaled(ring, &base, intermediate_t, &as_plain_0);
    ring.pb_to_crt(&mut constant);
    let mut acc = ring.allocate_wide(1, &base);
    ring.fma_scalar_big(&mut acc, &constant, &delta);
    acc.add_all(ring.backend(), noisy_message.data.as_ref().slice(0..1, .., .., ..), noisy_message.bound);
    // no period-basis sign correction here, since we are in CRT basis
    let round_to_floor = ZZbig.mul_ref_snd(int_cast(intermediate_t / modulus.t() / 2, ZZbig, ZZi128), &delta);
    let round_tensor = ring.scalar_tensor_big(&base, &round_to_floor);
    acc.add_all(ring.backend(), round_tensor.as_ref(), 1);
    ring.reduce_wide_into(noisy_message.rows_mut(0..1), &acc);

    // move the coefficients back into the slots and floor-divide away the noise
    let mut in_slots = coeffs_to_slots_dbfv(ring, crt_transform, intermediate_modulus, noisy_message, gks);
    ring.crt_to_pb(&mut in_slots);
    let mut result = hom_floordiv_dbfv(ring, intermediate_modulus, modulus, in_slots, rk, debug_sk);
    ring.pb_to_crt(&mut result);
    return result;
}

#[cfg(test)]
mod tests {
    use self::homcrt::required_galois_elements;
    use super::*;
    use crate::field::{field_18, field_32, field_36864, field_49152};
    use crate::hal::cpu::CpuBackend;
    use crate::ring::cmod;

    #[test]
    fn test_bootstrap() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 30u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_18(), 800..820, 20);
        let t = 1i128 << 15;
        let modulus = DBFVModulus::new(5, 3);
        let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), 20);
        let intermediate_t = intermediate_modulus.t();
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field());
        let sparse = create_sparse_key_encaps(&ring, intermediate_modulus, &sk, 4, &mut rng, log2_alpha);
        let digits = select_digits(base.len(), base.len());
        let gks = required_galois_elements(ring.field(), &crt_transform)
            .into_iter()
            .map(|g| (g, gen_gk(&ring, &sk, g, &digits, &mut rng, log2_alpha)))
            .collect::<Vec<_>>();
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let ct = bfv_enc(&ring, &base, t, &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        let res = bootstrap_bfv_dbfv(&ring, &crt_transform, modulus, ct, &rk, &gks, &sparse, None);
        // the [Lʼ : Q] trace factor is cancelled by the α-correction; see [`bootstrap_bfv_dbfv()`]
        let expected = cmod(2025, t);
        assert_eq!(encode_scalar(&ring, expected), dbfv_dec(&ring, modulus, &res, &sk));
    }

    /// The same bootstrap over a field in which `(2)` is not totally inert in `Lʼ`, so that the
    /// CRT transform coefficients live in a product of two Galois rings rather than one (see
    /// [`crate::bootstrap::homcrt::CRTTransformFactor::new()`]).
    #[test]
    fn test_bootstrap_field_32() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 30u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_32(), 800..820, 20);
        assert_eq!(2, ring.field().lprime_embedding_transversal().len());
        let t = 1i128 << 15;
        let modulus = DBFVModulus::new(5, 3);
        let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), 20);
        let intermediate_t = intermediate_modulus.t();
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field());
        let sparse = create_sparse_key_encaps(&ring, intermediate_modulus, &sk, 2, &mut rng, log2_alpha);
        let digits = select_digits(base.len(), base.len());
        let gks = required_galois_elements(ring.field(), &crt_transform)
            .into_iter()
            .map(|g| (g, gen_gk(&ring, &sk, g, &digits, &mut rng, log2_alpha)))
            .collect::<Vec<_>>();
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let slot_values = vec![cmod(2025, t), cmod(-777, t)];
        let ptxt = crt_transform.reduce_t(t).pack_slot_values(&ring, &slot_values);
        let ct = bfv_enc(&ring, &base, t, &ptxt, &sk, &mut rng, log2_alpha);
        let res = bootstrap_bfv_dbfv(&ring, &crt_transform, modulus, ct, &rk, &gks, &sparse, None);
        assert_eq!(ptxt, dbfv_dec(&ring, modulus, &res, &sk));
    }

    /// The hamming weight of the sparse key of the full-size tests. Together with the
    /// amplification of the field (see `test_sparse_key_amplification()`) it fixes the number of
    /// digits the bootstrap has to extract.
    const TEST_SPARSE_SK_HWT: usize = 32;

    const SAMPLE_SPARSE_SK_ATTEMPTS: usize = 128;

    /// The full-size bootstrapping over `field_36864()`; very slow, hence ignored by default
    #[test]
    #[ignore]
    fn test_bootstrap_field_36864() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 31u64;
        let mut rng = move || splitmix64(&mut state);
        let log2_alpha = 960.;
        let r = 64;
        let v = 12;
        let ring = RingContext::new(CpuBackend, field_36864(), 960..980, r + v);
        let modulus = DBFVModulus::new(8, 8);
        let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), r + v);
        let intermediate_t = intermediate_modulus.t();
        let base = ring.initial_rns_base();
        let sk = gen_sk(&ring, &mut rng);
        let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field());
        let sparse = create_sparse_key_encaps_with_bound(
            &ring,
            intermediate_modulus,
            &sk,
            TEST_SPARSE_SK_HWT,
            v as f64 - 2.,
            SAMPLE_SPARSE_SK_ATTEMPTS,
            &mut rng,
            log2_alpha,
        )
        .0;
        let digits_gks = select_digits(base.len(), base.len());
        let gks = required_galois_elements(ring.field(), &crt_transform)
            .into_iter()
            .map(|g| (g, gen_gk(&ring, &sk, g, &digits_gks, &mut rng, log2_alpha)))
            .collect::<Vec<_>>();
        let digits_rk = select_digits((base.len() - 1) / 2 + 1, base.len());
        let rk = gen_rk(&ring, &sk, &digits_rk, &mut rng, log2_alpha);

        let ct = bfv_enc(&ring, &base, modulus.t(), &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        let res = bootstrap_bfv_dbfv(&ring, &crt_transform, modulus, ct, &rk, &gks, &sparse, None);
        println!("noise pbinf norm: {:?}", dbfv_log2_noise_pbinf_norm(&ring, modulus, &res, &sk));
        println!("remaining precision: {}", ZZbig.abs_log2_ceil(&ring.modulus_of(&res.rns_base)).unwrap());
        let slots =
            crt_transform.reduce_t(modulus.t()).extract_slot_values(&ring, &dbfv_dec(&ring, modulus, &res, &sk));
        assert_eq!((0..ring.slot_count()).map(|_| cmod(2025, modulus.t())).collect::<Vec<_>>(), slots);
    }

    /// The full-size bootstrapping over `field_49152()`; very slow, hence ignored by default
    #[test]
    #[ignore]
    fn test_bootstrap_field_49152() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 31u64;
        let mut rng = move || splitmix64(&mut state);
        let log2_alpha = 1280.;
        let r = 64;
        let v = 12;
        let ring = RingContext::new(CpuBackend, field_49152(), 1280..1300, r + v);
        let modulus = DBFVModulus::new(8, 8);
        let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), r + v);
        let intermediate_t = intermediate_modulus.t();
        let base = ring.initial_rns_base();
        let sk = gen_sk(&ring, &mut rng);
        let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field());
        let sparse = create_sparse_key_encaps_with_bound(
            &ring,
            intermediate_modulus,
            &sk,
            TEST_SPARSE_SK_HWT,
            v as f64 - 2.,
            SAMPLE_SPARSE_SK_ATTEMPTS,
            &mut rng,
            log2_alpha,
        )
        .0;
        let digits_gks = select_digits(base.len(), base.len());
        let gks = required_galois_elements(ring.field(), &crt_transform)
            .into_iter()
            .map(|g| (g, gen_gk(&ring, &sk, g, &digits_gks, &mut rng, log2_alpha)))
            .collect::<Vec<_>>();
        let digits_rk = select_digits((base.len() - 1) / 2 + 1, base.len());
        let rk = gen_rk(&ring, &sk, &digits_rk, &mut rng, log2_alpha);

        let ct = bfv_enc(&ring, &base, modulus.t(), &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        let res = bootstrap_bfv_dbfv(&ring, &crt_transform, modulus, ct, &rk, &gks, &sparse, None);
        println!("noise pbinf norm: {:?}", dbfv_log2_noise_pbinf_norm(&ring, modulus, &res, &sk));
        println!("remaining precision: {}", ZZbig.abs_log2_ceil(&ring.modulus_of(&res.rns_base)).unwrap());
        let slots =
            crt_transform.reduce_t(modulus.t()).extract_slot_values(&ring, &dbfv_dec(&ring, modulus, &res, &sk));
        assert_eq!((0..ring.slot_count()).map(|_| cmod(2025, modulus.t())).collect::<Vec<_>>(), slots);
    }
}
