//! The BFV scheme on top of [`crate::ring`].
//!
//! A ciphertext is a [`RingEls`] batch of two elements `(c0, c1)` (batch dimension 0), such
//! that the plaintext is recovered as `round(t * (c0 + c1 * s) / Q) mod t`. The RNS base of a
//! ciphertext is an arbitrary prefix of the initial base `q` of the ring context, so
//! ciphertexts at reduced precision (obtained by [`mod_switch_down()`], "dynamic scaling") are
//! first-class: all operations derive the effective modulus from the ciphertext's own RNS
//! base, and restrict key switching keys to that base on the fly (as copy-free views).
//!
//! All randomness is drawn from a caller-provided `FnMut() -> u64`; errors are rounded
//! Gaussians sampled coefficient-wise in the period basis, and secret keys are sparse ternary
//! vectors in the period basis.
//!
//! # Noise parametrization
//!
//! Encryption and key generation are parametrized by `log2_alpha`, the binary logarithm of the
//! *modulus-to-noise ratio* `α = Q/σ` that the resulting object's security relies on, rather
//! than by `σ` directly; the error is derived as `σ = Q(base)/α` (see
//! [`sigma_for_log2_alpha()`]). Since `α` — not `σ` — is what determines RLWE hardness, this
//! keeps the security assumption fixed and explicit when the ciphertext modulus changes:
//! raising `Q` raises the error of every object automatically, instead of silently weakening
//! the assumption. It also lets [`crate::dbfv::dbfv_enc()`] express its weaker assumption
//! directly, and it accounts correctly for objects living over smaller bases (whose own `Q` is
//! smaller, so the same `α` demands less noise).

use std::ops::Range;

use feanor_math::integer::{BigIntRing, IntegerRingStore, int_cast};
use feanor_math::pid::EuclideanRingStore;
use feanor_math::ring::{El, RingStore};
use feanor_math::rings::zn::ZnRingStore;
use tracing::{instrument, warn};

use crate::hal::cpu::CpuBackend;
use crate::hal::{Backend, TensorI64Ref, TensorMut, TensorOwned, TensorRef};
use crate::keystore::KeyStore;
use crate::ring::{
    Basis, RNSBaseIndices, RingContext, RingEls, RingElsMut, RingElsRef, WideRingEls, ZZbig, ZZi128, cmod,
};

pub const SK_HWT: usize = 128;

/// A ciphertext `(c0, c1)` with `m ≈ (c0 + c1 s) t / Q`; a [`RingEls`] batch of size 2.
pub type BFVCiphertext<B> = RingEls<B>;

/// A secret key, as a single ring element in CRT basis over the initial RNS base.
pub type SecretKey<B> = RingEls<B>;

/// A key switching key: for each digit (a subrange of the positions of the RNS base), an
/// encryption of `old_sk * factor_digit` under the new key, where `factor_digit` is `1` modulo
/// the digit's primes and `0` modulo the others.
///
/// A key over the initial RNS base can directly switch ciphertexts that live over a prefix of
/// it: the key-switching functions restrict the stored keys to the ciphertext's base as
/// copy-free views (see [`clip_digits()`]).
pub struct KeySwitchKey<B: Backend> {
    pub digits: Vec<Range<usize>>,
    /// one size-2 batch per digit, in CRT basis
    pub keys: Vec<RingEls<B>>,
    /// the `log2` of the error standard deviation the key was generated with, recorded so that
    /// noise estimates can account for the actual key noise instead of assuming [`MIN_SIGMA`];
    /// see [`KeySwitchKey::log2_key_switch_noise()`]
    pub log2_sigma: f64,
}

impl<B: Backend> KeySwitchKey<B> {
    /// A (low) estimate of the `log2` absolute noise that key-switching with this key adds to a
    /// ciphertext over `base` (of which the key's own base must be an extension): the gadget
    /// product `Σ_i d_i ksk_i` has `|d_i| <= D_i` for the digit moduli `D_i`, so the error
    /// contributes `~ max_i D_i · σ`.
    ///
    /// Both factors have to be derived rather than assumed: `σ` follows the ciphertext modulus
    /// (see the module documentation), and `D_i` depends on how many RNS factors each gadget
    /// digit spans — a key with two primes per digit contributes `2^57` times more noise than one
    /// with a single prime. In the spirit of the other `*_lower_estimate` functions this leaves
    /// out the digit count and the period-basis expansion factor; callers add their own margin.
    pub fn log2_key_switch_noise(&self, ring: &RingContext<B>, base: &RNSBaseIndices) -> f64 {
        clip_digits(&self.digits, base.len())
            .into_iter()
            .map(|digit| ring.log2_modulus_of(&base[digit]))
            .fold(0., f64::max)
            + self.log2_sigma
    }

    /// The size of the stored key material in bytes.
    pub fn size_bytes(&self, ring: &RingContext<B>) -> usize {
        return self
            .keys
            .iter()
            .map(|key| key.count() * key.rns_base.len() * ring.n() * std::mem::size_of::<i64>())
            .sum();
    }
}

impl KeySwitchKey<CpuBackend> {
    /// Moves the key material into `store`, so that it occupies file-backed pages the kernel can
    /// reclaim instead of resident memory; see [`crate::keystore`]. The key stays usable exactly
    /// as before, and the pages are faulted back in from the file when it is next read.
    ///
    /// Offload the Galois keys of a linear transform, not the ciphertext buffers: the transform
    /// streams each key once per dBFV digit, but rereads its `babys` buffer once per giant step,
    /// which is only affordable from RAM.
    pub fn offload_to_disk(self, store: &KeyStore) -> std::io::Result<Self> {
        let Self { digits, keys, log2_sigma } = self;
        let keys = keys
            .into_iter()
            .map(|key| {
                let (rns_base, basis, bound) = (key.rns_base, key.basis, key.bound);
                let mut moved = RingEls::new(key.data.offload(store)?, basis, rns_base);
                moved.bound = bound;
                Ok(moved)
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        return Ok(Self { digits, keys, log2_sigma });
    }
}

/// Clips gadget digits to the length-`base_len` prefix of the RNS base: digits entirely
/// outside the prefix are dropped (they must form a suffix), the remaining ones are clipped.
pub fn clip_digits(digits: &[Range<usize>], base_len: usize) -> Vec<Range<usize>> {
    digits
        .iter()
        .filter(|digit| digit.start < base_len)
        .map(|digit| digit.start..usize::min(digit.end, base_len))
        .collect()
}

/// The error standard deviation below which no encryption is created, whatever the requested
/// modulus-to-noise ratio (see [`sigma_for_log2_alpha()`]).
///
/// The modulus-to-noise ratio generally determines security, but only as long as the noise
/// itself is a meaningful discrete Gaussian: once `σ` approaches 1 the absolute magnitude
/// matters too (at `σ = 0` an RLWE sample degenerates into a solvable linear system). Objects
/// over small RNS bases — e.g. the sparse-key-encapsulation key switching key, whose base holds
/// only a handful of primes — would otherwise be handed a `σ` far below 1. Note that such
/// objects do not *need* the ratio the large-base objects assume (sparse-key encapsulation
/// fundamentally relies on a much smaller ratio, because the key is very sparse); the floor
/// simply gives them the smallest sensible error instead of none.
pub const MIN_SIGMA: f64 = 3.2;

/// The largest `σ` that [`sample_gaussian()`] still samples on the full integer grid: the
/// rounded `f64` product `g * σ` has a spacing of `2^(log2(g σ) - 52)`, so beyond this the
/// samples start to skip integers. Exceeded only by extreme parameters; a warning is logged.
const MAX_EXACT_SIGMA_LOG2: f64 = 52.0;

/// The error standard deviation of an encryption over `base` that relies on RLWE hardness at
/// the modulus-to-noise ratio `α = 2^log2_alpha`, i.e. `σ = Q(base)/α`, floored at
/// [`MIN_SIGMA`]; see the module documentation.
pub fn sigma_for_log2_alpha<B: Backend>(ring: &RingContext<B>, base: &RNSBaseIndices, log2_alpha: f64) -> f64 {
    let log2_sigma = ring.log2_modulus_of(base) - log2_alpha;
    if log2_sigma > MAX_EXACT_SIGMA_LOG2 {
        // not a security condition, but a limit of the f64-based sampler
        warn!(
            "error sigma 2^{:.1} exceeds the largest exactly sampled value 2^{:.0}; the sampled \
             errors will skip integers",
            log2_sigma, MAX_EXACT_SIGMA_LOG2
        );
    }
    return f64::max(MIN_SIGMA, log2_sigma.exp2());
}

/// The modulus-to-noise ratio at which an encryption over `base` with error `sigma` is secure,
/// i.e. the inverse of [`sigma_for_log2_alpha()`] (ignoring its floor). Mainly useful to pin a
/// concrete `σ` in tests, and to report the assumption a given parameter choice makes.
pub fn log2_alpha_for_sigma<B: Backend>(ring: &RingContext<B>, base: &RNSBaseIndices, sigma: f64) -> f64 {
    ring.log2_modulus_of(base) - sigma.log2()
}

/// Samples a rounded Gaussian.
fn sample_gaussian<R: FnMut() -> u64>(rng: &mut R, sigma: f64) -> i128 {
    let u1 = ((rng() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = (rng() >> 11) as f64 / (1u64 << 53) as f64;
    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    return (g * sigma).round() as i128;
}

/// Generates a secret key: sparse ternary with hamming weight [`SK_HWT`] in the period basis,
/// returned in CRT basis over the initial RNS base.
#[instrument(skip_all)]
pub fn gen_sk<B: Backend, R: FnMut() -> u64>(ring: &RingContext<B>, rng: &mut R) -> SecretKey<B> {
    let n = ring.n();
    let mut coeffs = vec![0i128; n];
    for _ in 0..SK_HWT {
        let j = rng() as usize % n;
        coeffs[j] = (rng() % 2) as i128 * 2 - 1;
    }
    let mut sk = ring.element_from_pb_coeffs(&ring.initial_rns_base(), &coeffs);
    ring.pb_to_crt(&mut sk);
    return sk;
}

/// Adds a Gaussian error, sampled coefficient-wise in the period basis, to the (period-basis)
/// element.
#[instrument(skip_all)]
pub fn add_error<B: Backend, R: FnMut() -> u64>(ring: &RingContext<B>, els: &mut RingEls<B>, rng: &mut R, sigma: f64) {
    assert_eq!(Basis::PeriodBasis, els.basis);
    assert_eq!(1, els.count());
    let coeffs = (0..ring.n()).map(|_| sample_gaussian(rng, sigma)).collect::<Vec<_>>();
    let error = ring.element_from_pb_coeffs(&els.rns_base, &coeffs);
    let mut acc = ring.allocate_wide(1, &els.rns_base);
    ring.add(&mut acc, &*els);
    ring.add(&mut acc, &error);
    ring.reduce_wide_into(els.as_mut(), &acc);
}

/// A single uniformly random ring element in CRT basis.
fn random_element<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    rng: &mut R,
) -> RingEls<B> {
    let n = ring.n();
    let mut values = Vec::with_capacity(base.len() * n);
    for idx in base.iter() {
        let p = ring.rns_factor_prime(*idx);
        // montgomery representatives of uniform values are uniform
        values.extend((0..n).map(|_| (rng() % p as u64) as i64));
    }
    let mut result = ring.allocate_in_basis(1, base, Basis::CRTBasis);
    ring.backend().hal_import(result.data.as_mut(), &values);
    return result;
}

/// Encodes a plaintext (coefficients w.r.t. the period basis, modulo `t`) as a ring element,
/// without scaling.
pub fn encode_pb_unscaled<B: Backend>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    t: i128,
    ptxt_pb: &[i128],
) -> RingEls<B> {
    let coeffs = ptxt_pb.iter().map(|c| cmod(*c, t)).collect::<Vec<_>>();
    return ring.element_from_pb_coeffs(base, &coeffs);
}

/// Encodes a plaintext scaled by `Δ = round(Q/t)`.
pub fn encode_pb_scaled<B: Backend>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    t: i128,
    ptxt_pb: &[i128],
) -> RingEls<B> {
    let mut result = encode_pb_unscaled(ring, base, t, ptxt_pb);
    let delta = ZZbig.rounded_div(ring.modulus_of(base), &int_cast(t, ZZbig, ZZi128));
    ring.mul_scalar_big(&mut result, &delta);
    return result;
}

/// Decodes the `idx`-th element of a (period-basis) batch as an unscaled plaintext modulo `t`.
pub fn decode_pb_unscaled<B: Backend>(ring: &RingContext<B>, t: i128, els: &RingEls<B>, idx: usize) -> Vec<i128> {
    assert_eq!(Basis::PeriodBasis, els.basis);
    let t_big = int_cast(t, ZZbig, ZZi128);
    return ring
        .element_lifts(els, idx)
        .into_iter()
        .map(|lift| cmod(int_cast(ZZbig.euclidean_rem(lift, &t_big), ZZi128, ZZbig), t))
        .collect();
}

/// Decodes the `idx`-th element of a (period-basis) batch as a `Δ`-scaled plaintext modulo
/// `t`, i.e. computes `round(coeff * t / Q) mod t` per coefficient.
pub fn decode_pb_scaled<B: Backend>(ring: &RingContext<B>, t: i128, els: &RingEls<B>, idx: usize) -> Vec<i128> {
    assert_eq!(Basis::PeriodBasis, els.basis);
    let t_big = int_cast(t, ZZbig, ZZi128);
    let q = ring.modulus_of(&els.rns_base);
    return ring
        .element_lifts(els, idx)
        .into_iter()
        .map(|lift| cmod(int_cast(ZZbig.rounded_div(ZZbig.mul_ref_snd(lift, &t_big), &q), ZZi128, ZZbig), t))
        .collect();
}

/// Encrypts zero over the given RNS base; the secret key may live over a larger base with the
/// given base as prefix. The error is derived from the assumed modulus-to-noise ratio
/// `2^log2_alpha` over `base` (see [`sigma_for_log2_alpha()`]).
#[instrument(skip_all)]
pub fn bfv_enc_zero<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    sk: &SecretKey<B>,
    rng: &mut R,
    log2_alpha: f64,
) -> BFVCiphertext<B> {
    let mut b = ring.allocate(1, base);
    add_error(ring, &mut b, rng, sigma_for_log2_alpha(ring, base, log2_alpha));
    ring.pb_to_crt(&mut b);
    return finish_enc(ring, base, sk, b, rng);
}

/// Encrypts the given plaintext (period-basis coefficients modulo `t`). The error is derived
/// from the assumed modulus-to-noise ratio `2^log2_alpha` over `base` (see
/// [`sigma_for_log2_alpha()`]).
#[instrument(skip_all)]
pub fn bfv_enc<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    t: i128,
    ptxt_pb: &[i128],
    sk: &SecretKey<B>,
    rng: &mut R,
    log2_alpha: f64,
) -> BFVCiphertext<B> {
    let mut b = encode_pb_scaled(ring, base, t, ptxt_pb);
    add_error(ring, &mut b, rng, sigma_for_log2_alpha(ring, base, log2_alpha));
    ring.pb_to_crt(&mut b);
    return finish_enc(ring, base, sk, b, rng);
}

/// Given `b = Δm + e` in CRT basis, samples `a` and creates the ciphertext `(b + a s, -a)`.
fn finish_enc<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    sk: &SecretKey<B>,
    b: RingEls<B>,
    rng: &mut R,
) -> BFVCiphertext<B> {
    let mut a = random_element(ring, base, rng);
    let mut ct = ring.allocate_in_basis(2, base, Basis::CRTBasis);
    let mut acc = ring.allocate_wide(1, base);
    ring.fma(&mut acc, &a, sk.restrict_base(base.len()));
    ring.add(&mut acc, &b);
    ring.reduce_wide_into(ct.rows_mut(0..1), &acc);
    ring.mul_scalar(&mut a, -1);
    ring.assign(ct.data.as_mut().slice(1..2, .., .., ..), a.data.as_ref());
    return ct;
}

/// Computes the noisy scaled plaintext `c0 + c1 s` of a CRT-basis ciphertext, in period basis.
#[instrument(skip_all)]
fn noisy_plaintext<'a, B: Backend>(ring: &RingContext<B>, ct: RingElsRef<'a, B>, sk: &SecretKey<B>) -> RingEls<B>
where
    B: 'a,
{
    assert_eq!(Basis::CRTBasis, ct.basis);
    assert_eq!(2, ct.count());
    let base = ct.rns_base;
    let mut acc = ring.allocate_wide(1, base);
    ring.fma(&mut acc, ct.rows(1..2), sk.restrict_base(base.len()));
    ring.add(&mut acc, ct.rows(0..1));
    let mut noisy = ring.reduce_wide(&acc, Basis::CRTBasis);
    ring.crt_to_pb(&mut noisy);
    return noisy;
}

/// Decrypts a CRT-basis ciphertext.
#[instrument(skip_all)]
pub fn bfv_dec<'a, B: Backend>(
    ring: &RingContext<B>,
    t: i128,
    ct: impl Into<RingElsRef<'a, B>>,
    sk: &SecretKey<B>,
) -> Vec<i128>
where
    B: 'a,
{
    let noisy = noisy_plaintext(ring, ct.into(), sk);
    return decode_pb_scaled(ring, t, &noisy, 0);
}

/// The `log2` of the largest period-basis coefficient of the ciphertext noise.
pub fn bfv_log2_noise_pbinf_norm<'a, B: Backend>(
    ring: &RingContext<B>,
    t: i128,
    ct: impl Into<RingElsRef<'a, B>>,
    sk: &SecretKey<B>,
) -> f64
where
    B: 'a,
{
    let ct = ct.into();
    assert_eq!(Basis::CRTBasis, ct.basis);
    let noisy = noisy_plaintext(ring, ct, sk);
    let ptxt = decode_pb_scaled(ring, t, &noisy, 0);
    let ptxt_enc = encode_pb_scaled(ring, ct.rns_base, t, &ptxt);
    let mut acc = ring.allocate_wide(1, ct.rns_base);
    ring.add(&mut acc, &noisy);
    ring.sub(&mut acc, &ptxt_enc);
    let noise = ring.reduce_wide(&acc, Basis::PeriodBasis);
    return ring
        .element_lifts(&noise, 0)
        .iter()
        .map(|lift| ZZbig.abs_log2_ceil(lift).unwrap_or(0) as f64)
        .fold(0.0, f64::max);
}

/// Splits `0..rns_base_len` into `digits` contiguous ranges of (almost) equal length, for use
/// as gadget-product digits.
pub fn select_digits(digits: usize, rns_base_len: usize) -> Vec<Range<usize>> {
    assert!(digits <= rns_base_len, "the number of gadget product digits may not exceed the number of RNS factors");
    let moduli_per_small_digit = rns_base_len / digits;
    let large_digits = rns_base_len % digits;
    let mut result = Vec::with_capacity(digits);
    let mut current = 0;
    for i in 0..digits {
        let len = if i < large_digits { moduli_per_small_digit + 1 } else { moduli_per_small_digit };
        result.push(current..current + len);
        current += len;
    }
    return result;
}

/// Decomposes a single period-basis element into its gadget digits: the `i`-th output element
/// is `lift(x mod D_i) + ε D_i` over the full RNS base of `els`, where `D_i` is the product of
/// the primes of the `i`-th digit. Returns a batch of `digits.len()` elements, in CRT basis.
/// The base conversions of all digits are computed in a single sequence of backend calls; see
/// [`RingContext::bconv_gadget_tensors()`].
#[instrument(skip_all)]
pub fn gadget_decompose<'a, B: Backend>(
    ring: &RingContext<B>,
    els: RingElsRef<'a, B>,
    digits: &[Range<usize>],
) -> RingEls<B>
where
    B: 'a,
{
    assert_eq!(Basis::PeriodBasis, els.basis);
    assert_eq!(1, els.count());
    ring.assert_mul_mont_safe(els.bound, els.rns_base);
    let base = els.rns_base;
    let mut result = ring.allocate(digits.len(), base);
    // the batched base conversion destroys its input, so work on a copy
    let mut tmp = ring.clone_els(els);
    ring.bconv_gadget_tensors(result.data.as_mut(), tmp.data.as_mut(), base, digits);
    ring.pb_to_crt(&mut result);
    return result;
}

/// Generates a key switching key from `old_sk` to `new_sk` over the given base. The error of
/// every gadget component is derived from the assumed modulus-to-noise ratio `2^log2_alpha` over
/// `base` — i.e. a key over a smaller base gets correspondingly less noise, and raising the
/// ciphertext modulus raises the key noise automatically.
#[instrument(skip_all)]
pub fn gen_ksk<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    base: &RNSBaseIndices,
    old_sk: &RingEls<B>,
    new_sk: &SecretKey<B>,
    digits: &[Range<usize>],
    rng: &mut R,
    log2_alpha: f64,
) -> KeySwitchKey<B> {
    let zn_rns = ring.zn_rns_for(base);
    let log2_sigma = sigma_for_log2_alpha(ring, base, log2_alpha).log2();
    let mut keys = Vec::with_capacity(digits.len());
    for digit in digits {
        let mut ct = bfv_enc_zero(ring, base, new_sk, rng, log2_alpha);
        let factor: El<BigIntRing> = zn_rns.smallest_lift(zn_rns.from_congruence((0..base.len()).map(|j| {
            let zn = ring.rns_factor_zn(base[j]);
            if digit.contains(&j) { zn.one() } else { zn.zero() }
        })));
        let mut acc = ring.allocate_wide(1, base);
        ring.fma_scalar_big(&mut acc, old_sk.restrict_base(base.len()), &factor);
        ring.add(&mut acc, ct.rows(0..1));
        ring.reduce_wide_into(ct.rows_mut(0..1), &acc);
        keys.push(ct);
    }
    return KeySwitchKey { digits: digits.to_vec(), keys, log2_sigma };
}

/// Generates a relinearization key, i.e. a key switching key from `sk^2` to `sk`.
#[instrument(skip_all)]
pub fn gen_rk<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    sk: &SecretKey<B>,
    digits: &[Range<usize>],
    rng: &mut R,
    log2_alpha: f64,
) -> KeySwitchKey<B> {
    let base = ring.initial_rns_base();
    let mut acc = ring.allocate_wide(1, &base);
    ring.fma(&mut acc, sk, sk);
    let sk_sqr = ring.reduce_wide(&acc, Basis::CRTBasis);
    return gen_ksk(ring, &base, &sk_sqr, sk, digits, rng, log2_alpha);
}

/// Generates a Galois key for the automorphism `g in (Z/mZ)*`, i.e. a key switching key from
/// `g(sk)` to `sk`.
#[instrument(skip_all)]
pub fn gen_gk<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    sk: &SecretKey<B>,
    g: u64,
    digits: &[Range<usize>],
    rng: &mut R,
    log2_alpha: f64,
) -> KeySwitchKey<B> {
    let base = ring.initial_rns_base();
    let mut gal_sk = ring.allocate_in_basis(1, &base, Basis::CRTBasis);
    ring.apply_galois(&mut gal_sk, sk, g);
    return gen_ksk(ring, &base, &gal_sk, sk, digits, rng, log2_alpha);
}

/// Accumulates the inner product of the gadget decomposition of `c1_pb` with the key switching
/// key, i.e. `(Σ_i d_i ksk_i0, Σ_i d_i ksk_i1)`, into the rows `0..2` of `acc`. The key may
/// live over a larger RNS base than `c1_pb`, of which `c1_pb`'s base must be a prefix; the key
/// is restricted on the fly (as copy-free views).
#[instrument(skip_all)]
fn key_switch_acc<'a, B: Backend>(
    ring: &RingContext<B>,
    acc: &mut WideRingEls<B>,
    c1_pb: RingElsRef<'a, B>,
    ksk: &KeySwitchKey<B>,
) where
    B: 'a,
{
    assert_eq!(1, c1_pb.data.shape()[0]);
    assert_eq!(Basis::PeriodBasis, c1_pb.basis);

    let base_len = c1_pb.rns_base.len();
    assert_eq!(c1_pb.rns_base, &ksk.keys[0].rns_base[..base_len]);
    let digits = clip_digits(&ksk.digits, base_len);
    let parts = gadget_decompose(ring, c1_pb, &digits);
    // the inner product `acc[0..2] = Σ_i d_i * ksk_i` over all digits is a single, cache-tiled
    // backend call instead of one fma (i.e. one full pass over the accumulator) per digit
    let terms = ksk
        .keys
        .iter()
        .enumerate()
        .take(digits.len())
        .map(|(i, key)| {
            let key = key.restrict_base(base_len);
            (parts.data.as_ref().slice(i..i + 1, .., .., ..), parts.bound, key.data, key.bound)
        })
        .collect::<Vec<_>>();
    if !terms.is_empty() {
        acc.inner_product(ring.backend(), 0..2, &terms);
    }
}

/// Switches the ciphertext `(c0_crt, c1_pb)` to the key targeted by `ksk`; `summand1` (if
/// given) is added to the second output component.
#[instrument(skip_all)]
pub fn key_switch<'a, 'b, B: Backend>(
    ring: &RingContext<B>,
    c0_crt: impl Into<RingElsRef<'a, B>>,
    c1_pb: impl Into<RingElsRef<'b, B>>,
    ksk: &KeySwitchKey<B>,
    summand1: Option<RingElsRef<B>>,
) -> BFVCiphertext<B>
where
    B: 'a + 'b,
{
    let (c0_crt, c1_pb) = (c0_crt.into(), c1_pb.into());
    assert_eq!(1, c1_pb.data.shape()[0]);
    assert_eq!(Basis::PeriodBasis, c1_pb.basis);
    assert_eq!(1, c0_crt.data.shape()[0]);
    assert_eq!(Basis::CRTBasis, c0_crt.basis);

    let mut acc = ring.allocate_wide(2, c1_pb.rns_base);
    key_switch_acc(ring, &mut acc, c1_pb, ksk);
    acc.add(ring.backend(), 0..1, c0_crt.data, c0_crt.bound);
    if let Some(summand1) = summand1 {
        acc.add(ring.backend(), 1..2, summand1.data, summand1.bound);
    }
    return ring.reduce_wide(&acc, Basis::CRTBasis);
}

/// Lifts a batch of period-basis elements over some base `q'` to the base `q' + extension`,
/// in CRT basis; the extension is derived from `q'` (see
/// [`RingContext::extension_rns_base_for()`]), so elements that have been rescaled to a
/// smaller base are lifted to a smaller extended base. If the same elements are additionally
/// available in CRT basis, passing them as `crt` avoids transforming the `q'`-part.
#[instrument(skip_all)]
pub fn lift_to_extended<'a, B: Backend>(
    ring: &RingContext<B>,
    pb: impl Into<RingElsRef<'a, B>>,
    crt: Option<RingElsRef<B>>,
) -> RingEls<B>
where
    B: 'a,
{
    let pb = pb.into();
    assert_eq!(Basis::PeriodBasis, pb.basis);
    assert!(pb.bound <= 2, "lifting requires representatives bounded by 2p (basis transform input)");
    let base_len = pb.rns_base.len();
    let extension = ring.extension_rns_base_for(pb.rns_base);
    let lifted_base = pb.rns_base.iter().copied().chain(extension.iter().copied()).collect::<Vec<_>>();
    let mut result = ring.allocate(pb.count(), &lifted_base);

    // base conversion destroys its input, so work on a copy
    let mut tmp = ring.clone_els(pb);
    ring.bconv_tensors(result.data.as_mut().slice(.., base_len.., .., ..), &extension, tmp.data.as_mut(), pb.rns_base);
    if let Some(crt) = crt {
        assert_eq!(Basis::CRTBasis, crt.basis);
        assert_eq!(pb.rns_base, crt.rns_base);
        ring.pb_to_crt_tensor(result.data.as_mut().slice(.., base_len.., .., ..), &extension);
        ring.assign(result.data.as_mut().slice(.., ..base_len, .., ..), crt.data);
        result.bound = crt.bound;
    } else {
        ring.assign(result.data.as_mut().slice(.., ..base_len, .., ..), pb.data);
        ring.pb_to_crt_tensor(result.data.as_mut(), &lifted_base);
    }
    result.basis = Basis::CRTBasis;
    return result;
}

/// Computes `round(t x / q')` over the base `q'`, for a batch `x` of period-basis elements
/// over the base `q' + extension` (as produced by multiplying two [`lift_to_extended()`]
/// outputs); the first `base_len` factors of the input base form `q'`.
#[instrument(skip_all)]
pub fn rescale_to_base<B: Backend>(ring: &RingContext<B>, mut x: RingEls<B>, base_len: usize, t: i128) -> RingEls<B> {
    assert_eq!(Basis::PeriodBasis, x.basis);
    let count = x.count();
    let base = x.rns_base[..base_len].to_vec();
    let extension = x.rns_base[base_len..].to_vec();
    let q = ring.modulus_of(&base);
    ring.mul_scalar(&mut x, t);

    // subtract (t x mod q') and divide by q', all modulo the extension primes; the transient
    // unreduced difference is immediately reduced by the montgomery multiplication
    let mut tmp = ring.allocate(count, &extension);
    ring.bconv_tensors(tmp.data.as_mut(), &extension, x.data.as_mut().slice(.., ..base_len, .., ..), &base);
    ring.backend().hal_sub_i64(x.data.as_mut().slice(.., base_len.., .., ..), tmp.data.as_ref());
    let q_inv = ring.inverse_scalar_tensor(&extension, &q);
    let ext_mods = ring.mods_for(&extension);
    ring.backend().hal_mul_mont64(x.data.as_mut().slice(.., base_len.., .., ..), q_inv.as_ref(), &ext_mods);

    // convert the result (exact over the extension) back to q'
    let mut result = ring.allocate(count, &base);
    ring.bconv_tensors(result.data.as_mut(), &base, x.data.as_mut().slice(.., base_len.., .., ..), &extension);
    return result;
}

/// Mod-switches a batch of period-basis elements down to the length-`new_len` prefix of its
/// RNS base, i.e. computes `round(Q' x / Q)`. For BFV ciphertexts this realizes "dynamic
/// scaling": if the noise permits, computation can continue on the smaller ciphertext.
#[instrument(skip_all)]
pub fn mod_switch_down<'a, B: Backend>(
    ring: &RingContext<B>,
    els: impl Into<RingElsRef<'a, B>>,
    new_len: usize,
) -> RingEls<B>
where
    B: 'a,
{
    let els = els.into();
    assert_eq!(Basis::PeriodBasis, els.basis);
    assert!(new_len <= els.rns_base.len());
    let kept = els.rns_base[..new_len].to_vec();
    let dropped = els.rns_base[new_len..].to_vec();
    if dropped.is_empty() {
        return ring.clone_els(els);
    }

    // base conversion destroys its input, so work on a copy of the dropped part
    let mut dropped_part = ring.allocate(els.count(), &dropped);
    ring.assign(dropped_part.data.as_mut(), els.data.slice(.., new_len.., .., ..));
    let mut tmp = ring.allocate(els.count(), &kept);
    ring.bconv_tensors(tmp.data.as_mut(), &kept, dropped_part.data.as_mut(), &dropped);

    // (x mod Q' - bconv(dropped part)) * (Q/Q')^-1; the transient unreduced difference is
    // immediately reduced by the montgomery multiplication
    let mut result = ring.clone_els(els.restrict_base(new_len));
    ring.assert_mul_mont_safe(result.bound + 1, &kept);
    ring.backend().hal_sub_i64(result.data.as_mut(), tmp.data.as_ref());
    let dropped_modulus = ring.modulus_of(&dropped);
    let inv = ring.inverse_scalar_tensor(&kept, &dropped_modulus);
    let mods = ring.mods_for(&kept);
    ring.backend().hal_mul_mont64(result.data.as_mut(), inv.as_ref(), &mods);
    result.bound = 1;
    return result;
}

/// Computes the 3-component product `(l0 r0, l0 r1 + l1 r0, l1 r1)` of two lifted ciphertexts
/// (given as almost-reduced 2-row tensor slices in CRT basis) into the rows
/// `acc_row..acc_row + 3` of a wide batch; see [`RingContext::reduce_wide()`].
pub fn hom_mul_raw_acc<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut WideRingEls<B>,
    acc_row: usize,
    lhs: TensorI64Ref<B>,
    rhs: TensorI64Ref<B>,
) {
    assert_eq!(2, lhs.shape()[0]);
    assert_eq!(2, rhs.shape()[0]);
    let (l0, l1) = (lhs.slice(0..1, .., .., ..), lhs.slice(1..2, .., .., ..));
    let (r0, r1) = (rhs.slice(0..1, .., .., ..), rhs.slice(1..2, .., .., ..));
    let backend = ring.backend();
    acc.fma(backend, acc_row..acc_row + 1, l0, 1, r0, 1);
    acc.fma(backend, acc_row + 1..acc_row + 2, l0, 1, r1, 1);
    acc.fma(backend, acc_row + 1..acc_row + 2, l1, 1, r0, 1);
    acc.fma(backend, acc_row + 2..acc_row + 3, l1, 1, r1, 1);
}

/// Relinearizes a 3-component product (period basis, batch size 3) into a 2-component
/// ciphertext in the requested output basis. The best choice of `output_basis` depends on
/// what the next operation needs:
///  - [`Basis::CRTBasis`]: the `(c0, c1)` rows of `prod` are transformed to the CRT basis in place
///    and added to the (CRT-basis) key-switch accumulator, which is then reduced; the result is
///    almost-reduced.
///  - [`Basis::PeriodBasis`]: the key-switch accumulator is reduced to a period-basis ciphertext,
///    to which the `(c0, c1)` rows are added *without* reduction: the result has representatives
///    bounded by `2p` (bound 2). This costs nothing extra when the ciphertext is subsequently fed
///    into fma accumulation (inner products, [`crate::dbfv`] reduction) or a basis transform, all
///    of which accept bound-2 inputs.
///
/// The contents of `prod` should be considered destroyed afterwards.
#[instrument(skip_all)]
pub fn relin<'a, B: Backend>(
    ring: &RingContext<B>,
    prod: RingElsMut<'a, B>,
    rk: &KeySwitchKey<B>,
    output_basis: Basis,
) -> BFVCiphertext<B>
where
    B: 'a,
{
    assert_eq!(Basis::PeriodBasis, prod.basis);
    let mut data = prod.data;
    let base = prod.rns_base;
    assert_eq!(3, TensorMut::shape(&data)[0]);

    let mut acc = ring.allocate_wide(2, base);
    let c2 = RingElsRef {
        rns_base: base,
        basis: Basis::PeriodBasis,
        bound: prod.bound,
        data: data.as_const().slice(2..3, .., .., ..),
    };
    key_switch_acc(ring, &mut acc, c2, rk);
    match output_basis {
        Basis::CRTBasis => {
            assert!(prod.bound <= 2, "basis transforms require representatives bounded by 2p");
            ring.pb_to_crt_tensor(data.reborrow().slice(0..2, .., .., ..), base);
            acc.add(ring.backend(), 0..2, data.as_const().slice(0..2, .., .., ..), 1);
            return ring.reduce_wide(&acc, Basis::CRTBasis);
        }
        Basis::PeriodBasis => {
            let mut switched = ring.reduce_wide(&acc, Basis::CRTBasis);
            ring.crt_to_pb(&mut switched);
            let c0_c1 = RingElsRef {
                rns_base: base,
                basis: Basis::PeriodBasis,
                bound: prod.bound,
                data: data.as_const().slice(0..2, .., .., ..),
            };
            ring.add_els(&mut switched, c0_c1);
            return switched;
        }
    }
}

/// Homomorphically multiplies two CRT-basis ciphertexts over the same RNS base; `rk` may be
/// a relinearization key over any base extending it. The output is in `output_basis`; see
/// [`relin()`] for the trade-off (a period-basis output is unreduced, bound 2).
#[instrument(skip_all)]
pub fn bfv_hom_mul<B: Backend>(
    ring: &RingContext<B>,
    t: i128,
    lhs: &BFVCiphertext<B>,
    rhs: &BFVCiphertext<B>,
    rk: &KeySwitchKey<B>,
    output_basis: Basis,
) -> BFVCiphertext<B> {
    assert_eq!(lhs.rns_base, rhs.rns_base);
    assert_eq!(2, lhs.data.as_ref().shape()[0]);
    assert_eq!(2, rhs.data.as_ref().shape()[0]);
    assert_eq!(Basis::CRTBasis, lhs.basis);
    assert_eq!(Basis::CRTBasis, rhs.basis);

    let base_len = lhs.rns_base.len();
    let lift = |ct: &BFVCiphertext<B>| {
        let mut ct_pb = ring.clone_els(ct);
        ring.crt_to_pb(&mut ct_pb);
        lift_to_extended(ring, &ct_pb, Some(ct.as_ref()))
    };
    let lhs_lifted = lift(lhs);
    let rhs_lifted = lift(rhs);
    let mut acc = ring.allocate_wide(3, &lhs_lifted.rns_base);
    hom_mul_raw_acc(ring, &mut acc, 0, lhs_lifted.data.as_ref(), rhs_lifted.data.as_ref());
    let mut prod = ring.reduce_wide(&acc, Basis::CRTBasis);
    ring.crt_to_pb(&mut prod);
    let mut rescaled = rescale_to_base(ring, prod, base_len, t);
    return relin(ring, rescaled.as_mut(), rk, output_basis);
}

/// Homomorphically applies the Galois automorphisms `gs` to the ciphertext. The gadget
/// decomposition of `c1` is shared between all automorphisms, so applying several
/// automorphisms to the same ciphertext at once is cheaper than applying them separately.
/// The Galois keys may live over a larger RNS base than the ciphertext; they are restricted
/// on the fly.
#[instrument(skip_all)]
pub fn bfv_hom_gal<'a, B: Backend>(
    ring: &RingContext<B>,
    ct: impl Into<RingElsRef<'a, B>>,
    gs: &[u64],
    gks: &[(u64, KeySwitchKey<B>)],
) -> Vec<BFVCiphertext<B>>
where
    B: 'a,
{
    let ct = ct.into();
    assert_eq!(Basis::CRTBasis, ct.basis);
    assert_eq!(2, ct.data.shape()[0]);

    // no automorphism to apply: nothing is shared, and the gadget decomposition below would
    // have no key to read its digit layout from. A BSGS split with a single baby step (which
    // is what `bsgs_split()` chooses for very short transforms) hits this.
    if gs.is_empty() {
        return Vec::new();
    }

    let base = ct.rns_base;
    let base_len = base.len();
    let gk = |g: u64| &gks.iter().find(|(g_, _)| *g_ == g).expect("required galois key not present").1;

    // the basis transform needs to mutate, so c1 is copied
    let mut c1 = ring.clone_els(ct.rows(1..2));
    ring.crt_to_pb(&mut c1);
    let digits = clip_digits(&gk(gs[0]).digits, base_len);
    let parts = gadget_decompose(ring, c1.as_ref(), &digits);

    let mut results = Vec::with_capacity(gs.len());
    let mut gal_parts = ring.allocate_in_basis(parts.count(), base, Basis::CRTBasis);
    let mut gal_c0 = ring.allocate_in_basis(1, base, Basis::CRTBasis);
    for g in gs {
        let gk = gk(*g);
        assert!(digits == clip_digits(&gk.digits, base_len), "all galois keys must use the same digits");
        ring.apply_galois(&mut gal_parts, &parts, *g);
        ring.apply_galois(&mut gal_c0, ct.rows(0..1), *g);
        let mut acc = ring.allocate_wide(2, base);
        // the key-switch inner product `acc[0..2] = Σ_i d_i * gk_i` over all digits is a
        // single cache-tiled backend call instead of one fma per digit
        let terms = gk
            .keys
            .iter()
            .enumerate()
            .take(digits.len())
            .map(|(i, key)| {
                let key = key.restrict_base(base_len);
                (gal_parts.data.as_ref().slice(i..i + 1, .., .., ..), gal_parts.bound, key.data, key.bound)
            })
            .collect::<Vec<_>>();
        if !terms.is_empty() {
            acc.inner_product(ring.backend(), 0..2, &terms);
        }
        acc.add(ring.backend(), 0..1, gal_c0.data.as_ref(), gal_c0.bound);
        results.push(ring.reduce_wide(&acc, Basis::CRTBasis));
    }
    return results;
}

/// Multiplies two plaintexts as elements of `O_L / t` (both given by period-basis coefficients
/// modulo `t`); mainly useful to compute expected results in tests.
pub fn mul_plain_pb<B: Backend>(ring: &RingContext<B>, t: i128, lhs: &[i128], rhs: &[i128]) -> Vec<i128> {
    let base = ring.initial_rns_base();
    assert!(
        ZZbig.abs_log2_floor(&ring.modulus_of(&base)).unwrap()
            >= ZZi128.abs_log2_ceil(&t).unwrap() as usize * 2 + ring.log2_pb_expansion_factor_bound()
    );
    let mut lhs_el = encode_pb_unscaled(ring, &base, t, lhs);
    let mut rhs_el = encode_pb_unscaled(ring, &base, t, rhs);
    ring.pb_to_crt(&mut lhs_el);
    ring.pb_to_crt(&mut rhs_el);
    let mut acc = ring.allocate_wide(1, &base);
    ring.fma(&mut acc, &lhs_el, &rhs_el);
    let mut prod = ring.reduce_wide(&acc, Basis::CRTBasis);
    ring.crt_to_pb(&mut prod);
    return decode_pb_unscaled(ring, t, &prod, 0);
}

/// Encodes a scalar as a plaintext (a scalar has constant period-basis coefficients, up to the
/// sign [`crate::field::FieldDescriptor::scale_q_to_m_period_basis()`]).
pub fn encode_scalar<B: Backend>(ring: &RingContext<B>, x: i128) -> Vec<i128> {
    let scale = ring.field().scale_q_to_m_period_basis() as i128;
    (0..ring.n()).map(|_| x * scale).collect()
}

/// Inverse of [`encode_scalar()`].
pub fn decode_scalar<B: Backend>(ring: &RingContext<B>, t: i128, data_pb: &[i128]) -> i128 {
    let val = data_pb[0];
    for i in 1..ring.n() {
        assert_eq!(val, data_pb[i], "ring element doesn't store a scalar");
    }
    let scale = ring.field().scale_q_to_m_period_basis() as i128;
    return cmod(val * scale, t);
}

/// A simple splitmix64 rng for tests and non-cryptographic sampling.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{field_18, field_192};
    use crate::hal::cpu::CpuBackend;
    use crate::ring::RingContext;

    #[test]
    fn test_enc_dec() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 1u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 255) - 127).collect();
        let sk = gen_sk(&ring, &mut rng);
        let ct = bfv_enc(&ring, &base, 256, &m, &sk, &mut rng, log2_alpha);
        assert_eq!(m, bfv_dec(&ring, 256, &ct, &sk));

        let m: Vec<i128> = (0..ring.n()).map(|_| 2).collect();
        let ct = bfv_enc(&ring, &base, 4, &m, &sk, &mut rng, log2_alpha);
        assert_eq!(m, bfv_dec(&ring, 4, &ct, &sk));
    }

    /// The key-switching noise estimate must follow both the key error and the gadget digit
    /// size. Assuming a single 57-bit digit at the minimal error — as the bootstrap's
    /// slots-to-coeffs base estimate used to — underestimates it by the full amount by which the
    /// key error exceeds [`MIN_SIGMA`], which is exactly what happens once the ciphertext modulus
    /// grows.
    #[test]
    fn test_key_switch_noise_follows_key() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 58u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 8);
        let base = ring.initial_rns_base();
        let (sk, sk2) = (gen_sk(&ring, &mut rng), gen_sk(&ring, &mut rng));
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);

        // one RNS factor per gadget digit, at the minimal error
        let one_prime = select_digits(base.len(), base.len());
        let ksk = gen_ksk(&ring, &base, &sk, &sk2, &one_prime, &mut rng, log2_alpha);
        let max_prime_bits = (0..base.len()).map(|i| ring.log2_modulus_of(&base[i..i + 1])).fold(0., f64::max);
        assert!((ksk.log2_key_switch_noise(&ring, &base) - (max_prime_bits + MIN_SIGMA.log2())).abs() < 1e-9);

        // a key generated at a 48 bits weaker assumption — as required when the ciphertext
        // modulus grows by 48 bits — carries exactly 48 bits more noise
        let noisy = gen_ksk(&ring, &base, &sk, &sk2, &one_prime, &mut rng, log2_alpha - 48.);
        assert!((noisy.log2_key_switch_noise(&ring, &base) - ksk.log2_key_switch_noise(&ring, &base) - 48.) < 1e-6);

        // wider gadget digits contribute correspondingly more noise, at the same error
        let two_primes = select_digits(base.len().div_ceil(2), base.len());
        let wide = gen_ksk(&ring, &base, &sk, &sk2, &two_primes, &mut rng, log2_alpha);
        assert!(wide.log2_key_switch_noise(&ring, &base) > ksk.log2_key_switch_noise(&ring, &base) + 40.);

        // restricting a key to a shorter base clips its digits, lowering the contribution
        let short = &base[..2];
        assert!(ksk.log2_key_switch_noise(&ring, short) <= ksk.log2_key_switch_noise(&ring, &base));
    }

    /// The noise parametrization by modulus-to-noise ratio: the error follows the modulus, so a
    /// fixed assumption cannot be weakened by raising `Q` — raising it raises the error of every
    /// key and ciphertext instead. Objects over a smaller base get the [`MIN_SIGMA`] floor rather
    /// than a meaninglessly small error.
    #[test]
    fn test_sigma_follows_modulus() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);

        // round trip: at the base it was derived for, the ratio yields exactly MIN_SIGMA
        assert!((sigma_for_log2_alpha(&ring, &base, log2_alpha) - MIN_SIGMA).abs() < 1e-9);

        // a larger modulus at the same assumption: the error grows by exactly the added bits
        let larger = (0..base.len() + 1).collect::<Vec<_>>();
        let added_bits = ring.log2_modulus_of(&larger) - ring.log2_modulus_of(&base);
        assert!(added_bits > 40., "the ring should have further factors to extend the base with");
        let log2_sigma_larger = sigma_for_log2_alpha(&ring, &larger, log2_alpha).log2();
        assert!((log2_sigma_larger - (MIN_SIGMA.log2() + added_bits)).abs() < 1e-6);

        // a smaller modulus (as used by the sparse-key encapsulation) hits the floor: its own
        // ratio is far below the assumption, which is what encapsulation needs anyway
        let ske = ring.sparse_key_encaps_rns_base();
        assert!(ring.log2_modulus_of(&ske) < ring.log2_modulus_of(&base));
        assert_eq!(MIN_SIGMA, sigma_for_log2_alpha(&ring, &ske, log2_alpha));
    }

    #[test]
    fn test_key_switch() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 2u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 255) - 127).collect();
        let sk1 = gen_sk(&ring, &mut rng);
        let sk2 = gen_sk(&ring, &mut rng);
        let ct1 = bfv_enc(&ring, &base, 256, &m, &sk1, &mut rng, log2_alpha);
        let digits = select_digits(base.len(), base.len());
        let ksk = gen_ksk(&ring, &base, &sk1, &sk2, &digits, &mut rng, log2_alpha);

        let mut c1 = ring.clone_els(ct1.rows(1..2));
        ring.crt_to_pb(&mut c1);
        let ct2 = key_switch(&ring, ct1.rows(0..1), &c1, &ksk, None);
        assert_eq!(m, bfv_dec(&ring, 256, &ct2, &sk2));
    }

    #[test]
    fn test_hom_mul_scalar() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 3u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let ct1 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 5), &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 7), &sk, &mut rng, log2_alpha);
        let res = bfv_hom_mul(&ring, 256, &ct1, &ct2, &rk, Basis::CRTBasis);
        assert_eq!(35, decode_scalar(&ring, 256, &bfv_dec(&ring, 256, &res, &sk)));
    }

    #[test]
    fn test_hom_mul_nonscalar() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 4u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let m1 = (0..ring.n()).map(|i| i as i128 % 256).collect::<Vec<_>>();
        let m2 = (0..ring.n()).map(|i| (i * i) as i128 % 256).collect::<Vec<_>>();
        let prod = mul_plain_pb(&ring, 256, &m1, &m2);

        let ct1 = bfv_enc(&ring, &base, 256, &m1, &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 256, &m2, &sk, &mut rng, log2_alpha);
        let res = bfv_hom_mul(&ring, 256, &ct1, &ct2, &rk, Basis::CRTBasis);
        assert_eq!(prod, bfv_dec(&ring, 256, &res, &sk));
    }

    #[test]
    fn test_hom_mul_large_t() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 5u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len() / 2, base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let m1 = (0..ring.n()).map(|i| (i as i128) << 24).collect::<Vec<_>>();
        let m2 = (0..ring.n()).map(|i| ((i * i) as i128) << 23).collect::<Vec<_>>();
        let prod = mul_plain_pb(&ring, 1 << 64, &m1, &m2);

        let ct1 = bfv_enc(&ring, &base, 1 << 64, &m1, &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 1 << 64, &m2, &sk, &mut rng, log2_alpha);
        let res = bfv_hom_mul(&ring, 1 << 64, &ct1, &ct2, &rk, Basis::CRTBasis);
        assert_eq!(prod, bfv_dec(&ring, 1 << 64, &res, &sk));
    }

    /// homomorphic addition without reduction: the sum ciphertext has representatives bounded
    /// by 2p (bound 2), which decryption (fma accumulation) accepts directly
    #[test]
    fn test_hom_add_unreduced() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 9u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let m1: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 100) - 50).collect();
        let m2: Vec<i128> = (0..ring.n()).map(|i| ((3 * i) as i128 % 100) - 50).collect();
        let ct1 = bfv_enc(&ring, &base, 256, &m1, &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 256, &m2, &sk, &mut rng, log2_alpha);

        let mut sum = ring.clone_els(&ct1);
        ring.add_els(&mut sum, &ct2);
        assert_eq!(2, sum.bound);
        let expected: Vec<i128> = m1.iter().zip(&m2).map(|(a, b)| cmod(a + b, 256)).collect();
        assert_eq!(expected, bfv_dec(&ring, 256, &sum, &sk));

        let mut diff = ring.clone_els(&ct1);
        ring.sub_els(&mut diff, &ct2);
        assert_eq!(2, diff.bound);
        let expected: Vec<i128> = m1.iter().zip(&m2).map(|(a, b)| cmod(a - b, 256)).collect();
        assert_eq!(expected, bfv_dec(&ring, 256, &diff, &sk));
    }

    /// multiplication with period-basis output: the result is unreduced (bound 2) in period
    /// basis, and both the basis transform and further accumulation accept it directly
    #[test]
    fn test_hom_mul_period_output() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 10u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let ct1 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 5), &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 7), &sk, &mut rng, log2_alpha);

        let mut res = bfv_hom_mul(&ring, 256, &ct1, &ct2, &rk, Basis::PeriodBasis);
        assert_eq!(Basis::PeriodBasis, res.basis);
        assert_eq!(2, res.bound);
        // the bound-2 result feeds directly into the basis transform
        ring.pb_to_crt(&mut res);
        assert_eq!(1, res.bound);
        assert_eq!(35, decode_scalar(&ring, 256, &bfv_dec(&ring, 256, &res, &sk)));

        // ... and (as a CRT-basis unreduced sum) into fma accumulation: 5*7 + 5*7 = 70
        let mut doubled = ring.clone_els(&res);
        ring.add_els(&mut doubled, &res);
        let mut acc = ring.allocate_wide(2, &base);
        ring.add(&mut acc, &doubled);
        let sum = ring.reduce_wide(&acc, Basis::CRTBasis);
        assert_eq!(70, decode_scalar(&ring, 256, &bfv_dec(&ring, 256, &sum, &sk)));
    }

    #[test]
    fn test_hom_gal() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 6u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_18(), 300..320, 4);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let g = ring.field().kq_cyclic_decomposition()[0].0;
        let gk = gen_gk(&ring, &sk, g, &digits, &mut rng, log2_alpha);

        let ptxt_pb = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
        let ct = bfv_enc(&ring, &base, 16, &ptxt_pb, &sk, &mut rng, log2_alpha);
        let [res] = bfv_hom_gal(&ring, &ct, &[g], &[(g, gk)]).try_into().ok().unwrap();
        let actual = bfv_dec(&ring, 16, &res, &sk);
        let expected = vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0];
        assert_eq!(expected, actual);
    }

    /// the full-degree field: a complete enc/mul/dec roundtrip over `field_36864()`
    #[test]
    fn test_enc_dec_hom_mul_field_36864() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 8u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, crate::field::field_36864(), 100..120, 8);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 255) - 127).collect();
        let ct = bfv_enc(&ring, &base, 256, &m, &sk, &mut rng, log2_alpha);
        assert_eq!(m, bfv_dec(&ring, 256, &ct, &sk));

        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let ct1 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 5), &sk, &mut rng, log2_alpha);
        let ct2 = bfv_enc(&ring, &base, 256, &encode_scalar(&ring, 7), &sk, &mut rng, log2_alpha);
        let res = bfv_hom_mul(&ring, 256, &ct1, &ct2, &rk, Basis::CRTBasis);
        assert_eq!(35, decode_scalar(&ring, 256, &bfv_dec(&ring, 256, &res, &sk)));
    }

    #[test]
    fn test_mod_switch_down_dynamic_scaling() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 7u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_18(), 300..320, 16);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let ct = bfv_enc(&ring, &base, 1 << 16, &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);

        // scale the ciphertext down to the sparse-key-encapsulation base and decrypt
        let mut ct_pb = ring.clone_els(&ct);
        ring.crt_to_pb(&mut ct_pb);
        let small_len = ring.sparse_key_encaps_rns_base().len();
        let mut switched = mod_switch_down(&ring, &ct_pb, small_len);
        ring.pb_to_crt(&mut switched);
        assert_eq!(2025, decode_scalar(&ring, 1 << 16, &bfv_dec(&ring, 1 << 16, &switched, &sk)));

        // dynamic scaling: continue computing (a multiplication) on a mid-sized base; the
        // relinearization key is restricted to the smaller base on the fly
        let mid_len = base.len() - 2;
        let mut switched_mid = mod_switch_down(&ring, &ct_pb, mid_len);
        ring.pb_to_crt(&mut switched_mid);
        let res = bfv_hom_mul(&ring, 1 << 16, &switched_mid, &switched_mid, &rk, Basis::CRTBasis);
        assert_eq!(cmod(2025 * 2025, 1 << 16), decode_scalar(&ring, 1 << 16, &bfv_dec(&ring, 1 << 16, &res, &sk)));
    }
}
