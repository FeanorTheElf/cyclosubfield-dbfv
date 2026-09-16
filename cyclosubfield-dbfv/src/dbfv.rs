//! The dBFV ("digit-BFV") scheme: plaintexts modulo `t = b^d` are represented by `d` BFV
//! ciphertexts ("digits"), where the `i`-th digit encrypts (roughly) the value divided by
//! `b^i`. This keeps the noise of high digits independent of the low digits, enabling e.g.
//! exact division by powers of two and cheap digit extraction.
//!
//! A dBFV ciphertext is a single [`RingEls`] batch whose first dimension has length
//! `2 * digits`: the rows `2i, 2i + 1` are the components `(c0, c1)` of the `i`-th digit.
//! A ciphertext may carry fewer than `d` digits (missing digits are zero); since all digits
//! live in one tensor, most homomorphic operations are single coarse-grained HAL calls over
//! the whole batch. After homomorphic operations, the digit values can grow beyond `b`; the
//! canonical form (each digit bounded by roughly `b`) is restored by [`dbfv_reduce()`], which
//! splits off the overflow of each digit and propagates it as a carry into the next digit.
//!
//! Fresh encryptions ([`dbfv_enc()`]) sample an independent BFV encryption of zero for every
//! digit, add the message to digit `0` and reduce; this is provably secure under a weaker
//! RLWE assumption than embedding a single BFV encryption as `[bfv_enc(m), 0, ..., 0]`; see
//! [`dbfv_enc()`].

use std::sync::{MappedRwLockReadGuard, RwLockReadGuard};

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::integer::*;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::ring::*;
use tracing::instrument;

use crate::bfv::*;
use crate::hal::{
    Backend, TensorI64Mut, TensorI64Owned, TensorI64Ref, TensorMut, TensorOwned, TensorRef, split_i128_limbs,
};
use crate::ring::{
    Basis, RNSBaseIndices, RingContext, RingEls, RingElsRef, WideRingEls, ZZbig, ZZi64, ZZi128, cmod, cmod_pow2,
};

/// A dBFV ciphertext: a [`RingEls`] batch of `2 * digits` elements, where the rows
/// `2i, 2i + 1` form the BFV ciphertext of the `i`-th digit; see the module documentation.
pub type DBFVCiphertext<B> = RingEls<B>;

/// The number of digits carried by a dBFV ciphertext.
pub fn dbfv_digit_count<B: Backend>(ct: &DBFVCiphertext<B>) -> usize {
    assert_eq!(0, ct.count() % 2);
    return ct.count() / 2;
}

/// A dBFV ciphertext with no digits (an encryption of zero), usable e.g. as the initial
/// summand of the `dbfv_hom_fma_*` functions.
pub fn dbfv_empty<B: Backend>(ring: &RingContext<B>, base: &RNSBaseIndices, basis: Basis) -> DBFVCiphertext<B> {
    ring.allocate_in_basis(0, base, basis)
}

/// The `i`-th digit as a standalone BFV ciphertext, as a copy-free view.
pub fn dbfv_digit<B: Backend>(ct: &DBFVCiphertext<B>, i: usize) -> RingElsRef<'_, B> { ct.rows(2 * i..2 * i + 2) }

/// The plaintext modulus configuration `t = 2^log2_t` with digit base `b = 2^log2_b`.
#[derive(Clone, Copy, Debug)]
pub struct DBFVModulus {
    log2_b: usize,
    log2_t: usize,
}

impl DBFVModulus {
    pub fn new(log2_b: usize, digits: usize) -> Self { Self::new_incomplete(log2_b, digits * log2_b) }

    /// A modulus whose total precision is not a multiple of the digit size (e.g. after exact
    /// division).
    pub fn new_incomplete(log2_b: usize, log2_t: usize) -> Self {
        assert!(log2_b >= 1 && log2_t >= 1 && log2_t <= 126, "t = 2^log2_t must be between 2 and 2^126");
        Self { log2_b, log2_t }
    }

    pub fn log2_t(&self) -> usize { self.log2_t }

    pub fn log2_b(&self) -> usize { self.log2_b }

    pub fn t(&self) -> i128 { 1 << self.log2_t }

    pub fn digits(&self) -> usize { (self.log2_t - 1) / self.log2_b + 1 }

    pub fn b(&self) -> i128 { 1 << self.log2_b }
}

/// Decomposes a scalar into balanced base-`b` digits (at most `max_digits` of them).
fn balanced_digits(mut x: i128, b: i128, max_digits: usize) -> Vec<i128> {
    let mut result = Vec::new();
    for _ in 0..max_digits {
        if x == 0 {
            break;
        }
        let rem = cmod(x, b);
        result.push(rem);
        x = (x - rem) / b;
    }
    return result;
}

/// Decomposes a coefficient vector into balanced base-`b` digit vectors.
fn balanced_digits_vec(x: &[i128], b: i128, max_digits: usize) -> Vec<Vec<i128>> {
    let mut result = Vec::new();
    let mut current = x.to_vec();
    for _ in 0..max_digits {
        if current.iter().all(|x| *x == 0) {
            break;
        }
        let rems = current.iter().map(|x| cmod(*x, b)).collect::<Vec<_>>();
        for (x, y) in current.iter_mut().zip(&rems) {
            *x = (*x - *y) / b;
        }
        result.push(rems);
    }
    return result;
}

/// An accumulator for dBFV ciphertexts, the dBFV counterpart of [`WideRingEls`]: it is
/// allocated once (with capacity for all `modulus.digits()` digits), supports repeated
/// homomorphic fused multiply-adds (digit convolutions with scalars, `Lʼ`-elements or
/// plaintexts, see `dbfv_acc_fma_*`) and additions/subtractions of dBFV ciphertexts, and the
/// delayed modular reduction happens once, when converting the accumulator back into a
/// ciphertext with [`dbfv_finish_accumulator()`]. Use this to compute inner products of dBFV
/// ciphertexts with scalar/plaintext vectors without per-summand reductions.
///
/// (For plain BFV ciphertexts no separate type is needed: a BFV ciphertext is a 2-element
/// [`RingEls`] batch, so a 2-row [`WideRingEls`] together with
/// [`RingContext::fma_scalar()`] & co. serves as its inner-product accumulator.)
pub struct DBFVAccumulator<B: Backend> {
    acc: WideRingEls<B>,
    modulus: DBFVModulus,
    basis: Basis,
    used_digits: usize,
}

impl<B: Backend> DBFVAccumulator<B> {
    /// The number of digits accumulated so far.
    pub fn digit_count(&self) -> usize { self.used_digits }

    pub fn digit_capacity(&self) -> usize { self.modulus.digits() }

    pub fn basis(&self) -> Basis { self.basis }

    pub fn rns_base(&self) -> &RNSBaseIndices { self.acc.rns_base() }
}

/// Allocates a zeroed dBFV accumulator with capacity for all `modulus.digits()` digits.
pub fn dbfv_allocate_accumulator<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    base: &RNSBaseIndices,
    basis: Basis,
) -> DBFVAccumulator<B> {
    DBFVAccumulator { acc: ring.allocate_wide(2 * modulus.digits(), base), modulus, basis, used_digits: 0 }
}

/// Adds a dBFV ciphertext to the accumulator, digit-wise.
pub fn dbfv_acc_add<B: Backend>(ring: &RingContext<B>, acc: &mut DBFVAccumulator<B>, ct: &DBFVCiphertext<B>) {
    let d = dbfv_digit_count(ct);
    if d == 0 {
        return;
    }
    assert!(d <= acc.digit_capacity());
    assert_eq!(acc.basis, ct.basis);
    assert_eq!(acc.rns_base(), &*ct.rns_base);
    acc.acc.add(ring.backend(), 0..2 * d, ct.data.as_ref(), ct.bound);
    acc.used_digits = usize::max(acc.used_digits, d);
}

/// Adds a scalar plaintext constant to the accumulator, digit-wise.
pub fn dbfv_acc_add_constant<B: Backend>(ring: &RingContext<B>, acc: &mut DBFVAccumulator<B>, mut scalar: i128) {
    let delta = ZZbig.rounded_div(ring.modulus_of(acc.rns_base()), &int_cast(acc.modulus.t(), ZZbig, ZZi128));
    if acc.basis == Basis::PeriodBasis {
        scalar *= ring.field().scale_q_to_m_period_basis() as i128;
    }
    let coeff = ZZbig.mul_ref_snd(int_cast(scalar, ZZbig, ZZi128), &delta);
    let tensor = ring.scalar_tensor_big(acc.rns_base(), &coeff);
    acc.acc.add(ring.backend(), 0..1, tensor.as_ref(), 1);
}

/// Subtracts a dBFV ciphertext from the accumulator, digit-wise.
pub fn dbfv_acc_sub<B: Backend>(ring: &RingContext<B>, acc: &mut DBFVAccumulator<B>, ct: &DBFVCiphertext<B>) {
    let d = dbfv_digit_count(ct);
    if d == 0 {
        return;
    }
    assert!(d <= acc.digit_capacity());
    assert_eq!(acc.basis, ct.basis);
    assert_eq!(acc.rns_base(), &*ct.rns_base);
    acc.acc.sub(ring.backend(), 0..2 * d, ct.data.as_ref(), ct.bound);
    acc.used_digits = usize::max(acc.used_digits, d);
}

/// The shared skeleton of the `dbfv_acc_fma_*` functions: accumulates the digit-convolution
/// `lhs * rhs`, where the `j`-th digit of `rhs` is given as a broadcastable montgomery-form
/// factor tensor (see [`RingContext::fma_tensor()`]). Digits beyond the accumulator's
/// capacity are dropped.
fn dbfv_acc_fma_tensors<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    lhs: &DBFVCiphertext<B>,
    rhs_tensors: &[TensorI64Owned<B>],
) {
    let lhs_digits = dbfv_digit_count(lhs);
    if lhs_digits == 0 || rhs_tensors.is_empty() {
        return;
    }
    assert!(lhs_digits <= acc.digit_capacity());
    assert_eq!(acc.basis, lhs.basis);
    assert_eq!(acc.rns_base(), &*lhs.rns_base);
    let new_digits = usize::min(acc.digit_capacity(), lhs_digits + rhs_tensors.len() - 1);
    for i in 0..lhs_digits {
        for j in 0..rhs_tensors.len() {
            let k = i + j;
            if k < new_digits {
                acc.acc.fma(
                    ring.backend(),
                    2 * k..2 * k + 2,
                    lhs.data.as_ref().slice(2 * i..2 * i + 2, .., .., ..),
                    lhs.bound,
                    rhs_tensors[j].as_ref(),
                    1,
                );
            }
        }
    }
    acc.used_digits = usize::max(acc.used_digits, new_digits);
}

/// Accumulates `lhs * rhs` for a scalar `rhs`, which is decomposed into balanced base-`b`
/// digits so that the noise growth per digit is bounded by `b/2`.
pub fn dbfv_acc_fma_scalar<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    lhs: &DBFVCiphertext<B>,
    rhs: i128,
) {
    let modulus = acc.modulus;
    let rhs_tensors = balanced_digits(rhs, modulus.b(), modulus.digits())
        .into_iter()
        .map(|digit| ring.scalar_tensor(acc.rns_base(), digit))
        .collect::<Vec<_>>();
    dbfv_acc_fma_tensors(ring, acc, lhs, &rhs_tensors);
}

/// Accumulates `lhs * rhs` for an element `rhs` of `Lʼ` (given by its period-basis
/// coefficients w.r.t. the basis induced by `M`), decomposed into balanced base-`b` digits
/// coefficient-wise. The accumulator and `lhs` must be in CRT basis.
pub fn dbfv_acc_fma_element_lprime<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    lhs: &DBFVCiphertext<B>,
    rhs: &[i128],
) {
    assert_eq!(Basis::CRTBasis, acc.basis);
    let modulus = acc.modulus;
    let rhs_tensors = balanced_digits_vec(rhs, modulus.b(), modulus.digits())
        .into_iter()
        .map(|coeffs| ring.lprime_element_tensor(acc.rns_base(), &coeffs))
        .collect::<Vec<_>>();
    dbfv_acc_fma_tensors(ring, acc, lhs, &rhs_tensors);
}

/// Accumulates `lhs * rhs` for a plaintext `rhs` (period-basis coefficients modulo `t`),
/// decomposed into balanced base-`b` digits coefficient-wise. The accumulator and `lhs` must
/// be in CRT basis.
pub fn dbfv_acc_fma_plain<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    lhs: &DBFVCiphertext<B>,
    rhs: &[i128],
) {
    assert_eq!(ring.n(), rhs.len());
    assert_eq!(Basis::CRTBasis, acc.basis);
    let modulus = acc.modulus;
    let rhs_tensors = balanced_digits_vec(rhs, modulus.b(), modulus.digits())
        .into_iter()
        .map(|coeffs| {
            let mut rhs_el = encode_pb_unscaled(ring, acc.rns_base(), modulus.t(), &coeffs);
            ring.pb_to_crt(&mut rhs_el);
            rhs_el.data
        })
        .collect::<Vec<_>>();
    dbfv_acc_fma_tensors(ring, acc, lhs, &rhs_tensors);
}

/// The shared skeleton of the `dbfv_acc_inner_product_*` functions: the multi-term,
/// cache-tiled counterpart of a chain of [`dbfv_acc_fma_tensors()`] calls. It accumulates the
/// sum of digit-convolutions `Σ_t lhs_t * rhs_t` into the accumulator, where the `j`-th digit
/// of the `t`-th `rhs` is a broadcastable montgomery-form factor tensor. All contributions to
/// one output digit are gathered into a single [`WideRingEls::inner_product()`] call, so the
/// `i128` accumulator is streamed through memory once per output digit instead of once per
/// fused multiply-add. The accumulator rows written to must still be fresh (the underlying
/// inner product overwrites), which holds for the freshly allocated accumulators of the
/// `dbfv_acc_inner_product_*` callers; digits beyond the accumulator's capacity are dropped.
fn dbfv_acc_inner_product_tensors<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    terms: &[(&DBFVCiphertext<B>, Vec<TensorI64Owned<B>>)],
) {
    let capacity = acc.digit_capacity();
    let mut max_new_digits = 0;
    for (lhs, rhs_tensors) in terms {
        let lhs_digits = dbfv_digit_count(lhs);
        if lhs_digits == 0 || rhs_tensors.is_empty() {
            continue;
        }
        assert!(lhs_digits <= capacity);
        assert_eq!(acc.basis, lhs.basis);
        assert_eq!(acc.rns_base(), &*lhs.rns_base);
        let new_digits = usize::min(capacity, lhs_digits + rhs_tensors.len() - 1);
        max_new_digits = usize::max(max_new_digits, new_digits);
    }
    if max_new_digits == 0 {
        return;
    }
    let backend = ring.backend();
    // one inner product per output digit, gathering the whole convolution diagonal `i + j = k`
    // across all terms
    for k in 0..max_new_digits {
        let mut k_terms: Vec<(TensorI64Ref<B>, u128, TensorI64Ref<B>, u128)> = Vec::new();
        for (lhs, rhs_tensors) in terms {
            let lhs_digits = dbfv_digit_count(lhs);
            if lhs_digits == 0 || rhs_tensors.is_empty() {
                continue;
            }
            let new_digits = usize::min(capacity, lhs_digits + rhs_tensors.len() - 1);
            if k >= new_digits {
                continue;
            }
            for j in 0..rhs_tensors.len() {
                if k < j {
                    break;
                }
                let i = k - j;
                if i >= lhs_digits {
                    continue;
                }
                k_terms.push((
                    lhs.data.as_ref().slice(2 * i..2 * i + 2, .., .., ..),
                    lhs.bound,
                    rhs_tensors[j].as_ref(),
                    1,
                ));
            }
        }
        if !k_terms.is_empty() {
            acc.acc.inner_product(backend, 2 * k..2 * k + 2, &k_terms);
        }
    }
    acc.used_digits = usize::max(acc.used_digits, max_new_digits);
}

/// The inner-product (multi-term) counterpart of [`dbfv_acc_fma_scalar()`]: accumulates
/// `Σ_t lhs_t * scalar_t` into the accumulator with a single reduction's worth of memory
/// traffic per output digit. Each `scalar_t` is decomposed into balanced base-`b` digits, as
/// in [`dbfv_acc_fma_scalar()`]. The accumulator rows must be fresh (see
/// [`dbfv_acc_inner_product_tensors()`]).
pub fn dbfv_acc_inner_product_scalar<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    terms: &[(&DBFVCiphertext<B>, i128)],
) {
    let modulus = acc.modulus;
    let prepared = terms
        .iter()
        .map(|(lhs, scalar)| {
            let rhs_tensors = balanced_digits(*scalar, modulus.b(), modulus.digits())
                .into_iter()
                .map(|digit| ring.scalar_tensor(acc.rns_base(), digit))
                .collect::<Vec<_>>();
            (*lhs, rhs_tensors)
        })
        .collect::<Vec<_>>();
    dbfv_acc_inner_product_tensors(ring, acc, &prepared);
}

/// The inner-product (multi-term) counterpart of [`dbfv_acc_fma_element_lprime()`]: accumulates
/// `Σ_t lhs_t * rhs_t` for elements `rhs_t` of `Lʼ` (period-basis coefficients), with a single
/// reduction's worth of memory traffic per output digit. The accumulator and all `lhs_t` must
/// be in CRT basis, and the accumulator rows must be fresh (see
/// [`dbfv_acc_inner_product_tensors()`]).
pub fn dbfv_acc_inner_product_element_lprime<B: Backend>(
    ring: &RingContext<B>,
    acc: &mut DBFVAccumulator<B>,
    terms: &[(&DBFVCiphertext<B>, &[i128])],
) {
    assert_eq!(Basis::CRTBasis, acc.basis);
    let modulus = acc.modulus;
    let prepared = terms
        .iter()
        .map(|(lhs, rhs)| {
            let rhs_tensors = balanced_digits_vec(rhs, modulus.b(), modulus.digits())
                .into_iter()
                .map(|coeffs| ring.lprime_element_tensor(acc.rns_base(), &coeffs))
                .collect::<Vec<_>>();
            (*lhs, rhs_tensors)
        })
        .collect::<Vec<_>>();
    dbfv_acc_inner_product_tensors(ring, acc, &prepared);
}

/// Converts the accumulator back into a dBFV ciphertext; this performs the modular reduction
/// that has been delayed during accumulation (only over the digits actually used).
#[instrument(skip_all)]
pub fn dbfv_finish_accumulator<B: Backend>(ring: &RingContext<B>, acc: &DBFVAccumulator<B>) -> DBFVCiphertext<B> {
    ring.reduce_wide_rows(&acc.acc, 0..2 * acc.used_digits, acc.basis)
}

/// Computes `summand + lhs * rhs` for a scalar `rhs`; the one-shot version of
/// [`dbfv_acc_fma_scalar()`]. For inner products, prefer accumulating several
/// `dbfv_acc_fma_*` steps into one [`DBFVAccumulator`] and reducing once.
#[instrument(skip_all)]
pub fn dbfv_hom_fma_scalar<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    lhs: &DBFVCiphertext<B>,
    rhs: i128,
    summand: DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    assert_eq!(summand.basis, lhs.basis);
    assert!(dbfv_digit_count(lhs) <= modulus.digits());
    assert!(dbfv_digit_count(&summand) <= modulus.digits());
    if dbfv_digit_count(lhs) == 0 || rhs == 0 {
        return summand;
    }
    let mut acc = dbfv_allocate_accumulator(ring, modulus, &lhs.rns_base, lhs.basis);
    dbfv_acc_add(ring, &mut acc, &summand);
    dbfv_acc_fma_scalar(ring, &mut acc, lhs, rhs);
    return dbfv_finish_accumulator(ring, &acc);
}

/// Computes `summand + lhs * rhs` for an element `rhs` of `Lʼ`; the one-shot version of
/// [`dbfv_acc_fma_element_lprime()`]. All ciphertexts must be in CRT basis.
#[instrument(skip_all)]
pub fn dbfv_hom_fma_element_lprime<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    lhs: &DBFVCiphertext<B>,
    rhs: &[i128],
    summand: DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    assert_eq!(Basis::CRTBasis, lhs.basis);
    assert_eq!(Basis::CRTBasis, summand.basis);
    assert!(dbfv_digit_count(lhs) <= modulus.digits());
    assert!(dbfv_digit_count(&summand) <= modulus.digits());
    if dbfv_digit_count(lhs) == 0 || rhs.iter().all(|x| *x == 0) {
        return summand;
    }
    let mut acc = dbfv_allocate_accumulator(ring, modulus, &lhs.rns_base, Basis::CRTBasis);
    dbfv_acc_add(ring, &mut acc, &summand);
    dbfv_acc_fma_element_lprime(ring, &mut acc, lhs, rhs);
    return dbfv_finish_accumulator(ring, &acc);
}

/// Computes `summand + lhs * rhs` for a plaintext `rhs` (period-basis coefficients modulo
/// `t`); the one-shot version of [`dbfv_acc_fma_plain()`]. All ciphertexts must be in CRT
/// basis.
#[instrument(skip_all)]
pub fn dbfv_hom_fma_plain<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    lhs: &DBFVCiphertext<B>,
    rhs: &[i128],
    summand: DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    assert_eq!(Basis::CRTBasis, lhs.basis);
    assert_eq!(Basis::CRTBasis, summand.basis);
    assert_eq!(ring.n(), rhs.len());
    assert!(dbfv_digit_count(lhs) <= modulus.digits());
    assert!(dbfv_digit_count(&summand) <= modulus.digits());
    if dbfv_digit_count(lhs) == 0 || rhs.iter().all(|x| *x == 0) {
        return summand;
    }
    let mut acc = dbfv_allocate_accumulator(ring, modulus, &lhs.rns_base, Basis::CRTBasis);
    dbfv_acc_add(ring, &mut acc, &summand);
    dbfv_acc_fma_plain(ring, &mut acc, lhs, rhs);
    return dbfv_finish_accumulator(ring, &acc);
}

/// Divides a dBFV ciphertext exactly by `2^log2_div` (which must divide the encrypted value),
/// by merging the lowest digits into a single one and shifting the higher digits down. The
/// result is w.r.t. the modulus with `log2_tʼ = log2_t - log2_div`.
///
/// The ciphertexts themselves are not rescaled: a ciphertext encrypting `Δ y` also encrypts
/// `Δʼ (y / 2^log2_div)` for the larger scaling factor `Δʼ = 2^log2_div Δ` of the smaller
/// plaintext modulus `tʼ`. Only the digit *structure* needs fixing, since the individual digit
/// values `y_i` are not divisible by `2^log2_div` (only their weighted sum `x = Σ_i b^i y_i`
/// is). With `k = ceil(log2_div / log2_b)`, the lowest `k + 1` digits are merged into the new
/// digit `0` (nothing is discarded: they enter with their weights `b^i`), and every higher
/// digit is shifted down by `k` positions, which compensates the weight `b^k` it loses by
/// scaling its value with `b^k`:
/// ```text
///   zʼ_0 = (y_0 + b y_1 + ... + b^k y_k) / 2^log2_div,    zʼ_j = b^k y_(j + k) / 2^log2_div
/// ```
/// The weighted sum `Σ_j b^j zʼ_j` is then exactly `x / 2^log2_div`, and all `zʼ_j` are
/// integral: `2^log2_div | b^k` handles the digits `j >= 1`, and the combined value of `zʼ_0`
/// is `≡ x mod b^(k + 1)` (resp. `≡ x mod t` if fewer than `k + 1` digits are present), hence
/// divisible by `2^log2_div` as well. Note that this holds for arbitrarily large digit values,
/// i.e. the input does not have to be reduced.
///
/// The merged digit `0` may be much larger than `b` — up to `b^(k + 1) / 2^log2_div`, i.e. `b`
/// times the bound of a canonical digit when `log2_div` is not a multiple of `log2_b`; a
/// subsequent [`dbfv_reduce()`] brings it back into the canonical range. Likewise, the noise of
/// the shifted digits grows by the factor `b^k / 2^log2_div <= b` relative to `Δʼ`, so an exact
/// division by a `log2_div` that is not a multiple of `log2_b` requires the input digit noise to
/// have `k log2_b - log2_div` bits of headroom below `Δ / 2`.
#[instrument(skip_all)]
pub fn dbfv_exact_div_pow2<B: Backend>(
    ring: &RingContext<B>,
    initial_modulus: DBFVModulus,
    val: DBFVCiphertext<B>,
    log2_div: usize,
) -> DBFVCiphertext<B> {
    let d = dbfv_digit_count(&val);
    assert!(d <= initial_modulus.digits());
    assert!(log2_div < initial_modulus.log2_t(), "the result modulus tʼ = t / 2^log2_div must be at least 2");
    if log2_div == 0 || d <= 1 {
        // a single digit is a plain BFV ciphertext, which needs no restructuring at all
        return val;
    }
    let log2_b = initial_modulus.log2_b();
    // the number of digit positions to shift down by; the shifted digits are scaled by b^k,
    // which makes their values divisible by 2^log2_div, and the lowest k + 1 digits are merged
    let k = (log2_div - 1) / log2_b + 1;
    let base = val.rns_base.clone();
    ring.assert_mul_mont_safe(val.bound, &base);
    let out_digits = usize::max(d.saturating_sub(k), 1);
    let mut result = ring.allocate_in_basis(2 * out_digits, &base, val.basis);

    // the digits `>= 1` are the input digits `>= k + 1`, shifted down by `k` and scaled by `b^k`
    if out_digits > 1 {
        let b_pow_k = ring.scalar_tensor_big(&base, &ZZbig.power_of_two(k * log2_b));
        let mods = ring.mods_for(&base);
        ring.assign(result.data.as_mut().slice(2.., .., .., ..), val.data.as_ref().slice(2 * (k + 1).., .., .., ..));
        ring.backend().hal_mul_mont64(result.data.as_mut().slice(2.., .., .., ..), b_pow_k.as_ref(), &mods);
    }

    // the merged digit_0 = Σ_(i <= min(k, d - 1)) b^i * old digit_i
    let factors = (0..=usize::min(k, d - 1))
        .map(|i| ring.scalar_tensor_big(&base, &ZZbig.power_of_two(i * log2_b)))
        .collect::<Vec<_>>();
    let terms = factors
        .iter()
        .enumerate()
        .map(|(i, factor)| (val.data.as_ref().slice(2 * i..2 * i + 2, .., .., ..), val.bound, factor.as_ref(), 1u128))
        .collect::<Vec<_>>();
    let mut acc = ring.allocate_wide(2, &base);
    acc.inner_product(ring.backend(), 0..2, &terms);
    ring.reduce_wide_into(result.rows_mut(0..2), &acc);
    return result;
}

/// Adds two dBFV ciphertexts digit-wise.
#[instrument(skip_all)]
pub fn dbfv_hom_add<B: Backend>(
    ring: &RingContext<B>,
    lhs: DBFVCiphertext<B>,
    rhs: &DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    dbfv_hom_add_sub(ring, lhs, rhs, false)
}

/// Subtracts a dBFV ciphertext from another digit-wise.
#[instrument(skip_all)]
pub fn dbfv_hom_sub<B: Backend>(
    ring: &RingContext<B>,
    lhs: DBFVCiphertext<B>,
    rhs: &DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    dbfv_hom_add_sub(ring, lhs, rhs, true)
}

fn dbfv_hom_add_sub<B: Backend>(
    ring: &RingContext<B>,
    lhs: DBFVCiphertext<B>,
    rhs: &DBFVCiphertext<B>,
    subtract: bool,
) -> DBFVCiphertext<B> {
    assert_eq!(lhs.basis, rhs.basis);
    let lhs_digits = dbfv_digit_count(&lhs);
    let rhs_digits = dbfv_digit_count(rhs);
    if rhs_digits == 0 {
        return lhs;
    }
    if lhs_digits > 0 {
        assert_eq!(lhs.basis, rhs.basis);
        assert_eq!(lhs.rns_base, rhs.rns_base);
    }
    let mut acc = ring.allocate_wide(2 * usize::max(lhs_digits, rhs_digits), &rhs.rns_base);
    if lhs_digits > 0 {
        acc.add(ring.backend(), 0..2 * lhs_digits, lhs.data.as_ref(), lhs.bound);
    }
    if subtract {
        acc.sub(ring.backend(), 0..2 * rhs_digits, rhs.data.as_ref(), rhs.bound);
    } else {
        acc.add(ring.backend(), 0..2 * rhs_digits, rhs.data.as_ref(), rhs.bound);
    }
    return ring.reduce_wide(&acc, rhs.basis);
}

/// The modulus-to-noise ratio at which [`dbfv_enc()`] creates the *individual* digit
/// zero-encryptions, given the ratio `2^log2_alpha` that the combined ciphertext relies on: the
/// combined noise is `Σ b^i e_i ≈ b^(digits - 1) σ`, so each digit is created at the
/// `b^(digits - 1)` times larger ratio. This is the entire security gain of dBFV encryption over
/// embedding a BFV ciphertext, expressed as a parameter.
pub fn dbfv_log2_alpha_per_digit(modulus: DBFVModulus, log2_alpha: f64) -> f64 {
    log2_alpha + (modulus.log2_b() * (modulus.digits() - 1)) as f64
}

/// Encrypts a plaintext (period-basis coefficients modulo `t`) as a dBFV ciphertext with all
/// `modulus.digits()` digits, in CRT basis over the given RNS base.
///
/// Every digit is an independent, fresh BFV encryption of zero; the `Δ`-scaled message is
/// added to the first component of digit `0`, and a final [`dbfv_reduce()`] restores the
/// canonical digit form. The reduction is not needed for correctness — without it, the
/// ciphertext decrypts to the same plaintext — but for security under the correct parameters:
/// with it, the encryption can be proven secure assuming hardness of RLWE with
/// modulus-to-noise ratio `error / (Q / b^(digits - 1))`, whereas interpreting a plain BFV
/// encryption as the single-digit dBFV ciphertext `[bfv_enc(m), 0, ..., 0]` requires hardness
/// at the ratio `error / Q`. At the same security level this allows a ciphertext modulus
/// larger by the factor `b^(digits - 1)` (e.g. `2^48` for `t = 2^64`, `b = 2^16`).
///
/// Accordingly, `log2_alpha` is the modulus-to-noise ratio of the *combined* RLWE sample, i.e.
/// the assumption the ciphertext as a whole relies on (see [`crate::bfv`]); each digit is
/// encrypted with the correspondingly smaller error `σ = Q / (b^(digits - 1) α)`, so that the
/// combined noise `Σ b^i e_i ≈ b^(digits - 1) σ` realizes exactly the ratio `α`. Since the
/// per-digit error is floored at [`MIN_SIGMA`], a ciphertext modulus beyond
/// `b^(digits - 1) α MIN_SIGMA` stops yielding the digit gain and starts raising the fresh
/// noise instead — the parametrization thus exposes when the gain has been spent, rather than
/// silently weakening the assumption.
///
/// The security argument: the reduction moves the ciphertext (viewed digit-wise as a
/// polynomial in a formal variable `B`, scale-invariantly) only by multiples of `B - b` and
/// `B^e`, so a *canonical* reduction publishes a function of the residue class alone, which is
/// determined by the combined pair `(Σ b^i a_i, Σ b^i (a_i s + e_i) + Δm)` — a single RLWE
/// sample with noise `Σ b^i e_i ~ b^(digits - 1) error`. Note that the fresh noise in the
/// *low* digits is essential: without it the combined noise is supported on `b^(digits - 1) Z`
/// and scaling by `b^-(digits - 1) mod Q` maps the sample back to an instance with ratio
/// `error / Q`.
///
/// # Security caveat (research code)
///
/// The proof requires the final reduction to output a *unique canonical* representative of
/// the residue class, and [`dbfv_reduce()`] only *almost* does:
///  - [`split_overflow()`] guarantees `|x_lo| <= 2bΔ` — a window four times wider than the `bΔ`
///    step between valid decompositions — and the choice within the window is made by the
///    BEHZ-style correction `α̂ = round(Σ_p lift_p round(γ/p) / γ)`, whose error radius
///    `len·max_p/(2γ) ≈ 2^-4` makes the rounding outcome depend on the fine RNS-lift structure of
///    the digit value for a constant fraction of coefficients. Class-equivalent inputs (which
///    differ by `bΔ h` in a digit, i.e. by only `~ h(Δt - Q)/Q` in the fractional position the
///    correction rounds) can therefore reduce to *different* representatives, so the published
///    ciphertext carries a small selection pattern that is a deterministic function of the
///    individual `a_i s + e_i` beyond the combined RLWE sample — information the security reduction
///    cannot simulate.
///  - Exact integer arithmetic (`Δt - Q ≠ 0`) additionally perturbs class-invariance by
///    `O(t)`-sized terms; these are statistically negligible whenever `Q >> 2^λ t²` and are not a
///    concern for the parameters used here.
///
/// For a provable instantiation, the encryption-time reduction (only; in-circuit reductions
/// act on already-published randomness and need no canonicity for IND-CPA) would have to use
/// a unique split, e.g. the exact `x_hi = round(cmod_Q(x)/(bΔ))` per coefficient, or
/// equivalently derive the digits by an exact canonical decomposition of the combined pair.
/// This is research code, so the almost-canonical [`dbfv_reduce()`] is used as-is.
#[instrument(skip_all)]
pub fn dbfv_enc<B: Backend, R: FnMut() -> u64>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    base: &RNSBaseIndices,
    ptxt_pb: &[i128],
    sk: &SecretKey<B>,
    rng: &mut R,
    log2_alpha: f64,
) -> DBFVCiphertext<B> {
    let digits = modulus.digits();
    let log2_alpha_digit = dbfv_log2_alpha_per_digit(modulus, log2_alpha);
    // digit 0 is a BFV encryption of the full message, i.e. an encryption of zero with Δm
    // added to its first component
    let digit0 = bfv_enc(ring, base, modulus.t(), ptxt_pb, sk, rng, log2_alpha_digit);
    if digits == 1 {
        // a single digit is a plain BFV ciphertext, for which the reduction is a no-op
        return digit0;
    }
    let mut ct = ring.allocate_in_basis(2 * digits, base, Basis::CRTBasis);
    ring.assign(ct.data.as_mut().slice(0..2, .., .., ..), digit0.data.as_ref());
    for i in 1..digits {
        let digit_i = bfv_enc_zero(ring, base, sk, rng, log2_alpha_digit);
        ring.assign(ct.data.as_mut().slice(2 * i..2 * i + 2, .., .., ..), digit_i.data.as_ref());
    }
    ring.crt_to_pb(&mut ct);
    let mut result = dbfv_reduce(ring, modulus, ct);
    ring.pb_to_crt(&mut result);
    return result;
}

/// Transforms all digits to CRT basis (a single HAL-level transform over the whole batch).
pub fn dbfv_pb_to_crt<B: Backend>(ring: &RingContext<B>, ct: &mut DBFVCiphertext<B>) { ring.pb_to_crt(ct); }

/// Transforms all digits to period basis (a single HAL-level transform over the whole batch).
pub fn dbfv_crt_to_pb<B: Backend>(ring: &RingContext<B>, ct: &mut DBFVCiphertext<B>) { ring.crt_to_pb(ct); }

/// Decrypts a dBFV ciphertext (in CRT basis).
#[instrument(skip_all)]
pub fn dbfv_dec<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    ct: &DBFVCiphertext<B>,
    sk: &SecretKey<B>,
) -> Vec<i128> {
    assert_eq!(Basis::CRTBasis, ct.basis);
    let d = dbfv_digit_count(ct);
    assert!(d <= modulus.digits());
    let mut result = vec![0i128; ring.n()];
    for i in 0..d {
        let part = bfv_dec(ring, modulus.t(), dbfv_digit(ct, i), sk);
        for j in 0..ring.n() {
            // for t close to 2^126, the sum of the digit contributions (each bounded by t/2)
            // can exceed the i128 range; since t divides 2^128, wrapping keeps it correct mod t
            result[j] = result[j]
                .wrapping_add(cmod(part[j], 1 << (modulus.log2_t() - i * modulus.log2_b())) << (i * modulus.log2_b()));
        }
    }
    for x in &mut result {
        *x = cmod_pow2(*x, modulus.log2_t());
    }
    return result;
}

pub fn dbfv_log2_noise_pbinf_norm<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    ct: &DBFVCiphertext<B>,
    sk: &SecretKey<B>,
) -> Vec<f64> {
    assert_eq!(Basis::CRTBasis, ct.basis);
    (0..dbfv_digit_count(ct)).map(|i| bfv_log2_noise_pbinf_norm(ring, modulus.t(), dbfv_digit(ct, i), sk)).collect()
}

/// The precomputed constants of [`split_overflow()`] for one RNS base and one power-of-two
/// modulus `R = 2^log2_r`. Instances are cached in the [`RingContext`] (keyed by the dBFV
/// modulus configuration and the RNS base); see [`split_overflow_data()`].
pub(crate) struct SplitOverflowData<B: Backend> {
    /// cache key: the digit size of the dBFV modulus this data was created for
    log2_b: usize,
    /// cache key: the total precision of the dBFV modulus this data was created for
    log2_t: usize,
    /// cache key: the RNS base this data was created for
    rns_base: Box<RNSBaseIndices>,
    log2_r: usize,
    /// the lifts of `R * (Q/p)^-1 mod p`, see [`RingContext::q_over_p_inv_tensor()`]
    r_q_over_p_inv: TensorI64Owned<B>,
    /// shape `[1, limbs + 1, r, 1]`: the rows `0..limbs` hold the 64-bit limbs of
    /// `Q/p mod 2^log2_r` (see [`split_i128_limbs()`]), the last row holds `round(γ / p)`
    /// (`γ = 2^log2_gamma`) per source prime
    matrix: TensorI64Owned<B>,
    log2_gamma: usize,
    /// `-(Q^-1) mod 2^log2_r`
    neg_q_inv_mod_r: i128,
    /// `Q mod 2^log2_r`
    q_mod_r: i128,
    /// entry `k` holds `mont^2(-b * Δ * 2^(64 k))` (see
    /// [`RingContext::scalar_tensor_big_mont2()`]), the factor for the `k`-th overflow limb
    neg_b_delta_mont2: Vec<TensorI64Owned<B>>,
}

/// Computes the inverse of an odd `q` modulo `2^128` by Newton iteration.
fn inv_mod_2_128(q: i128) -> i128 {
    let mut x: i128 = q; // correct mod 8
    for _ in 0..6 {
        x = x.wrapping_mul(2i128.wrapping_sub(q.wrapping_mul(x)));
    }
    return x;
}

impl<B: Backend> SplitOverflowData<B> {
    /// The number of 64-bit limbs of the overflow representation: values modulo
    /// `R = 2^log2_r` fit into a single `i64` limb for `log2_r <= 62`; larger `R` (up to
    /// `2^126`) uses two limbs, see [`Backend::bconv_correct_pow2_wide()`].
    fn limbs(&self) -> usize { if self.log2_r <= 62 { 1 } else { 2 } }

    fn new(ring: &RingContext<B>, modulus: DBFVModulus, base: &[usize]) -> Self {
        let log2_r = modulus.log2_t() - modulus.log2_b();
        assert!(log2_r >= 1 && log2_r <= 126, "t/b must be between 2 and 2^126");
        let limbs = if log2_r <= 62 { 1 } else { 2 };
        let q = ring.modulus_of(base);
        let r_big = ZZbig.power_of_two(log2_r);

        let max_p = base.iter().map(|i| ring.rns_factor_prime(*i)).max().unwrap();
        let min_p = base.iter().map(|i| ring.rns_factor_prime(*i)).min().unwrap();
        let log2_gamma = usize::max(64, ZZi64.abs_log2_ceil(&(4 * base.len() as i64 * max_p + 1)).unwrap());
        let gamma = ZZbig.power_of_two(log2_gamma);
        // neither of the two inner products computed by the matmul must overflow an i128: the
        // correction row exactly, the value row modulo 2^128 (which wrapping arithmetic gives)
        assert!(
            ((1i128 << log2_gamma) / min_p as i128 + 1)
                .checked_mul(max_p as i128)
                .and_then(|x| x.checked_mul(base.len() as i128))
                .is_some()
        );

        let q_over_p_mod_r = base
            .iter()
            .map(|i| {
                let q_over_p = ZZbig.checked_div(&q, &int_cast(ring.rns_factor_prime(*i), ZZbig, ZZi64)).unwrap();
                int_cast(ZZbig.euclidean_rem(q_over_p, &r_big), ZZi128, ZZbig)
            })
            .collect::<Vec<_>>();
        let gamma_over_p = base
            .iter()
            .map(|i| {
                i64::try_from(int_cast(
                    ZZbig.rounded_div(ZZbig.clone_el(&gamma), &int_cast(ring.rns_factor_prime(*i), ZZbig, ZZi64)),
                    ZZi128,
                    ZZbig,
                ))
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut matrix_values = Vec::with_capacity((limbs + 1) * base.len());
        matrix_values.extend(q_over_p_mod_r.iter().map(|c| split_i128_limbs(*c).0));
        if limbs == 2 {
            matrix_values.extend(q_over_p_mod_r.iter().map(|c| split_i128_limbs(*c).1));
        }
        matrix_values.extend(gamma_over_p);
        let mut matrix = ring.backend().hal_allocate_i64([1, limbs + 1, base.len(), 1]);
        ring.backend().hal_import(matrix.as_mut(), &matrix_values);

        let q_mod_r = int_cast(ZZbig.euclidean_rem(ZZbig.clone_el(&q), &r_big), ZZi128, ZZbig);
        // the inverse mod 2^log2_r only depends on q mod 2^log2_r, so inverting the residue
        // mod 2^128 and reducing gives the correct value
        let neg_q_inv_mod_r = (-inv_mod_2_128(q_mod_r)).rem_euclid(1i128 << log2_r);
        let delta = ZZbig.rounded_div(ZZbig.clone_el(&q), &int_cast(modulus.t(), ZZbig, ZZi128));
        let neg_b_delta = ZZbig.negate(ZZbig.mul(delta, int_cast(modulus.b(), ZZbig, ZZi128)));
        return Self {
            log2_b: modulus.log2_b(),
            log2_t: modulus.log2_t(),
            rns_base: base.to_vec().into_boxed_slice(),
            log2_r,
            r_q_over_p_inv: ring.q_over_p_inv_tensor(base, 1i128 << log2_r),
            matrix,
            log2_gamma,
            neg_q_inv_mod_r,
            q_mod_r,
            neg_b_delta_mont2: (0..limbs)
                .map(|k| {
                    let scaled = ZZbig.mul(ZZbig.clone_el(&neg_b_delta), ZZbig.power_of_two(64 * k));
                    ring.scalar_tensor_big_mont2(base, &scaled)
                })
                .collect(),
        };
    }
}

/// The [`SplitOverflowData`] for the given dBFV modulus and RNS base, from the cache in the
/// [`RingContext`] (creating and caching it on first use).
fn split_overflow_data<'a, B: Backend>(
    ring: &'a RingContext<B>,
    modulus: DBFVModulus,
    base: &RNSBaseIndices,
) -> MappedRwLockReadGuard<'a, SplitOverflowData<B>> {
    let matches = |data: &SplitOverflowData<B>| {
        data.log2_b == modulus.log2_b() && data.log2_t == modulus.log2_t() && &*data.rns_base == base
    };
    let read = ring.split_overflow_cache.read().unwrap();
    if let Some((idx, _)) = read.iter().enumerate().find(|(_, data)| matches(data)) {
        return RwLockReadGuard::map(read, |data| &data[idx]);
    }
    drop(read);
    let mut write = ring.split_overflow_cache.write().unwrap();
    let result = SplitOverflowData::new(ring, modulus, base);
    let idx = write.len();
    write.push(result);
    drop(write);
    let read = ring.split_overflow_cache.read().unwrap();
    return RwLockReadGuard::map(read, |data| &data[idx]);
}

/// Computes the map `x -> (x_lo, x_hi)` with `x = x_lo + (b Δ) x_hi mod Q` and
/// `|x_lo| <= 2 b Δ`, where `Δ = round(Q/t)` and `x_hi` is a plain integer vector modulo
/// `R = t/b`. The input is a whole batch (as a raw tensor slice) in period basis; `x_lo`
/// replaces the input in-place, and `x_hi` is returned coefficient-wise as a
/// `[count, limbs, deg_Lʼ, deg_K]` tensor of plain values in `(-R/2, R/2]`, split into one or
/// two 64-bit limbs per [`split_i128_limbs()`] (`limbs == 1` for `R <= 2^62`).
///
/// All computation happens through the backend: the per-coefficient overflow is a base
/// conversion to the power-of-two modulus `R` in the style of [BEHZ16] (see
/// [`crate::ring::BConvCache`]), realized as one [`Backend::matmul_i64()`] computing the
/// value and correction inner products, followed by one [`Backend::bconv_correct_pow2()`]
/// (resp. [`Backend::bconv_correct_pow2_wide()`] for two-limb `R`).
///
/// The decomposition is *not unique* (the `2bΔ` window admits several valid `x_hi`, and the
/// approximate BEHZ correction decides between them based on the fine RNS-lift structure of
/// `x`); this is irrelevant for correctness, but matters for the provable security of fresh
/// encryptions — see the security caveat on [`dbfv_enc()`].
#[instrument(skip_all)]
fn split_overflow<B: Backend>(
    ring: &RingContext<B>,
    data: &SplitOverflowData<B>,
    x: TensorI64Mut<B>,
    x_bound: u128,
    base: &RNSBaseIndices,
) -> TensorI64Owned<B> {
    let shape = TensorMut::shape(&x);
    assert_eq!(base.len(), shape[1]);
    ring.assert_mul_mont_safe(x_bound, base);
    let limbs = data.limbs();
    let mods = ring.mods_for(base);

    // plain lifts of (x * R) * (Q/p)^-1 mod p, in (-p, p)
    let mut lifts = ring.backend().hal_allocate_i64(shape);
    ring.backend().hal_assign(lifts.as_mut(), x.as_const());
    ring.backend().hal_mul_mont64(lifts.as_mut(), data.r_q_over_p_inv.as_ref(), &mods);

    // per coefficient: Σ_p lift_p (Q/p) mod R (in limbs), and the γ-scaled estimate of the
    // number of multiples of Q contained in Σ_p lift_p (Q/p)
    let mut sums = ring.backend().hal_allocate_i128([shape[0], limbs + 1, shape[2], shape[3]]);
    ring.backend().hal_matmul_i64(sums.as_mut(), data.matrix.as_ref(), lifts.as_ref());

    // overflow = -Q^-1 (Σ_p lift_p (Q/p) - correction * Q) mod R, per [BEHZ16]
    let mut overflow = ring.backend().hal_allocate_i64([shape[0], limbs, shape[2], shape[3]]);
    if limbs == 1 {
        ring.backend().hal_bconv_correct_pow2(
            overflow.as_mut(),
            sums.as_ref(),
            i64::try_from(data.q_mod_r).unwrap(),
            i64::try_from(data.neg_q_inv_mod_r).unwrap(),
            data.log2_gamma,
            data.log2_r,
        );
    } else {
        ring.backend().hal_bconv_correct_pow2_wide(
            overflow.as_mut(),
            sums.as_ref(),
            data.q_mod_r,
            data.neg_q_inv_mod_r,
            data.log2_gamma,
            data.log2_r,
        );
    }

    // x_lo = x + overflow * (-b Δ), with overflow = Σ_k limb_k 2^(64 k) folded into the
    // per-limb factors mont^2(-b Δ 2^(64 k)); the limb values are plain (not montgomery-form
    // and possibly much larger than the RNS factors), hence the unbounded fma
    let mut acc = ring.allocate_wide(shape[0], base);
    for k in 0..limbs {
        acc.fma_unbounded_all(
            ring.backend(),
            overflow.as_ref().slice(.., k..k + 1, .., ..),
            data.neg_b_delta_mont2[k].as_ref(),
        );
    }
    acc.add_all(ring.backend(), x.as_const(), x_bound);
    ring.reduce_wide_into_tensor(x, &acc);
    return overflow;
}

/// Performs the dBFV-reduction: reduces every digit modulo `b Δ` and propagates the carry to
/// the next digit. All overflows of digits at index `>= d` are dropped. Input and output are
/// in period basis.
///
/// The carry chain is deliberately kept sequential (each digit is split *after* receiving the
/// carry from below), even though a single-pass parallel variant (split all digits at once,
/// then add the downscaled overflows to the one-higher digits) would allow larger, batched
/// backend calls. The parallel variant does not re-split a digit after it receives its carry,
/// so its output digits are bounded by `2bΔ + Δ·|overflow of the digit below|` instead of
/// `2bΔ`; since the overflow scales with the *input* digit magnitude `M` as `M/b`, iterated
/// multiplication drives the digit magnitude through `M ↦ 2b + d·M²/b`, which diverges for
/// every `b` and `d ≥ 1` (the map `c ↦ 2 + d·c²` on `c = M/b` has no fixed point). The
/// resulting per-multiplication noise amplification empirically breaks the digit-extraction
/// circuits (homomorphic floor division and bootstrapping), so the sequential chain — whose
/// output digits are always bounded by `2bΔ` — is required for correctness.
#[instrument(skip_all)]
pub fn dbfv_reduce<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    ct: DBFVCiphertext<B>,
) -> DBFVCiphertext<B> {
    let d = dbfv_digit_count(&ct);
    assert!(d <= modulus.digits());
    if modulus.digits() == 1 || d == 0 {
        return ct;
    }
    assert_eq!(Basis::PeriodBasis, ct.basis);
    let base = ct.rns_base.to_vec();
    let data = split_overflow_data(ring, modulus, &base);
    if modulus.digits() == 1 {
        let mut ct = ct;
        split_overflow(ring, &data, ct.data.as_mut(), ct.bound, &base);
        ct.bound = 1;
        return ct;
    }

    let delta = ZZbig.rounded_div(ring.modulus_of(&base), &int_cast(modulus.t(), ZZbig, ZZi128));
    let delta_mont2 = (0..data.limbs())
        .map(|k| {
            let scaled = ZZbig.mul(ZZbig.clone_el(&delta), ZZbig.power_of_two(64 * k));
            ring.scalar_tensor_big_mont2(&base, &scaled)
        })
        .collect::<Vec<_>>();

    let mut result = ring.allocate(2 * modulus.digits(), &base);
    ring.assign(result.data.as_mut().slice(0..2 * d, .., .., ..), ct.data.as_ref());
    let mut carry: Option<TensorI64Owned<B>> = None;
    for i in 0..modulus.digits() {
        // rows copied from the (possibly unreduced) input have its bound; rows beyond the
        // input digits are zero
        let input_row_bound = if i < d { ct.bound } else { 1 };
        if let Some(carry) = &carry {
            // digit_i += Δ * carry; the carry limbs are plain integers, hence unbounded fma
            let mut acc = ring.allocate_wide(2, &base);
            for (k, factor) in delta_mont2.iter().enumerate() {
                acc.fma_unbounded_all(ring.backend(), carry.as_ref().slice(.., k..k + 1, .., ..), factor.as_ref());
            }
            acc.add_all(ring.backend(), result.data.as_ref().slice(2 * i..2 * i + 2, .., .., ..), input_row_bound);
            ring.reduce_wide_into(result.rows_mut(2 * i..2 * i + 2), &acc);
        }
        // after a carry the digit row was rewritten with reduced values
        let row_bound = if carry.is_none() { input_row_bound } else { 1 };
        carry = Some(split_overflow(
            ring,
            &data,
            result.data.as_mut().slice(2 * i..2 * i + 2, .., .., ..),
            row_bound,
            &base,
        ));
    }
    // every digit row was rewritten by its split_overflow, so the result is almost-reduced
    result.bound = 1;
    return result;
}

/// Homomorphically multiplies two dBFV ciphertexts (in period basis); the result is in
/// `output_basis` (see [`relin()`]: a period-basis output is unreduced, bound 2, which is the
/// natural choice when the result is subsequently reduced ([`dbfv_reduce()`]) or fed into
/// inner-product accumulation). The digit products are computed as a convolution of the digit
/// vectors, with all digit-pair products at the same output index sharing one
/// relinearization.
#[instrument(skip_all)]
pub fn dbfv_hom_mul<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    lhs: DBFVCiphertext<B>,
    rhs: DBFVCiphertext<B>,
    rk: &KeySwitchKey<B>,
    output_basis: Basis,
) -> DBFVCiphertext<B> {
    assert_eq!(Basis::PeriodBasis, lhs.basis);
    assert_eq!(Basis::PeriodBasis, rhs.basis);
    assert!(dbfv_digit_count(&lhs) <= modulus.digits());
    assert!(dbfv_digit_count(&rhs) <= modulus.digits());
    let lhs_red = dbfv_reduce(ring, modulus, lhs);
    let rhs_red = dbfv_reduce(ring, modulus, rhs);
    if dbfv_digit_count(&lhs_red) == 0 {
        return lhs_red;
    } else if dbfv_digit_count(&rhs_red) == 0 {
        return rhs_red;
    }
    let base_len = lhs_red.rns_base.len();
    let lhs_lifted = lift_to_extended(ring, &lhs_red, None);
    let rhs_lifted = lift_to_extended(ring, &rhs_red, None);
    return dbfv_mul_convolve(ring, modulus, &lhs_lifted, &rhs_lifted, base_len, rk, output_basis);
}

/// Homomorphically squares a dBFV ciphertext (in period basis); the result is in
/// `output_basis`, see [`dbfv_hom_mul()`].
#[instrument(skip_all)]
pub fn dbfv_hom_square<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    val: DBFVCiphertext<B>,
    rk: &KeySwitchKey<B>,
    output_basis: Basis,
) -> DBFVCiphertext<B> {
    assert_eq!(Basis::PeriodBasis, val.basis);
    assert!(dbfv_digit_count(&val) <= modulus.digits());
    let val_red = dbfv_reduce(ring, modulus, val);
    if dbfv_digit_count(&val_red) == 0 {
        return val_red;
    }
    let base_len = val_red.rns_base.len();
    let lifted = lift_to_extended(ring, &val_red, None);
    return dbfv_mul_convolve(ring, modulus, &lifted, &lifted, base_len, rk, output_basis);
}

#[instrument(skip_all)]
fn dbfv_mul_convolve<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    lhs_lifted: &RingEls<B>,
    rhs_lifted: &RingEls<B>,
    base_len: usize,
    rk: &KeySwitchKey<B>,
    output_basis: Basis,
) -> DBFVCiphertext<B> {
    let d1 = dbfv_digit_count(lhs_lifted);
    let d2 = dbfv_digit_count(rhs_lifted);
    let lifted_base = &lhs_lifted.rns_base;
    let d_out = usize::min(modulus.digits(), d1 + d2 - 1);

    // all 3-component digit products in one wide batch (rows 3k..3k+3 belong to
    // output digit k), so that basis transform and rescaling are single coarse HAL calls
    let mut acc = ring.allocate_wide(3 * d_out, lifted_base);
    for k in 0..d_out {
        let mut terms0 = Vec::new();
        let mut terms1 = Vec::new();
        let mut terms2 = Vec::new();
        for j in k.saturating_sub(d2 - 1)..=usize::min(k, d1 - 1) {
            terms0.push((
                lhs_lifted.data.as_ref().slice(2 * j..2 * j + 1, .., .., ..),
                lhs_lifted.bound,
                rhs_lifted.data.as_ref().slice(2 * (k - j)..2 * (k - j) + 1, .., .., ..),
                rhs_lifted.bound,
            ));
            terms1.push((
                lhs_lifted.data.as_ref().slice(2 * j..2 * j + 1, .., .., ..),
                lhs_lifted.bound,
                rhs_lifted.data.as_ref().slice(2 * (k - j) + 1..2 * (k - j) + 2, .., .., ..),
                rhs_lifted.bound,
            ));
            terms1.push((
                lhs_lifted.data.as_ref().slice(2 * j + 1..2 * j + 2, .., .., ..),
                lhs_lifted.bound,
                rhs_lifted.data.as_ref().slice(2 * (k - j)..2 * (k - j) + 1, .., .., ..),
                rhs_lifted.bound,
            ));
            terms2.push((
                lhs_lifted.data.as_ref().slice(2 * j + 1..2 * j + 2, .., .., ..),
                lhs_lifted.bound,
                rhs_lifted.data.as_ref().slice(2 * (k - j) + 1..2 * (k - j) + 2, .., .., ..),
                rhs_lifted.bound,
            ));
        }
        acc.inner_product(ring.backend(), 3 * k..3 * k + 1, &terms0);
        acc.inner_product(ring.backend(), 3 * k + 1..3 * k + 2, &terms1);
        acc.inner_product(ring.backend(), 3 * k + 2..3 * k + 3, &terms2);
    }
    let mut prod = ring.reduce_wide(&acc, Basis::CRTBasis);
    ring.crt_to_pb(&mut prod);
    let mut rescaled = rescale_to_base(ring, prod, base_len, modulus.t());

    let mut result = ring.allocate_in_basis(2 * d_out, &rescaled.rns_base, output_basis);
    for k in 0..d_out {
        let relinearized = relin(ring, rescaled.rows_mut(3 * k..3 * k + 3), rk, output_basis);
        ring.assign(result.data.as_mut().slice(2 * k..2 * k + 2, .., .., ..), relinearized.data.as_ref());
        result.bound = relinearized.bound;
    }
    return result;
}

#[cfg(test)]
mod tests {
    use feanor_math::homomorphism::Homomorphism;
    use feanor_math::ordered::OrderedRingStore;

    use super::*;
    use crate::field::field_192;
    use crate::hal::cpu::CpuBackend;
    use crate::ring::RingContext;

    fn check_split_overflow(seed: u64, modulus: DBFVModulus) {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = seed;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, modulus.log2_t());
        let base = ring.initial_rns_base();
        let n = ring.n();
        let coeffs = (0..n).map(|_| (rng() >> 20) as i128 - (1 << 43)).collect::<Vec<_>>();
        let x_small = ring.element_from_pb_coeffs(&base, &coeffs);
        // make coefficients uniform mod Q by adding large multiples
        let big = ZZbig.power_of_two(190);
        let mut x2 = ring.element_from_pb_coeffs(&base, &(0..n).map(|_| (rng() >> 40) as i128).collect::<Vec<_>>());
        ring.mul_scalar_big(&mut x2, &big);
        let mut acc = ring.allocate_wide(1, &base);
        ring.add(&mut acc, &x_small);
        ring.add(&mut acc, &x2);
        let mut x = ring.reduce_wide(&acc, Basis::PeriodBasis);
        let x_before = ring.element_lifts(&x, 0);

        let data = SplitOverflowData::new(&ring, modulus, &base);
        let limbs = data.limbs();
        let overflow = split_overflow(&ring, &data, x.data.as_mut(), x.bound, &base);
        let mut overflow_limbs = vec![0i64; limbs * n];
        ring.backend().hal_export(overflow.as_ref(), &mut overflow_limbs);
        let overflow_values =
            (0..n).map(|j| (0..limbs).map(|k| (overflow_limbs[k * n + j] as i128) << (64 * k)).sum::<i128>());

        let q = ring.modulus_of(&base);
        let delta = ZZbig.rounded_div(ZZbig.clone_el(&q), &int_cast(modulus.t(), ZZbig, ZZi128));
        let scale = ZZbig.mul(delta, int_cast(modulus.b(), ZZbig, ZZi128));
        let bound = ZZbig.int_hom().mul_ref_map(&scale, &2);
        let x_after = ring.element_lifts(&x, 0);
        for (j, overflow_value) in overflow_values.enumerate() {
            // x_before = x_after + overflow * (b Δ) mod q
            let reconstructed = ZZbig.euclidean_rem(
                ZZbig.add_ref_fst(&x_after[j], ZZbig.mul_ref_snd(int_cast(overflow_value, ZZbig, ZZi128), &scale)),
                &q,
            );
            let expected = ZZbig.euclidean_rem(ZZbig.clone_el(&x_before[j]), &q);
            assert!(ZZbig.eq_el(&expected, &reconstructed), "coefficient {}", j);
            assert!(ZZbig.is_lt(&ZZbig.abs(ZZbig.clone_el(&x_after[j])), &bound), "coefficient {} too large", j);
        }
    }

    #[test]
    fn test_split_overflow() { check_split_overflow(10, DBFVModulus::new(8, 3)); }

    /// `R = t/b = 2^100` exceeds 64 bits, so the overflow digit is returned in two limbs
    #[test]
    fn test_split_overflow_large_r() { check_split_overflow(18, DBFVModulus::new_incomplete(8, 108)); }

    #[test]
    fn test_split_overflow_large_r_reduce_config() { check_split_overflow(20, DBFVModulus::new(27, 4)); }

    /// a plain BFV ciphertext is a valid single-digit dBFV ciphertext, which [`dbfv_dec()`]
    /// accepts directly
    #[test]
    fn test_dbfv_dec_of_bfv_enc() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 11u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let ct = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, DBFVModulus::new(16, 4), &ct, &sk));
        assert_eq!(5 << 31, result);
    }

    /// the proper dBFV encryption: a fresh zero-encryption per digit plus the final reduction,
    /// so the ciphertext comes out with all digits populated (and reduced)
    #[test]
    fn test_dbfv_enc_dec() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 22u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(16, 4);
        let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 - 96) << 41).collect();
        let ct = dbfv_enc(&ring, modulus, &base, &m, &sk, &mut rng, log2_alpha);
        assert_eq!(modulus.digits(), dbfv_digit_count(&ct));
        assert_eq!(Basis::CRTBasis, ct.basis);
        assert_eq!(m, dbfv_dec(&ring, modulus, &ct, &sk));
    }

    /// The dBFV digit structure — not a larger error — is what absorbs a `b^(digits - 1)` times
    /// larger ciphertext modulus: at a given assumption the individual digit encryptions are
    /// created at a `b^(digits - 1)` times larger ratio, hence with correspondingly less noise
    /// than a plain BFV encryption relying on the same assumption.
    #[test]
    fn test_dbfv_enc_digit_ratio_carries_the_gain() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let modulus = DBFVModulus::new(16, 4);
        let log2_gain = (modulus.log2_b() * (modulus.digits() - 1)) as f64;
        assert_eq!(48., log2_gain);

        // the assumption at which a plain BFV encryption would carry the minimal error ...
        let log2_alpha_bfv = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        // ... is `log2_gain` bits stronger than the one a dBFV encryption needs for it
        let log2_alpha_dbfv = log2_alpha_bfv - log2_gain;
        assert_eq!(log2_alpha_bfv, dbfv_log2_alpha_per_digit(modulus, log2_alpha_dbfv));
        let sigma_digit = sigma_for_log2_alpha(&ring, &base, dbfv_log2_alpha_per_digit(modulus, log2_alpha_dbfv));
        assert!((sigma_digit - MIN_SIGMA).abs() < 1e-9);

        // at that same (weaker) assumption a BFV encryption needs `2^log2_gain` times the error
        let sigma_bfv = sigma_for_log2_alpha(&ring, &base, log2_alpha_dbfv);
        assert!((sigma_bfv.log2() - (MIN_SIGMA.log2() + log2_gain)).abs() < 1e-6);

        // and the encryption is correct at that weaker assumption, i.e. the gain is real
        let mut state = 25u64;
        let mut rng = move || splitmix64(&mut state);
        let sk = gen_sk(&ring, &mut rng);
        let ct = dbfv_enc(&ring, modulus, &base, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha_dbfv);
        assert_eq!(modulus.digits(), dbfv_digit_count(&ct));
        assert_eq!(5 << 31, decode_scalar(&ring, modulus.t(), &dbfv_dec(&ring, modulus, &ct, &sk)));
    }

    /// dBFV encryption for an incomplete modulus, and the degenerate single-digit case
    #[test]
    fn test_dbfv_enc_dec_incomplete_and_single_digit() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 23u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);

        let modulus = DBFVModulus::new_incomplete(16, 60);
        let ct = dbfv_enc(&ring, modulus, &base, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        assert_eq!(modulus.digits(), dbfv_digit_count(&ct));
        assert_eq!(5 << 31, decode_scalar(&ring, modulus.t(), &dbfv_dec(&ring, modulus, &ct, &sk)));

        // a single digit stays a plain BFV ciphertext
        let modulus = DBFVModulus::new(16, 1);
        let ct = dbfv_enc(&ring, modulus, &base, &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        assert_eq!(1, dbfv_digit_count(&ct));
        assert_eq!(2025, decode_scalar(&ring, modulus.t(), &dbfv_dec(&ring, modulus, &ct, &sk)));
    }

    /// dBFV-encrypted ciphertexts (all digits populated) feed through homomorphic
    /// multiplication like the single-digit embeddings do
    #[test]
    fn test_dbfv_enc_hom_mul() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 24u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let modulus = DBFVModulus::new(16, 4);
        let mut ct1 = dbfv_enc(&ring, modulus, &base, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        let mut ct2 = dbfv_enc(&ring, modulus, &base, &encode_scalar(&ring, 7 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct1);
        dbfv_crt_to_pb(&ring, &mut ct2);
        let product = dbfv_hom_mul(&ring, modulus, ct1, ct2, &rk, Basis::CRTBasis);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, modulus, &product, &sk));
        assert_eq!(cmod(35 << 62, 1 << 64), result);
    }

    #[test]
    fn test_dbfv_reduce() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 12u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let mut ct = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct);
        let mut ct = dbfv_reduce(&ring, DBFVModulus::new(16, 4), ct);
        assert_eq!(4, dbfv_digit_count(&ct));
        dbfv_pb_to_crt(&ring, &mut ct);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, DBFVModulus::new(16, 4), &ct, &sk));
        assert_eq!(5 << 31, result);
    }

    /// `R = t/b = 2^81` exceeds 64 bits, so reduction propagates two-limb carries. Note that
    /// large `t` needs `Q` considerably larger than `t^2`: splitting shifts a digit by
    /// multiples of `Q`, each of which perturbs the digit noise by `ε = Δt - Q` (`|ε| <= t/2`),
    /// and decryption requires the total to stay below `Δ/2 = Q/(2t)`.
    #[test]
    fn test_dbfv_reduce_large_r() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 19u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 260..280, 108);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let mut ct = bfv_enc(&ring, &base, 1 << 108, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct);
        let mut ct = dbfv_reduce(&ring, DBFVModulus::new(27, 4), ct);
        assert_eq!(4, dbfv_digit_count(&ct));
        dbfv_pb_to_crt(&ring, &mut ct);
        let result = decode_scalar(&ring, 1 << 108, &dbfv_dec(&ring, DBFVModulus::new(27, 4), &ct, &sk));
        assert_eq!(5 << 31, result);
    }

    #[test]
    fn test_dbfv_mul() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 13u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let mut ct1 = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        let mut ct2 = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 7 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct1);
        dbfv_crt_to_pb(&ring, &mut ct2);
        let product = dbfv_hom_mul(&ring, DBFVModulus::new(16, 4), ct1, ct2, &rk, Basis::CRTBasis);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, DBFVModulus::new(16, 4), &product, &sk));
        assert_eq!(cmod(35 << 62, 1 << 64), result);
    }

    #[test]
    fn test_dbfv_square() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 14u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let mut ct = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct);
        let product = dbfv_hom_square(&ring, DBFVModulus::new(16, 4), ct, &rk, Basis::CRTBasis);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, DBFVModulus::new(16, 4), &product, &sk));
        assert_eq!(cmod(25 << 62, 1 << 64), result);
    }

    /// multiplication with period-basis output: the digits come back unreduced (bound 2) in
    /// period basis — the natural representation for the follow-up dBFV reduction, which
    /// accepts them directly
    #[test]
    fn test_dbfv_mul_period_output() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 21u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 64);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let modulus = DBFVModulus::new(16, 4);
        let mut ct1 = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 5 << 31), &sk, &mut rng, log2_alpha);
        let mut ct2 = bfv_enc(&ring, &base, 1 << 64, &encode_scalar(&ring, 7 << 31), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct1);
        dbfv_crt_to_pb(&ring, &mut ct2);

        let product = dbfv_hom_mul(&ring, modulus, ct1, ct2, &rk, Basis::PeriodBasis);
        assert_eq!(Basis::PeriodBasis, product.basis);
        assert_eq!(2, product.bound);
        // the unreduced period-basis product feeds directly into the dBFV reduction ...
        let mut reduced = dbfv_reduce(&ring, modulus, product);
        assert_eq!(1, reduced.bound);
        dbfv_pb_to_crt(&ring, &mut reduced);
        let result = decode_scalar(&ring, 1 << 64, &dbfv_dec(&ring, modulus, &reduced, &sk));
        assert_eq!(cmod(35 << 62, 1 << 64), result);
    }

    #[test]
    fn test_dbfv_exact_div_pow2() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 15u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let mut ct = bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, 5 << 16), &sk, &mut rng, log2_alpha);
        dbfv_crt_to_pb(&ring, &mut ct);
        let ct = dbfv_reduce(&ring, DBFVModulus::new(8, 4), ct);
        let mut ct = dbfv_exact_div_pow2(&ring, DBFVModulus::new(8, 4), ct, 5);
        dbfv_pb_to_crt(&ring, &mut ct);
        let result = decode_scalar(&ring, 1 << 27, &dbfv_dec(&ring, DBFVModulus::new_incomplete(8, 27), &ct, &sk));
        assert_eq!(5 << 11, result);
    }

    /// exact division by more than a whole digit (`log2_div > log2_b`), which drops
    /// `ceil(log2_div / log2_b)` digits
    #[test]
    fn test_dbfv_exact_div_pow2_multiple_digits() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 45u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 300..320, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(8, 4);
        for log2_div in [8, 9, 13, 16, 17, 24, 25, 31] {
            let value = 12345i128 << log2_div;
            let mut ct = bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, value), &sk, &mut rng, log2_alpha);
            dbfv_crt_to_pb(&ring, &mut ct);
            let ct = dbfv_reduce(&ring, modulus, ct);
            let mut res = dbfv_exact_div_pow2(&ring, modulus, ct, log2_div);
            let res_modulus = DBFVModulus::new_incomplete(8, 32 - log2_div);
            assert_eq!(usize::max(4usize.saturating_sub((log2_div - 1) / 8 + 1), 1), dbfv_digit_count(&res));
            // the result is canonical again after a reduction, and still holds the same value
            // (the reduction needs t > b, i.e. it is meaningless for a degenerate modulus)
            if res_modulus.log2_t() > res_modulus.log2_b() {
                res = dbfv_reduce(&ring, res_modulus, res);
            }
            dbfv_pb_to_crt(&ring, &mut res);
            let result = decode_scalar(&ring, res_modulus.t(), &dbfv_dec(&ring, res_modulus, &res, &sk));
            assert_eq!(cmod(12345, res_modulus.t()), result, "log2_div = {}", log2_div);
        }
    }

    /// exact division of an input whose digit values are far outside the canonical range — in
    /// particular a digit `0` much larger than `b`, and (for `log2_div = 17`) fewer digits than
    /// the merge would consume. Nothing may be discarded: the merged digit carries the exact
    /// value of all digits it absorbs.
    #[test]
    fn test_dbfv_exact_div_pow2_large_low_digit() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 46u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 300..320, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(8, 4);
        let total = |digits: &[i128]| digits.iter().enumerate().map(|(i, y)| y << (8 * i)).sum::<i128>();
        for log2_div in [9usize, 13, 17] {
            // digit 0 is ~b^2, and is adjusted so that the total value is divisible by 2^log2_div
            let mut digits = vec![128768i128, 777, 5, -3];
            digits[0] -= total(&digits).rem_euclid(1 << log2_div);
            let value = total(&digits);
            assert_eq!(0, value % (1 << log2_div));

            let mut ct = ring.allocate_in_basis(2 * digits.len(), &base, Basis::CRTBasis);
            for (i, y) in digits.iter().enumerate() {
                let part = bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, *y), &sk, &mut rng, log2_alpha);
                ring.assign(ct.data.as_mut().slice(2 * i..2 * i + 2, .., .., ..), part.data.as_ref());
            }
            assert_eq!(value, decode_scalar(&ring, 1 << 32, &dbfv_dec(&ring, modulus, &ct, &sk)));

            ring.crt_to_pb(&mut ct);
            let mut res = dbfv_exact_div_pow2(&ring, modulus, ct, log2_div);
            let res_modulus = DBFVModulus::new_incomplete(8, 32 - log2_div);
            dbfv_pb_to_crt(&ring, &mut res);
            let result = decode_scalar(&ring, res_modulus.t(), &dbfv_dec(&ring, res_modulus, &res, &sk));
            assert_eq!(cmod(value >> log2_div, res_modulus.t()), result, "log2_div = {}", log2_div);
        }
    }

    #[test]
    fn test_dbfv_hom_fma_scalar() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 16u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(8, 4);
        let ct = bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        let res = dbfv_hom_fma_scalar(&ring, modulus, &ct, 4321, dbfv_empty(&ring, &base, ct.basis));
        let result = decode_scalar(&ring, 1 << 32, &dbfv_dec(&ring, modulus, &res, &sk));
        assert_eq!(2025 * 4321, result);
    }

    /// an inner product of several ciphertexts with scalar coefficients, accumulated in one
    /// [`DBFVAccumulator`] with a single reduction at the end
    #[test]
    fn test_dbfv_inner_product() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 17u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(8, 4);

        let values: [i128; 3] = [2025, -311, 7777];
        let coeffs: [i128; 3] = [4321, 987, -55];
        let cts =
            values.map(|value| bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, value), &sk, &mut rng, log2_alpha));

        let mut acc = dbfv_allocate_accumulator(&ring, modulus, &base, Basis::CRTBasis);
        for (ct, coeff) in cts.iter().zip(coeffs) {
            dbfv_acc_fma_scalar(&ring, &mut acc, ct, coeff);
        }
        // additions of whole ciphertexts also go through the accumulator
        dbfv_acc_add(&ring, &mut acc, &cts[0]);
        dbfv_acc_sub(&ring, &mut acc, &cts[1]);
        let res = dbfv_finish_accumulator(&ring, &acc);

        let expected = values.iter().zip(coeffs).map(|(v, c)| v * c).sum::<i128>() + values[0] - values[1];
        let result = decode_scalar(&ring, 1 << 32, &dbfv_dec(&ring, modulus, &res, &sk));
        assert_eq!(cmod(expected, 1 << 32), result);
    }

    /// [`dbfv_acc_inner_product_scalar()`] must produce exactly the same accumulator as the
    /// chain of [`dbfv_acc_fma_scalar()`] calls it replaces (identical modular residues, not
    /// just the same decoded plaintext).
    #[test]
    fn test_dbfv_inner_product_scalar_matches_fma_chain() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 44u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 200..220, 32);
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let modulus = DBFVModulus::new(8, 4);

        let values: [i128; 4] = [2025, -311, 7777, 42];
        let coeffs: [i128; 4] = [4321, 987, -55, 130000];
        let cts =
            values.map(|value| bfv_enc(&ring, &base, 1 << 32, &encode_scalar(&ring, value), &sk, &mut rng, log2_alpha));

        // reference: a chain of fused multiply-adds
        let mut chain = dbfv_allocate_accumulator(&ring, modulus, &base, Basis::CRTBasis);
        for (ct, coeff) in cts.iter().zip(coeffs) {
            dbfv_acc_fma_scalar(&ring, &mut chain, ct, coeff);
        }
        let chain_res = dbfv_finish_accumulator(&ring, &chain);

        // the gathered inner product
        let mut acc = dbfv_allocate_accumulator(&ring, modulus, &base, Basis::CRTBasis);
        let terms = cts.iter().zip(coeffs).map(|(ct, coeff)| (ct, coeff)).collect::<Vec<_>>();
        dbfv_acc_inner_product_scalar(&ring, &mut acc, &terms);
        let ip_res = dbfv_finish_accumulator(&ring, &acc);

        assert_eq!(dbfv_digit_count(&chain_res), dbfv_digit_count(&ip_res));
        let mut a = vec![0; chain_res.data.as_ref().shape().iter().product()];
        let mut b = vec![0; ip_res.data.as_ref().shape().iter().product()];
        ring.backend().hal_export(chain_res.data.as_ref(), &mut a);
        ring.backend().hal_export(ip_res.data.as_ref(), &mut b);
        assert_eq!(a, b);
    }
}
