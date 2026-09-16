//! The ring `R_q = O_L / q O_L` for an RNS modulus `q`, on top of the tensor HAL.
//!
//! # Data layout
//!
//! A batch of ring elements is stored as a shape `[count, r, deg_Lʼ, deg_K]` tensor of
//! montgomery-form residues; entry `[i, j, k, l]` is a coefficient of the `i`-th element
//! modulo the `j`-th RNS factor. The coefficients of one element are either
//!  - w.r.t. the period basis ([`Basis::PeriodBasis`]): coefficient `(k, l)` corresponds to the
//!    Gaussian period `η_σ` with `σ` the element of [`FieldDescriptor::lq_transversal()`] at flat
//!    index `k * deg_K + l`, or
//!  - w.r.t. the CRT basis ([`Basis::CRTBasis`]): the values of the element modulo the primes
//!    `σ(𝔭)` of `O_L` over the RNS factor, in the same transversal order.
//!
//! Since both bases are normal bases, the Galois action permutes the coefficients in the same
//! way in either basis; see [`RingContext::apply_galois()`].
//!
//! # RNS factors
//!
//! Every RNS factor is a 57-bit prime `p ≡ 1 mod rou_order` (see
//! [`FieldDescriptor::rou_order()`]), which guarantees that `p` splits completely in `M` (so
//! the CRT basis of `O_L/(p)` exists) and that `Z/pZ` contains the roots of unity needed for
//! the group-NTTs over `Gal(Lʼ|Q)` (along dimension 2) and `Gal(K|Q)` (along dimension 3).
//!
//! The change of basis between period basis and CRT basis is the composition
//! `iNTT ∘ ((K̂_Lʼ ⊗ K̂_K) ⊙ _) ∘ NTT` of group-NTTs over `Gal(L|Q)` and a pointwise
//! multiplication with the NTT of the CRT kernel `K = K_Lʼ ⊗ K_K`, whose entries are the
//! reductions `η_σ mod 𝔭` of the periods themselves. Indeed, since the transversal order of
//! the CRT basis matches the transversal order of the period basis, the change of basis is the
//! `Gal(L|Q)`-circulant map `(x_σ)_σ -> (Σ_τ x_τ η_(στ) mod 𝔭)_σ`, i.e. a group convolution
//! with the kernel `(η_σ mod 𝔭)_σ`; the kernel factors into an `Lʼ`-part and a `K`-part since
//! `L = Lʼ ⊗ K` and `ζ_m = ζ_mʼ ζ_(m_K)`.
//!
//! # Value bounds
//!
//! All persistent data is kept in montgomery form. Every [`RingEls`], [`RingElsRef`] and
//! [`RingElsMut`] tracks a worst-case bound on its representatives, in multiples of the
//! respective RNS factor ([`RingEls::bound`]): a batch with bound `k` has all coefficients of
//! residues modulo `p` in `[-k p, k p]`. Every HAL operation with modular reduction produces
//! *almost-reduced* values (bound 1); unreduced values arise from [`RingContext::add_els()`]
//! and [`RingContext::sub_els()`], which add batches entry-wise *without* reduction (the
//! bounds add up). Operations assert the input bounds they can handle — in particular the
//! NTT-based basis transforms accept inputs up to bound 2, so a single deferred addition
//! (e.g. an unreduced relinearization output) costs nothing.
//!
//! For longer sums and products, [`WideRingEls`] holds ring elements with 128-bit entries:
//! fused multiply-adds and additions accumulate there without intermediate reductions, and
//! [`RingContext::reduce_wide()`] performs the single modular reduction back to an
//! (almost-reduced) [`RingEls`] at the end.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::{Arc, MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::integer::{BigIntRing, IntegerRingStore, int_cast};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::{El, RingStore};
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::zn_rns::Zn as ZnRns;
use tracing::instrument;

use crate::field::FieldDescriptor;
#[cfg(test)]
use crate::hal::cpu::CpuBackend;
use crate::hal::*;
#[cfg(test)]
use crate::{montgomery_repr, pos_repr};

#[allow(non_upper_case_globals)]
pub const ZZbig: BigIntRing = BigIntRing::RING;
#[allow(non_upper_case_globals)]
pub const ZZi64: StaticRing<i64> = StaticRing::RING;
#[allow(non_upper_case_globals)]
pub const ZZi128: StaticRing<i128> = StaticRing::RING;

/// the extended modulus is chosen to have at least `2 * log2(q) + log2(t) + EXTENDED_Q_EXTRA_BITS`
/// bits; these extra bits have to account for the period-basis expansion factor and any
/// sums/inner products performed in lifted form
const EXTENDED_Q_EXTRA_BITS: usize = 60;

/// Reduces `a` into `(-2^(log2_modulus - 1), 2^(log2_modulus - 1)]` modulo `2^log2_modulus`
/// (with `log2_modulus <= 126`). Unlike [`cmod()`], the result depends only on `a` modulo
/// `2^128`, so this correctly reduces sums that were accumulated with wrapping arithmetic.
pub fn cmod_pow2(a: i128, log2_modulus: usize) -> i128 {
    assert!(log2_modulus >= 1 && log2_modulus <= 126);
    let shift = 128 - log2_modulus as u32;
    let result = (a << shift) >> shift;
    return if result == -(1 << (log2_modulus - 1)) { result + (1 << log2_modulus) } else { result };
}

/// Reduces `a` into the symmetric interval `(-modulus/2, modulus/2]`-ish around zero.
pub fn cmod(a: i128, modulus: i128) -> i128 {
    let modulus_half = modulus / 2;
    let mut result = a % modulus;
    if result.abs() >= modulus_half {
        if result < 0 {
            result += modulus;
        } else if result > modulus_half {
            result -= modulus;
        }
    }
    return result;
}

/// The montgomery representation `x * 2^64 mod p`, reduced to `(-p, p)`.
pub fn mont_repr(x: i64, p: i64) -> i64 { (((x as i128) << 64) % p as i128) as i64 }

fn mul_mod(a: i64, b: i64, p: i64) -> i64 { ((a as i128 * b as i128) % p as i128) as i64 }

fn pow_mod(mut b: i64, mut e: u64, p: i64) -> i64 {
    let mut result = 1;
    b = b.rem_euclid(p);
    while e > 0 {
        if e & 1 == 1 {
            result = mul_mod(result, b, p);
        }
        b = mul_mod(b, b, p);
        e >>= 1;
    }
    return result;
}

/// The plain (non-montgomery) representative of a montgomery-form value `x`, in `[0, p)`.
pub(crate) fn from_mont_repr(x: i64, p: i64) -> i64 {
    let r_inv = pow_mod(mont_repr(1, p), (p - 2) as u64, p);
    return mul_mod(x.rem_euclid(p), r_inv, p);
}

/// A list of indices into the RNS factors of a [`RingContext`], defining the RNS base of a
/// (batch of) ring element(s).
pub type RNSBaseIndices = [usize];

fn largest_prime_leq_congruent_to_one(leq_than: i64, congruent_to_one_mod: i64) -> Option<i64> {
    assert!(leq_than > congruent_to_one_mod);
    let mut current = leq_than - (leq_than - 1) % congruent_to_one_mod;
    while !is_prime(&ZZi64, &current, 10) {
        current -= congruent_to_one_mod;
        if current <= 0 {
            return None;
        }
    }
    return Some(current);
}

/// Samples primes `≡ 1 mod congruent_to_one_mod` with `min_bits .. max_bits` total bits,
/// each prime having at most `max_bits_each_modulus` bits, starting with the given primes.
fn extend_sampled_primes(
    begin_with: &[i64],
    min_bits: usize,
    max_bits: usize,
    max_bits_each_modulus: usize,
    congruent_to_one_mod: i64,
) -> Option<Vec<i64>> {
    assert!(max_bits > min_bits);
    let mut result = begin_with.to_vec();
    let mut current_bits = result.iter().map(|i| (*i as f64).log2()).sum::<f64>();
    assert!((current_bits.floor() as usize) < max_bits);
    let mut current_upper_bound = ZZi64.power_of_two(max_bits_each_modulus);

    while current_bits < min_bits as f64 {
        let remaining_bits = min_bits as f64 - current_bits;
        let allowed_bits = if remaining_bits < max_bits_each_modulus as f64 {
            f64::min(max_bits as f64 - current_bits, max_bits_each_modulus as f64)
        } else {
            let required_number_of_primes = (remaining_bits / max_bits_each_modulus as f64).ceil();
            f64::min((max_bits as f64 - current_bits) / required_number_of_primes, max_bits_each_modulus as f64)
        };
        current_upper_bound = i64::min(current_upper_bound, ZZi64.power_of_two(allowed_bits.floor() as usize));

        let mut prime = largest_prime_leq_congruent_to_one(current_upper_bound, congruent_to_one_mod)?;
        current_upper_bound = prime - 1;
        while begin_with.iter().any(|p| *p == prime) {
            prime = largest_prime_leq_congruent_to_one(current_upper_bound, congruent_to_one_mod)?;
            current_upper_bound = prime - 1;
        }
        current_bits += (prime as f64).log2();
        result.push(prime);
    }
    return Some(result);
}

/// Whether ring element coefficients are w.r.t. the period basis or the CRT basis; see the
/// module-level documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    PeriodBasis,
    CRTBasis,
}

/// A batch of ring elements; see the module-level documentation for the data layout and the
/// representative bounds. Borrowed views into a batch (or a part of one) are
/// [`RingElsRef`] and [`RingElsMut`]; both slicing along the batch dimension
/// ([`RingEls::rows()`]) and restriction to a prefix of the RNS base
/// ([`RingEls::restrict_base()`]) are copy-free.
pub struct RingEls<B: Backend> {
    backend: PhantomData<B>,
    pub rns_base: Box<RNSBaseIndices>,
    pub basis: Basis,
    /// worst-case bound on the representatives, in multiples of the respective RNS factor:
    /// residues modulo `p` lie in `[-bound * p, bound * p]`. Almost-reduced batches (the
    /// result of every reducing operation) have bound 1; unreduced additions
    /// ([`RingContext::add_els()`]) increase it.
    pub bound: u128,
    pub data: TensorI64Owned<B>,
}

/// An immutable view of (a part of) a batch of ring elements; see [`RingEls`].
pub struct RingElsRef<'a, B: Backend + 'a> {
    pub rns_base: &'a RNSBaseIndices,
    pub basis: Basis,
    /// see [`RingEls::bound`]
    pub bound: u128,
    pub data: TensorI64Ref<'a, B>,
}

impl<'a, B: Backend> Clone for RingElsRef<'a, B> {
    fn clone(&self) -> Self { *self }
}

impl<'a, B: Backend> Copy for RingElsRef<'a, B> {}

/// A mutable view of (a part of) a batch of ring elements; see [`RingEls`]. Note that the
/// bound is a copy: operations that increase the representatives of a view cannot record
/// this in the owning batch, so unreduced additions are only offered on owned batches
/// ([`RingContext::add_els()`]).
pub struct RingElsMut<'a, B: Backend + 'a> {
    pub rns_base: &'a RNSBaseIndices,
    pub basis: Basis,
    /// see [`RingEls::bound`]
    pub bound: u128,
    pub data: TensorI64Mut<'a, B>,
}

impl<B: Backend> RingEls<B> {
    /// Wraps an almost-reduced (bound 1) tensor as a batch of ring elements.
    pub fn new(data: TensorI64Owned<B>, basis: Basis, rns_base: Box<RNSBaseIndices>) -> Self {
        Self { backend: PhantomData, rns_base, basis, bound: 1, data }
    }

    pub fn count(&self) -> usize { self.data.as_ref().shape()[0] }

    pub fn as_ref(&self) -> RingElsRef<'_, B> {
        RingElsRef { rns_base: &self.rns_base, basis: self.basis, bound: self.bound, data: self.data.as_ref() }
    }

    pub fn as_mut(&mut self) -> RingElsMut<'_, B> {
        RingElsMut { rns_base: &self.rns_base, basis: self.basis, bound: self.bound, data: self.data.as_mut() }
    }

    /// The restriction to a prefix of the RNS base, as a copy-free view.
    pub fn restrict_base(&self, base_len: usize) -> RingElsRef<'_, B> { self.as_ref().restrict_base(base_len) }

    /// The given range of batch elements, as a copy-free view.
    pub fn rows(&self, rows: Range<usize>) -> RingElsRef<'_, B> { self.as_ref().rows(rows) }

    /// The given range of batch elements, as a copy-free mutable view.
    pub fn rows_mut(&mut self, rows: Range<usize>) -> RingElsMut<'_, B> {
        RingElsMut {
            rns_base: &self.rns_base,
            basis: self.basis,
            bound: self.bound,
            data: self.data.as_mut().slice(rows, .., .., ..),
        }
    }
}

impl<'a, B: Backend> RingElsRef<'a, B> {
    pub fn count(&self) -> usize { self.data.shape()[0] }

    /// The restriction to a prefix of the RNS base, as a copy-free view.
    pub fn restrict_base(self, base_len: usize) -> RingElsRef<'a, B> {
        assert!(base_len <= self.rns_base.len());
        RingElsRef {
            rns_base: &self.rns_base[..base_len],
            basis: self.basis,
            bound: self.bound,
            data: self.data.slice(.., ..base_len, .., ..),
        }
    }

    /// The given range of batch elements, as a copy-free view.
    pub fn rows(self, rows: Range<usize>) -> RingElsRef<'a, B> {
        RingElsRef {
            rns_base: self.rns_base,
            basis: self.basis,
            bound: self.bound,
            data: self.data.slice(rows, .., .., ..),
        }
    }
}

impl<'a, B: Backend> From<&'a RingEls<B>> for RingElsRef<'a, B> {
    fn from(els: &'a RingEls<B>) -> Self { els.as_ref() }
}

impl<'a, B: Backend> From<&'a mut RingEls<B>> for RingElsMut<'a, B> {
    fn from(els: &'a mut RingEls<B>) -> Self { els.as_mut() }
}

/// A batch of ring elements with 128-bit entries (in the doubly-montgomery domain), used to
/// accumulate sums, differences and products of montgomery-form values, delaying the
/// (comparatively expensive) modular reduction to the single conversion back to a [`RingEls`]
/// at the end ([`RingContext::reduce_wide()`]). Long ring element sums and all fused
/// multiply-adds go through a [`WideRingEls`]; an addition is realized as an fma with
/// `mont(1)`, so that the added value enters the wide batch in the same doubly-montgomery
/// domain as products of two montgomery-form factors. (Short sums that stay within the `i64`
/// range can instead be performed unreduced on a [`RingEls`] directly, see
/// [`RingContext::add_els()`].)
///
/// The wrapper tracks, per batch element (dimension 0), a worst-case bound on the absolute
/// value of the accumulated entries, and panics before an accumulation could overflow the
/// `i128` range. The accumulation steps take the operands' representative bounds (in
/// multiples of the respective RNS factor, matching [`RingEls::bound`]) as parameters;
/// [`WideRingEls::fma_unbounded()`] allows one operand to be an arbitrary `i64`.
///
/// The bound also determines how the delayed reduction is performed: as long as all entries
/// are bounded by `2^63 min_p`, a single montgomery reduction ([`Backend::red_mont64()`])
/// of the doubly-montgomery values directly yields montgomery form; only larger
/// entries need the full double reduction [`Backend::red_mont64_double()`] followed
/// by [`Backend::i64_to_mont64()`].
///
/// The wide tensor is not explicitly zeroed: the first accumulation step of each batch
/// element (recognized by its bound still being zero) *overwrites* the tensor
/// ([`Backend::mul_i64()`]) instead of adding to it ([`Backend::fma_i64()`]). Rows that are
/// never accumulated into reduce to zero, since [`Backend::allocate_i128()`] returns
/// zero-initialized memory.
pub struct WideRingEls<B: Backend> {
    rns_base: Box<RNSBaseIndices>,
    data: TensorI128Owned<B>,
    /// `mont(1)` per RNS factor, shape `[1, r, 1, 1]`; multiplying by this lifts plain
    /// additions into the doubly-montgomery domain
    one: TensorI64Owned<B>,
    /// `mont(-1)` per RNS factor
    neg_one: TensorI64Owned<B>,
    /// the largest RNS factor of the base; almost-reduced operands are bounded by it
    max_p: i64,
    /// the smallest RNS factor of the base; determines up to which bound a single montgomery
    /// reduction suffices when reducing the wide batch
    min_p: i64,
    /// per batch element: worst-case bound on the absolute value of the accumulated entries
    bound: Vec<u128>,
}

impl<B: Backend> WideRingEls<B> {
    pub fn as_ref(&self) -> TensorI128Ref<'_, B> { self.data.as_ref() }

    pub fn rns_base(&self) -> &RNSBaseIndices { &self.rns_base }

    pub fn count(&self) -> usize { self.bound.len() }

    fn record_step(&mut self, rows: &Range<usize>, step_bound: u128) {
        for i in rows.clone() {
            self.bound[i] = self.bound[i]
                .checked_add(step_bound)
                .filter(|bound| *bound <= i128::MAX as u128)
                .expect("wide ring element overflow: too many accumulation steps");
        }
    }

    /// Whether finishing row `i` requires the full double reduction, or whether a single
    /// montgomery reduction suffices (i.e. all entries are bounded by `2^63 min_p`).
    fn needs_full_reduction(&self, row: usize) -> bool { self.bound[row] > (self.min_p as u128) << 63 }

    /// Performs `data[rows] += lhs * rhs` pointwise, except that rows that have not been
    /// accumulated into yet (recognized by `bound[row] == 0`) are *overwritten* with
    /// `lhs * rhs`, so that fresh accumulators need neither zeroing nor reading back the
    /// zeros. `lhs`/`rhs` are broadcast in dimension 0 if they have length 1 there, otherwise
    /// their dimension 0 must match `rows`. (A free function over the individual fields, so
    /// that callers can pass `self.one`/`self.neg_one` as an operand.)
    fn accumulate(
        backend: &B,
        data: &mut TensorI128Owned<B>,
        bound: &[u128],
        rows: Range<usize>,
        lhs: TensorI64Ref<B>,
        rhs: TensorI64Ref<B>,
    ) {
        fn slice_operand<'a, B: Backend>(
            tensor: TensorI64Ref<'a, B>,
            offset: usize,
            run: &Range<usize>,
        ) -> TensorI64Ref<'a, B> {
            if tensor.shape()[0] == 1 { tensor } else { tensor.slice(run.start - offset..run.end - offset, .., .., ..) }
        }
        // maximal runs of rows with equal freshness, each handled by a single backend call
        let mut start = rows.start;
        while start < rows.end {
            let fresh = bound[start] == 0;
            let mut end = start + 1;
            while end < rows.end && (bound[end] == 0) == fresh {
                end += 1;
            }
            let run = start..end;
            let dst = data.as_mut().slice(run.clone(), .., .., ..);
            if fresh {
                backend.hal_mul_i64(
                    dst,
                    slice_operand::<B>(lhs, rows.start, &run),
                    slice_operand::<B>(rhs, rows.start, &run),
                );
            } else {
                backend.hal_fma_i64(
                    dst,
                    slice_operand::<B>(lhs, rows.start, &run),
                    slice_operand::<B>(rhs, rows.start, &run),
                );
            }
            start = end;
        }
    }

    /// Computes `wide[rows] += lhs * rhs` pointwise (with broadcasting; `rows` selects a range
    /// of the batch dimension 0). `lhs_bound` and `rhs_bound` are worst-case bounds on the
    /// operand entries, in multiples of the respective RNS factor (matching
    /// [`RingEls::bound`]; almost-reduced ring elements and the montgomery-form factor
    /// tensors of [`RingContext::scalar_tensor()`] & co. have bound 1). Panics if any of the
    /// selected rows could overflow.
    pub fn fma(
        &mut self,
        backend: &B,
        rows: Range<usize>,
        lhs: TensorI64Ref<B>,
        lhs_bound: u128,
        rhs: TensorI64Ref<B>,
        rhs_bound: u128,
    ) {
        Self::accumulate(backend, &mut self.data, &self.bound, rows.clone(), lhs, rhs);
        let step = lhs_bound
            .checked_mul(rhs_bound)
            .and_then(|b| b.checked_mul((self.max_p as u128) * (self.max_p as u128)))
            .expect("wide ring element overflow: operand bounds too large");
        self.record_step(&rows, step);
    }

    /// [`WideRingEls::fma()`] over the whole batch.
    pub fn fma_all(
        &mut self,
        backend: &B,
        lhs: TensorI64Ref<B>,
        lhs_bound: u128,
        rhs: TensorI64Ref<B>,
        rhs_bound: u128,
    ) {
        let rows = 0..self.bound.len();
        self.fma(backend, rows, lhs, lhs_bound, rhs, rhs_bound);
    }

    /// Computes `wide[rows] = Σ_t lhs_t * rhs_t` in a single backend call, the multi-term,
    /// cache-tiled counterpart of a chain of [`WideRingEls::fma()`]s (see
    /// [`Backend::inner_product_i64()`]). Since the underlying HAL call *overwrites* the
    /// destination, all selected rows must still be fresh (never accumulated into yet); this
    /// matches the common case of building an inner product into a freshly allocated batch.
    /// Each term is `(lhs, lhs_bound, rhs, rhs_bound)` with operands and bounds as in
    /// [`WideRingEls::fma()`]; operands are broadcast in dimension 0 if they have length 1
    /// there, otherwise their dimension 0 must have length `rows.len()`. Panics if any row
    /// could overflow.
    pub fn inner_product(
        &mut self,
        backend: &B,
        rows: Range<usize>,
        terms: &[(TensorI64Ref<B>, u128, TensorI64Ref<B>, u128)],
    ) {
        assert!(!terms.is_empty(), "inner product needs at least one term");
        assert!(rows.clone().all(|i| self.bound[i] == 0), "WideRingEls::inner_product requires fresh rows");
        let dst = self.data.as_mut().slice(rows.clone(), .., .., ..);
        for (lhs, _, rhs, _) in terms {
            assert!(lhs.shape()[0] == 1 || lhs.shape()[0] == rows.len());
            assert!(rhs.shape()[0] == 1 || rhs.shape()[0] == rows.len());
        }
        let lhs = terms.iter().map(|(l, ..)| *l).collect::<Vec<_>>();
        let rhs = terms.iter().map(|(_, _, r, _)| *r).collect::<Vec<_>>();
        backend.hal_inner_product_i64(dst, &lhs, &rhs);
        let mut step: u128 = 0;
        for (_, lhs_bound, _, rhs_bound) in terms {
            let term = lhs_bound
                .checked_mul(*rhs_bound)
                .and_then(|b| b.checked_mul((self.max_p as u128) * (self.max_p as u128)))
                .expect("wide ring element overflow: operand bounds too large");
            step = step.checked_add(term).expect("wide ring element overflow: too many accumulation steps");
        }
        self.record_step(&rows, step);
    }

    /// The variant of [`WideRingEls::fma()`] where `lhs` may be an arbitrary `i64`
    /// tensor (e.g. plain, non-montgomery values); `rhs` must be bounded by the largest
    /// RNS factor of the base.
    pub fn fma_unbounded(&mut self, backend: &B, rows: Range<usize>, lhs: TensorI64Ref<B>, rhs: TensorI64Ref<B>) {
        Self::accumulate(backend, &mut self.data, &self.bound, rows.clone(), lhs, rhs);
        self.record_step(&rows, (self.max_p as u128) << 63);
    }

    /// [`WideRingEls::fma_unbounded()`] over the whole batch.
    pub fn fma_unbounded_all(&mut self, backend: &B, lhs: TensorI64Ref<B>, rhs: TensorI64Ref<B>) {
        let rows = 0..self.bound.len();
        self.fma_unbounded(backend, rows, lhs, rhs);
    }

    /// Computes `wide[rows] += src` for a montgomery-form tensor `src` (with broadcasting),
    /// realized as an fma with `mont(1)`; `src_bound` as in [`WideRingEls::fma()`].
    pub fn add(&mut self, backend: &B, rows: Range<usize>, src: TensorI64Ref<B>, src_bound: u128) {
        Self::accumulate(backend, &mut self.data, &self.bound, rows.clone(), src, self.one.as_ref());
        let step = src_bound
            .checked_mul((self.max_p as u128) * (self.max_p as u128))
            .expect("wide ring element overflow: operand bound too large");
        self.record_step(&rows, step);
    }

    /// [`WideRingEls::add()`] over the whole batch.
    pub fn add_all(&mut self, backend: &B, src: TensorI64Ref<B>, src_bound: u128) {
        let rows = 0..self.bound.len();
        self.add(backend, rows, src, src_bound);
    }

    /// Computes `wide[rows] -= src` for a montgomery-form tensor `src` (with broadcasting),
    /// realized as an fma with `mont(-1)`; `src_bound` as in [`WideRingEls::fma()`].
    pub fn sub(&mut self, backend: &B, rows: Range<usize>, src: TensorI64Ref<B>, src_bound: u128) {
        Self::accumulate(backend, &mut self.data, &self.bound, rows.clone(), src, self.neg_one.as_ref());
        let step = src_bound
            .checked_mul((self.max_p as u128) * (self.max_p as u128))
            .expect("wide ring element overflow: operand bound too large");
        self.record_step(&rows, step);
    }

    /// [`WideRingEls::sub()`] over the whole batch.
    pub fn sub_all(&mut self, backend: &B, src: TensorI64Ref<B>, src_bound: u128) {
        let rows = 0..self.bound.len();
        self.sub(backend, rows, src, src_bound);
    }
}

/// One RNS factor. The per-prime data for the period-basis/CRT-basis transforms is not stored
/// here: the kernels are memoized in [`RingContext::rns_factor_kernels`] (see
/// [`RnsFactorKernels`]) and gathered into per-RNS-base tensors on demand (see
/// [`TransformCache`]).
struct RnsFactor<B: Backend> {
    p: i64,
    zn: Zn,
    modulus: B::Modulus,
    /// whether `p ≡ 1 mod rou_order`, i.e. whether the factor supports the period-basis/CRT-basis
    /// transforms; `false` for factors of a [`RingContext::new_raw()`] context, for which only
    /// base conversions work
    supports_transform: bool,
}

/// The NTT-domain CRT kernels of one RNS factor, kept on the host (montgomery form, NTT
/// domain, reduced to `(-p, p)`), so that the per-RNS-base kernel tensors of [`TransformCache`]
/// can be created by simple concatenation.
///
/// Computing these costs `O(m_K + [M_K : K] deg_K)` per factor — by far the most expensive part
/// of [`RingContext::compute_rns_factor_transform()`], and for the larger fields more expensive
/// than the transforms they serve. They depend only on the factor (and the field), not on the
/// RNS base it appears in, so they are memoized per factor in
/// [`RingContext::rns_factor_kernels`]; without that, every newly encountered RNS base would
/// recompute them for all of its factors.
struct RnsFactorKernels {
    /// length `deg_Lʼ`
    crt_kernel_lprime: Vec<i64>,
    /// length `deg_K`
    crt_kernel_k: Vec<i64>,
    /// pointwise inverse of `crt_kernel_lprime`
    inv_crt_kernel_lprime: Vec<i64>,
    /// pointwise inverse of `crt_kernel_k`
    inv_crt_kernel_k: Vec<i64>,
}

/// The NTT plans and NTT-domain CRT kernels of one RNS factor. This is a transient result of
/// [`RingContext::compute_rns_factor_transform()`]; the NTT plans are cheap to build and hence
/// rebuilt per RNS base, while the kernels are shared with the memo (see [`RnsFactorKernels`]).
struct RnsFactorTransform<B: Backend> {
    ntt_lprime: B::NTTPlan,
    ntt_k: B::NTTPlan,
    kernels: Arc<RnsFactorKernels>,
}

/// Cached per-RNS-base data: the gathered moduli, NTT plans and kernel tensors for all factors
/// of one RNS base, so that a basis transform of a whole batch of ring elements is a handful
/// of HAL calls.
struct TransformCache<B: Backend> {
    rns_base: Box<RNSBaseIndices>,
    mods: Vec<B::Modulus>,
    ntts_lprime: Vec<B::NTTPlan>,
    ntts_k: Vec<B::NTTPlan>,
    /// shape `[1, r, deg_Lʼ, 1]`, NTT domain, montgomery form
    crt_kernel_lprime: TensorI64Owned<B>,
    /// shape `[1, r, 1, deg_K]`, NTT domain, montgomery form
    crt_kernel_k: TensorI64Owned<B>,
    inv_crt_kernel_lprime: TensorI64Owned<B>,
    inv_crt_kernel_k: TensorI64Owned<B>,
}

pub struct RingContext<B: Backend> {
    backend: B,
    field: FieldDescriptor,
    rns_factors: Vec<RnsFactor<B>>,
    /// the first `initial_rns_base_len` factors form the initial modulus `q` of fresh
    /// ciphertexts; the remaining factors are the pool from which extension bases for lifted
    /// multiplication are drawn (see [`RingContext::extension_rns_base_for()`])
    initial_rns_base_len: usize,
    /// the plaintext modulus bits the extension pool was sized for
    t_bits: usize,
    sparse_key_encaps_len: usize,
    bconv_cache: RwLock<Vec<BConvCache<B>>>,
    gadget_bconv_cache: RwLock<Vec<GadgetBConvCache<B>>>,
    transform_cache: RwLock<Vec<TransformCache<B>>>,
    /// the memoized CRT kernels of the RNS factors, keyed by their prime; see
    /// [`RnsFactorKernels`]
    rns_factor_kernels: RwLock<HashMap<i64, Arc<RnsFactorKernels>>>,
    permutation_cache: RwLock<HashMap<Vec<i64>, (B::PermutationPlan, B::PermutationPlan)>>,
    /// cached [`crate::dbfv::SplitOverflowData`]; storing dBFV-specific data here is a slight
    /// break of abstraction, accepted so that the (expensive to compute) constants are shared
    /// across all dBFV reductions
    pub(crate) split_overflow_cache: RwLock<Vec<crate::dbfv::SplitOverflowData<B>>>,
}

impl<B: Backend> RingContext<B> {
    /// Creates a ring context whose initial RNS base `q` has `log2_q` bits, and whose extension
    /// pool supports (lifted) multiplication of `q`-sized elements at a plaintext modulus
    /// of `t_bits` bits.
    pub fn new(backend: B, field: FieldDescriptor, log2_q: Range<usize>, t_bits: usize) -> Self {
        let rou_order = field.rou_order() as i64;
        let rns_base = extend_sampled_primes(&[], log2_q.start, log2_q.end, 57, rou_order).unwrap();
        let extended_base = extend_sampled_primes(
            &rns_base,
            2 * log2_q.end + EXTENDED_Q_EXTRA_BITS + t_bits,
            2 * log2_q.end + EXTENDED_Q_EXTRA_BITS + t_bits + 30,
            57,
            rou_order,
        )
        .unwrap();

        let mut result = Self {
            backend,
            rns_factors: Vec::new(),
            initial_rns_base_len: rns_base.len(),
            t_bits,
            sparse_key_encaps_len: usize::min(3, rns_base.len()),
            bconv_cache: RwLock::new(Vec::new()),
            gadget_bconv_cache: RwLock::new(Vec::new()),
            split_overflow_cache: RwLock::new(Vec::new()),
            transform_cache: RwLock::new(Vec::new()),
            rns_factor_kernels: RwLock::new(HashMap::new()),
            permutation_cache: RwLock::new(HashMap::new()),
            field,
        };
        for p in extended_base {
            result.add_rns_factor(p);
        }
        return result;
    }

    /// Creates a ring context over the given primes without basis-transform support (the
    /// primes are not required to be compatible with the field); only base conversions work on
    /// such a context.
    #[cfg(test)]
    pub fn new_raw(backend: B, field: FieldDescriptor, primes: &[i64]) -> Self {
        let mut result = Self {
            backend,
            rns_factors: Vec::new(),
            initial_rns_base_len: primes.len(),
            t_bits: 0,
            sparse_key_encaps_len: usize::min(3, primes.len()),
            bconv_cache: RwLock::new(Vec::new()),
            gadget_bconv_cache: RwLock::new(Vec::new()),
            split_overflow_cache: RwLock::new(Vec::new()),
            transform_cache: RwLock::new(Vec::new()),
            rns_factor_kernels: RwLock::new(HashMap::new()),
            permutation_cache: RwLock::new(HashMap::new()),
            field,
        };
        for p in primes {
            result.add_rns_factor_raw(*p);
        }
        return result;
    }

    fn add_rns_factor_raw(&mut self, p: i64) -> usize {
        assert!(p >= self.min_rns_factor() && p <= self.max_rns_factor());
        assert!(self.rns_factors.iter().all(|f| f.p != p));
        let zn = Zn::new(p as u64);
        let modulus = self.backend.hal_create_modulus(p);
        self.rns_factors.push(RnsFactor { p, zn, modulus, supports_transform: false });
        return self.rns_factors.len() - 1;
    }

    fn add_rns_factor(&mut self, p: i64) -> usize {
        assert_eq!(0, (p - 1) % self.field.rou_order() as i64, "RNS factor incompatible with the field");
        let idx = self.add_rns_factor_raw(p);
        self.rns_factors[idx].supports_transform = true;
        return idx;
    }

    pub fn backend(&self) -> &B { &self.backend }

    pub fn field(&self) -> &FieldDescriptor { &self.field }

    pub fn n(&self) -> usize { self.field.deg_l() }

    pub fn slot_count(&self) -> usize { self.field.deg_k() }

    /// The indices of the initial RNS base `q`, over which fresh ciphertexts (and keys) are
    /// created. Functions operating on existing ring elements or ciphertexts must derive the
    /// RNS base from their inputs instead, since elements may have been rescaled to a smaller
    /// base.
    pub fn initial_rns_base(&self) -> Vec<usize> { (0..self.initial_rns_base_len).collect() }

    pub fn initial_rns_base_len(&self) -> usize { self.initial_rns_base_len }

    /// The indices of the extension base to use for lifted multiplication of elements over
    /// `base`: enough factors (disjoint from any initial-base prefix) such that the product of
    /// `base` and the extension leaves room for a lifted product plus the plaintext scaling,
    /// i.e. has at least `2 log2(Q) + t_bits + EXTENDED_Q_EXTRA_BITS` bits. In particular, the
    /// extension is smaller for elements that have been rescaled to a smaller base.
    pub fn extension_rns_base_for(&self, base: &RNSBaseIndices) -> Vec<usize> {
        assert!(base.iter().all(|i| *i < self.initial_rns_base_len));
        let required_bits =
            (ZZbig.abs_log2_ceil(&self.modulus_of(base)).unwrap_or(0) + self.t_bits + EXTENDED_Q_EXTRA_BITS) as f64;
        let mut result = Vec::new();
        let mut bits = 0.0f64;
        for idx in self.initial_rns_base_len..self.rns_factors.len() {
            if bits >= required_bits {
                break;
            }
            result.push(idx);
            bits += (self.rns_factors[idx].p as f64).log2();
        }
        assert!(bits >= required_bits, "extension pool too small for lifted multiplication over this base");
        return result;
    }

    /// The indices of the small RNS base used for the sparse-key encapsulation during
    /// bootstrapping.
    pub fn sparse_key_encaps_rns_base(&self) -> Vec<usize> { (0..self.sparse_key_encaps_len).collect() }

    pub fn rns_factor_prime(&self, idx: usize) -> i64 { self.rns_factors[idx].p }

    pub fn rns_factor_zn(&self, idx: usize) -> &Zn { &self.rns_factors[idx].zn }

    /// The feanor-math RNS ring `Z/QZ` for the RNS base given by `base`, used for host-side
    /// encoding/decoding of ring elements.
    pub fn zn_rns_for(&self, base: &RNSBaseIndices) -> ZnRns<Zn, BigIntRing> {
        ZnRns::new(base.iter().map(|i| self.rns_factors[*i].zn).collect(), ZZbig)
    }

    /// The product of the primes of the RNS base given by `base`.
    pub fn modulus_of(&self, base: &RNSBaseIndices) -> El<BigIntRing> {
        ZZbig.prod(base.iter().map(|i| int_cast(self.rns_factors[*i].p, ZZbig, ZZi64)))
    }

    /// The `log2` of [`RingContext::modulus_of()`], as a sum of the per-prime logarithms. Unlike
    /// `ZZbig.abs_log2_ceil(&ring.modulus_of(base))` this is exact to within floating-point
    /// error rather than rounded to a whole bit, which matters when it is used to *derive* a
    /// parameter (see [`crate::bfv::sigma_for_log2_alpha()`]).
    pub fn log2_modulus_of(&self, base: &RNSBaseIndices) -> f64 {
        base.iter().map(|i| (self.rns_factors[*i].p as f64).log2()).sum()
    }

    fn min_rns_factor(&self) -> i64 { 1 << 40 }

    fn max_rns_factor(&self) -> i64 { (1 << 57) - 1 }

    /// Returns a bound on
    /// ```text
    ///   max_(x, y != 0) log2 |xy|_pb∞ - log2 |x|_pb∞ - log2 |y|_pb∞
    /// ```
    /// where `|x|_pb∞` is the largest magnitude among coefficients in the period-basis
    /// representation of `x`.
    pub fn log2_pb_expansion_factor_bound(&self) -> usize {
        ZZi64.abs_log2_ceil(&(self.field.m() as i64)).unwrap() + self.field.m_factors().len() + 1
    }

    pub fn allocate(&self, els_count: usize, rns_base: &RNSBaseIndices) -> RingEls<B> {
        self.allocate_in_basis(els_count, rns_base, Basis::PeriodBasis)
    }

    pub fn allocate_in_basis(&self, els_count: usize, rns_base: &RNSBaseIndices, basis: Basis) -> RingEls<B> {
        RingEls {
            backend: PhantomData,
            rns_base: rns_base.to_owned().into_boxed_slice(),
            basis,
            bound: 1,
            data: self.backend.hal_allocate_i64([
                els_count,
                rns_base.len(),
                self.field.deg_lprime(),
                self.field.deg_k(),
            ]),
        }
    }

    /// Allocates a zeroed wide batch matching a batch of `els_count` ring elements over
    /// the given RNS base; see [`WideRingEls`] and [`RingContext::reduce_wide()`].
    pub fn allocate_wide(&self, els_count: usize, rns_base: &RNSBaseIndices) -> WideRingEls<B> {
        // no explicit zeroing: the first accumulation into each row overwrites it, see
        // [`WideRingEls::accumulate()`]
        let data =
            self.backend.hal_allocate_i128([els_count, rns_base.len(), self.field.deg_lprime(), self.field.deg_k()]);
        return WideRingEls {
            rns_base: rns_base.to_owned().into_boxed_slice(),
            data,
            one: self.scalar_tensor(rns_base, 1),
            neg_one: self.scalar_tensor(rns_base, -1),
            max_p: rns_base.iter().map(|i| self.rns_factors[*i].p).max().unwrap_or(1),
            min_p: rns_base.iter().map(|i| self.rns_factors[*i].p).min().unwrap_or(1),
            bound: vec![0; els_count],
        };
    }

    /// Converts a wide batch back into a batch of (almost-reduced, montgomery-form) ring
    /// elements; this performs the modular reduction that has been delayed during
    /// accumulation.
    pub fn reduce_wide(&self, wide: &WideRingEls<B>, basis: Basis) -> RingEls<B> {
        self.reduce_wide_rows(wide, 0..wide.count(), basis)
    }

    /// The version of [`RingContext::reduce_wide()`] that only reduces the given range
    /// of batch elements.
    pub fn reduce_wide_rows(&self, wide: &WideRingEls<B>, rows: Range<usize>, basis: Basis) -> RingEls<B> {
        let mut result = self.allocate_in_basis(rows.len(), &wide.rns_base, basis);
        self.reduce_wide_tensor(result.data.as_mut(), wide, rows);
        return result;
    }

    /// The version of [`RingContext::reduce_wide()`] that writes the reduced elements
    /// into an existing (mutable view of a) batch. Note that the written rows are
    /// almost-reduced, so the bound of the (owner of the) destination view remains valid.
    pub fn reduce_wide_into(&self, dst: RingElsMut<B>, wide: &WideRingEls<B>) {
        assert_eq!(dst.rns_base, &*wide.rns_base);
        self.reduce_wide_into_tensor(dst.data, wide);
    }

    /// The raw-tensor version of [`RingContext::reduce_wide_into()`].
    pub fn reduce_wide_into_tensor(&self, dst: TensorI64Mut<B>, wide: &WideRingEls<B>) {
        self.reduce_wide_tensor(dst, wide, 0..wide.count());
    }

    /// Reduces wide entries (in the doubly-montgomery domain) into almost-reduced
    /// montgomery-form values, overwriting `dst` (whose dimension 0 must match `rows`). Rows
    /// whose entries are bounded by `2^63 min_p` need only a single montgomery reduction,
    /// which directly yields montgomery form; larger rows need the full double reduction
    /// (producing the plain values, which are then converted to montgomery form). Maximal
    /// runs of rows with the same requirement are handled by joint backend calls.
    #[instrument(skip_all)]
    fn reduce_wide_tensor(&self, mut dst: TensorI64Mut<B>, wide: &WideRingEls<B>, rows: Range<usize>) {
        let mods = self.mods_for(&wide.rns_base);
        let mut start = rows.start;
        while start < rows.end {
            let full_reduction = wide.needs_full_reduction(start);
            let mut end = start + 1;
            while end < rows.end && wide.needs_full_reduction(end) == full_reduction {
                end += 1;
            }
            let mut dst_run = dst.reborrow().slice(start - rows.start..end - rows.start, .., .., ..);
            let wide_run = wide.data.as_ref().slice(start..end, .., .., ..);
            if full_reduction {
                self.backend.hal_red_mont64_double(dst_run.reborrow(), wide_run, &mods);
                self.backend.hal_i64_to_mont64(dst_run, &mods);
            } else {
                self.backend.hal_red_mont64(dst_run, wide_run, &mods);
            }
            start = end;
        }
    }

    /// The largest RNS factor of the given base.
    fn max_p_of(&self, base: &RNSBaseIndices) -> i64 { base.iter().map(|i| self.rns_factors[*i].p).max().unwrap_or(1) }

    pub fn mods_for(&self, base: &RNSBaseIndices) -> Vec<B::Modulus> {
        base.iter().map(|i| self.rns_factors[*i].modulus.clone()).collect()
    }

    /// Creates a `[1, r, 1, 1]`-shaped tensor holding `mont(scalar mod p)` for each RNS factor
    /// `p` of `base`, suitable for broadcast multiplication/fma with ring element batches.
    pub fn scalar_tensor(&self, base: &RNSBaseIndices, scalar: i128) -> TensorI64Owned<B> {
        let values = base
            .iter()
            .map(|i| {
                let p = self.rns_factors[*i].p;
                mont_repr((scalar % p as i128) as i64, p)
            })
            .collect::<Vec<_>>();
        let mut result = self.backend.hal_allocate_i64([1, base.len(), 1, 1]);
        self.backend.hal_import(result.as_mut(), &values);
        return result;
    }

    /// Like [`RingContext::scalar_tensor()`], but for an arbitrarily large scalar.
    pub fn scalar_tensor_big(&self, base: &RNSBaseIndices, scalar: &El<BigIntRing>) -> TensorI64Owned<B> {
        let values = base
            .iter()
            .map(|i| {
                let zn = &self.rns_factors[*i].zn;
                let reduced = zn.coerce(&ZZbig, ZZbig.clone_el(scalar));
                mont_repr(zn.smallest_lift(reduced), self.rns_factors[*i].p)
            })
            .collect::<Vec<_>>();
        let mut result = self.backend.hal_allocate_i64([1, base.len(), 1, 1]);
        self.backend.hal_import(result.as_mut(), &values);
        return result;
    }

    /// Like [`RingContext::scalar_tensor_big()`], but holding `scalar * 2^128 mod p` (the
    /// "doubly montgomery" form). Multiplying this with plain (non-montgomery) values and
    /// reducing via [`RingContext::reduce_wide()`] yields montgomery-form products.
    pub fn scalar_tensor_big_mont2(&self, base: &RNSBaseIndices, scalar: &El<BigIntRing>) -> TensorI64Owned<B> {
        let values = base
            .iter()
            .map(|i| {
                let zn = &self.rns_factors[*i].zn;
                let p = self.rns_factors[*i].p;
                let reduced = zn.smallest_lift(zn.coerce(&ZZbig, ZZbig.clone_el(scalar)));
                mont_repr(mont_repr(reduced, p), p)
            })
            .collect::<Vec<_>>();
        let mut result = self.backend.hal_allocate_i64([1, base.len(), 1, 1]);
        self.backend.hal_import(result.as_mut(), &values);
        return result;
    }

    /// Multiplies all elements by the given scalar. The result is almost-reduced (bound 1).
    pub fn mul_scalar(&self, els: &mut RingEls<B>, scalar: i128) {
        self.assert_mul_mont_safe(els.bound, &els.rns_base);
        let tensor = self.scalar_tensor(&els.rns_base, scalar);
        let mods = self.mods_for(&els.rns_base);
        self.backend.hal_mul_mont64(els.data.as_mut(), tensor.as_ref(), &mods);
        els.bound = 1;
    }

    /// Multiplies all elements by the given scalar. The result is almost-reduced (bound 1).
    pub fn mul_scalar_big(&self, els: &mut RingEls<B>, scalar: &El<BigIntRing>) {
        self.assert_mul_mont_safe(els.bound, &els.rns_base);
        let tensor = self.scalar_tensor_big(&els.rns_base, scalar);
        let mods = self.mods_for(&els.rns_base);
        self.backend.hal_mul_mont64(els.data.as_mut(), tensor.as_ref(), &mods);
        els.bound = 1;
    }

    /// Multiplies every element of the batch (which must be in CRT basis) by the `Lʼ`-element
    /// with the given period-basis coefficients (see [`RingContext::lprime_element_tensor()`]).
    /// The result is almost-reduced (bound 1).
    pub fn mul_lprime_element(&self, els: &mut RingEls<B>, coeffs: &[i128]) {
        assert_eq!(Basis::CRTBasis, els.basis, "multiplication by an Lʼ-element requires CRT basis");
        self.assert_mul_mont_safe(els.bound, &els.rns_base);
        let tensor = self.lprime_element_tensor(&els.rns_base, coeffs);
        let mods = self.mods_for(&els.rns_base);
        self.backend.hal_mul_mont64(els.data.as_mut(), tensor.as_ref(), &mods);
        els.bound = 1;
    }

    /// Asserts that a montgomery multiplication of entries bounded by `bound * p` with an
    /// almost-reduced factor cannot exceed the input contract of [`Backend::mul_mont64()`]
    /// (product within `[-2^63 p, 2^63 p]`).
    pub(crate) fn assert_mul_mont_safe(&self, bound: u128, base: &RNSBaseIndices) {
        assert!(
            bound.saturating_mul(self.max_p_of(base) as u128) <= 1 << 63,
            "representatives too large for a montgomery multiplication"
        );
    }

    /// Adds `src` to the wide batch (element-wise over the batch; `src` may have batch size
    /// 1, in which case it is broadcast); the operand's representative bound enters the
    /// overflow accounting. See [`WideRingEls`].
    pub fn add<'a>(&self, wide: &mut WideRingEls<B>, src: impl Into<RingElsRef<'a, B>>)
    where
        B: 'a,
    {
        let src = src.into();
        assert_eq!(&*wide.rns_base, src.rns_base);
        wide.add_all(&self.backend, src.data, src.bound);
    }

    /// Subtracts `src` from the wide batch; see [`RingContext::add()`].
    pub fn sub<'a>(&self, wide: &mut WideRingEls<B>, src: impl Into<RingElsRef<'a, B>>)
    where
        B: 'a,
    {
        let src = src.into();
        assert_eq!(&*wide.rns_base, src.rns_base);
        wide.sub_all(&self.backend, src.data, src.bound);
    }

    /// Adds `src` to `dst` entry-wise *without* modular reduction: the representatives grow,
    /// which is recorded in `dst.bound` (the bounds add up). Use this for short sums whose
    /// result is consumed by bound-tolerant operations (fma accumulation into a
    /// [`WideRingEls`], the basis transforms up to bound 2, ...); longer sums go through a
    /// [`WideRingEls`]. `src` may have batch size 1 (broadcast) or match `dst`. Panics if
    /// entries could leave the `i64` range.
    pub fn add_els<'a>(&self, dst: &mut RingEls<B>, src: impl Into<RingElsRef<'a, B>>)
    where
        B: 'a,
    {
        let src = src.into();
        self.add_sub_els(dst, src, false);
    }

    /// Subtracts `src` from `dst` entry-wise without modular reduction; see
    /// [`RingContext::add_els()`].
    pub fn sub_els<'a>(&self, dst: &mut RingEls<B>, src: impl Into<RingElsRef<'a, B>>)
    where
        B: 'a,
    {
        let src = src.into();
        self.add_sub_els(dst, src, true);
    }

    fn add_sub_els(&self, dst: &mut RingEls<B>, src: RingElsRef<B>, subtract: bool) {
        assert_eq!(&*dst.rns_base, src.rns_base);
        assert_eq!(dst.basis, src.basis);
        let new_bound = dst.bound.checked_add(src.bound).unwrap();
        assert!(
            new_bound.saturating_mul(self.max_p_of(&dst.rns_base) as u128) <= i64::MAX as u128,
            "unreduced addition could overflow the i64 range; use a WideRingEls instead"
        );
        if subtract {
            self.backend.hal_sub_i64(dst.data.as_mut(), src.data);
        } else {
            self.backend.hal_add_i64(dst.data.as_mut(), src.data);
        }
        dst.bound = new_bound;
    }

    /// Creates a `[1, r, 1, 1]`-shaped tensor holding the plain (non-montgomery) smallest
    /// lifts of `scale * (Q/p)^-1 mod p`, where `Q` is the product of the base.
    /// Montgomery-multiplying montgomery-form values by this yields the plain values
    /// `lift(x * scale * (Q/p)^-1 mod p)` in `(-p, p)`, as used by base-conversion-style
    /// algorithms (which pass `scale = 1`; see [`BConvCache`]).
    pub fn q_over_p_inv_tensor(&self, base: &RNSBaseIndices, scale: i128) -> TensorI64Owned<B> {
        let q = self.modulus_of(base);
        let values = base
            .iter()
            .map(|i| {
                let zn = &self.rns_factors[*i].zn;
                let q_over_p =
                    zn.coerce(&ZZbig, ZZbig.checked_div(&q, &int_cast(self.rns_factors[*i].p, ZZbig, ZZi64)).unwrap());
                zn.smallest_lift(zn.mul(zn.invert(&q_over_p).unwrap(), zn.coerce(&ZZi128, scale)))
            })
            .collect::<Vec<_>>();
        let mut result = self.backend.hal_allocate_i64([1, base.len(), 1, 1]);
        self.backend.hal_import(result.as_mut(), &values);
        return result;
    }

    /// Computes `wide += lhs * rhs` pointwise into a wide batch (see
    /// [`RingContext::reduce_wide()`]). Both operands must be in CRT basis. Either
    /// operand may have batch size 1 for broadcasting.
    pub fn fma<'a, 'b>(
        &self,
        wide: &mut WideRingEls<B>,
        lhs: impl Into<RingElsRef<'a, B>>,
        rhs: impl Into<RingElsRef<'b, B>>,
    ) where
        B: 'a + 'b,
    {
        let (lhs, rhs) = (lhs.into(), rhs.into());
        assert_eq!(lhs.rns_base, rhs.rns_base);
        assert_eq!(&*wide.rns_base, lhs.rns_base);
        assert_eq!(Basis::CRTBasis, lhs.basis);
        assert_eq!(Basis::CRTBasis, rhs.basis);
        wide.fma_all(&self.backend, lhs.data, lhs.bound, rhs.data, rhs.bound);
    }

    /// Computes `wide += els * scalar` into a wide batch (see
    /// [`RingContext::reduce_wide()`]).
    pub fn fma_scalar<'a>(&self, wide: &mut WideRingEls<B>, els: impl Into<RingElsRef<'a, B>>, scalar: i128)
    where
        B: 'a,
    {
        let els = els.into();
        assert_eq!(&*wide.rns_base, els.rns_base);
        let tensor = self.scalar_tensor(els.rns_base, scalar);
        wide.fma_all(&self.backend, els.data, els.bound, tensor.as_ref(), 1);
    }

    /// Computes `wide += els * scalar` into a wide batch (see
    /// [`RingContext::reduce_wide()`]).
    pub fn fma_scalar_big<'a>(
        &self,
        wide: &mut WideRingEls<B>,
        els: impl Into<RingElsRef<'a, B>>,
        scalar: &El<BigIntRing>,
    ) where
        B: 'a,
    {
        let els = els.into();
        assert_eq!(&*wide.rns_base, els.rns_base);
        let tensor = self.scalar_tensor_big(els.rns_base, scalar);
        wide.fma_all(&self.backend, els.data, els.bound, tensor.as_ref(), 1);
    }

    /// Transforms all elements from the period basis to the CRT basis. Accepts inputs up to
    /// bound 2 (the NTTs accept inputs in `[-2p, 2p]`); the result is almost-reduced.
    pub fn pb_to_crt(&self, els: &mut RingEls<B>) {
        assert_eq!(Basis::PeriodBasis, els.basis);
        assert!(els.bound <= 2, "basis transforms require representatives bounded by 2p");
        let base = els.rns_base.clone();
        self.pb_to_crt_tensor(els.data.as_mut(), &base);
        els.basis = Basis::CRTBasis;
        els.bound = 1;
    }

    /// Transforms all elements from the CRT basis to the period basis. Accepts inputs up to
    /// bound 2; the result is almost-reduced.
    pub fn crt_to_pb(&self, els: &mut RingEls<B>) {
        assert_eq!(Basis::CRTBasis, els.basis);
        assert!(els.bound <= 2, "basis transforms require representatives bounded by 2p");
        let base = els.rns_base.clone();
        self.crt_to_pb_tensor(els.data.as_mut(), &base);
        els.basis = Basis::PeriodBasis;
        els.bound = 1;
    }

    /// The raw-tensor version of [`RingContext::pb_to_crt()`]. The input entries must be
    /// within `[-2p, 2p]` (bound 2), since the NTTs require inputs in that range; the output
    /// is almost-reduced.
    #[instrument(skip_all)]
    pub fn pb_to_crt_tensor(&self, mut data: TensorI64Mut<B>, base: &RNSBaseIndices) {
        let cache = self.transform_cache(base);
        // the CRT-kernel multiplication is the tensor product of a dim-2 (Lʼ) and a dim-3 (K)
        // factor. Since a dim-2 scaling commutes with the dim-3 transform (and vice versa), we
        // may interleave each transform with its own kernel factor and fuse the transform's
        // output reduction into that multiplication
        self.backend.hal_ntt_mul_mont64_dim(
            data.reborrow(),
            &cache.ntts_lprime,
            DataDim::Dim2,
            cache.crt_kernel_lprime.as_ref(),
            &cache.mods,
        );
        self.backend.hal_ntt_mul_mont64_dim(
            data.reborrow(),
            &cache.ntts_k,
            DataDim::Dim3,
            cache.crt_kernel_k.as_ref(),
            &cache.mods,
        );
        self.backend.hal_intt_mont64(data, &cache.ntts_lprime, &cache.ntts_k);
    }

    /// The raw-tensor version of [`RingContext::crt_to_pb()`]; requires input within
    /// `[-2p, 2p]`, see [`RingContext::pb_to_crt_tensor()`].
    #[instrument(skip_all)]
    pub fn crt_to_pb_tensor(&self, mut data: TensorI64Mut<B>, base: &RNSBaseIndices) {
        let cache = self.transform_cache(base);
        self.backend.hal_ntt_mul_mont64_dim(
            data.reborrow(),
            &cache.ntts_lprime,
            DataDim::Dim2,
            cache.inv_crt_kernel_lprime.as_ref(),
            &cache.mods,
        );
        self.backend.hal_ntt_mul_mont64_dim(
            data.reborrow(),
            &cache.ntts_k,
            DataDim::Dim3,
            cache.inv_crt_kernel_k.as_ref(),
            &cache.mods,
        );
        self.backend.hal_intt_mont64(data, &cache.ntts_lprime, &cache.ntts_k);
    }

    /// Creates a `[1, r, 1, 1]`-shaped tensor holding `mont((scalar mod p)^-1)` for each RNS
    /// factor `p` of `base`; the scalar must be invertible modulo each factor.
    pub fn inverse_scalar_tensor(&self, base: &RNSBaseIndices, scalar: &El<BigIntRing>) -> TensorI64Owned<B> {
        let values = base
            .iter()
            .map(|i| {
                let zn = &self.rns_factors[*i].zn;
                let reduced = zn.invert(&zn.coerce(&ZZbig, ZZbig.clone_el(scalar))).unwrap();
                mont_repr(zn.smallest_lift(reduced), self.rns_factors[*i].p)
            })
            .collect::<Vec<_>>();
        let mut result = self.backend.hal_allocate_i64([1, base.len(), 1, 1]);
        self.backend.hal_import(result.as_mut(), &values);
        return result;
    }

    /// Applies the Galois automorphism given by `g in (Z/mZ)*` to all elements of `src`,
    /// writing the results to `dst`. Works in both bases (both are normal bases, so the
    /// automorphism acts by permuting coefficients).
    #[instrument(skip_all)]
    pub fn apply_galois<'a>(&self, dst: &mut RingEls<B>, src: impl Into<RingElsRef<'a, B>>, g: u64)
    where
        B: 'a,
    {
        let src = src.into();
        assert_eq!(&*dst.rns_base, src.rns_base);
        let shift = self.field.gal_to_shift(g);
        dst.basis = src.basis;
        dst.bound = src.bound;
        {
            let cache = self.permutation_cache.read().unwrap();
            if let Some((perm_lprime, perm_k)) = cache.get(&shift) {
                return self.apply_galois_perms(dst, src, perm_lprime, perm_k);
            }
        }
        let lprime_factors = self.field.lprime_q_cyclic_decomposition().len();
        let orders = self.field.lq_cyclic_orders();
        let perm_lprime = self.backend.hal_create_permutation(
            &orders[..lprime_factors],
            &shift[..lprime_factors].iter().map(|s| *s as usize).collect::<Vec<_>>(),
        );
        let perm_k = self.backend.hal_create_permutation(
            &orders[lprime_factors..],
            &shift[lprime_factors..].iter().map(|s| *s as usize).collect::<Vec<_>>(),
        );
        self.apply_galois_perms(dst, src, &perm_lprime, &perm_k);
        self.permutation_cache.write().unwrap().insert(shift, (perm_lprime, perm_k));
    }

    fn apply_galois_perms(
        &self,
        dst: &mut RingEls<B>,
        src: RingElsRef<B>,
        perm_lprime: &B::PermutationPlan,
        perm_k: &B::PermutationPlan,
    ) {
        let mut tmp = self.backend.hal_allocate_i64(src.data.shape());
        self.backend.hal_permute(tmp.as_mut(), src.data, perm_lprime, DataDim::Dim2);
        self.backend.hal_permute(dst.data.as_mut(), tmp.as_ref(), perm_k, DataDim::Dim3);
    }

    /// Embeds an element of `Lʼ` (given by its coefficients w.r.t. the period basis of `Lʼ`
    /// induced by `M`) into the ring, in CRT basis, as a `[1, r, deg_Lʼ, 1]`-shaped tensor
    /// suitable for broadcast multiplication with ring element batches in CRT basis.
    pub fn lprime_element_tensor(&self, base: &RNSBaseIndices, coeffs: &[i128]) -> TensorI64Owned<B> {
        assert_eq!(self.field.deg_lprime(), coeffs.len());
        let r = base.len();
        let mut values = Vec::with_capacity(r * coeffs.len());
        for idx in base.iter() {
            let p = self.rns_factors[*idx].p;
            for c in coeffs {
                values.push(mont_repr((*c % p as i128) as i64, p));
            }
        }
        let mut result = self.backend.hal_allocate_i64([1, r, self.field.deg_lprime(), 1]);
        self.backend.hal_import(result.as_mut(), &values);
        // apply the Lʼ-part of the CRT transform along dimension 2
        let cache = self.transform_cache(base);
        self.backend.hal_ntt_mul_mont64_dim(
            result.as_mut(),
            &cache.ntts_lprime,
            DataDim::Dim2,
            cache.crt_kernel_lprime.as_ref(),
            &cache.mods,
        );
        self.backend.hal_intt_mont64_dim(result.as_mut(), &cache.ntts_lprime, DataDim::Dim2);
        return result;
    }

    fn transform_cache<'a>(&'a self, rns_base: &RNSBaseIndices) -> MappedRwLockReadGuard<'a, TransformCache<B>> {
        let read = self.transform_cache.read().unwrap();
        if let Some((idx, _)) = read.iter().enumerate().find(|(_, cache)| &*cache.rns_base == rns_base) {
            return RwLockReadGuard::map(read, |data| &data[idx]);
        }
        drop(read);
        let mut write = self.transform_cache.write().unwrap();
        let result = self.create_transform_cache(rns_base);
        let idx = write.len();
        write.push(result);
        drop(write);
        let read = self.transform_cache.read().unwrap();
        return RwLockReadGuard::map(read, |data| &data[idx]);
    }

    #[instrument(skip_all)]
    fn create_transform_cache(&self, rns_base: &RNSBaseIndices) -> TransformCache<B> {
        let r = rns_base.len();
        let deg_lprime = self.field.deg_lprime();
        let deg_k = self.field.deg_k();
        let mut kernel_lprime_data = Vec::with_capacity(r * deg_lprime);
        let mut kernel_k_data = Vec::with_capacity(r * deg_k);
        let mut inv_kernel_lprime_data = Vec::with_capacity(r * deg_lprime);
        let mut inv_kernel_k_data = Vec::with_capacity(r * deg_k);
        let mut ntts_lprime = Vec::with_capacity(r);
        let mut ntts_k = Vec::with_capacity(r);
        for idx in rns_base.iter() {
            assert!(self.rns_factors[*idx].supports_transform, "RNS factor doesn't support the basis transforms");
            let transform = self.compute_rns_factor_transform(self.rns_factors[*idx].p);
            ntts_lprime.push(transform.ntt_lprime);
            ntts_k.push(transform.ntt_k);
            kernel_lprime_data.extend_from_slice(&transform.kernels.crt_kernel_lprime);
            kernel_k_data.extend_from_slice(&transform.kernels.crt_kernel_k);
            inv_kernel_lprime_data.extend_from_slice(&transform.kernels.inv_crt_kernel_lprime);
            inv_kernel_k_data.extend_from_slice(&transform.kernels.inv_crt_kernel_k);
        }
        let import = |data: &[i64], shape: [usize; DIMS]| {
            let mut tensor = self.backend.hal_allocate_i64(shape);
            self.backend.hal_import(tensor.as_mut(), data);
            tensor
        };
        return TransformCache {
            rns_base: rns_base.to_owned().into_boxed_slice(),
            mods: self.mods_for(rns_base),
            ntts_lprime,
            ntts_k,
            crt_kernel_lprime: import(&kernel_lprime_data, [1, r, deg_lprime, 1]),
            crt_kernel_k: import(&kernel_k_data, [1, r, 1, deg_k]),
            inv_crt_kernel_lprime: import(&inv_kernel_lprime_data, [1, r, deg_lprime, 1]),
            inv_crt_kernel_k: import(&inv_kernel_k_data, [1, r, 1, deg_k]),
        };
    }

    /// Precomputes the CRT kernels of every RNS factor (see [`RnsFactorKernels`]), so that no
    /// later basis transform pays for them. This is pure setup — it changes nothing about the
    /// results, only about when the work happens — and is worth calling before timing anything,
    /// since otherwise the first transform over each newly encountered RNS base absorbs the
    /// (field-dependent, and for large `m_K` substantial) cost of its factors.
    pub fn precompute_rns_factor_transforms(&self) {
        for factor in &self.rns_factors {
            if factor.supports_transform {
                self.compute_rns_factor_transform(factor.p);
            }
        }
    }

    /// Computes the NTT plans for a new RNS factor `p` and looks up (or computes and memoizes)
    /// its CRT kernels; see the module-level documentation for the meaning of the kernels.
    fn compute_rns_factor_transform(&self, p: i64) -> RnsFactorTransform<B> {
        let field = &self.field;
        let modulus = self.backend.hal_create_modulus(p);
        let orders = field.lq_cyclic_orders();
        let lprime_factors = field.lprime_q_cyclic_decomposition().len();
        let ntt_lprime = self.backend.hal_create_ntt_plan(&modulus, &orders[..lprime_factors]);
        let ntt_k = self.backend.hal_create_ntt_plan(&modulus, &orders[lprime_factors..]);

        if let Some(kernels) = self.rns_factor_kernels.read().unwrap().get(&p) {
            return RnsFactorTransform { ntt_lprime, ntt_k, kernels: Arc::clone(kernels) };
        }
        let kernels = Arc::new(self.compute_rns_factor_kernels(p, &ntt_lprime, &ntt_k));
        self.rns_factor_kernels.write().unwrap().insert(p, Arc::clone(&kernels));
        return RnsFactorTransform { ntt_lprime, ntt_k, kernels };
    }

    /// Computes the NTT-domain CRT kernels of the RNS factor `p`; see [`RnsFactorKernels`].
    #[instrument(skip_all)]
    fn compute_rns_factor_kernels(&self, p: i64, ntt_lprime: &B::NTTPlan, ntt_k: &B::NTTPlan) -> RnsFactorKernels {
        let field = &self.field;
        let m = field.m() as i64;

        // find a primitive m-th root of unity mod p
        let m_factors = field.m_factors();
        let zeta = (2..)
            .map(|candidate| pow_mod(candidate, ((p - 1) / m) as u64, p))
            .find(|zeta| m_factors.iter().all(|f| pow_mod(*zeta, (m / *f as i64) as u64, p) != 1))
            .unwrap();
        // ζ_mʼ and ζ_(m_K) as powers of ζ_m
        let zeta_mprime = pow_mod(zeta, crate::field::crt2(1, field.m_prime(), 0, field.m_k()), p);
        let zeta_mk = pow_mod(zeta, crate::field::crt2(0, field.m_prime(), 1, field.m_k()), p);

        // kernel_Lʼ[j] = Σ_(x in Gal(Mʼ|Lʼ)) ζ_mʼ^(x g_j^-1) = Tr_(Mʼ|Lʼ)(g_j^-1(ζ_mʼ)) mod 𝔭
        let compute_kernel = |zeta_part: i64, part_order: u64, subgroup: &[u64], transversal: &[u64]| {
            // power table of zeta_part
            let mut powers = Vec::with_capacity(part_order as usize);
            let mut current = 1;
            for _ in 0..part_order {
                powers.push(current);
                current = mul_mod(current, zeta_part, p);
            }
            transversal
                .iter()
                .map(|g| {
                    let g_inv = crate::field::inv_mod(*g, field.m());
                    let mut sum: i64 = 0;
                    for x in subgroup {
                        let exponent = (*x as u128 * g_inv as u128 % field.m() as u128) as u64 % part_order;
                        sum = (sum + powers[exponent as usize]) % p;
                    }
                    sum
                })
                .collect::<Vec<_>>()
        };
        let kernel_lprime = compute_kernel(
            zeta_mprime,
            field.m_prime(),
            field.gal_mprime_lprime_elements(),
            &field.lprime_q_transversal(),
        );
        let kernel_k = compute_kernel(zeta_mk, field.m_k(), field.gal_mk_k_elements(), &field.kq_transversal());

        // transform the kernels into the NTT domain (montgomery form), and invert them there
        let to_ntt_domain = |kernel: &[i64], plan: &B::NTTPlan, shape: [usize; DIMS], dim: DataDim| {
            let mont_kernel = kernel.iter().map(|x| mont_repr(*x, p)).collect::<Vec<_>>();
            let mut tensor = self.backend.hal_allocate_i64(shape);
            self.backend.hal_import(tensor.as_mut(), &mont_kernel);
            self.backend.hal_ntt_mont64_dim(tensor.as_mut(), std::slice::from_ref(plan), dim);
            let mut result = vec![0; kernel.len()];
            self.backend.hal_export(tensor.as_ref(), &mut result);
            for x in &mut result {
                *x = x.rem_euclid(p);
            }
            result
        };
        let deg_lprime = field.deg_lprime();
        let deg_k = field.deg_k();
        let ntt_kernel_lprime = to_ntt_domain(&kernel_lprime, ntt_lprime, [1, 1, deg_lprime, 1], DataDim::Dim2);
        let ntt_kernel_k = to_ntt_domain(&kernel_k, ntt_k, [1, 1, 1, deg_k], DataDim::Dim3);
        let invert = |kernel: &[i64]| {
            kernel
                .iter()
                .map(|x| {
                    let plain = from_mont_repr(*x, p);
                    assert!(plain != 0, "CRT kernel is not invertible");
                    mont_repr(pow_mod(plain, (p - 2) as u64, p), p)
                })
                .collect::<Vec<_>>()
        };
        return RnsFactorKernels {
            inv_crt_kernel_lprime: invert(&ntt_kernel_lprime),
            inv_crt_kernel_k: invert(&ntt_kernel_k),
            crt_kernel_lprime: ntt_kernel_lprime,
            crt_kernel_k: ntt_kernel_k,
        };
    }

    /// Copies tensor contents; broadcasting `src` is supported.
    pub fn assign(&self, dst: TensorI64Mut<B>, src: TensorI64Ref<B>) { self.backend.hal_assign(dst, src); }

    /// Copies (a view of) a batch of ring elements into a new owned batch. Note that
    /// restriction to a smaller RNS base ([`RingEls::restrict_base()`]) and row selection
    /// ([`RingEls::rows()`]) are copy-free views; a copy is only needed when the data is
    /// subsequently mutated (e.g. transformed in place, or destroyed by a base conversion).
    pub fn clone_els<'a>(&self, els: impl Into<RingElsRef<'a, B>>) -> RingEls<B>
    where
        B: 'a,
    {
        let els = els.into();
        let mut result = self.allocate_in_basis(els.count(), els.rns_base, els.basis);
        self.assign(result.data.as_mut(), els.data);
        result.bound = els.bound;
        return result;
    }

    /// Computes `wide += els * factor` for a broadcastable tensor `factor` (e.g. a scalar
    /// tensor, or an embedded `Lʼ`-element from [`RingContext::lprime_element_tensor()`] if
    /// `els` is in CRT basis). The factor must be in montgomery form (bound 1).
    pub fn fma_tensor<'a>(&self, wide: &mut WideRingEls<B>, els: impl Into<RingElsRef<'a, B>>, factor: TensorI64Ref<B>)
    where
        B: 'a,
    {
        let els = els.into();
        assert_eq!(&*wide.rns_base, els.rns_base);
        wide.fma_all(&self.backend, els.data, els.bound, factor, 1);
    }

    /// Creates a single ring element with the given period-basis coefficients (of length
    /// `deg_L`, in transversal order).
    pub fn element_from_pb_coeffs(&self, base: &RNSBaseIndices, coeffs: &[i128]) -> RingEls<B> {
        assert_eq!(self.n(), coeffs.len());
        let mut values = Vec::with_capacity(base.len() * coeffs.len());
        for idx in base.iter() {
            let p = self.rns_factors[*idx].p;
            for c in coeffs {
                values.push(mont_repr((*c % p as i128) as i64, p));
            }
        }
        let mut result = self.allocate(1, base);
        self.backend.hal_import(result.data.as_mut(), &values);
        return result;
    }

    /// Returns the smallest lifts (w.r.t. the product of the elements' RNS base) of all
    /// coefficients of the `idx`-th element of the batch, in transversal order.
    pub fn element_lifts(&self, els: &RingEls<B>, idx: usize) -> Vec<El<BigIntRing>> {
        let n = self.n();
        let r = els.rns_base.len();
        let mut data = vec![0i64; r * n];
        self.backend.hal_export(els.data.as_ref().slice(idx..idx + 1, .., .., ..), &mut data);
        let zn_rns = self.zn_rns_for(&els.rns_base);
        return (0..n)
            .map(|j| {
                zn_rns.smallest_lift(zn_rns.from_congruence((0..r).map(|i| {
                    let p = self.rns_factors[els.rns_base[i]].p;
                    let zn = &self.rns_factors[els.rns_base[i]].zn;
                    zn.coerce(&ZZi64, from_mont_repr(data[i * n + j], p))
                })))
            })
            .collect();
    }

    fn bconv_cache<'a>(
        &'a self,
        src_rns_base: &RNSBaseIndices,
        dst_rns_base: &RNSBaseIndices,
    ) -> MappedRwLockReadGuard<'a, BConvCache<B>> {
        let read = self.bconv_cache.read().unwrap();
        if let Some((idx, _)) = read
            .iter()
            .enumerate()
            .find(|(_, bconv)| &*bconv.src_rns_base == src_rns_base && &*bconv.dst_rns_base == dst_rns_base)
        {
            return RwLockReadGuard::map(read, |data| &data[idx]);
        }
        drop(read);
        let mut write = self.bconv_cache.write().unwrap();
        let result =
            BConvCache::new(self, src_rns_base.to_vec().into_boxed_slice(), dst_rns_base.to_vec().into_boxed_slice());
        let idx = write.len();
        write.push(result);
        drop(write);
        let read = self.bconv_cache.read().unwrap();
        return RwLockReadGuard::map(read, |data| &data[idx]);
    }

    /// Computes an RNS base conversion. Concretely, computes the map
    /// ```text
    ///   Z/qZ -> Z/q'Z,    x -> lift(x) + ε_x q mod q'
    /// ```
    /// where `ε_x in {-1, 0, 1}`, and `ε_x = 0` if `lift(x) <= q/4`.
    /// This map is applied separately to each slice `src[i, *, k, l]`, and the output is written to
    /// `dst[i, *, k, l]`.
    ///
    /// This function destroys the data in `src`. The result is almost-reduced; the input may
    /// have any representative size that permits a montgomery multiplication (the conversion
    /// is exact modulo the source primes, whatever the representatives).
    pub fn bconv(&self, dst: &mut RingEls<B>, src: &mut RingEls<B>) {
        self.assert_mul_mont_safe(src.bound, &src.rns_base);
        dst.basis = src.basis;
        dst.bound = 1;
        let (src_base, dst_base) = (src.rns_base.clone(), dst.rns_base.clone());
        self.bconv_tensors(dst.data.as_mut(), &dst_base, src.data.as_mut(), &src_base);
    }

    /// The raw-tensor version of [`RingContext::bconv()`].
    #[instrument(skip_all)]
    pub fn bconv_tensors(
        &self,
        dst: TensorI64Mut<B>,
        dst_rns_base: &RNSBaseIndices,
        src: TensorI64Mut<B>,
        src_rns_base: &RNSBaseIndices,
    ) {
        self.bconv_cache(src_rns_base, dst_rns_base).bconv(self, dst, src);
    }

    fn gadget_bconv_cache<'a>(
        &'a self,
        rns_base: &RNSBaseIndices,
        digits: &[Range<usize>],
    ) -> MappedRwLockReadGuard<'a, GadgetBConvCache<B>> {
        let read = self.gadget_bconv_cache.read().unwrap();
        if let Some((idx, _)) =
            read.iter().enumerate().find(|(_, cache)| &*cache.rns_base == rns_base && cache.digits == digits)
        {
            return RwLockReadGuard::map(read, |data| &data[idx]);
        }
        drop(read);
        let mut write = self.gadget_bconv_cache.write().unwrap();
        let result = GadgetBConvCache::new(self, rns_base.to_vec().into_boxed_slice(), digits.to_vec());
        let idx = write.len();
        write.push(result);
        drop(write);
        let read = self.gadget_bconv_cache.read().unwrap();
        return RwLockReadGuard::map(read, |data| &data[idx]);
    }

    /// Computes the base conversions of all gadget digits of a single element in one sequence
    /// of backend calls: for each digit `D_i` (a range of positions of `base`), the slice
    /// `dst[i, *, k, l]` receives the base conversion (in the sense of
    /// [`RingContext::bconv()`]) of `src[0, D_i, k, l]` from the digit's base to the full
    /// `base`.
    ///
    /// This function destroys the data in `src`.
    #[instrument(skip_all)]
    pub fn bconv_gadget_tensors(
        &self,
        dst: TensorI64Mut<B>,
        src: TensorI64Mut<B>,
        base: &RNSBaseIndices,
        digits: &[Range<usize>],
    ) {
        self.gadget_bconv_cache(base, digits).bconv(self, dst, src);
    }
}

/// Stores the precomputed data for the RNS base conversion of [`RingContext::bconv()`].
///
/// # Algorithm
///
/// This is a variant of the classical [BEHZ16]-style base conversion. Concretely, observe that
/// ```text
///   x = Σ_(p|q) lift(x * (q/p)^-1 mod p) q/p mod q
/// ```
/// Now we estimate how many multiples of `q` are removed by the final `mod q`, by choosing `q̃
/// << q`, and compute
/// ```text
///   c = rounddiv( Σ_(p|q) lift(x * (q/p)^-1 mod p) round(q̃/p), q̃)
/// ```
/// Then we find that `x_res mod q'`, given in RNS form by
/// ```text
///   x_res mod p' = Σ_(p|q) lift(x * (q/p)^-1 mod p) q/p mod p' - (q mod p') c
/// ```
/// satisfies the claimed identity (assuming `q̃` has enough precision).
struct BConvCache<B: Backend> {
    src_rns_base: Box<RNSBaseIndices>,
    dst_rns_base: Box<RNSBaseIndices>,
    log2_qtilde: usize,
    q_over_p_inv: TensorI64Owned<B>,
    matrix: TensorI64Owned<B>,
    q_mod_p_: TensorI64Owned<B>,
    acc_before_reduce: usize,
}

impl<B: Backend> BConvCache<B> {
    fn new(ring: &RingContext<B>, src_rns_base: Box<RNSBaseIndices>, dst_rns_base: Box<RNSBaseIndices>) -> Self {
        let r = src_rns_base.len();
        let r_ = dst_rns_base.len();

        let max_p = src_rns_base.iter().chain(dst_rns_base.iter()).map(|i| ring.rns_factors[*i].p).max().unwrap();
        let min_p = src_rns_base.iter().map(|i| ring.rns_factors[*i].p).min().unwrap();
        let q = ZZbig.prod(src_rns_base.iter().map(|i| int_cast(ring.rns_factors[*i].p, ZZbig, ZZi64)));
        let log2_qtilde = usize::max(64, ZZi128.abs_log2_ceil(&(4 * r as i128 * max_p as i128 + 1)).unwrap());
        let qtilde: i128 = 1 << log2_qtilde;
        let correct_vec_max_coeff = (qtilde - 1) / min_p as i128 + 1;
        assert!(correct_vec_max_coeff <= i64::MAX as i128, "coefficients of correction vector don't fit into an i64");
        assert!(
            (r as i128 * max_p as i128 * correct_vec_max_coeff) <= i128::MAX,
            "correction vector inner product overflows"
        );
        let acc_before_reduce = (i128::MAX / (max_p as i128 * max_p as i128)) as usize;
        assert!(
            acc_before_reduce >= 16,
            "to avoid overflow, modular reductions have to be performed too often to be efficient"
        );
        assert!(
            r.div_ceil(acc_before_reduce) as i128 * max_p as i128 <= i64::MAX as i128,
            "overflow during accumulation"
        );

        let q_over_p_inv = ring.q_over_p_inv_tensor(&src_rns_base, 1);

        let mut matrix_vec = Vec::with_capacity((r_ + 1) * r);
        for i in 0..r_ {
            let zp_ = ring.rns_factors[dst_rns_base[i]].zn;
            let r_sqr = zp_.pow(zp_.coerce(&ZZi128, 1 << 64), 2);
            for j in 0..r {
                let zp = ring.rns_factors[src_rns_base[j]].zn;
                let value = zp_.coerce(&ZZbig, ZZbig.checked_div(&q, &int_cast(*zp.modulus(), ZZbig, ZZi64)).unwrap());
                matrix_vec.push(i64::try_from(zp_.smallest_lift(zp_.mul(value, r_sqr))).unwrap());
            }
        }
        for j in 0..r {
            let p = ring.rns_factors[src_rns_base[j]].p;
            matrix_vec.push(i64::try_from(ZZi128.rounded_div(qtilde, &(p as i128))).unwrap());
        }
        let mut matrix = ring.backend.hal_allocate_i64([1, r_ + 1, r, 1]);
        ring.backend.hal_import(matrix.as_mut(), &matrix_vec);

        let mut q_mod_p_vec = Vec::with_capacity(r_);
        for i in 0..r_ {
            let zp_ = ring.rns_factors[dst_rns_base[i]].zn;
            let neg_r = zp_.coerce(&ZZi128, -1 << 64);
            q_mod_p_vec.push(
                i64::try_from(zp_.smallest_lift(zp_.mul(zp_.coerce(&ZZbig, ZZbig.clone_el(&q)), neg_r))).unwrap(),
            );
        }
        let mut q_mod_p_ = ring.backend.hal_allocate_i64([1, r_, 1, 1]);
        ring.backend.hal_import(q_mod_p_.as_mut(), &q_mod_p_vec);
        return Self { dst_rns_base, src_rns_base, log2_qtilde, q_over_p_inv, matrix, q_mod_p_, acc_before_reduce };
    }

    fn bconv(&self, ring: &RingContext<B>, mut dst: TensorI64Mut<B>, mut src: TensorI64Mut<B>) {
        let r = self.src_rns_base.len();
        let r_ = self.dst_rns_base.len();
        let src_shape = TensorMut::shape(&src);
        let shape = [src_shape[0], 1, src_shape[2], src_shape[3]];
        assert_eq!([shape[0], r_, shape[2], shape[3]], TensorMut::shape(&dst));

        let src_mods = ring.mods_for(&self.src_rns_base);
        ring.backend.hal_mul_mont64(src.reborrow(), self.q_over_p_inv.as_ref(), &src_mods);

        let dst_mods = ring.mods_for(&self.dst_rns_base);
        let mut correction = ring.backend.hal_allocate_i128([shape[0], 1, shape[2], shape[3]]);
        let mut tmp = ring.backend.hal_allocate_i128([shape[0], r_ + 1, shape[2], shape[3]]);
        // the first chunk overwrites dst/correction, later chunks accumulate; this avoids
        // zeroing them upfront (the loop runs at least once since r >= 1)
        for (chunk, j) in (0..r).step_by(self.acc_before_reduce).enumerate() {
            ring.backend.hal_matmul_i64(
                tmp.as_mut(),
                self.matrix.as_ref().slice(.., .., j..usize::min(j + self.acc_before_reduce, r), ..),
                src.as_const().slice(.., j..usize::min(j + self.acc_before_reduce, r), .., ..),
            );
            if chunk == 0 {
                ring.backend.hal_red_mont64_double(dst.reborrow(), tmp.as_ref().slice(.., ..r_, .., ..), &dst_mods);
                ring.backend.hal_assign_i128(correction.as_mut(), tmp.as_ref().slice(.., r_.., .., ..));
            } else {
                ring.backend.hal_add_red_mont64_double(dst.reborrow(), tmp.as_ref().slice(.., ..r_, .., ..), &dst_mods);
                ring.backend.hal_add_i128(correction.as_mut(), tmp.as_ref().slice(.., r_.., .., ..));
            }
        }
        ring.backend.hal_bconv_correct_mont64(
            dst.reborrow(),
            correction.as_ref(),
            self.q_mod_p_.as_ref(),
            &dst_mods,
            self.log2_qtilde,
        );

        ring.backend.hal_i64_to_mont64(dst, &dst_mods);
    }
}

/// Stores the precomputed data for the batched gadget-digit base conversion
/// [`RingContext::bconv_gadget_tensors()`].
///
/// This is the same [BEHZ16]-style algorithm as [`BConvCache`], except that the source RNS
/// base varies along the first (batch) dimension: the `i`-th output row is the conversion of
/// the `i`-th gadget digit `D_i` (a contiguous, disjoint range of positions of a common RNS
/// base `q`) to the full base `q`. The conversions of all digits share a single sequence of
/// backend calls: the matmul's first dimension enumerates the digits, with the conversion
/// matrix zeroed outside each digit's columns (so output row `i` only reads the residues of
/// `D_i`), and the single source element is broadcast along the first dimension.
struct GadgetBConvCache<B: Backend> {
    rns_base: Box<RNSBaseIndices>,
    digits: Vec<Range<usize>>,
    log2_qtilde: usize,
    /// shape `[1, r, 1, 1]`: position `u` holds the plain lift of `(D(u)/p_u)^-1 mod p_u`,
    /// where `D(u)` is the modulus of the digit containing `u` (zero outside all digits)
    d_over_p_inv: TensorI64Owned<B>,
    /// shape `[d, r + 1, r, 1]`: `matrix[i, j, u] = lift(D_i/p_u * 2^128 mod p_j)` for
    /// `u in D_i` and zero otherwise; row `r` is the correction row `round(q̃/p_u)`
    matrix: TensorI64Owned<B>,
    /// shape `[d, r, 1, 1]`: `-D_i * 2^64 mod p_j`
    d_mod_p: TensorI64Owned<B>,
    acc_before_reduce: usize,
}

impl<B: Backend> GadgetBConvCache<B> {
    fn new(ring: &RingContext<B>, rns_base: Box<RNSBaseIndices>, digits: Vec<Range<usize>>) -> Self {
        let r = rns_base.len();
        let d = digits.len();
        assert!(d > 0);
        assert!(digits.iter().all(|digit| digit.start < digit.end && digit.end <= r));
        assert!(digits.windows(2).all(|pair| pair[0].end <= pair[1].start), "digits must be sorted and disjoint");

        let max_p = rns_base.iter().map(|i| ring.rns_factors[*i].p).max().unwrap();
        let min_p = rns_base.iter().map(|i| ring.rns_factors[*i].p).min().unwrap();
        let max_digit_len = digits.iter().map(|digit| digit.len()).max().unwrap();
        let log2_qtilde = usize::max(64, ZZi64.abs_log2_ceil(&(4 * max_digit_len as i64 * max_p + 1)).unwrap());
        let qtilde: i128 = 1 << log2_qtilde;
        let correct_vec_max_coeff = (qtilde - 1) / min_p as i128 + 1;
        assert!(correct_vec_max_coeff <= i64::MAX as i128, "coefficients of correction vector don't fit into an i64");
        assert!(
            (max_digit_len as i128)
                .checked_mul(max_p as i128)
                .and_then(|x| x.checked_mul(correct_vec_max_coeff))
                .is_some(),
            "correction vector inner product overflows"
        );
        let acc_before_reduce = (i128::MAX / (max_p as i128 * max_p as i128)) as usize;
        assert!(
            acc_before_reduce >= 16,
            "to avoid overflow, modular reductions have to be performed too often to be efficient"
        );
        assert!(
            r.div_ceil(acc_before_reduce) as i128 * max_p as i128 <= i64::MAX as i128,
            "overflow during accumulation"
        );

        let digit_moduli = digits.iter().map(|digit| ring.modulus_of(&rns_base[digit.clone()])).collect::<Vec<_>>();

        let mut d_over_p_inv_vec = vec![0i64; r];
        for (digit, d_mod) in digits.iter().zip(&digit_moduli) {
            for u in digit.clone() {
                let zp = &ring.rns_factors[rns_base[u]].zn;
                let d_over_p = zp.coerce(
                    &ZZbig,
                    ZZbig.checked_div(d_mod, &int_cast(ring.rns_factors[rns_base[u]].p, ZZbig, ZZi64)).unwrap(),
                );
                d_over_p_inv_vec[u] = zp.smallest_lift(zp.invert(&d_over_p).unwrap());
            }
        }
        let mut d_over_p_inv = ring.backend.hal_allocate_i64([1, r, 1, 1]);
        ring.backend.hal_import(d_over_p_inv.as_mut(), &d_over_p_inv_vec);

        let mut matrix_vec = vec![0i64; d * (r + 1) * r];
        let mut d_mod_p_vec = Vec::with_capacity(d * r);
        for (i, (digit, d_mod)) in digits.iter().zip(&digit_moduli).enumerate() {
            for j in 0..r {
                let zp_ = ring.rns_factors[rns_base[j]].zn;
                let r_sqr = zp_.pow(zp_.coerce(&ZZi128, 1 << 64), 2);
                for u in digit.clone() {
                    let p_u = ring.rns_factors[rns_base[u]].p;
                    let value = zp_.coerce(&ZZbig, ZZbig.checked_div(d_mod, &int_cast(p_u, ZZbig, ZZi64)).unwrap());
                    matrix_vec[(i * (r + 1) + j) * r + u] =
                        i64::try_from(zp_.smallest_lift(zp_.mul(value, r_sqr))).unwrap();
                }
                let neg_r = zp_.coerce(&ZZi128, -1 << 64);
                d_mod_p_vec.push(
                    i64::try_from(zp_.smallest_lift(zp_.mul(zp_.coerce(&ZZbig, ZZbig.clone_el(d_mod)), neg_r)))
                        .unwrap(),
                );
            }
            for u in digit.clone() {
                let p_u = ring.rns_factors[rns_base[u]].p;
                matrix_vec[(i * (r + 1) + r) * r + u] =
                    i64::try_from(ZZi128.rounded_div(qtilde, &(p_u as i128))).unwrap();
            }
        }
        let mut matrix = ring.backend.hal_allocate_i64([d, r + 1, r, 1]);
        ring.backend.hal_import(matrix.as_mut(), &matrix_vec);
        let mut d_mod_p = ring.backend.hal_allocate_i64([d, r, 1, 1]);
        ring.backend.hal_import(d_mod_p.as_mut(), &d_mod_p_vec);

        return Self { rns_base, digits, log2_qtilde, d_over_p_inv, matrix, d_mod_p, acc_before_reduce };
    }

    fn bconv(&self, ring: &RingContext<B>, mut dst: TensorI64Mut<B>, mut src: TensorI64Mut<B>) {
        let r = self.rns_base.len();
        let d = self.digits.len();
        let src_shape = TensorMut::shape(&src);
        assert_eq!([1, r, src_shape[2], src_shape[3]], src_shape);
        assert_eq!([d, r, src_shape[2], src_shape[3]], TensorMut::shape(&dst));
        let mods = ring.mods_for(&self.rns_base);

        // plain lifts of x_u * (D(u)/p_u)^-1 mod p_u, in (-p_u, p_u)
        ring.backend.hal_mul_mont64(src.reborrow(), self.d_over_p_inv.as_ref(), &mods);

        // per digit i: Σ_(u in D_i) lift_u (D_i/p_u) mod p_j (value rows, doubly-montgomery),
        // and the q̃-scaled estimate of the number of multiples of D_i (correction row); the
        // matrix is zero outside each digit's columns, and src is broadcast along dimension 0
        let mut correction = ring.backend.hal_allocate_i128([d, 1, src_shape[2], src_shape[3]]);
        let mut tmp = ring.backend.hal_allocate_i128([d, r + 1, src_shape[2], src_shape[3]]);
        // the first chunk overwrites dst/correction, later chunks accumulate; this avoids
        // zeroing them upfront (the loop runs at least once since r >= 1)
        for (chunk, u) in (0..r).step_by(self.acc_before_reduce).enumerate() {
            let end = usize::min(u + self.acc_before_reduce, r);
            ring.backend.hal_matmul_i64(
                tmp.as_mut(),
                self.matrix.as_ref().slice(.., .., u..end, ..),
                src.as_const().slice(.., u..end, .., ..),
            );
            if chunk == 0 {
                ring.backend.hal_red_mont64_double(dst.reborrow(), tmp.as_ref().slice(.., ..r, .., ..), &mods);
                ring.backend.hal_assign_i128(correction.as_mut(), tmp.as_ref().slice(.., r.., .., ..));
            } else {
                ring.backend.hal_add_red_mont64_double(dst.reborrow(), tmp.as_ref().slice(.., ..r, .., ..), &mods);
                ring.backend.hal_add_i128(correction.as_mut(), tmp.as_ref().slice(.., r.., .., ..));
            }
        }
        ring.backend.hal_bconv_correct_mont64(
            dst.reborrow(),
            correction.as_ref(),
            self.d_mod_p.as_ref(),
            &mods,
            self.log2_qtilde,
        );

        ring.backend.hal_i64_to_mont64(dst.reborrow(), &mods);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{field_2, field_18};

    #[test]
    fn test_bconv() {
        feanor_tracing::DelayedLogger::init_test();
        let rns_factors: [i64; 13] = [
            144115188075855859,
            144115188075855847,
            144115188075855823,
            144115188075855811,
            144115188075855803,
            144115188075855761,
            144115188075855677,
            144115188075855599,
            144115188075855509,
            144115188075855449,
            144115188075855439,
            144115188075855421,
            144115188075855413,
        ];
        let ring = RingContext::new_raw(CpuBackend, field_2(), &rns_factors);
        let from = &rns_factors[..7];
        let to = &rns_factors[7..13];
        let src_base = (0..7).collect::<Vec<_>>();
        let dst_base = (7..13).collect::<Vec<_>>();
        let mut src = ring.backend().hal_allocate_i64([2, 7, 8, 1]);
        let mut dst = ring.backend().hal_allocate_i64([2, 6, 8, 1]);

        let data1: [i128; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let data2: [i128; 8] = [1 << 64, 3 << 100, -7 << 100, 9 << 120, -1 << 120, 5 << 120, 0, 7 << 120];
        let mut src_vec = Vec::new();
        for i in 0..from.len() {
            for j in 0..8 {
                src_vec.push(montgomery_repr(data1[j] % from[i] as i128, from[i]));
            }
        }
        for i in 0..from.len() {
            for j in 0..8 {
                src_vec.push(montgomery_repr(data2[j] % from[i] as i128, from[i]));
            }
        }
        ring.backend().hal_import(src.as_mut(), &src_vec);
        ring.bconv_tensors(dst.as_mut(), &dst_base, src.as_mut(), &src_base);

        let mut dst_vec = (0..(16 * to.len())).map(|_| 0).collect::<Vec<_>>();
        ring.backend().hal_export(dst.as_ref(), &mut dst_vec);

        for i in 0..to.len() {
            for j in 0..8 {
                assert_eq!(pos_repr(montgomery_repr(data1[j], to[i]), to[i]), pos_repr(dst_vec[i * 8 + j], to[i]));
            }
        }
        for i in 0..to.len() {
            for j in 0..8 {
                assert_eq!(
                    pos_repr(montgomery_repr(data2[j], to[i]), to[i]),
                    pos_repr(dst_vec[8 * to.len() + i * 8 + j], to[i])
                );
            }
        }
    }

    /// the batched gadget-digit conversion must agree with converting every digit separately
    #[test]
    fn test_bconv_gadget() {
        feanor_tracing::DelayedLogger::init_test();
        let rns_factors: [i64; 7] = [
            144115188075855859,
            144115188075855847,
            144115188075855823,
            144115188075855811,
            144115188075855803,
            144115188075855761,
            144115188075855677,
        ];
        let ring = RingContext::new_raw(CpuBackend, field_2(), &rns_factors);
        let base = (0..7).collect::<Vec<_>>();
        let digits = vec![0..3usize, 3..5, 5..7];

        let data: [i128; 8] = [0, 1, -1 << 100, 3 << 100, -7 << 90, 9 << 120, 12345, 7 << 120];
        let mut src_vec = Vec::new();
        for i in 0..7 {
            for j in 0..8 {
                src_vec.push(montgomery_repr(data[j] % rns_factors[i] as i128, rns_factors[i]));
            }
        }

        // batched conversion of all digits at once
        let mut src = ring.backend().hal_allocate_i64([1, 7, 8, 1]);
        ring.backend().hal_import(src.as_mut(), &src_vec);
        let mut dst = ring.backend().hal_allocate_i64([3, 7, 8, 1]);
        ring.bconv_gadget_tensors(dst.as_mut(), src.as_mut(), &base, &digits);
        let mut out = vec![0i64; 3 * 7 * 8];
        ring.backend().hal_export(dst.as_ref(), &mut out);

        // reference: every digit converted separately
        for (i, digit) in digits.iter().enumerate() {
            let digit_base = base[digit.clone()].to_vec();
            let mut digit_src = ring.backend().hal_allocate_i64([1, digit.len(), 8, 1]);
            ring.backend().hal_import(digit_src.as_mut(), &src_vec[digit.start * 8..digit.end * 8]);
            let mut expected = ring.backend().hal_allocate_i64([1, 7, 8, 1]);
            ring.bconv_tensors(expected.as_mut(), &base, digit_src.as_mut(), &digit_base);
            let mut expected_vec = vec![0i64; 7 * 8];
            ring.backend().hal_export(expected.as_ref(), &mut expected_vec);
            for (pos, expected) in expected_vec.iter().enumerate() {
                let p = rns_factors[pos / 8];
                assert_eq!(
                    pos_repr(*expected, p),
                    pos_repr(out[i * 7 * 8 + pos], p),
                    "digit {}, flat position {}",
                    i,
                    pos
                );
            }
        }
    }

    /// multiplying the CRT transform of `1_(O_K) = scale * (1, ..., 1)_pb` with anything must
    /// be the identity, and squaring in CRT basis and transforming back must match the known
    /// period-basis relation of field_18 (ported from the reference implementation)
    #[test]
    fn test_pb_crt_transform_period_basis_relation() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = [0];
        let p = ring.rns_factor_prime(0);

        // the element with period-basis coefficients (1, 0, 0, 0, 0, 0) per Lʼ-block; its
        // square has coefficients [3, 4, 2, -2, -4, -2] in each block (computed with sage)
        let mut els = ring.allocate(1, &base);
        let mut data = vec![0i64; 18];
        for block in 0..3 {
            data[block * 6] = mont_repr(1, p);
        }
        ring.backend().hal_import(els.data.as_mut(), &data);
        ring.pb_to_crt(&mut els);
        let mut acc = ring.allocate_wide(1, &base);
        ring.fma(&mut acc, &els, &els);
        let mut sqr = ring.reduce_wide(&acc, Basis::CRTBasis);
        ring.crt_to_pb(&mut sqr);

        let mut out = vec![0i64; 18];
        ring.backend().hal_export(sqr.data.as_ref(), &mut out);
        let expected_in_k: [i64; 6] = [3, 4, 2, -2, -4, -2];
        for block in 0..3 {
            for j in 0..6 {
                assert_eq!(
                    pos_repr(montgomery_repr(expected_in_k[j], p), p),
                    pos_repr(out[block * 6 + j], p),
                    "block {}, coefficient {}",
                    block,
                    j
                );
            }
        }
    }

    #[test]
    fn test_pb_crt_roundtrip() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = ring.initial_rns_base();
        let mut els = ring.allocate(2, &base);
        let mut data = Vec::new();
        let mut state: u64 = 123;
        for _ in 0..(2 * base.len() * 18) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            data.push((state >> 12) as i64 % 1000000);
        }
        ring.backend().hal_import(els.data.as_mut(), &data);
        ring.pb_to_crt(&mut els);
        ring.crt_to_pb(&mut els);
        let mut out = vec![0i64; data.len()];
        ring.backend().hal_export(els.data.as_ref(), &mut out);
        for (pos, (expected, actual)) in data.iter().zip(&out).enumerate() {
            let p = ring.rns_factor_prime(base[(pos / 18) % base.len()]);
            assert_eq!(pos_repr(*expected, p), pos_repr(*actual, p), "at flat position {}", pos);
        }
    }

    /// multiplication in the CRT basis must be multiplication in `O_L`: squaring a period basis
    /// element must give coefficients bounded by `m`
    #[test]
    fn test_crt_transform_multiply_small() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = [0];
        let p = ring.rns_factor_prime(0);
        let mut els = ring.allocate(1, &base);
        let mut data = vec![0i64; 18];
        data[0] = mont_repr(1, p);
        ring.backend().hal_import(els.data.as_mut(), &data);
        ring.pb_to_crt(&mut els);
        let mut acc = ring.allocate_wide(1, &base);
        ring.fma(&mut acc, &els, &els);
        let mut sqr = ring.reduce_wide(&acc, Basis::CRTBasis);
        ring.crt_to_pb(&mut sqr);
        let mut out = vec![0i64; 18];
        ring.backend().hal_export(sqr.data.as_ref(), &mut out);
        for (i, x) in out.iter().enumerate() {
            let plain = from_mont_repr(*x, p);
            let small = i64::min(plain, p - plain);
            assert!(
                small <= ring.field().m() as i64,
                "expected coefficient {} of the square to be at most {}, got {}",
                i,
                ring.field().m(),
                small
            );
        }
    }

    /// unreduced additions on [`RingEls`] must match the reduced result of the wide path,
    /// track the representative bound, and panic before entries could overflow an i64
    #[test]
    fn test_add_els_unreduced() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = ring.initial_rns_base();
        let r = base.len();
        let mut a = ring.allocate(1, &base);
        let mut b = ring.allocate(1, &base);
        let a_vals: Vec<i64> =
            (0..r * 18).map(|pos| mont_repr((pos % 37) as i64 - 18, ring.rns_factor_prime(base[pos / 18]))).collect();
        let b_vals: Vec<i64> =
            (0..r * 18).map(|pos| mont_repr((pos % 23) as i64 - 11, ring.rns_factor_prime(base[pos / 18]))).collect();
        ring.backend().hal_import(a.data.as_mut(), &a_vals);
        ring.backend().hal_import(b.data.as_mut(), &b_vals);

        // a + b - b unreduced; bounds accumulate per operation
        let mut sum = ring.clone_els(&a);
        ring.add_els(&mut sum, &b);
        assert_eq!(2, sum.bound);
        ring.sub_els(&mut sum, &b);
        assert_eq!(3, sum.bound);

        // must agree (mod p) with the reduced result of the wide path
        let mut wide = ring.allocate_wide(1, &base);
        ring.add(&mut wide, &a);
        let reference = ring.reduce_wide(&wide, Basis::PeriodBasis);
        let mut sum_out = vec![0i64; r * 18];
        let mut ref_out = vec![0i64; r * 18];
        ring.backend().hal_export(sum.data.as_ref(), &mut sum_out);
        ring.backend().hal_export(reference.data.as_ref(), &mut ref_out);
        for pos in 0..r * 18 {
            let p = ring.rns_factor_prime(base[pos / 18]);
            assert!(sum_out[pos].unsigned_abs() as u128 <= 3 * p as u128);
            assert_eq!(pos_repr(ref_out[pos], p), pos_repr(sum_out[pos], p), "at {}", pos);
        }

        // repeated unreduced additions must panic before the i64 range could overflow
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut runaway = ring.clone_els(&a);
            for _ in 0..128 {
                let copy = ring.clone_els(&runaway);
                ring.add_els(&mut runaway, &copy);
            }
        }));
        assert!(result.is_err(), "expected an overflow panic");
    }

    /// wide batches are not zeroed on allocation: the first accumulation into a row must
    /// overwrite it, rows that are never accumulated into must finish to zero, and row ranges
    /// mixing fresh and already-touched rows (as well as rows needing different reduction
    /// kinds when finishing) must be dispatched correctly
    #[test]
    fn test_wide_freshness() {
        feanor_tracing::DelayedLogger::init_test();
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = ring.initial_rns_base();
        let r = base.len();
        let mut els = ring.allocate(4, &base);
        let values: Vec<i64> = (0..4 * r * 18).map(|pos| (pos % 50) as i64 + 1).collect();
        let values_mont: Vec<i64> = values
            .iter()
            .enumerate()
            .map(|(pos, x)| mont_repr(*x, ring.rns_factor_prime(base[(pos / 18) % r])))
            .collect();
        ring.backend().hal_import(els.data.as_mut(), &values_mont);

        // rows 1..3 are accumulated into first; the subsequent accumulation over 0..4 covers
        // the runs [0..1] (fresh), [1..3] (touched), [3..4] (fresh)
        let mut acc = ring.allocate_wide(4, &base);
        acc.add(ring.backend(), 1..3, els.data.as_ref().slice(1..3, .., .., ..), els.bound);
        acc.add(ring.backend(), 0..4, els.data.as_ref(), els.bound);
        // push row 0 above the single-reduction bound, so that finishing needs the full
        // double reduction for row 0 but not for rows 1..4
        let extra_adds = 100;
        for _ in 0..extra_adds {
            acc.add(ring.backend(), 0..1, els.data.as_ref().slice(0..1, .., .., ..), els.bound);
        }
        assert!(acc.needs_full_reduction(0) && !acc.needs_full_reduction(1));
        let finished = ring.reduce_wide(&acc, Basis::PeriodBasis);
        let mut out = vec![0i64; 4 * r * 18];
        ring.backend().hal_export(finished.data.as_ref(), &mut out);
        for pos in 0..4 * r * 18 {
            let p = ring.rns_factor_prime(base[(pos / 18) % r]);
            assert!(out[pos].abs() <= p);
            let multiplier = match pos / (r * 18) {
                0 => 1 + extra_adds,
                1 | 2 => 2,
                _ => 1,
            };
            assert_eq!(pos_repr(multiplier * values[pos], p), pos_repr(from_mont_repr(out[pos], p), p), "at {}", pos);
        }

        // a wide batch that is never accumulated into reduces to zero
        let acc = ring.allocate_wide(2, &base);
        let finished = ring.reduce_wide(&acc, Basis::PeriodBasis);
        let mut out = vec![1i64; 2 * r * 18];
        ring.backend().hal_export(finished.data.as_ref(), &mut out);
        assert!(out.iter().all(|x| *x == 0));
    }

    #[test]
    fn test_apply_galois() {
        feanor_tracing::DelayedLogger::init_test();
        // group structure of Gal(L|Q) for field_18 is 3 x 2 x 3
        let ring = RingContext::new(CpuBackend, field_18(), 100..120, 4);
        let base = [0, 1];
        let mut els = ring.allocate(1, &base);
        let data: Vec<i64> = (0..36).collect();
        ring.backend().hal_import(els.data.as_mut(), &data);

        // pick the generator of the first K-factor (order 2); it shifts dimension 1 by 1
        let g = ring.field().lq_cyclic_decomposition()[1].0;
        let mut dst = ring.allocate(1, &base);
        ring.apply_galois(&mut dst, &els, g);
        let mut out = vec![0i64; 36];
        ring.backend().hal_export(dst.data.as_ref(), &mut out);
        for j in 0..2 {
            for a0 in 0..3 {
                for a1 in 0..2 {
                    for a2 in 0..3 {
                        let src_pos = j * 18 + (a0 * 2 + a1) * 3 + a2;
                        let dst_pos = j * 18 + (a0 * 2 + (a1 + 1) % 2) * 3 + a2;
                        assert_eq!(data[src_pos], out[dst_pos]);
                    }
                }
            }
        }

        // applying g for all g in the transversal, starting from a fixed element, permutes
        // transitively; composing shifts must be consistent: g1 then g2 equals g1*g2
        let g1 = ring.field().lq_cyclic_decomposition()[0].0;
        let g2 = ring.field().lq_cyclic_decomposition()[2].0;
        let mut tmp = ring.allocate(1, &base);
        ring.apply_galois(&mut tmp, &els, g1);
        ring.apply_galois(&mut dst, &tmp, g2);
        let mut expected = ring.allocate(1, &base);
        ring.apply_galois(&mut expected, &els, (g1 as u128 * g2 as u128 % ring.field().m() as u128) as u64);
        let mut out2 = vec![0i64; 36];
        ring.backend().hal_export(dst.data.as_ref(), &mut out);
        ring.backend().hal_export(expected.data.as_ref(), &mut out2);
        assert_eq!(out, out2);
    }
}
