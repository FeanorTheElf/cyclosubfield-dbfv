use std::ops::RangeBounds;

pub mod cpu;
pub mod mont;
pub mod ntt;

pub const DIMS: usize = 4;

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub enum Dim {
    Dim0,
    Dim1,
    Dim2,
    Dim3,
}

pub trait TensorRef: Copy {
    fn shape(self) -> [usize; DIMS];

    fn slice<B0: RangeBounds<usize>, B1: RangeBounds<usize>, B2: RangeBounds<usize>, B3: RangeBounds<usize>>(
        self,
        idx0: B0,
        idx1: B1,
        idx2: B2,
        idx3: B3,
    ) -> Self;
}

pub trait TensorMut<'b, T: Copy, Tensor: 'b + TensorOwned<T>>: 'b + Sized {
    fn shape(&self) -> [usize; DIMS];
    fn reborrow<'a>(&'a mut self) -> Tensor::Mut<'a>;
    fn as_const<'a>(&'a self) -> Tensor::Ref<'a>;
    fn into_const(self) -> Tensor::Ref<'b>;

    fn split(self, at: usize, dim: Dim) -> (Self, Self);

    fn slice<B0: RangeBounds<usize>, B1: RangeBounds<usize>, B2: RangeBounds<usize>, B3: RangeBounds<usize>>(
        self,
        idx0: B0,
        idx1: B1,
        idx2: B2,
        idx3: B3,
    ) -> Self;
}

pub trait TensorOwned<T: Copy>: Sized {
    type Ref<'a>: TensorRef
    where
        Self: 'a;

    type Mut<'a>: TensorMut<'a, T, Self>
    where
        Self: 'a;

    fn shape(self) -> [usize; DIMS];
    fn as_ref<'a>(&'a self) -> Self::Ref<'a>;
    fn as_mut<'a>(&'a mut self) -> Self::Mut<'a>;
}

pub type TensorI64Mut<'a, B> = <<B as Backend>::TensorI64 as TensorOwned<i64>>::Mut<'a>;
pub type TensorI64Ref<'a, B> = <<B as Backend>::TensorI64 as TensorOwned<i64>>::Ref<'a>;
pub type TensorI64Owned<B> = <B as Backend>::TensorI64;
pub type TensorI128Mut<'a, B> = <<B as Backend>::TensorI128 as TensorOwned<i128>>::Mut<'a>;
pub type TensorI128Ref<'a, B> = <<B as Backend>::TensorI128 as TensorOwned<i128>>::Ref<'a>;
pub type TensorI128Owned<B> = <B as Backend>::TensorI128;
pub type Moduli<B> = [<B as Backend>::Modulus];

/// Splits `x` into two signed 64-bit limbs `(lo, hi)` with `x = hi * 2^64 + lo`, where `lo` is
/// the low 64 bits of `x` reinterpreted as an `i64`. This is the limb representation used by
/// [`Backend::bconv_correct_pow2()`]; it requires `|x| < 2^126` (so that `hi` fits).
pub fn split_i128_limbs(x: i128) -> (i64, i64) {
    debug_assert!(x.unsigned_abs() < 1 << 126);
    let lo = x as i64;
    let hi = ((x - lo as i128) >> 64) as i64;
    return (lo, hi);
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub enum DataDim {
    Dim2,
    Dim3,
}

/// A backend for tensor computations with integers and modular arithmetic.
///
/// # Broadcasting
///
/// Every operation that acts pointwise on tensors (i.e. [`Backend::add()`],
/// [`Backend::sub()`], [`Backend::fma()`]), the source tensors may have length 1 in
/// some dimensions, in which case this dimension is broadcasted to the full output length.
///
/// # Modular arithmetic
///
/// Each tensor has 4 dimensions. The 1-st one of them (using zero-indexing) is the only one in
/// which the modulus for modular arithmetic can vary. Concretely, if a tensor represents modular
/// numbers, the entry `tensor[i, j, k, l]` is defined w.r.t. `moduli[j]`.
///
/// # NTTs
///
/// The backend supports modular group-NTTs along the 2nd and 3rd dimension, either separately or
/// both. NTT plans are created via [`Backend::create_ntt_plan()`]. That function takes an "inner
/// shape" parameter, which is a list `[n0, ..., n(r - 1)]`, and has total length `n = n0 * ... *
/// n(r - 1)`. The resulting modular NTT then represents the group-NTT over the group `C_n0 x ... x
/// C_n(r - 1)`; it can then be applied, either along the 2nd dimension (i.e. separately on each
/// slice `dst[i, j, *, l]`) or along the 3rd dimension (i.e. separately on each slice `dst[i, j, k,
/// *]`), or both. Of course, the total length of the NTT plan `n` must be equal to the length of
/// the corresponding dimension.
pub trait Backend {
    type TensorI64: Send + Sync + TensorOwned<i64>;
    type TensorI128: Send + Sync + TensorOwned<i128>;
    type Modulus: Send + Sync + Clone;
    type NTTPlan: Send + Sync + Clone;
    type PermutationPlan: Send + Sync + Clone;

    /// Allocates a zero-initialized tensor. Zeroing on allocation is essentially free (fresh
    /// zero pages/`calloc`); consumers must nevertheless prefer the overwriting operations
    /// ([`Backend::mul_i64()`], [`Backend::red_mont64()`], ...) over accumulating into the
    /// zeros, since the latter reads the destination once more.
    fn hal_allocate_i64(&self, shape: [usize; DIMS]) -> TensorI64Owned<Self>;

    /// Allocates a zero-initialized tensor; see [`Backend::allocate_i64()`].
    fn hal_allocate_i128(&self, shape: [usize; DIMS]) -> TensorI128Owned<Self>;

    fn hal_import(&self, dst: TensorI64Mut<Self>, src: &[i64]);
    fn hal_export(&self, src: TensorI64Ref<Self>, dst: &mut [i64]);

    /// Copies `src` into `dst`; as for the pointwise arithmetic operations, `src` may be
    /// broadcast in any dimension.
    fn hal_assign(&self, dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>);

    /// Copies `src` into `dst`, narrowing every entry to 64 bits; requires that every entry of
    /// `src` fits into an `i64`. As for [`Backend::assign()`], `src` may be broadcast in any
    /// dimension.
    fn hal_assign_narrow(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>);

    /// The `i128` version of [`Backend::assign()`].
    fn hal_assign_i128(&self, dst: TensorI128Mut<Self>, src: TensorI128Ref<Self>);

    fn hal_add_i64(&self, dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>);
    fn hal_sub_i64(&self, dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>);
    fn hal_add_i128(&self, dst: TensorI128Mut<Self>, src: TensorI128Ref<Self>);

    /// Computes `dst += lhs * rhs` pointwise (with broadcasting).
    fn hal_fma_i64(&self, dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>);

    /// The overwriting counterpart of [`Backend::fma_i64()`]: computes `dst = lhs * rhs`
    /// pointwise (with broadcasting), without reading `dst`. Use this for the first
    /// accumulation step instead of zeroing `dst` and accumulating.
    fn hal_mul_i64(&self, dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>);

    /// Computes the term-wise inner product
    /// ```text
    ///   dst = Σ_t lhs[t] * rhs[t]
    /// ```
    /// pointwise (with per-term broadcasting), *overwriting* `dst` (i.e. without reading its
    /// previous value). This is the multi-term counterpart of [`Backend::mul_i64()`]
    /// followed by a chain of [`Backend::fma_i64()`]s, and produces the identical result; it
    /// exists so that the backend can keep the `i128` accumulator resident in a fast cache
    /// while summing the terms, instead of streaming the whole accumulator through memory once
    /// per term. `lhs` and `rhs` must have the same length (at least one term); each `lhs[t]`
    /// and `rhs[t]` is broadcast to the shape of `dst` just like the operands of
    /// [`Backend::fma_i64()`].
    fn hal_inner_product_i64(&self, dst: TensorI128Mut<Self>, lhs: &[TensorI64Ref<Self>], rhs: &[TensorI64Ref<Self>]);

    /// The matmul dimension is the 1-st one (using zero-indexing).
    ///
    /// In other words, this computes
    /// ```text
    ///   dst[i, j, k, l] = Σ_u lhs[i, j, u] * rhs[i, u, k, l]
    /// ```
    /// Requires that the 3-rd dimension of `lhs` has length 1. Broadcasting of `lhs` and/or
    /// `rhs` in the 0-th dimension is also supported.
    fn hal_matmul_i64(&self, dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>);

    fn hal_create_modulus(&self, prime: i64) -> Self::Modulus;

    /// Expects the component-wise product of `lhs` and `rhs` to be in `[-2^63 N, 2^63 N]`. The
    /// output is then the component-wise montgomery multiplication, reduced to `[-N, N]`.
    fn hal_mul_mont64(&self, dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>, mods: &Moduli<Self>);

    /// Computes `dst += double_red_mont(src)`, where `double_red_mont` is the component-wise
    /// double-montgomery reduction. In particular, `double_red_mont` works correctly on any
    /// 128-bit input (doesn't have to be bounded by `2^63 N`), and computes a value congruent to
    /// `2^(-128) * input` modulo `N`, and in the interval `[-N, N]`.
    fn hal_add_red_mont64_double(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>);

    /// The overwriting counterpart of [`Backend::add_red_mont64_double()`]: computes
    /// `dst = double_red_mont(src)` without reading `dst`.
    fn hal_red_mont64_double(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>);

    /// Expects `src` to be within `[-2^63 N, 2^63 N]`. Computes `dst += red_mont(src)`, where
    /// `red_mont` is the component-wise montgomery reduction. In particular, `red_mont`
    /// computes a value congruent to `2^(-64) * input` modulo `N`, and in the interval `[-N,
    /// N]`.
    fn hal_add_red_mont64(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>);

    /// The overwriting counterpart of [`Backend::add_red_mont64()`]: computes
    /// `dst = red_mont(src)` without reading `dst`; `src` must be within `[-2^63 N, 2^63 N]`.
    fn hal_red_mont64(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>);

    /// Expects `src` to be within `[-2^63 N, 2^63 N]`. The output is then the montgomery form of
    /// the input, reduced to `[-N, N]`.
    fn hal_i128_to_mont64(&self, dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>);

    /// Works for any 64-bit input. The output is then the montgomery form of the
    /// input, reduced to `[-N, N]`.
    fn hal_i64_to_mont64(&self, dst: TensorI64Mut<Self>, mods: &Moduli<Self>);

    fn hal_create_ntt_plan(&self, modulus: &Self::Modulus, inner_shape: &[usize]) -> Self::NTTPlan;

    /// Applies the group-NTT of `ntt[j]` to every slice `dst[i, j, *, l]` (`dim == Dim2`) or
    /// `dst[i, j, k, *]` (`dim == Dim3`). The transforms work in the montgomery domain
    /// (montgomery-form inputs map to montgomery-form outputs). Input entries must be within
    /// `[-2N, 2N]` (w.r.t. the modulus `N` of `ntt[j]`); output entries are reduced into
    /// `[-N, N]`.
    fn hal_ntt_mont64_dim(&self, dst: TensorI64Mut<Self>, ntt: &[Self::NTTPlan], dim: DataDim);

    /// The inverse of [`Backend::ntt_mont64_dim()`]; same input/output bounds.
    fn hal_intt_mont64_dim(&self, dst: TensorI64Mut<Self>, ntt: &[Self::NTTPlan], dim: DataDim);

    /// Applies the forward group-NTT of `ntt[j]` along `dim` (as [`Backend::ntt_mont64_dim()`])
    /// and then montgomery-multiplies the result pointwise by `mul` (as
    /// [`Backend::mul_mont64()`], so `mul` may be broadcast). This is equivalent to an
    /// `ntt_mont64_dim` followed by a `mul_mont64`, but fuses the two: the transform leaves its
    /// output unreduced (skipping its final reduction pass) and the multiplication performs the
    /// reduction. Input entries must be within `[-2N, 2N]`; output entries are reduced into
    /// `[-N, N]`.
    ///
    /// This is used by the CRT transforms, whose forward transform is always followed by a
    /// pointwise multiplication with a CRT-kernel tensor.
    fn hal_ntt_mul_mont64_dim(
        &self,
        dst: TensorI64Mut<Self>,
        ntt: &[Self::NTTPlan],
        dim: DataDim,
        mul: TensorI64Ref<Self>,
        mods: &Moduli<Self>,
    );

    /// Shortcut for [`Backend::ntt_mont64_dim()`] in the last two dimensions.
    fn hal_ntt_mont64(&self, mut dst: TensorI64Mut<Self>, dim2: &[Self::NTTPlan], dim3: &[Self::NTTPlan]) {
        self.hal_ntt_mont64_dim(dst.reborrow(), dim2, DataDim::Dim2);
        self.hal_ntt_mont64_dim(dst.reborrow(), dim3, DataDim::Dim3);
    }

    /// Shortcut for [`Backend::intt_mont64_dim()`] in the last two dimensions.
    fn hal_intt_mont64(&self, mut dst: TensorI64Mut<Self>, dim2: &[Self::NTTPlan], dim3: &[Self::NTTPlan]) {
        self.hal_intt_mont64_dim(dst.reborrow(), dim2, DataDim::Dim2);
        self.hal_intt_mont64_dim(dst.reborrow(), dim3, DataDim::Dim3);
    }

    /// Computes the correction step required for base conversions.
    ///
    /// Concretely, this computes
    /// ```text
    ///   dst[i, j, k, l] += mul_mont(rounddiv(lhs[i, 0, k, l], 2^shift), rhs[i, j, 0, 0])
    /// ```
    /// Here `shift` is at least 64 and `mul_mont` is classic montgomery multiplication w.r.t.
    /// `mods[j]`, as specified by [`Backend::mul_mont64()`]. `rhs` has shape `[s_0, s_1, 1, 1]` and
    /// may be broadcast in the 0-th dimension if `s_0 = 1`. `rhs[i, j, 0, 0]` should be
    /// bounded by `N`, in which case the bound condition of [`Backend::mul_mont64()`] is
    /// satisfied.
    fn hal_bconv_correct_mont64(
        &self,
        dst: TensorI64Mut<Self>,
        lhs: TensorI128Ref<Self>,
        rhs: TensorI64Ref<Self>,
        mods: &Moduli<Self>,
        shift: usize,
    );

    /// Computes the correction and reduction step of a base conversion to a power-of-two
    /// modulus `2^log2_modulus`.
    ///
    /// Concretely, this computes
    /// ```text
    ///   dst[i, 0, k, l] = cmod((src[i, 0, k, l] - rounddiv(src[i, 1, k, l], 2^shift) * factor0) * factor1, 2^log2_modulus)
    /// ```
    /// where `cmod` reduces into `[-2^(log2_modulus - 1), 2^(log2_modulus - 1)]`.
    ///
    /// `src` should have shape `[s_0, 2, s_2, s_3]`, where `src[i, 0, j, k]` stores the modular
    /// result, and `src[i, 1, j, k]` stores the correction term.
    ///
    /// Requires `log2_modulus <= 62` (so the result fits into an `i64`) and `shift >= 1`. Since the
    /// modulus divides `2^128`, all products may be computed with wrapping `i128` arithmetic;
    /// only the rounded division must be exact, i.e. `src[i, 1, k, l]` must not overflow.
    ///
    /// For larger moduli (up to `2^126`), use [`Backend::bconv_correct_pow2_wide()`].
    fn hal_bconv_correct_pow2(
        &self,
        dst: TensorI64Mut<Self>,
        src: TensorI128Ref<Self>,
        factor0: i64,
        factor1: i64,
        shift: usize,
        log2_modulus: usize,
    );

    /// Computes the correction and reduction step of a base conversion to a power-of-two
    /// modulus `2^log2_modulus`. This is the 128-bit analogue of [`Backend::bconv_correct_pow2()`].
    ///
    /// The variant of [`Backend::bconv_correct_pow2()`] for moduli `2^log2_modulus` of up to
    /// 126 bits, whose residues are represented as two 64-bit limbs per [`split_i128_limbs()`].
    ///
    /// This computes
    /// ```text
    ///   dst_i128[i, 0, k, l] = cmod((src_i128[i, k, l] - rounddiv(src[i, 2, k, l], 2^shift) * factor0) * factor1, 2^log2_modulus)
    /// ```
    /// where `cmod` reduces into `[-2^(log2_modulus - 1), 2^(log2_modulus - 1)]`.
    ///
    /// `src` should have shape `[s_0, 3, s_2, s_3]`, where `src[i, 0, k, l]` and `src[i, 1, k, l]`
    /// such that `src_i128[i, k, l] = src[i, 0, k, l] + 2^64 src[i, 1, k, l]`. Note that `src[i, 0,
    /// k, l]` and `src[i, 1, k, l]` may be 128-bit values themselves. Moreover, `src[i, 2, k,
    /// l]` then stores the correction term.
    ///
    /// `dst` should have shape `[s_0, 2, s_2, s_3]`, where `dst[i, 0, k, l]` and `dst[i, 1, k, l]`
    /// store the two 64-bit limbs of the 128-bit value `dst_i128[i, k, l]`.
    ///
    /// Requires `log2_modulus <= 126` (so the result fits into an `i128`) and `shift >= 1`. Since
    /// the modulus divides `2^128`, all products may be computed with wrapping `i128`
    /// arithmetic; only the rounded division must be exact, i.e. `src[i, 1, k, l]` must not
    /// overflow.
    ///
    /// # Why not a 128-bit destination?
    ///
    /// Because `dst` represents the dBFV overflow, and will later be used as argument of
    /// [`Backend::fma_i64()`]. Similarly, `src_i128` should be the output of a 128 x 64 -> 192 bit
    /// matmul, which we also don't want to support in the HAL.
    fn hal_bconv_correct_pow2_wide(
        &self,
        dst: TensorI64Mut<Self>,
        src: TensorI128Ref<Self>,
        factor0: i128,
        factor1: i128,
        shift: usize,
        log2_modulus: usize,
    );

    /// Creates the permutation that interprets a slice as a shape-`inner_shape` tensor, and
    /// performs a cyclic rotation of `shifts` steps along each dimension.
    fn hal_create_permutation(&self, inner_shape: &[usize], shifts: &[usize]) -> Self::PermutationPlan;

    fn hal_permute(
        &self,
        dst: TensorI64Mut<Self>,
        src: TensorI64Ref<Self>,
        permutation: &Self::PermutationPlan,
        dim: DataDim,
    );
}
