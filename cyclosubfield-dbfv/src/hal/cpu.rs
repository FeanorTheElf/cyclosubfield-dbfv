//! A simple, single-threaded reference implementation of [`Backend`] on the CPU.
//!
//! Tensors are stored as contiguous, row-major `Vec`s; views (obtained via slicing or splitting)
//! are strided views into such a buffer. Since slicing never changes strides, the stride of
//! dimension 3 is always 1, so every line `tensor[i, j, k, *]` is a contiguous block; all inner
//! loops of the operations below run over such lines.
//!
//! Views store a raw pointer to the element at index `[0, 0, 0, 0]` instead of a slice, since
//! [`TensorMut::split()`] must produce two mutable views whose underlying memory ranges may
//! interleave (even though the sets of addressed elements are disjoint). Mutable access is only
//! ever materialized as a reference to a single line, i.e. to elements addressed by the view
//! itself; two views can only address a common element if one was derived from the other, in
//! which case the borrow checker prevents concurrent use.
//!
//! All modular arithmetic is implemented on top of the montgomery reduction routines of
//! [`MontgomeryModulus`], with `R = 2^64`. NTTs are implemented as lazy radix-2/radix-3
//! Cooley-Tukey transforms whose twiddle multiplications are Barrett-style multiplications
//! by plain constants; see [`super::ntt`].

use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::slice::{from_raw_parts, from_raw_parts_mut};

use memmap2::MmapMut;
use tracing::instrument;

use super::mont::MontgomeryModulus;
use super::{
    Backend, DIMS, DataDim, Dim, Moduli, TensorI64Mut, TensorI64Ref, TensorI128Mut, TensorI128Ref, TensorMut,
    TensorOwned, TensorRef,
};
use crate::hal::split_i128_limbs;
use crate::keystore::KeyStore;

/// Tensors

fn strides_for(shape: [usize; DIMS]) -> [usize; DIMS] {
    return [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1];
}

fn dim_index(dim: Dim) -> usize {
    return match dim {
        Dim::Dim0 => 0,
        Dim::Dim1 => 1,
        Dim::Dim2 => 2,
        Dim::Dim3 => 3,
    };
}

fn resolve_range<B: RangeBounds<usize>>(bounds: B, len: usize) -> (usize, usize) {
    let start = match bounds.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match bounds.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(start <= end && end <= len, "slice bounds {}..{} invalid for dimension of length {}", start, end, len);
    return (start, end);
}

fn slice_view<B0: RangeBounds<usize>, B1: RangeBounds<usize>, B2: RangeBounds<usize>, B3: RangeBounds<usize>>(
    shape: [usize; DIMS],
    strides: [usize; DIMS],
    idx0: B0,
    idx1: B1,
    idx2: B2,
    idx3: B3,
) -> (usize, [usize; DIMS]) {
    let (s0, e0) = resolve_range(idx0, shape[0]);
    let (s1, e1) = resolve_range(idx1, shape[1]);
    let (s2, e2) = resolve_range(idx2, shape[2]);
    let (s3, e3) = resolve_range(idx3, shape[3]);
    let offset = s0 * strides[0] + s1 * strides[1] + s2 * strides[2] + s3 * strides[3];
    return (offset, [e0 - s0, e1 - s1, e2 - s2, e3 - s3]);
}

/// Where the elements of a [`CpuTensor`] live: either on the heap, or in a file mapped into
/// the address space. The mapped variant exists so that objects which are large, read-mostly
/// and streamed in a predictable order — Galois keys during a linear transform, see
/// [`crate::keystore`] — can be held on an SSD instead of in RAM.
enum CpuTensorData<T: Copy> {
    Heap(Vec<T>),
    /// the mapping covers at least `len` elements starting at its base address, which is page
    /// aligned and hence aligned for `T`
    Mapped {
        map: MmapMut,
        len: usize,
    },
}

impl<T: Copy> CpuTensorData<T> {
    fn len(&self) -> usize {
        return match self {
            Self::Heap(data) => data.len(),
            Self::Mapped { len, .. } => *len,
        };
    }

    fn as_slice(&self) -> &[T] {
        return match self {
            Self::Heap(data) => data,
            // the mapping is at least `len` elements long and page aligned; the elements were
            // written as a `[T]` and `T: Copy` has no invalid bit patterns that we introduce
            Self::Mapped { map, len } => unsafe { from_raw_parts(map.as_ptr() as *const T, *len) },
        };
    }

    fn as_ptr(&self) -> *const T {
        return match self {
            Self::Heap(data) => data.as_ptr(),
            Self::Mapped { map, .. } => map.as_ptr() as *const T,
        };
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        return match self {
            Self::Heap(data) => data.as_mut_ptr(),
            Self::Mapped { map, .. } => map.as_mut_ptr() as *mut T,
        };
    }
}

impl<T: Copy> Debug for CpuTensorData<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Heap(data) => write!(f, "Heap({} elements)", data.len()),
            Self::Mapped { len, .. } => write!(f, "Mapped({} elements)", len),
        };
    }
}

/// An owned, contiguous, row-major tensor.
#[derive(Debug)]
pub struct CpuTensor<T: Copy> {
    data: CpuTensorData<T>,
    shape: [usize; DIMS],
}

/// Cloning always produces a heap-backed tensor; a mapped tensor is a cache of data that lives
/// in a file, and duplicating the mapping would not duplicate the storage.
impl<T: Copy> Clone for CpuTensor<T> {
    fn clone(&self) -> Self {
        return Self { data: CpuTensorData::Heap(self.data.as_slice().to_vec()), shape: self.shape };
    }
}

impl<T: Copy> CpuTensor<T> {
    pub fn new(data: Vec<T>, shape: [usize; DIMS]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        return Self { data: CpuTensorData::Heap(data), shape };
    }

    pub fn zeros(shape: [usize; DIMS]) -> Self
    where
        T: Default,
    {
        return Self { data: CpuTensorData::Heap(vec![T::default(); shape.iter().product()]), shape };
    }

    /// The elements in row-major order.
    pub fn data(&self) -> &[T] { return self.data.as_slice(); }

    /// Whether the elements are held in a mapped file rather than on the heap.
    pub fn is_mapped(&self) -> bool { return matches!(self.data, CpuTensorData::Mapped { .. }); }

    /// Moves the elements into `store`, replacing the heap allocation by a mapping of the
    /// store's backing file; see [`crate::keystore::KeyStore`]. Returns the tensor unchanged if
    /// it is already mapped.
    ///
    /// The elements keep their values and stay accessible exactly as before — only the pages
    /// backing them become evictable without consuming swap, so the kernel can drop them under
    /// memory pressure and fault them back in from the file.
    pub fn offload(self, store: &KeyStore) -> std::io::Result<Self> {
        if self.is_mapped() {
            return Ok(self);
        }
        let map = store.store(self.data.as_slice())?;
        return Ok(Self { data: CpuTensorData::Mapped { map, len: self.data.len() }, shape: self.shape });
    }
}

/// An immutable, strided view into a [`CpuTensor`]. `ptr` points to the element at index
/// `[0, 0, 0, 0]`; the stride of dimension 3 is always 1, so lines `self[i, j, k, *]` are
/// contiguous.
#[derive(Debug, Clone, Copy)]
pub struct CpuTensorRef<'a, T: Copy> {
    ptr: *const T,
    shape: [usize; DIMS],
    strides: [usize; DIMS],
    tensor: PhantomData<&'a [T]>,
}

/// The mutable version of [`CpuTensorRef`].
#[derive(Debug)]
pub struct CpuTensorMut<'a, T: Copy> {
    ptr: *mut T,
    shape: [usize; DIMS],
    strides: [usize; DIMS],
    tensor: PhantomData<&'a mut [T]>,
}

impl<'a, T: Copy> CpuTensorRef<'a, T> {
    /// The contiguous last-dimension line `self[i, j, k, *]`.
    #[inline]
    fn line(self, i: usize, j: usize, k: usize) -> &'a [T] {
        debug_assert!(i < self.shape[0] && j < self.shape[1] && k < self.shape[2]);
        debug_assert_eq!(self.strides[3], 1);
        let offset = i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
        return unsafe { from_raw_parts(self.ptr.add(offset), self.shape[3]) };
    }

    /// The element `self[idx]`.
    #[inline]
    fn at(self, idx: [usize; DIMS]) -> T {
        debug_assert!((0..DIMS).all(|d| idx[d] < self.shape[d]));
        let offset: usize = (0..DIMS).map(|d| idx[d] * self.strides[d]).sum();
        return unsafe { *self.ptr.add(offset) };
    }
}

impl<'a, T: Copy> CpuTensorMut<'a, T> {
    /// The contiguous last-dimension line `self[i, j, k, *]`.
    #[inline]
    fn line(&mut self, i: usize, j: usize, k: usize) -> &mut [T] {
        debug_assert!(i < self.shape[0] && j < self.shape[1] && k < self.shape[2]);
        debug_assert_eq!(self.strides[3], 1);
        let offset = i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
        return unsafe { from_raw_parts_mut(self.ptr.add(offset), self.shape[3]) };
    }
}

impl<'a, T: Copy> TensorRef for CpuTensorRef<'a, T> {
    fn shape(self) -> [usize; DIMS] { return self.shape; }

    fn slice<B0: RangeBounds<usize>, B1: RangeBounds<usize>, B2: RangeBounds<usize>, B3: RangeBounds<usize>>(
        self,
        idx0: B0,
        idx1: B1,
        idx2: B2,
        idx3: B3,
    ) -> Self {
        let (offset, shape) = slice_view(self.shape, self.strides, idx0, idx1, idx2, idx3);
        return Self { ptr: unsafe { self.ptr.add(offset) }, shape, strides: self.strides, tensor: PhantomData };
    }
}

impl<'b, T: Copy + 'b> TensorMut<'b, T, CpuTensor<T>> for CpuTensorMut<'b, T> {
    fn shape(&self) -> [usize; DIMS] { return self.shape; }

    fn reborrow<'a>(&'a mut self) -> CpuTensorMut<'a, T> {
        return CpuTensorMut { ptr: self.ptr, shape: self.shape, strides: self.strides, tensor: PhantomData };
    }

    fn as_const<'a>(&'a self) -> CpuTensorRef<'a, T> {
        return CpuTensorRef { ptr: self.ptr, shape: self.shape, strides: self.strides, tensor: PhantomData };
    }

    fn into_const(self) -> CpuTensorRef<'b, T> {
        return CpuTensorRef { ptr: self.ptr, shape: self.shape, strides: self.strides, tensor: PhantomData };
    }

    fn split(self, at: usize, dim: Dim) -> (Self, Self) {
        let d = dim_index(dim);
        assert!(at <= self.shape[d], "cannot split dimension of length {} at {}", self.shape[d], at);
        let mut first_shape = self.shape;
        first_shape[d] = at;
        let mut second_shape = self.shape;
        second_shape[d] = self.shape[d] - at;
        // the two views address disjoint sets of elements, even though their memory ranges may
        // interleave
        let first = Self { ptr: self.ptr, shape: first_shape, strides: self.strides, tensor: PhantomData };
        let second = Self {
            ptr: unsafe { self.ptr.add(at * self.strides[d]) },
            shape: second_shape,
            strides: self.strides,
            tensor: PhantomData,
        };
        return (first, second);
    }

    fn slice<B0: RangeBounds<usize>, B1: RangeBounds<usize>, B2: RangeBounds<usize>, B3: RangeBounds<usize>>(
        self,
        idx0: B0,
        idx1: B1,
        idx2: B2,
        idx3: B3,
    ) -> Self {
        let (offset, shape) = slice_view(self.shape, self.strides, idx0, idx1, idx2, idx3);
        return Self { ptr: unsafe { self.ptr.add(offset) }, shape, strides: self.strides, tensor: PhantomData };
    }
}

impl<T: Copy> TensorOwned<T> for CpuTensor<T> {
    type Ref<'a>
        = CpuTensorRef<'a, T>
    where
        Self: 'a;
    type Mut<'a>
        = CpuTensorMut<'a, T>
    where
        Self: 'a;

    fn shape(self) -> [usize; DIMS] { return self.shape; }

    fn as_ref<'a>(&'a self) -> Self::Ref<'a> {
        return CpuTensorRef {
            ptr: CpuTensorData::as_ptr(&self.data),
            shape: self.shape,
            strides: strides_for(self.shape),
            tensor: PhantomData,
        };
    }

    fn as_mut<'a>(&'a mut self) -> Self::Mut<'a> {
        return CpuTensorMut {
            ptr: CpuTensorData::as_mut_ptr(&mut self.data),
            shape: self.shape,
            strides: strides_for(self.shape),
            tensor: PhantomData,
        };
    }
}

/// Pointwise iteration helpers

/// Computes the effective strides of a broadcasted source: dimensions of length 1 that are
/// expanded to the destination length get stride 0.
fn broadcast_strides(src_shape: [usize; DIMS], src_strides: [usize; DIMS], dst_shape: [usize; DIMS]) -> [usize; DIMS] {
    let mut strides = src_strides;
    for d in 0..DIMS {
        if src_shape[d] != dst_shape[d] {
            assert_eq!(
                src_shape[d], 1,
                "cannot broadcast dimension {} of length {} to length {}",
                d, src_shape[d], dst_shape[d]
            );
            strides[d] = 0;
        }
    }
    return strides;
}

/// Calls `f(j, dst_line)` for every last-dimension line of `dst`.
fn for_each_line1<T: Copy>(dst: &mut CpuTensorMut<T>, mut f: impl FnMut(usize, &mut [T])) {
    let [d0, d1, d2, d3] = dst.shape;
    let ds = dst.strides;
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let doff = i * ds[0] + j * ds[1] + k * ds[2];
                f(j, unsafe { from_raw_parts_mut(dst.ptr.add(doff), d3) });
            }
        }
    }
}

/// Calls `f(j, dst_line, src_line, src_broadcasts_dim3)` for every last-dimension line of `dst`,
/// with `src` broadcast to the shape of `dst`.
fn for_each_line2<T: Copy, S: Copy>(
    dst: &mut CpuTensorMut<T>,
    src: &CpuTensorRef<S>,
    mut f: impl FnMut(usize, &mut [T], &[S], bool),
) {
    let [d0, d1, d2, d3] = dst.shape;
    let ds = dst.strides;
    let ss = broadcast_strides(src.shape, src.strides, dst.shape);
    let sb = src.shape[3] != d3;
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let doff = i * ds[0] + j * ds[1] + k * ds[2];
                let soff = i * ss[0] + j * ss[1] + k * ss[2];
                f(
                    j,
                    unsafe { from_raw_parts_mut(dst.ptr.add(doff), d3) },
                    unsafe { from_raw_parts(src.ptr.add(soff), if sb { 1 } else { d3 }) },
                    sb,
                );
            }
        }
    }
}

/// The two-source version of [`for_each_line2()`].
fn for_each_line3<T: Copy, A: Copy, B: Copy>(
    dst: &mut CpuTensorMut<T>,
    lhs: &CpuTensorRef<A>,
    rhs: &CpuTensorRef<B>,
    mut f: impl FnMut(usize, &mut [T], &[A], bool, &[B], bool),
) {
    let [d0, d1, d2, d3] = dst.shape;
    let ds = dst.strides;
    let ls = broadcast_strides(lhs.shape, lhs.strides, dst.shape);
    let rs = broadcast_strides(rhs.shape, rhs.strides, dst.shape);
    let lb = lhs.shape[3] != d3;
    let rb = rhs.shape[3] != d3;
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let doff = i * ds[0] + j * ds[1] + k * ds[2];
                let loff = i * ls[0] + j * ls[1] + k * ls[2];
                let roff = i * rs[0] + j * rs[1] + k * rs[2];
                f(
                    j,
                    unsafe { from_raw_parts_mut(dst.ptr.add(doff), d3) },
                    unsafe { from_raw_parts(lhs.ptr.add(loff), if lb { 1 } else { d3 }) },
                    lb,
                    unsafe { from_raw_parts(rhs.ptr.add(roff), if rb { 1 } else { d3 }) },
                    rb,
                );
            }
        }
    }
}

/// Applies `f` pointwise over a destination line and a (possibly length-1 broadcasted) source
/// line. The broadcast case is dispatched outside of the inner loop, so both loop bodies are free
/// of stride computations and bounds checks.
#[inline(always)]
fn bin_line<T: Copy, S: Copy>(dst: &mut [T], src: &[S], src_bcast: bool, mut f: impl FnMut(&mut T, S)) {
    if src_bcast {
        let s = src[0];
        for d in dst.iter_mut() {
            f(d, s);
        }
    } else {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            f(d, *s);
        }
    }
}

/// The two-source version of [`bin_line()`].
#[inline(always)]
fn tern_line<T: Copy, A: Copy, B: Copy>(
    dst: &mut [T],
    lhs: &[A],
    lhs_bcast: bool,
    rhs: &[B],
    rhs_bcast: bool,
    mut f: impl FnMut(&mut T, A, B),
) {
    match (lhs_bcast, rhs_bcast) {
        (false, false) => {
            for ((d, l), r) in dst.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
                f(d, *l, *r);
            }
        }
        (false, true) => {
            let r = rhs[0];
            for (d, l) in dst.iter_mut().zip(lhs.iter()) {
                f(d, *l, r);
            }
        }
        (true, false) => {
            let l = lhs[0];
            for (d, r) in dst.iter_mut().zip(rhs.iter()) {
                f(d, l, *r);
            }
        }
        (true, true) => {
            let l = lhs[0];
            let r = rhs[0];
            for d in dst.iter_mut() {
                f(d, l, r);
            }
        }
    }
}

/// NTT plans are implemented in [`super::ntt`] (lazy Barrett-style transforms);
/// [`CpuNTTPlan`] is re-exported from there.
pub use super::ntt::CpuNTTPlan;

#[derive(Debug, Clone)]
pub struct CpuPermutationPlan {
    len: usize,
    /// `dst[e] = src[perm[e]]`
    perm: Vec<u32>,
}

/// The backend

/// A simple, single-threaded, pure-CPU implementation of [`Backend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    fn ntt_dim_impl(
        &self,
        mut dst: TensorI64Mut<Self>,
        ntt: &[CpuNTTPlan],
        dim: DataDim,
        inverse: bool,
        reduce_output: bool,
    ) {
        let [d0, d1, d2, d3] = dst.shape;
        assert_eq!(ntt.len(), d1);
        let strides = dst.strides;
        debug_assert_eq!(1, strides[3]);
        match dim {
            DataDim::Dim2 => {
                for plan in ntt {
                    assert_eq!(plan.len(), d2);
                }
                for i in 0..d0 {
                    for j in 0..d1 {
                        let data = unsafe { dst.ptr.add(i * strides[0] + j * strides[1]) };
                        unsafe { ntt[j].apply(data, strides[2], d3, inverse, reduce_output) };
                    }
                }
            }
            DataDim::Dim3 => {
                for plan in ntt {
                    assert_eq!(plan.len(), d3);
                }
                for i in 0..d0 {
                    for j in 0..d1 {
                        for k in 0..d2 {
                            let data = unsafe { dst.ptr.add(i * strides[0] + j * strides[1] + k * strides[2]) };
                            unsafe { ntt[j].apply(data, 1, 1, inverse, reduce_output) };
                        }
                    }
                }
            }
        }
        // silence unused warnings when dst is fully consumed by raw-pointer access
        let _ = dst.reborrow();
    }

    fn mul_mont64_impl(&self, mut dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| {
                    *d = modulus.red128((*d as i128) * (s as i128));
                },
            );
        });
    }
}

impl Backend for CpuBackend {
    type TensorI64 = CpuTensor<i64>;
    type TensorI128 = CpuTensor<i128>;
    type Modulus = MontgomeryModulus;
    type NTTPlan = CpuNTTPlan;
    type PermutationPlan = CpuPermutationPlan;

    #[instrument(skip_all)]
    fn hal_allocate_i64(&self, shape: [usize; DIMS]) -> CpuTensor<i64> { return CpuTensor::zeros(shape); }

    #[instrument(skip_all)]
    fn hal_allocate_i128(&self, shape: [usize; DIMS]) -> CpuTensor<i128> { return CpuTensor::zeros(shape); }

    #[instrument(skip_all)]
    fn hal_import(&self, mut dst: TensorI64Mut<Self>, src: &[i64]) {
        assert_eq!(src.len(), dst.shape.iter().product::<usize>());
        let mut pos = 0;
        for_each_line1(&mut dst, |_, line| {
            line.copy_from_slice(&src[pos..pos + line.len()]);
            pos += line.len();
        });
    }

    #[instrument(skip_all)]
    fn hal_export(&self, src: TensorI64Ref<Self>, dst: &mut [i64]) {
        assert_eq!(dst.len(), src.shape.iter().product::<usize>());
        let [d0, d1, d2, d3] = src.shape;
        let mut pos = 0;
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    dst[pos..pos + d3].copy_from_slice(src.line(i, j, k));
                    pos += d3;
                }
            }
        }
    }

    #[instrument(skip_all)]
    fn hal_assign(&self, mut dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = s,
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_assign_narrow(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| {
                    debug_assert!(i64::try_from(s).is_ok(), "assign_narrow: entry doesn't fit into an i64");
                    *d = s as i64;
                },
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_assign_i128(&self, mut dst: TensorI128Mut<Self>, src: TensorI128Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = s,
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_add_i64(&self, mut dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = d.wrapping_add(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_add_i128(&self, mut dst: TensorI128Mut<Self>, src: TensorI128Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = d.wrapping_add(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_sub_i64(&self, mut dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>) {
        for_each_line2(&mut dst, &src, |_, dl, sl, sb| {
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = d.wrapping_sub(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_fma_i64(&self, mut dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>) {
        for_each_line3(&mut dst, &lhs, &rhs, |_, dl, ll, lb, rl, rb| {
            tern_line(
                dl,
                ll,
                lb,
                rl,
                rb,
                #[inline(always)]
                |d, l, r| {
                    *d = d.wrapping_add((l as i128) * (r as i128));
                },
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_mul_i64(&self, mut dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>) {
        for_each_line3(&mut dst, &lhs, &rhs, |_, dl, ll, lb, rl, rb| {
            tern_line(
                dl,
                ll,
                lb,
                rl,
                rb,
                #[inline(always)]
                |d, l, r| {
                    *d = (l as i128) * (r as i128);
                },
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_inner_product_i64(
        &self,
        mut dst: TensorI128Mut<Self>,
        lhs: &[TensorI64Ref<Self>],
        rhs: &[TensorI64Ref<Self>],
    ) {
        let n = lhs.len();
        assert_eq!(n, rhs.len());
        assert!(n >= 1, "inner_product_i64 requires at least one term");
        let dst_shape = dst.shape;
        let ds = dst.strides;
        debug_assert_eq!(ds[3], 1);
        // Precompute, per term, the broadcast strides into the destination shape, the
        // dimension-3 broadcast flag and the base pointer; these are loop-invariant.
        let terms: Vec<(([usize; DIMS], bool, *const i64), ([usize; DIMS], bool, *const i64))> = (0..n)
            .map(|t| {
                let l = &lhs[t];
                let r = &rhs[t];
                (
                    (broadcast_strides(l.shape, l.strides, dst_shape), l.shape[3] != dst_shape[3], l.ptr),
                    (broadcast_strides(r.shape, r.strides, dst_shape), r.shape[3] != dst_shape[3], r.ptr),
                )
            })
            .collect();
        // Number of `i128` accumulators kept resident while summing all terms. Chosen so a tile
        // comfortably fits into L1 (256 * 16 B = 4 KiB); the whole tile is loaded once, all
        // terms are accumulated into it, and it is written back once.
        const TILE: usize = 256;
        let [d0, d1, d2, d3] = dst_shape;
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    let doff = i * ds[0] + j * ds[1] + k * ds[2];
                    let dst_line = unsafe { from_raw_parts_mut(dst.ptr.add(doff), d3) };
                    let mut base = 0;
                    while base < d3 {
                        let end = usize::min(base + TILE, d3);
                        let dst_tile = &mut dst_line[base..end];
                        for (t, ((ls, lb, lp), (rs, rb, rp))) in terms.iter().enumerate() {
                            let loff = i * ls[0] + j * ls[1] + k * ls[2];
                            let roff = i * rs[0] + j * rs[1] + k * rs[2];
                            let lhs_tile = unsafe {
                                from_raw_parts(
                                    lp.add(loff + if *lb { 0 } else { base }),
                                    if *lb { 1 } else { dst_tile.len() },
                                )
                            };
                            let rhs_tile = unsafe {
                                from_raw_parts(
                                    rp.add(roff + if *rb { 0 } else { base }),
                                    if *rb { 1 } else { dst_tile.len() },
                                )
                            };
                            if t == 0 {
                                tern_line(
                                    dst_tile,
                                    lhs_tile,
                                    *lb,
                                    rhs_tile,
                                    *rb,
                                    #[inline(always)]
                                    |d, l, r| *d = (l as i128) * (r as i128),
                                );
                            } else {
                                tern_line(
                                    dst_tile,
                                    lhs_tile,
                                    *lb,
                                    rhs_tile,
                                    *rb,
                                    #[inline(always)]
                                    |d, l, r| *d = d.wrapping_add((l as i128) * (r as i128)),
                                );
                            }
                        }
                        base = end;
                    }
                }
            }
        }
        let _ = dst.reborrow();
    }

    #[instrument(skip_all)]
    fn hal_matmul_i64(&self, mut dst: TensorI128Mut<Self>, lhs: TensorI64Ref<Self>, rhs: TensorI64Ref<Self>) {
        let [i_len, j_len, k_len, l_len] = dst.shape;
        let u_len = lhs.shape[2];
        assert!(lhs.shape[0] == i_len || lhs.shape[0] == 1);
        assert_eq!(lhs.shape[1], j_len);
        assert_eq!(lhs.shape[3], 1);
        assert!(rhs.shape[0] == i_len || rhs.shape[0] == 1);
        assert_eq!(rhs.shape[1..], [u_len, k_len, l_len]);
        for i in 0..i_len {
            let li = if lhs.shape[0] == 1 { 0 } else { i };
            let ri = if rhs.shape[0] == 1 { 0 } else { i };
            for j in 0..j_len {
                for k in 0..k_len {
                    let dst_line = dst.line(i, j, k);
                    dst_line.fill(0);
                    for u in 0..u_len {
                        let c = lhs.at([li, j, u, 0]) as i128;
                        if c == 0 {
                            continue;
                        }
                        for (d, r) in dst_line.iter_mut().zip(rhs.line(ri, u, k).iter()) {
                            *d = d.wrapping_add(c * (*r as i128));
                        }
                    }
                }
            }
        }
    }

    #[instrument(skip_all)]
    fn hal_create_modulus(&self, prime: i64) -> Self::Modulus { return MontgomeryModulus::new(prime); }

    #[instrument(skip_all)]
    fn hal_mul_mont64(&self, dst: TensorI64Mut<Self>, src: TensorI64Ref<Self>, mods: &Moduli<Self>) {
        self.mul_mont64_impl(dst, src, mods);
    }

    #[instrument(skip_all)]
    fn hal_add_red_mont64_double(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| {
                    *d = d.wrapping_add(modulus.double_red128(s));
                },
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_red_mont64_double(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = modulus.double_red128(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_add_red_mont64(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| {
                    *d = d.wrapping_add(modulus.red128(s));
                },
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_red_mont64(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = modulus.red128(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_i128_to_mont64(&self, mut dst: TensorI64Mut<Self>, src: TensorI128Ref<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line2(&mut dst, &src, |j, dl, sl, sb| {
            let modulus = &mods[j];
            bin_line(
                dl,
                sl,
                sb,
                #[inline(always)]
                |d, s| *d = modulus.to_mont128(s),
            );
        });
    }

    #[instrument(skip_all)]
    fn hal_i64_to_mont64(&self, mut dst: TensorI64Mut<Self>, mods: &Moduli<Self>) {
        assert_eq!(mods.len(), dst.shape[1]);
        for_each_line1(&mut dst, |j, line| {
            let modulus = &mods[j];
            for d in line {
                *d = modulus.to_mont64(*d);
            }
        });
    }

    #[instrument(skip_all)]
    fn hal_create_ntt_plan(&self, modulus: &Self::Modulus, inner_shape: &[usize]) -> Self::NTTPlan {
        return CpuNTTPlan::new(modulus.modulus(), inner_shape);
    }

    #[instrument(skip_all)]
    fn hal_ntt_mont64_dim(&self, dst: TensorI64Mut<Self>, ntt: &[Self::NTTPlan], dim: DataDim) {
        self.ntt_dim_impl(dst, ntt, dim, false, true);
    }

    #[instrument(skip_all)]
    fn hal_intt_mont64_dim(&self, dst: TensorI64Mut<Self>, ntt: &[Self::NTTPlan], dim: DataDim) {
        self.ntt_dim_impl(dst, ntt, dim, true, true);
    }

    #[instrument(skip_all)]
    fn hal_ntt_mul_mont64_dim(
        &self,
        mut dst: TensorI64Mut<Self>,
        ntt: &[Self::NTTPlan],
        dim: DataDim,
        mul: TensorI64Ref<Self>,
        mods: &Moduli<Self>,
    ) {
        // the forward transform leaves its output unreduced (bounded by i64::MAX), and the
        // montgomery multiplication both applies `mul` and reduces into [-N, N]; the unreduced
        // output times a montgomery constant in [-N, N] stays within the `2^63 N` bound of the
        // montgomery reduction
        self.ntt_dim_impl(dst.reborrow(), ntt, dim, false, false);
        self.mul_mont64_impl(dst, mul, mods);
    }

    #[instrument(skip_all)]
    fn hal_bconv_correct_mont64(
        &self,
        mut dst: TensorI64Mut<Self>,
        lhs: TensorI128Ref<Self>,
        rhs: TensorI64Ref<Self>,
        mods: &Moduli<Self>,
        shift: usize,
    ) {
        let [i_len, j_len, k_len, l_len] = dst.shape;
        assert!(shift >= 64);
        assert_eq!(lhs.shape, [i_len, 1, k_len, l_len]);
        assert!(rhs.shape[0] == i_len || rhs.shape[0] == 1);
        assert_eq!(rhs.shape[1..], [j_len, 1, 1]);
        assert_eq!(mods.len(), j_len);
        let half: i128 = 1 << (shift - 1);
        for i in 0..i_len {
            let ri = if rhs.shape[0] == 1 { 0 } else { i };
            for j in 0..j_len {
                let modulus = &mods[j];
                let r = rhs.at([ri, j, 0, 0]) as i128;
                for k in 0..k_len {
                    let dst_line = dst.line(i, j, k);
                    let lhs_line = lhs.line(i, 0, k);
                    for (d, l) in dst_line.iter_mut().zip(lhs_line.iter()) {
                        *d = d.wrapping_add(modulus.red128(((*l + half) >> shift) * r));
                    }
                }
            }
        }
    }

    #[instrument(skip_all)]
    fn hal_bconv_correct_pow2(
        &self,
        mut dst: TensorI64Mut<Self>,
        src: TensorI128Ref<Self>,
        factor0: i64,
        factor1: i64,
        shift: usize,
        log2_modulus: usize,
    ) {
        let [i_len, j_len, k_len, l_len] = dst.shape;
        assert_eq!(1, j_len);
        assert_eq!(src.shape, [i_len, 2, k_len, l_len]);
        assert!(shift >= 1 && shift < 127);
        assert!(log2_modulus >= 1 && log2_modulus <= 62);
        let half: i128 = 1 << (shift - 1);
        let modulus: i128 = 1 << log2_modulus;
        let modulus_half = modulus / 2;
        for i in 0..i_len {
            for k in 0..k_len {
                let dst_line = dst.line(i, 0, k);
                let value_line = src.line(i, 0, k);
                let correction_line = src.line(i, 1, k);
                for ((d, v), c) in dst_line.iter_mut().zip(value_line.iter()).zip(correction_line.iter()) {
                    let quot = (*c + half) >> shift;
                    let reduced = v
                        .wrapping_sub(quot.wrapping_mul(factor0 as i128))
                        .wrapping_mul(factor1 as i128)
                        .rem_euclid(modulus);
                    *d = (if reduced > modulus_half { reduced - modulus } else { reduced }) as i64;
                }
            }
        }
    }

    #[instrument(skip_all)]
    fn hal_bconv_correct_pow2_wide(
        &self,
        mut dst: TensorI64Mut<Self>,
        src: TensorI128Ref<Self>,
        factor0: i128,
        factor1: i128,
        shift: usize,
        log2_modulus: usize,
    ) {
        let [i_len, j_len, k_len, l_len] = dst.shape;
        assert_eq!(2, j_len);
        assert_eq!(src.shape, [i_len, 3, k_len, l_len]);
        assert!(shift >= 1 && shift < 127);
        assert!(log2_modulus >= 1 && log2_modulus <= 126);
        let half: i128 = 1 << (shift - 1);
        let modulus: i128 = 1 << log2_modulus;
        let modulus_half = modulus / 2;
        let mut centered = vec![0i128; l_len];
        for i in 0..i_len {
            for k in 0..k_len {
                let value_lo_line = src.line(i, 0, k);
                let value_hi_line = src.line(i, 1, k);
                let correction_line = src.line(i, 2, k);
                for (c, ((v_lo, v_hi), corr)) in
                    centered.iter_mut().zip(value_lo_line.iter().zip(value_hi_line.iter()).zip(correction_line.iter()))
                {
                    let quot = (*corr + half) >> shift;
                    let v = v_lo.wrapping_add(v_hi.wrapping_shl(64));
                    let reduced = v.wrapping_sub(quot.wrapping_mul(factor0)).wrapping_mul(factor1).rem_euclid(modulus);
                    *c = if reduced > modulus_half { reduced - modulus } else { reduced };
                }
                for (d, c) in dst.line(i, 0, k).iter_mut().zip(centered.iter()) {
                    *d = split_i128_limbs(*c).0;
                }
                for (d, c) in dst.line(i, 1, k).iter_mut().zip(centered.iter()) {
                    *d = split_i128_limbs(*c).1;
                }
            }
        }
    }

    #[instrument(skip_all)]
    fn hal_create_permutation(&self, inner_shape: &[usize], shifts: &[usize]) -> Self::PermutationPlan {
        assert_eq!(inner_shape.len(), shifts.len());
        let n: usize = inner_shape.iter().product();
        assert!(n <= u32::MAX as usize);
        let mut perm = Vec::with_capacity(n);
        let mut idx = vec![0; inner_shape.len()];
        for _ in 0..n {
            // the entry at (multi-)index `a` moves to index `a + shifts`, so the entry ending up
            // at `idx` comes from `idx - shifts`
            let mut src = 0;
            for (t, &m) in inner_shape.iter().enumerate() {
                src = src * m + (idx[t] + m - shifts[t] % m) % m;
            }
            perm.push(src as u32);
            for t in (0..inner_shape.len()).rev() {
                idx[t] += 1;
                if idx[t] < inner_shape[t] {
                    break;
                }
                idx[t] = 0;
            }
        }
        return CpuPermutationPlan { len: n, perm };
    }

    #[instrument(skip_all)]
    fn hal_permute(
        &self,
        mut dst: TensorI64Mut<Self>,
        src: TensorI64Ref<Self>,
        permutation: &Self::PermutationPlan,
        dim: DataDim,
    ) {
        assert_eq!(dst.shape, src.shape);
        let [d0, d1, d2, d3] = dst.shape;
        let perm = &permutation.perm;
        match dim {
            DataDim::Dim2 => {
                assert_eq!(permutation.len, d2);
                for i in 0..d0 {
                    for j in 0..d1 {
                        for e in 0..d2 {
                            dst.line(i, j, e).copy_from_slice(src.line(i, j, perm[e] as usize));
                        }
                    }
                }
            }
            DataDim::Dim3 => {
                assert_eq!(permutation.len, d3);
                for i in 0..d0 {
                    for j in 0..d1 {
                        for k in 0..d2 {
                            let dst_line = dst.line(i, j, k);
                            let src_line = src.line(i, j, k);
                            for (d, e) in dst_line.iter_mut().zip(perm.iter()) {
                                *d = src_line[*e as usize];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{montgomery_repr, pos_repr};

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        return *state >> 33;
    }

    fn flat_index(shape: [usize; DIMS], idx: [usize; DIMS]) -> usize {
        return ((idx[0] * shape[1] + idx[1]) * shape[2] + idx[2]) * shape[3] + idx[3];
    }

    #[test]
    fn test_slice_import_export() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 3, 4, 5];
        let mut tensor = CpuTensor::<i64>::zeros(shape);
        let data: Vec<i64> = (0..120).collect();
        backend.hal_import(tensor.as_mut(), &data);
        assert_eq!(tensor.data(), &data);

        let view = tensor.as_ref().slice(1..2, 0..=1, 2.., 1..4);
        assert_eq!(view.shape(), [1, 2, 2, 3]);
        let mut out = vec![0; 12];
        backend.hal_export(view, &mut out);
        let mut expected = Vec::new();
        for i in 1..2 {
            for j in 0..2 {
                for k in 2..4 {
                    for l in 1..4 {
                        expected.push(data[flat_index(shape, [i, j, k, l])]);
                    }
                }
            }
        }
        assert_eq!(out, expected);

        // import into a strided view only touches the view
        let mut tensor2 = tensor.clone();
        let patch: Vec<i64> = (1000..1012).collect();
        backend.hal_import(tensor2.as_mut().slice(1..2, 0..2, 2..4, 1..4), &patch);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    for l in 0..5 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        if i == 1 && j < 2 && (2..4).contains(&k) && (1..4).contains(&l) {
                            let patch_pos = ((j * 2) + (k - 2)) * 3 + (l - 1);
                            assert_eq!(tensor2.data()[pos], patch[patch_pos]);
                        } else {
                            assert_eq!(tensor2.data()[pos], data[pos]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_split() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 3, 2, 2];
        let mut tensor = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(tensor.as_mut(), &(0..24).collect::<Vec<_>>());

        let (mut first, mut second) = tensor.as_mut().split(2, Dim::Dim1);
        assert_eq!(TensorMut::shape(&first), [2, 2, 2, 2]);
        assert_eq!(TensorMut::shape(&second), [2, 1, 2, 2]);
        backend.hal_import(first.reborrow(), &[100; 16]);
        backend.hal_import(second.reborrow(), &[200; 8]);
        // reading through one view sees the writes through the other only in its own elements
        let mut out = vec![0; 16];
        backend.hal_export(first.into_const(), &mut out);
        assert_eq!(out, [100; 16]);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    for l in 0..2 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        assert_eq!(tensor.data()[pos], if j < 2 { 100 } else { 200 });
                    }
                }
            }
        }
    }

    #[test]
    fn test_add_sub_broadcast() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 2, 3, 4];
        let mut dst = CpuTensor::<i64>::zeros(shape);
        let data: Vec<i64> = (0..48).collect();
        backend.hal_import(dst.as_mut(), &data);

        let mut src_j = CpuTensor::<i64>::zeros([1, 2, 1, 1]);
        backend.hal_import(src_j.as_mut(), &[10, 20]);
        backend.hal_add_i64(dst.as_mut(), src_j.as_ref());
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        assert_eq!(dst.data()[pos], data[pos] + [10, 20][j]);
                    }
                }
            }
        }

        let mut src_l = CpuTensor::<i64>::zeros([2, 2, 3, 1]);
        let src_l_data: Vec<i64> = (0..12).map(|x| x * 100).collect();
        backend.hal_import(src_l.as_mut(), &src_l_data);
        backend.hal_sub_i64(dst.as_mut(), src_l.as_ref());
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let expected = data[pos] + [10, 20][j] - src_l_data[(i * 2 + j) * 3 + k];
                        assert_eq!(dst.data()[pos], expected);
                    }
                }
            }
        }

        // add128 with broadcasting in dimension 0
        let mut dst128 = CpuTensor::<i128>::new(vec![1; 8], [2, 1, 2, 2]);
        let src128 = CpuTensor::<i128>::new(vec![5, 6, 7, 8], [1, 1, 2, 2]);
        backend.hal_add_i128(dst128.as_mut(), src128.as_ref());
        assert_eq!(dst128.data(), &[6, 7, 8, 9, 6, 7, 8, 9]);
    }

    #[test]
    fn test_fma_and_matmul() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 2, 3, 4];
        let mut state = 42;
        let lhs_data: Vec<i64> = (0..48).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let rhs_data: Vec<i64> = (0..48).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let mut lhs = CpuTensor::<i64>::zeros(shape);
        let mut rhs = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(lhs.as_mut(), &lhs_data);
        backend.hal_import(rhs.as_mut(), &rhs_data);

        let mut acc = CpuTensor::<i128>::zeros(shape);
        backend.hal_fma_i64(acc.as_mut(), lhs.as_ref(), rhs.as_ref());
        backend.hal_fma_i64(acc.as_mut(), lhs.as_ref(), rhs.as_ref());
        for pos in 0..48 {
            assert_eq!(acc.data()[pos], 2 * (lhs_data[pos] as i128) * (rhs_data[pos] as i128));
        }

        // mul_i64 overwrites without reading the destination
        let mut prod = CpuTensor::<i128>::new(vec![-77; 48], shape);
        backend.hal_mul_i64(prod.as_mut(), lhs.as_ref(), rhs.as_ref());
        for pos in 0..48 {
            assert_eq!(prod.data()[pos], (lhs_data[pos] as i128) * (rhs_data[pos] as i128));
        }

        let (i_len, j_len, k_len, l_len, u_len) = (2, 3, 2, 4, 5);
        let dst_shape = [i_len, j_len, k_len, l_len];
        let lhs_shape = [1, j_len, u_len, 1];
        let rhs_shape = [i_len, u_len, k_len, l_len];
        let mm_lhs_data: Vec<i64> = (0..j_len * u_len).map(|_| lcg(&mut state) as i64 % 100 - 50).collect();
        let mm_rhs_data: Vec<i64> =
            (0..i_len * u_len * k_len * l_len).map(|_| lcg(&mut state) as i64 % 100 - 50).collect();
        let mut mm_lhs = CpuTensor::<i64>::zeros(lhs_shape);
        let mut mm_rhs = CpuTensor::<i64>::zeros(rhs_shape);
        backend.hal_import(mm_lhs.as_mut(), &mm_lhs_data);
        backend.hal_import(mm_rhs.as_mut(), &mm_rhs_data);
        let mut dst = CpuTensor::<i128>::new(vec![-1; 48], dst_shape);

        // matmul overwrites dst, so running it twice must give the same result
        backend.hal_matmul_i64(dst.as_mut(), mm_lhs.as_ref(), mm_rhs.as_ref());
        backend.hal_matmul_i64(dst.as_mut(), mm_lhs.as_ref(), mm_rhs.as_ref());
        for i in 0..i_len {
            for j in 0..j_len {
                for k in 0..k_len {
                    for l in 0..l_len {
                        let mut expected = 0i128;
                        for u in 0..u_len {
                            let lhs_val = mm_lhs_data[j * u_len + u] as i128;
                            let rhs_val = mm_rhs_data[flat_index(rhs_shape, [i, u, k, l])] as i128;
                            expected += lhs_val * rhs_val;
                        }
                        assert_eq!(dst.data()[flat_index(dst_shape, [i, j, k, l])], expected);
                    }
                }
            }
        }

        // matmul with rhs broadcast in dimension 0: per-row lhs, shared rhs
        let lhs_shape_rows = [i_len, j_len, u_len, 1];
        let rhs_shape_bcast = [1, u_len, k_len, l_len];
        let mm_lhs_rows_data: Vec<i64> =
            (0..i_len * j_len * u_len).map(|_| lcg(&mut state) as i64 % 100 - 50).collect();
        let mm_rhs_bcast_data: Vec<i64> =
            (0..u_len * k_len * l_len).map(|_| lcg(&mut state) as i64 % 100 - 50).collect();
        let mut mm_lhs_rows = CpuTensor::<i64>::zeros(lhs_shape_rows);
        let mut mm_rhs_bcast = CpuTensor::<i64>::zeros(rhs_shape_bcast);
        backend.hal_import(mm_lhs_rows.as_mut(), &mm_lhs_rows_data);
        backend.hal_import(mm_rhs_bcast.as_mut(), &mm_rhs_bcast_data);
        backend.hal_matmul_i64(dst.as_mut(), mm_lhs_rows.as_ref(), mm_rhs_bcast.as_ref());
        for i in 0..i_len {
            for j in 0..j_len {
                for k in 0..k_len {
                    for l in 0..l_len {
                        let mut expected = 0i128;
                        for u in 0..u_len {
                            let lhs_val = mm_lhs_rows_data[(i * j_len + j) * u_len + u] as i128;
                            let rhs_val = mm_rhs_bcast_data[flat_index(rhs_shape_bcast, [0, u, k, l])] as i128;
                            expected += lhs_val * rhs_val;
                        }
                        assert_eq!(dst.data()[flat_index(dst_shape, [i, j, k, l])], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_inner_product() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 2, 3, 4];
        let n = 5;
        let mut state = 99;
        // n term pairs, each a full-shape tensor
        let lhs_data: Vec<Vec<i64>> =
            (0..n).map(|_| (0..48).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect()).collect();
        let rhs_data: Vec<Vec<i64>> =
            (0..n).map(|_| (0..48).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect()).collect();
        let lhs: Vec<CpuTensor<i64>> = lhs_data
            .iter()
            .map(|d| {
                let mut t = CpuTensor::<i64>::zeros(shape);
                backend.hal_import(t.as_mut(), d);
                t
            })
            .collect();
        let rhs: Vec<CpuTensor<i64>> = rhs_data
            .iter()
            .map(|d| {
                let mut t = CpuTensor::<i64>::zeros(shape);
                backend.hal_import(t.as_mut(), d);
                t
            })
            .collect();
        let lhs_refs: Vec<_> = lhs.iter().map(|t| t.as_ref()).collect();
        let rhs_refs: Vec<_> = rhs.iter().map(|t| t.as_ref()).collect();

        // inner_product_i64 overwrites dst, so a stale value must be ignored, and the result
        // must equal the term-wise sum of products
        let mut dst = CpuTensor::<i128>::new(vec![-123; 48], shape);
        backend.hal_inner_product_i64(dst.as_mut(), &lhs_refs, &rhs_refs);
        for pos in 0..48 {
            let expected: i128 = (0..n).map(|t| (lhs_data[t][pos] as i128) * (rhs_data[t][pos] as i128)).sum();
            assert_eq!(dst.data()[pos], expected);
        }

        // it must be equivalent to the mul_i64 + fma_i64 chain it replaces
        let mut chain = CpuTensor::<i128>::new(vec![7; 48], shape);
        backend.hal_mul_i64(chain.as_mut(), lhs_refs[0], rhs_refs[0]);
        for t in 1..n {
            backend.hal_fma_i64(chain.as_mut(), lhs_refs[t], rhs_refs[t]);
        }
        assert_eq!(chain.data(), dst.data());

        // a single term behaves like mul_i64
        let mut single = CpuTensor::<i128>::new(vec![55; 48], shape);
        backend.hal_inner_product_i64(single.as_mut(), &lhs_refs[0..1], &rhs_refs[0..1]);
        for pos in 0..48 {
            assert_eq!(single.data()[pos], (lhs_data[0][pos] as i128) * (rhs_data[0][pos] as i128));
        }

        // broadcasting: lhs broadcast in dimension 0, rhs broadcast in dimension 3
        let lhs_b_data: Vec<i64> = (0..24).map(|_| lcg(&mut state) as i64 % 200 - 100).collect();
        let rhs_b_data: Vec<i64> = (0..12).map(|_| lcg(&mut state) as i64 % 200 - 100).collect();
        let mut lhs_b = CpuTensor::<i64>::zeros([1, 2, 3, 4]);
        let mut rhs_b = CpuTensor::<i64>::zeros([2, 2, 3, 1]);
        backend.hal_import(lhs_b.as_mut(), &lhs_b_data);
        backend.hal_import(rhs_b.as_mut(), &rhs_b_data);
        let mut bdst = CpuTensor::<i128>::new(vec![0; 48], shape);
        backend.hal_inner_product_i64(bdst.as_mut(), &[lhs_b.as_ref(), lhs_refs[0]], &[rhs_b.as_ref(), rhs_refs[0]]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let lb = lhs_b_data[flat_index([1, 2, 3, 4], [0, j, k, l])] as i128;
                        let rb = rhs_b_data[flat_index([2, 2, 3, 1], [i, j, k, 0])] as i128;
                        let expected = lb * rb + (lhs_data[0][pos] as i128) * (rhs_data[0][pos] as i128);
                        assert_eq!(bdst.data()[pos], expected);
                    }
                }
            }
        }

        // a line longer than the internal tile, to exercise the tiling loop
        let long_shape = [1, 1, 1, 600];
        let la: Vec<i64> = (0..600).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let lb2: Vec<i64> = (0..600).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let ra: Vec<i64> = (0..600).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let rb2: Vec<i64> = (0..600).map(|_| lcg(&mut state) as i64 % 1000 - 500).collect();
        let (mut lat, mut lbt, mut rat, mut rbt) = (
            CpuTensor::<i64>::zeros(long_shape),
            CpuTensor::<i64>::zeros(long_shape),
            CpuTensor::<i64>::zeros(long_shape),
            CpuTensor::<i64>::zeros(long_shape),
        );
        backend.hal_import(lat.as_mut(), &la);
        backend.hal_import(lbt.as_mut(), &lb2);
        backend.hal_import(rat.as_mut(), &ra);
        backend.hal_import(rbt.as_mut(), &rb2);
        let mut ldst = CpuTensor::<i128>::new(vec![-9; 600], long_shape);
        backend.hal_inner_product_i64(ldst.as_mut(), &[lat.as_ref(), lbt.as_ref()], &[rat.as_ref(), rbt.as_ref()]);
        for pos in 0..600 {
            assert_eq!(
                ldst.data()[pos],
                (la[pos] as i128) * (ra[pos] as i128) + (lb2[pos] as i128) * (rb2[pos] as i128)
            );
        }
    }

    #[test]
    fn test_mont_ops() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let mods = [backend.hal_create_modulus(97), backend.hal_create_modulus(65537)];
        let shape = [2, 2, 3, 4];

        // mul_mont: with one factor in montgomery form, the result is the plain product
        let mut state = 17;
        let a_data: Vec<i64> = (0..48).map(|_| lcg(&mut state) as i64 % 200 - 100).collect();
        let b_data: Vec<i64> = (0..48).map(|_| lcg(&mut state) as i64 % 200 - 100).collect();
        let mut dst = CpuTensor::<i64>::zeros(shape);
        let a_mont: Vec<i64> = (0..48).map(|pos| mods[(pos / 12) % 2].to_mont128(a_data[pos] as i128)).collect();
        backend.hal_import(dst.as_mut(), &a_mont);
        let mut src = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(src.as_mut(), &b_data);
        backend.hal_mul_mont64(dst.as_mut(), src.as_ref(), &mods);
        for j in 0..2 {
            let p = mods[j].modulus();
            for i in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let out = dst.data()[pos];
                        assert!(out.abs() <= p);
                        let expected = (a_data[pos] as i128) * (b_data[pos] as i128);
                        assert_eq!(pos_repr(out, p), pos_repr(expected, p));
                    }
                }
            }
        }

        // fma_double_mont: dst += src * 2^(-128) mod p
        let src128_data: Vec<i128> = (0..48).map(|x| (x as i128 - 24) * 1000000007 * 1000000007).collect();
        let mut src128 = CpuTensor::<i128>::zeros(shape);
        let mut pos = 0;
        for_each_line1(&mut src128.as_mut(), |_, line| {
            line.copy_from_slice(&src128_data[pos..pos + line.len()]);
            pos += line.len();
        });
        let mut acc = CpuTensor::<i64>::new(vec![1000; 48], shape);
        backend.hal_add_red_mont64_double(acc.as_mut(), src128.as_ref(), &mods);
        for j in 0..2 {
            let p = mods[j].modulus();
            for i in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let red = acc.data()[pos] - 1000;
                        assert!(red.abs() <= p);
                        // red == src * 2^(-128) mod p, i.e. red * 2^128 == src mod p
                        assert_eq!(montgomery_repr(montgomery_repr(red, p), p), pos_repr(src128_data[pos], p));
                    }
                }
            }
        }

        // red_mont64_double: the overwriting variant ignores the previous destination
        let mut overwritten = CpuTensor::<i64>::new(vec![1000; 48], shape);
        backend.hal_red_mont64_double(overwritten.as_mut(), src128.as_ref(), &mods);
        for pos in 0..48 {
            assert_eq!(overwritten.data()[pos], acc.data()[pos] - 1000);
        }

        // repr_to_mont: dst = src * 2^64 mod p
        let mut mont = CpuTensor::<i64>::zeros(shape);
        backend.hal_i128_to_mont64(mont.as_mut(), src128.as_ref(), &mods);
        for j in 0..2 {
            let p = mods[j].modulus();
            for i in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let out = mont.data()[pos];
                        assert!(out.abs() <= p);
                        let expected = montgomery_repr(src128_data[pos], p);
                        assert_eq!(pos_repr(out, p), expected);
                    }
                }
            }
        }

        // add_red_mont64: dst += src * 2^(-64) mod p
        let mut acc = CpuTensor::<i64>::new(vec![1000; 48], shape);
        backend.hal_add_red_mont64(acc.as_mut(), src128.as_ref(), &mods);
        for j in 0..2 {
            let p = mods[j].modulus();
            for i in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let red = acc.data()[pos] - 1000;
                        assert!(red.abs() <= p);
                        // red == src * 2^(-64) mod p, i.e. red * 2^64 == src mod p
                        assert_eq!(pos_repr(montgomery_repr(red, p), p), pos_repr(src128_data[pos], p));
                    }
                }
            }
        }

        // red_mont64: the overwriting variant ignores the previous destination
        let mut overwritten = CpuTensor::<i64>::new(vec![1000; 48], shape);
        backend.hal_red_mont64(overwritten.as_mut(), src128.as_ref(), &mods);
        for pos in 0..48 {
            assert_eq!(overwritten.data()[pos], acc.data()[pos] - 1000);
        }

        // i64_to_mont64: dst = dst * 2^64 mod p, in place, for arbitrary 64-bit inputs
        let plain_data: Vec<i64> = (0..48).map(|x| (x as i64 - 24) * 187654321987654321).collect();
        let mut in_place = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(in_place.as_mut(), &plain_data);
        backend.hal_i64_to_mont64(in_place.as_mut(), &mods);
        for j in 0..2 {
            let p = mods[j].modulus();
            for i in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let out = in_place.data()[pos];
                        assert!(out.abs() <= p);
                        assert_eq!(pos_repr(out, p), montgomery_repr(plain_data[pos], p));
                    }
                }
            }
        }
    }

    #[test]
    fn test_assign_narrow() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let shape = [2, 2, 3, 4];
        let src_data: Vec<i128> = (0..48).map(|x| (x as i128 - 24) * 123456789123).collect();
        let mut src = CpuTensor::<i128>::zeros(shape);
        let mut pos = 0;
        for_each_line1(&mut src.as_mut(), |_, line| {
            line.copy_from_slice(&src_data[pos..pos + line.len()]);
            pos += line.len();
        });
        let mut dst = CpuTensor::<i64>::zeros(shape);
        backend.hal_assign_narrow(dst.as_mut(), src.as_ref());
        assert_eq!(dst.data(), &src_data.iter().map(|x| *x as i64).collect::<Vec<_>>());

        // broadcasting along dimensions 0 and 3
        let bcast = CpuTensor::<i128>::new((0..6).map(|x| x as i128 + 100).collect(), [1, 2, 3, 1]);
        backend.hal_assign_narrow(dst.as_mut(), bcast.as_ref());
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        assert_eq!(dst.data()[flat_index(shape, [i, j, k, l])], (j * 3 + k) as i64 + 100);
                    }
                }
            }
        }

        // assign_i128 keeps the full 128-bit values, with the same broadcasting
        let mut dst128 = CpuTensor::<i128>::new(vec![-1; 48], shape);
        backend.hal_assign_i128(dst128.as_mut(), src.as_ref());
        assert_eq!(dst128.data(), &src_data);
        let bcast_large = CpuTensor::<i128>::new((0..6).map(|x| (x as i128 + 100) << 80).collect(), [1, 2, 3, 1]);
        backend.hal_assign_i128(dst128.as_mut(), bcast_large.as_ref());
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..4 {
                        assert_eq!(dst128.data()[flat_index(shape, [i, j, k, l])], ((j * 3 + k) as i128 + 100) << 80);
                    }
                }
            }
        }
    }

    #[test]
    fn test_bconv_correct_mont() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let mods = [backend.hal_create_modulus(97), backend.hal_create_modulus(65537)];
        let shape = [2, 2, 2, 3];
        let shift = 66;

        let lhs_small = [3i128, -7, 12, 0, 25, -1, 8, 4, -13, 6, 2, 9];
        let mut lhs = CpuTensor::<i128>::zeros([2, 1, 2, 3]);
        let mut pos = 0;
        for_each_line1(&mut lhs.as_mut(), |_, line| {
            for x in line.iter_mut() {
                // add a low-order part that must be shifted away
                *x = (lhs_small[pos] << shift) + (lhs_small[pos].rem_euclid(5) << 20) + 12345;
                pos += 1;
            }
        });
        let rhs_data = [1001, -4242];
        let mut rhs = CpuTensor::<i64>::zeros([1, 2, 1, 1]);
        backend.hal_import(rhs.as_mut(), &rhs_data);

        let mut dst = CpuTensor::<i64>::new(vec![500; 24], shape);
        backend.hal_bconv_correct_mont64(dst.as_mut(), lhs.as_ref(), rhs.as_ref(), &mods, shift);
        for i in 0..2 {
            for j in 0..2 {
                let p = mods[j].modulus();
                for k in 0..2 {
                    for l in 0..3 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let out = dst.data()[pos] - 500;
                        assert!(out.abs() <= p);
                        let expected = lhs_small[(i * 2 + k) * 3 + l] * (rhs_data[j] as i128);
                        // out == (lhs >> shift) * rhs * 2^(-64) mod p, i.e. out * 2^64 == (lhs >>
                        // shift) * rhs mod p
                        assert_eq!(montgomery_repr(out, p), pos_repr(expected, p));
                    }
                }
            }
        }

        // per-row rhs (no broadcast in dimension 0)
        let rhs_rows_data = [1001, -4242, 77, -33];
        let mut rhs_rows = CpuTensor::<i64>::zeros([2, 2, 1, 1]);
        backend.hal_import(rhs_rows.as_mut(), &rhs_rows_data);
        let mut dst = CpuTensor::<i64>::new(vec![500; 24], shape);
        backend.hal_bconv_correct_mont64(dst.as_mut(), lhs.as_ref(), rhs_rows.as_ref(), &mods, shift);
        for i in 0..2 {
            for j in 0..2 {
                let p = mods[j].modulus();
                for k in 0..2 {
                    for l in 0..3 {
                        let pos = flat_index(shape, [i, j, k, l]);
                        let out = dst.data()[pos] - 500;
                        assert!(out.abs() <= p);
                        let expected = lhs_small[(i * 2 + k) * 3 + l] * (rhs_rows_data[i * 2 + j] as i128);
                        assert_eq!(montgomery_repr(out, p), pos_repr(expected, p));
                    }
                }
            }
        }
    }

    #[test]
    fn test_ntt_roundtrip() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let q: i64 = 144115188075849889; // 57-bit prime with q - 1 divisible by 32 * 9
        assert_eq!(0, (q - 1) % (16 * 9));
        let modulus = backend.hal_create_modulus(q);
        let plan = backend.hal_create_ntt_plan(&modulus, &[4, 3]);
        let mut state = 7;

        let shape = [2, 2, 12, 5];
        let data: Vec<i64> = (0..240).map(|_| (lcg(&mut state) % (2 * q as u64)) as i64 - q).collect();
        let mut tensor = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(tensor.as_mut(), &data);
        backend.hal_ntt_mont64_dim(tensor.as_mut(), &[plan.clone(), plan.clone()], DataDim::Dim2);
        backend.hal_intt_mont64_dim(tensor.as_mut(), &[plan.clone(), plan.clone()], DataDim::Dim2);
        let mut out = vec![0; 240];
        backend.hal_export(tensor.as_ref(), &mut out);
        for pos in 0..240 {
            assert_eq!(pos_repr(data[pos], q), pos_repr(out[pos], q));
        }

        // the same along dimension 3
        let shape = [2, 2, 5, 12];
        let mut tensor = CpuTensor::<i64>::zeros(shape);
        backend.hal_import(tensor.as_mut(), &data);
        backend.hal_ntt_mont64_dim(tensor.as_mut(), &[plan.clone(), plan.clone()], DataDim::Dim3);
        backend.hal_intt_mont64_dim(tensor.as_mut(), &[plan.clone(), plan.clone()], DataDim::Dim3);
        backend.hal_export(tensor.as_ref(), &mut out);
        for pos in 0..240 {
            assert_eq!(pos_repr(data[pos], q), pos_repr(out[pos], q));
        }
    }

    /// The NTT of a plan with inner shape `[n0, ..., nr]` must diagonalize the group convolution
    /// over `C_n0 x ... x C_nr`, whatever its internal ordering conventions are.
    #[test]
    fn test_ntt_convolution_theorem() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let q: i64 = 144115188075849889;
        let modulus = backend.hal_create_modulus(q);
        let inner_shape = [8, 9, 2];
        let n: usize = inner_shape.iter().product();
        let plan = backend.hal_create_ntt_plan(&modulus, &inner_shape);
        let mut state = 42;

        let a: Vec<i64> = (0..n).map(|_| (lcg(&mut state) % 1000) as i64 - 500).collect();
        let b: Vec<i64> = (0..n).map(|_| (lcg(&mut state) % 1000) as i64 - 500).collect();
        let mut expected = vec![0i128; n];
        for i0 in 0..8 {
            for i1 in 0..9 {
                for i2 in 0..2 {
                    for j0 in 0..8 {
                        for j1 in 0..9 {
                            for j2 in 0..2 {
                                let dst = (((i0 + j0) % 8) * 9 + (i1 + j1) % 9) * 2 + (i2 + j2) % 2;
                                expected[dst] += (a[(i0 * 9 + i1) * 2 + i2] * b[(j0 * 9 + j1) * 2 + j2]) as i128;
                            }
                        }
                    }
                }
            }
        }

        let a_mont: Vec<i64> = a.iter().map(|x| montgomery_repr(*x, q)).collect();
        let b_mont: Vec<i64> = b.iter().map(|x| montgomery_repr(*x, q)).collect();
        let mut a_tensor = CpuTensor::<i64>::zeros([1, 1, n, 1]);
        let mut b_tensor = CpuTensor::<i64>::zeros([1, 1, n, 1]);
        backend.hal_import(a_tensor.as_mut(), &a_mont);
        backend.hal_import(b_tensor.as_mut(), &b_mont);
        backend.hal_ntt_mont64_dim(a_tensor.as_mut(), &[plan.clone()], DataDim::Dim2);
        backend.hal_ntt_mont64_dim(b_tensor.as_mut(), &[plan.clone()], DataDim::Dim2);
        backend.hal_mul_mont64(a_tensor.as_mut(), b_tensor.as_ref(), &[modulus]);
        backend.hal_intt_mont64_dim(a_tensor.as_mut(), &[plan.clone()], DataDim::Dim2);
        let mut out = vec![0; n];
        backend.hal_export(a_tensor.as_ref(), &mut out);
        for pos in 0..n {
            assert_eq!(pos_repr(montgomery_repr(expected[pos], q), q), pos_repr(out[pos], q));
        }
    }

    /// `ntt_mul_mont64_dim` must produce exactly the same result as an `ntt_mont64_dim`
    /// followed by a `mul_mont64`, including when `mul` is broadcast, and along both dimensions.
    #[test]
    fn test_ntt_mul_mont64_dim() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let q: i64 = 144115188075849889;
        let modulus = backend.hal_create_modulus(q);
        let plan = backend.hal_create_ntt_plan(&modulus, &[8, 3]);
        let mut state = 123;
        let n = 24;

        // inputs must be within [-2q, 2q] for the forward transform
        let make_data = |state: &mut u64, len: usize| -> Vec<i64> {
            (0..len).map(|_| (lcg(state) % (4 * q as u64)) as i64 - 2 * q).collect()
        };

        for dim in [DataDim::Dim2, DataDim::Dim3] {
            let (shape, kernel_shape) = match dim {
                DataDim::Dim2 => ([2, 2, n, 3], [1, 2, n, 1]),
                DataDim::Dim3 => ([2, 2, 3, n], [1, 2, 1, n]),
            };
            let data = make_data(&mut state, shape.iter().product());
            // a montgomery-form kernel (broadcast over dims 0 and one data dimension)
            let kernel: Vec<i64> = (0..kernel_shape.iter().product::<usize>())
                .map(|_| montgomery_repr((lcg(&mut state) % (2 * q as u64)) as i64 - q, q))
                .collect();
            let mut kernel_tensor = CpuTensor::<i64>::zeros(kernel_shape);
            backend.hal_import(kernel_tensor.as_mut(), &kernel);
            let plans = [plan.clone(), plan.clone()];
            let mods = [modulus, modulus];

            // reference: separate transform and multiplication
            let mut reference = CpuTensor::<i64>::zeros(shape);
            backend.hal_import(reference.as_mut(), &data);
            backend.hal_ntt_mont64_dim(reference.as_mut(), &plans, dim);
            backend.hal_mul_mont64(reference.as_mut(), kernel_tensor.as_ref(), &mods);

            // fused
            let mut fused = CpuTensor::<i64>::zeros(shape);
            backend.hal_import(fused.as_mut(), &data);
            backend.hal_ntt_mul_mont64_dim(fused.as_mut(), &plans, dim, kernel_tensor.as_ref(), &mods);

            let len = shape.iter().product();
            let mut ref_out = vec![0; len];
            let mut fused_out = vec![0; len];
            backend.hal_export(reference.as_ref(), &mut ref_out);
            backend.hal_export(fused.as_ref(), &mut fused_out);
            for pos in 0..len {
                assert!(fused_out[pos].abs() <= q, "fused output not reduced at position {}", pos);
                assert_eq!(ref_out[pos], fused_out[pos], "dim = {:?}, position {}", dim, pos);
            }
        }
    }

    #[test]
    fn test_permute() {
        feanor_tracing::DelayedLogger::init_test();
        let backend = CpuBackend;
        let inner_shape = [3, 4];
        let shifts = [1, 2];
        let plan = backend.hal_create_permutation(&inner_shape, &shifts);

        // along dimension 3
        let shape3 = [1, 2, 2, 12];
        let data: Vec<i64> = (0..48).collect();
        let mut src = CpuTensor::<i64>::zeros(shape3);
        backend.hal_import(src.as_mut(), &data);
        let mut dst = CpuTensor::<i64>::zeros(shape3);
        backend.hal_permute(dst.as_mut(), src.as_ref(), &plan, DataDim::Dim3);
        for i in 0..1 {
            for j in 0..2 {
                for k in 0..2 {
                    for a0 in 0..3 {
                        for a1 in 0..4 {
                            let from = flat_index(shape3, [i, j, k, a0 * 4 + a1]);
                            let to = flat_index(shape3, [i, j, k, ((a0 + 1) % 3) * 4 + (a1 + 2) % 4]);
                            assert_eq!(dst.data()[to], data[from]);
                        }
                    }
                }
            }
        }

        // along dimension 2
        let shape2 = [1, 2, 12, 2];
        let mut src = CpuTensor::<i64>::zeros(shape2);
        backend.hal_import(src.as_mut(), &data);
        let mut dst = CpuTensor::<i64>::zeros(shape2);
        backend.hal_permute(dst.as_mut(), src.as_ref(), &plan, DataDim::Dim2);
        for j in 0..2 {
            for a0 in 0..3 {
                for a1 in 0..4 {
                    for l in 0..2 {
                        let from = flat_index(shape2, [0, j, a0 * 4 + a1, l]);
                        let to = flat_index(shape2, [0, j, ((a0 + 1) % 3) * 4 + (a1 + 2) % 4, l]);
                        assert_eq!(dst.data()[to], data[from]);
                    }
                }
            }
        }
    }
}
