//! Lazy NTTs built on Barrett-style multiplication by constants.
//!
//! # Multiplication by constants
//!
//! All twiddle multiplications are performed by [`BarrettMul`]: for a fixed constant `z`
//! (stored as its centered representative modulo the odd modulus `N`, so `|z| <= (N - 1)/2`),
//! we precompute `u = round(-z * 2^64 / N)`, which fits into an `i64`. Given `|x| <= 2^63`,
//! ```text
//!   q = round(u * x / 2^64),    t = z * x + q * N   (wrapping 64-bit arithmetic)
//! ```
//! satisfies `t = z * x mod N` and `|t| <= (3/4) N`: writing `u = -z 2^64 / N + δ1` and
//! `q = u x / 2^64 + δ2` with `|δ1|, |δ2| <= 1/2` gives `t = N (δ1 x / 2^64 + δ2)`, so
//! `|t| <= N (2^63 / 2^65 + 1/2) = (3/4) N`. Since the exact value fits into an `i64`, the
//! wrapping arithmetic computes it exactly. In contrast to a montgomery multiplication, this
//! doesn't introduce a factor `2^-64`, so the transforms map montgomery-form inputs to
//! montgomery-form outputs with *plain* twiddle constants.
//!
//! # The transforms, and how lazy they are
//!
//! The forward transform of one cyclic factor `C_len` is the Cooley-Tukey style splitting of
//! `X^len - 1` into linear factors: a stage splits the residue of `f = a + X^m b` modulo
//! `X^(2m) - v^2` into the residues `a ± v b` modulo `X^m ∓ v` (radix 2; the radix-3 stage
//! splits `X^(3m) - v^3` into `X^m - ζ^k v` analogously, `ζ` a primitive cube root of unity).
//! The output is the vector of evaluations of `f` at all `len`-th roots of unity, in the
//! digit-reversed order induced by the recursion; since the "NTT domain" is only ever used
//! for pointwise multiplication with data that went through the same plan, the ordering is
//! invisible from outside.
//!
//! The butterfly `(a + v b, a - v b)` reduces only the product `t = v b` (into
//! `[-N, N]`, via [`BarrettMul`]); the sums are *not* reduced. Starting from inputs bounded
//! by `2N`, the entries therefore grow by `N` per radix-2 stage (by `2N` per radix-3 stage,
//! whose butterfly adds two reduced products). Only when the *next* stage could exceed
//! `i64::MAX` is a reduction pass (a Barrett multiplication by `1`, bringing all entries
//! back to `[-N, N]`) inserted; for 57-bit moduli this never happens (that would take more
//! than 61 radix-2 stages, i.e. `len > 2^61`). Only one final reduction is needed to bring
//! the output back to `[-N, N]`, and it is fused into the last butterfly stage — unless the
//! caller requests unreduced output (see below), in which case even that is skipped.
//!
//! # Twiddle-1 butterflies
//!
//! Because this is a *cyclic* (not negacyclic) NTT, every stage has exactly one block whose
//! splitting constant is `v = ω^0 = 1` (the block covering the residue modulo `X^m - 1`).
//! Its butterfly is `(a + b, a - b)`, so the Barrett multiplication `t = v b` degenerates to
//! a plain reduction of `b`. We special-case it and skip the reduction, adding/subtracting
//! `b` directly. The catch is bookkeeping: the reduction of `b` was what kept the block's
//! entries bounded (`a` grows only additively, `t` is capped at `N`); without it, both
//! operands are only bounded by the running `bound`, so the output grows to `2 bound`
//! (radix 2) resp. `3 bound` (radix 3) instead of `bound + N` resp. `bound + 2N`. The
//! forward transform therefore only skips the reduction while `2 bound` (resp. `3 bound`)
//! still fits into an `i64`; once it would not, the twiddle-1 block falls back to the Barrett
//! reduction (`v = 1` reduces `b` for free), which keeps the running bound additive and
//! avoids any full reduction pass. In the *inverse* transform the skip is always free: the
//! entry bound there is already dominated by the unmultiplied `y0 + y1` branch (it doubles
//! resp. triples per stage regardless), so dropping the multiplication of the *other* branch
//! never enlarges it.
//!
//! # Unreduced output and fused multiplication
//!
//! The CRT transforms follow every forward transform with a pointwise montgomery
//! multiplication by a kernel tensor, which itself reduces its result into `[-N, N]`. It is
//! therefore wasteful for the forward transform to first reduce its output. When the caller
//! passes `reduce_output = false`, the last stage leaves its output *unreduced* — still
//! bounded by `i64::MAX`, hence a valid `mul_mont64` operand (the product with a montgomery
//! constant in `[-N, N]` stays within the `2^63 N` bound of the montgomery reduction) — and
//! the subsequent multiplication performs the reduction. The HAL exposes this as
//! `ntt_mul_mont64_dim`, which fuses the reduction of the forward transform into the kernel
//! multiplication.
//!
//! The inverse transform runs the recursion backwards, with butterfly
//! `(y0 + y1, v^-1 (y0 - y1))` (up to the deferred global division by `len`, fused into the
//! last processed stage). Laziness helps less here: the branch `y0 + y1` is not multiplied
//! by anything, so it cannot be Barrett-reduced for free, and the entry bound *doubles* per
//! radix-2 stage (triples per radix-3 stage) instead of growing additively. This is inherent
//! to inverting the forward butterfly — the inverse of `(a + vb, a - vb)` necessarily forms
//! the unscaled sum `y0 + y1 = 2a` (reducing it would cost an extra multiplication per
//! butterfly, which is exactly the cost the lazy approach avoids); using the sum-forming
//! butterfly for the forward direction instead would only move the problem, since one of the
//! two directions always ends up with the unmultiplied branch. Consequently the inverse
//! needs a reduction pass every few stages (for 57-bit moduli: every 6 radix-2 stages), which
//! is still far cheaper than reducing every butterfly.
//!
//! Both transforms track a rigorous worst-case entry bound (`u128`, in absolute value) and
//! decide reduction passes from it; the caller provides the input bound and receives the
//! output bound.

/// Multiplication modulo `q` (any `q < 2^63`).
pub(super) fn mul_mod(a: i64, b: i64, q: i64) -> i64 { ((a as i128 * b as i128) % q as i128) as i64 }

pub(super) fn pow_mod(mut b: i64, mut e: u64, q: i64) -> i64 {
    let mut result = 1;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_mod(result, b, q);
        }
        b = mul_mod(b, b, q);
        e >>= 1;
    }
    return result;
}

/// Finds an element of order exactly `n` in `Z/qZ` (`q` prime, `n` a power of 2 or 3),
/// deterministically.
pub(super) fn find_root_of_unity(q: i64, n: usize) -> i64 {
    assert!(n > 1);
    assert_eq!((q - 1) % n as i64, 0, "modulus doesn't support roots of unity of order {}", n);
    let p: usize = if n.is_power_of_two() { 2 } else { 3 };
    for candidate in 2.. {
        let root = pow_mod(candidate, ((q - 1) / n as i64) as u64, q);
        if pow_mod(root, (n / p) as u64, q) != 1 {
            return root;
        }
    }
    unreachable!()
}

/// `round(a / b)` for `b > 0`, rounding half away from zero.
fn rounded_div(a: i128, b: i128) -> i128 {
    debug_assert!(b > 0);
    return if a >= 0 { (a + b / 2) / b } else { -((-a + b / 2) / b) };
}

/// Multiplication by a fixed constant `z` modulo an odd `N < 2^62`, reducing into `[-N, N]`;
/// see the module documentation for the algorithm and its error analysis.
#[derive(Debug, Clone, Copy)]
pub struct BarrettMul {
    /// the centered representative of the constant, `|z| <= (N - 1)/2`
    z: i64,
    /// `round(-z * 2^64 / N)`; `|u| < 2^63` since `|z| < N/2`
    u: i64,
}

impl BarrettMul {
    pub fn new(value: i64, n: i64) -> Self {
        assert!(n > 1 && n % 2 == 1 && n < (1 << 62));
        let mut z = value % n;
        if 2 * z > n {
            z -= n;
        } else if 2 * z < -n {
            z += n;
        }
        let u = rounded_div(-((z as i128) << 64), n as i128);
        return Self { z, u: i64::try_from(u).unwrap() };
    }

    /// Given `|x| <= 2^63`, returns a value congruent to `z * x` modulo `N` and bounded by
    /// `(3/4) N < N` in absolute value.
    #[inline(always)]
    pub fn mul(self, x: i64, n: i64) -> i64 {
        let q = (((self.u as i128) * (x as i128) + (1i128 << 63)) >> 64) as i64;
        return self.z.wrapping_mul(x).wrapping_add(q.wrapping_mul(n));
    }

    /// Whether the constant is `1` (its centered representative equals `1`), so that
    /// [`BarrettMul::mul()`] would only reduce its argument; the twiddle-1 fast path skips it.
    #[inline(always)]
    pub fn is_one(self) -> bool { self.z == 1 }
}

/// Reduces the `len * vec_len` addressed entries into `[-N, N]` by Barrett-multiplying with
/// `1`.
///
/// Safety: the caller must ensure the addressed memory is valid and exclusive.
unsafe fn reduce_pass(one: BarrettMul, n: i64, data: *mut i64, elem_stride: usize, len: usize, vec_len: usize) {
    for i in 0..len {
        for l in 0..vec_len {
            unsafe {
                let x = data.add(i * elem_stride + l);
                *x = one.mul(*x, n);
            }
        }
    }
}

/// `mul.mul(x, n)` if `reduce` is set, `x` otherwise; `reduce` is always loop-invariant, so
/// the branch vanishes after loop unswitching.
#[inline(always)]
fn reduce_if(reduce: bool, mul: BarrettMul, n: i64, x: i64) -> i64 { if reduce { mul.mul(x, n) } else { x } }

/// The lazy DFT along one cyclic factor `C_len`, `len` a power of 2 or of 3; see the module
/// documentation.
///
/// The stage-`s` twiddles are indexed by the `radix^s` *blocks* of the stage (all butterflies
/// of a block share the splitting constant `v`), following the residue tree of the recursive
/// splitting: the block covering positions `[b * len/radix^s, (b+1) * len/radix^s)` holds the
/// residue of the input modulo `X^(len/radix^s) - ω^e[s][b]`, where `e[0] = [0]` and the
/// (in-order) children of a node `e` are `e/radix + k * len/radix` for `k < radix`.
#[derive(Debug, Clone)]
pub(super) enum LazyFactorNTT {
    Radix2 {
        n: i64,
        one: BarrettMul,
        len: usize,
        /// `twiddles[s][b] = v`, the splitting constant `ω^(e[s][b] / 2)` of block `b` in
        /// stage `s`
        twiddles: Vec<Vec<BarrettMul>>,
        /// `inv_twiddles[s][b] = v^-1`
        inv_twiddles: Vec<Vec<BarrettMul>>,
    },
    Radix3 {
        n: i64,
        one: BarrettMul,
        len: usize,
        /// `twiddles[s][6b .. 6b + 6] = [v, ζv, ζ²v, v², ζv², ζ²v²]` for the splitting
        /// constant `v = ω^(e[s][b] / 3)` of block `b` in stage `s`
        twiddles: Vec<Vec<BarrettMul>>,
        /// `inv_twiddles[s][2b .. 2b + 2] = [v^-1, v^-2]`
        inv_twiddles: Vec<Vec<BarrettMul>>,
        /// the primitive cube root of unity `ζ = ω^(len/3)` and its square
        zeta: BarrettMul,
        zeta2: BarrettMul,
    },
}

impl LazyFactorNTT {
    pub(super) fn new(n: i64, len: usize) -> Self {
        if len.is_power_of_two() {
            assert!(len >= 2);
            // a stage on entries reduced to [-N, N] outputs entries bounded by 2N
            assert!(2 * (n as u128) <= i64::MAX as u128, "modulus too large for lazy radix-2 butterflies");
            let one = BarrettMul::new(1, n);
            let root = find_root_of_unity(n, len);
            let inv_root = pow_mod(root, len as u64 - 1, n);
            let mut twiddles = Vec::new();
            let mut inv_twiddles = Vec::new();
            let mut exps: Vec<usize> = vec![0];
            while exps.len() < len {
                twiddles.push(exps.iter().map(|e| BarrettMul::new(pow_mod(root, (e / 2) as u64, n), n)).collect());
                inv_twiddles
                    .push(exps.iter().map(|e| BarrettMul::new(pow_mod(inv_root, (e / 2) as u64, n), n)).collect());
                exps = exps.iter().flat_map(|e| [e / 2, e / 2 + len / 2]).collect();
            }
            return Self::Radix2 { n, one, len, twiddles, inv_twiddles };
        } else {
            let mut l = len;
            while l > 1 {
                assert_eq!(0, l % 3, "NTT length must be a power of 2 or a power of 3");
                l /= 3;
            }
            assert!(len >= 3);
            // a stage on entries reduced to [-N, N] outputs entries bounded by 3N
            assert!(3 * (n as u128) <= i64::MAX as u128, "modulus too large for lazy radix-3 butterflies");
            let one = BarrettMul::new(1, n);
            let root = find_root_of_unity(n, len);
            let inv_root = pow_mod(root, len as u64 - 1, n);
            let zeta_plain = pow_mod(root, (len / 3) as u64, n);
            let zeta2_plain = mul_mod(zeta_plain, zeta_plain, n);
            let mut twiddles = Vec::new();
            let mut inv_twiddles = Vec::new();
            let mut exps: Vec<usize> = vec![0];
            while exps.len() < len {
                let mut stage_twiddles = Vec::with_capacity(6 * exps.len());
                let mut stage_inv_twiddles = Vec::with_capacity(2 * exps.len());
                for e in &exps {
                    let v = pow_mod(root, (e / 3) as u64, n);
                    let v_sqr = mul_mod(v, v, n);
                    stage_twiddles.push(BarrettMul::new(v, n));
                    stage_twiddles.push(BarrettMul::new(mul_mod(zeta_plain, v, n), n));
                    stage_twiddles.push(BarrettMul::new(mul_mod(zeta2_plain, v, n), n));
                    stage_twiddles.push(BarrettMul::new(v_sqr, n));
                    stage_twiddles.push(BarrettMul::new(mul_mod(zeta_plain, v_sqr, n), n));
                    stage_twiddles.push(BarrettMul::new(mul_mod(zeta2_plain, v_sqr, n), n));
                    let v_inv = pow_mod(inv_root, (e / 3) as u64, n);
                    stage_inv_twiddles.push(BarrettMul::new(v_inv, n));
                    stage_inv_twiddles.push(BarrettMul::new(mul_mod(v_inv, v_inv, n), n));
                }
                twiddles.push(stage_twiddles);
                inv_twiddles.push(stage_inv_twiddles);
                exps = exps.iter().flat_map(|e| [e / 3, e / 3 + len / 3, e / 3 + 2 * (len / 3)]).collect();
            }
            return Self::Radix3 {
                n,
                one,
                len,
                twiddles,
                inv_twiddles,
                zeta: BarrettMul::new(zeta_plain, n),
                zeta2: BarrettMul::new(zeta2_plain, n),
            };
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Radix2 { len, .. } => *len,
            Self::Radix3 { len, .. } => *len,
        }
    }

    /// Applies the forward transform to the `len * vec_len` elements at
    /// `data[i * elem_stride .. i * elem_stride + vec_len]` (`i < len`), i.e. to `vec_len`
    /// interleaved transforms whose element `i` occupies a contiguous line of length
    /// `vec_len`. All input entries must be bounded by `in_bound <= i64::MAX` in absolute
    /// value; returns the corresponding bound on the output entries (also `<= i64::MAX`).
    ///
    /// If `reduce_output` is set, the last stage additionally Barrett-reduces its outputs,
    /// so the returned bound is `N`; this costs exactly as many multiplications as a separate
    /// reduction pass, but saves traversing the data once more.
    ///
    /// Safety: the caller must ensure the addressed memory is valid and exclusive.
    pub(super) unsafe fn forward(
        &self,
        data: *mut i64,
        elem_stride: usize,
        vec_len: usize,
        in_bound: u128,
        reduce_output: bool,
    ) -> u128 {
        debug_assert!(in_bound <= i64::MAX as u128);
        let mut bound = in_bound;
        match self {
            Self::Radix2 { n, one, len, twiddles, .. } => {
                let n = *n;
                let one = *one;
                let mut stage_len = *len;
                for (s, stage_twiddles) in twiddles.iter().enumerate() {
                    let fuse = reduce_output && s + 1 == twiddles.len();
                    // non-twiddle-1 outputs are a ± t with |t| <= N, so the bound grows by N
                    if bound + n as u128 > i64::MAX as u128 {
                        unsafe { reduce_pass(one, n, data, elem_stride, *len, vec_len) };
                        bound = n as u128;
                    }
                    // a twiddle-1 block only reduces the twiddle multiplication (into a plain
                    // add/sub) while the resulting a ± b (bounded by 2 bound) still fits; then
                    // its output grows to 2 bound instead of bound + N
                    let skip_ok = 2 * bound <= i64::MAX as u128;
                    let new_bound = if fuse {
                        n as u128
                    } else if skip_ok {
                        2 * bound
                    } else {
                        bound + n as u128
                    };
                    let half = stage_len / 2;
                    for (block, block_start) in (0..*len).step_by(stage_len).enumerate() {
                        let v = stage_twiddles[block];
                        let skip = skip_ok && v.is_one();
                        for j in 0..half {
                            let x = unsafe { data.add((block_start + j) * elem_stride) };
                            let y = unsafe { data.add((block_start + j + half) * elem_stride) };
                            for l in 0..vec_len {
                                unsafe {
                                    let a = *x.add(l);
                                    let t = if skip { *y.add(l) } else { v.mul(*y.add(l), n) };
                                    *x.add(l) = reduce_if(fuse, one, n, a + t);
                                    *y.add(l) = reduce_if(fuse, one, n, a - t);
                                }
                            }
                        }
                    }
                    bound = new_bound;
                    stage_len = half;
                }
            }
            Self::Radix3 { n, one, len, twiddles, .. } => {
                let n = *n;
                let one = *one;
                let mut stage_len = *len;
                for (s, stage_twiddles) in twiddles.iter().enumerate() {
                    let fuse = reduce_output && s + 1 == twiddles.len();
                    // non-twiddle-1 outputs are a + t + t' with |t|, |t'| <= N (bound + 2N)
                    if bound + 2 * n as u128 > i64::MAX as u128 {
                        unsafe { reduce_pass(one, n, data, elem_stride, *len, vec_len) };
                        bound = n as u128;
                    }
                    // in a twiddle-1 block, `v` and `v²` degenerate to 1, so `t_b = b` and
                    // `t_c = c` skip their Barrett reductions (the `ζ`-scaled products stay
                    // reduced); the largest output a + b + c is then bounded by 3 bound
                    let skip_ok = 3 * bound <= i64::MAX as u128;
                    let new_bound = if fuse {
                        n as u128
                    } else if skip_ok {
                        3 * bound
                    } else {
                        bound + 2 * n as u128
                    };
                    let third = stage_len / 3;
                    for (block, block_start) in (0..*len).step_by(stage_len).enumerate() {
                        // copy the block twiddles into locals: reading them through a slice
                        // inside the loop would force reloads after every (potentially
                        // aliasing) store through the raw data pointer
                        let [w_b, w_bz, w_bzz, w_c, w_cz, w_czz]: [BarrettMul; 6] =
                            stage_twiddles[6 * block..6 * block + 6].try_into().unwrap();
                        // w_b == 1 (i.e. v == 1) implies w_c == v² == 1
                        let skip = skip_ok && w_b.is_one();
                        for j in 0..third {
                            let x = unsafe { data.add((block_start + j) * elem_stride) };
                            let y = unsafe { data.add((block_start + j + third) * elem_stride) };
                            let z = unsafe { data.add((block_start + j + 2 * third) * elem_stride) };
                            for l in 0..vec_len {
                                unsafe {
                                    let a = *x.add(l);
                                    let b = *y.add(l);
                                    let c = *z.add(l);
                                    let t_b = if skip { b } else { w_b.mul(b, n) }; // v b
                                    let t_bz = w_bz.mul(b, n); // ζ v b
                                    let t_bzz = w_bzz.mul(b, n); // ζ² v b
                                    let t_c = if skip { c } else { w_c.mul(c, n) }; // v² c
                                    let t_cz = w_cz.mul(c, n); // ζ v² c
                                    let t_czz = w_czz.mul(c, n); // ζ² v² c
                                    *x.add(l) = reduce_if(fuse, one, n, a + t_b + t_c);
                                    *y.add(l) = reduce_if(fuse, one, n, a + t_bz + t_czz);
                                    *z.add(l) = reduce_if(fuse, one, n, a + t_bzz + t_cz);
                                }
                            }
                        }
                    }
                    bound = new_bound;
                    stage_len = third;
                }
            }
        }
        return bound;
    }

    /// Inverts [`LazyFactorNTT::forward()`], except for the global scaling by `len^-1` (which
    /// the caller defers to the end of the whole multi-factor transform). Bound handling as
    /// in [`LazyFactorNTT::forward()`].
    ///
    /// If `final_scale` is given, the last processed stage multiplies *all* its outputs by it
    /// (this works because that stage's splitting constant is `v = ω^0 = 1` by construction,
    /// so its outputs are plain linear combinations); the caller passes the deferred global
    /// `len^-1` here, which both applies the scaling and reduces the output into `[-N, N]`
    /// without traversing the data once more.
    ///
    /// Safety: see [`LazyFactorNTT::forward()`].
    pub(super) unsafe fn backward(
        &self,
        data: *mut i64,
        elem_stride: usize,
        vec_len: usize,
        in_bound: u128,
        final_scale: Option<BarrettMul>,
    ) -> u128 {
        debug_assert!(in_bound <= i64::MAX as u128);
        let mut bound = in_bound;
        match self {
            Self::Radix2 { n, one, len, inv_twiddles, .. } => {
                let n = *n;
                for (s, stage_twiddles) in inv_twiddles.iter().enumerate().rev() {
                    let fused_scale = if s == 0 { final_scale } else { None };
                    // the unmultiplied output y0 + y1 doubles the bound per stage
                    if 2 * bound > i64::MAX as u128 {
                        unsafe { reduce_pass(*one, n, data, elem_stride, *len, vec_len) };
                        bound = n as u128;
                    }
                    bound = if fused_scale.is_some() { n as u128 } else { 2 * bound };
                    let stage_len = *len >> s;
                    let half = stage_len / 2;
                    for (block, block_start) in (0..*len).step_by(stage_len).enumerate() {
                        // in the fused stage (a single block with v = 1), both outputs are
                        // multiplied by the deferred scaling constant instead
                        let (fuse, y_mul) = match fused_scale {
                            Some(scale) => {
                                debug_assert_eq!(1, stage_twiddles[block].z);
                                (true, scale)
                            }
                            None => (false, stage_twiddles[block]),
                        };
                        // when v^-1 == 1 (and this is not the fused-scale stage, which must
                        // apply the deferred len^-1) the `y0 - y1` branch skips its Barrett
                        // reduction; this never enlarges the bound, since the `y0 + y1` branch
                        // (bounded by 2 bound) already dominates it
                        let skip = !fuse && y_mul.is_one();
                        for j in 0..half {
                            let x = unsafe { data.add((block_start + j) * elem_stride) };
                            let y = unsafe { data.add((block_start + j + half) * elem_stride) };
                            for l in 0..vec_len {
                                unsafe {
                                    let y0 = *x.add(l);
                                    let y1 = *y.add(l);
                                    *x.add(l) = reduce_if(fuse, y_mul, n, y0 + y1);
                                    *y.add(l) = if skip { y0 - y1 } else { y_mul.mul(y0 - y1, n) };
                                }
                            }
                        }
                    }
                }
            }
            Self::Radix3 { n, one, len, inv_twiddles, zeta, zeta2, .. } => {
                let n = *n;
                for (s, stage_twiddles) in inv_twiddles.iter().enumerate().rev() {
                    let fused_scale = if s == 0 { final_scale } else { None };
                    // the unmultiplied output y0 + y1 + y2 triples the bound per stage; the
                    // multiplied outputs are safe as well, since |y0 + ζ^k y1 + ζ^k' y2| is
                    // bounded by `bound + 2N <= 3 bound`
                    if 3 * bound > i64::MAX as u128 {
                        unsafe { reduce_pass(*one, n, data, elem_stride, *len, vec_len) };
                        bound = n as u128;
                    }
                    bound = if fused_scale.is_some() { n as u128 } else { 3 * bound };
                    let stage_len = *len / 3usize.pow(s as u32);
                    let third = stage_len / 3;
                    for (block, block_start) in (0..*len).step_by(stage_len).enumerate() {
                        // in the fused stage (a single block with v = 1), all outputs are
                        // multiplied by the deferred scaling constant instead
                        let (fuse, y_mul, z_mul) = match fused_scale {
                            Some(scale) => {
                                debug_assert_eq!(1, stage_twiddles[2 * block].z);
                                debug_assert_eq!(1, stage_twiddles[2 * block + 1].z);
                                (true, scale, scale)
                            }
                            None => (false, stage_twiddles[2 * block], stage_twiddles[2 * block + 1]),
                        };
                        let (zeta, zeta2) = (*zeta, *zeta2);
                        // when v^-1 == 1 (implying v^-2 == 1) and this is not the fused-scale
                        // stage, the outer multiplications of the `b`/`c` branches degenerate
                        // to 1 and are skipped; the `y0 + y1 + y2` branch (bounded by 3 bound)
                        // still dominates, so the bound is unchanged
                        let skip = !fuse && y_mul.is_one();
                        for j in 0..third {
                            let x = unsafe { data.add((block_start + j) * elem_stride) };
                            let y = unsafe { data.add((block_start + j + third) * elem_stride) };
                            let z = unsafe { data.add((block_start + j + 2 * third) * elem_stride) };
                            for l in 0..vec_len {
                                unsafe {
                                    let y0 = *x.add(l);
                                    let y1 = *y.add(l);
                                    let y2 = *z.add(l);
                                    let y1z = zeta.mul(y1, n); // ζ y1
                                    let y1zz = zeta2.mul(y1, n); // ζ² y1
                                    let y2z = zeta.mul(y2, n); // ζ y2
                                    let y2zz = zeta2.mul(y2, n); // ζ² y2
                                    // 3a = y0 + y1 + y2, 3vb = y0 + ζ²y1 + ζy2,
                                    // 3v²c = y0 + ζy1 + ζ²y2
                                    *x.add(l) = reduce_if(fuse, y_mul, n, y0 + y1 + y2);
                                    *y.add(l) = if skip { y0 + y1zz + y2z } else { y_mul.mul(y0 + y1zz + y2z, n) };
                                    *z.add(l) = if skip { y0 + y1z + y2zz } else { z_mul.mul(y0 + y1z + y2zz, n) };
                                }
                            }
                        }
                    }
                }
            }
        }
        return bound;
    }
}

/// The multi-dimensional group-NTT over `C_n0 x ... x C_n(r-1)` for one modulus; see
/// [`super::Backend::create_ntt_plan()`]. Requires each `n_i` to be a power of 2 or a power
/// of 3.
///
/// Inputs must be bounded by `2N` in absolute value. The inverse transform reduces its output
/// into `[-N, N]` (via the final `len^-1` scaling); the forward transform does so only when
/// `reduce_output` is set (via its final reduction pass), and otherwise leaves the output
/// unreduced but bounded by `i64::MAX`. See [`CpuNTTPlan::apply()`].
#[derive(Debug, Clone)]
pub struct CpuNTTPlan {
    n: i64,
    factors: Vec<LazyFactorNTT>,
    len: usize,
    /// the centered `len^-1 mod N` (plain, not montgomery form), applied at the end of the
    /// inverse transform
    inv_len: BarrettMul,
    /// multiplication by `1`; the final reduction pass of the forward transform
    one: BarrettMul,
}

impl CpuNTTPlan {
    pub(super) fn new(n: i64, inner_shape: &[usize]) -> Self {
        let factors = inner_shape.iter().filter(|l| **l > 1).map(|l| LazyFactorNTT::new(n, *l)).collect::<Vec<_>>();
        let len: usize = inner_shape.iter().product();
        let inv_len = pow_mod(len as i64, (n - 2) as u64, n);
        return Self { n, factors, len, inv_len: BarrettMul::new(inv_len, n), one: BarrettMul::new(1, n) };
    }

    pub(super) fn len(&self) -> usize { self.len }

    /// Applies the multi-dimensional transform along an axis of length `self.len` with
    /// element stride `elem_stride`, vectorized over a contiguous inner dimension of length
    /// `vec_len`. Input entries must be bounded by `2N`.
    ///
    /// The inverse transform always reduces its output into `[-N, N]` (via the deferred
    /// `len^-1` scaling). The forward transform reduces its output into `[-N, N]` only if
    /// `reduce_output` is set; otherwise its output is left unreduced (still bounded by
    /// `i64::MAX`), for a caller that reduces it via a subsequent montgomery multiplication.
    /// `reduce_output` is ignored for the inverse transform.
    ///
    /// Safety: the caller must ensure the addressed memory is valid and exclusive.
    pub(super) unsafe fn apply(
        &self,
        data: *mut i64,
        elem_stride: usize,
        vec_len: usize,
        inverse: bool,
        reduce_output: bool,
    ) {
        // the axis index decomposes as (i_0, ..., i_(r-1)) with i_0 most significant; the
        // transform for factor t acts on i_t, for every combination of the remaining digits.
        // The entry bound is threaded through all factors; the per-(prefix, suffix) sub-calls
        // of one factor make identical (deterministic) reduction decisions, so they all
        // return the same bound. The last factor also brings the output back to [-N, N], by
        // fusing the final reduction (forward) resp. the deferred scaling by len^-1 (inverse)
        // into its last stage.
        let mut bound: u128 = 2 * self.n as u128;
        let mut suffix_len: usize = self.factors.iter().map(|f| f.len()).product();
        let mut prefix_len = 1;
        for (f, factor) in self.factors.iter().enumerate() {
            let is_last_factor = f + 1 == self.factors.len();
            let final_scale = if is_last_factor { Some(self.inv_len) } else { None };
            // only the last forward factor reduces the output, and only if the caller asked
            let fuse_forward = is_last_factor && reduce_output;
            suffix_len /= factor.len();
            let mut new_bound = bound;
            if vec_len == elem_stride {
                // the axis data is one contiguous block per prefix, so the suffix loop can be
                // folded into the vectorized inner dimension: one wide transform per prefix
                // instead of `suffix_len` narrow ones, amortizing the per-call overhead and
                // lengthening the contiguous inner loops
                let inner = suffix_len * elem_stride;
                for prefix in 0..prefix_len {
                    let block = unsafe { data.add(prefix * factor.len() * inner) };
                    new_bound = if inverse {
                        unsafe { factor.backward(block, inner, inner, bound, final_scale) }
                    } else {
                        unsafe { factor.forward(block, inner, inner, bound, fuse_forward) }
                    };
                }
            } else {
                for prefix in 0..prefix_len {
                    for suffix in 0..suffix_len {
                        let offset = (prefix * factor.len() * suffix_len + suffix) * elem_stride;
                        new_bound = if inverse {
                            unsafe {
                                factor.backward(data.add(offset), suffix_len * elem_stride, vec_len, bound, final_scale)
                            }
                        } else {
                            unsafe {
                                factor.forward(data.add(offset), suffix_len * elem_stride, vec_len, bound, fuse_forward)
                            }
                        };
                    }
                }
            }
            prefix_len *= factor.len();
            bound = new_bound;
        }
        debug_assert!(bound <= i64::MAX as u128);
        if self.factors.is_empty() {
            // a length-1 transform is the identity; the inverse still scales by 1^-1 = 1 and
            // the forward still reduces (when requested), both a single reduction pass. When
            // the forward output may stay unreduced, nothing has to be done (the input, bounded
            // by 2N, is already a valid unreduced output)
            if inverse || reduce_output {
                let final_mul = if inverse { self.inv_len } else { self.one };
                for i in 0..self.len {
                    for l in 0..vec_len {
                        unsafe {
                            let x = data.add(i * elem_stride + l);
                            *x = final_mul.mul(*x, self.n);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pos_repr;

    /// 57-bit prime with `q - 1` divisible by `2^7 * 3^4`.
    const Q57: i64 = 144115188075835777;
    /// prime slightly below `2^62` (the largest supported radix-2 modulus size) with
    /// `q - 1` divisible by 16; forces a reduction pass in every radix-2 stage
    const Q62_RADIX2: i64 = 4611686018427379009;
    /// prime slightly below `i64::MAX / 3` (the largest supported radix-3 modulus size) with
    /// `q - 1` divisible by 81; forces frequent radix-3 reduction passes
    const Q61_RADIX3: i64 = 3074457345618256921;

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        return *state >> 33;
    }

    /// A random value in `[-bound, bound]` (the lcg only yields 31 bits per call, so combine
    /// three calls to cover the full range).
    fn random_entry(state: &mut u64, bound: i64) -> i64 {
        let r = ((lcg(state) as u128) << 62) ^ ((lcg(state) as u128) << 31) ^ (lcg(state) as u128);
        return ((r % (2 * bound as u128 + 1)) as i128 - bound as i128) as i64;
    }

    fn forward(ntt: &LazyFactorNTT, data: &mut [i64], in_bound: u128) -> u128 {
        assert_eq!(ntt.len(), data.len());
        return unsafe { ntt.forward(data.as_mut_ptr(), 1, 1, in_bound, false) };
    }

    fn backward(ntt: &LazyFactorNTT, data: &mut [i64], in_bound: u128) -> u128 {
        assert_eq!(ntt.len(), data.len());
        return unsafe { ntt.backward(data.as_mut_ptr(), 1, 1, in_bound, None) };
    }

    #[test]
    fn test_barrett_mul() {
        feanor_tracing::DelayedLogger::init_test();
        for n in [97i64, 65537, Q57, Q62_RADIX2, Q61_RADIX3] {
            let mut state = 1;
            let mut values = vec![0, 1, -1, n - 1, -(n - 1), (n - 1) / 2, -(n - 1) / 2];
            for _ in 0..20 {
                values.push(random_entry(&mut state, n - 1));
            }
            let inputs =
                [0, 1, -1, i64::MAX, i64::MIN, i64::MIN + 1, n, -n, 2 * n.min(i64::MAX / 2), 123456789123456789 % n];
            for z in values {
                let mul = BarrettMul::new(z, n);
                for x in inputs {
                    let out = mul.mul(x, n);
                    assert!(out.abs() <= n, "output {} out of range for modulus {}", out, n);
                    let expected = ((z as i128) * (x as i128)).rem_euclid(n as i128);
                    assert_eq!(expected, (out as i128).rem_euclid(n as i128), "z = {}, x = {}, n = {}", z, x, n);
                }
            }
        }
    }

    /// The forward transform must be the evaluation of the input (as coefficients of a
    /// polynomial modulo `X^len - 1`) at all `len`-th roots of unity, in some order that is a
    /// permutation. The order is recovered from the transform of `X` (whose evaluations are
    /// the roots themselves), then verified against naively evaluated random inputs.
    #[test]
    fn test_forward_is_permuted_dft() {
        feanor_tracing::DelayedLogger::init_test();
        for (q, lens) in
            [(Q57, vec![2usize, 4, 16, 128, 3, 9, 81]), (Q62_RADIX2, vec![2, 4, 16]), (Q61_RADIX3, vec![3, 9, 27, 81])]
        {
            for len in lens {
                let ntt = LazyFactorNTT::new(q, len);
                let root = find_root_of_unity(q, len);
                let roots = (0..len).map(|e| pow_mod(root, e as u64, q)).collect::<Vec<_>>();

                // transform of X: position p holds the root ω^(e_p)
                let mut x_poly = vec![0i64; len];
                x_poly[1] = 1;
                forward(&ntt, &mut x_poly, 2 * q as u128);
                let leaf_exps = x_poly
                    .iter()
                    .map(|out| {
                        roots.iter().position(|r| pos_repr(*out, q) == *r).expect("output is not a root of unity")
                    })
                    .collect::<Vec<_>>();
                let mut seen = vec![false; len];
                for e in &leaf_exps {
                    assert!(!seen[*e], "root ω^{} appears twice in the transform of X", e);
                    seen[*e] = true;
                }

                // random input, evaluated naively at ω^(e_p)
                let mut state = 42 + len as u64;
                let data: Vec<i64> = (0..len).map(|_| random_entry(&mut state, 2 * q)).collect();
                let mut transformed = data.clone();
                let out_bound = forward(&ntt, &mut transformed, 2 * q as u128);
                assert!(out_bound <= i64::MAX as u128);
                for (p, e) in leaf_exps.iter().enumerate() {
                    assert!(transformed[p].abs() as u128 <= out_bound);
                    let mut expected: i128 = 0;
                    for (j, x) in data.iter().enumerate() {
                        expected += (mul_mod(*x % q, roots[(e * j) % len], q) as i128) % q as i128;
                    }
                    assert_eq!(
                        pos_repr(expected, q as i128),
                        pos_repr(transformed[p], q) as i128,
                        "q = {}, len = {}, position {}",
                        q,
                        len,
                        p
                    );
                }
            }
        }
    }

    /// `backward . forward` must be the identity scaled by `len` (the factor-level transforms
    /// defer the division by `len` to the caller), for extreme inputs at the documented bound
    /// `2N` as well as random ones.
    #[test]
    fn test_factor_roundtrip() {
        feanor_tracing::DelayedLogger::init_test();
        for (q, lens) in
            [(Q57, vec![2usize, 4, 16, 128, 3, 9, 81]), (Q62_RADIX2, vec![2, 4, 16]), (Q61_RADIX3, vec![3, 9, 27, 81])]
        {
            for len in lens {
                let ntt = LazyFactorNTT::new(q, len);
                let mut state = 7 + len as u64;
                let mut extreme = vec![2 * q, -2 * q];
                for _ in 2..len {
                    extreme.push([2 * q, -2 * q, 0, q, -q][lcg(&mut state) as usize % 5]);
                }
                let random: Vec<i64> = (0..len).map(|_| random_entry(&mut state, 2 * q)).collect();
                for data in [extreme, random] {
                    let mut transformed = data.clone();
                    let bound = forward(&ntt, &mut transformed, 2 * q as u128);
                    let bound = backward(&ntt, &mut transformed, bound);
                    assert!(bound <= i64::MAX as u128);
                    for pos in 0..len {
                        let expected = ((data[pos] as i128) * (len as i128)).rem_euclid(q as i128);
                        assert_eq!(expected, (transformed[pos] as i128).rem_euclid(q as i128), "position {}", pos);
                    }
                }
            }
        }
    }

    /// The transforms must also work on strided/vectorized data exactly as on contiguous
    /// data.
    #[test]
    fn test_strided_vectorized() {
        feanor_tracing::DelayedLogger::init_test();
        let q = Q57;
        for len in [16usize, 27] {
            let ntt = LazyFactorNTT::new(q, len);
            let (elem_stride, vec_len) = (7usize, 3usize);
            let mut state = 99;
            let mut strided = vec![0i64; len * elem_stride];
            let mut contiguous = vec![vec![0i64; len]; vec_len];
            for i in 0..len {
                for l in 0..vec_len {
                    let value = random_entry(&mut state, 2 * q);
                    strided[i * elem_stride + l] = value;
                    contiguous[l][i] = value;
                }
            }
            unsafe { ntt.forward(strided.as_mut_ptr(), elem_stride, vec_len, 2 * q as u128, false) };
            for column in &mut contiguous {
                forward(&ntt, column, 2 * q as u128);
            }
            for i in 0..len {
                for l in 0..vec_len {
                    assert_eq!(contiguous[l][i], strided[i * elem_stride + l]);
                }
            }
        }

        // the same at the plan level (multi-factor), where `vec_len != elem_stride` takes the
        // non-contiguous fallback of [`CpuNTTPlan::apply()`]; must match the contiguous path
        let plan = CpuNTTPlan::new(q, &[4, 3, 2]);
        let len = 24;
        let (elem_stride, vec_len) = (7usize, 3usize);
        let mut state = 77;
        let mut strided = vec![0i64; len * elem_stride];
        let mut contiguous = vec![vec![0i64; len]; vec_len];
        for i in 0..len {
            for l in 0..vec_len {
                let value = random_entry(&mut state, 2 * q);
                strided[i * elem_stride + l] = value;
                contiguous[l][i] = value;
            }
        }
        for inverse in [false, true] {
            unsafe { plan.apply(strided.as_mut_ptr(), elem_stride, vec_len, inverse, true) };
            for column in &mut contiguous {
                unsafe { plan.apply(column.as_mut_ptr(), 1, 1, inverse, true) };
            }
            for i in 0..len {
                for l in 0..vec_len {
                    assert_eq!(
                        contiguous[l][i],
                        strided[i * elem_stride + l],
                        "i = {}, l = {}, inverse = {}",
                        i,
                        l,
                        inverse
                    );
                }
            }
        }
    }

    /// The plan-level transform must reduce its output into `[-N, N]`, roundtrip correctly
    /// (including the `len^-1` scaling) and diagonalize the multi-factor group convolution.
    /// The shapes cover the fused final stage for both radices, multi-factor plans and
    /// length-1 factors.
    #[test]
    fn test_plan_roundtrip_and_convolution() {
        feanor_tracing::DelayedLogger::init_test();
        for (q, inner_shape) in [
            (Q57, [16usize, 2]),
            (Q57, [27, 1]),
            (Q57, [4, 9]),
            (Q57, [1, 1]),
            (Q62_RADIX2, [16, 2]),
            (Q61_RADIX3, [27, 1]),
        ] {
            let len: usize = inner_shape.iter().product();
            let plan = CpuNTTPlan::new(q, &inner_shape);
            let mut state = 5;

            let a: Vec<i64> = (0..len).map(|_| random_entry(&mut state, 2 * q)).collect();
            let b: Vec<i64> = (0..len).map(|_| random_entry(&mut state, q / 4)).collect();
            let mut expected = vec![0i128; len];
            for i0 in 0..inner_shape[0] {
                for i1 in 0..inner_shape[1] {
                    for j0 in 0..inner_shape[0] {
                        for j1 in 0..inner_shape[1] {
                            let dst = ((i0 + j0) % inner_shape[0]) * inner_shape[1] + (i1 + j1) % inner_shape[1];
                            let product = mul_mod(a[i0 * inner_shape[1] + i1] % q, b[j0 * inner_shape[1] + j1], q);
                            expected[dst] = (expected[dst] + product as i128) % q as i128;
                        }
                    }
                }
            }

            // roundtrip
            let mut data = a.clone();
            unsafe { plan.apply(data.as_mut_ptr(), 1, 1, false, true) };
            for x in &data {
                assert!(x.abs() <= q);
            }
            unsafe { plan.apply(data.as_mut_ptr(), 1, 1, true, true) };
            for pos in 0..len {
                assert!(data[pos].abs() <= q);
                assert_eq!(pos_repr(a[pos], q), pos_repr(data[pos], q));
            }

            // convolution: pointwise products bounded by N * N << 2^63 N would overflow the
            // i64 entries, so multiply pointwise with a naive modular multiplication
            let mut a_hat = a.clone();
            let mut b_hat = b.clone();
            unsafe { plan.apply(a_hat.as_mut_ptr(), 1, 1, false, true) };
            unsafe { plan.apply(b_hat.as_mut_ptr(), 1, 1, false, true) };
            for pos in 0..len {
                a_hat[pos] = mul_mod(a_hat[pos], b_hat[pos], q);
            }
            unsafe { plan.apply(a_hat.as_mut_ptr(), 1, 1, true, true) };
            for pos in 0..len {
                assert_eq!(pos_repr(expected[pos], q as i128), pos_repr(a_hat[pos], q) as i128, "position {}", pos);
            }

            // the unreduced forward output (reduce_output = false) must agree with the reduced
            // one modulo q, and stay within i64::MAX so it remains a valid montgomery operand
            let mut reduced = a.clone();
            let mut unreduced = a.clone();
            unsafe { plan.apply(reduced.as_mut_ptr(), 1, 1, false, true) };
            unsafe { plan.apply(unreduced.as_mut_ptr(), 1, 1, false, false) };
            for pos in 0..len {
                assert_eq!(pos_repr(reduced[pos], q), pos_repr(unreduced[pos], q), "position {}", pos);
            }
        }
    }
}
