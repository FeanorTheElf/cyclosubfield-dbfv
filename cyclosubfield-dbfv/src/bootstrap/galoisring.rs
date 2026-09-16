//! Minimal arithmetic in Galois rings `GR(2^e, D) = Z_(2^e)[y] / (g(y))` (for a monic lift
//! `g` of an irreducible degree-`D` polynomial over `F_2`), plus Gaussian elimination modulo
//! `2^e`. This is only used during the setup of the homomorphic CRT transform (see
//! [`crate::bootstrap::homcrt`]), so simple schoolbook algorithms suffice.

/// polynomials over `F_2` as bit vectors (bit `i` of the vector = coefficient of `y^i`),
/// supporting arbitrary degrees
mod f2 {
    pub type Poly = Vec<u64>;

    pub fn from_bit(i: usize) -> Poly {
        let mut result = vec![0u64; i / 64 + 1];
        result[i / 64] = 1 << (i % 64);
        return result;
    }

    pub fn one() -> Poly { vec![1] }

    pub fn is_zero(a: &Poly) -> bool { a.iter().all(|w| *w == 0) }

    pub fn get_bit(a: &Poly, i: usize) -> bool { i / 64 < a.len() && (a[i / 64] >> (i % 64)) & 1 == 1 }

    pub fn degree(a: &Poly) -> i64 {
        for (i, w) in a.iter().enumerate().rev() {
            if *w != 0 {
                return (i * 64 + 63 - w.leading_zeros() as usize) as i64;
            }
        }
        return -1;
    }

    fn xor_shifted(dst: &mut Poly, src: &Poly, shift: usize) {
        let (words, bits) = (shift / 64, shift % 64);
        if dst.len() < src.len() + words + 1 {
            dst.resize(src.len() + words + 1, 0);
        }
        for (i, w) in src.iter().enumerate() {
            if *w != 0 {
                dst[i + words] ^= w << bits;
                if bits > 0 {
                    dst[i + words + 1] ^= w >> (64 - bits);
                }
            }
        }
    }

    pub fn mul(a: &Poly, b: &Poly) -> Poly {
        let mut result = vec![0u64; a.len() + b.len() + 1];
        for i in 0..=degree(a).max(0) as usize {
            if get_bit(a, i) {
                xor_shifted(&mut result, b, i);
            }
        }
        return result;
    }

    pub fn rem(a: &Poly, g: &Poly) -> Poly {
        let dg = degree(g);
        assert!(dg >= 0);
        let mut a = a.clone();
        while degree(&a) >= dg {
            let shift = (degree(&a) - dg) as usize;
            xor_shifted(&mut a, g, shift);
        }
        // normalize the length, so that equal polynomials have equal representations
        a.resize((dg as usize) / 64 + 1, 0);
        return a;
    }

    pub fn mul_mod(a: &Poly, b: &Poly, g: &Poly) -> Poly { rem(&mul(a, b), g) }

    /// `base^e mod g`, with the exponent given as a bit vector (little endian).
    pub fn pow_mod(base: &Poly, e: &[u64], g: &Poly) -> Poly {
        let mut result = rem(&one(), g);
        let mut base = rem(base, g);
        for i in 0..64 * e.len() {
            if (e[i / 64] >> (i % 64)) & 1 == 1 {
                result = mul_mod(&result, &base, g);
            }
            base = mul_mod(&base, &base, g);
        }
        return result;
    }

    fn gcd(a: &Poly, b: &Poly) -> Poly {
        let (mut a, mut b) = (a.clone(), b.clone());
        while !is_zero(&b) {
            let r = rem(&a, &b);
            a = b;
            b = r;
        }
        return a;
    }

    fn xor(a: &Poly, b: &Poly) -> Poly {
        let mut result = a.clone();
        if result.len() < b.len() {
            result.resize(b.len(), 0);
        }
        for (i, w) in b.iter().enumerate() {
            result[i] ^= w;
        }
        return result;
    }

    /// Whether `g` is irreducible of degree `d`, via the standard criterion
    /// `y^(2^d) = y mod g` and `gcd(y^(2^(d/r)) - y, g) = 1` for all primes `r | d`.
    fn is_irreducible(g: &Poly, d: usize) -> bool {
        let y = rem(&from_bit(1), g);
        let frobenius_power = |k: usize| {
            let mut current = y.clone();
            for _ in 0..k {
                current = mul_mod(&current, &current, g);
            }
            current
        };
        if frobenius_power(d) != y {
            return false;
        }
        let mut rest = d;
        let mut r = 2;
        while rest > 1 {
            if rest % r == 0 {
                while rest % r == 0 {
                    rest /= r;
                }
                if degree(&gcd(&xor(&frobenius_power(d / r), &y), g)) != 0 {
                    return false;
                }
            }
            r += 1;
        }
        return true;
    }

    /// Finds an irreducible polynomial of degree `d` over `F_2`, deterministically.
    pub fn find_irreducible(d: usize) -> Poly {
        assert!(d >= 1);
        if d == 1 {
            return from_bit(1);
        }
        // iterate over polynomials y^d + (candidate) with odd candidate; an irreducible one
        // of low weight is found quickly
        for candidate in (1u64..).step_by(2) {
            let mut g = from_bit(d);
            g[0] ^= candidate;
            if is_irreducible(&g, d) {
                return g;
            }
        }
        unreachable!()
    }

    /// The inverse of `a` modulo `g` (`g` irreducible of degree `d`), via Fermat:
    /// `a^(2^d - 2)`.
    pub fn inv_mod(a: &Poly, g: &Poly) -> Poly {
        let d = degree(g) as usize;
        assert!(!is_zero(a));
        // exponent 2^d - 2 as a bit vector
        let mut e = vec![0u64; (d - 1) / 64 + 1];
        for i in 1..d {
            e[i / 64] |= 1 << (i % 64);
        }
        return pow_mod(a, &e, g);
    }
}

/// The inverse of an odd `x` modulo `2^128`, by Newton iteration.
fn inv_mod_2_128(x: i128) -> i128 {
    assert!(x & 1 == 1);
    let mut result: i128 = x; // correct mod 8
    for _ in 0..6 {
        result = result.wrapping_mul(2i128.wrapping_sub(x.wrapping_mul(result)));
    }
    return result;
}

/// The Galois ring `GR(2^e, D)`; elements are coefficient vectors of length `D` with entries
/// in `[0, 2^e)`.
pub struct GaloisRing {
    e: usize,
    d: usize,
    /// the irreducible degree-`d` polynomial over `F_2` (with 0/1-lifted coefficients, `g` is
    /// the modulus of the ring)
    g: f2::Poly,
    mask: i128,
}

pub type GrEl = Vec<i128>;

impl GaloisRing {
    pub fn new(e: usize, d: usize) -> Self {
        assert!(e >= 1 && e <= 100);
        assert!(d >= 1);
        return Self { e, d, g: f2::find_irreducible(d), mask: (1i128 << e) - 1 };
    }

    pub fn e(&self) -> usize { self.e }

    pub fn d(&self) -> usize { self.d }

    pub fn zero(&self) -> GrEl { vec![0; self.d] }

    pub fn one(&self) -> GrEl {
        let mut result = self.zero();
        result[0] = 1;
        return result;
    }

    pub fn from_scalar(&self, x: i128) -> GrEl {
        let mut result = self.zero();
        result[0] = x & self.mask;
        return result;
    }

    pub fn add(&self, lhs: &GrEl, rhs: &GrEl) -> GrEl {
        lhs.iter().zip(rhs).map(|(a, b)| (a + b) & self.mask).collect()
    }

    pub fn sub(&self, lhs: &GrEl, rhs: &GrEl) -> GrEl {
        lhs.iter().zip(rhs).map(|(a, b)| (a - b) & self.mask).collect()
    }

    pub fn mul_scalar(&self, lhs: &GrEl, rhs: i128) -> GrEl {
        lhs.iter().map(|a| a.wrapping_mul(rhs) & self.mask).collect()
    }

    pub fn mul(&self, lhs: &GrEl, rhs: &GrEl) -> GrEl {
        let mut product = vec![0i128; 2 * self.d - 1];
        for (i, a) in lhs.iter().enumerate() {
            if *a != 0 {
                for (j, b) in rhs.iter().enumerate() {
                    product[i + j] = (product[i + j] + a.wrapping_mul(*b)) & self.mask;
                }
            }
        }
        // reduce by the monic polynomial g (with 0/1 coefficients)
        for i in (self.d..2 * self.d - 1).rev() {
            let coeff = product[i];
            if coeff != 0 {
                product[i] = 0;
                for j in 0..self.d {
                    if f2::get_bit(&self.g, j) {
                        product[i - self.d + j] = (product[i - self.d + j] - coeff) & self.mask;
                    }
                }
            }
        }
        product.truncate(self.d);
        return product;
    }

    pub fn pow(&self, base: &GrEl, mut e: u128) -> GrEl {
        let mut result = self.one();
        let mut base = base.clone();
        while e > 0 {
            if e & 1 == 1 {
                result = self.mul(&result, &base);
            }
            base = self.mul(&base, &base);
            e >>= 1;
        }
        return result;
    }

    pub fn is_zero(&self, x: &GrEl) -> bool { x.iter().all(|c| *c == 0) }

    pub fn eq(&self, lhs: &GrEl, rhs: &GrEl) -> bool { lhs == rhs }

    fn to_f2(&self, x: &GrEl) -> f2::Poly {
        let mut result = vec![0u64; (self.d - 1) / 64 + 1];
        for (i, c) in x.iter().enumerate() {
            result[i / 64] |= ((c & 1) as u64) << (i % 64);
        }
        return result;
    }

    fn from_f2(&self, x: &f2::Poly) -> GrEl { (0..self.d).map(|i| f2::get_bit(x, i) as i128).collect() }

    /// The inverse of `x` (which must be a unit, i.e. nonzero modulo 2): invert in `F_(2^d)`
    /// and Hensel-lift.
    pub fn inv(&self, x: &GrEl) -> GrEl {
        let x_mod_2 = self.to_f2(x);
        assert!(!f2::is_zero(&x_mod_2), "element is not a unit");
        let mut result = self.from_f2(&f2::inv_mod(&x_mod_2, &self.g));
        // Newton: r <- r (2 - x r), doubling the 2-adic precision each step
        for _ in 0..7 {
            let two_minus = self.sub(&self.from_scalar(2), &self.mul(x, &result));
            result = self.mul(&result, &two_minus);
        }
        return result;
    }

    /// Finds an element of order exactly `n` (an odd prime with `n | 2^d - 1`), i.e. a
    /// primitive `n`-th root of unity: constructed in `F_(2^d)` and Hensel-lifted to satisfy
    /// `x^n = 1` exactly.
    pub fn root_of_unity(&self, n: usize) -> GrEl {
        assert!(n % 2 == 1);
        if n == 1 {
            return self.one();
        }
        // cofactor = (2^d - 1) / n as a bit vector, by binary long division of 11...1 by n
        let mut cofactor = vec![0u64; (self.d - 1) / 64 + 1];
        let mut remainder: u128 = 0;
        for i in (0..self.d).rev() {
            remainder = (remainder << 1) | 1;
            if remainder >= n as u128 {
                remainder -= n as u128;
                cofactor[i / 64] |= 1 << (i % 64);
            }
        }
        assert_eq!(0, remainder, "no root of unity of order {} in F_2^{}", n, self.d);
        let n_bits = [n as u64];
        let mut root_f2 = f2::one();
        for candidate in 2usize.. {
            // small deterministic starting elements: y^k + 1
            let mut start = f2::from_bit(candidate);
            start[0] ^= 1;
            let x = f2::pow_mod(&start, &cofactor, &self.g);
            if f2::degree(&x) > 0 && f2::degree(&f2::pow_mod(&x, &n_bits, &self.g)) == 0 {
                root_f2 = x;
                break;
            }
        }
        let mut root = self.from_f2(&root_f2);
        // Hensel: root <- root - (root^n - 1) / (n root^(n-1))
        for _ in 0..8 {
            let power = self.pow(&root, n as u128 - 1);
            let numerator = self.sub(&self.mul(&power, &root), &self.one());
            if self.is_zero(&numerator) {
                break;
            }
            let denominator = self.mul_scalar(&power, n as i128);
            root = self.sub(&root, &self.mul(&numerator, &self.inv(&denominator)));
        }
        assert!(self.eq(&self.pow(&root, n as u128), &self.one()));
        return root;
    }

    /// The smallest (balanced) lift of a coefficient.
    pub fn smallest_lift(&self, c: i128) -> i128 {
        let c = c & self.mask;
        if c > self.mask / 2 { c - self.mask - 1 } else { c }
    }
}

/// Solves `matrix * sol = rhs` modulo `2^e`, where `matrix` is a (row-major) `rows x cols`
/// matrix and `rhs` consists of `rhs_cols` (row-major) columns. Requires the system to be
/// uniquely solvable modulo 2, which for our use is guaranteed since the columns of `matrix`
/// are part of a basis of a direct summand. Returns the `cols x rhs_cols` solution.
pub fn solve_mod_2e(
    e: usize,
    rows: usize,
    cols: usize,
    rhs_cols: usize,
    matrix: &mut [i128],
    rhs: &mut [i128],
) -> Vec<i128> {
    assert!(rows >= cols);
    let mask = (1i128 << e) - 1;
    let mut pivot_rows: Vec<usize> = Vec::with_capacity(cols);
    for col in 0..cols {
        // find a row (not yet used as pivot) with a unit (odd) entry in this column
        let pivot = (0..rows)
            .find(|row| !pivot_rows.contains(row) && matrix[row * cols + col] & 1 == 1)
            .expect("matrix is not invertible modulo 2");
        let inv = inv_mod_2_128(matrix[pivot * cols + col]) & mask;
        for j in 0..cols {
            matrix[pivot * cols + j] = matrix[pivot * cols + j].wrapping_mul(inv) & mask;
        }
        for j in 0..rhs_cols {
            rhs[pivot * rhs_cols + j] = rhs[pivot * rhs_cols + j].wrapping_mul(inv) & mask;
        }
        for row in 0..rows {
            if row != pivot {
                let factor = matrix[row * cols + col];
                if factor != 0 {
                    for j in 0..cols {
                        matrix[row * cols + j] =
                            (matrix[row * cols + j] - factor.wrapping_mul(matrix[pivot * cols + j])) & mask;
                    }
                    for j in 0..rhs_cols {
                        rhs[row * rhs_cols + j] =
                            (rhs[row * rhs_cols + j] - factor.wrapping_mul(rhs[pivot * rhs_cols + j])) & mask;
                    }
                }
            }
        }
        pivot_rows.push(pivot);
    }
    // consistency: all non-pivot rows must have a zero right-hand side now
    for row in 0..rows {
        if !pivot_rows.contains(&row) {
            for j in 0..rhs_cols {
                assert_eq!(0, rhs[row * rhs_cols + j] & mask, "linear system is not solvable");
            }
        }
    }
    let mut solution = vec![0i128; cols * rhs_cols];
    for (col, pivot) in pivot_rows.iter().enumerate() {
        for j in 0..rhs_cols {
            solution[col * rhs_cols + j] = rhs[pivot * rhs_cols + j] & mask;
        }
    }
    return solution;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galois_ring_basic() {
        feanor_tracing::DelayedLogger::init_test();
        let gr = GaloisRing::new(20, 12);
        let a = (0..12).map(|i| (i * i * 37 + 5) as i128).collect::<Vec<_>>();
        let inv = gr.inv(&a);
        assert!(gr.eq(&gr.one(), &gr.mul(&a, &inv)));

        // 13 | 2^12 - 1 = 4095
        let root = gr.root_of_unity(13);
        assert!(gr.eq(&gr.one(), &gr.pow(&root, 13)));
        assert!(!gr.eq(&gr.one(), &root));
        // the powers sum to zero
        let mut sum = gr.zero();
        for i in 0..13 {
            sum = gr.add(&sum, &gr.pow(&root, i));
        }
        assert!(gr.is_zero(&sum));
    }

    /// large-degree Galois rings (more than 64 bits worth of polynomial coefficients) as
    /// needed by `field_192`, where `ord_1777(2) = 74`
    #[test]
    fn test_galois_ring_large_degree() {
        feanor_tracing::DelayedLogger::init_test();
        let gr = GaloisRing::new(4, 74);
        let root = gr.root_of_unity(1777);
        assert!(gr.eq(&gr.one(), &gr.pow(&root, 1777)));
        assert!(!gr.eq(&gr.one(), &root));
        let inv = gr.inv(&root);
        assert!(gr.eq(&gr.one(), &gr.mul(&root, &inv)));
    }

    #[test]
    fn test_solve_mod_2e() {
        feanor_tracing::DelayedLogger::init_test();
        let e = 10;
        // matrix [[1, 2], [3, 5], [4, 6]] * [x, y] = known rhs
        let mut matrix = vec![1, 2, 3, 5, 4, 6];
        let x = 17i128;
        let y = 900i128;
        let mask = (1i128 << e) - 1;
        let mut rhs = vec![(x + 2 * y) & mask, (3 * x + 5 * y) & mask, (4 * x + 6 * y) & mask];
        let sol = solve_mod_2e(e, 3, 2, 1, &mut matrix, &mut rhs);
        assert_eq!(vec![x & mask, y & mask], sol);
    }
}
