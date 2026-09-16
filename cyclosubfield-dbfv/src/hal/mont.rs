use feanor_math::pid::PrincipalIdealRingStore;
use feanor_math::primitive_int::StaticRing;

#[derive(Debug, Clone, Copy)]
pub struct MontgomeryModulus {
    N: i64,
    neg_inv_N: i64,
    R_sqr_mod_N: i64,
    R_cbe_mod_N: i64,
}

impl MontgomeryModulus {
    pub(crate) fn new(N: i64) -> Self {
        assert!(N > 1);
        let (s, _, d) = StaticRing::<i128>::RING.extended_ideal_gen(&(N as i128), &(1 << 64));
        assert!(d == 1 || d == -1);
        let pmod = |x: i128| {
            let mut res = (x % N as i128) + N as i128;
            if res > N as i128 {
                res -= N as i128;
            }
            return res as i64;
        };
        let R_mod_N = pmod(1 << 64) as i128;
        let R_sqr_mod_N = pmod(R_mod_N * R_mod_N);
        return Self {
            N,
            // -N^(-1) is needed modulo 2^64 (not modulo N), so truncate to the low 64 bits
            neg_inv_N: (-s * d) as i64,
            R_sqr_mod_N,
            R_cbe_mod_N: pmod((R_sqr_mod_N as i128) * pmod(1 << 64) as i128),
        };
    }

    /// The modulus `N`.
    pub(crate) fn modulus(&self) -> i64 { self.N }

    /// Given an input `x` with `|x| <= 2^63 * N + N`, returns an output that is in `[-N, N]` and
    /// congruent to `2^(-64) x mod N`.
    #[inline(always)]
    pub(crate) fn red128(&self, x: i128) -> i64 {
        let m = (x as i64).wrapping_mul(self.neg_inv_N);
        let t = (m as i128) * (self.N as i128);
        return ((x + t) >> 64) as i64;
    }

    /// Works for any 64-bit input, returns an output that is in `[-N, N]` and
    /// congruent to `2^(-64) x mod N`.
    #[inline(always)]
    #[allow(unused)]
    fn red64(&self, x: i64) -> i64 {
        let m = x.wrapping_mul(self.neg_inv_N);
        let t = (m as i128) * (self.N as i128);
        return (((x as i128) + t) >> 64) as i64;
    }

    /// Given any `i128` input `x`, returns an output that is in `[-N, N]` and
    /// congruent to `2^(-128) x mod N`.
    #[inline(always)]
    pub(crate) fn double_red128(&self, x: i128) -> i64 {
        let m = (x as i64).wrapping_mul(self.neg_inv_N);
        let t = (m as i128) * (self.N as i128);
        let carry = (((x as u64 as u128) + (t as u64 as u128)) >> 64) as i128;
        let r1 = (x >> 64) + (t >> 64) + carry;
        self.red128(r1)
    }

    /// Computes the montgomery representation of any 64-bit integer.
    #[inline(always)]
    pub(crate) fn to_mont64(&self, x: i64) -> i64 { self.red128((x as i128) * (self.R_sqr_mod_N as i128)) }

    /// Given an input `x` with `|x| <= 2^63 * N + N`, returns its montgomery representation.
    #[inline(always)]
    pub(crate) fn to_mont128(&self, x: i128) -> i64 {
        self.red128((self.red128(x) as i128) * (self.R_cbe_mod_N as i128))
    }
}
