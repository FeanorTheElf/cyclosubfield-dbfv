#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![feature(allocator_api)]
#![feature(iter_array_chunks)]
#![feature(iter_order_by)]
#![feature(iter_map_windows)]

use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::group::*;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;

mod bconv;
mod bfv;
mod bootstrap;
mod crt;
mod dbfv;
mod digitex;
mod fft;
mod fielddesc;
mod homcrt;
mod matmul;
mod ntt;
mod ring;

pub const ZZi128: StaticRing<i128> = StaticRing::RING;

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

pub fn pmod(a: i64, modulus: usize) -> usize {
    assert!(modulus >= 1);
    let result = ((a % modulus as i64) + modulus as i64) as usize;
    if result >= modulus { result - modulus } else { result }
}

pub fn log3_exact(order: usize) -> usize {
    if order == 1 {
        return 0;
    }
    let (p, e) = is_prime_power(ZZi64, &(order as i64)).unwrap();
    assert_eq!(3, p);
    e
}

pub fn log2_exact(order: usize) -> usize {
    let e = ZZi64.abs_log2_ceil(&(order as i64)).unwrap();
    assert_eq!(1 << e, order);
    e
}

pub fn group_linear_combine<'a, I: IntoIterator<Item = (&'a GEl<MultGroup<Zn64B>>, i64)>>(
    G: &MultGroup<Zn64B>,
    combination: I,
) -> GEl<MultGroup<Zn64B>> {
    combination.into_iter().fold(G.identity(), |current, next| {
        G.op(current, G.pow(next.0.clone(), next.1))
    })
}

fn main() {}
