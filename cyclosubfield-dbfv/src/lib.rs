//! An implementation of BFV and dBFV over non-cyclotomic (abelian, non-Galois-closed) number
//! rings, built on a tensor-based hardware abstraction layer ([`hal`]).
//!
//! - [`field`]: hardcoded descriptions of the supported number fields.
//! - [`ring`]: the RNS ring `R_q = O_L/q`, with the period-basis/CRT-basis transforms.
//! - [`bfv`]: the BFV scheme, including dynamic scaling (mod-switching to smaller RNS bases).
//! - [`dbfv`]: the digit-decomposed dBFV scheme.
//! - [`bootstrap`]: bootstrapping (homomorphic linear transforms and digit extraction).
//! - [`keystore`]: disk backing for key material that exceeds the machine's memory.
#![feature(mapped_lock_guards)]
#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]

pub mod bfv;
pub mod bootstrap;
pub mod dbfv;
pub mod field;
pub mod hal;
pub mod keystore;
pub mod ring;

#[cfg(test)]
use feanor_math::primitive_int::PrimitiveInt;

#[cfg(test)]
pub(crate) fn pos_repr<T: PrimitiveInt, S: PrimitiveInt>(x: T, modulus: S) -> S {
    let modulus_i128 = Into::<i128>::into(modulus);
    let res = (Into::<i128>::into(x) % modulus_i128) + modulus_i128;
    if res >= modulus_i128 { S::try_from(res - modulus_i128).ok().unwrap() } else { S::try_from(res).ok().unwrap() }
}

/// The montgomery representation of `x` modulo `N`, i.e. `pos_repr(x * 2^64 mod N)`.
///
/// The input is reduced modulo `N` before multiplying by `R = 2^64`, so all intermediate
/// products stay bounded by `N^2` and fit into an `i128` (as long as `N < 2^63`).
#[cfg(test)]
pub(crate) fn montgomery_repr<T: PrimitiveInt, S: PrimitiveInt>(x: T, modulus: S) -> S {
    let modulus_i128 = Into::<i128>::into(modulus);
    let x_mod = Into::<i128>::into(x).rem_euclid(modulus_i128);
    let r_mod = (1i128 << 64).rem_euclid(modulus_i128);
    pos_repr(x_mod * r_mod, modulus)
}
