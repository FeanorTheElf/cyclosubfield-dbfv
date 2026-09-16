use std::fmt::Write;

use feanor_math::matrix::{Submatrix, SubmatrixMut};
use feanor_math::prelude::*;
use feanor_math::ring_impls::float_complex::Complex64;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::*;
use feanor_math::seq::VectorView;
use tracing::instrument;

use crate::bconv::BConv;
use crate::bfv::*;
use crate::ring::{RingContext, RingElementData};
use crate::*;

pub type DBFVCiphertext<'a> = Vec<(RingElementData<'a>, RingElementData<'a>)>;

const ZZi128: StaticRing<i128> = StaticRing::RING;

#[derive(Clone, Copy, Debug)]
pub struct DBFVModulus {
    log2_b: usize,
    log2_t: usize,
}

impl DBFVModulus {
    pub fn new(log2_b: usize, digits: usize) -> Self {
        Self {
            log2_b,
            log2_t: digits * log2_b,
        }
    }

    pub fn new_incomplete(log2_b: usize, log2_t: usize) -> Self { Self { log2_b, log2_t } }

    pub fn log2_t(&self) -> usize { self.log2_t }

    pub fn log2_b(&self) -> usize { self.log2_b }

    pub fn t(&self) -> i128 { 1 << self.log2_t }

    pub fn digits(&self) -> usize { (self.log2_t - 1) / self.log2_b + 1 }

    pub fn b(&self) -> i128 { 1 << self.log2_b }
}

#[instrument(skip_all)]
pub fn dbfv_hom_fma_scalar(
    ring: &RingContext,
    modulus: DBFVModulus,
    lhs: &DBFVCiphertext,
    rhs: i128,
    summand: DBFVCiphertext,
) -> DBFVCiphertext<'static> {
    assert!(lhs.len() <= modulus.digits());
    assert!(summand.len() <= modulus.digits());
    let rhs_decomp = (0..modulus.digits())
        .scan(rhs, |current, _| {
            if *current == 0 {
                return None;
            }
            let rem = cmod(*current, modulus.b());
            *current = (*current - rem) / modulus.b();
            return Some(rem);
        })
        .collect::<Vec<_>>();
    debug_assert!(rhs_decomp.len() <= modulus.digits());

    // let mut report = String::new();
    // writeln!(&mut report, "hom_fma_scalar");
    // writeln!(&mut report, "scalar: {:?}", rhs_decomp);
    // write!(&mut report, "lhs err: ");
    // for ct in lhs {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);
    // write!(&mut report, "dst err: ");
    // for ct in &summand {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }

    let mut summand = summand
        .into_iter()
        .map(|ct| (ct.0.to_static(), ct.1.to_static()))
        .collect::<Vec<_>>();

    if lhs.len() == 0 || rhs_decomp.len() == 0 {
        return summand;
    }
    summand.resize_with(
        usize::max(
            summand.len(),
            usize::min(modulus.digits(), lhs.len() + rhs_decomp.len() - 1),
        ),
        || (ring.init_zero(), ring.init_zero()),
    );

    for (i, lhs) in lhs.iter().enumerate() {
        for (j, rhs) in rhs_decomp.iter().enumerate() {
            if i + j < modulus.digits() {
                ring.fma_scalar(lhs.0.borrow(), *rhs, summand[i + j].0.borrow_mut());
                ring.fma_scalar(lhs.1.borrow(), *rhs, summand[i + j].1.borrow_mut());
            }
        }
    }
    // write!(&mut report, "res err: ");
    // for ct in &summand {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // println!();
    // println!("{}", report);
    return summand;
}

#[instrument(skip_all)]
pub fn dbfv_hom_fma_element_Lʼ(
    ring: &RingContext,
    modulus: DBFVModulus,
    lhs: &DBFVCiphertext,
    rhs: &[i128],
    summand: DBFVCiphertext,
) -> DBFVCiphertext<'static> {
    assert!(lhs.len() <= modulus.digits());
    assert!(summand.len() <= modulus.digits());
    let rhs_decomp = (0..modulus.digits())
        .scan(rhs.to_vec(), |current, _| {
            if current.iter().all(|x| *x == 0) {
                return None;
            }
            let rems = current.iter().map(|x| cmod(*x, modulus.b())).collect::<Vec<_>>();
            for (x, y) in current.iter_mut().zip(rems.iter()) {
                *x = (*x - *y) / modulus.b();
            }
            return Some(rems);
        })
        .collect::<Vec<_>>();
    debug_assert!(rhs_decomp.len() <= modulus.digits());

    let mut summand = summand
        .into_iter()
        .map(|ct| (ct.0.to_static(), ct.1.to_static()))
        .collect::<Vec<_>>();

    if lhs.len() == 0 || rhs_decomp.len() == 0 {
        return summand;
    }
    summand.resize_with(
        usize::max(
            summand.len(),
            usize::min(modulus.digits(), lhs.len() + rhs_decomp.len() - 1),
        ),
        || (ring.init_zero(), ring.init_zero()),
    );

    for (i, lhs) in lhs.iter().enumerate() {
        for (j, rhs) in rhs_decomp.iter().enumerate() {
            if i + j < modulus.digits() {
                ring.fma_element_Lʼ(lhs.0.borrow(), &rhs, summand[i + j].0.borrow_mut());
                ring.fma_element_Lʼ(lhs.1.borrow(), &rhs, summand[i + j].1.borrow_mut());
            }
        }
    }
    return summand;
}

#[instrument(skip_all)]
pub fn dbfv_hom_fma_plain(
    ring: &RingContext,
    modulus: DBFVModulus,
    lhs_crt: &DBFVCiphertext,
    rhs: &[i128],
    summand_crt: DBFVCiphertext,
) -> DBFVCiphertext<'static> {
    assert_eq!(ring.n(), rhs.len());
    assert!(lhs_crt.len() <= modulus.digits());
    assert!(summand_crt.len() <= modulus.digits());
    let rhs_decomp = (0..modulus.digits())
        .scan(rhs.to_vec(), |current, _| {
            if current.iter().all(|x| *x == 0) {
                return None;
            }
            let rems = current.iter().map(|x| cmod(*x, modulus.b())).collect::<Vec<_>>();
            for (x, y) in current.iter_mut().zip(rems.iter()) {
                *x = (*x - *y) / modulus.b();
            }
            return Some(rems);
        })
        .collect::<Vec<_>>();
    debug_assert!(rhs_decomp.len() <= modulus.digits());

    let mut summand = summand_crt
        .into_iter()
        .map(|ct| (ct.0.to_static(), ct.1.to_static()))
        .collect::<Vec<_>>();

    if lhs_crt.len() == 0 || rhs_decomp.len() == 0 {
        return summand;
    }
    summand.resize_with(
        usize::min(modulus.digits(), lhs_crt.len() + rhs_decomp.len() - 1),
        || (ring.init_zero(), ring.init_zero()),
    );

    for (j, rhs) in rhs_decomp.iter().enumerate() {
        let mut rhs_in_ring = ring.plaintext_into_ciphertext_ring_unscaled_pb(modulus.t(), rhs);
        ring.pb_to_crt(rhs_in_ring.borrow_mut());
        for (i, lhs) in lhs_crt.iter().enumerate() {
            if i + j < modulus.digits() {
                ring.fma_crt(lhs.0.borrow(), rhs_in_ring.borrow(), summand[i + j].0.borrow_mut());
                ring.fma_crt(lhs.1.borrow(), rhs_in_ring.borrow(), summand[i + j].1.borrow_mut());
            }
        }
    }
    return summand;
}

#[instrument(skip_all)]
pub fn dbfv_exact_div_pow2(
    ring: &RingContext,
    initial_modulus: DBFVModulus,
    val: DBFVCiphertext,
    log2_div: usize,
) -> DBFVCiphertext<'static> {
    assert!(val.len() <= initial_modulus.digits());
    if log2_div == 0 {
        return val.into_iter().map(|ct| (ct.0.to_static(), ct.1.to_static())).collect();
    } else if log2_div > initial_modulus.log2_b() {
        unimplemented!("the case that more than a whole digit is removed is not implemented")
    }

    let mut val_it = val.into_iter();
    let mut res = if let Some(val) = val_it.next() {
        (val.0.to_static(), val.1.to_static())
    } else {
        return Vec::new();
    };
    if let Some(val) = val_it.next() {
        ring.fma_scalar(val.0.borrow(), initial_modulus.b(), res.0.borrow_mut());
        ring.fma_scalar(val.1.borrow(), initial_modulus.b(), res.1.borrow_mut());
    }
    let mut result = vec![res];

    for val in val_it {
        let mut res = (val.0.to_static(), val.1.to_static());
        ring.mul_scalar(res.0.borrow_mut(), initial_modulus.b());
        ring.mul_scalar(res.1.borrow_mut(), initial_modulus.b());
        result.push(res);
    }
    return result;
}

#[instrument(skip_all)]
pub fn dbfv_hom_add(ring: &RingContext, lhs: DBFVCiphertext, rhs: &DBFVCiphertext) -> DBFVCiphertext<'static> {
    let mut lhs: DBFVCiphertext<'static> = lhs
        .into_iter()
        .map(|lhs| (lhs.0.to_static(), lhs.1.to_static()))
        .collect::<Vec<_>>();
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        ring.add(rhs.0.borrow(), lhs.0.borrow_mut());
        ring.add(rhs.1.borrow(), lhs.1.borrow_mut());
    }
    lhs.extend((lhs.len()..rhs.len()).map(|i| (rhs[i].0.clone().to_static(), rhs[i].1.clone().to_static())));
    return lhs;
}

#[instrument(skip_all)]
pub fn dbfv_hom_sub(ring: &RingContext, lhs: DBFVCiphertext, rhs: &DBFVCiphertext) -> DBFVCiphertext<'static> {
    let mut lhs: DBFVCiphertext<'static> = lhs
        .into_iter()
        .map(|lhs| (lhs.0.to_static(), lhs.1.to_static()))
        .collect::<Vec<_>>();
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        ring.sub(rhs.0.borrow(), lhs.0.borrow_mut());
        ring.sub(rhs.1.borrow(), lhs.1.borrow_mut());
    }
    lhs.extend((lhs.len()..rhs.len()).map(|i| {
        let mut res = (rhs[i].0.clone().to_static(), rhs[i].1.clone().to_static());
        ring.mul_scalar(res.0.borrow_mut(), -1);
        ring.mul_scalar(res.1.borrow_mut(), -1);
        return res;
    }));
    return lhs;
}

#[instrument(skip_all)]
pub fn dbfv_log2_noise_pb_norm(
    ring: &RingContext,
    modulus: DBFVModulus,
    ct_crt: &DBFVCiphertext,
    sk_crt: &RingElementData,
) -> Vec<f64> {
    ct_crt
        .iter()
        .map(|ct| log2_noise_pbinf_norm(ring, modulus.t(), ct, sk_crt))
        .collect()
}

#[instrument(skip_all)]
pub fn dbfv_pb_to_crt(ring: &RingContext, ct: &mut DBFVCiphertext) {
    for (ct0, ct1) in ct {
        ring.pb_to_crt(ct0.borrow_mut());
        ring.pb_to_crt(ct1.borrow_mut());
    }
}

#[instrument(skip_all)]
pub fn dbfv_crt_to_pb(ring: &RingContext, ct: &mut DBFVCiphertext) {
    for (ct0, ct1) in ct {
        ring.crt_to_pb(ct0.borrow_mut());
        ring.crt_to_pb(ct1.borrow_mut());
    }
}

#[instrument(skip_all)]
pub fn dbfv_dec(
    ring: &RingContext,
    modulus: DBFVModulus,
    ct_crt: DBFVCiphertext,
    sk_crt: &RingElementData,
) -> Vec<i128> {
    assert!(ct_crt.len() <= modulus.digits());
    let mut result = (0..ring.n()).map(|_| 0).collect::<Vec<_>>();
    for (i, ct) in ct_crt.into_iter().enumerate() {
        let part = dec(ring, modulus.t(), ct, sk_crt);
        for j in 0..ring.n() {
            result[j] += cmod(part[j], 1 << (modulus.log2_t() - i * modulus.log2_b())) << (i * modulus.log2_b());
        }
    }
    for j in 0..ring.n() {
        result[j] = cmod(result[j], modulus.t());
    }
    return result;
}

/// input in period-basis, output in CRT basis
#[instrument(skip_all)]
pub fn dbfv_hom_square_full_crt(
    ring: &RingContext,
    modulus: DBFVModulus,
    val_pb: DBFVCiphertext,
    rk: &KSKey,
) -> DBFVCiphertext<'static> {
    assert!(val_pb.len() <= modulus.digits());
    let val_red_pb = dbfv_reduce(ring, modulus, val_pb);
    if val_red_pb.len() == 0 {
        return val_red_pb;
    }

    // let compute_size = |x: &[Zn64BEl]| {
    //     let mut data = (0..ring.n()).map(|j| ZZbig.rounded_div(ZZbig.mul(int_cast(modulus.t(), ZZbig,
    // ZZi128), ring.rns_base().smallest_lift(ring.rns_base().from_congruence((0..ring.rns_base().
    // len()).map(|i| x[i * ring.n() + j])))), ring.rns_base().modulus())).map(|x|
    // Complex64::RING.from_f64(ZZbig.to_float_approx(&x))).collect::<Vec<_>>();
    //     ring.complex_crt_transform().crt_transform(&mut data);
    //     return data
    //         .iter()
    //         .map(|x| Complex64::RING.abs(*x).log2())
    //         .max_by(f64::total_cmp)
    //         .unwrap();
    // };

    // let mut report = String::new();
    // writeln!(&mut report, "hom_sqr");
    // write!(&mut report, "lhs size: ");
    // for ct in &val_red_pb {
    //     write!(&mut report, "{}, ", compute_size(ct.1.borrow()));
    // }
    // writeln!(&mut report);
    // write!(&mut report, "lhs error: ");
    // for ct in &val_red_pb {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);

    let mut lift = lift_to_extended_rns_basis(ring);
    let mut rescale = rescale_to_basis(ring, modulus.t());
    let val_lift_crt = val_red_pb
        .into_iter()
        .map(|(ct0, ct1)| (lift(&ct0, None), lift(&ct1, None)))
        .collect::<Vec<_>>();

    let result = (0..usize::min(modulus.digits(), val_lift_crt.len() + val_lift_crt.len() - 1))
        .map(|i| {
            let mut out_mulres_crt = [
                ring.init_zero_extended(),
                ring.init_zero_extended(),
                ring.init_zero_extended(),
            ];
            for j in i.saturating_sub(val_lift_crt.len() - 1)..=usize::min(i, val_lift_crt.len() - 1) {
                out_mulres_crt = hom_mul_raw_add(ring, &val_lift_crt[j], &val_lift_crt[i - j], out_mulres_crt);
            }
            ring.crt_to_pb(out_mulres_crt[0].borrow_mut());
            ring.crt_to_pb(out_mulres_crt[1].borrow_mut());
            ring.crt_to_pb(out_mulres_crt[2].borrow_mut());
            return relin(ring, out_mulres_crt.map(&mut rescale), rk);
        })
        .collect::<Vec<_>>();

    // write!(&mut report, "res error: ");
    // for ct_crt in &result {
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);
    // println!();
    // println!("{}", report);
    return result;
}

/// input in period-basis, output in CRT basis
#[instrument(skip_all)]
pub fn dbfv_hom_mul_full_crt(
    ring: &RingContext,
    modulus: DBFVModulus,
    lhs_pb: DBFVCiphertext,
    rhs_pb: DBFVCiphertext,
    rk: &KSKey,
) -> DBFVCiphertext<'static> {
    assert!(lhs_pb.len() <= modulus.digits());
    assert!(rhs_pb.len() <= modulus.digits());
    let lhs_red_pb = dbfv_reduce(ring, modulus, lhs_pb);
    let rhs_red_pb = dbfv_reduce(ring, modulus, rhs_pb);
    if lhs_red_pb.len() == 0 {
        return lhs_red_pb;
    } else if rhs_red_pb.len() == 0 {
        return rhs_red_pb;
    }

    // let compute_size = |x: &[Zn64BEl]| {
    //     let mut data = (0..ring.n()).map(|j| ZZbig.rounded_div(ZZbig.mul(int_cast(modulus.t(), ZZbig,
    // ZZi128), ring.rns_base().smallest_lift(ring.rns_base().from_congruence((0..ring.rns_base().
    // len()).map(|i| x[i * ring.n() + j])))), ring.rns_base().modulus())).map(|x|
    // Complex64::RING.from_f64(ZZbig.to_float_approx(&x))).collect::<Vec<_>>();
    //     ring.complex_crt_transform().crt_transform(&mut data);
    //     return data
    //         .iter()
    //         .map(|x| Complex64::RING.abs(*x).log2())
    //         .max_by(f64::total_cmp)
    //         .unwrap();
    // };

    // let mut report = String::new();
    // writeln!(&mut report, "hom_mul");
    // write!(&mut report, "lhs size: ");
    // for ct in &lhs_red_pb {
    //     write!(&mut report, "{}, ", compute_size(ct.1.borrow()));
    // }
    // writeln!(&mut report);
    // write!(&mut report, "lhs error: ");
    // for ct in &lhs_red_pb {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);
    // write!(&mut report, "rhs size: ");
    // for ct in &rhs_red_pb {
    //     write!(&mut report, "{}, ", compute_size(ct.1.borrow()));
    // }
    // writeln!(&mut report);
    // write!(&mut report, "rhs error: ");
    // for ct in &rhs_red_pb {
    //     let mut ct_crt = ct.clone();
    //     ring.pb_to_crt(ct_crt.0.borrow_mut());
    //     ring.pb_to_crt(ct_crt.1.borrow_mut());
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);

    let mut lift = lift_to_extended_rns_basis(ring);
    let mut rescale = rescale_to_basis(ring, modulus.t());
    let lhs_lift_crt = lhs_red_pb
        .into_iter()
        .map(|(ct0, ct1)| (lift(&ct0, None), lift(&ct1, None)))
        .collect::<Vec<_>>();
    let rhs_lift_crt = rhs_red_pb
        .into_iter()
        .map(|(ct0, ct1)| (lift(&ct0, None), lift(&ct1, None)))
        .collect::<Vec<_>>();

    let result = (0..usize::min(modulus.digits(), lhs_lift_crt.len() + rhs_lift_crt.len() - 1))
        .map(|i| {
            let mut out_mulres_crt = [
                ring.init_zero_extended(),
                ring.init_zero_extended(),
                ring.init_zero_extended(),
            ];
            for j in i.saturating_sub(rhs_lift_crt.len() - 1)..=usize::min(i, lhs_lift_crt.len() - 1) {
                out_mulres_crt = hom_mul_raw_add(ring, &lhs_lift_crt[j], &rhs_lift_crt[i - j], out_mulres_crt);
            }
            ring.crt_to_pb(out_mulres_crt[0].borrow_mut());
            ring.crt_to_pb(out_mulres_crt[1].borrow_mut());
            ring.crt_to_pb(out_mulres_crt[2].borrow_mut());
            return relin(ring, out_mulres_crt.map(&mut rescale), rk);
        })
        .collect::<Vec<_>>();

    // write!(&mut report, "res error: ");
    // for ct_crt in &result {
    //     write!(&mut report, "{}, ",  log2_noise_caninf_norm(ring, modulus.t(), &ct_crt,
    // DebugSK.lock().unwrap().as_ref().unwrap())); }
    // writeln!(&mut report);
    // println!();
    // println!("{}", report);
    return result;
}

/// Performs the dBFV-reduction, i.e. reduces every digit (a BFV ciphertext)
/// modulo `q/p^(e(d - 1))` and propagates the carry to the next digit.
/// All overflows and digits above `d` are dropped. The full plaintext
/// modulus is thus `t = p^(ed)`.
#[instrument(skip_all)]
pub fn dbfv_reduce(ring: &RingContext, modulus: DBFVModulus, mut ct_pb: DBFVCiphertext) -> DBFVCiphertext<'static> {
    assert!(ct_pb.len() <= modulus.digits());
    if ct_pb.len() == 0 {
        return Vec::new();
    }
    let mut split_overflow = split_overflow_pb(ring, modulus);
    if modulus.digits() == 1 {
        let (ct0, ct1) = ct_pb.remove(0);
        return vec![(split_overflow(ct0).0, split_overflow(ct1).0)];
    }
    let n = ring.n();
    let Delta = ring.rns_base().coerce(
        &ZZbig,
        ZZbig.rounded_div(ring.rns_base().modulus().clone(), &int_cast(modulus.t(), ZZbig, ZZi128)),
    );
    let Delta_congruence = ring.rns_base().get_congruence(&Delta);
    let mut result = Vec::new();
    let mut overflow: Option<(Vec<i64>, Vec<i64>)> = None;
    for (mut ct0, mut ct1) in ct_pb.into_iter().take(modulus.digits()) {
        let ct0_data = ct0.borrow_mut();
        let ct1_data = ct1.borrow_mut();
        if let Some((o0, o1)) = overflow {
            for (i, Zp) in ring.rns_base().as_iter().enumerate() {
                let hom = Zp.can_hom(&ZZi64).unwrap();
                let coeff = Delta_congruence.at(i);
                for j in 0..n {
                    ct0_data[i * n + j] = Zp.fma(coeff, &hom.map(o0[j]), ct0_data[i * n + j]);
                    ct1_data[i * n + j] = Zp.fma(coeff, &hom.map(o1[j]), ct1_data[i * n + j]);
                }
            }
        }
        let (new0, overflow0) = split_overflow(ct0);
        let (new1, overflow1) = split_overflow(ct1);
        result.push((new0, new1));
        overflow = Some((overflow0, overflow1));
    }
    let mut overflow = if let Some(overflow) = overflow {
        overflow
    } else {
        return result;
    };
    while result.len() < modulus.digits() {
        let mut ct0 = ring.init_zero();
        let mut ct1 = ring.init_zero();
        let ct0_data = ct0.borrow_mut();
        let ct1_data = ct1.borrow_mut();
        let (o0, o1) = overflow;
        for (i, Zp) in ring.rns_base().as_iter().enumerate() {
            let hom = Zp.can_hom(&ZZi64).unwrap();
            let coeff = Delta_congruence.at(i);
            for j in 0..n {
                ct0_data[i * n + j] = Zp.fma(coeff, &hom.map(o0[j]), ct0_data[i * n + j]);
                ct1_data[i * n + j] = Zp.fma(coeff, &hom.map(o1[j]), ct1_data[i * n + j]);
            }
        }
        let (new0, overflow0) = split_overflow(ct0);
        let (new1, overflow1) = split_overflow(ct1);
        result.push((new0, new1));
        overflow = (overflow0, overflow1);
    }

    return result;
}

/// Computes the function `x -> x_lo, x_hi` where `x_lo + p^e Δ x_hi = x`,
/// and `|x_lo| <= p^e Δ`. As usual, `Δ = round(q/t)`
///
/// input in period-basis, output in period-basis
#[instrument(skip_all)]
pub fn split_overflow_pb<'a>(
    ring: &'a RingContext,
    modulus: DBFVModulus,
) -> impl use<'a> + FnMut(RingElementData) -> (RingElementData<'static>, Vec<i64>) {
    let n = ring.n();
    let q_to_R = BConv::new(
        ring.rns_base().as_iter().cloned().collect(),
        vec![Zn64B::new(u64::try_from(modulus.t() / modulus.b()).unwrap())],
    );
    let ZR = &q_to_R.output_rings()[0];
    let R: i128 = modulus.t() / modulus.b();
    let Delta = ring.rns_base().coerce(
        &ZZbig,
        ZZbig.rounded_div(ring.rns_base().modulus().clone(), &int_cast(modulus.t(), ZZbig, ZZi128)),
    );
    let scale = ring
        .rns_base()
        .neg(ring.rns_base().can_hom(&ZZi128).unwrap().mul_map(Delta, modulus.b()));
    let neg_q_inv_mod_R = ZR.neg(
        ZR.invert(&ZR.coerce(&ZZbig, ring.rns_base().modulus().clone()))
            .unwrap(),
    );
    move |x_pb| {
        let ZR = &q_to_R.output_rings()[0];
        let mut x_R_pb = x_pb.clone();
        ring.mul_scalar(x_R_pb.borrow_mut(), R);
        let mut overflow = (0..ring.n()).map(|_| ZR.zero()).collect::<Vec<_>>();
        q_to_R.bconv(
            Submatrix::from_1d(x_R_pb.borrow(), ring.rns_base().len(), ring.n()),
            SubmatrixMut::from_1d(&mut overflow, 1, ring.n()),
        );
        for x in &mut overflow {
            ZR.mul_assign(x, neg_q_inv_mod_R);
        }
        let overflow = overflow.into_iter().map(|x| ZR.smallest_lift(x)).collect::<Vec<_>>();

        let mut res_pb = x_pb.to_static();
        let res_pb_data = res_pb.borrow_mut();
        let scale_congruence = ring.rns_base().get_congruence(&scale);
        for (i, Zp) in ring.rns_base().as_iter().enumerate() {
            let scale_coeff = scale_congruence.at(i);
            let hom = Zp.can_hom(&ZZi64).unwrap();
            for j in 0..ring.n() {
                res_pb_data[i * n + j] = Zp.fma(&hom.map(overflow[j]), scale_coeff, res_pb_data[i * n + j])
            }
        }

        return (res_pb, overflow);
    }
}

#[cfg(test)]
use std::cmp::Ordering;

#[cfg(test)]
use rand::rngs::StdRng;
#[cfg(test)]
use rand::{Rng, SeedableRng};

#[cfg(test)]
use crate::fielddesc::field_192;

#[test]
fn test_split_overflow_pb() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 16);
    let n = ring.n();
    let mut x_pb = ring.init_zero();
    let x_pb_data = x_pb.borrow_mut();
    for (i, Zp) in ring.rns_base().as_iter().enumerate() {
        for j in 0..n {
            x_pb_data[i * n + j] = Zp.random_element(|| rng.next_u64());
        }
    }

    let Delta = ZZbig.rounded_div(ring.rns_base().modulus().clone(), &ZZbig.power_of_two(24));
    let scale = ZZbig.mul(Delta, ZZbig.power_of_two(8));
    let bound = ZZbig.int_hom().mul_ref_map(&scale, &2);
    let mut split_overflow = split_overflow_pb(&ring, DBFVModulus::new(8, 3));
    let (y_pb, overflow) = split_overflow(x_pb.clone());
    for j in 0..n {
        let x_coeff = ring.rns_base().smallest_lift(
            ring.rns_base()
                .from_congruence((0..ring.rns_base().len()).map(|i| x_pb.borrow()[i * n + j])),
        );
        let y_coeff = ring.rns_base().smallest_lift(
            ring.rns_base()
                .from_congruence((0..ring.rns_base().len()).map(|i| y_pb.borrow()[i * n + j])),
        );
        let overflow_coeff = int_cast(overflow[j], ZZbig, ZZi64);

        assert_el_eq!(
            ZZbig,
            x_coeff,
            ZZbig.euclidean_rem(
                ZZbig.add_ref_fst(&y_coeff, ZZbig.mul_ref(&overflow_coeff, &scale)),
                ring.rns_base().modulus()
            )
        );
        assert!(
            ZZbig.abs_cmp(&y_coeff, &bound) == Ordering::Less,
            "expected {}-th coefficient {} to be < 2 q/R = 2 * {}/{} ~ {}",
            j,
            ZZbig.formatted_el(&y_coeff),
            ZZbig.formatted_el(ring.rns_base().modulus()),
            ZZbig.formatted_el(&ZZbig.power_of_two(16)),
            ZZbig.formatted_el(&bound)
        );
    }
}

#[test]
fn test_dbfv_enc_dec() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 64);
    let sk = gen_sk(&ring, &mut rng);
    let ct_crt = enc_crt(
        &ring,
        1 << 64,
        &encode_scalar(&ring, 1 << 64, 5 << 31),
        &sk,
        &mut rng,
        3.2,
    );
    let result = decode_scalar(
        &ring,
        1 << 64,
        dbfv_dec(&ring, DBFVModulus::new(16, 4), vec![ct_crt], &sk),
    );
    assert_eq!(5 << 31, result);
}

#[test]
fn test_dbfv_reduce() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 64);
    let sk = gen_sk(&ring, &mut rng);
    let mut ct = vec![enc_crt(
        &ring,
        1 << 64,
        &encode_scalar(&ring, 1 << 64, 5 << 31),
        &sk,
        &mut rng,
        3.2,
    )];
    dbfv_crt_to_pb(&ring, &mut ct);
    let mut ct = dbfv_reduce(&ring, DBFVModulus::new(16, 4), ct.clone());
    dbfv_pb_to_crt(&ring, &mut ct);

    let result = decode_scalar(&ring, 1 << 64, dbfv_dec(&ring, DBFVModulus::new(16, 4), ct, &sk));
    assert_eq!(5 << 31, result);
}

#[test]
fn test_dbfv_mul() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 64);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);
    let ct1_crt = enc_crt(
        &ring,
        1 << 64,
        &encode_scalar(&ring, 1 << 64, 5 << 31),
        &sk,
        &mut rng,
        3.2,
    );
    let mut ct1_pb = ct1_crt.clone();
    ring.crt_to_pb(ct1_pb.0.borrow_mut());
    ring.crt_to_pb(ct1_pb.1.borrow_mut());
    let ct2_crt = enc_crt(
        &ring,
        1 << 64,
        &encode_scalar(&ring, 1 << 64, 7 << 31),
        &sk,
        &mut rng,
        3.2,
    );
    let mut ct2_pb = ct2_crt.clone();
    ring.crt_to_pb(ct2_pb.0.borrow_mut());
    ring.crt_to_pb(ct2_pb.1.borrow_mut());
    let product = dbfv_hom_mul_full_crt(&ring, DBFVModulus::new(16, 4), vec![ct1_pb], vec![ct2_pb], &rk);
    let result = decode_scalar(&ring, 1 << 64, dbfv_dec(&ring, DBFVModulus::new(16, 4), product, &sk));
    assert_eq!(cmod(35 << 62, 1 << 64), result);
}

#[test]
fn test_dbfv_square() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 64);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);
    let ct_crt = enc_crt(
        &ring,
        1 << 64,
        &encode_scalar(&ring, 1 << 64, 5 << 31),
        &sk,
        &mut rng,
        3.2,
    );
    let mut ct_pb = ct_crt.clone();
    ring.crt_to_pb(ct_pb.0.borrow_mut());
    ring.crt_to_pb(ct_pb.1.borrow_mut());
    let product = dbfv_hom_square_full_crt(&ring, DBFVModulus::new(16, 4), vec![ct_pb], &rk);
    let result = decode_scalar(&ring, 1 << 64, dbfv_dec(&ring, DBFVModulus::new(16, 4), product, &sk));
    assert_eq!(cmod(25 << 62, 1 << 64), result);
}

#[test]
fn test_dbfv_exact_div_pow2() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 32);
    let sk = gen_sk(&ring, &mut rng);
    let mut ct = vec![enc_crt(
        &ring,
        1 << 32,
        &encode_scalar(&ring, 1 << 32, 5 << 16),
        &sk,
        &mut rng,
        3.2,
    )];
    dbfv_crt_to_pb(&ring, &mut ct);
    ct = dbfv_reduce(&ring, DBFVModulus::new(8, 4), ct);
    ct = dbfv_exact_div_pow2(&ring, DBFVModulus::new(8, 4), ct, 5);
    dbfv_pb_to_crt(&ring, &mut ct);
    let result = decode_scalar(
        &ring,
        1 << 27,
        dbfv_dec(&ring, DBFVModulus::new_incomplete(8, 27), ct, &sk),
    );
    assert_eq!(5 << 11, result);
}
