use std::iter::repeat_n;
use std::ops::Range;
use std::sync::Mutex;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::matrix::{Submatrix, SubmatrixMut};
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use feanor_math::ring_impls::zn::zn_rns::ZnRNS;
use feanor_math::ring_properties::divisibility::DivisibilityRingStore;
use feanor_math::ring_properties::finite::FiniteRingStore;
use feanor_math::seq::{VectorFn, VectorView};
use rand::*;
use rand_distr::{Distribution, StandardNormal};
use tracing::instrument;

use crate::bconv::BConv;
use crate::fielddesc::field_36864;
use crate::ring::*;
use crate::*;

pub static DebugSK: Mutex<Option<RingElementData<'static>>> = Mutex::new(None);
pub static DebugSparseSK: Mutex<Option<RingElementData<'static>>> = Mutex::new(None);

pub type KSKey = Vec<(Range<usize>, RingElementData<'static>, RingElementData<'static>)>;

/// adds an error term to the given value, in period basis;
/// currently the error is sampled coefficient-wise in period basis
#[instrument(skip_all)]
pub fn add_error<R: CryptoRng>(ring: &RingContext, add_to_pb: &mut [El<Zn64B>], mut rng: R, sigma: f64) {
    let error = (0..ring.n()).map(|_| (Distribution::<f64>::sample(&StandardNormal, &mut rng) * sigma).round() as i32);
    let n = ring.n();
    let used_rns_factors = add_to_pb.len() / n;
    assert_eq!(used_rns_factors * n, add_to_pb.len());
    for (j, e) in error.enumerate() {
        for (i, Zp) in ring.rns_base().as_iter().enumerate().take(used_rns_factors) {
            Zp.add_assign(&mut add_to_pb[i * n + j], Zp.int_hom().map(e));
        }
    }
}

const SK_HWT: usize = 128;

/// generates a secret key, in crt basis;
/// currently, the sk is sampled from the sparse ternary distribution in period basis
#[instrument(skip_all)]
pub fn gen_sk<R: CryptoRng>(ring: &RingContext, mut rng: R) -> RingElementData<'static> {
    let mut result = ring.init_zero();
    let result_data = result.borrow_mut();
    let n = ring.n();
    for _ in 0..SK_HWT {
        let j = rng.next_u32() as usize % n;
        let coeff = (rng.next_u32() % 2) as i32 * 2 - 1;
        for (i, Zp) in ring.rns_base().as_iter().enumerate() {
            result_data[i * n + j] = Zp.int_hom().map(coeff);
        }
    }
    println!();
    println!(
        "secret key log2 caninf-norm {}",
        ring.log2_caninf_norm_small(&result_data)
    );
    ring.pb_to_crt(result_data);
    return result;
}

#[instrument(skip_all)]
pub fn enc_zero_crt<R: CryptoRng>(
    ring: &RingContext,
    sk_crt: &RingElementData,
    mut rng: R,
    sigma: f64,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let n = ring.n();
    let mut a_crt = ring.init_zero();
    let a_crt_data = a_crt.borrow_mut();
    for (i, Zp) in ring.rns_base().as_iter().enumerate() {
        for j in 0..n {
            a_crt_data[i * n + j] = Zp.random_element(|| rng.next_u64());
        }
    }

    let mut b_pb = ring.init_zero();
    let b_pb_data = b_pb.borrow_mut();
    add_error(ring, b_pb_data, rng, sigma);
    ring.pb_to_crt(b_pb_data);

    let mut b_crt = b_pb;
    let b_crt_data = b_crt.borrow_mut();
    ring.fma_crt(a_crt.borrow(), sk_crt.borrow(), b_crt_data);

    ring.mul_scalar(a_crt.borrow_mut(), -1);
    return (b_crt, a_crt);
}

#[instrument(skip_all)]
pub fn enc_crt<R: CryptoRng>(
    ring: &RingContext,
    t: i128,
    ptxt_pb: &[i128],
    sk_crt: &RingElementData,
    mut rng: R,
    sigma: f64,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let n = ring.n();
    assert_eq!(n, ptxt_pb.len());
    let mut a_crt = ring.init_zero();
    let a_crt_data = a_crt.borrow_mut();
    for (i, Zp) in ring.rns_base().as_iter().enumerate() {
        for j in 0..n {
            a_crt_data[i * n + j] = Zp.random_element(|| rng.next_u64());
        }
    }

    let mut b_pb = ring.plaintext_into_ciphertext_ring_scaled_pb(t, ptxt_pb);
    add_error(ring, b_pb.borrow_mut(), rng, sigma);
    ring.pb_to_crt(b_pb.borrow_mut());
    let mut b_crt = b_pb;

    let b_crt_data = b_crt.borrow_mut();
    ring.fma_crt(a_crt.borrow(), sk_crt.borrow(), b_crt_data);

    ring.mul_scalar(a_crt.borrow_mut(), -1);
    return (b_crt, a_crt);
}

#[instrument(skip_all)]
pub fn log2_noise_caninf_norm(
    ring: &RingContext,
    t: i128,
    ct_crt: &(RingElementData, RingElementData),
    sk_crt: &RingElementData,
) -> f64 {
    let ptxt = dec(ring, t, ct_crt.clone(), sk_crt);
    let ptxt_enc = ring.plaintext_into_ciphertext_ring_scaled_pb(t, &ptxt);
    let mut ct0_crt = ct_crt.0.clone();
    let ct0_crt_data = ct0_crt.borrow_mut();
    ring.fma_crt(ct_crt.1.borrow(), sk_crt.borrow(), ct0_crt_data);
    ring.crt_to_pb(ct0_crt_data);
    ring.sub(ptxt_enc.borrow(), ct0_crt_data);
    return ring.log2_caninf_norm(ct0_crt.borrow());
}

#[instrument(skip_all)]
pub fn log2_noise_pbinf_norm(
    ring: &RingContext,
    t: i128,
    ct_crt: &(RingElementData, RingElementData),
    sk_crt: &RingElementData,
) -> f64 {
    let ptxt = dec(ring, t, ct_crt.clone(), sk_crt);
    let ptxt_enc = ring.plaintext_into_ciphertext_ring_scaled_pb(t, &ptxt);
    let mut ct0_crt = ct_crt.0.clone();
    let ct0_crt_data = ct0_crt.borrow_mut();
    ring.fma_crt(ct_crt.1.borrow(), sk_crt.borrow(), ct0_crt_data);
    ring.crt_to_pb(ct0_crt_data);
    ring.sub(ptxt_enc.borrow(), ct0_crt_data);
    return ring.log2_pbinf_norm(ct0_crt.borrow());
}

#[instrument(skip_all)]
pub fn dec(
    ring: &RingContext,
    t: i128,
    mut ct_crt: (RingElementData, RingElementData),
    sk_crt: &RingElementData,
) -> Vec<i128> {
    let n = ring.n();
    let ct0_crt_data = ct_crt.0.borrow_mut();
    let rns_base = if ct0_crt_data.len() == ring.sparse_key_encaps_rns_base().len() * ring.n() {
        ring.sparse_key_encaps_rns_base()
    } else {
        ring.rns_base()
    };
    ring.fma_crt(
        ct_crt.1.borrow(),
        &sk_crt.borrow()[..(n * rns_base.len())],
        ct0_crt_data,
    );
    ring.crt_to_pb(ct0_crt_data);

    return ring.plaintext_from_ciphertext_ring_scaled_pb(t, ct_crt.0);
}

/// input in period basis, output in period basis
#[instrument(skip_all)]
pub fn gadget_decompose_pb<I: IntoIterator<Item = Range<usize>>>(
    n: usize,
    in_rns_base: &ZnRNS<Zn64B, BigIntRing>,
    out_rns_base: &ZnRNS<Zn64B, BigIntRing>,
    el_pb: &RingElementData,
    digits: I,
) -> Vec<RingElementData<'static>> {
    let mut result = Vec::new();
    let el_as_matrix = Submatrix::from_1d(el_pb.borrow(), in_rns_base.len(), n);
    let homs = out_rns_base
        .as_iter()
        .map(|Zp| Zp.can_hom(&ZZi64).unwrap())
        .collect::<Vec<_>>();

    for digit in digits {
        let conversion = BConv::new(
            digit.iter().map(|idx| *in_rns_base.at(idx)).collect::<Vec<_>>(),
            homs.iter().map(|h| **h.codomain()).collect::<Vec<_>>(),
        );
        let mut res = Vec::with_capacity(n * out_rns_base.len());
        res.extend(out_rns_base.as_iter().flat_map(|Zp| repeat_n(Zp.zero(), n)));
        conversion.bconv(
            el_as_matrix.restrict_rows(digit.clone()),
            SubmatrixMut::from_1d(&mut res, out_rns_base.len(), n),
        );
        result.push(RingElementData::Owned(res));
    }
    return result;
}

#[instrument(skip_all)]
pub fn select_digits(digits: usize, rns_base_len: usize) -> Vec<Range<usize>> {
    assert!(
        digits <= rns_base_len,
        "the number of gadget product digits may not exceed the number of RNS factors"
    );
    let moduli_per_small_digit = rns_base_len / digits;
    let large_digits = rns_base_len % digits;
    let small_digits = digits - large_digits;
    let mut result = Vec::with_capacity(digits);
    let mut current = 0;
    for _ in 0..large_digits {
        current += moduli_per_small_digit + 1;
        result.push(current);
    }
    for _ in 0..small_digits {
        current += moduli_per_small_digit;
        result.push(current);
    }
    return [0]
        .into_iter()
        .chain(result)
        .map_windows(|[from, to]| (*from)..(*to))
        .collect();
}

#[instrument(skip_all)]
pub fn enc_zero_small_rns_base<R: CryptoRng>(
    ring: &RingContext,
    sk_crt: &RingElementData,
    mut rng: R,
    sigma: f64,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let n = ring.n();
    let rns_base = ring.sparse_key_encaps_rns_base();
    let mut a_crt = ring.init_zero_with_len(rns_base.len());
    let a_crt_data = a_crt.borrow_mut();
    for (i, Zp) in rns_base.as_iter().enumerate() {
        for j in 0..n {
            a_crt_data[i * n + j] = Zp.random_element(|| rng.next_u64());
        }
    }

    let mut b_pb = ring.init_zero_with_len(rns_base.len());
    let b_pb_data = b_pb.borrow_mut();
    add_error(ring, b_pb_data, rng, sigma);
    ring.pb_to_crt(b_pb_data);

    let mut b_crt = b_pb;
    let b_crt_data = b_crt.borrow_mut();
    ring.fma_crt(a_crt.borrow(), sk_crt.borrow(), b_crt_data);

    ring.mul_scalar(a_crt.borrow_mut(), -1);
    return (b_crt, a_crt);
}

#[instrument(skip_all)]
pub fn gen_ksk_small_rns_base<R: CryptoRng>(
    ring: &RingContext,
    old_sk: &RingElementData,
    new_sk: &RingElementData,
    mut rng: R,
    sigma: f64,
) -> KSKey {
    let digits = (0..ring.sparse_key_encaps_rns_base().len())
        .map(|i| i..(i + 1))
        .collect::<Vec<_>>();
    let rns_base = ring.sparse_key_encaps_rns_base();
    let mut res = Vec::with_capacity(digits.len());
    for digit in digits.into_iter() {
        let (mut c0_crt, c1_crt) = enc_zero_small_rns_base(ring, new_sk, &mut rng, sigma);
        let factor = rns_base.from_congruence((0..rns_base.len()).map(|j| {
            let Fp = rns_base.at(j);
            if digit.contains(&j) { Fp.one() } else { Fp.zero() }
        }));
        let c0_crt_data = c0_crt.borrow_mut();
        ring.fma_scalar_large(old_sk.borrow(), &factor, c0_crt_data);
        res.push((digit, c0_crt, c1_crt));
    }
    return res;
}

#[instrument(skip_all)]
pub fn gen_ksk<I: IntoIterator<Item = Range<usize>>, R: CryptoRng>(
    ring: &RingContext,
    old_sk: &RingElementData,
    new_sk: &RingElementData,
    digits: I,
    mut rng: R,
    sigma: f64,
) -> KSKey {
    let digits = digits.into_iter().collect::<Vec<_>>();
    let rns_base = ring.rns_base();
    let mut res = Vec::with_capacity(digits.len());
    for digit in digits.into_iter() {
        let (mut c0_crt, c1_crt) = enc_zero_crt(ring, new_sk, &mut rng, sigma);
        let factor = rns_base.from_congruence((0..rns_base.len()).map(|j| {
            let Fp = rns_base.at(j);
            if digit.contains(&j) { Fp.one() } else { Fp.zero() }
        }));
        let c0_crt_data = c0_crt.borrow_mut();
        ring.fma_scalar_large(old_sk.borrow(), &factor, c0_crt_data);
        res.push((digit, c0_crt, c1_crt));
    }
    return res;
}

#[instrument(skip_all)]
pub fn gen_rk<I: IntoIterator<Item = Range<usize>>, R: CryptoRng>(
    ring: &RingContext,
    sk: &RingElementData,
    digits: I,
    rng: R,
    sigma: f64,
) -> KSKey {
    let mut sk_sqr = ring.init_zero();
    ring.fma_crt(sk.borrow(), sk.borrow(), sk_sqr.borrow_mut());
    gen_ksk(ring, &sk_sqr, sk, digits, rng, sigma)
}

#[instrument(skip_all)]
pub fn gen_gk<I: IntoIterator<Item = Range<usize>>, R: CryptoRng>(
    ring: &RingContext,
    sk: &RingElementData,
    g: GEl<MultGroup<Zn64B>>,
    digits: I,
    rng: R,
    sigma: f64,
) -> KSKey {
    let shift = ring.field_descriptor().Gal_to_shift(&g);
    let mut gal_sk = ring.init_zero();
    ring.apply_galois(gal_sk.borrow_mut(), sk.borrow(), &shift);
    gen_ksk(ring, &gal_sk, sk, digits, rng, sigma)
}

#[instrument(skip_all)]
pub fn key_switch_crt(
    ring: &RingContext,
    ct0_crt: RingElementData,
    ct1_pb: RingElementData,
    ksk: &KSKey,
    out1_crt_summand: Option<RingElementData>,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let n = ring.n();
    let used_rns_factors = ct0_crt.borrow().len() / n;
    let rns_base = if used_rns_factors == ring.sparse_key_encaps_rns_base().len() {
        ring.sparse_key_encaps_rns_base()
    } else {
        ring.rns_base()
    };
    assert_eq!(n * rns_base.len(), ct0_crt.borrow().len());
    assert_eq!(n * rns_base.len(), ct1_pb.borrow().len());
    let mut ct1_decomp = gadget_decompose_pb(
        ring.n(),
        rns_base,
        rns_base,
        &ct1_pb,
        ksk.iter().map(|(d, ..)| d.clone()),
    );
    let mut res0_crt = ct0_crt.into_owned();
    let res0_crt_data = &mut res0_crt[..];
    let mut res1_crt = if let Some(out1_crt) = out1_crt_summand {
        RingElementData::Owned(out1_crt.into_owned())
    } else {
        ring.init_zero_with_len(rns_base.len())
    };
    let res1_crt_data = res1_crt.borrow_mut();
    for (part, (_, ksk0_crt, ksk1_crt)) in ct1_decomp.iter_mut().zip(ksk.iter()) {
        ring.pb_to_crt(part.borrow_mut());
        ring.fma_crt(part.borrow(), ksk0_crt.borrow(), res0_crt_data);
        ring.fma_crt(part.borrow(), ksk1_crt.borrow(), res1_crt_data);
    }
    return (RingElementData::Owned(res0_crt), res1_crt);
}

/// input in period basis, output in CRT basis
#[instrument(skip_all)]
pub fn lift_to_extended_rns_basis<'a>(
    ring: &'a RingContext,
) -> impl use<'a> + FnMut(&RingElementData, Option<&RingElementData>) -> RingElementData<'static> {
    let conversion = BConv::new(
        ring.rns_base().as_iter().cloned().collect(),
        ring.extended_rns_base()
            .as_iter()
            .skip(ring.rns_base().len())
            .cloned()
            .collect(),
    );
    move |x_pb, x_crt| {
        let mut res = ring.init_zero_extended();
        let res_pb_data = res.borrow_mut();
        conversion.bconv(
            Submatrix::from_1d(x_pb.borrow(), ring.rns_base().len(), ring.n()),
            SubmatrixMut::from_1d(
                &mut res_pb_data[(ring.n() * ring.rns_base().len())..],
                ring.extended_rns_base().len() - ring.rns_base().len(),
                ring.n(),
            ),
        );
        if let Some(x_crt) = x_crt {
            for i in ring.rns_base().len()..ring.extended_rns_base().len() {
                ring.crt_transforms()[i].crt_transform(&mut res_pb_data[(i * ring.n())..((i + 1) * ring.n())]);
            }
            let res_crt_data = res_pb_data;
            res_crt_data[..(ring.n() * ring.rns_base().len())].copy_from_slice(x_crt.borrow());
        } else {
            res_pb_data[..(ring.n() * ring.rns_base().len())].copy_from_slice(x_pb.borrow());
            ring.pb_to_crt(res_pb_data);
        }
        return res;
    }
}

/// input in period basis, output in period basis
#[instrument(skip_all)]
pub fn rescale_to_basis<'a>(
    ring: &'a RingContext,
    t: i128,
) -> impl use<'a> + FnMut(RingElementData) -> RingElementData<'static> {
    let n = ring.n();
    let q_l = ring.rns_base().len();
    let ext_l = ring.extended_rns_base().len() - q_l;
    let q_to_ext = BConv::new(
        ring.rns_base().as_iter().cloned().collect(),
        ring.extended_rns_base().as_iter().skip(q_l).cloned().collect(),
    );
    let ext_to_q = BConv::new(
        ring.extended_rns_base().as_iter().skip(q_l).cloned().collect(),
        ring.rns_base().as_iter().cloned().collect(),
    );
    let mut tmp = Vec::with_capacity(n * ext_l);
    tmp.extend(
        ring.extended_rns_base()
            .as_iter()
            .skip(q_l)
            .flat_map(|Zp| repeat_n(Zp.zero(), n)),
    );

    move |x_pb| {
        let mut x_pb_data = x_pb.into_owned();
        ring.mul_scalar(&mut x_pb_data, t);
        let (x_q_pb_data, x_ext_pb_data) = x_pb_data.split_at_mut(n * q_l);
        q_to_ext.bconv(
            Submatrix::from_1d(&x_q_pb_data, q_l, n),
            SubmatrixMut::from_1d(&mut tmp, ext_l, n),
        );
        for (i, Zp) in ring.extended_rns_base().as_iter().skip(q_l).enumerate() {
            let q_inv = Zp
                .invert(&Zp.coerce(&ZZbig, ring.rns_base().modulus().clone()))
                .unwrap();
            for j in 0..n {
                Zp.sub_assign(&mut x_ext_pb_data[i * n + j], tmp[i * n + j]);
                Zp.mul_assign(&mut x_ext_pb_data[i * n + j], q_inv);
            }
        }
        ext_to_q.bconv(
            Submatrix::from_1d(x_ext_pb_data, ext_l, n),
            SubmatrixMut::from_1d(x_q_pb_data, q_l, n),
        );
        x_pb_data.truncate(q_l * n);
        return RingElementData::Owned(x_pb_data);
    }
}

/// input in period basis, output in period basis
#[instrument(skip_all)]
pub fn rescale_to_sparse_key_encaps_base<'a>(
    ring: &'a RingContext,
    t: i128,
) -> impl use<'a> + FnMut(RingElementData) -> RingElementData<'static> {
    let n = ring.n();
    let q_l = ring.rns_base().len();
    let ext_l = ring.extended_rns_base().len() - q_l;
    let q_to_ext = BConv::new(
        ring.rns_base().as_iter().cloned().collect(),
        ring.extended_rns_base().as_iter().skip(q_l).cloned().collect(),
    );
    let ext_to_q = BConv::new(
        ring.extended_rns_base().as_iter().skip(q_l).cloned().collect(),
        ring.rns_base().as_iter().cloned().collect(),
    );
    let mut tmp = Vec::with_capacity(n * ext_l);
    tmp.extend(
        ring.extended_rns_base()
            .as_iter()
            .skip(q_l)
            .flat_map(|Zp| repeat_n(Zp.zero(), n)),
    );

    move |x_pb| {
        let mut x_pb_data = x_pb.into_owned();
        ring.mul_scalar(&mut x_pb_data, t);
        let (x_q_pb_data, x_ext_pb_data) = x_pb_data.split_at_mut(n * q_l);
        q_to_ext.bconv(
            Submatrix::from_1d(&x_q_pb_data, q_l, n),
            SubmatrixMut::from_1d(&mut tmp, ext_l, n),
        );
        for (i, Zp) in ring.extended_rns_base().as_iter().skip(q_l).enumerate() {
            let q_inv = Zp
                .invert(&Zp.coerce(&ZZbig, ring.rns_base().modulus().clone()))
                .unwrap();
            for j in 0..n {
                Zp.sub_assign(&mut x_ext_pb_data[i * n + j], tmp[i * n + j]);
                Zp.mul_assign(&mut x_ext_pb_data[i * n + j], q_inv);
            }
        }
        ext_to_q.bconv(
            Submatrix::from_1d(x_ext_pb_data, ext_l, n),
            SubmatrixMut::from_1d(x_q_pb_data, q_l, n),
        );
        x_pb_data.truncate(q_l * n);
        return RingElementData::Owned(x_pb_data);
    }
}

/// input in period basis, output in period basis
pub fn mod_switch_to_sparse_key_encaps(
    ring: &RingContext,
    ct_pb: &(RingElementData, RingElementData),
) -> (RingElementData<'static>, RingElementData<'static>) {
    let n = ring.n();
    let small_rns_basis = ring.sparse_key_encaps_rns_base();
    let bconv = BConv::new(
        ring.rns_base().as_iter().skip(small_rns_basis.len()).cloned().collect(),
        small_rns_basis.as_iter().cloned().collect(),
    );
    let mut result = (
        ring.init_zero_with_len(small_rns_basis.len()),
        ring.init_zero_with_len(small_rns_basis.len()),
    );
    bconv.bconv(
        Submatrix::from_1d(ct_pb.0.borrow(), ring.rns_base().len(), n)
            .restrict_rows(small_rns_basis.len()..ring.rns_base().len()),
        SubmatrixMut::from_1d(result.0.borrow_mut(), small_rns_basis.len(), n),
    );
    bconv.bconv(
        Submatrix::from_1d(ct_pb.1.borrow(), ring.rns_base().len(), n)
            .restrict_rows(small_rns_basis.len()..ring.rns_base().len()),
        SubmatrixMut::from_1d(result.1.borrow_mut(), small_rns_basis.len(), n),
    );
    let res0_data = result.0.borrow_mut();
    let res1_data = result.1.borrow_mut();
    let quo = ZZbig
        .checked_div(ring.rns_base().modulus(), small_rns_basis.modulus())
        .unwrap();
    let quo_inv = small_rns_basis.invert(&small_rns_basis.coerce(&ZZbig, quo)).unwrap();
    let quo_inv_congruence = small_rns_basis.get_congruence(&quo_inv);
    for (i, Zp) in small_rns_basis.as_iter().enumerate() {
        let scale = quo_inv_congruence.at(i);
        for j in 0..n {
            res0_data[i * n + j] = Zp.mul_ref_fst(scale, Zp.sub(ct_pb.0.borrow()[i * n + j], res0_data[i * n + j]));
            res1_data[i * n + j] = Zp.mul_ref_fst(scale, Zp.sub(ct_pb.1.borrow()[i * n + j], res1_data[i * n + j]));
        }
    }
    return result;
}

/// input in CRT basis, output in CRT basis
#[instrument(skip_all)]
pub fn hom_square_full_crt(
    ring: &RingContext,
    t: i128,
    val_lft_crt: (RingElementData, RingElementData),
    rk: &KSKey,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let mut lift = lift_to_extended_rns_basis(&ring);
    let mut rescale = rescale_to_basis(&ring, t);
    let mut lifted = |ct: RingElementData| {
        let mut ct_pb = ct.clone();
        ring.crt_to_pb(ct_pb.borrow_mut());
        lift(&ct_pb, Some(&ct))
    };
    let val0_lifted = lifted(val_lft_crt.0);
    let val1_lifted = lifted(val_lft_crt.1);
    let mut res = hom_mul_raw(
        &ring,
        (val0_lifted.clone(), val1_lifted.clone()),
        (val0_lifted, val1_lifted),
    );
    ring.crt_to_pb(res[0].borrow_mut());
    ring.crt_to_pb(res[1].borrow_mut());
    ring.crt_to_pb(res[2].borrow_mut());
    let rescaled = res.map(&mut rescale);
    return relin(&ring, rescaled, &rk);
}

/// input in CRT basis, output in CRT basis
#[instrument(skip_all)]
pub fn hom_mul_full_crt(
    ring: &RingContext,
    t: i128,
    lhs_lft_crt: (RingElementData, RingElementData),
    rhs_lft_crt: (RingElementData, RingElementData),
    rk: &KSKey,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let mut lift = lift_to_extended_rns_basis(&ring);
    let mut rescale = rescale_to_basis(&ring, t);
    let mut lifted = |ct: RingElementData| {
        let mut ct_pb = ct.clone();
        ring.crt_to_pb(ct_pb.borrow_mut());
        lift(&ct_pb, Some(&ct))
    };
    let mut res = hom_mul_raw(
        &ring,
        (lifted(lhs_lft_crt.0), lifted(lhs_lft_crt.1)),
        (lifted(rhs_lft_crt.0), lifted(rhs_lft_crt.1)),
    );
    ring.crt_to_pb(res[0].borrow_mut());
    ring.crt_to_pb(res[1].borrow_mut());
    ring.crt_to_pb(res[2].borrow_mut());
    let rescaled = res.map(&mut rescale);
    return relin(&ring, rescaled, &rk);
}

#[instrument(skip_all)]
pub fn hom_mul_raw(
    ring: &RingContext,
    lhs_lft_crt: (RingElementData, RingElementData),
    rhs_lft_crt: (RingElementData, RingElementData),
) -> [RingElementData<'static>; 3] {
    let n = ring.n();
    let mut lhs0_lft_crt_data = lhs_lft_crt.0.into_owned();
    let mut lhs1_lft_crt_data = lhs_lft_crt.1.into_owned();
    let mut rhs0_lft_crt_data = rhs_lft_crt.0.into_owned();
    let rhs1_lft_crt_data = rhs_lft_crt.1.borrow();
    for (i, Zp) in ring.extended_rns_base().as_iter().enumerate() {
        for j in 0..n {
            let [l0, l1, r0, r1] = [
                lhs0_lft_crt_data[i * n + j],
                lhs1_lft_crt_data[i * n + j],
                rhs0_lft_crt_data[i * n + j],
                rhs1_lft_crt_data[i * n + j],
            ];
            let [o0, o1, o2] = [Zp.mul(l0, r0), Zp.fma(&l1, &r0, Zp.mul(l0, r1)), Zp.mul(l1, r1)];
            lhs0_lft_crt_data[i * n + j] = o0;
            lhs1_lft_crt_data[i * n + j] = o1;
            rhs0_lft_crt_data[i * n + j] = o2;
        }
    }
    return [
        RingElementData::Owned(lhs0_lft_crt_data),
        RingElementData::Owned(lhs1_lft_crt_data),
        RingElementData::Owned(rhs0_lft_crt_data),
    ];
}

#[instrument(skip_all)]
pub fn hom_mul_raw_add(
    ring: &RingContext,
    lhs_lft_crt: &(RingElementData, RingElementData),
    rhs_lft_crt: &(RingElementData, RingElementData),
    out_mulres_crt: [RingElementData; 3],
) -> [RingElementData<'static>; 3] {
    let n = ring.n();
    let mut out_mulres_crt_data = out_mulres_crt.map(RingElementData::into_owned);
    let lhs0_lft_crt_data = lhs_lft_crt.0.borrow();
    let lhs1_lft_crt_data = lhs_lft_crt.1.borrow();
    let rhs0_lft_crt_data = rhs_lft_crt.0.borrow();
    let rhs1_lft_crt_data = rhs_lft_crt.1.borrow();
    for (i, Zp) in ring.extended_rns_base().as_iter().enumerate() {
        for j in 0..n {
            let [l0, l1, r0, r1] = [
                lhs0_lft_crt_data[i * n + j],
                lhs1_lft_crt_data[i * n + j],
                rhs0_lft_crt_data[i * n + j],
                rhs1_lft_crt_data[i * n + j],
            ];
            let [o0, o1, o2] = [Zp.mul(l0, r0), Zp.fma(&l1, &r0, Zp.mul(l0, r1)), Zp.mul(l1, r1)];
            Zp.add_assign(&mut out_mulres_crt_data[0][i * n + j], o0);
            Zp.add_assign(&mut out_mulres_crt_data[1][i * n + j], o1);
            Zp.add_assign(&mut out_mulres_crt_data[2][i * n + j], o2);
        }
    }
    return out_mulres_crt_data.map(RingElementData::Owned);
}

/// input in period basis, output in CRT basis
pub fn relin(
    ring: &RingContext,
    [mut c0_pb, mut c1_pb, c2_pb]: [RingElementData; 3],
    rk: &KSKey,
) -> (RingElementData<'static>, RingElementData<'static>) {
    ring.pb_to_crt(c0_pb.borrow_mut());
    ring.pb_to_crt(c1_pb.borrow_mut());
    return key_switch_crt(&ring, c0_pb, c2_pb, &rk, Some(c1_pb));
}

/// input in CRT basis, output in CRT basis
#[instrument(skip_all)]
pub fn hom_gal(
    ring: &RingContext,
    ctxt_crt: (RingElementData, RingElementData),
    exponents: &[GEl<MultGroup<Zn64B>>],
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> Vec<(RingElementData<'static>, RingElementData<'static>)> {
    let digits = gks[0].1.iter().map(|(d, ..)| d.clone()).collect::<Vec<_>>();
    let GalMQ = ring.field_descriptor().GalMQ();
    let gk = |g| {
        &gks.iter()
            .filter(|(g_, _)| GalMQ.eq_el(g_, &g))
            .next()
            .expect("required galois key not present")
            .1
    };
    if exponents.len() > 1 {
        let mut ct1 = ctxt_crt.1;
        ring.crt_to_pb(ct1.borrow_mut());
        let ct1_decomp = gadget_decompose_pb(ring.n(), ring.rns_base(), ring.rns_base(), &ct1, digits.iter().cloned())
            .into_iter()
            .map(|mut part| {
                ring.pb_to_crt(part.borrow_mut());
                part
            })
            .collect::<Vec<_>>();
        let mut tmp = ring.init_zero();
        let mut results = Vec::new();
        for g in exponents {
            let shift = ring.field_descriptor().Gal_to_shift(&g);
            let mut res0 = ring.init_zero();
            let mut res1 = ring.init_zero();
            ring.apply_galois(res0.borrow_mut(), ctxt_crt.0.borrow(), &shift);
            let gk = gk(*g);
            assert!(digits.iter().eq(gk.iter().map(|(d, ..)| d)));
            for ((_, gk0, gk1), decomp_part) in gk.iter().zip(&ct1_decomp) {
                ring.apply_galois(tmp.borrow_mut(), decomp_part.borrow(), &shift);
                ring.fma_crt(gk0.borrow(), tmp.borrow(), res0.borrow_mut());
                ring.fma_crt(gk1.borrow(), tmp.borrow(), res1.borrow_mut());
            }
            results.push((res0, res1));
        }
        return results;
    } else {
        let g = &exponents[0];
        let shift = ring.field_descriptor().Gal_to_shift(&g);
        let mut res0 = ring.init_zero();
        ring.apply_galois(res0.borrow_mut(), ctxt_crt.0.borrow(), &shift);
        let mut ct1_pb = ctxt_crt.0.to_static();
        ring.apply_galois(ct1_pb.borrow_mut(), ctxt_crt.1.borrow(), &shift);
        ring.crt_to_pb(ct1_pb.borrow_mut());
        let ct1_decomp = gadget_decompose_pb(
            ring.n(),
            ring.rns_base(),
            ring.rns_base(),
            &ct1_pb,
            digits.iter().cloned(),
        )
        .into_iter()
        .map(|mut part| {
            ring.pb_to_crt(part.borrow_mut());
            part
        })
        .collect::<Vec<_>>();
        let gk = gk(*g);
        assert!(digits.iter().eq(gk.iter().map(|(d, ..)| d)));
        let mut res1 = ring.init_zero();
        for ((_, gk0, gk1), decomp_part) in gk.iter().zip(&ct1_decomp) {
            ring.fma_crt(gk0.borrow(), decomp_part.borrow(), res0.borrow_mut());
            ring.fma_crt(gk1.borrow(), decomp_part.borrow(), res1.borrow_mut());
        }
        return vec![(res0, res1)];
    }
}

#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::fielddesc::{field_18, field_192};

pub fn encode_scalar(ring: &RingContext, _t: i128, x: i128) -> Vec<i128> {
    let scale = ring.field_descriptor().scale_Q_to_M_period_basis() as i128;
    (0..ring.n()).map(|_| x * scale).collect()
}

pub fn decode_scalar(ring: &RingContext, t: i128, data_pb: Vec<i128>) -> i128 {
    let val = data_pb[0];
    for i in 1..ring.n() {
        assert_eq!(val, data_pb[i], "ring element doesn't store a scalar");
    }
    let scale = ring.field_descriptor().scale_Q_to_M_period_basis() as i128;
    return cmod(val * scale, t);
}

#[test]
fn test_enc_dec() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 100..120, 8);
    let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 255) - 127).collect();
    let sk = gen_sk(&ring, &mut rng);
    let ct = enc_crt(&ring, 256, &m, &sk, &mut rng, 3.2);
    let res = dec(&ring, 256, ct, &sk);
    assert_eq!(m, res);

    let ring = RingContext::new(field_192(), 100..120, 2);
    let m: Vec<i128> = (0..ring.n()).map(|_| 2).collect();
    let sk = gen_sk(&ring, &mut rng);
    let ct = enc_crt(&ring, 4, &m, &sk, &mut rng, 3.2);
    let res = dec(&ring, 4, ct, &sk);
    assert_eq!(m, res);
}

#[test]
fn test_key_switch() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 100..120, 8);
    let m: Vec<i128> = (0..ring.n()).map(|i| (i as i128 % 255) - 127).collect();
    let sk1 = gen_sk(&ring, &mut rng);
    let sk2 = gen_sk(&ring, &mut rng);
    let mut ct1 = enc_crt(&ring, 256, &m, &sk1, &mut rng, 3.2);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let ksk = gen_ksk(&ring, &sk1, &sk2, digits, rng, 3.2);
    ring.crt_to_pb(ct1.1.borrow_mut());
    let ct2 = key_switch_crt(&ring, ct1.0, ct1.1, &ksk, None);
    let res = dec(&ring, 256, ct2, &sk2);
    assert_eq!(m, res);
}

#[test]
fn test_hom_mul_scalar() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 100..120, 8);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);
    let ct1_crt = enc_crt(&ring, 256, &encode_scalar(&ring, 256, 5), &sk, &mut rng, 3.2);
    let mut ct1_pb = ct1_crt.clone();
    ring.crt_to_pb(ct1_pb.0.borrow_mut());
    ring.crt_to_pb(ct1_pb.1.borrow_mut());
    let ct2_crt = enc_crt(&ring, 256, &encode_scalar(&ring, 256, 7), &sk, &mut rng, 3.2);
    let mut ct2_pb = ct2_crt.clone();
    ring.crt_to_pb(ct2_pb.0.borrow_mut());
    ring.crt_to_pb(ct2_pb.1.borrow_mut());
    let mut lift = lift_to_extended_rns_basis(&ring);
    let mut rescale = rescale_to_basis(&ring, 256);
    let mut res = hom_mul_raw(
        &ring,
        (lift(&ct1_pb.0, Some(&ct1_crt.0)), lift(&ct1_pb.1, Some(&ct1_crt.1))),
        (lift(&ct2_pb.0, Some(&ct2_crt.0)), lift(&ct2_pb.1, Some(&ct2_crt.1))),
    );
    ring.crt_to_pb(res[0].borrow_mut());
    ring.crt_to_pb(res[1].borrow_mut());
    ring.crt_to_pb(res[2].borrow_mut());
    let rescaled = res.map(&mut rescale);
    let res_crt = relin(&ring, rescaled, &rk);
    let res = dec(&ring, 256, res_crt, &sk);
    assert_eq!(encode_scalar(&ring, 256, 35), res);
}

#[test]
fn test_hom_mul_nonscalar() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 100..120, 8);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let m1 = (0..ring.n()).map(|i| i as i128 % 256).collect::<Vec<_>>();
    let m2 = (0..ring.n()).map(|i| (i * i) as i128 % 256).collect::<Vec<_>>();
    let prod = ring.mul_plain_pb(256, &m1, &m2);

    let ct1_crt = enc_crt(&ring, 256, &m1, &sk, &mut rng, 3.2);
    let ct2_crt = enc_crt(&ring, 256, &m2, &sk, &mut rng, 3.2);
    let res_crt = hom_mul_full_crt(&ring, 256, ct1_crt, ct2_crt, &rk);
    let res = dec(&ring, 256, res_crt, &sk);
    assert_eq!(prod, res);
}

#[test]
fn test_hom_mul_large_t() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 64);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len() / 2, ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let m1 = (0..ring.n()).map(|i| (i as i128) << 24).collect::<Vec<_>>();
    let m2 = (0..ring.n()).map(|i| ((i * i) as i128) << 23).collect::<Vec<_>>();
    let prod = ring.mul_plain_pb(1 << 64, &m1, &m2);

    let ct1_crt = enc_crt(&ring, 1 << 64, &m1, &sk, &mut rng, 3.2);
    let ct2_crt = enc_crt(&ring, 1 << 64, &m2, &sk, &mut rng, 3.2);
    let res_crt = hom_mul_full_crt(&ring, 1 << 64, ct1_crt, ct2_crt, &rk);
    println!("{}", log2_noise_pbinf_norm(&ring, 1 << 64, &res_crt, &sk));
    let res = dec(&ring, 1 << 64, res_crt, &sk);
    assert_eq!(prod, res);
}

#[test]
fn test_hom_gal() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_18(), 300..320, 4);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let g = &ring.field_descriptor().GalKQ_cyclic_decomposition()[0].0;
    let gk = gen_gk(&ring, &sk, *g, digits, &mut rng, 3.2);

    let ptxt_pb = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
    let ctxt_crt = enc_crt(&ring, 16, &ptxt_pb, &sk, &mut rng, 3.2);
    let [res] = hom_gal(&ring, ctxt_crt, &[*g], &[(*g, gk)]).try_into().ok().unwrap();
    println!("{}", log2_noise_pbinf_norm(&ring, 1 << 64, &res, &sk));
    let actual = dec(&ring, 16, res, &sk);
    let expected = vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0];
    assert_eq!(&expected, &actual);
}

#[test]
fn test_mod_switch_to_sparse_key_encaps() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_18(), 300..320, 16);
    let sk = gen_sk(&ring, &mut rng);

    let ctxt_crt = enc_crt(&ring, 1 << 16, &encode_scalar(&ring, 1 << 16, 2025), &sk, &mut rng, 3.2);
    let mut ctxt_pb = ctxt_crt;
    ring.crt_to_pb(ctxt_pb.0.borrow_mut());
    ring.crt_to_pb(ctxt_pb.1.borrow_mut());
    let mut modswitched = mod_switch_to_sparse_key_encaps(&ring, &ctxt_pb);
    ring.pb_to_crt(modswitched.0.borrow_mut());
    ring.pb_to_crt(modswitched.1.borrow_mut());
    let res = dec(&ring, 1 << 16, modswitched, &sk);
    assert_eq!(encode_scalar(&ring, 1 << 16, 2025), res);
}
