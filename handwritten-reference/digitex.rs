use std::collections::HashMap;

use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use fhe_ir::GenericInstruction::*;
use fhe_ir::Program;
use tracing::{Level, instrument, span};

use crate::ZZi128;
use crate::bfv::*;
use crate::dbfv::*;
use crate::ring::{RingContext, RingElementData};

/// Homomorphically computes the base-2 digit extraction
/// `Z/2^e -> (Z/2^eZ)^l`, where the `i`-th output is equal to
/// the lowest bit of the input modulo `2^min(2^i, e)` and
/// `l = ceil(log2(e))`. The results are subtraced from the
/// corresponding values in `subtract_from_crt`, and the final
/// result (i.e. the lowest bit of the input mod `2^e`) is
/// returned.
#[instrument(skip_all)]
pub fn hom_digit_extract_subtract<'a, I: IntoIterator<Item = (&'a mut [El<Zn64B>], &'a mut [El<Zn64B>])>>(
    ring: &RingContext,
    e: usize,
    ct_crt: (RingElementData, RingElementData),
    subtract_from_crt: I,
    rk: &KSKey,
) -> (RingElementData<'static>, RingElementData<'static>) {
    let mut results = hom_digit_extract_bfv(ring, e, ct_crt, rk);
    assert_eq!(ZZi64.abs_log2_ceil(&(e as i64)).unwrap() + 1, results.len());
    let mut subtract_from_it = subtract_from_crt.into_iter();
    for x in subtract_from_it.by_ref().take(1) {
        assert!(e > 1);
        ring.sub(results[1].0.borrow(), x.0);
        ring.sub(results[1].1.borrow(), x.1);
    }
    for x in subtract_from_it.by_ref().take(2) {
        assert!(e > 2);
        ring.sub(results[2].0.borrow(), x.0);
        ring.sub(results[2].1.borrow(), x.1);
    }
    for x in subtract_from_it.by_ref().take(4) {
        assert!(e > 4);
        ring.sub(results[3].0.borrow(), x.0);
        ring.sub(results[3].1.borrow(), x.1);
    }
    for x in subtract_from_it.by_ref().take(8) {
        assert!(e > 8);
        ring.sub(results[4].0.borrow(), x.0);
        ring.sub(results[4].1.borrow(), x.1);
    }
    for x in subtract_from_it.by_ref().take(7) {
        assert!(e > 16);
        ring.sub(results[5].0.borrow(), x.0);
        ring.sub(results[5].1.borrow(), x.1);
    }
    return results.pop().unwrap();
}

/// Outputs are `x2, x4, x8, x16, x23` where `xe` is congruent to `lift(x mod 2)` modulo `2^e`
const DIGIT_DECOMPOSE_PROGRAM: &'static str = r#"
    func(%0) {
        return %0
        %1 = mul %0, %0
        return %1
        %2 = mul %1, %1
        return %2
        %3 = inner_prod %1, %2, coefficients = [@0, @1]
        %4 = mul %3, %3
        %11 = inner_prod %1, %4, coefficients = [@16, @17]
        return %11
        %5 = inner_prod %1, %2, %4, coefficients = [@2, @3, @4]
        %6 = inner_prod %1, %2, %4, coefficients = [@5, @6, @7]
        %7 = mul %5, %6
        %12 = inner_prod %1, %2, %4, %7, coefficients = [@18, @19, @20, @21]
        return %12
        %8 = inner_prod %1, %2, %4, %7, coefficients = [@8, @9, @10, @11]
        %9 = inner_prod %1, %2, %4, %7, coefficients = [@12, @13, @14, @15]
        %10 = mul %8, %9
        %13 = inner_prod %1, %2, %4, %7, %10, coefficients = [@22, @23, @24, @25, @26]
        return %13
    }
    @0: 94
    @1: 121
    @10: 19902061394
    @11: 879825
    @12: 1922883005169
    @13: 400982412
    @14: 17155160050
    @15: 758385
    @16: 112
    @17: 1
    @18: 2535424
    @19: 528
    @2: 12656
    @20: 22620
    @21: 1
    @22: 16641840644096
    @23: 3469149649
    @24: 148471549968
    @25: 6563608
    @26: 1
    @3: 226
    @4: 113
    @5: 33720
    @6: 2
    @7: 301
    @8: 2230777446111
    @9: 472843148"#;

/// Outputs are `x2, x4, x8, x80` where `xe` is congruent to `lift(x mod 2)` modulo `2^e`
const DIGIT_EXTRACT_PROGRAM: &'static str = r#"
    func(%0) {
        %1 = mul %0, %0
        %2 = mul %1, %1
        %3 = inner_prod %1, %2, coefficients = [@0, @1]
        %4 = mul %3, %3
        %5 = inner_prod %1, %2, %4, coefficients = [@2, @3, @4]
        %6 = inner_prod %1, %2, %4, coefficients = [@5, @6, @7]
        %7 = mul %5, %6
        %8 = inner_prod %1, %2, %4, %7, coefficients = [@8, @9, @10, @11]
        %9 = inner_prod %1, %2, %4, %7, coefficients = [@12, @13, @14, @15]
        %10 = mul %8, %9
        %11 = inner_prod %1, %2, %4, %7, %10, coefficients = [@16, @17, @18, @19, @20]
        %12 = mul %11, %11
        %13 = inner_prod %1, %2, %4, %7, %10, %12, coefficients = [@21, @22, @23, @24, @25, @26]
        %14 = mul %13, %12
        %15 = inner_prod %1, %2, %4, %7, %10, %12, %14, coefficients = [@27, @28, @29, @30, @31, @32, @33]
        %16 = add_ptx %15, @34
        %17 = inner_prod %1, %2, %4, %7, %10, %12, %14, coefficients = [@35, @36, @37, @38, @39, @40, @41]
        %18 = mul %16, %17
        %19 = inner_prod %1, %4, coefficients = [@42, @43]
        %20 = inner_prod %1, %2, %4, %7, %10, %12, %18, coefficients = [@44, @45, @46, @47, @48, @49, @50]
        return %1
        return %2
        return %19
        return %20
    }
    @0: 94
    @1: 121
    @10: 88914
    @11: -168751
    @12: 142065
    @13: 426380
    @14: 456690
    @15: -290191
    @16: 0
    @17: 460241
    @18: -520176
    @19: 272152
    @2: 12656
    @20: 1
    @21: 0
    @22: -32106558060291480110540
    @23: 64784965186867951517504
    @24: -70466562361570490438944
    @25: 403089028126664267137012
    @26: 1
    @27: 0
    @28: -306594109877109866124476
    @29: 55340232221151557156928
    @3: 226
    @30: 66482065641637241715808
    @31: -201964177491009025911804
    @32: 982289121925033623516
    @33: 1
    @34: 152453116397172589318468
    @35: 0
    @36: -541458055423539766375756
    @37: -321268494787747983014080
    @38: 389373873907872951793376
    @39: -168833825134626671122316
    @4: 113
    @40: 982289121925033623520
    @41: 1
    @42: 112
    @43: 1
    @44: 0
    @45: 138276793813981485937712
    @46: -394317601587992658717952
    @47: 277439031009004544298112
    @48: -62423781945432606733008
    @49: 376442706312192718108336
    @5: 33720
    @50: 1
    @6: 2
    @7: 301
    @8: 163551
    @9: -64628
    "#;

#[instrument(skip_all)]
pub fn hom_digit_extract_bfv<'a>(
    ring: &RingContext,
    e: usize,
    ct_crt: (RingElementData, RingElementData),
    rk: &KSKey,
) -> Vec<(RingElementData<'static>, RingElementData<'static>)> {
    assert!(e >= 1);
    assert!(e <= 23);
    let program = Program::parse_check(DIGIT_DECOMPOSE_PROGRAM.as_bytes()).unwrap();
    return evaluate_polynomial_bfv(
        &program,
        ring,
        1 << e,
        vec![ct_crt],
        ZZi64.abs_log2_ceil(&(e as i64)).unwrap() + 1,
        rk,
    );
}

#[instrument(skip_all)]
pub fn hom_digit_extract_dbfv<'a>(
    ring: &RingContext,
    modulus: DBFVModulus,
    extract_digit_e: usize,
    ct_crt: DBFVCiphertext,
    rk: &KSKey,
) -> Vec<DBFVCiphertext<'static>> {
    assert!(extract_digit_e >= 1);
    assert!(extract_digit_e <= 23);
    let program = Program::parse_check(DIGIT_DECOMPOSE_PROGRAM.as_bytes()).unwrap();
    return evaluate_polynomial_dbfv(
        &program,
        ring,
        modulus,
        vec![ct_crt],
        ZZi64.abs_log2_ceil(&(extract_digit_e as i64)).unwrap() + 1,
        rk,
    );
}

#[instrument(skip_all)]
pub fn hom_floordiv_dbfv<'a>(
    ring: &RingContext,
    initial_modulus: DBFVModulus,
    final_modulus: DBFVModulus,
    ct_crt: DBFVCiphertext,
    rk: &KSKey,
) -> DBFVCiphertext<'static> {
    let v = initial_modulus.log2_t() - final_modulus.log2_t();
    assert_eq!(final_modulus.log2_b(), initial_modulus.log2_b());
    let log2_b = final_modulus.log2_b();

    if v > 8 {
        unimplemented!("current IR only supports eliminating up to 8 digits");
    } else if initial_modulus.log2_t() > 80 {
        unimplemented!("current IR only supports precision up to 80 bits");
    }
    let program: Program<i128> = Program::parse_check(DIGIT_EXTRACT_PROGRAM.as_bytes()).unwrap();

    let mut result = ct_crt
        .iter()
        .map(|ct| (ct.0.clone().to_static(), ct.1.clone().to_static()))
        .collect::<Vec<_>>();
    let mut parts = (0..v).map(|_| ct_crt.clone()).collect::<Vec<_>>();
    for i in 0..v {
        let mut parts_it = parts.into_iter().fuse();
        let next = dbfv_exact_div_pow2(ring, initial_modulus, parts_it.next().unwrap(), i);
        let current_modulus = DBFVModulus::new_incomplete(log2_b, initial_modulus.log2_t() - i);

        // let noises = dbfv_log2_noise_pb_norm(ring, current_modulus, &next,
        // DebugSK.lock().unwrap().as_ref().unwrap()); let values =
        // crt_transform.unwrap().reduce_t(current_modulus.t()).extract_slot_values(ring, &dbfv_dec(ring,
        // current_modulus, next.clone(), DebugSK.lock().unwrap().as_ref().unwrap())); println!();
        // println!("stage {} begin: {:?}", i, noises);
        // println!("{:?}", &values[..20]);

        let to_subtract = evaluate_polynomial_dbfv(&program, ring, current_modulus, vec![next], usize::MAX, rk);
        let last = to_subtract.last().unwrap();

        // let noises = dbfv_log2_noise_pb_norm(ring, current_modulus, &last,
        // DebugSK.lock().unwrap().as_ref().unwrap()); let values =
        // crt_transform.unwrap().reduce_t(current_modulus.t()).extract_slot_values(ring, &dbfv_dec(ring,
        // current_modulus, last.clone(), DebugSK.lock().unwrap().as_ref().unwrap())); println!();
        // println!("stage {} end: {:?}", i, noises);
        // println!("{:?}", &values[..20]);

        result = dbfv_hom_sub(ring, result, &last);

        // let values = crt_transform.unwrap().reduce_t(current_modulus.t()).extract_slot_values(ring,
        // &dbfv_dec(ring, initial_modulus, result.clone(), DebugSK.lock().unwrap().as_ref().unwrap()));
        // println!();
        // println!("{:?}", &values[..20]);

        let mut new_parts = Vec::new();
        for (subtract, e) in to_subtract.iter().zip([2, 4, 8]) {
            for lhs in parts_it.by_ref().take(e / 2) {
                new_parts.push(dbfv_hom_sub(ring, lhs, &subtract));
            }
        }
        assert!(parts_it.next().is_none());
        parts = new_parts;
    }
    let result = dbfv_exact_div_pow2(ring, initial_modulus, result, v);
    // let values = crt_transform.unwrap().reduce_t(final_modulus.t()).extract_slot_values(ring,
    // &dbfv_dec(ring, final_modulus, result.clone(), DebugSK.lock().unwrap().as_ref().unwrap()));
    // println!();
    // println!("{:?}", &values[..20]);
    return result;
}

fn evaluate_polynomial_bfv(
    program: &Program<i128>,
    ring: &RingContext,
    t: i128,
    ctxts_crt: Vec<(RingElementData, RingElementData)>,
    until_outputs: usize,
    rk: &KSKey,
) -> Vec<(RingElementData<'static>, RingElementData<'static>)> {
    assert_eq!(ctxts_crt.len(), program.inputs().len());
    let mut data = HashMap::new();
    for (input, value) in program.inputs().zip(ctxts_crt) {
        data.insert(input, value);
    }
    let Delta = ring.rns_base().coerce(
        &ZZbig,
        ZZbig.rounded_div(ring.rns_base().modulus().clone(), &int_cast(t, ZZbig, ZZi128)),
    );
    let mut results = Vec::new();
    for inst in program.instructions_with_data() {
        #[allow(unused_variables)]
        match inst {
            InnerProduct {
                out,
                values,
                coefficients,
            } => {
                let mut res0 = ring.init_zero();
                let mut res1 = ring.init_zero();
                for (coeff, val) in coefficients.iter().zip(values) {
                    let coeff = **coeff;
                    let val = data.get(val).unwrap();
                    ring.fma_scalar(val.0.borrow(), coeff, res0.borrow_mut());
                    ring.fma_scalar(val.1.borrow(), coeff, res1.borrow_mut());
                }
                _ = data.insert(out, (res0, res1));
            }
            AddPtxCtx {
                out,
                value: val,
                plaintext,
            } => {
                let mut val = data.get(val).unwrap().clone();
                let coeff = ring
                    .rns_base()
                    .mul_ref_snd(ring.rns_base().coerce(&ZZi128, *plaintext), &Delta);
                ring.add_scalar_large(&coeff, val.0.borrow_mut());
                _ = data.insert(out, val);
            }
            AddCtxCtx { out, lhs, rhs } => {
                let lhs = data.get(lhs).unwrap();
                let rhs = data.get(rhs).unwrap();
                let mut res = lhs.clone();
                ring.add(rhs.0.borrow(), res.0.borrow_mut());
                ring.add(rhs.1.borrow(), res.1.borrow_mut());
                _ = data.insert(out, res);
            }
            MulCtxCtx { out, lhs, rhs } => {
                if lhs == rhs {
                    let val = data.get(lhs).unwrap().clone();
                    let res = hom_square_full_crt(ring, t, val, rk);
                    _ = data.insert(out, res);
                } else {
                    let lhs = data.get(lhs).unwrap().clone();
                    let rhs = data.get(rhs).unwrap().clone();
                    let res = hom_mul_full_crt(ring, t, lhs, rhs, rk);
                    _ = data.insert(out, res);
                }
            }
            Return { val } => {
                let val = data.get(val).unwrap();
                results.push((val.0.clone().to_static(), val.1.clone().to_static()));
                if results.len() == until_outputs {
                    return results;
                }
            }
            MulPtxCtx { out, value, plaintext } => unreachable!(),
            Galois { out, val, exponents } => unreachable!(),
            Zero { out } => unreachable!(),
            MulIntCtx { out, value, integer } => unreachable!(),
            Copy { out, val } => unreachable!(),
        }
    }
    return results;
}

fn evaluate_polynomial_dbfv(
    program: &Program<i128>,
    ring: &RingContext,
    modulus: DBFVModulus,
    ctxts_crt: Vec<DBFVCiphertext>,
    until_outputs: usize,
    rk: &KSKey,
) -> Vec<DBFVCiphertext<'static>> {
    assert_eq!(ctxts_crt.len(), program.inputs().len());

    let mut data = HashMap::new();
    for (input, mut value) in program.inputs().zip(ctxts_crt) {
        dbfv_crt_to_pb(ring, &mut value);
        data.insert(input, value);
    }
    let Delta = ring.rns_base().coerce(
        &ZZbig,
        ZZbig.rounded_div(ring.rns_base().modulus().clone(), &int_cast(modulus.t(), ZZbig, ZZi128)),
    );
    let mut results = Vec::new();
    for inst in program.instructions_with_data() {
        #[allow(unused_variables)]
        match inst {
            InnerProduct {
                out,
                values,
                coefficients,
            } => span!(Level::INFO, "inner_product").in_scope(|| {
                let mut res = Vec::new();
                for (val, coeff) in values.iter().zip(coefficients) {
                    let val = data.get(val).unwrap();
                    res = dbfv_hom_fma_scalar(ring, modulus, val, *coeff, res);
                }
                _ = data.insert(out, res);
            }),
            AddPtxCtx {
                out,
                value: val,
                plaintext,
            } => span!(Level::INFO, "add_ptxt_ctxt").in_scope(|| {
                let mut val = data.get(val).unwrap().clone();
                let scale = ring.field_descriptor().scale_Q_to_M_period_basis() as i128;
                let coeff = ring
                    .rns_base()
                    .mul_ref_snd(ring.rns_base().coerce(&ZZi128, plaintext * scale), &Delta);
                ring.add_scalar_large(&coeff, val[0].0.borrow_mut());
                _ = data.insert(out, val);
            }),
            AddCtxCtx { out, lhs, rhs } => span!(Level::INFO, "add_ctxt_ctxt").in_scope(|| {
                let lhs = data.get(lhs).unwrap().clone();
                let rhs = data.get(rhs).unwrap();
                let res = dbfv_hom_add(ring, lhs, rhs);
                _ = data.insert(out, res);
            }),
            MulCtxCtx { out, lhs, rhs } => span!(Level::INFO, "mul_ctxt_ctxt").in_scope(|| {
                if lhs == rhs {
                    let val = data.get(lhs).unwrap().clone();
                    let mut res = dbfv_hom_square_full_crt(ring, modulus, val, rk);
                    dbfv_crt_to_pb(ring, &mut res);
                    _ = data.insert(out, res);
                } else {
                    let lhs = data.get(lhs).unwrap().clone();
                    let rhs = data.get(rhs).unwrap().clone();
                    let mut res = dbfv_hom_mul_full_crt(ring, modulus, lhs, rhs, rk);
                    dbfv_crt_to_pb(ring, &mut res);
                    _ = data.insert(out, res);
                }
            }),
            Return { val } => {
                let val = data.get(val).unwrap();
                let mut val: Vec<(RingElementData<'_>, RingElementData<'_>)> = val
                    .iter()
                    .map(|ct| (ct.0.clone().to_static(), ct.1.clone().to_static()))
                    .collect();
                dbfv_pb_to_crt(ring, &mut val);
                results.push(val);
                if results.len() == until_outputs {
                    return results;
                }
            }
            MulPtxCtx { out, value, plaintext } => unreachable!(),
            Galois { out, val, exponents } => unreachable!(),
            Zero { out } => unreachable!(),
            MulIntCtx { out, value, integer } => unreachable!(),
            Copy { out, val } => unreachable!(),
        }
    }
    return results;
}

#[cfg(test)]
use feanor_math::seq::VectorView;
#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::fielddesc::*;
#[cfg(test)]
use crate::pmod;

#[test]
fn test_hom_digit_extract_bfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 4);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let ct_crt = enc_crt(&ring, 16, &encode_scalar(&ring, 16, 9), &sk, &mut rng, 3.2);
    let res_crt = hom_digit_extract_bfv(&ring, 4, ct_crt, &rk);
    let [res1, res2, res4] = res_crt.try_into().ok().unwrap();
    assert_eq!(1, pmod(decode_scalar(&ring, 16, dec(&ring, 16, res1, &sk)) as i64, 2));
    assert_eq!(1, pmod(decode_scalar(&ring, 16, dec(&ring, 16, res2, &sk)) as i64, 4));
    assert_eq!(1, pmod(decode_scalar(&ring, 16, dec(&ring, 16, res4, &sk)) as i64, 16));
}

#[test]
fn test_hom_digit_extract_dbfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 200..220, 16);
    let t = 1 << 16;
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let ct_crt = enc_crt(&ring, t, &encode_scalar(&ring, t, 2025), &sk, &mut rng, 3.2);
    assert_eq!(
        2025,
        decode_scalar(
            &ring,
            t,
            dbfv_dec(&ring, DBFVModulus::new(4, 4), vec![ct_crt.clone()], &sk)
        )
    );
    let res_crt = hom_digit_extract_dbfv(&ring, DBFVModulus::new(4, 4), 16, vec![ct_crt], &rk);
    let [res1, res2, res4, res8, res16] = res_crt.try_into().ok().unwrap();
    assert_eq!(
        1,
        pmod(
            decode_scalar(&ring, t, dbfv_dec(&ring, DBFVModulus::new(4, 4), res1, &sk)) as i64,
            2
        )
    );
    assert_eq!(
        1,
        pmod(
            decode_scalar(&ring, t, dbfv_dec(&ring, DBFVModulus::new(4, 4), res2, &sk)) as i64,
            4
        )
    );
    assert_eq!(
        1,
        pmod(
            decode_scalar(&ring, t, dbfv_dec(&ring, DBFVModulus::new(4, 4), res4, &sk)) as i64,
            16
        )
    );
    assert_eq!(
        1,
        pmod(
            decode_scalar(&ring, t, dbfv_dec(&ring, DBFVModulus::new(4, 4), res8, &sk)) as i64,
            256
        )
    );
    assert_eq!(
        1,
        pmod(
            decode_scalar(&ring, t, dbfv_dec(&ring, DBFVModulus::new(4, 4), res16, &sk)) as i64,
            65536
        )
    );
}

#[test]
fn test_hom_floordiv_dbfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_192(), 300..320, 16);
    let t = 1 << 16;
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let ct_crt = enc_crt(&ring, t, &encode_scalar(&ring, t, 2025), &sk, &mut rng, 3.2);
    let res_crt = hom_floordiv_dbfv(&ring, DBFVModulus::new(4, 4), DBFVModulus::new(4, 3), vec![ct_crt], &rk);
    println!(
        "{:?}",
        res_crt
            .iter()
            .map(|ct| log2_noise_pbinf_norm(&ring, 1 << 12, ct, &sk))
            .collect::<Vec<_>>()
    );
    let res = dbfv_dec(&ring, DBFVModulus::new(4, 3), res_crt, &sk);
    assert_eq!(encode_scalar(&ring, 1 << 12, 126), res);
}
