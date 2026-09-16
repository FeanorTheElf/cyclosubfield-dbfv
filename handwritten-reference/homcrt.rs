use std::collections::HashMap;
use std::convert::identity;
use std::sync::Mutex;

use feanor_math::group::MultGroup;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use fhe_ir::GenericInstruction::*;
use fhe_ir::{Instruction, Program, ValueList};
use tracing::instrument;

use crate::bfv::*;
use crate::crt::*;
use crate::dbfv::*;
use crate::fielddesc::field_36864;
use crate::ring::*;

#[instrument(skip_all)]
pub fn slots_to_coeffs_bfv(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
    ctxt_crt: (RingElementData, RingElementData),
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> (RingElementData<'static>, RingElementData<'static>) {
    let transform = create_slots_to_coeffs(ring, crt_transform);
    let [res] = evaluate_linear_transform_bfv(&transform, &ring, vec![ctxt_crt], &gks)
        .try_into()
        .ok()
        .unwrap();
    return res;
}

#[instrument(skip_all)]
pub fn coeffs_to_slots_bfv(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
    ctxt_crt: (RingElementData, RingElementData),
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> (RingElementData<'static>, RingElementData<'static>) {
    let transform = create_coeffs_to_slots(ring, crt_transform);
    let [res] = evaluate_linear_transform_bfv(&transform, &ring, vec![ctxt_crt], &gks)
        .try_into()
        .ok()
        .unwrap();
    return res;
}

#[instrument(skip_all)]
pub fn coeffs_to_slots_dbfv(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    ctxt_crt: DBFVCiphertext,
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> DBFVCiphertext<'static> {
    assert_eq!(modulus.t(), crt_transform.t());
    let transform = create_coeffs_to_slots(ring, crt_transform);
    let [res] = evaluate_linear_transform_dbfv(&transform, &ring, modulus, vec![ctxt_crt], &gks)
        .try_into()
        .ok()
        .unwrap();
    return res;
}

#[instrument(skip_all)]
pub fn slots_to_coeffs_dbfv(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    ctxt_crt: DBFVCiphertext,
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> DBFVCiphertext<'static> {
    assert_eq!(modulus.t(), crt_transform.t());
    let transform = create_slots_to_coeffs(ring, crt_transform);
    let [res] = evaluate_linear_transform_dbfv(&transform, &ring, modulus, vec![ctxt_crt], &gks)
        .try_into()
        .ok()
        .unwrap();
    return res;
}

/// Creates the map that moves the slots values to the coefficients w.r.t. the period basis on K
/// (induced by M_K).
///
/// This means that if you want to move the slot values to the coefficient w.r.t. the period basis
/// on K induced by M, you need to multiply with [`FieldDescriptor::scale_M_K_to_M_period_basis()`].
#[instrument(skip_all)]
pub fn create_slots_to_coeffs(ring: &RingContext, crt_transform: &UnsplitCRTTransform) -> Program<ValueList<i128>> {
    let mut instructions = Vec::new();
    let mut ptxt_table = HashMap::new();
    for (stage, factor) in crt_transform.factors().iter().enumerate() {
        bsgs_cyclic_linear_transform(
            ring,
            *factor.gen_Jp(),
            factor.u_p(),
            &mut instructions,
            &mut ptxt_table,
            stage,
        );
        if stage == crt_transform.factors().len() - 1 {
            instructions.push(Return { val: "%out" });
        } else {
            instructions.push(Copy {
                out: "%in",
                val: "%out",
            });
        }
    }
    let result = Program::new(&["%in"], instructions, ptxt_table);
    result.check().unwrap();
    return result;
}

/// Creates the map that moves the coefficients w.r.t. the period basis on K (induced by M_K), to
/// the slots.
///
/// This means that if you want to move the coefficient w.r.t. the period basis on K induced by M to
/// the slots, you need to multiply with [`FieldDescriptor::scale_M_K_to_M_period_basis()`].
#[instrument(skip_all)]
pub fn create_coeffs_to_slots(ring: &RingContext, crt_transform: &UnsplitCRTTransform) -> Program<ValueList<i128>> {
    let mut instructions = Vec::new();
    let mut ptxt_table = HashMap::new();
    for (stage, factor) in crt_transform.factors().iter().enumerate() {
        bsgs_cyclic_linear_transform(
            ring,
            *factor.gen_Jp(),
            factor.inv_u_p(),
            &mut instructions,
            &mut ptxt_table,
            stage,
        );
        if stage == crt_transform.factors().len() - 1 {
            instructions.push(Return { val: "%out" });
        } else {
            instructions.push(Copy {
                out: "%in",
                val: "%out",
            });
        }
    }
    let result = Program::new(&["%in"], instructions, ptxt_table);
    result.check().unwrap();
    return result;
}

pub fn slots_to_coeffs_galois_keys(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
) -> Vec<GEl<MultGroup<Zn64B>>> {
    required_galois_keys(ring, &create_slots_to_coeffs(ring, crt_transform))
}

pub fn required_galois_keys(ring: &RingContext, program: &Program<ValueList<i128>>) -> Vec<GEl<MultGroup<Zn64B>>> {
    let GalMQ = ring.field_descriptor().GalMQ();
    let mut result = Vec::new();
    for instruction in program.instructions() {
        if let Instruction::Galois {
            out: _,
            val: _,
            exponents,
        } = &instruction
        {
            for exp in exponents {
                result.push(
                    GalMQ
                        .from_ring_el(GalMQ.underlying_ring().coerce(&ZZi64, *exp))
                        .unwrap(),
                );
            }
        }
    }
    result.sort_unstable_by_key(|g| GalMQ.underlying_ring().smallest_positive_lift(*GalMQ.as_ring_el(g)));
    result.dedup_by(|g, h| GalMQ.eq_el(g, h));
    return result;
}

fn baby_step_names(count: usize) -> Vec<&'static str> {
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap();
    if names.len() < count {
        let to_add = (names.len()..count).map(|i| &*format!("%bs{}", i).leak());
        names.extend(to_add);
    }
    return names[..count].to_vec();
}

#[instrument(skip_all)]
fn bsgs_cyclic_linear_transform(
    ring: &RingContext,
    automorphism: GEl<MultGroup<Zn64B>>,
    transform_coefficients: &Vec<Vec<i128>>,
    instructions: &mut Vec<Instruction>,
    ptxt_table: &mut HashMap<&'static str, ValueList<i128>>,
    stage: usize,
) -> &'static str {
    const UNHOISTED_OVERHEAD: usize = 3;
    let GalMQ = ring.field_descriptor().GalMQ();
    let len = transform_coefficients.len();
    let baby_steps = (1..=len)
        .min_by_key(|count| count + UNHOISTED_OVERHEAD * ((len - 1) / count + 1))
        .unwrap();
    let giant_steps = (len - 1) / baby_steps + 1;

    let baby_step_names = baby_step_names(baby_steps).into_iter().skip(1).collect::<Vec<_>>();
    let baby_step_exponents = (1..baby_steps)
        .map(|i| {
            GalMQ
                .underlying_ring()
                .smallest_positive_lift(*GalMQ.as_ring_el(&GalMQ.pow(automorphism, i as i64)))
        })
        .collect();
    instructions.push(Instruction::Galois {
        out: baby_step_names.clone(),
        val: "%in",
        exponents: baby_step_exponents,
    });
    let baby_step_names = ["%in"].into_iter().chain(baby_step_names).collect::<Vec<_>>();
    let mut coeff_names = (0..giant_steps)
        .map(|i| {
            (0..usize::min(baby_steps, len - i * baby_steps))
                .map(|j| &*format!("@it{}_gs{}_bs{}", stage, i, j).leak())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (val, name) in transform_coefficients.iter().zip(coeff_names.iter().flat_map(identity)) {
        _ = ptxt_table.insert(*name, ValueList::from(val.clone()));
    }
    let giant_step_auto = GalMQ
        .underlying_ring()
        .smallest_positive_lift(*GalMQ.as_ring_el(&GalMQ.pow(automorphism, baby_steps as i64)));
    let output = "%out";
    let conjugate = "%conj";
    let coeffs = coeff_names.pop().unwrap();
    let values = baby_step_names[..coeffs.len()].to_owned();
    instructions.push(InnerProduct {
        out: output,
        values,
        coefficients: coeffs,
    });
    while let Some(coeffs) = coeff_names.pop() {
        instructions.push(Galois {
            out: vec![conjugate],
            val: output,
            exponents: vec![giant_step_auto],
        });
        instructions.push(InnerProduct {
            out: output,
            values: baby_step_names.clone(),
            coefficients: coeffs,
        });
        instructions.push(AddCtxCtx {
            out: output,
            lhs: output,
            rhs: conjugate,
        });
    }
    return output;
}

fn evaluate_linear_transform_bfv(
    program: &Program<ValueList<i128>>,
    ring: &RingContext,
    ctxts_crt: Vec<(RingElementData, RingElementData)>,
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> Vec<(RingElementData<'static>, RingElementData<'static>)> {
    assert_eq!(ctxts_crt.len(), program.inputs().len());
    let mut data = HashMap::new();
    for (input, value) in program.inputs().zip(ctxts_crt) {
        data.insert(input, value);
    }
    let GalMQ = ring.field_descriptor().GalMQ();
    let mut results = Vec::new();
    for inst in program.instructions_with_data() {
        #[allow(unused_variables)]
        match inst {
            Galois { out, val, exponents } => {
                let val = data.get(val).unwrap();
                let exponents = exponents
                    .iter()
                    .map(|g| GalMQ.from_ring_el(GalMQ.underlying_ring().coerce(&ZZi64, *g)).unwrap())
                    .collect::<Vec<_>>();
                for (out, result) in out.iter().zip(hom_gal(ring, val.clone(), &exponents, gks)) {
                    _ = data.insert(out, result)
                }
            }
            InnerProduct {
                out,
                values,
                coefficients,
            } => {
                let mut res0 = ring.init_zero();
                let mut res1 = ring.init_zero();
                for (coeff, val) in coefficients.iter().zip(values) {
                    let val = data.get(val).unwrap();
                    ring.fma_element_Lʼ(val.0.borrow(), &***coeff, res0.borrow_mut());
                    ring.fma_element_Lʼ(val.1.borrow(), &***coeff, res1.borrow_mut());
                }
                _ = data.insert(out, (res0, res1));
            }
            AddCtxCtx { out, lhs, rhs } => {
                let lhs = data.get(lhs).unwrap();
                let rhs = data.get(rhs).unwrap();
                let mut res = lhs.clone();
                ring.add(rhs.0.borrow(), res.0.borrow_mut());
                ring.add(rhs.1.borrow(), res.1.borrow_mut());
                _ = data.insert(out, res);
            }
            Return { val } => {
                let val = data.get(val).unwrap();
                results.push((val.0.clone().to_static(), val.1.clone().to_static()));
            }
            Copy { out, val } => {
                let val = data.get(val).unwrap();
                _ = data.insert(out, val.clone());
            }
            Zero { out } => unreachable!(),
            MulPtxCtx { out, value, plaintext } => unreachable!(),
            MulIntCtx { out, value, integer } => unreachable!(),
            AddPtxCtx { out, value, plaintext } => unreachable!(),
            MulCtxCtx { out, lhs, rhs } => unreachable!(),
        }
    }
    return results;
}

fn evaluate_linear_transform_dbfv(
    program: &Program<ValueList<i128>>,
    ring: &RingContext,
    modulus: DBFVModulus,
    ctxts_crt: Vec<DBFVCiphertext>,
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
) -> Vec<DBFVCiphertext<'static>> {
    assert_eq!(ctxts_crt.len(), program.inputs().len());
    let mut data = HashMap::new();
    for (input, value) in program.inputs().zip(ctxts_crt) {
        data.insert(input, value);
    }
    let GalMQ = ring.field_descriptor().GalMQ();
    let mut results = Vec::new();
    for inst in program.instructions_with_data() {
        #[allow(unused_variables)]
        match inst {
            Galois { out, val, exponents } => {
                let val = data.get(val).unwrap();
                let exponents = exponents
                    .iter()
                    .map(|g| GalMQ.from_ring_el(GalMQ.underlying_ring().coerce(&ZZi64, *g)).unwrap())
                    .collect::<Vec<_>>();
                let mut results = (0..exponents.len()).map(|_| Vec::new()).collect::<Vec<_>>();
                for ct in val {
                    for (append_to, result) in results.iter_mut().zip(hom_gal(ring, ct.clone(), &exponents, gks)) {
                        append_to.push(result);
                    }
                }
                for (out, result) in out.iter().zip(results) {
                    _ = data.insert(*out, result);
                }
            }
            InnerProduct {
                out,
                values,
                coefficients,
            } => {
                let mut res = Vec::new();
                for (coeff, val) in coefficients.iter().zip(values) {
                    let val = data.get(val).unwrap();
                    res = dbfv_hom_fma_element_Lʼ(ring, modulus, val, coeff, res);
                }
                _ = data.insert(out, res);
            }
            AddCtxCtx { out, lhs, rhs } => {
                let lhs = data.get(lhs).unwrap().clone();
                let rhs = data.get(rhs).unwrap();
                let res = dbfv_hom_add(ring, lhs, rhs);
                _ = data.insert(out, res);
            }
            Copy { out, val } => {
                let val = data.get(val).unwrap();
                _ = data.insert(out, val.clone());
            }
            Return { val } => {
                let val = data.get(val).unwrap();
                results.push(
                    val.iter()
                        .map(|ct| (ct.0.clone().to_static(), ct.1.clone().to_static()))
                        .collect(),
                );
            }
            Zero { out } => unreachable!(),
            MulPtxCtx { out, value, plaintext } => unreachable!(),
            MulIntCtx { out, value, integer } => unreachable!(),
            AddPtxCtx { out, value, plaintext } => unreachable!(),
            MulCtxCtx { out, lhs, rhs } => unreachable!(),
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
use crate::cmod;
#[cfg(test)]
use crate::fielddesc::field_18;

#[test]
fn test_hom_linear_transform_field18_bfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_18(), 400..420, 4);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits.iter().cloned(), &mut rng, 3.2);
    let crt_transform = UnsplitCRTTransform::new(16, ring.field_descriptor());
    let c2s = create_coeffs_to_slots(&ring, &crt_transform);
    let s2c = create_slots_to_coeffs(&ring, &crt_transform);
    let gks = required_galois_keys(&ring, &s2c)
        .into_iter()
        .map(|g| (g, gen_gk(&ring, &sk, g, digits.iter().cloned(), &mut rng, 3.2)))
        .collect::<Vec<_>>();

    let slot_values = (0..ring.slot_count()).map(|i| cmod(i as i128, 16)).collect::<Vec<_>>();
    let ptxt_pb = (0..ring.field_descriptor().deg_Lʼ())
        .flat_map(|_| slot_values.iter())
        .copied()
        .collect::<Vec<_>>();
    let ctxt_crt = enc_crt(&ring, 16, &ptxt_pb, &sk, &mut rng, 3.2);

    let [c2s_crt] = evaluate_linear_transform_bfv(&c2s, &ring, vec![ctxt_crt], &gks)
        .try_into()
        .ok()
        .unwrap();
    let prod_crt = hom_mul_full_crt(&ring, 16, c2s_crt.clone(), c2s_crt, &rk);
    let [res_crt] = evaluate_linear_transform_bfv(&s2c, &ring, vec![prod_crt], &gks)
        .try_into()
        .ok()
        .unwrap();

    println!("final noise = {:?}", log2_noise_pbinf_norm(&ring, 16, &res_crt, &sk));
    let expected = slot_values.iter().map(|x| cmod(x * x, 16)).collect::<Vec<_>>();
    let actual = dec(&ring, 16, res_crt, &sk)[..ring.slot_count()]
        .iter()
        .map(|x| cmod(x * ring.field_descriptor().scale_M_K_to_M_period_basis() as i128, 16))
        .collect::<Vec<_>>();
    assert_eq!(&expected, &actual);
}

#[test]
#[ignore]
fn test_hom_linear_transform_field_36864_dbfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_36864(), 400..420, 16);
    let sk = gen_sk(&ring, &mut rng);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let crt_transform = UnsplitCRTTransform::new(1 << 16, ring.field_descriptor());
    let gks = slots_to_coeffs_galois_keys(&ring, &crt_transform)
        .into_iter()
        .map(|g| (g, gen_gk(&ring, &sk, g, digits.iter().cloned(), &mut rng, 3.2)))
        .collect::<Vec<_>>();

    let slot_values = (0..ring.slot_count())
        .map(|i| cmod(i as i128, 1 << 16))
        .collect::<Vec<_>>();
    let ptxt_pb = (0..ring.field_descriptor().deg_Lʼ())
        .flat_map(|_| {
            slot_values
                .iter()
                .map(|x| *x * ring.field_descriptor().scale_M_K_to_M_period_basis() as i128)
        })
        .collect::<Vec<_>>();
    let ctxt_crt = vec![enc_crt(&ring, 1 << 16, &ptxt_pb, &sk, &mut rng, 3.2)];

    let res_crt = coeffs_to_slots_dbfv(&ring, &crt_transform, DBFVModulus::new(4, 4), ctxt_crt, &gks);

    let actual = crt_transform.extract_slot_values(&ring, &dbfv_dec(&ring, DBFVModulus::new(4, 4), res_crt, &sk));
    assert_eq!(&slot_values, &actual);
}
