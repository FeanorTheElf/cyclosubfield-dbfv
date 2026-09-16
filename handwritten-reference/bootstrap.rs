use feanor_math::group::MultGroup;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use feanor_math::seq::VectorView;
use rand::*;
use tracing::instrument;

use crate::ZZi128;
use crate::bfv::*;
use crate::crt::UnsplitCRTTransform;
use crate::dbfv::*;
use crate::digitex::hom_floordiv_dbfv;
use crate::homcrt::*;
use crate::ring::{RingContext, RingElementData};

fn gen_sparse_sk_in_K<R: CryptoRng>(ring: &RingContext, hwt: usize, mut rng: R) -> Vec<i128> {
    let mut sparse_sk = (0..ring.field_descriptor().deg_K()).map(|_| 0).collect::<Vec<_>>();
    for _ in 0..hwt {
        let j = rng.next_u32() as usize % ring.field_descriptor().deg_K();
        sparse_sk[j] = (rng.next_u32() % 3) as i128 - 1;
    }
    let sparse_sk_in_L = (0..ring.field_descriptor().deg_Lʼ())
        .flat_map(|_| {
            sparse_sk
                .iter()
                .map(|x| x * ring.field_descriptor().scale_M_K_to_M_period_basis() as i128)
        })
        .collect::<Vec<_>>();
    return sparse_sk_in_L;
}

fn gen_sparse_sk_in_L<R: CryptoRng>(ring: &RingContext, hwt: usize, mut rng: R) -> Vec<i128> {
    let mut sparse_sk = (0..ring.field_descriptor().deg_L()).map(|_| 0).collect::<Vec<_>>();
    for _ in 0..hwt {
        let j = rng.next_u32() as usize % ring.field_descriptor().deg_L();
        sparse_sk[j] = (rng.next_u32() % 3) as i128 - 1;
    }
    return sparse_sk;
}

#[instrument(skip_all)]
pub fn create_sparse_key_encaps<R: CryptoRng>(
    ring: &RingContext,
    intermediate_t: i128,
    sk: &RingElementData,
    hwt: usize,
    mut rng: R,
    log2_caninf_norm_bound: f64,
) -> (KSKey, DBFVCiphertext<'static>) {
    let mut log2_encountered_norm = f64::INFINITY;
    for attempt in 0..2048 {
        let sparse_sk_in_K = gen_sparse_sk_in_K(ring, hwt, &mut rng);
        let mut sparse_sk = ring.plaintext_into_ciphertext_ring_unscaled_pb(intermediate_t, &sparse_sk_in_K);
        let log2_caninf_norm = ring.log2_caninf_norm_small(sparse_sk.borrow());
        log2_encountered_norm = f64::min(log2_encountered_norm, log2_caninf_norm);
        if log2_caninf_norm <= log2_caninf_norm_bound {
            let sparse_sk_enc = vec![enc_crt(ring, intermediate_t, &sparse_sk_in_K, &sk, &mut rng, 3.2)];
            let small_rns_base = ring.sparse_key_encaps_rns_base();
            println!();
            println!(
                "sparse secret key log2 caninf-norm {}, found after {} attempts",
                log2_caninf_norm, attempt
            );
            ring.pb_to_crt(sparse_sk.borrow_mut());
            *DebugSparseSK.lock().unwrap() =
                Some(RingElementData::Borrowed(&sparse_sk.borrow()[..(small_rns_base.len() * ring.n())]).to_static());
            let sparse_ksk = gen_ksk_small_rns_base(
                ring,
                &RingElementData::Borrowed(&sk.borrow()[..(small_rns_base.len() * ring.n())]),
                &RingElementData::Borrowed(&sparse_sk.borrow()[..(small_rns_base.len() * ring.n())]),
                rng,
                3.2,
            );
            return (sparse_ksk, sparse_sk_enc);
        }
    }
    unreachable!(
        "couldn't find a secret key with given sparsity and norm bound; smallest encountered log2 caninf norm is {}",
        log2_encountered_norm
    )
}

#[instrument(skip_all)]
pub fn bootstrap_bfv_dbfv(
    ring: &RingContext,
    crt_transform: &UnsplitCRTTransform,
    modulus: DBFVModulus,
    ctxt_crt: DBFVCiphertext,
    rk: &KSKey,
    gks: &[(GEl<MultGroup<Zn64B>>, KSKey)],
    sparse_key_encaps: &(KSKey, DBFVCiphertext),
) -> DBFVCiphertext<'static> {
    let intermediate_t = crt_transform.t();
    let reduced_crt_transform = crt_transform.reduce_t(modulus.t());
    let in_coeffs = slots_to_coeffs_dbfv(ring, &reduced_crt_transform, modulus, ctxt_crt, gks);
    let mut in_coeffs_bfv = (ring.init_zero(), ring.init_zero());
    for (i, ct) in in_coeffs.into_iter().enumerate() {
        ring.fma_scalar(ct.0.borrow(), 1 << (i * modulus.log2_b()), in_coeffs_bfv.0.borrow_mut());
        ring.fma_scalar(ct.1.borrow(), 1 << (i * modulus.log2_b()), in_coeffs_bfv.1.borrow_mut());
    }

    ring.crt_to_pb(in_coeffs_bfv.0.borrow_mut());
    ring.crt_to_pb(in_coeffs_bfv.1.borrow_mut());
    let mut modswitched = mod_switch_to_sparse_key_encaps(ring, &in_coeffs_bfv);
    ring.pb_to_crt(modswitched.0.borrow_mut());
    ring.pb_to_crt(modswitched.1.borrow_mut());

    ring.crt_to_pb(modswitched.1.borrow_mut());
    let keyswitched = key_switch_crt(ring, modswitched.0, modswitched.1, &sparse_key_encaps.0, None);

    let n = ring.n();
    let deg_K = ring.field_descriptor().deg_K();
    let trace = |x: &mut [_]| {
        for (i, Zp) in ring.sparse_key_encaps_rns_base().as_iter().enumerate() {
            for j in 0..deg_K {
                let new_val = Zp.sum((0..ring.field_descriptor().deg_Lʼ()).map(|l| x[i * n + l * deg_K + j]));
                for l in 0..ring.field_descriptor().deg_Lʼ() {
                    x[i * n + l * deg_K + j] = new_val;
                }
            }
        }
    };
    let mut traced = keyswitched;
    trace(traced.0.borrow_mut());
    trace(traced.1.borrow_mut());

    ring.crt_to_pb(traced.0.borrow_mut());
    ring.crt_to_pb(traced.1.borrow_mut());
    let as_plain = (
        ring.plaintext_from_ciphertext_ring_scaled_pb(intermediate_t, traced.0),
        ring.plaintext_from_ciphertext_ring_scaled_pb(intermediate_t, traced.1),
    );

    let Delta = ring.rns_base().coerce(
        &ZZbig,
        ZZbig.rounded_div(
            ring.rns_base().modulus().clone(),
            &int_cast(intermediate_t, ZZbig, ZZi128),
        ),
    );
    let intermediate_modulus =
        DBFVModulus::new_incomplete(modulus.log2_b(), ZZi128.abs_log2_ceil(&intermediate_t).unwrap());

    let mut noisy_message_crt = dbfv_hom_fma_plain(
        ring,
        intermediate_modulus,
        &sparse_key_encaps.1,
        &as_plain.1,
        Vec::new(),
    );
    let mut constant = ring.plaintext_into_ciphertext_ring_unscaled_pb(intermediate_t, &as_plain.0);
    ring.pb_to_crt(constant.borrow_mut());
    ring.fma_scalar_large(constant.borrow(), &Delta, noisy_message_crt[0].0.borrow_mut());

    let round_to_floor_constant = ring.rns_base().mul(
        Delta,
        ring.rns_base().coerce(
            &ZZi128,
            intermediate_t / modulus.t() / 2, // no scale_Q_to_M, since we are in CRT basis
        ),
    );
    ring.add_scalar_large(&round_to_floor_constant, noisy_message_crt[0].0.borrow_mut());

    // let noises = dbfv_log2_noise_pb_norm(
    //     ring,
    //     intermediate_modulus,
    //     &noisy_message_crt,
    //     DebugSK.lock().unwrap().as_ref().unwrap(),
    // );
    // println!();
    // println!("before c2s: {:?}", noises);

    let in_slots = coeffs_to_slots_dbfv(ring, &crt_transform, intermediate_modulus, noisy_message_crt, &gks);

    // let noises = dbfv_log2_noise_pb_norm(
    //     ring,
    //     intermediate_modulus,
    //     &in_slots,
    //     DebugSK.lock().unwrap().as_ref().unwrap(),
    // );
    // let values = crt_transform.extract_slot_values(ring, &dbfv_dec(ring, intermediate_modulus,
    // in_slots.clone(), DebugSK.lock().unwrap().as_ref().unwrap())); println!();
    // println!("before floordiv: {:?}", noises);
    // println!("{:?}", &values[..20]);

    let result = hom_floordiv_dbfv(ring, intermediate_modulus, modulus, in_slots, &rk);

    // let noises = dbfv_log2_noise_pb_norm(
    //     ring,
    //     intermediate_modulus,
    //     &result,
    //     DebugSK.lock().unwrap().as_ref().unwrap(),
    // );
    // println!();
    // println!("after bootstrap: {:?}", noises);
    return result;
}

#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::fielddesc::{field_18, field_36864};

#[test]
fn test_bootstrap() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_18(), 300..320, 20);
    let t = 1 << 15;
    let intermediate_t = 1 << 20;
    let sk = gen_sk(&ring, &mut rng);
    let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field_descriptor());
    let sparse_sk = create_sparse_key_encaps(&ring, intermediate_t, &sk, 4, &mut rng, f64::INFINITY);
    let digits = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let gks = slots_to_coeffs_galois_keys(&ring, &crt_transform)
        .into_iter()
        .map(|g| (g, gen_gk(&ring, &sk, g, digits.iter().cloned(), &mut rng, 3.2)))
        .collect::<Vec<_>>();
    let rk = gen_rk(&ring, &sk, digits, &mut rng, 3.2);

    let ct_crt = enc_crt(&ring, t, &encode_scalar(&ring, t, 2025), &sk, &mut rng, 3.2);
    let res_crt = bootstrap_bfv_dbfv(
        &ring,
        &crt_transform,
        DBFVModulus::new(5, 3),
        vec![ct_crt],
        &rk,
        &gks,
        &sparse_sk,
    );
    assert_eq!(
        encode_scalar(&ring, 1 << 16, 2025),
        dbfv_dec(&ring, DBFVModulus::new(5, 3), res_crt, &sk)
    )
}

#[test]
#[ignore]
fn estimate_performance_bootstrap_dbfv() {
    feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_36864(), 900..920, 80);
    let t = 1 << 64;
    let intermediate_t = 1 << 72;
    let sk = gen_sk(&ring, &mut rng);
    *DebugSK.lock().unwrap() = Some(sk.clone());
    let crt_transform = UnsplitCRTTransform::new(intermediate_t, ring.field_descriptor());
    let sparse_sk = create_sparse_key_encaps(&ring, intermediate_t, &sk, 16, &mut rng, 6.0);
    let digits_gks = select_digits(ring.rns_base().len(), ring.rns_base().len());
    let gks = slots_to_coeffs_galois_keys(&ring, &crt_transform)
        .into_iter()
        .map(|g| (g, gen_gk(&ring, &sk, g, digits_gks.iter().cloned(), &mut rng, 3.2)))
        .collect::<Vec<_>>();
    let digits_rk = select_digits((ring.rns_base().len() - 1) / 2 + 1, ring.rns_base().len());
    let rk = gen_rk(&ring, &sk, digits_rk, &mut rng, 3.2);

    let ct_crt = enc_crt(&ring, t, &encode_scalar(&ring, t, 2025), &sk, &mut rng, 3.2);
    let res_crt = bootstrap_bfv_dbfv(
        &ring,
        &crt_transform,
        DBFVModulus::new(16, 4),
        vec![ct_crt],
        &rk,
        &gks,
        &sparse_sk,
    );
    assert_eq!(
        (0..ring.slot_count()).map(|_| 24300).collect::<Vec<_>>(),
        crt_transform
            .reduce_t(t)
            .extract_slot_values(&ring, &dbfv_dec(&ring, DBFVModulus::new(16, 4), res_crt, &sk))
    )
}

#[test]
fn exp_gen_sk() {
    // feanor_tracing::DelayedLogger::init_test();
    let mut rng = StdRng::from_seed([0; 32]);
    let ring = RingContext::new(field_36864(), 200..220, 16);
    println!("in K");
    let mut list = (0..32)
        .map(|_| {
            let s = ring.plaintext_into_ciphertext_ring_unscaled_pb(16, &gen_sparse_sk_in_K(&ring, 16, &mut rng));
            ring.log2_caninf_norm(s.borrow())
        })
        .collect::<Vec<_>>();
    list.sort_unstable_by(f64::total_cmp);
    println!("{}", list[list.len() / 2]);
    println!("in L");
    let mut list = (0..32)
        .map(|_| {
            let s = ring.plaintext_into_ciphertext_ring_unscaled_pb(16, &gen_sparse_sk_in_L(&ring, 16, &mut rng));
            ring.log2_caninf_norm(s.borrow())
        })
        .collect::<Vec<_>>();
    list.sort_unstable_by(f64::total_cmp);
    println!("{}", list[list.len() / 2]);
}
