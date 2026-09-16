use std::ops::Range;
use std::time::Instant;

use cyclosubfield_dbfv::bfv::{
    KeySwitchKey, SecretKey, bfv_enc, decode_scalar, encode_scalar, gen_gk, gen_rk, gen_sk, select_digits,
};
use cyclosubfield_dbfv::bootstrap::homcrt::{UnsplitCRTTransform, coeffs_to_slots_dbfv, required_galois_elements};
use cyclosubfield_dbfv::bootstrap::{bootstrap_bfv_dbfv, create_sparse_key_encaps_with_bound};
use cyclosubfield_dbfv::dbfv::{DBFVModulus, dbfv_dec, dbfv_hom_add, dbfv_hom_mul, dbfv_log2_noise_pbinf_norm};
use cyclosubfield_dbfv::field::{FieldDescriptor, field_32768, field_32768_ex, field_49152, field_49152_ex};
use cyclosubfield_dbfv::hal::cpu::CpuBackend;
use cyclosubfield_dbfv::keystore::KeyStore;
use cyclosubfield_dbfv::ring::{Basis, RingContext, ZZbig, cmod};
use feanor_math::integer::IntegerRingStore;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::{Benchmark, DType, NoiseBudget, NoiseInfo, Reporter, Scheme};

/// Generates the Galois keys the transform needs, moving each one onto disk as soon as it is
/// built if `DBFV_KEY_STORE_DIR` names a directory (see [`cyclosubfield_dbfv::keystore`]).
///
/// At the large parameter sets the key material does not fit in RAM — at `field_65536()` it is
/// tens of gigabytes — while the ciphertext buffers of the transform do. Offloading here keeps
/// only one key resident at a time during generation, and lets the kernel stream the keys back
/// from the file during the transform, where each is read once per dBFV digit in a fixed order.
fn gen_galois_keys<R: FnMut() -> u64>(
    ring: &RingContext<CpuBackend>,
    sk: &SecretKey<CpuBackend>,
    crt_transform: &UnsplitCRTTransform,
    digits: &[Range<usize>],
    rng: &mut R,
    log2_alpha: f64,
) -> Vec<(u64, KeySwitchKey<CpuBackend>)> {
    let store = KeyStore::from_env().expect("could not create the key store");
    let elements = required_galois_elements(ring.field(), crt_transform);
    let start = Instant::now();
    let gks = elements
        .into_iter()
        .map(|g| {
            let gk = gen_gk(ring, sk, g, digits, rng, log2_alpha);
            let gk = match &store {
                Some(store) => gk.offload_to_disk(store).expect("could not offload a galois key"),
                None => gk,
            };
            (g, gk)
        })
        .collect::<Vec<_>>();
    let size: usize = gks.iter().map(|(_, gk)| gk.size_bytes(ring)).sum();
    println!(
        "  generated {} galois keys ({:.1} GiB{}) in {:.0} s",
        gks.len(),
        size as f64 / (1u64 << 30) as f64,
        match &store {
            Some(store) => format!(", on disk at {}", store.path().display()),
            None => format!(", in memory"),
        },
        start.elapsed().as_secs_f64()
    );
    return gks;
}

fn param_str(field: &FieldDescriptor, modulus: DBFVModulus, log2_alpha: f64) -> String {
    if modulus.digits() == 1 {
        format!("n = {}, m = {}, log2(α) = {}, t = 2^{}", field.deg_l(), field.m(), log2_alpha, modulus.log2_t())
    } else {
        format!(
            "n = {}, m = {}, log2(α) = {}, t = 2^{}, b = 2^{}",
            field.deg_l(),
            field.m(),
            log2_alpha,
            modulus.log2_t(),
            modulus.log2_b(),
        )
    }
}

pub fn bootstrap_benchmark(
    reporter: &mut Reporter,
    field: FieldDescriptor,
    modulus: DBFVModulus,
    log2_alpha: f64,
    max_log_amplification: f64,
    v: usize,
) {
    let param_str = param_str(&field, modulus, log2_alpha);
    let mut rng = StdRng::from_seed([0; 32]);
    let log2_q = (log2_alpha.ceil() as usize)..(log2_alpha.ceil() as usize + 30);
    let num_slots = field.deg_k();
    let ring = RingContext::new(CpuBackend, field, log2_q, modulus.log2_t() + v);
    // one-time, RNS-base-independent setup; see the note in
    // `RingContext::precompute_rns_factor_transforms()`
    ring.precompute_rns_factor_transforms();
    let base = ring.initial_rns_base();
    let sk = gen_sk(&ring, &mut || rng.next_u64());
    let intermediate_modulus = DBFVModulus::new_incomplete(modulus.log2_b(), modulus.log2_t() + v);
    let crt_transform = UnsplitCRTTransform::new(1 << intermediate_modulus.log2_t(), ring.field());
    let sparse = create_sparse_key_encaps_with_bound(
        &ring,
        intermediate_modulus,
        &sk,
        32,
        max_log_amplification,
        1024,
        &mut || rng.next_u64(),
        log2_alpha,
    )
    .0;
    let digits_gks = select_digits(base.len().div_ceil(2), base.len());
    let gks = gen_galois_keys(&ring, &sk, &crt_transform, &digits_gks, &mut || rng.next_u64(), log2_alpha);
    let digits_rk = select_digits(base.len().div_ceil(4), base.len());
    let rk = gen_rk(&ring, &sk, &digits_rk, &mut || rng.next_u64(), log2_alpha);
    let ct = bfv_enc(&ring, &base, modulus.t(), &encode_scalar(&ring, 1), &sk, &mut || rng.next_u64(), log2_alpha);
    let start = Instant::now();
    let res = bootstrap_bfv_dbfv(&ring, &crt_transform, modulus, ct, &rk, &gks, &sparse, None);
    let end = Instant::now();
    let time = (end - start).as_millis();
    let final_noise = dbfv_log2_noise_pbinf_norm(&ring, modulus, &res, &sk).iter().map(|x| *x as usize).max().unwrap();
    let final_precision = ZZbig.abs_log2_ceil(&ring.modulus_of(&res.rns_base)).unwrap();
    let budget = final_precision as i64 - final_noise as i64 - modulus.log2_t() as i64;
    let result = dbfv_dec(&ring, modulus, &res, &sk);
    let scheme = if DBFVModulus::new_incomplete(modulus.log2_b(), modulus.log2_t() + v).digits() > 1 {
        Scheme::CycloSubringDBFV
    } else {
        Scheme::CycloSubringBFV
    };
    let noise = NoiseInfo::new(
        crt_transform
            .reduce_t(modulus.t())
            .pack_slot_values(&ring, &(0..ring.slot_count()).map(|_| cmod(1, modulus.t())).collect::<Vec<_>>())
            == result,
        NoiseBudget::Left(budget),
    );
    let dtype = DType::Uint(modulus.log2_t());
    reporter.report(
        Benchmark::Bootstrap,
        dtype,
        scheme,
        param_str.as_str(),
        u64::try_from(time).unwrap(),
        num_slots,
        noise,
    );
}

pub fn add_mul_benchmark(reporter: &mut Reporter, field: FieldDescriptor, modulus: DBFVModulus, log2_alpha: f64) {
    let param_str = param_str(&field, modulus, log2_alpha);
    let mut rng = StdRng::from_seed([0; 32]);
    let dtype = DType::Uint(modulus.log2_t());
    let scheme = if modulus.digits() > 1 { Scheme::CycloSubringDBFV } else { Scheme::CycloSubringBFV };
    let num_slots = field.deg_k();
    let log2_q = (log2_alpha.ceil() as usize)..(log2_alpha.ceil() as usize + 30);
    let ring = RingContext::new(CpuBackend, field, log2_q, modulus.log2_t());
    // one-time, RNS-base-independent setup; see the note in
    // `RingContext::precompute_rns_factor_transforms()`
    ring.precompute_rns_factor_transforms();
    let base = ring.initial_rns_base();
    let sk = gen_sk(&ring, &mut || rng.next_u64());
    let digits_rk = select_digits(base.len().div_ceil(4), base.len());
    let rk = gen_rk(&ring, &sk, &digits_rk, &mut || rng.next_u64(), log2_alpha);
    let mut ct_pre =
        bfv_enc(&ring, &base, modulus.t(), &encode_scalar(&ring, 1), &sk, &mut || rng.next_u64(), log2_alpha);
    ring.crt_to_pb(&mut ct_pre);
    let mut ct1 = dbfv_hom_mul(&ring, modulus, ring.clone_els(&ct_pre), ring.clone_els(&ct_pre), &rk, Basis::CRTBasis);
    let ct2 = dbfv_hom_mul(&ring, modulus, ring.clone_els(&ct_pre), ct_pre, &rk, Basis::PeriodBasis);
    let input_noise = dbfv_log2_noise_pbinf_norm(&ring, modulus, &ct1, &sk).iter().map(|x| *x as usize).max().unwrap();
    ring.crt_to_pb(&mut ct1);

    {
        let start = Instant::now();
        let mut ct_res =
            dbfv_hom_mul(&ring, modulus, ring.clone_els(&ct1), ring.clone_els(&ct2), &rk, Basis::PeriodBasis);
        let end = Instant::now();
        ring.pb_to_crt(&mut ct_res);
        let time = (end - start).as_millis();
        let final_noise =
            dbfv_log2_noise_pbinf_norm(&ring, modulus, &ct_res, &sk).iter().map(|x| *x as usize).max().unwrap();
        let used_noise = final_noise as i64 - input_noise as i64;
        let result = dbfv_dec(&ring, modulus, &ct_res, &sk);
        let noise = NoiseInfo::new(
            (decode_scalar(&ring, modulus.t(), &result) + modulus.t()) % modulus.t() == 1 % modulus.t(),
            NoiseBudget::Consumed(used_noise),
        );
        reporter.report(
            Benchmark::Mul,
            dtype,
            scheme,
            param_str.as_str(),
            u64::try_from(time).unwrap(),
            num_slots,
            noise,
        );
    }
    {
        let start = Instant::now();
        let mut ct_res = dbfv_hom_add(&ring, ring.clone_els(&ct1), &ct2);
        let end = Instant::now();
        ring.pb_to_crt(&mut ct_res);
        let time = (end - start).as_millis();
        let final_noise =
            dbfv_log2_noise_pbinf_norm(&ring, modulus, &ct_res, &sk).iter().map(|x| *x as usize).max().unwrap();
        let used_noise = final_noise as i64 - input_noise as i64;
        let result = dbfv_dec(&ring, modulus, &ct_res, &sk);
        let noise = NoiseInfo::new(
            (decode_scalar(&ring, modulus.t(), &result) + modulus.t()) % modulus.t() == 2 % modulus.t(),
            NoiseBudget::Consumed(used_noise),
        );
        reporter.report(
            Benchmark::Add,
            dtype,
            scheme,
            param_str.as_str(),
            u64::try_from(time).unwrap(),
            num_slots,
            noise,
        );
    }
}

pub fn coeffs_to_slots_benchmark(
    reporter: &mut Reporter,
    field: FieldDescriptor,
    modulus: DBFVModulus,
    log2_alpha: f64,
) {
    let param_str = param_str(&field, modulus, log2_alpha);
    let mut rng = StdRng::from_seed([0; 32]);
    let log2_q = (log2_alpha.ceil() as usize)..(log2_alpha.ceil() as usize + 30);
    let num_slots = field.deg_k();
    let ring = RingContext::new(CpuBackend, field, log2_q, modulus.log2_t());
    // one-time, RNS-base-independent setup; see the note in
    // `RingContext::precompute_rns_factor_transforms()`
    ring.precompute_rns_factor_transforms();
    let base = ring.initial_rns_base();
    let sk = gen_sk(&ring, &mut || rng.next_u64());
    let crt_transform = UnsplitCRTTransform::new(1 << modulus.log2_t(), ring.field());
    let digits_gks = select_digits(base.len().div_ceil(2), base.len());
    let gks = gen_galois_keys(&ring, &sk, &crt_transform, &digits_gks, &mut || rng.next_u64(), log2_alpha);
    let ct = bfv_enc(
        &ring,
        &base,
        modulus.t(),
        &(0..ring.field().deg_lprime())
            .flat_map(|_| (0..ring.field().deg_k()).map(|i| if i == 1 { 1 } else { 0 }))
            .collect::<Vec<_>>(),
        &sk,
        &mut || rng.next_u64(),
        200.,
    );
    let initial_noise = dbfv_log2_noise_pbinf_norm(&ring, modulus, &ct, &sk).iter().map(|x| *x as usize).max().unwrap();
    let start = Instant::now();
    let res = coeffs_to_slots_dbfv(&ring, &crt_transform, modulus, ct, &gks);
    let end = Instant::now();
    let time = (end - start).as_millis();
    let final_noise = dbfv_log2_noise_pbinf_norm(&ring, modulus, &res, &sk).iter().map(|x| *x as usize).max().unwrap();
    let consumed = final_noise as i64 - initial_noise as i64;
    let result = dbfv_dec(&ring, modulus, &res, &sk);
    let scheme = if modulus.digits() > 1 { Scheme::CycloSubringDBFV } else { Scheme::CycloSubringBFV };
    let noise = NoiseInfo::new(
        crt_transform.reduce_t(modulus.t()).pack_slot_values(
            &ring,
            &(0..ring.field().deg_k()).map(|i| if i == 1 { 1 } else { 0 }).collect::<Vec<_>>(),
        ) == result,
        NoiseBudget::Consumed(consumed),
    );
    let dtype = DType::Uint(modulus.log2_t());
    reporter.report(
        Benchmark::LinTransform,
        dtype,
        scheme,
        param_str.as_str(),
        u64::try_from(time).unwrap(),
        num_slots,
        noise,
    );
}

pub fn benchmark_cyclotomic_subring_bfv(reporter: &mut Reporter) {
    let log2_t = 1;
    let modulus = DBFVModulus::new_incomplete(20, log2_t);
    coeffs_to_slots_benchmark(reporter, field_32768_ex(), modulus, 850.);
    coeffs_to_slots_benchmark(reporter, field_32768(), modulus, 850.);
    coeffs_to_slots_benchmark(reporter, field_49152_ex(), modulus, 1280.);
    coeffs_to_slots_benchmark(reporter, field_49152(), modulus, 1280.);
    bootstrap_benchmark(reporter, field_32768_ex(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_32768(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_49152_ex(), modulus, 1280., 8.5, 11);
    bootstrap_benchmark(reporter, field_49152(), modulus, 1280., 9.5, 12);
    add_mul_benchmark(reporter, field_32768_ex(), modulus, 850.);
    add_mul_benchmark(reporter, field_32768(), modulus, 850.);
    add_mul_benchmark(reporter, field_49152_ex(), modulus, 1280.);
    add_mul_benchmark(reporter, field_49152(), modulus, 1280.);
}

pub fn benchmark_cyclotomic_subring_dbfv(reporter: &mut Reporter) {
    let modulus = DBFVModulus::new(8, 8);
    bootstrap_benchmark(reporter, field_32768_ex(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_32768(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_49152_ex(), modulus, 1280., 8.5, 11);
    bootstrap_benchmark(reporter, field_49152(), modulus, 1280., 9.5, 12);
    add_mul_benchmark(reporter, field_32768_ex(), modulus, 850.);
    add_mul_benchmark(reporter, field_32768(), modulus, 850.);
    add_mul_benchmark(reporter, field_49152_ex(), modulus, 1280.);
    add_mul_benchmark(reporter, field_49152(), modulus, 1280.);

    let modulus = DBFVModulus::new_incomplete(6, 64);
    bootstrap_benchmark(reporter, field_32768_ex(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_32768(), modulus, 850., 6.1, 8);
    bootstrap_benchmark(reporter, field_49152_ex(), modulus, 1280., 8.5, 11);
    bootstrap_benchmark(reporter, field_49152(), modulus, 1280., 9.5, 12);
    add_mul_benchmark(reporter, field_32768_ex(), modulus, 850.);
    add_mul_benchmark(reporter, field_32768(), modulus, 850.);
    add_mul_benchmark(reporter, field_49152_ex(), modulus, 1280.);
    add_mul_benchmark(reporter, field_49152(), modulus, 1280.);
}
