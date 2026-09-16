use std::time::Instant;

use tfhe::prelude::{FheDecrypt, FheTryEncrypt};
use tfhe::{ConfigBuilder, FheUint64, generate_keys, set_server_key};

use crate::{Benchmark, DType, NoiseInfo, Reporter, Scheme};

pub fn benchmark_tfhe(reporter: &mut Reporter) {
    let config = ConfigBuilder::default().build();
    let config_json = serde_json::to_value(&config).unwrap();
    let lwe_dim = config_json["inner"]["block_parameters"]["Standard"]["PBS"]["lwe_dimension"].as_u64().unwrap();
    let rlwe_dim = config_json["inner"]["block_parameters"]["Standard"]["PBS"]["polynomial_size"].as_u64().unwrap();
    let params_str = format!("n = {}, N = {}", lwe_dim, rlwe_dim);
    let (sk, evk) = generate_keys(config);
    set_server_key(evk);

    let a = FheUint64::try_encrypt(17u64, &sk).unwrap();
    let b = FheUint64::try_encrypt(256u64, &sk).unwrap();

    _ = &a + &b;
    _ = &a * &b;

    let start = Instant::now();
    let c = &a + &b;
    let end = Instant::now();
    assert_eq!(17 + 256, FheDecrypt::<u64>::decrypt(&c, &sk));
    reporter.report(
        Benchmark::Add,
        DType::Uint(64),
        Scheme::TFHE,
        &params_str,
        (end - start).as_millis() as u64,
        1,
        NoiseInfo::not_applicable(),
    );

    let start = Instant::now();
    let c = &a * &b;
    let end = Instant::now();
    assert_eq!(17 * 256, FheDecrypt::<u64>::decrypt(&c, &sk));
    reporter.report(
        Benchmark::Mul,
        DType::Uint(64),
        Scheme::TFHE,
        &params_str,
        (end - start).as_millis() as u64,
        1,
        NoiseInfo::not_applicable(),
    );
}
