use std::ops::Range;
use std::time::Instant;

use cyclosubfield_dbfv::ring::{ZZbig, ZZi64};
use feanor_math::algorithms::discrete_log::multiplicative_order;
use feanor_math::delegate::DelegateRing;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{IntegerRingStore, int_cast};
use feanor_math::ring::{RingExtensionStore, RingStore};
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::seq::VectorView;
use fheanor::bfv::bootstrap::{SparseKeyEncapsulationKey, ThinBootstrapper};
use fheanor::bfv::eval::{EncodedBFVPlaintextRing, EncodedBFVPlaintextRingBase};
use fheanor::bfv::{BFVInstantiation, CompositeBFV, SecretKeyDistribution};
use fheanor::circuit::create_circuit_cached;
use fheanor::filename_keys;
use fheanor::gadget_product::digits::RNSGadgetVectorDigitIndices;
use fheanor::lin_transform::composite;
use fheanor::number_ring::hypercube::isomorphism::HypercubeIsomorphism;
use fheanor::number_ring::hypercube::structure::HypercubeStructure;
use fheanor::number_ring::{NumberRingDescriptor, NumberRingQuotientStore};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{Benchmark, CACHE_DIR, DType, NoiseBudget, NoiseInfo, Reporter, Scheme};

pub fn bootstrap_benchmark(reporter: &mut Reporter, (m1, m2): (usize, usize), log2_q: Range<usize>, v: usize) {
    let mut rng = StdRng::from_seed([0; 32]);
    let inst = CompositeBFV::new(m1, m2);
    let t = 2;
    let param_str =
        format!("n = {}, log2(q) = {}, t = {}", inst.number_ring().rank(), (log2_q.end + log2_q.start) / 2, t);
    let ring = Zn::new((m1 * m2) as u64);
    let f = multiplicative_order(ring.int_hom().map(2), &ring);
    let P = inst.create_plaintext_ring(int_cast(t, ZZbig, ZZi64));
    let (C, C_mul) = inst.create_ciphertext_rings(log2_q);
    let gk_digits = RNSGadgetVectorDigitIndices::select_digits(C.base_ring().len().div_ceil(2), C.base_ring().len());
    let rk_digits = RNSGadgetVectorDigitIndices::select_digits(C.base_ring().len().div_ceil(4), C.base_ring().len());
    let bootstrapper = ThinBootstrapper::build_odd(&inst, &P, &C, v, None, 4, &gk_digits, Some(CACHE_DIR));

    let sk = CompositeBFV::gen_sk(&C, &mut rng, SecretKeyDistribution::UniformTernary);
    let gk = CompositeBFV::gen_gks(&C, &mut rng, &sk, bootstrapper.required_galois_keys(&P), &gk_digits, 3.2);
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, &rk_digits, 3.2);
    let sparse_key_encaps = SparseKeyEncapsulationKey::<CompositeBFV>::new(
        bootstrapper.intermediate_plaintext_ring(),
        &C,
        &sk,
        2,
        32,
        &mut rng,
        3.2,
    );

    let m = P.one();
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2);
    let start = Instant::now();
    let ct_result = bootstrapper.bootstrap_thin(&C, &C_mul, ct, &rk, &gk, Some(&sparse_key_encaps), None);
    let end = Instant::now();
    let time = (end - start).as_millis();
    let noise_budget = CompositeBFV::noise_budget(&P, &C, &ct_result, &sk);
    let result = CompositeBFV::dec(&P, &C, ct_result, &sk);
    let noise = NoiseInfo::new(P.is_one(&result), NoiseBudget::Left(noise_budget as i64));
    reporter.report(
        Benchmark::Bootstrap,
        DType::Uint(ZZi64.abs_log2_ceil(&t).unwrap()),
        Scheme::CycloRingBFV,
        &param_str,
        u64::try_from(time).unwrap(),
        P.rank() / f as usize,
        noise,
    );
}

pub fn coeffs_to_slots_benchmark(reporter: &mut Reporter, (m1, m2): (usize, usize), log2_q: Range<usize>) {
    let mut rng = StdRng::from_seed([0; 32]);
    let inst = CompositeBFV::new(m1, m2);
    let t = 2;
    let param_str =
        format!("n = {}, log2(q) = {}, t = {}", inst.number_ring().rank(), (log2_q.end + log2_q.start) / 2, t);
    let ring = Zn::new((m1 * m2) as u64);
    let f = multiplicative_order(ring.int_hom().map(2), &ring);
    let P = inst.create_plaintext_ring(int_cast(t, ZZbig, ZZi64));
    let (C, _) = inst.create_ciphertext_rings(log2_q);
    let gk_digits = RNSGadgetVectorDigitIndices::select_digits(C.base_ring().len().div_ceil(2), C.base_ring().len());
    let H = HypercubeIsomorphism::new(
        &P,
        &HypercubeStructure::halevi_shoup_hypercube(P.acting_galois_group(), int_cast(2, ZZbig, ZZi64)),
        Some(CACHE_DIR),
    );
    let circuit = create_circuit_cached(
        &P,
        &filename_keys![coeffs2slots, m: P.number_ring().m(), p: 2, e: ZZi64.abs_log2_ceil(&t).unwrap(), lvl: 4, cm: &inst.cost_model()],
        Some(CACHE_DIR),
        || composite::powcoeffs_to_slots_thin(&H, 4, &inst.cost_model()),
    );

    let sk = CompositeBFV::gen_sk(&C, &mut rng, SecretKeyDistribution::UniformTernary);
    let gks = CompositeBFV::gen_gks(
        &C,
        &mut rng,
        &sk,
        circuit.required_galois_keys(P.acting_galois_group()),
        &gk_digits,
        3.2,
    );

    // Encode the circuit's constants into the ciphertext ring once, up front. Every constant
    // is a full-rank element of `R/tR`, so using one straight from the plaintext ring means
    // lifting all `n` coefficients into the RNS base and running a forward NTT over the whole
    // base (~56 ms at m = 42799) — and `hom_mul_plain()` redoes that on *every* use, which for
    // this circuit is 376 times. That setup belongs to key generation, not to the transform:
    // `ThinBootstrapper` likewise converts both of its transforms to `EncodedBFVPlaintextRing`
    // at construction time, so this also makes the measurement consistent with the bootstrap
    // benchmark, which calls the very same circuit.
    let start = Instant::now();
    let EP = EncodedBFVPlaintextRingBase::<CompositeBFV>::new(P.clone(), C.clone());
    let circuit = circuit.change_ring_uniform(|c| {
        c.change_ring::<EncodedBFVPlaintextRingBase<CompositeBFV>, _>(|x| EP.get_ring().rev_delegate(x))
    });
    println!("  encoded the transform constants in {:.0} s", start.elapsed().as_secs_f64());

    let m = P.one();
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2);
    let ct = CompositeBFV::hom_galois(&C, ct, &gks[0].0, &gks[0].1);
    let initial_noise_budget = CompositeBFV::noise_budget(&P, &C, &ct, &sk);
    let start = Instant::now();
    let [ct_result] = circuit
        .evaluate_bfv::<CompositeBFV, &EncodedBFVPlaintextRing<CompositeBFV>>(
            &EP,
            &P,
            &C,
            None,
            &[ct],
            None,
            &gks,
            None,
        )
        .try_into()
        .ok()
        .unwrap();
    let end = Instant::now();
    let time = (end - start).as_millis();
    let final_noise_budget = CompositeBFV::noise_budget(&P, &C, &ct_result, &sk);
    let result = CompositeBFV::dec(&P, &C, ct_result, &sk);
    let noise = NoiseInfo::new(
        H.get_slot_values(&result)
            .enumerate()
            .all(|(i, x)| if i == 0 { H.slot_ring().is_one(&x) } else { H.slot_ring().is_zero(&x) }),
        NoiseBudget::Consumed(initial_noise_budget as i64 - final_noise_budget as i64),
    );
    reporter.report(
        Benchmark::LinTransform,
        DType::Uint(ZZi64.abs_log2_ceil(&t).unwrap()),
        Scheme::CycloRingBFV,
        &param_str,
        u64::try_from(time).unwrap(),
        P.rank() / f as usize,
        noise,
    );
}

pub fn add_mul_benchmark(reporter: &mut Reporter, (m1, m2): (usize, usize), log2_q: Range<usize>) {
    let mut rng = StdRng::from_seed([0; 32]);
    let inst = CompositeBFV::new(m1, m2);
    let t = 2;
    let param_str =
        format!("n = {}, log2(q) = {}, t = {}", inst.number_ring().rank(), (log2_q.end + log2_q.start) / 2, t);
    let ring = Zn::new((m1 * m2) as u64);
    let f = multiplicative_order(ring.int_hom().map(2), &ring);
    let P = inst.create_plaintext_ring(int_cast(t, ZZbig, ZZi64));
    let (C, C_mul) = inst.create_ciphertext_rings(log2_q);
    let rk_digits = RNSGadgetVectorDigitIndices::select_digits(C.base_ring().len().div_ceil(4), C.base_ring().len());
    let sk = CompositeBFV::gen_sk(&C, &mut rng, SecretKeyDistribution::UniformTernary);
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, &rk_digits, 3.2);

    let ct_pre = CompositeBFV::enc_sym(&P, &C, &mut rng, &P.one(), &sk, 3.2);
    let ct1 = CompositeBFV::hom_mul(
        &P,
        &C,
        &C_mul,
        CompositeBFV::clone_ct(&C, &ct_pre),
        CompositeBFV::clone_ct(&C, &ct_pre),
        &rk,
    );
    let ct2 = CompositeBFV::hom_mul(
        &P,
        &C,
        &C_mul,
        CompositeBFV::clone_ct(&C, &ct_pre),
        CompositeBFV::clone_ct(&C, &ct_pre),
        &rk,
    );
    let initial_noise = CompositeBFV::noise_budget(&P, &C, &ct1, &sk);

    {
        let start = Instant::now();
        let ct_result = CompositeBFV::hom_mul(
            &P,
            &C,
            &C_mul,
            CompositeBFV::clone_ct(&C, &ct1),
            CompositeBFV::clone_ct(&C, &ct2),
            &rk,
        );
        let end = Instant::now();
        let time = (end - start).as_millis();
        let noise_used = initial_noise as i64 - CompositeBFV::noise_budget(&P, &C, &ct_result, &sk) as i64;
        let result = CompositeBFV::dec(&P, &C, ct_result, &sk);
        let noise = NoiseInfo::new(P.is_one(&result), NoiseBudget::Consumed(noise_used));
        reporter.report(
            Benchmark::Mul,
            DType::Uint(ZZi64.abs_log2_ceil(&t).unwrap()),
            Scheme::CycloRingBFV,
            &param_str,
            u64::try_from(time).unwrap(),
            P.rank() / f as usize,
            noise,
        );
    }
    {
        let start = Instant::now();
        let ct_result = CompositeBFV::hom_add(&C, ct1, &ct2);
        let end = Instant::now();
        let time = (end - start).as_millis();
        let noise_used = initial_noise as i64 - CompositeBFV::noise_budget(&P, &C, &ct_result, &sk) as i64;
        let result = CompositeBFV::dec(&P, &C, ct_result, &sk);
        let noise = NoiseInfo::new(P.eq_el(&result, &P.int_hom().map(2)), NoiseBudget::Consumed(noise_used));
        reporter.report(
            Benchmark::Add,
            DType::Uint(ZZi64.abs_log2_ceil(&t).unwrap()),
            Scheme::CycloRingBFV,
            &param_str,
            u64::try_from(time).unwrap(),
            P.rank() / f as usize,
            noise,
        );
    }
}

pub fn benchmark_cyclotomic_bfv(reporter: &mut Reporter) {
    coeffs_to_slots_benchmark(reporter, (37, 949), 805..820);
    coeffs_to_slots_benchmark(reporter, (337, 127), 1100..1120);
    add_mul_benchmark(reporter, (37, 949), 805..820);
    add_mul_benchmark(reporter, (337, 127), 1100..1120);
    bootstrap_benchmark(reporter, (37, 949), 805..820, 6);
    bootstrap_benchmark(reporter, (337, 127), 1100..1120, 6);
}
