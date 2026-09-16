//! Homomorphic digit extraction: evaluating the (hardcoded) digit-extraction polynomials on
//! BFV and dBFV ciphertexts, and the resulting homomorphic floor division for dBFV.
//!
//! The polynomial programs are ported verbatim from the reference implementation; they encode
//! straight-line programs computing (variants of) the digit extraction polynomials used for
//! BFV-style bootstrapping, i.e. polynomials `f` with `f(x) ≡ lift(x mod 2) mod 2^e`.

use feanor_math::integer::IntegerRingStore;
use tracing::instrument;

use crate::bfv::*;
use crate::bootstrap::RNS_FACTOR_SIZE_LOG2;
use crate::dbfv::*;
use crate::hal::Backend;
use crate::ring::{Basis, RingContext, ZZi64};

#[instrument(skip_all)]
pub fn hom_eval_digit_extract_circuit_dbfv<B: Backend>(
    ring: &RingContext<B>,
    modulus: DBFVModulus,
    ct: DBFVCiphertext<B>,
    subtract_intermediate: &mut [DBFVAccumulator<B>],
    subtract_final: &mut DBFVAccumulator<B>,
    mults: usize,
    rk: &KeySwitchKey<B>,
    rns_base_lens: &[usize],
    _debug_sk: Option<&SecretKey<B>>,
) {
    assert_eq!(rns_base_lens.len(), mults + 1);
    assert!(subtract_intermediate.len() < 1 << mults);
    assert!(subtract_intermediate.len() < 16);
    let mut subtract_intermediate = subtract_intermediate.into_iter().peekable();

    let modswitch_to = |x: DBFVCiphertext<B>, j| {
        let new_rns_base_len = rns_base_lens[j];
        let pre_rns_base_len = x.rns_base.len();
        if new_rns_base_len < pre_rns_base_len {
            let result = mod_switch_down(ring, &x, new_rns_base_len);
            result
        } else {
            assert_eq!(new_rns_base_len, pre_rns_base_len);
            x
        }
    };
    let sub_from = |target: &mut DBFVAccumulator<B>, value: &DBFVCiphertext<B>| {
        if target.rns_base().len() < value.rns_base.len() {
            assert_eq!(target.rns_base(), &value.rns_base[..target.rns_base().len()]);
            dbfv_acc_sub(ring, target, &mod_switch_down(ring, value, target.rns_base().len()));
        } else {
            assert_eq!(target.rns_base(), &*value.rns_base);
            dbfv_acc_sub(ring, target, value);
        }
    };

    let x2 = modswitch_to(dbfv_hom_square(ring, modulus, ct, rk, Basis::PeriodBasis), 1);
    for target in subtract_intermediate.by_ref().take(1) {
        sub_from(target, &x2);
    }
    if mults <= 1 {
        sub_from(subtract_final, &x2);
        return;
    }

    let x4 = modswitch_to(dbfv_hom_square(ring, modulus, ring.clone_els(&x2), rk, Basis::PeriodBasis), 2);
    for target in subtract_intermediate.by_ref().take(2) {
        sub_from(target, &x4);
    }
    if mults <= 2 {
        sub_from(subtract_final, &x4);
        return;
    }

    let x2 = modswitch_to(x2, 2);
    let mut val_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(ring, &mut val_acc, &[(&x2, 94), (&x4, 121)]);
    let val = dbfv_finish_accumulator(ring, &val_acc);
    drop(val_acc);
    let pre_x8 = modswitch_to(dbfv_hom_square(ring, modulus, val, rk, Basis::PeriodBasis), 3);
    let x2 = modswitch_to(x2, 3);
    let x4 = modswitch_to(x4, 3);

    if subtract_intermediate.peek().is_some() || mults <= 3 {
        let mut x8_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
        dbfv_acc_inner_product_scalar(ring, &mut x8_acc, &[(&x2, 112)]);
        dbfv_acc_add(ring, &mut x8_acc, &pre_x8);
        let x8 = dbfv_finish_accumulator(ring, &x8_acc);
        drop(x8_acc);
        for target in subtract_intermediate.by_ref().take(4) {
            sub_from(target, &x8);
        }
        if mults <= 3 {
            sub_from(subtract_final, &x8);
            return;
        }
    }

    let mut lhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(ring, &mut lhs_acc, &[(&x2, 12656), (&x4, 226), (&pre_x8, 113)]);
    let lhs = dbfv_finish_accumulator(ring, &lhs_acc);
    drop(lhs_acc);
    let mut rhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(ring, &mut rhs_acc, &[(&x2, 33720), (&x4, 2), (&pre_x8, 301)]);
    let rhs = dbfv_finish_accumulator(ring, &rhs_acc);
    drop(rhs_acc);
    let pre_x16 = modswitch_to(dbfv_hom_mul(ring, modulus, lhs, rhs, rk, Basis::PeriodBasis), 4);
    let x2 = modswitch_to(x2, 4);
    let x4 = modswitch_to(x4, 4);
    let pre_x8 = modswitch_to(pre_x8, 4);

    if subtract_intermediate.peek().is_some() || mults <= 4 {
        let mut x16_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
        dbfv_acc_inner_product_scalar(ring, &mut x16_acc, &[(&x2, 2535424), (&x4, 528), (&pre_x8, 22620)]);
        dbfv_acc_add(ring, &mut x16_acc, &pre_x16);
        let x16 = dbfv_finish_accumulator(ring, &x16_acc);
        drop(x16_acc);
        for target in subtract_intermediate.by_ref().take(8) {
            sub_from(target, &x16);
        }
        if mults <= 4 {
            sub_from(subtract_final, &x16);
            return;
        }
    }

    let mut lhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut lhs_acc,
        &[(&x2, 163551), (&x4, -64628), (&pre_x8, 88914), (&pre_x16, -168751)],
    );
    let lhs = dbfv_finish_accumulator(ring, &lhs_acc);
    drop(lhs_acc);
    let mut rhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut rhs_acc,
        &[(&x2, 142065), (&x4, 426380), (&pre_x8, 456690), (&pre_x16, -290191)],
    );
    let rhs = dbfv_finish_accumulator(ring, &rhs_acc);
    drop(rhs_acc);
    let pre_x23 = modswitch_to(dbfv_hom_mul(ring, modulus, lhs, rhs, rk, Basis::PeriodBasis), 5);

    let x2 = modswitch_to(x2, 5);
    let x4 = modswitch_to(x4, 5);
    let pre_x8 = modswitch_to(pre_x8, 5);
    let pre_x16 = modswitch_to(pre_x16, 5);
    let mut val_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(ring, &mut val_acc, &[(&x4, 460241), (&pre_x8, -520176), (&pre_x16, 272152)]);
    dbfv_acc_add(ring, &mut val_acc, &pre_x23);
    let val = dbfv_finish_accumulator(ring, &val_acc);
    drop(val_acc);
    let s1 = modswitch_to(dbfv_hom_square(ring, modulus, val, rk, Basis::PeriodBasis), 6);

    let x2 = modswitch_to(x2, 6);
    let x4 = modswitch_to(x4, 6);
    let pre_x8 = modswitch_to(pre_x8, 6);
    let pre_x16 = modswitch_to(pre_x16, 6);
    let pre_x23 = modswitch_to(pre_x23, 6);
    let mut val_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut val_acc,
        &[
            (&x4, -32106558060291480110540),
            (&pre_x8, 64784965186867951517504),
            (&pre_x16, -70466562361570490438944),
            (&pre_x23, 403089028126664267137012),
        ],
    );
    dbfv_acc_add(ring, &mut val_acc, &s1);
    let val = dbfv_finish_accumulator(ring, &val_acc);
    drop(val_acc);
    let s2 = modswitch_to(dbfv_hom_mul(ring, modulus, val, ring.clone_els(&s1), rk, Basis::PeriodBasis), 7);

    let x2 = modswitch_to(x2, 7);
    let x4 = modswitch_to(x4, 7);
    let pre_x8 = modswitch_to(pre_x8, 7);
    let pre_x16 = modswitch_to(pre_x16, 7);
    let pre_x23 = modswitch_to(pre_x23, 7);
    let s1 = modswitch_to(s1, 7);
    let mut lhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut lhs_acc,
        &[
            (&x4, -306594109877109866124476),
            (&pre_x8, 55340232221151557156928),
            (&pre_x16, 66482065641637241715808),
            (&pre_x23, -201964177491009025911804),
            (&s1, 982289121925033623516),
        ],
    );
    dbfv_acc_add(ring, &mut lhs_acc, &s2);
    dbfv_acc_add_constant(ring, &mut lhs_acc, 152453116397172589318468);
    let lhs = dbfv_finish_accumulator(ring, &lhs_acc);
    drop(lhs_acc);
    let mut rhs_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut rhs_acc,
        &[
            (&x4, -541458055423539766375756),
            (&pre_x8, -321268494787747983014080),
            (&pre_x16, 389373873907872951793376),
            (&pre_x23, -168833825134626671122316),
            (&s1, 982289121925033623520),
        ],
    );
    dbfv_acc_add(ring, &mut rhs_acc, &s2);
    let rhs = dbfv_finish_accumulator(ring, &rhs_acc);
    drop(rhs_acc);
    let s3 = modswitch_to(dbfv_hom_mul(ring, modulus, lhs, rhs, rk, Basis::PeriodBasis), 8);

    let x2 = modswitch_to(x2, 8);
    let x4 = modswitch_to(x4, 8);
    let pre_x8 = modswitch_to(pre_x8, 8);
    let pre_x16 = modswitch_to(pre_x16, 8);
    let pre_x23 = modswitch_to(pre_x23, 8);
    let s1 = modswitch_to(s1, 8);

    let mut res_acc = dbfv_allocate_accumulator(ring, modulus, &x2.rns_base, Basis::PeriodBasis);
    dbfv_acc_inner_product_scalar(
        ring,
        &mut res_acc,
        &[
            (&x4, 138276793813981485937712),
            (&pre_x8, -394317601587992658717952),
            (&pre_x16, 277439031009004544298112),
            (&pre_x23, -62423781945432606733008),
            (&s1, 376442706312192718108336),
        ],
    );
    dbfv_acc_add(ring, &mut res_acc, &s3);
    let res = dbfv_finish_accumulator(ring, &res_acc);
    sub_from(subtract_final, &res);
}

/// Homomorphically computes `floor(x / 2^v)` where `v = log2(t_initial) - log2(t_final)`, by
/// repeatedly extracting and subtracting the lowest bit. Input and output in CRT basis; the
/// output is w.r.t. `final_modulus`.
///
/// # Noise growth and modulus-switching
///
/// To get better performance, we use dynamic scaling. We make the following assumptions:
///  - after the first multiplication, the noise in the input ciphertext is greater than the
///    additive noise amount introduced by keyswitching
///  - thus, whenever multiplications increase the relative noise by 60 bits, we can drop an RNS
///    factor
///
/// The exact algorithm uses the digit circuit as implemented by
/// [`hom_eval_digit_extract_circuit_dbfv_new()`]. Concretely, we compute a sequence of values
/// ```text
///   x_00 = input
///   x_i(j + 1) = degree-2 computation in x_i0, ..., x_ij
///   x_(i + 1)0 = sum/difference of x_jk for 0 <= j <= i, where k = ceil(log2(i + 2 - j))
/// ```
/// such that `x_ij` is the `i`-th digit of the input modulo `2^(roughly 2^j)`.
///
/// This allows us to compute the noise levels (above baseline) in advance:
/// ```text
///   n_00 = 0
///   n_i(j + 1) = n_ij + hom_mul_noise + scalar_mul_noise
///   n_(i + 1)0 = n_i1
/// ```
/// This gives rise to the `rns_base_lens` value below.
#[instrument(skip_all)]
pub fn hom_floordiv_dbfv<B: Backend>(
    ring: &RingContext<B>,
    initial_modulus: DBFVModulus,
    final_modulus: DBFVModulus,
    ct: DBFVCiphertext<B>,
    rk: &KeySwitchKey<B>,
    debug_sk: Option<&SecretKey<B>>,
) -> DBFVCiphertext<B> {
    let v = initial_modulus.log2_t() - final_modulus.log2_t();
    let e = initial_modulus.log2_t();
    let mults = if e <= 16 { ZZi64.abs_log2_ceil(&(e as i64)).unwrap() } else { 8 };
    assert_eq!(final_modulus.log2_b(), initial_modulus.log2_b());
    // the circuit can feed the extracted bit to at most 1 + 2 + 4 + 8 = 15 pending values (see
    // [`hom_eval_digit_extract_circuit_dbfv_new()`]), i.e. v - 1 <= 15
    if v > 16 {
        unimplemented!("the polynomial program only supports eliminating up to 16 digits");
    } else if initial_modulus.log2_t() > 80 {
        unimplemented!("the polynomial program only supports precision up to 80 bits");
    }
    let mut noises = (0..v).map(|_i| (0..=mults).map(|_j| 0.).collect::<Vec<_>>()).collect::<Vec<_>>();
    for i in 0..v {
        if i > 0 {
            noises[i][0] = noises[i - 1][1];
        }
        for j in 1..=mults {
            noises[i][j] = ring
                .field()
                .noise_growth_hom_mul_scalar_lower_estimate(i128::min(initial_modulus.b(), initial_modulus.t()))
                + ring.field().noise_growth_hom_mul_lower_estimate(i128::min(initial_modulus.b(), initial_modulus.t()))
                + noises[i][j - 1];
        }
    }
    // as in [`crate::homsgn`], these are purely multiplicative growth estimates: the absolute
    // noise contributed by each relinearization with `rk` (see
    // [`KeySwitchKey::log2_key_switch_noise()`]) is not modelled, and dominates once `rk` is
    // generated with an error well above `MIN_SIGMA`
    let baseline_rns_base_len = ct.rns_base.len();
    let rns_base_lens = noises
        .into_iter()
        .map(|noises| {
            noises
                .into_iter()
                .map(|n| baseline_rns_base_len.checked_sub((n / RNS_FACTOR_SIZE_LOG2).floor() as usize).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let add_to = |target: &mut DBFVAccumulator<B>, value: &DBFVCiphertext<B>| {
        if target.rns_base().len() < value.rns_base.len() {
            assert_eq!(target.rns_base(), &value.rns_base[..target.rns_base().len()]);
            dbfv_acc_add(ring, target, &mod_switch_down(ring, value, target.rns_base().len()));
        } else {
            assert_eq!(target.rns_base(), &*value.rns_base);
            dbfv_acc_add(ring, target, value);
        }
    };
    let result_rns_base_len = rns_base_lens.last().unwrap().last().unwrap();
    let mut result =
        dbfv_allocate_accumulator(ring, initial_modulus, &ct.rns_base[..*result_rns_base_len], Basis::PeriodBasis);
    add_to(&mut result, &ct);
    let mut parts = (0..v)
        .map(|i| {
            dbfv_allocate_accumulator(ring, initial_modulus, &ct.rns_base[..rns_base_lens[i][0]], Basis::PeriodBasis)
        })
        .collect::<Vec<_>>();
    for part in &mut parts {
        add_to(part, &ct);
    }

    for i in 0..v {
        let current_acc = parts.remove(0);
        let current = dbfv_finish_accumulator(ring, &current_acc);
        drop(current_acc);
        let current = dbfv_exact_div_pow2(ring, initial_modulus, current, i);
        let current_modulus = DBFVModulus::new_incomplete(initial_modulus.log2_b(), initial_modulus.log2_t() - i);

        hom_eval_digit_extract_circuit_dbfv(
            ring,
            current_modulus,
            current,
            &mut parts[..],
            &mut result,
            mults,
            rk,
            &rns_base_lens[i],
            debug_sk,
        );
    }
    let result = dbfv_finish_accumulator(ring, &result);
    return dbfv_exact_div_pow2(ring, initial_modulus, result, v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::field_192;
    use crate::hal::cpu::CpuBackend;
    use crate::ring::{RingContext, cmod};

    fn pmod(a: i128, modulus: i128) -> i128 { a.rem_euclid(modulus) }

    #[test]
    fn test_hom_floordiv_dbfv() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 22u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 800..820, 16);
        let t = 1i128 << 16;
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);

        let mut ct = bfv_enc(&ring, &base, t, &encode_scalar(&ring, 2025), &sk, &mut rng, log2_alpha);
        ring.crt_to_pb(&mut ct);
        let mut res = hom_floordiv_dbfv(&ring, DBFVModulus::new(4, 4), DBFVModulus::new(4, 3), ct, &rk, None);
        ring.pb_to_crt(&mut res);
        let result = dbfv_dec(&ring, DBFVModulus::new(4, 3), &res, &sk);
        assert_eq!(encode_scalar(&ring, 126), result);
    }

    /// the same with a precision `> 16` bits, which selects the full 8-multiplication circuit
    #[test]
    fn test_hom_floordiv_dbfv_deep_circuit() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 24u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 900..920, 36);
        let t = 1i128 << 36;
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let initial_modulus = DBFVModulus::new_incomplete(8, 36);

        for v in [1usize, 5, 10, 12] {
            let final_modulus = DBFVModulus::new_incomplete(8, 36 - v);
            let value = 12345678901i128;
            let mut ct = bfv_enc(&ring, &base, t, &encode_scalar(&ring, value), &sk, &mut rng, log2_alpha);
            ring.crt_to_pb(&mut ct);
            let mut res = hom_floordiv_dbfv(&ring, initial_modulus, final_modulus, ct, &rk, Some(&sk));
            ring.pb_to_crt(&mut res);
            let result = decode_scalar(&ring, final_modulus.t(), &dbfv_dec(&ring, final_modulus, &res, &sk));
            let expected = cmod(pmod(value, t) >> v, final_modulus.t());
            assert_eq!(expected, result, "v = {}", v);
        }
    }

    /// the floor division on an input as it occurs inside a bootstrap: several dBFV digits, and a
    /// non-scalar (but `K`-valued) plaintext, whose slot values are floor-divided individually
    #[test]
    fn test_hom_floordiv_dbfv_multi_digit_input() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 25u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 900..920, 36);
        let t = 1i128 << 36;
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let initial_modulus = DBFVModulus::new_incomplete(8, 36);
        let transform = crate::bootstrap::homcrt::UnsplitCRTTransform::new(t, ring.field());
        let slot_values = (0..ring.slot_count()).map(|i| cmod(i as i128 * 1234567 + 777, t)).collect::<Vec<_>>();

        for v in [1usize, 12] {
            let final_modulus = DBFVModulus::new_incomplete(8, 36 - v);
            let ptxt = transform.pack_slot_values(&ring, &slot_values);
            let mut ct = bfv_enc(&ring, &base, t, &ptxt, &sk, &mut rng, log2_alpha);
            ring.crt_to_pb(&mut ct);
            // the input of a bootstrap's floor division carries all digits
            let ct = dbfv_reduce(&ring, initial_modulus, ct);
            assert_eq!(initial_modulus.digits(), dbfv_digit_count(&ct));
            let mut res = hom_floordiv_dbfv(&ring, initial_modulus, final_modulus, ct, &rk, Some(&sk));
            ring.pb_to_crt(&mut res);
            let out = transform
                .reduce_t(final_modulus.t())
                .extract_slot_values(&ring, &dbfv_dec(&ring, final_modulus, &res, &sk));
            let expected =
                slot_values.iter().map(|x| cmod(x.rem_euclid(t) >> v, final_modulus.t())).collect::<Vec<_>>();
            assert_eq!(expected, out, "v = {}", v);
        }
    }

    /// eliminating more than a whole digit worth of bits (`v > log2_b`), which requires the
    /// exact divisions to drop several digits at once and (for `v > 7`) the `x16` output of the
    /// digit-extraction circuit
    #[test]
    fn test_hom_floordiv_dbfv_multiple_digits() {
        feanor_tracing::DelayedLogger::init_test();
        let mut state = 23u64;
        let mut rng = move || splitmix64(&mut state);
        let ring = RingContext::new(CpuBackend, field_192(), 900..920, 16);
        let t = 1i128 << 16;
        let base = ring.initial_rns_base();
        let log2_alpha = log2_alpha_for_sigma(&ring, &base, MIN_SIGMA);
        let sk = gen_sk(&ring, &mut rng);
        let digits = select_digits(base.len(), base.len());
        let rk = gen_rk(&ring, &sk, &digits, &mut rng, log2_alpha);
        let initial_modulus = DBFVModulus::new(4, 4);

        for v in [5usize, 6, 9] {
            let final_modulus = DBFVModulus::new_incomplete(4, 16 - v);
            let mut ct = bfv_enc(&ring, &base, t, &encode_scalar(&ring, 40529), &sk, &mut rng, log2_alpha);
            ring.crt_to_pb(&mut ct);
            let mut res = hom_floordiv_dbfv(&ring, initial_modulus, final_modulus, ct, &rk, Some(&sk));
            ring.pb_to_crt(&mut res);
            let result = decode_scalar(&ring, final_modulus.t(), &dbfv_dec(&ring, final_modulus, &res, &sk));
            let expected = cmod(pmod(40529, t) >> v, final_modulus.t());
            assert_eq!(expected, result, "v = {}", v);
        }
    }
}
