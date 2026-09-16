use std::alloc::Global;
use std::mem::swap;

use feanor_math::algorithms::convolution::DefaultConvolutionRing;
use feanor_math::algorithms::cyclotomic::get_prim_root_of_unity;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::linsolve::LinSolveRingStore;
use feanor_math::group::*;
use feanor_math::matrix::*;
use feanor_math::prelude::*;
use feanor_math::ring_impls::extension::extension_impl::{MonogenicExtensionImpl, MonogenicExtensionImplBase};
use feanor_math::ring_impls::extension::galois_field::GaloisField;
use feanor_math::ring_impls::extension::*;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use feanor_math::ring_impls::zn::zn_big::ZnGB;
use feanor_math::ring_impls::zn::*;
use feanor_math::seq::{VectorFn, VectorView};
use tracing::{Level, instrument, span};

use crate::fielddesc::*;
use crate::ring::{RingContext, RingElementData};
use crate::*;

type GaloisRing = MonogenicExtensionImpl<ZnGB<BigIntRing>>;
type GaloisRingBase = MonogenicExtensionImplBase<ZnGB<BigIntRing>>;

/// Represents the CRT transform for O_K/(2).
///
/// In contrast to [`crate::ntt::PlainCRTTransform`], the prime `(2)` usually doesn't split
/// completely in M or L, thus this transform is significantly more complicated.
pub struct UnsplitCRTTransform {
    t: i128,
    factors: Vec<CRTTransformFactor>,
}

impl UnsplitCRTTransform {
    pub fn new(t: i128, field_descriptor: &FieldDescriptor) -> Self {
        let mut factors = Vec::new();
        for p in field_descriptor.m_K_factors() {
            factors.push(CRTTransformFactor::new(t, *p, field_descriptor));
        }
        return Self { t, factors };
    }

    pub fn t(&self) -> i128 { self.t }

    pub fn factors(&self) -> &[CRTTransformFactor] { &self.factors }

    pub fn reduce_t(&self, new_t: i128) -> Self {
        assert!(self.t % new_t == 0);
        Self {
            factors: self.factors.iter().map(|f| f.reduce_t(new_t)).collect(),
            t: new_t,
        }
    }

    #[instrument(skip_all)]
    pub fn extract_slot_values(&self, ring: &RingContext, ptxt_pb: &[i128]) -> Vec<i128> {
        assert_eq!(ring.n(), ptxt_pb.len());
        let field_descriptor = ring.field_descriptor();
        let deg_K = field_descriptor.deg_K();
        for i in 1..field_descriptor.deg_Lʼ() {
            assert_eq!(
                &ptxt_pb[..deg_K],
                &ptxt_pb[(i * deg_K)..((i + 1) * deg_K)],
                "input plaintext is not contained in K"
            );
        }

        assert!(
            self.factors.len() * (2 * ZZi128.abs_log2_ceil(&self.t).unwrap() + ring.log2_pb_expansion_factor_bound())
                < ZZbig.abs_log2_floor(ring.rns_base().modulus()).unwrap()
        );
        let mut embedded = ring.plaintext_into_ciphertext_ring_unscaled_pb(self.t, ptxt_pb);
        for factor in &self.factors {
            embedded = factor.apply_plain(ring, embedded);
        }
        let mut result = ring.plaintext_from_ciphertext_ring_unscaled_pb(self.t, embedded);
        for i in 1..field_descriptor.deg_Lʼ() {
            debug_assert_eq!(&result[..deg_K], &result[(i * deg_K)..((i + 1) * deg_K)]);
        }
        result.truncate(deg_K);
        for x in &mut result {
            *x = cmod(
                *x * ring.field_descriptor().scale_M_K_to_M_period_basis() as i128,
                self.t,
            );
        }
        return result;
    }

    #[instrument(skip_all)]
    pub fn pack_slot_values(&self, ring: &RingContext, slot_values: &[i128]) -> Vec<i128> {
        let field_descriptor = ring.field_descriptor();
        let deg_K = field_descriptor.deg_K();
        assert_eq!(deg_K, slot_values.len());

        assert!(
            self.factors.len() * (2 * ZZi128.abs_log2_ceil(&self.t).unwrap() + ring.log2_pb_expansion_factor_bound())
                < ZZbig.abs_log2_floor(ring.rns_base().modulus()).unwrap()
        );
        let mut embedded = ring.plaintext_into_ciphertext_ring_unscaled_pb(
            self.t,
            &(0..field_descriptor.deg_Lʼ())
                .flat_map(|_| slot_values.iter().copied())
                .collect::<Vec<_>>(),
        );
        for factor in &self.factors {
            embedded = factor.apply_inv_plain(ring, embedded);
        }
        let mut result = ring.plaintext_from_ciphertext_ring_unscaled_pb(self.t, embedded);
        for i in 1..field_descriptor.deg_Lʼ() {
            debug_assert_eq!(
                &result[..deg_K],
                &result[(i * deg_K)..((i + 1) * deg_K)],
                "input plaintext is not contained in K"
            );
        }
        for x in &mut result {
            *x = cmod(
                *x * ring.field_descriptor().scale_M_K_to_M_period_basis() as i128,
                self.t,
            );
        }
        return result;
    }
}

/// Represents the CRT transform factor u_p for a prime p|m_K.
///
/// The CRT transform factor g_p is the map that represents the coefficients-to-slots transform on
/// `F_q ⊗ Z[ζ_p]/(2)`. It is defined as
/// ```text
///   g_p = \sum_σ (σ^-1(ζ) mod \p) σ    in F_q[G_p]
/// ```
/// where G_p is the Galois group of the p-th cyclotomic number field `Q(ζ_p)`, σ ranges through
/// G_p, F_q is a finite field of degree `f_p = ord_(Z/pZ)*(2)`, and \p is some prime of `F_q ⊗
/// Z[ζ_p]/(2)` over 2. Note that `F_q ⊗ Z[ζ_p]/(2) ≌ F_q[G_p]` as G_p-algebras.
///
/// g_p is then mapped into `F_q[G_L]` by first embedding it into `F_q ⊗ O_(M_K)/(2)`, and then
/// taking it modulo `Gal(M_K|K)`. We denote this image by u_p. In other words, u_p is the image of
/// g_p under the map
/// ```text
///   F_q[Gal(M_K|Q)] -> F_q[G_L],    \sum_σ a_σ σ -> \sum_σ ( \sum_τ a_τ ) σ
/// ```
/// where σ runs through `Gal(M_K|Q)` resp. G_L, and τ runs through `σ Gal(M_K|K)`. Note that this
/// means that the coefficients are actually in a subfield F' of F_q, which has degree only
/// ```text
///   d_p = f_p / gcd(f_p, #H_p),    where H_p = Gal(M_K|K) \cap (Z/pZ)*
/// ```
pub struct CRTTransformFactor {
    p: i64,
    t: i128,
    /// for each coefficient, store its coefficients w.r.t. the period basis of Lʼ induced by M
    /// (i.e. already sign corrected)
    u_p: Vec<Vec<i128>>,
    /// for each coefficient, store its coefficients w.r.t. the period basis of Lʼ induced by M
    /// (i.e. already sign corrected)
    inv_u_p: Vec<Vec<i128>>,
    gen_Jp_dlog: Vec<i64>,
    gen_Jp: GEl<MultGroup<Zn64B>>,
}

impl CRTTransformFactor {
    #[instrument(skip_all)]
    pub fn new(t: i128, p: i64, field_descriptor: &FieldDescriptor) -> Self {
        assert!(field_descriptor.m() as i64 % p == 0);
        let (two, e) = is_prime_power(ZZi128, &t).unwrap();
        assert!(two == 2);

        let f_p = field_descriptor.f_p(p);
        let GF = GaloisField::new(
            2,
            ZZi64.lcm(&(f_p as i64), &(field_descriptor.deg_Mʼ() as i64)) as usize,
        );
        let base_ring = ZnGB::new(ZZbig, ZZbig.power_of_two(e));
        let GR = RingValue::from(GF.get_ring().galois_ring_with_convolution(
            base_ring.clone(),
            Global,
            DefaultConvolutionRing::create_default_convolution(base_ring, None),
        ));
        let Z2e = GR.base_ring();
        let GalMQ = field_descriptor.GalMQ();

        // compute the period basis of F_(p^d)
        let d = field_descriptor.d();
        let rou_mʼ = compute_root_of_unity(&GF, &GR, field_descriptor.mʼ() as usize);
        let rou_mʼ_powers = span!(Level::INFO, "powers_of_rou_mʼ").in_scope(|| {
            (0..field_descriptor.mʼ())
                .scan(GR.one(), |current, _| {
                    let result = current.clone();
                    GR.mul_assign_ref(current, &rou_mʼ);
                    return Some(result);
                })
                .collect::<Vec<_>>()
        });
        let mut matrix = OwnedMatrix::zero(GR.rank(), d, Z2e);
        for (j, g) in field_descriptor.GalLʼQ_transversal().enumerate() {
            let period_basis_el = GR.sum(field_descriptor.GalMʼLʼ().enumerate_elements().map(|h| {
                let galois_el = GalMQ.op_ref(&g, &h);
                let power = GalMQ
                    .underlying_ring()
                    .smallest_positive_lift(*GalMQ.as_ring_el(&galois_el))
                    % field_descriptor.mʼ() as i64;
                rou_mʼ_powers[power as usize].clone()
            }));
            let period_basis_el = GR.inclusion().mul_map(
                period_basis_el,
                GR.base_ring()
                    .coerce(&ZZi64, field_descriptor.scale_Mʼ_to_M_period_basis()),
            );
            let current_wrt_basis = GR.wrt_power_basis(&period_basis_el);
            for i in 0..GR.rank() {
                *matrix.at_mut(i, j) = current_wrt_basis.at(i);
            }
        }

        // compute the crt kernel
        let rou_p = compute_root_of_unity(&GF, &GR, p as usize);
        let rou_p_powers = span!(Level::INFO, "powers_of_rou_p").in_scope(|| {
            (0..p)
                .scan(GR.one(), |current, _| {
                    let result = current.clone();
                    GR.mul_assign_ref(current, &rou_p);
                    return Some(result);
                })
                .collect::<Vec<_>>()
        });
        let [gen_Jp] = field_descriptor.I_p(p).generators().try_into().unwrap();
        let Jp_ord = ZZi64
            .checked_div(
                &(p - 1),
                &int_cast(field_descriptor.H_p(p).subgroup_order().clone(), ZZi64, ZZbig),
            )
            .unwrap() as usize;
        let u_p_coeffs = (0..Jp_ord).map(|i| {
            GR.sum(field_descriptor.H_p(p).enumerate_elements().map(|g| {
                let galois_el = GalMQ.op(g, GalMQ.pow(gen_Jp.clone(), -(i as i64)));
                let power = GalMQ
                    .underlying_ring()
                    .smallest_positive_lift(*GalMQ.as_ring_el(&galois_el))
                    % p;
                rou_p_powers[power as usize].clone()
            }))
        });
        let conjugation = GalMQ.from_ring_el(GalMQ.underlying_ring().neg_one()).unwrap();
        let scale = GR.base_ring().invert(&GR.base_ring().coerce(&ZZi64, p)).unwrap();
        let H_p_ord = int_cast(field_descriptor.H_p(p).subgroup_order().clone(), ZZi64, ZZbig);
        let summand = GR.inclusion().map(GR.base_ring().coerce(&ZZi64, -H_p_ord));
        let inv_u_p_coeffs = (0..Jp_ord).map(|i| {
            let mut result = GR.sum(field_descriptor.H_p(p).enumerate_elements().map(|g| {
                let galois_el = GalMQ.op(GalMQ.op(conjugation, g), GalMQ.pow(gen_Jp.clone(), i as i64));
                let power = GalMQ
                    .underlying_ring()
                    .smallest_positive_lift(*GalMQ.as_ring_el(&galois_el))
                    % p;
                GR.pow(rou_p.clone(), power as usize)
            }));
            GR.add_assign_ref(&mut result, &summand);
            GR.inclusion().mul_assign_ref_map(&mut result, &scale);
            result
        });

        // map coefficients to period basis
        let mut rhs = OwnedMatrix::zero(GR.rank(), Jp_ord * 2, Z2e);
        for (j, coeff) in u_p_coeffs.enumerate() {
            let coeff_wrt_basis = GR.wrt_power_basis(&coeff);
            for i in 0..GR.rank() {
                *rhs.at_mut(i, j) = coeff_wrt_basis.at(i);
            }
        }
        for (j, coeff) in inv_u_p_coeffs.enumerate() {
            let coeff_wrt_basis = GR.wrt_power_basis(&coeff);
            for i in 0..GR.rank() {
                *rhs.at_mut(i, j + Jp_ord) = coeff_wrt_basis.at(i);
            }
        }
        let mut sol = OwnedMatrix::zero(d, Jp_ord * 2, Z2e);
        Z2e.solve_right(matrix.data_mut(), rhs.data_mut(), sol.data_mut())
            .assert_solved();
        let u_p = (0..Jp_ord)
            .map(|j| sol.data().col_at(j).as_iter().cloned().collect::<Vec<_>>())
            .map(|coeff| {
                coeff
                    .into_iter()
                    .map(|c| int_cast(Z2e.smallest_lift(c), ZZi128, ZZbig))
                    .collect()
            })
            .collect();
        let inv_u_p = (Jp_ord..(Jp_ord * 2))
            .map(|j| sol.data().col_at(j).as_iter().cloned().collect::<Vec<_>>())
            .map(|coeff| {
                coeff
                    .into_iter()
                    .map(|c| int_cast(Z2e.smallest_lift(c), ZZi128, ZZbig))
                    .collect()
            })
            .collect();

        let gen_Jp_dlog = field_descriptor.Gal_to_shift(&gen_Jp);
        return Self {
            u_p,
            inv_u_p,
            p,
            gen_Jp_dlog,
            gen_Jp,
            t,
        };
    }

    pub fn t(&self) -> i128 { self.t }

    pub fn u_p(&self) -> &Vec<Vec<i128>> { &self.u_p }

    pub fn inv_u_p(&self) -> &Vec<Vec<i128>> { &self.inv_u_p }

    pub fn gen_Jp_dlog(&self) -> &[i64] { &self.gen_Jp_dlog }

    pub fn gen_Jp(&self) -> &GEl<MultGroup<Zn64B>> { &self.gen_Jp }

    pub fn reduce_t(&self, new_t: i128) -> Self {
        assert!(self.t % new_t == 0);
        Self {
            gen_Jp: *self.gen_Jp(),
            gen_Jp_dlog: self.gen_Jp_dlog.clone(),
            p: self.p,
            t: new_t,
            u_p: self
                .u_p
                .iter()
                .map(|coeff| coeff.iter().map(|c| cmod(*c, new_t)).collect())
                .collect(),
            inv_u_p: self
                .inv_u_p
                .iter()
                .map(|coeff| coeff.iter().map(|c| cmod(*c, new_t)).collect())
                .collect(),
        }
    }

    /// Applies `u_p` to the given element of `R_q`.
    #[instrument(skip_all)]
    pub fn apply_plain(&self, ring: &RingContext, mut data_crt: RingElementData) -> RingElementData<'static> {
        let mut result_crt = ring.init_zero();
        let mut data_crt = data_crt.borrow_mut();
        let mut tmp = ring.init_zero();
        let mut tmp = tmp.borrow_mut();
        for i in 0..self.u_p.len() {
            ring.fma_element_Lʼ(data_crt, &self.u_p[i], result_crt.borrow_mut());
            ring.apply_galois(tmp, data_crt, &self.gen_Jp_dlog);
            swap(&mut tmp, &mut data_crt);
        }
        return result_crt;
    }

    /// Applies `u_p^-1` to the given element of `R_q`.
    #[instrument(skip_all)]
    pub fn apply_inv_plain(&self, ring: &RingContext, mut data_crt: RingElementData) -> RingElementData<'static> {
        let mut result_crt = ring.init_zero();
        let mut data_crt = data_crt.borrow_mut();
        let mut tmp = ring.init_zero();
        let mut tmp = tmp.borrow_mut();
        for i in 0..self.inv_u_p.len() {
            ring.fma_element_Lʼ(data_crt, &self.inv_u_p[i], result_crt.borrow_mut());
            ring.apply_galois(tmp, data_crt, &self.gen_Jp_dlog);
            swap(&mut tmp, &mut data_crt);
        }
        return result_crt;
    }
}

#[instrument(skip_all)]
fn compute_root_of_unity(GF: &GaloisField, GR: &GaloisRing, p: usize) -> El<GaloisRing> {
    let e = ZZbig.abs_log2_ceil(GR.base_ring().modulus()).unwrap();
    let rou = get_prim_root_of_unity(&GF, &int_cast(p as i64, ZZbig, ZZi64)).unwrap();
    let mut rou = GR.from_power_basis(GF.wrt_power_basis(&rou).iter().map(|c| {
        GR.base_ring().get_ring().from_int_promise_reduced(int_cast(
            GF.base_ring().smallest_positive_lift(c),
            ZZbig,
            ZZi64,
        ))
    }));
    for _ in 0..ZZi64.abs_log2_ceil(&(e as i64)).unwrap() {
        let rou_power = GR.pow(rou.clone(), p - 1);
        let delta = GR
            .checked_div(
                &GR.sub(GR.mul_ref(&rou_power, &rou), GR.one()),
                &GR.int_hom().mul_map(rou_power, p as i32),
            )
            .unwrap();
        GR.sub_assign(&mut rou, delta);
    }
    return rou;
}

#[test]
fn test_crt_transform_factor_field_2() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_2(), 100..120, 1);
    let crt_transform = CRTTransformFactor::new(2, 7, ring.field_descriptor());

    // since (Z/7Z)*/D is cyclic, there is only one ordering of the period basis
    let mut data = ring.plaintext_into_ciphertext_ring_unscaled_pb(2, &[0, 1]);
    ring.pb_to_crt(data.borrow_mut());
    let mut result = crt_transform.apply_plain(&ring, data);
    ring.crt_to_pb(result.borrow_mut());
    let result = ring.plaintext_from_ciphertext_ring_unscaled_pb(2, result);
    let nonzero = result.iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
    assert_eq!(nonzero, [1]);
}

#[test]
fn test_crt_transform_factor_field_16() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_16(), 100..120, 1);
    let crt_transform = CRTTransformFactor::new(2, 257, ring.field_descriptor());

    let slot_unit_vector = vec![1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0];
    assert_eq!(
        slot_unit_vector,
        ring.mul_plain_pb(2, &slot_unit_vector, &slot_unit_vector)
    );
    let mut data = ring.plaintext_into_ciphertext_ring_unscaled_pb(2, &slot_unit_vector);
    ring.pb_to_crt(data.borrow_mut());
    let mut result = crt_transform.apply_plain(&ring, data);
    ring.crt_to_pb(result.borrow_mut());
    let result = ring.plaintext_from_ciphertext_ring_unscaled_pb(2, result);
    let nonzero = result.iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
    assert_eq!(nonzero, [1]);
}

#[test]
fn test_crt_transform_factor_field_18() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_18(), 100..120, 4);
    let crt_factor_7 = CRTTransformFactor::new(16, 7, ring.field_descriptor());
    let crt_factor_13 = CRTTransformFactor::new(16, 13, ring.field_descriptor());

    let slot_unit_vector_in_K = [6, -3, -2, -2, -7, 7];
    let slot_unit_vector = (0..3)
        .flat_map(|_| slot_unit_vector_in_K.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(
        slot_unit_vector,
        ring.mul_plain_pb(16, &slot_unit_vector, &slot_unit_vector)
    );
    let mut data = ring.plaintext_into_ciphertext_ring_unscaled_pb(16, &slot_unit_vector);
    ring.pb_to_crt(data.borrow_mut());
    let mut result = crt_factor_13.apply_plain(&ring, crt_factor_7.apply_plain(&ring, data));
    ring.crt_to_pb(result.borrow_mut());
    let result = ring.plaintext_from_ciphertext_ring_unscaled_pb(16, result);
    assert_eq!(&result[..6], &result[6..12]);
    assert_eq!(&result[..6], &result[12..]);
    let nonzero = result[..6].iter().copied().filter(|&x| x != 0).collect::<Vec<_>>();
    assert_eq!(vec![-1], nonzero);
}

#[test]
fn test_crt_transform_field_18() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_18(), 200..220, 4);
    let crt_transform = UnsplitCRTTransform::new(16, ring.field_descriptor());

    let ptxt_slots = (0..6).map(|i| cmod(i, 16)).collect::<Vec<_>>();
    let ptxt_pb = crt_transform.pack_slot_values(&ring, &ptxt_slots);
    let res_pb = ring.mul_plain_pb(16, &ptxt_pb, &ptxt_pb);
    let res_slots = crt_transform.extract_slot_values(&ring, &res_pb);
    let expected = (0..6).map(|i| cmod(i * i, 16)).collect::<Vec<_>>();
    assert_eq!(expected, res_slots);
}

#[test]
fn test_crt_transform_field_192() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_192(), 300..320, 4);
    let crt_transform = UnsplitCRTTransform::new(16, ring.field_descriptor());

    let ptxt1_slots = (0..96).map(|i| cmod(i + 5, 16)).collect::<Vec<_>>();
    let ptxt1_pb = crt_transform.pack_slot_values(&ring, &ptxt1_slots);
    let ptxt2_slots = (0..96).map(|i| cmod(i % 7 + 2, 16)).collect::<Vec<_>>();
    let ptxt2_pb = crt_transform.pack_slot_values(&ring, &ptxt2_slots);
    let res_pb = ring.mul_plain_pb(16, &ptxt1_pb, &ptxt2_pb);
    let res_slots = crt_transform.extract_slot_values(&ring, &res_pb);
    let expected = (0..96).map(|i| cmod((i + 5) * (i % 7 + 2), 16)).collect::<Vec<_>>();
    assert_eq!(expected, res_slots);
}

#[test]
#[ignore]
fn test_crt_transform_field_36864() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RingContext::new(field_36864(), 800..820, 80);
    let crt_transform = UnsplitCRTTransform::new(1 << 80, ring.field_descriptor());

    let ptxt1_slots = (0..ring.slot_count())
        .map(|i| cmod(i as i128 + 5, 1 << 80))
        .collect::<Vec<_>>();
    let ptxt1_pb = crt_transform.pack_slot_values(&ring, &ptxt1_slots);
    let ptxt2_slots = (0..ring.slot_count())
        .map(|i| cmod(i as i128 % 7 + 2, 1 << 80))
        .collect::<Vec<_>>();
    let ptxt2_pb = crt_transform.pack_slot_values(&ring, &ptxt2_slots);
    let res_pb = ring.mul_plain_pb(1 << 80, &ptxt1_pb, &ptxt2_pb);
    let res_slots = crt_transform.extract_slot_values(&ring, &res_pb);
    let expected = (0..ring.slot_count())
        .map(|i| cmod((i as i128 + 5) * (i as i128 % 7 + 2), 1 << 80))
        .collect::<Vec<_>>();
    assert_eq!(expected, res_slots);
}
