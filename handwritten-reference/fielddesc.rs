use feanor_math::algorithms::cyclotomic::get_prim_root_of_unity;
use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::subgroup::Subgroup;
use feanor_math::group::*;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Describes a number field L that can be used to instantiate (d)BFV.
///
/// L is the compositum of two disjoint fields K and Lʼ.
///  - K is a subfield of the splitting field of (2) in a cyclotomic field M_K.
///  - Lʼ is a subfield of some prime cyclotomic field Mʼ, with the restriction that (2) is totally
///    inert in Lʼ.
///  - Lʼ must have degree at least d (see below).
///  - Thus L = LʼK is a subfield of M := MʼM_K.
///
/// We also define the additional the notion:
///  - I_p is the inertia subgroup of p|m, in other words the (Z/pZ)* factor in Gal(M|Q).
///  - f_p is the order of 2 in (Z/pZ)*, or equivalently the order of the projection of 2 into I_p
///    (identifying 2 with the Frobenius of (2) in Gal(M|Q)).
///  - H_p = I_p \cap Gal(M|L).
///  - d_p = f_p / gcd(f_p, #H_p), or equivalently the order of the projection of 2 into I_p / H_p.
///  - d is the least common multiple of all d_p, p|m_K.
pub struct FieldDescriptor {
    GalMQ: MultGroup<Zn64B>,
    /// as a subgroup of [`FieldDescriptor::GalMQ`]; the first
    /// [`FieldDescriptor::GalLQ_cyclic_decomposition`] generators exactly match
    /// [`FieldDescriptor::GalLQ_cyclic_decomposition`].
    GalMQ_with_gens: Subgroup<MultGroup<Zn64B>>,
    m_factors: Vec<i64>,
    /// as a subgroup of [`FieldDescriptor::GalMQ`]
    GalML: Subgroup<MultGroup<Zn64B>>,
    GalKQ_cyclic_decomposition: Vec<(GEl<MultGroup<Zn64B>>, usize)>,
    deg_L: usize,
    mʼ: i64,
    /// as a subgroup of [`FieldDescriptor::GalMQ`]
    GalMʼLʼ: Subgroup<MultGroup<Zn64B>>,
    /// as a subgroup of [`FieldDescriptor::GalMQ`]
    GalM_KK: Subgroup<MultGroup<Zn64B>>,
    GalLʼQ_cyclic_decomposition: Vec<(GEl<MultGroup<Zn64B>>, usize)>,
    deg_Lʼ: usize,
    deg_Mʼ: usize,
    I_p_s: Vec<Subgroup<MultGroup<Zn64B>>>,
    H_p_s: Vec<Subgroup<MultGroup<Zn64B>>>,
    f_p_s: Vec<usize>,
    d_p_s: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FieldDescriptorData {
    pub m: i64,
    pub mʼ: i64,
    pub GalMʼLʼ_generators: Vec<i64>,
    pub GalM_KK_generators: Vec<i64>,
    pub GalLʼQ_cyclic_decomposition: Vec<(i64, usize)>,
    pub GalKQ_cyclic_decomposition: Vec<(i64, usize)>,
}

impl FieldDescriptor {
    #[instrument(skip_all)]
    pub fn new(data: FieldDescriptorData) -> Self {
        assert!(data.m % 2 == 1);
        assert!(data.m % data.mʼ == 0);
        assert!(data.mʼ == 1 || is_prime(ZZi64, &data.mʼ, 10), "mʼ must be a prime");

        let m = data.m;
        let mʼ = data.mʼ;
        let m_factors = if mʼ == 1 { None } else { Some(mʼ) }
            .into_iter()
            .chain(factor(ZZi64, m / mʼ).into_iter().map(|(p, e)| {
                assert!(e == 1);
                p
            }))
            .collect::<Vec<_>>();
        let deg_M = ZZi64.prod(m_factors.iter().map(|p| p - 1));
        let deg_Mʼ = if mʼ == 1 { 1 } else { mʼ as usize - 1 };

        let GalMQ = MultGroup::new(Zn64B::new(m as u64));
        let to_GalMQ = |x: &i64| GalMQ.from_ring_el(GalMQ.underlying_ring().coerce(&ZZi64, *x)).unwrap();
        let GalMʼLʼ = Subgroup::for_zn(GalMQ, data.GalMʼLʼ_generators.iter().map(to_GalMQ).collect());
        let GalM_KK = Subgroup::for_zn(GalMQ, data.GalM_KK_generators.iter().map(to_GalMQ).collect());
        let GalML = GalMʼLʼ.sum(&GalM_KK);
        let deg_L = (deg_M / int_cast(GalML.subgroup_order().clone(), ZZi64, ZZbig)) as usize;
        let deg_Lʼ = (deg_Mʼ as i64 / int_cast(GalMʼLʼ.subgroup_order().clone(), ZZi64, ZZbig)) as usize;
        let GalLʼQ_cyclic_decomposition = data
            .GalLʼQ_cyclic_decomposition
            .iter()
            .map(|(g, n)| (to_GalMQ(g), *n))
            .collect::<Vec<_>>();
        let GalKQ_cyclic_decomposition = data
            .GalKQ_cyclic_decomposition
            .iter()
            .map(|(g, n)| (to_GalMQ(g), *n))
            .collect::<Vec<_>>();
        let GalMQ_with_gens = Subgroup::for_zn(
            GalMQ,
            GalLʼQ_cyclic_decomposition
                .iter()
                .chain(&GalKQ_cyclic_decomposition)
                .map(|(g, _)| g)
                .chain(GalM_KK.generators())
                .chain(GalMʼLʼ.generators())
                .cloned()
                .collect(),
        );

        let mut result = Self {
            GalKQ_cyclic_decomposition,
            GalLʼQ_cyclic_decomposition,
            GalML,
            GalMQ,
            GalMQ_with_gens,
            GalMʼLʼ,
            GalM_KK,
            H_p_s: Vec::new(),
            I_p_s: Vec::new(),
            d_p_s: Vec::new(),
            f_p_s: Vec::new(),
            deg_L,
            deg_Lʼ,
            deg_Mʼ,
            m_factors,
            mʼ,
        };
        result.init_per_prime_info();
        return result;
    }

    pub fn data(&self) -> FieldDescriptorData {
        let GalMQ_to_int = |g| {
            self.GalMQ()
                .underlying_ring()
                .smallest_positive_lift(*self.GalMQ().as_ring_el(g))
        };
        FieldDescriptorData {
            m: self.m() as i64,
            mʼ: self.mʼ() as i64,
            GalMʼLʼ_generators: self.GalMʼLʼ().generators().iter().map(GalMQ_to_int).collect(),
            GalM_KK_generators: self.GalM_KK.generators().iter().map(GalMQ_to_int).collect(),
            GalLʼQ_cyclic_decomposition: self
                .GalLʼQ_cyclic_decomposition
                .iter()
                .map(|(g, l)| (GalMQ_to_int(g), *l))
                .collect(),
            GalKQ_cyclic_decomposition: self
                .GalKQ_cyclic_decomposition
                .iter()
                .map(|(g, l)| (GalMQ_to_int(g), *l))
                .collect(),
        }
    }

    /// Creates a new [`FieldDescriptor`] describing the given K and using Lʼ = Q.
    ///
    /// Note that the ordering of [`FieldDescriptor::GalLQ_transversal()`] is not deterministic -
    /// different calls with the same Galois groups will describe the same field, but potentially
    /// with permuted period bases. Therefore, if you want lists of coefficients to represent
    /// the same element w.r.t. multiple [`FieldDescriptors`], the [`FieldDescriptor`] has to be
    /// created from a [`FieldDescriptorData`], which fixes the concrete isomorphism `Gal(L|Q) ->
    /// C_n1 x ... x C_nr`. You can first create a [`FieldDescriptor`] using this function, and
    /// then extract the data via [`FieldDescriptor::data()`].
    #[instrument(skip_all)]
    pub fn create_with_trivial_Lʼ(GalM_KQ: MultGroup<Zn64B>, GalM_KK: Subgroup<MultGroup<Zn64B>>) -> Self {
        assert!(GalM_KK.parent().get_group() == GalM_KQ.get_group());
        let m = *GalM_KQ.underlying_ring().modulus();
        assert!(m % 2 == 1);

        // check that (2) splits in K
        assert!(
            GalM_KK.contains(
                &GalM_KQ
                    .from_ring_el(GalM_KQ.underlying_ring().int_hom().map(2))
                    .unwrap()
            ),
            "(2) must split completely in K"
        );

        let m_factors = factor(ZZi64, m)
            .into_iter()
            .map(|(p, e)| {
                assert!(e == 1);
                p
            })
            .collect::<Vec<_>>();
        let deg_M = ZZi64.prod(m_factors.iter().map(|p| p - 1));

        let mut GalKQ_cyclic_decomposition = Subgroup::zn_unit_group(GalM_KQ)
            .quotient_cyclic_decomposition(&GalM_KK)
            .into_iter()
            .map(|(g, n)| (g, int_cast(n, ZZi64, ZZbig) as usize))
            .collect::<Vec<_>>();
        GalKQ_cyclic_decomposition.sort_unstable_by_key(|(_, l)| *l);
        let GalMQ_with_gens = Subgroup::for_zn(
            GalM_KQ,
            GalKQ_cyclic_decomposition
                .iter()
                .map(|(g, _)| g)
                .chain(GalM_KK.generators())
                .cloned()
                .collect(),
        );
        let deg_K = deg_M / int_cast(GalM_KK.subgroup_order().clone(), ZZi64, ZZbig);

        let mut result = Self {
            GalKQ_cyclic_decomposition,
            GalLʼQ_cyclic_decomposition: Vec::new(),
            GalML: GalM_KK.clone(),
            GalMQ: GalM_KQ,
            GalMQ_with_gens,
            GalMʼLʼ: Subgroup::for_zn(GalM_KQ, vec![]),
            GalM_KK,
            H_p_s: Vec::new(),
            I_p_s: Vec::new(),
            d_p_s: Vec::new(),
            f_p_s: Vec::new(),
            deg_L: deg_K as usize,
            deg_Lʼ: 1,
            deg_Mʼ: 1,
            m_factors,
            mʼ: 1,
        };
        result.init_per_prime_info();
        return result;
    }

    /// Creates a new [`FieldDescriptor`] describing the given K and Lʼ.
    ///
    /// Note that the ordering of [`FieldDescriptor::GalLQ_transversal()`] is not deterministic -
    /// different calls with the same Galois groups will describe the same field, but potentially
    /// with permuted period bases. Therefore, if you want lists of coefficients to represent
    /// the same element w.r.t. multiple [`FieldDescriptors`], the [`FieldDescriptor`] has to be
    /// created from a [`FieldDescriptorData`], which fixes the concrete isomorphism `Gal(L|Q) ->
    /// C_n1 x ... x C_nr`. You can first create a [`FieldDescriptor`] using this function, and
    /// then extract the data via [`FieldDescriptor::data()`].
    #[instrument(skip_all)]
    pub fn create_with_Lʼ(
        GalM_KQ: MultGroup<Zn64B>,
        GalM_KK: Subgroup<MultGroup<Zn64B>>,
        GalMʼQ: MultGroup<Zn64B>,
        GalMʼLʼ: Subgroup<MultGroup<Zn64B>>,
    ) -> Self {
        assert!(GalM_KK.parent().get_group() == GalM_KQ.get_group());
        assert!(GalMʼLʼ.parent().get_group() == GalMʼQ.get_group());

        let m_K = *GalM_KQ.underlying_ring().modulus();
        let mʼ = *GalMʼQ.underlying_ring().modulus();
        assert!(m_K % 2 == 1);
        assert!(mʼ % 2 == 1);
        assert!(ZZi64.is_unit(&ZZi64.gcd(&m_K, &mʼ)), "m_K and must be coprime");
        assert!(is_prime(ZZi64, &mʼ, 10), "mʼ must be a prime");

        // check that (2) splits in K
        assert!(
            GalM_KK.contains(
                &GalM_KQ
                    .from_ring_el(GalM_KQ.underlying_ring().int_hom().map(2))
                    .unwrap()
            ),
            "(2) must split completely in K"
        );

        // check that (2) is totally inert in Lʼ
        assert!(
            GalMʼLʼ.generators().len() <= 1,
            "GalMʼLʼ must be cyclic, since mʼ is a prime"
        );
        let Dʼ = Subgroup::for_zn(
            GalMʼQ,
            vec![GalMʼQ.from_ring_el(GalMʼQ.underlying_ring().int_hom().map(2)).unwrap()],
        );
        assert!(
            int_cast(Dʼ.sum(&GalMʼLʼ).subgroup_order().clone(), ZZi64, ZZbig) == mʼ - 1,
            "(2) must be totally inert in Lʼ"
        );

        let m = m_K * mʼ;
        let m_factors = [mʼ]
            .into_iter()
            .chain(factor(ZZi64, m_K).into_iter().map(|(p, e)| {
                assert!(e == 1);
                p
            }))
            .collect::<Vec<_>>();
        let deg_M = ZZi64.prod(m_factors.iter().map(|p| p - 1));
        let deg_Mʼ = mʼ as usize - 1;

        // now embed everything into GalMQ
        let GalMQ = MultGroup::new(Zn64B::new(m as u64));
        let to_GalMQ = |x: Option<&MultGroupEl<_>>, y: Option<&MultGroupEl<_>>| {
            let res = ZZi64.inv_crt(
                [
                    &x.map(|x| GalMʼQ.underlying_ring().smallest_positive_lift(*GalMʼQ.as_ring_el(x)))
                        .unwrap_or(1),
                    &y.map(|y| GalM_KQ.underlying_ring().smallest_positive_lift(*GalM_KQ.as_ring_el(y)))
                        .unwrap_or(1),
                ],
                [&mʼ, &m_K],
            );
            GalMQ.from_ring_el(GalMQ.underlying_ring().coerce(&ZZi64, res)).unwrap()
        };
        let GalMʼQ = Subgroup::for_zn(
            GalMQ,
            Subgroup::zn_unit_group(GalMʼQ)
                .generators()
                .iter()
                .map(|g| to_GalMQ(Some(g), None))
                .collect(),
        );
        let GalMʼLʼ = Subgroup::for_zn(
            GalMQ,
            GalMʼLʼ.generators().iter().map(|g| to_GalMQ(Some(g), None)).collect(),
        );
        let GalM_KQ = Subgroup::for_zn(
            GalMQ,
            Subgroup::zn_unit_group(GalM_KQ)
                .generators()
                .iter()
                .map(|g| to_GalMQ(None, Some(g)))
                .collect(),
        );
        let GalM_KK = Subgroup::for_zn(
            GalMQ,
            GalM_KK.generators().iter().map(|g| to_GalMQ(None, Some(g))).collect(),
        );
        let GalML = GalMʼLʼ.sum(&GalM_KK);

        // now compute cyclic decompositions
        let deg_L = (deg_M / int_cast(GalML.subgroup_order().clone(), ZZi64, ZZbig)) as usize;
        let deg_Lʼ = (deg_Mʼ as i64 / int_cast(GalMʼLʼ.subgroup_order().clone(), ZZi64, ZZbig)) as usize;
        let GalLʼQ_cyclic_decomposition = GalMʼQ
            .quotient_cyclic_decomposition(&GalMʼLʼ)
            .into_iter()
            .map(|(g, n)| (g, int_cast(n, ZZi64, ZZbig) as usize))
            .collect::<Vec<_>>();

        let mut GalKQ_cyclic_decomposition = GalM_KQ
            .quotient_cyclic_decomposition(&GalM_KK)
            .into_iter()
            .map(|(g, n)| (g, int_cast(n, ZZi64, ZZbig) as usize))
            .collect::<Vec<_>>();
        GalKQ_cyclic_decomposition.sort_unstable_by_key(|(_, l)| *l);
        let GalMQ_with_gens = Subgroup::for_zn(
            GalMQ,
            GalLʼQ_cyclic_decomposition
                .iter()
                .chain(&GalKQ_cyclic_decomposition)
                .map(|(g, _)| g)
                .chain(GalM_KK.generators())
                .chain(GalMʼLʼ.generators())
                .cloned()
                .collect(),
        );

        let mut result = Self {
            GalKQ_cyclic_decomposition,
            GalLʼQ_cyclic_decomposition,
            GalML,
            GalMQ,
            GalMQ_with_gens,
            GalMʼLʼ,
            GalM_KK,
            H_p_s: Vec::new(),
            I_p_s: Vec::new(),
            d_p_s: Vec::new(),
            f_p_s: Vec::new(),
            deg_L,
            deg_Lʼ,
            deg_Mʼ,
            m_factors,
            mʼ,
        };
        result.init_per_prime_info();
        return result;
    }

    fn init_per_prime_info(&mut self) {
        for p in &self.m_factors {
            let Zp = Zn64B::new(*p as u64).as_field().unwrap();
            let g = get_prim_root_of_unity(Zp, &int_cast(*p - 1, ZZbig, ZZi64)).unwrap();
            let g = self
                .GalMQ
                .from_ring_el(self.GalMQ.underlying_ring().coerce(
                    &ZZi64,
                    ZZi64.inv_crt([&Zp.smallest_positive_lift(g), &1], [p, &(self.m() as i64 / p)]),
                ))
                .unwrap();
            let I_p = Subgroup::for_zn(self.GalMQ, vec![g]);
            let GalK_pQ = MultGroup::new(Zp);
            let f_p = int_cast(
                Subgroup::for_zn(
                    GalK_pQ,
                    vec![
                        GalK_pQ
                            .from_ring_el(GalK_pQ.underlying_ring().int_hom().map(2))
                            .unwrap(),
                    ],
                )
                .subgroup_order()
                .clone(),
                ZZi64,
                ZZbig,
            ) as usize;
            let H_p = I_p.intersection(&self.GalML);
            let d_p = ZZi64
                .checked_div(
                    &(f_p as i64),
                    &ZZi64.gcd(&(f_p as i64), &int_cast(H_p.subgroup_order().clone(), ZZi64, ZZbig)),
                )
                .unwrap() as usize;
            self.I_p_s.push(I_p);
            self.f_p_s.push(f_p);
            self.H_p_s.push(H_p);
            self.d_p_s.push(d_p);
        }
        if self.mʼ > 1 {
            assert_eq!(
                self.d_p_s.iter().skip(1).fold(1, |x, y| ZZi64.lcm(&x, &(*y as i64))) as usize,
                self.deg_Lʼ
            );
        } else {
            assert!(self.d_p_s.iter().all(|d_p| *d_p == 1));
        }
    }

    fn get_p_idx(&self, p: i64) -> usize {
        self.m_factors
            .iter()
            .enumerate()
            .filter(|(_, p_)| p == **p_)
            .next()
            .expect("prime doesn't divide m")
            .0
    }

    pub fn m(&self) -> usize { *self.GalMQ.underlying_ring().modulus() as usize }

    pub fn mʼ(&self) -> usize { self.mʼ as usize }

    pub fn m_K(&self) -> usize { self.m() / self.mʼ() }

    pub fn deg_L(&self) -> usize { self.deg_L }

    pub fn deg_K(&self) -> usize { self.deg_L() / self.deg_Lʼ() }

    pub fn deg_Lʼ(&self) -> usize { self.deg_Lʼ }

    pub fn deg_Mʼ(&self) -> usize { self.deg_Mʼ }

    pub fn GalMQ(&self) -> &MultGroup<Zn64B> { &self.GalMQ }

    pub fn GalML(&self) -> &Subgroup<MultGroup<Zn64B>> { &self.GalML }

    pub fn GalMʼLʼ(&self) -> &Subgroup<MultGroup<Zn64B>> { &self.GalMʼLʼ }

    pub fn GalM_KK(&self) -> &Subgroup<MultGroup<Zn64B>> { &self.GalM_KK }

    pub fn m_factors(&self) -> &[i64] { &self.m_factors }

    pub fn m_K_factors(&self) -> &[i64] {
        debug_assert!(self.mʼ == 1 || self.m_factors[0] == self.mʼ);
        if self.mʼ == 1 {
            &self.m_factors
        } else {
            &self.m_factors[1..]
        }
    }

    pub fn Gal_to_shift(&self, g: &GEl<MultGroup<Zn64B>>) -> Vec<i64> {
        self.GalMQ_with_gens
            .dlog(g)
            .unwrap()
            .into_iter()
            .take(self.GalLQ_cyclic_decomposition().len())
            .map(|x| int_cast(x, ZZi64, ZZbig))
            .collect::<Vec<_>>()
    }

    /// Returns `(g1, n1), ..., (gr, nr)` such that we have an isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr  ->  Gal(L|Q)
    ///                 e_i  ->  g_i mod Gal(M|L)
    /// ```
    /// The ordering is also compatible with [`FieldDescriptor::GalLQ_transversal()`], which
    /// directly induces the ordering of the period basis.
    pub fn GalLQ_cyclic_decomposition<'a>(&'a self) -> Vec<(GEl<MultGroup<Zn64B>>, usize)> {
        self.GalLʼQ_cyclic_decomposition
            .iter()
            .chain(&self.GalKQ_cyclic_decomposition)
            .cloned()
            .collect()
    }

    /// Returns `(g1, n1), ..., (gr, nr)` such that we have an isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr  ->  Gal(K|Q)
    ///                 e_i  ->  g_i mod Gal(M|K)
    /// ```
    /// The ordering is also compatible with [`FieldDescriptor::GalLQ_transversal()`], which
    /// directly induces the ordering of the period basis.
    pub fn GalKQ_cyclic_decomposition<'a>(&'a self) -> &'a [(GEl<MultGroup<Zn64B>>, usize)] {
        &self.GalKQ_cyclic_decomposition
    }

    /// Returns `(g1, n1), ..., (gr, nr)` such that we have an isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr  ->  Gal(Lʼ|Q)
    ///                 e_i  ->  g_i mod Gal(M|Lʼ)
    /// ```
    /// The ordering is also compatible with [`FieldDescriptor::GalLQ_transversal()`], which
    /// directly induces the ordering of the period basis.
    pub fn GalLʼQ_cyclic_decomposition<'a>(&'a self) -> &'a [(GEl<MultGroup<Zn64B>>, usize)] {
        &self.GalLʼQ_cyclic_decomposition
    }

    /// Returns an iterator that yields a representative in `Gal(M|Q)` for every element in
    /// `Gal(L|Q) = Gal(M|Q)/Gal(M|L)`. The order is compatible with
    /// [`FieldDescriptor::GalLQ_cyclic_decomposition()`].
    ///
    /// Note that the order of the returned elements directly corresponds to the ordering of the
    /// period basis of L. Indeed, the period basis is exactly `Tr_(M|L)(σ(ζ))` where `σ` runs
    /// through the elements returned by this function. Note that the concrete choice of
    /// representative `σ` of an element in `Gal(M|Q)/Gal(M|L)` doesn't matter (it's vanishes via
    /// the trace), but a different ordering can result in a different ordering of the period basis
    /// coefficients.
    pub fn GalLQ_transversal<'a>(&'a self) -> impl use<'a> + Iterator<Item = GEl<MultGroup<Zn64B>>> {
        multi_cartesian_product(
            self.GalLQ_cyclic_decomposition().into_iter().map(|(g, order)| {
                (0..order).scan(self.GalMQ.identity(), move |current, _| {
                    let result = current.clone();
                    *current = self.GalMQ.op_ref(current, &g);
                    Some(result)
                })
            }),
            |els| {
                els.iter()
                    .fold(self.GalMQ.identity(), |x, y| self.GalMQ.op_ref_snd(x, y))
            },
            |_, x| x.clone(),
        )
    }

    /// Returns an iterator that yields a representative in `Gal(M|Q)` for every element in
    /// `Gal(K|Q) = Gal(M|Q)/Gal(M|K)`. The order is compatible with
    /// [`FieldDescriptor::GalKQ_cyclic_decomposition()`].
    ///
    /// Note that the order of the returned elements directly corresponds to the ordering of the
    /// period basis of K. Indeed, the period basis is exactly `Tr_(M|K)(σ(ζ))` where `σ` runs
    /// through the elements returned by this function. Note that the concrete choice of
    /// representative `σ` of an element in `Gal(M|Q)/Gal(M|K)` doesn't matter (it's vanishes via
    /// the trace), but a different ordering can result in a different ordering of the period basis
    /// coefficients.
    pub fn GalKQ_transversal<'a>(&'a self) -> impl use<'a> + Iterator<Item = GEl<MultGroup<Zn64B>>> {
        multi_cartesian_product(
            self.GalKQ_cyclic_decomposition().iter().map(|(g, order)| {
                (0..*order).scan(self.GalMQ.identity(), move |current, _| {
                    let result = current.clone();
                    *current = self.GalMQ.op_ref(current, g);
                    Some(result)
                })
            }),
            |els| {
                els.iter()
                    .fold(self.GalMQ.identity(), |x, y| self.GalMQ.op_ref_snd(x, y))
            },
            |_, x| x.clone(),
        )
    }

    /// Returns an iterator that yields a representative in `Gal(M|Q)` for every element in
    /// `Gal(Lʼ|Q) = Gal(M|Q)/Gal(M|Lʼ)`. The order is compatible with
    /// [`FieldDescriptor::GalLʼQ_cyclic_decomposition()`].
    ///
    /// Note that the order of the returned elements directly corresponds to the ordering of the
    /// period basis of Lʼ. Indeed, the period basis is exactly `Tr_(M|Lʼ)(σ(ζ))` where `σ` runs
    /// through the elements returned by this function. Note that the concrete choice of
    /// representative `σ` of an element in `Gal(M|Q)/Gal(M|Lʼ)` doesn't matter (it's vanishes via
    /// the trace), but a different ordering can result in a different ordering of the period basis
    /// coefficients.
    pub fn GalLʼQ_transversal(&self) -> impl Iterator<Item = GEl<MultGroup<Zn64B>>> {
        multi_cartesian_product(
            self.GalLʼQ_cyclic_decomposition.iter().map(|(g, order)| {
                (0..*order).scan(self.GalMQ.identity(), move |current, _| {
                    let result = current.clone();
                    *current = self.GalMQ.op_ref(current, g);
                    Some(result)
                })
            }),
            |els| {
                els.iter()
                    .fold(self.GalMQ.identity(), |x, y| self.GalMQ.op_ref_snd(x, y))
            },
            |_, x| x.clone(),
        )
    }

    pub fn d(&self) -> usize {
        self.m_factors
            .iter()
            .filter(|p| **p != self.mʼ)
            .map(|p| self.d_p(*p) as i64)
            .fold(1, |x, y| ZZi64.lcm(&x, &y)) as usize
    }

    /// Returns the group `H_p = I_p \cap Gal(M|L)`
    pub fn H_p(&self, p: i64) -> &Subgroup<MultGroup<Zn64B>> { &self.H_p_s[self.get_p_idx(p)] }

    pub fn I_p(&self, p: i64) -> &Subgroup<MultGroup<Zn64B>> { &self.I_p_s[self.get_p_idx(p)] }

    /// Returns the order f_p of the frobenius in the group `I_p`
    pub fn f_p(&self, p: i64) -> usize { self.f_p_s[self.get_p_idx(p)] }

    /// Returns the order d_p of the frobenius in the group `I_p / H_p`
    pub fn d_p(&self, p: i64) -> usize {
        let p_idx = self.get_p_idx(p);
        return ZZi64
            .checked_div(
                &(self.f_p_s[p_idx] as i64),
                &int_cast(self.H_p_s[p_idx].subgroup_order().clone(), ZZi64, ZZbig),
            )
            .unwrap() as usize;
    }

    /// Returns the value `t in {-1, 1}` such that the period basis of Q induced by Q (i.e. the
    /// basis { 1 }) is the same as the period basis of Q induced by M.
    pub fn scale_Q_to_M_period_basis(&self) -> i64 { if self.m_factors.len() % 2 == 1 { -1 } else { 1 } }

    /// Returns the value `t in {-1, 1}` such that the period basis of Lʼ induced by Mʼ is the same
    /// as the period basis of Lʼ induced by M.
    pub fn scale_Mʼ_to_M_period_basis(&self) -> i64 {
        self.scale_Q_to_M_period_basis() * self.scale_M_K_to_M_period_basis()
    }

    /// Returns the value `t in {-1, 1}` such that the period basis of K induced by M_K is the same
    /// as the period basis of Lʼ induced by M.
    pub fn scale_M_K_to_M_period_basis(&self) -> i64 { if self.mʼ == 1 { 1 } else { -1 } }
}

/// ```text
/// m = 8536255 = 5 * 7 * 13 * 73 * 257
/// [L : Q] = 36864 = 2^12 * 3^2 ~ 2^15.169925001442312
/// [M : L] = 144
/// K = Q(ζ_656635)^D
/// slot count = [K : Q] = 3072
/// D = <[3, 4, 32, 80]> <= C4 x C6 x C72 x C256
/// d = 12, d_5 = 4, d_7 = 3, d_73 = 3, d_257 = 4
/// #H_5 = 1, #H_7 = 1, #H_73 = 3, #H_257 = 4
/// #J_5 = 4, #J_7 = 6, #J_73 = 24, #J_257 = 64
/// L' = Q(ζ_13)
/// ```
pub fn field_36864() -> FieldDescriptor {
    FieldDescriptor::new(FieldDescriptorData {
        m: 8536255,
        mʼ: 13,
        GalMʼLʼ_generators: vec![],
        GalM_KK_generators: vec![3283177],
        GalLʼQ_cyclic_decomposition: vec![(3939811, 4), (7879621, 3)],
        GalKQ_cyclic_decomposition: vec![(7316791, 2), (1169351, 3), (4560466, 8), (8071246, 64)],
    })
}

/// ```text
/// m = 90627 = 3 * 17 * 1777
/// [L : Q] = 192 = 2^6 * 3 ~ 2^7.584962500721156
/// [M : L] = 296
/// K = Q(ζ_30209)^D
/// slot count = [K : Q] = 96
/// D = <[2, 168]> <= C16 x C1776
/// d = 2, d_17 = 2, d_1777 = 2
/// #H_17 = 4, #H_1777 = 37
/// #J_17 = 4, #J_1777 = 48
/// L' = Q(ζ_3)
/// ```
pub fn field_192() -> FieldDescriptor {
    FieldDescriptor::new(FieldDescriptorData {
        m: 90627,
        mʼ: 3,
        GalMʼLʼ_generators: vec![],
        GalM_KK_generators: vec![30211],
        GalLʼQ_cyclic_decomposition: vec![(60419, 2)],
        GalKQ_cyclic_decomposition: vec![(45427, 2), (4183, 3), (47533, 16)],
    })
}

/// ```text
/// m = 1729 = 7 * 13 * 19
/// [L : Q] = 18 = 2 * 3^2 ~ 2^4.169925001442312
/// [M : L] = 72
/// K = Q(ζ_91)^D
/// slot count = [K : Q] = 6
/// D = <[4, 5]> <= C6 x C12
/// d = 3, d_7 = 3, d_13 = 3
/// #H_7 = 1, #H_13 = 4
/// #J_7 = 6, #J_13 = 3
/// L' = degree-3 subfield of Q(ζ_19)
/// ```
pub fn field_18() -> FieldDescriptor {
    FieldDescriptor::new(FieldDescriptorData {
        m: 1729,
        mʼ: 19,
        GalMʼLʼ_generators: vec![274],
        GalM_KK_generators: vec![457],
        GalLʼQ_cyclic_decomposition: vec![(1548, 3)],
        GalKQ_cyclic_decomposition: vec![(1483, 2), (932, 3)],
    })
}

/// ```text
/// m = 257
/// [L : Q] = 16 = 2^4
/// [M : L] = 16
/// K = Q(ζ_257)^D
/// slot count = [K : Q] = 16
/// D = <[16]> <= C256
/// d = d_257 = 1
/// #H_257 = 16
/// #J_257 = 16
/// L' = Q
/// ```
pub fn field_16() -> FieldDescriptor {
    FieldDescriptor::new(FieldDescriptorData {
        m: 257,
        mʼ: 1,
        GalMʼLʼ_generators: vec![],
        GalM_KK_generators: vec![2],
        GalLʼQ_cyclic_decomposition: vec![],
        GalKQ_cyclic_decomposition: vec![(155, 16)],
    })
}

/// ```text
/// m = 7
/// [L : Q] = 2
/// [M : L] = 3
/// K = Q(ζ_7)^D
/// slot count = [K : Q] = 2
/// D = <[2]> <= C6
/// d = d_7 = 1
/// #H_7 = 3
/// #J_7 = 2
/// L' = Q
/// ```
pub fn field_2() -> FieldDescriptor {
    FieldDescriptor::new(FieldDescriptorData {
        m: 7,
        mʼ: 1,
        GalMʼLʼ_generators: vec![],
        GalM_KK_generators: vec![2],
        GalLʼQ_cyclic_decomposition: vec![],
        GalKQ_cyclic_decomposition: vec![(6, 2)],
    })
}

#[test]
fn test_field_136864() {
    feanor_tracing::DelayedLogger::init_test();
    let field = field_36864();
    assert_eq!(8536255, field.m());
    assert_eq!(36864, field.deg_L());
    assert_eq!(3072, field.deg_K());
    assert_eq!(12, field.deg_Lʼ());
}

#[test]
fn test_field_192() {
    feanor_tracing::DelayedLogger::init_test();
    let field = field_192();
    assert_eq!(90627, field.m());
    assert_eq!(192, field.deg_L());
    assert_eq!(96, field.deg_K());
    assert_eq!(2, field.deg_Lʼ());
}

#[test]
fn test_field_18() {
    feanor_tracing::DelayedLogger::init_test();
    let field = field_18();
    assert_eq!(1729, field.m());
    assert_eq!(18, field.deg_L());
    assert_eq!(6, field.deg_K());
    assert_eq!(3, field.deg_Lʼ());
}

#[test]
fn test_field_16() {
    feanor_tracing::DelayedLogger::init_test();
    let field = field_16();
    assert_eq!(257, field.m());
    assert_eq!(16, field.deg_L());
    assert_eq!(16, field.deg_K());
    assert_eq!(1, field.deg_Lʼ());
}

#[test]
fn test_field_2() {
    feanor_tracing::DelayedLogger::init_test();
    let field = field_2();
    assert_eq!(7, field.m());
    assert_eq!(2, field.deg_L());
    assert_eq!(2, field.deg_K());
    assert_eq!(1, field.deg_Lʼ());
}
