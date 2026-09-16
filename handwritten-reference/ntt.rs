use feanor_math::algorithms::cyclotomic::{get_prim_root_of_unity, get_prim_root_of_unity_pow2};
use feanor_math::algorithms::fft::cooley_tuckey::{CooleyTuckeyButterfly, bitreverse};
use feanor_math::algorithms::fft::radix3::CooleyTukeyRadix3Butterfly;
use feanor_math::group::AbelianGroupStore;
use feanor_math::homomorphism::CanHom;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::{Zn64B, Zn64BBase, Zn64BFastmul, Zn64BFastmulBase};
use tracing::instrument;

use crate::fielddesc::*;
use crate::*;

/// For an abelian number field `L` described by a [`FieldDescriptor`] and a prime
/// `p` that splits completely in `O_L` and additionally has suitable power-of-two
/// and power-of-three roots of unity, a [`PlainCRTTransform`] models the CRT transform,
/// which is the change-of-basis transform between the Gaussian period basis and
/// the CRT basis, consisting of elements of `O_L/(p)` that are `= 1` modulo one
/// prime ideal of `O_L/(p)` and `= 0` modulo all other prime ideals.
pub struct SplitCRTTransform {
    ring: Zn64B,
    deg_K: usize,
    deg_Lʼ: usize,
    ntts_K: Vec<NTTEitherRadix>,
    ntts_Lʼ: Vec<NTTEitherRadix>,
    ntt_crt_kernel_K: Vec<El<Zn64B>>,
    ntt_crt_kernel_Lʼ: Vec<El<Zn64B>>,
    ntt_inv_crt_kernel_K: Vec<El<Zn64B>>,
    ntt_inv_crt_kernel_Lʼ: Vec<El<Zn64B>>,
    ntt_dual_crt_kernel: Vec<El<Zn64B>>,
}

impl SplitCRTTransform {
    #[instrument(skip_all)]
    pub fn new(Zp: Zn64B, field_descriptor: &FieldDescriptor) -> SplitCRTTransform {
        let m = field_descriptor.m();
        let Zp_field = Zp.as_field().unwrap();

        let ntts_Lʼ = field_descriptor
            .GalLʼQ_cyclic_decomposition()
            .iter()
            .map(|(_, l)| NTTEitherRadix::new(Zp, *l))
            .collect::<Vec<_>>();
        let ntts_K = field_descriptor
            .GalKQ_cyclic_decomposition()
            .iter()
            .map(|(_, l)| NTTEitherRadix::new(Zp, *l))
            .collect::<Vec<_>>();
        let mut result = SplitCRTTransform {
            deg_Lʼ: field_descriptor.deg_Lʼ(),
            deg_K: field_descriptor.deg_K(),
            ntt_crt_kernel_K: Vec::new(),
            ntt_crt_kernel_Lʼ: Vec::new(),
            ntt_inv_crt_kernel_K: Vec::new(),
            ntt_inv_crt_kernel_Lʼ: Vec::new(),
            ntt_dual_crt_kernel: Vec::new(),
            ring: Zp,
            ntts_K,
            ntts_Lʼ,
        };

        let rou = get_prim_root_of_unity(Zp_field, &int_cast(m as i64, ZZbig, ZZi64)).unwrap();
        let rou = Zp_field.get_ring().unwrap_element(rou);
        let GalMQ = field_descriptor.GalMQ();

        let GalMʼLʼ_elements = field_descriptor.GalMʼLʼ().enumerate_elements().collect::<Vec<_>>();
        let rou_Mʼ = Zp.pow(
            rou,
            pmod(
                ZZi64.inv_crt(
                    [&1, &0],
                    [&(field_descriptor.mʼ() as i64), &(field_descriptor.m_K() as i64)],
                ),
                field_descriptor.m(),
            ),
        );
        let mut crt_kernel_Lʼ = Vec::new();
        for index_gal in field_descriptor.GalLʼQ_transversal() {
            let index_gal_inv = GalMQ.inv(&index_gal);
            crt_kernel_Lʼ.push(Zp.sum(GalMʼLʼ_elements.iter().map(|x| {
                Zp.pow(
                    rou_Mʼ,
                    GalMQ
                        .underlying_ring()
                        .smallest_positive_lift(*GalMQ.as_ring_el(&GalMQ.op_ref(x, &index_gal_inv)))
                        as usize,
                )
            })));
        }

        let GalM_KK_elements = field_descriptor.GalM_KK().enumerate_elements().collect::<Vec<_>>();
        let rou_M_K = Zp.pow(
            rou,
            pmod(
                ZZi64.inv_crt(
                    [&0, &1],
                    [&(field_descriptor.mʼ() as i64), &(field_descriptor.m_K() as i64)],
                ),
                field_descriptor.m(),
            ),
        );
        let mut crt_kernel_K = Vec::new();
        for index_gal in field_descriptor.GalKQ_transversal() {
            let index_gal_inv = GalMQ.inv(&index_gal);
            crt_kernel_K.push(Zp.sum(GalM_KK_elements.iter().map(|x| {
                Zp.pow(
                    rou_M_K,
                    GalMQ
                        .underlying_ring()
                        .smallest_positive_lift(*GalMQ.as_ring_el(&GalMQ.op_ref(x, &index_gal_inv)))
                        as usize,
                )
            })));
        }

        let mut rev_crt_kernel = multi_cartesian_product(
            field_descriptor.GalLQ_cyclic_decomposition().iter().map(|(_, l)| 0..*l),
            |idxs| {
                let idx_Lʼ = field_descriptor
                    .GalLʼQ_cyclic_decomposition()
                    .iter()
                    .map(|(_, l)| *l)
                    .zip(idxs)
                    .fold(0, |current, (l, i)| current * l + ((l - i) % l));
                let idx_K = field_descriptor
                    .GalKQ_cyclic_decomposition()
                    .iter()
                    .map(|(_, l)| *l)
                    .zip(idxs.iter().skip(field_descriptor.GalLʼQ_cyclic_decomposition().len()))
                    .fold(0, |current, (l, i)| current * l + ((l - i) % l));
                return Zp.mul_ref(&crt_kernel_Lʼ[idx_Lʼ], &crt_kernel_K[idx_K]);
            },
            |_, x| *x,
        )
        .collect::<Vec<_>>();
        result.multidim_ntt(&mut rev_crt_kernel);
        let ntt_dual_crt_kernel = rev_crt_kernel.into_iter().map(|x| Zp.invert(&x).unwrap()).collect();
        result.ntt_dual_crt_kernel = ntt_dual_crt_kernel;

        result.multidim_ntt_Lʼ(&mut crt_kernel_Lʼ);
        result.ntt_crt_kernel_Lʼ = crt_kernel_Lʼ;

        let mut ntt_inv_crt_kernel_Lʼ = Vec::new();
        ntt_inv_crt_kernel_Lʼ.extend(result.ntt_crt_kernel_Lʼ.iter().map(|x| Zp.invert(x).unwrap()));
        result.ntt_inv_crt_kernel_Lʼ = ntt_inv_crt_kernel_Lʼ;

        result.multidim_ntt_K(&mut crt_kernel_K);
        result.ntt_crt_kernel_K = crt_kernel_K;

        let mut ntt_inv_crt_kernel_K = Vec::new();
        ntt_inv_crt_kernel_K.extend(result.ntt_crt_kernel_K.iter().map(|x| Zp.invert(x).unwrap()));
        result.ntt_inv_crt_kernel_K = ntt_inv_crt_kernel_K;

        return result;
    }

    #[instrument(skip_all)]
    pub fn dual_crt_transform(&self, data: &mut [El<Zn64B>]) {
        self.multidim_ntt(data);
        for (twiddle, x) in self.ntt_dual_crt_kernel.iter().zip(data.iter_mut()) {
            self.ring.mul_assign_ref(x, twiddle);
        }
        self.multidim_intt(data);
    }

    pub fn ring(&self) -> &Zn64B { &self.ring }

    #[instrument(skip_all)]
    fn multidim_ntt(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_K * self.deg_Lʼ, data.len());

        let mut block_len = self.deg_K * self.deg_Lʼ;
        for ntt in self.ntts_Lʼ.iter().chain(&self.ntts_K) {
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.ntt(chunk, block_len / ntt.len());
            }
            block_len /= ntt.len();
        }
        debug_assert_eq!(1, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_ntt_K(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_K, data.len());

        let mut block_len = self.deg_K;
        for ntt in self.ntts_K.iter() {
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.ntt(chunk, block_len / ntt.len());
            }
            block_len /= ntt.len();
        }
        debug_assert_eq!(1, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_ntt_Lʼ(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_Lʼ, data.len());

        let mut block_len = self.deg_Lʼ;
        for ntt in self.ntts_Lʼ.iter() {
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.ntt(chunk, block_len / ntt.len());
            }
            block_len /= ntt.len();
        }
        debug_assert_eq!(1, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_intt(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_K * self.deg_Lʼ, data.len());

        let mut block_len = 1;
        for ntt in self.ntts_Lʼ.iter().chain(&self.ntts_K).rev() {
            block_len *= ntt.len();
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.intt(chunk, block_len / ntt.len());
            }
        }
        debug_assert_eq!(self.deg_K * self.deg_Lʼ, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_intt_Lʼ(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_Lʼ, data.len());

        let mut block_len = 1;
        for ntt in self.ntts_Lʼ.iter().rev() {
            block_len *= ntt.len();
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.intt(chunk, block_len / ntt.len());
            }
        }
        debug_assert_eq!(self.deg_Lʼ, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_intt_K(&self, data: &mut [El<Zn64B>]) {
        assert_eq!(self.deg_K, data.len());

        let mut block_len = 1;
        for ntt in self.ntts_K.iter().rev() {
            block_len *= ntt.len();
            for chunk in data.chunks_exact_mut(block_len) {
                ntt.intt(chunk, block_len / ntt.len());
            }
        }
        debug_assert_eq!(self.deg_K, block_len);
    }

    #[instrument(skip_all)]
    pub fn crt_transform(&self, data: &mut [El<Zn64B>]) {
        self.multidim_ntt(data);
        for (twiddle, chunk) in self.ntt_crt_kernel_Lʼ.iter().zip(data.chunks_exact_mut(self.deg_K)) {
            for x in chunk {
                self.ring.mul_assign(x, *twiddle);
            }
        }
        for chunk in data.chunks_exact_mut(self.deg_K) {
            for (twiddle, x) in self.ntt_crt_kernel_K.iter().zip(chunk) {
                self.ring.mul_assign(x, *twiddle);
            }
        }
        self.multidim_intt(data);
    }

    #[instrument(skip_all)]
    pub fn crt_transform_Lʼ(&self, data: &mut [El<Zn64B>]) {
        self.multidim_ntt_Lʼ(data);
        for (twiddle, x) in self.ntt_crt_kernel_Lʼ.iter().zip(data.iter_mut()) {
            self.ring.mul_assign(x, *twiddle);
        }
        self.multidim_intt_Lʼ(data);
    }

    #[instrument(skip_all)]
    pub fn crt_transform_K(&self, data: &mut [El<Zn64B>]) {
        self.multidim_ntt_K(data);
        for (twiddle, x) in self.ntt_crt_kernel_K.iter().zip(data.iter_mut()) {
            self.ring.mul_assign(x, *twiddle);
        }
        self.multidim_intt_K(data);
    }

    #[instrument(skip_all)]
    pub fn inv_crt_transform(&self, data: &mut [El<Zn64B>]) {
        self.multidim_ntt(data);
        for (twiddle, chunk) in self.ntt_inv_crt_kernel_Lʼ.iter().zip(data.chunks_exact_mut(self.deg_K)) {
            for x in chunk {
                self.ring.mul_assign(x, *twiddle);
            }
        }
        for chunk in data.chunks_exact_mut(self.deg_K) {
            for (twiddle, x) in self.ntt_inv_crt_kernel_K.iter().zip(chunk) {
                self.ring.mul_assign(x, *twiddle);
            }
        }
        self.multidim_intt(data);
    }
}

pub enum NTTEitherRadix {
    Radix2(NTTRadix2),
    Radix3(NTTRadix3),
}

impl NTTEitherRadix {
    pub fn new(Zp: Zn64B, len: usize) -> Self {
        let (p, e) = is_prime_power(ZZi64, &(len as i64)).unwrap();
        match p {
            2 => Self::Radix2(NTTRadix2::new(Zp, e)),
            3 => Self::Radix3(NTTRadix3::new(Zp, e)),
            _ => unimplemented!(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NTTEitherRadix::Radix2(ntt) => ntt.len(),
            NTTEitherRadix::Radix3(ntt) => ntt.len(),
        }
    }

    pub fn ntt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        match self {
            NTTEitherRadix::Radix2(ntt) => ntt.ntt(data, block_len),
            NTTEitherRadix::Radix3(ntt) => ntt.ntt(data, block_len),
        }
    }

    pub fn intt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        match self {
            NTTEitherRadix::Radix2(ntt) => ntt.intt(data, block_len),
            NTTEitherRadix::Radix3(ntt) => ntt.intt(data, block_len),
        }
    }
}

pub struct NTTRadix2 {
    hom: CanHom<Zn64BFastmul, Zn64B>,
    twiddles: Vec<El<Zn64BFastmul>>,
    inv_twiddles: Vec<El<Zn64BFastmul>>,
    log2_n: usize,
    rou: El<Zn64B>,
    inv_n: El<Zn64B>,
}

impl NTTRadix2 {
    #[instrument(skip_all)]
    pub fn new(Zp: Zn64B, log2_n: usize) -> Self {
        let Zp_as_field = Zp.as_field().unwrap();
        let rou = get_prim_root_of_unity_pow2(Zp_as_field, log2_n).unwrap();
        let Zp_fastmul = Zn64BFastmul::new(Zp).unwrap();
        let rou = Zp_as_field.get_ring().unwrap_element(rou);
        let inv_rou = Zp.invert(&rou).unwrap();
        let mut twiddles = Vec::with_capacity(1 << log2_n);
        twiddles.resize_with(1 << log2_n, || Zp_fastmul.one());
        let mut inv_twiddles = Vec::with_capacity(1 << log2_n);
        inv_twiddles.resize_with(1 << log2_n, || Zp_fastmul.one());
        let to_fastmul = Zp.can_iso(&Zp_fastmul).unwrap();

        for stage in 0..log2_n {
            let butterfly_count = 1 << stage;
            let base_rou = Zp.pow(rou, 1 << (log2_n - stage - 1));
            let base_inv_rou = Zp.pow(inv_rou, 1 << (log2_n - stage - 1));
            let mut current_rou = Zp.one();
            let mut current_inv_rou = Zp.one();
            for i in 0..butterfly_count {
                twiddles[bitreverse(i, stage) + butterfly_count] = to_fastmul.map(current_rou);
                inv_twiddles[bitreverse(i, stage) + butterfly_count] = to_fastmul.map(current_inv_rou);
                Zp.mul_assign(&mut current_rou, base_rou);
                Zp.mul_assign(&mut current_inv_rou, base_inv_rou);
            }
        }
        return Self {
            inv_n: Zp.invert(&Zp.coerce(&ZZi64, 1 << log2_n)).unwrap(),
            hom: Zp.into_can_hom(Zp_fastmul).ok().unwrap(),
            inv_twiddles,
            log2_n,
            twiddles,
            rou,
        };
    }

    pub fn len(&self) -> usize { self.twiddles.len() }

    pub fn ntt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        assert_eq!(block_len << self.log2_n, data.len());
        for x in data.iter_mut() {
            <Zn64BBase as CooleyTuckeyButterfly<Zn64BFastmulBase>>::prepare_for_fft(self.hom.codomain().get_ring(), x);
        }

        for stage in 0..self.log2_n {
            let butterfly_count = 1 << stage;
            let effective_block_len = block_len << (self.log2_n - stage - 1);
            for i in 0..butterfly_count {
                let twiddle = &self.twiddles[i + butterfly_count];
                for j in 0..effective_block_len {
                    let [a, b] = unsafe {
                        data.get_disjoint_unchecked_mut([
                            i * 2 * effective_block_len + j,
                            i * 2 * effective_block_len + effective_block_len + j,
                        ])
                    };
                    CooleyTuckeyButterfly::butterfly_new(&self.hom, a, b, twiddle);
                }
            }
        }
    }

    pub fn intt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        assert_eq!(block_len << self.log2_n, data.len());
        for x in data.iter_mut() {
            <Zn64BBase as CooleyTuckeyButterfly<Zn64BFastmulBase>>::prepare_for_inv_fft(
                self.hom.codomain().get_ring(),
                x,
            );
        }

        for stage in (0..self.log2_n).rev() {
            let butterfly_count = 1 << stage;
            let effective_block_len = block_len << (self.log2_n - stage - 1);
            for i in 0..butterfly_count {
                let twiddle = &self.inv_twiddles[i + butterfly_count];
                for j in 0..effective_block_len {
                    let [a, b] = unsafe {
                        data.get_disjoint_unchecked_mut([
                            i * 2 * effective_block_len + j,
                            i * 2 * effective_block_len + effective_block_len + j,
                        ])
                    };
                    CooleyTuckeyButterfly::inv_butterfly_new(&self.hom, a, b, twiddle);
                }
            }
        }

        for x in data.iter_mut() {
            self.hom.codomain().mul_assign(x, self.inv_n);
        }
    }
}

fn radix3_reverse(mut i: usize, digits: usize) -> usize {
    let mut result = 0;
    for _ in 0..digits {
        let (quo, rem) = (i / 3, i % 3);
        result = (result * 3) + rem;
        i = quo;
    }
    return result;
}

pub struct NTTRadix3 {
    hom: CanHom<Zn64BFastmul, Zn64B>,
    twiddle_z: El<Zn64BFastmul>,
    twiddles_t: Vec<El<Zn64BFastmul>>,
    twiddles_zt_sqr: Vec<El<Zn64BFastmul>>,
    inv_twiddles_t: Vec<El<Zn64BFastmul>>,
    inv_twiddles_t_sqr: Vec<El<Zn64BFastmul>>,
    log3_n: usize,
    rou: El<Zn64B>,
    inv_n: El<Zn64B>,
}

impl NTTRadix3 {
    #[instrument(skip_all)]
    pub fn new(Zp: Zn64B, log3_n: usize) -> Self {
        let Zp_as_field = Zp.as_field().unwrap();
        let n = 3usize.pow(log3_n as u32);
        let rou = get_prim_root_of_unity(Zp_as_field, &ZZbig.pow(int_cast(3, ZZbig, ZZi64), log3_n)).unwrap();
        let Zp_fastmul = Zn64BFastmul::new(Zp).unwrap();
        let rou = Zp_as_field.get_ring().unwrap_element(rou);

        let inv_rou = Zp.invert(&rou).unwrap();
        let twiddle_z = Zp.pow(rou, n / 3);
        let mut twiddles_t = Vec::with_capacity(n);
        twiddles_t.resize_with(n, || Zp_fastmul.one());
        let mut twiddles_zt_sqr = Vec::with_capacity(n);
        twiddles_zt_sqr.resize_with(n, || Zp_fastmul.one());
        let mut inv_twiddles_t = Vec::with_capacity(n);
        inv_twiddles_t.resize_with(n, || Zp_fastmul.one());
        let mut inv_twiddles_t_sqr = Vec::with_capacity(n);
        inv_twiddles_t_sqr.resize_with(n, || Zp_fastmul.one());
        let to_fastmul = Zp.can_iso(&Zp_fastmul).unwrap();

        for stage in 0..log3_n {
            let butterfly_count = 3usize.pow(stage as u32);
            let base_rou = Zp.pow(rou, 3usize.pow((log3_n - stage - 1) as u32));
            let base_inv_rou = Zp.pow(inv_rou, 3usize.pow((log3_n - stage - 1) as u32));
            let mut current_rou = Zp.one();
            let mut current_inv_rou = Zp.one();
            for i in 0..butterfly_count {
                twiddles_t[radix3_reverse(i, stage) + butterfly_count] = to_fastmul.map(current_rou);
                twiddles_zt_sqr[radix3_reverse(i, stage) + butterfly_count] =
                    to_fastmul.map(Zp.pow(Zp.mul(current_rou, twiddle_z), 2));
                inv_twiddles_t[radix3_reverse(i, stage) + butterfly_count] = to_fastmul.map(current_inv_rou);
                inv_twiddles_t_sqr[radix3_reverse(i, stage) + butterfly_count] =
                    to_fastmul.map(Zp.pow(current_inv_rou, 2));
                Zp.mul_assign(&mut current_rou, base_rou);
                Zp.mul_assign(&mut current_inv_rou, base_inv_rou);
            }
        }
        return Self {
            inv_n: Zp.invert(&Zp.coerce(&ZZi64, n as i64)).unwrap(),
            hom: Zp.into_can_hom(Zp_fastmul).ok().unwrap(),
            log3_n,
            twiddle_z: to_fastmul.map(twiddle_z),
            twiddles_t,
            twiddles_zt_sqr,
            inv_twiddles_t,
            inv_twiddles_t_sqr,
            rou,
        };
    }

    pub fn len(&self) -> usize { self.twiddles_t.len() }

    pub fn ntt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        assert_eq!(block_len * 3usize.pow(self.log3_n as u32), data.len());
        for x in data.iter_mut() {
            <Zn64BBase as CooleyTukeyRadix3Butterfly<Zn64BFastmulBase>>::prepare_for_fft(
                self.hom.codomain().get_ring(),
                x,
            );
        }

        for stage in 0..self.log3_n {
            let butterfly_count = 3usize.pow(stage as u32);
            let effective_block_len = block_len * 3usize.pow((self.log3_n - stage - 1) as u32);
            for i in 0..butterfly_count {
                let z = self.twiddle_z;
                let t = &self.twiddles_t[i + butterfly_count];
                let zt_sqr = &self.twiddles_zt_sqr[i + butterfly_count];
                for j in 0..effective_block_len {
                    let [a, b, c] = unsafe {
                        data.get_disjoint_unchecked_mut([
                            i * 3 * effective_block_len + j,
                            i * 3 * effective_block_len + effective_block_len + j,
                            i * 3 * effective_block_len + 2 * effective_block_len + j,
                        ])
                    };
                    CooleyTukeyRadix3Butterfly::butterfly(&self.hom, a, b, c, &z, &t, &zt_sqr);
                }
            }
        }
    }

    pub fn intt(&self, data: &mut [El<Zn64B>], block_len: usize) {
        assert_eq!(block_len * 3usize.pow(self.log3_n as u32), data.len());
        for x in data.iter_mut() {
            <Zn64BBase as CooleyTukeyRadix3Butterfly<Zn64BFastmulBase>>::prepare_for_inv_fft(
                self.hom.codomain().get_ring(),
                x,
            );
        }
        for stage in (0..self.log3_n).rev() {
            let butterfly_count = 3usize.pow(stage as u32);
            let effective_block_len = block_len * 3usize.pow((self.log3_n - stage - 1) as u32);
            for i in 0..butterfly_count {
                let z = self.twiddle_z;
                let t = &self.inv_twiddles_t[i + butterfly_count];
                let t_sqr = &self.inv_twiddles_t_sqr[i + butterfly_count];
                for j in 0..effective_block_len {
                    let [a, b, c] = unsafe {
                        data.get_disjoint_unchecked_mut([
                            i * 3 * effective_block_len + j,
                            i * 3 * effective_block_len + effective_block_len + j,
                            i * 3 * effective_block_len + 2 * effective_block_len + j,
                        ])
                    };
                    CooleyTukeyRadix3Butterfly::inv_butterfly(&self.hom, a, b, c, &z, &t, &t_sqr);
                }
            }
        }
        for x in data.iter_mut() {
            self.hom.codomain().mul_assign(x, self.inv_n);
        }
    }
}

#[cfg(test)]
use feanor_math::algorithms::matmul::ComputeInnerProduct;

#[test]
fn test_ntt_radix2() {
    feanor_tracing::DelayedLogger::init_test();
    let Fp = Zn64B::new(65537);
    let n = 16;
    let fft = NTTRadix2::new(Fp, 4);
    let rou = fft.rou;
    let original = (0..n).map(|x| Fp.int_hom().map(x)).collect::<Vec<_>>();
    let mut data = original.clone();
    fft.ntt(&mut data, 1);
    for (i, actual) in data.iter().enumerate() {
        assert_el_eq!(
            Fp,
            Fp.sum(
                original
                    .iter()
                    .enumerate()
                    .map(|(j, c)| Fp.mul(*c, Fp.pow(rou, bitreverse(i, 4) * j)))
            ),
            actual
        );
    }
}

#[test]
fn test_intt_radix2() {
    feanor_tracing::DelayedLogger::init_test();
    let Fp = Zn64B::new(65537);
    let n = 16;
    let fft = NTTRadix2::new(Fp, 4);
    let original = (0..n).map(|x| Fp.int_hom().map(x)).collect::<Vec<_>>();
    let mut data = original.clone();
    fft.ntt(&mut data, 1);
    fft.intt(&mut data, 1);
    for (expected, actual) in original.iter().zip(data.iter()) {
        assert_el_eq!(Fp, expected, actual);
    }
}

#[test]
fn test_ntt_radix3() {
    feanor_tracing::DelayedLogger::init_test();
    let Fp = Zn64B::new(163);
    let n = 27;
    let fft = NTTRadix3::new(Fp, 3);
    let rou = fft.rou;
    let original = (0..n).map(|x| Fp.int_hom().map(x)).collect::<Vec<_>>();
    let mut data = original.clone();
    fft.ntt(&mut data, 1);
    for (i, actual) in data.iter().enumerate() {
        assert_el_eq!(
            Fp,
            Fp.sum(
                original
                    .iter()
                    .enumerate()
                    .map(|(j, c)| Fp.mul(*c, Fp.pow(rou, radix3_reverse(i, 3) * j)))
            ),
            actual
        );
    }
}

#[test]
fn test_intt_radix3() {
    feanor_tracing::DelayedLogger::init_test();
    let Fp = Zn64B::new(163);
    let n = 27;
    let fft = NTTRadix3::new(Fp, 3);
    let original = (0..n).map(|x| Fp.int_hom().map(x)).collect::<Vec<_>>();
    let mut data = original.clone();
    fft.ntt(&mut data, 1);
    fft.intt(&mut data, 1);
    for (expected, actual) in original.iter().zip(data.iter()) {
        assert_el_eq!(Fp, expected, actual);
    }
}

#[test]
fn test_plain_crt_transform_multiply_small() {
    feanor_tracing::DelayedLogger::init_test();
    let Zp = Zn64B::new(7250161);
    let field_desc = field_192();
    let transform = SplitCRTTransform::new(Zp, &field_desc);
    let mut data = [Zp.one()]
        .into_iter()
        .chain((1..192).map(|_| Zp.zero()))
        .collect::<Vec<_>>();
    transform.crt_transform(&mut data);
    data.iter_mut().for_each(|x| Zp.square(x));
    transform.inv_crt_transform(&mut data);

    for (i, x) in data.into_iter().enumerate() {
        assert!(
            Zp.smallest_lift(x).abs() <= field_desc.m() as i64,
            "expected multiplication result to be at most {} in absolute value, but got {} at index {}",
            field_desc.m(),
            Zp.smallest_lift(x),
            i
        );
    }
}

#[test]
fn test_dual_crt_transform() {
    feanor_tracing::DelayedLogger::init_test();
    let Zp = Zn64B::new(7250161);
    let field_desc = field_192();
    let transform = SplitCRTTransform::new(Zp, &field_desc);
    let mut primal = [Zp.one()]
        .into_iter()
        .chain((1..192).map(|_| Zp.zero()))
        .collect::<Vec<_>>();
    transform.crt_transform(&mut primal);
    let mut dual = [Zp.one()]
        .into_iter()
        .chain((1..192).map(|_| Zp.zero()))
        .collect::<Vec<_>>();
    transform.dual_crt_transform(&mut dual);
    let result = Zp.get_ring().inner_product_ref(primal.iter().zip(&dual));
    assert_el_eq!(Zp, Zp.one(), result);

    let mut dual = [Zp.zero(), Zp.one()]
        .into_iter()
        .chain((2..192).map(|_| Zp.zero()))
        .collect::<Vec<_>>();
    transform.dual_crt_transform(&mut dual);
    let result = Zp.get_ring().inner_product_ref(primal.iter().zip(&dual));
    assert_el_eq!(Zp, Zp.zero(), result);
}

#[test]
fn test_plain_crt_transform_period_basis_relation() {
    feanor_tracing::DelayedLogger::init_test();
    let Zp = Zn64B::new(1037441497);
    let field_desc = field_18();
    let transform = SplitCRTTransform::new(Zp, &field_desc);
    let mut data = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        .into_iter()
        .map(|x| Zp.int_hom().map(x))
        .collect::<Vec<_>>();
    transform.crt_transform(&mut data);
    data.iter_mut().for_each(|x| Zp.square(x));
    transform.inv_crt_transform(&mut data);

    let expected_in_K = [3, 4, 2, -2, -4, -2];
    let expected = (0..3).flat_map(|_| expected_in_K.iter().copied()).collect::<Vec<_>>();
    for (expected, actual) in expected.iter().zip(data.iter()) {
        assert_el_eq!(Zp, Zp.int_hom().map(*expected), actual);
    }
}

#[test]
fn test_plain_crt_transform_minpoly() {
    feanor_tracing::DelayedLogger::init_test();
    let Zp = Zn64B::new(7250161);
    let transform = SplitCRTTransform::new(Zp, &field_192());

    // minpoly (mod 7250161) of the first period basis element, i.e. Tr_(M|L)(ζ), as computed with sage
    #[rustfmt::skip]
    let coefficients: [i64; 193] = [
        3179273, 2994446, 7073685, 639340, 6045694, 432130, 5494804, 1586050, 4024764, 2498828, 1440117, 352602, 3781025, 4344632, 5931514, 
        5878587, 3538253, 1243463, 1420235, 3881076, 3110882, 2555293, 1520334, 5153320, 1250343, 6186881, 3714465, 660468, 4363445, 4374644, 
        4286078, 5310517, 422171, 2761586, 3183753, 5051396, 3281477, 5497552, 5692780, 5439851, 3005673, 6210249, 3587556, 1371788, 753824, 
        2840984, 362545, 5501635, 4196355, 6215278, 6523475, 5432620, 3603024, 3383664, 2552143, 5759219, 5970991, 1494034, 1165096, 1571672, 
        2590629, 3219842, 5093123, 988720, 6536220, 4643588, 7206309, 1469818, 133241, 6906525, 3035879, 2351796, 180354, 3112566, 3594713, 
        4638265, 3976744, 5963187, 958565, 781243, 7196859, 2168376, 5292678, 2966630, 550224, 1672260, 4752962, 5932308, 525306, 3758711, 
        2114688, 4806161, 5915570, 6925890, 6282966, 454931, 3422496, 6587337, 4109166, 3587879, 7130630, 3356604, 6314701, 3418098, 1003048, 
        197893, 3928053, 2544956, 5790299, 1067070, 6597269, 1827043, 1378997, 51904, 706324, 2432373, 2429969, 2303467, 938082, 4471522, 
        3525084, 7223864, 3393746, 412776, 5684384, 4627038, 1614225, 4047090, 2522994, 5662782, 5734924, 2329660, 5376907, 6681060, 3232590, 
        6373129, 2530422, 1171959, 3894829, 7101094, 4028786, 4804812, 2593112, 1997546, 1709608, 5566459, 6797346, 1783830, 3740433, 5867056, 
        3874156, 6566585, 3682630, 3368954, 6365569, 5265813, 1610414, 5731959, 5919574, 2595768, 1876904, 2106899, 1176203, 2669308, 5694302, 
        244423, 6852963, 1308932, 5546146, 606853, 3956284, 2904142, 5527645, 575831, 1137369, 6245907, 6895852, 2395719, 895380, 4098520, 
        2319840, 6980233, 339610, 3426968, 1003535, 3636443, 3472067, 3263347, 7191329, 41279, 7246441, 1, 1
    ];
    let mut data = [Zp.one()]
        .into_iter()
        .chain((1..192).map(|_| Zp.zero()))
        .collect::<Vec<_>>();
    transform.crt_transform(&mut data);

    for x in data {
        assert_el_eq!(
            Zp,
            Zp.zero(),
            Zp.get_ring().inner_product(
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (Zp.coerce(&ZZi64, *c), Zp.pow(x, i)))
            )
        );
    }
}

#[test]
fn test_plain_inv_crt_transform() {
    feanor_tracing::DelayedLogger::init_test();
    let Zp = Zn64B::new(7250161);
    let transform = SplitCRTTransform::new(Zp, &field_192());
    let mut data = (0..192).map(|_| Zp.zero()).collect::<Vec<_>>();

    for i in 0..192 {
        data[i] = Zp.one();
        transform.crt_transform(&mut data);
        transform.inv_crt_transform(&mut data);
        for (j, x) in data.iter().enumerate() {
            if i == j {
                assert_el_eq!(Zp, Zp.one(), x);
            } else {
                assert_el_eq!(Zp, Zp.zero(), x);
            }
        }
        data[i] = Zp.zero();
    }
}
