use std::f64::consts::PI;

use feanor_math::algorithms::fft::cooley_tuckey::{CooleyTuckeyButterfly, bitreverse};
use feanor_math::algorithms::fft::radix3::CooleyTukeyRadix3Butterfly;
use feanor_math::group::AbelianGroupStore;
use feanor_math::prelude::*;
use feanor_math::ring_impls::float_complex::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use tracing::instrument;

use crate::fielddesc::*;
use crate::*;

pub const CC: Complex64 = Complex64::RING;

pub fn exp_2pii(phi: f64) -> El<Complex64> { CC.exp(CC.mul(Complex64::I, CC.from_f64(2. * PI * phi))) }

pub struct ComplexCRTTransform {
    n: usize,
    ffts: Vec<FFTEitherRadix>,
    fft_crt_kernel: Vec<El<Complex64>>,
    fft_inv_crt_kernel: Vec<El<Complex64>>,
}

impl ComplexCRTTransform {
    #[instrument(skip_all)]
    pub fn new(field_descriptor: &FieldDescriptor) -> ComplexCRTTransform {
        let m = field_descriptor.m();
        let n = field_descriptor.deg_L();

        let cyclic_factors = field_descriptor.GalLQ_cyclic_decomposition();
        let ffts = cyclic_factors
            .iter()
            .map(|(_, l)| FFTEitherRadix::new(*l))
            .collect::<Vec<_>>();
        let mut result = ComplexCRTTransform {
            n,
            fft_crt_kernel: Vec::new(),
            fft_inv_crt_kernel: Vec::new(),
            ffts,
        };

        let GalMQ = field_descriptor.GalMQ();
        let mut crt_kernel = Vec::with_capacity(n);
        let relative_galois_group_els = field_descriptor.GalML().enumerate_elements().collect::<Vec<_>>();
        for index_gal in field_descriptor.GalLQ_transversal() {
            let index_gal_inv = GalMQ.inv(&index_gal);
            crt_kernel.push(CC.sum(relative_galois_group_els.iter().map(|x| {
                exp_2pii(
                    GalMQ
                        .underlying_ring()
                        .smallest_positive_lift(*GalMQ.as_ring_el(&GalMQ.op_ref(x, &index_gal_inv)))
                        as f64
                        / m as f64,
                )
            })));
        }
        result.multidim_fft(&mut crt_kernel);
        result.fft_crt_kernel = crt_kernel;

        let mut ntt_inv_crt_kernel = Vec::with_capacity(n);
        ntt_inv_crt_kernel.extend(result.fft_crt_kernel.iter().map(|x| CC.invert(x).unwrap()));
        result.fft_inv_crt_kernel = ntt_inv_crt_kernel;

        return result;
    }

    #[instrument(skip_all)]
    fn multidim_fft(&self, data: &mut [El<Complex64>]) {
        assert_eq!(self.n, data.len());

        let mut block_len = self.n;
        for fft in &self.ffts {
            for chunk in data.chunks_exact_mut(block_len) {
                fft.fft(chunk, block_len / fft.len());
            }
            block_len /= fft.len();
        }
        debug_assert_eq!(1, block_len);
    }

    #[instrument(skip_all)]
    fn multidim_ifft(&self, data: &mut [El<Complex64>]) {
        assert_eq!(self.n, data.len());

        let mut block_len = 1;
        for fft in self.ffts.iter().rev() {
            block_len *= fft.len();
            for chunk in data.chunks_exact_mut(block_len) {
                fft.ifft(chunk, block_len / fft.len());
            }
        }
        debug_assert_eq!(self.n, block_len);
    }

    #[instrument(skip_all)]
    pub fn crt_transform(&self, data: &mut [El<Complex64>]) {
        self.multidim_fft(data);
        for (twiddle, x) in self.fft_crt_kernel.iter().zip(data.iter_mut()) {
            CC.mul_assign(x, *twiddle);
        }
        self.multidim_ifft(data);
    }

    pub fn inv_crt_transform(&self, data: &mut [El<Complex64>]) {
        self.multidim_fft(data);
        for (twiddle, x) in self.fft_inv_crt_kernel.iter().zip(data.iter_mut()) {
            CC.mul_assign(x, *twiddle);
        }
        self.multidim_ifft(data);
    }
}

pub enum FFTEitherRadix {
    Radix2(FFTRadix2),
    Radix3(FFTRadix3),
}

impl FFTEitherRadix {
    pub fn new(len: usize) -> Self {
        let (p, e) = is_prime_power(ZZi64, &(len as i64)).unwrap();
        match p {
            2 => Self::Radix2(FFTRadix2::new(e)),
            3 => Self::Radix3(FFTRadix3::new(e)),
            _ => unimplemented!(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            FFTEitherRadix::Radix2(fft) => fft.len(),
            FFTEitherRadix::Radix3(fft) => fft.len(),
        }
    }

    pub fn fft(&self, data: &mut [El<Complex64>], block_len: usize) {
        match self {
            FFTEitherRadix::Radix2(fft) => fft.fft(data, block_len),
            FFTEitherRadix::Radix3(fft) => fft.fft(data, block_len),
        }
    }

    pub fn ifft(&self, data: &mut [El<Complex64>], block_len: usize) {
        match self {
            FFTEitherRadix::Radix2(fft) => fft.ifft(data, block_len),
            FFTEitherRadix::Radix3(fft) => fft.ifft(data, block_len),
        }
    }
}

pub struct FFTRadix2 {
    twiddles: Vec<El<Complex64>>,
    inv_twiddles: Vec<El<Complex64>>,
    log2_n: usize,
}

impl FFTRadix2 {
    #[instrument(skip_all)]
    pub fn new(log2_n: usize) -> Self {
        let mut twiddles = Vec::with_capacity(1 << log2_n);
        twiddles.resize_with(1 << log2_n, || CC.one());
        let mut inv_twiddles = Vec::with_capacity(1 << log2_n);
        inv_twiddles.resize_with(1 << log2_n, || CC.one());

        for stage in 0..log2_n {
            let butterfly_count = 1 << stage;
            let k = 1 << (stage + 1);
            for i in 0..butterfly_count {
                twiddles[bitreverse(i, stage) + butterfly_count] = exp_2pii(i as f64 / k as f64);
                inv_twiddles[bitreverse(i, stage) + butterfly_count] = exp_2pii(-1. * i as f64 / k as f64);
            }
        }
        return Self {
            inv_twiddles,
            log2_n,
            twiddles,
        };
    }

    pub fn len(&self) -> usize { self.twiddles.len() }

    #[instrument(skip_all, fields(parallel = block_len))]
    pub fn fft(&self, data: &mut [El<Complex64>], block_len: usize) {
        assert_eq!(block_len << self.log2_n, data.len());
        for x in data.iter_mut() {
            <Complex64Base as CooleyTuckeyButterfly<Complex64Base>>::prepare_for_fft(CC.get_ring(), x);
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
                    CooleyTuckeyButterfly::butterfly_new(&CC.identity(), a, b, twiddle);
                }
            }
        }
    }

    #[instrument(skip_all, fields(parallel = block_len))]
    pub fn ifft(&self, data: &mut [El<Complex64>], block_len: usize) {
        assert_eq!(block_len << self.log2_n, data.len());
        for x in data.iter_mut() {
            <Complex64Base as CooleyTuckeyButterfly<Complex64Base>>::prepare_for_inv_fft(CC.get_ring(), x);
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
                    CooleyTuckeyButterfly::inv_butterfly_new(&CC.identity(), a, b, twiddle);
                }
            }
        }

        let inv_n = CC.from_f64(1. / (1 << self.log2_n) as f64);
        for x in data.iter_mut() {
            CC.mul_assign(x, inv_n);
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

pub struct FFTRadix3 {
    twiddle_z: El<Complex64>,
    twiddles_t: Vec<El<Complex64>>,
    twiddles_zt_sqr: Vec<El<Complex64>>,
    inv_twiddles_t: Vec<El<Complex64>>,
    inv_twiddles_t_sqr: Vec<El<Complex64>>,
    log3_n: usize,
}

impl FFTRadix3 {
    #[instrument(skip_all)]
    pub fn new(log3_n: usize) -> Self {
        let n = 3usize.pow(log3_n as u32);

        let mut twiddles_t = Vec::with_capacity(n);
        twiddles_t.resize_with(n, || CC.one());
        let mut twiddles_zt_sqr = Vec::with_capacity(n);
        twiddles_zt_sqr.resize_with(n, || CC.one());
        let mut inv_twiddles_t = Vec::with_capacity(n);
        inv_twiddles_t.resize_with(n, || CC.one());
        let mut inv_twiddles_t_sqr = Vec::with_capacity(n);
        inv_twiddles_t_sqr.resize_with(n, || CC.one());

        for stage in 0..log3_n {
            let butterfly_count = 3usize.pow(stage as u32);
            let k = 3usize.pow(stage as u32 + 1);
            for i in 0..butterfly_count {
                twiddles_t[radix3_reverse(i, stage) + butterfly_count] = exp_2pii(i as f64 / k as f64);
                twiddles_zt_sqr[radix3_reverse(i, stage) + butterfly_count] =
                    exp_2pii(2. * i as f64 / k as f64 + 2. / 3.);
                inv_twiddles_t[radix3_reverse(i, stage) + butterfly_count] = exp_2pii(-1. * i as f64 / k as f64);
                inv_twiddles_t_sqr[radix3_reverse(i, stage) + butterfly_count] = exp_2pii(-2. * i as f64 / k as f64);
            }
        }
        return Self {
            log3_n,
            twiddle_z: exp_2pii(1. / 3.),
            twiddles_t,
            twiddles_zt_sqr,
            inv_twiddles_t,
            inv_twiddles_t_sqr,
        };
    }

    pub fn len(&self) -> usize { self.twiddles_t.len() }

    #[instrument(skip_all, fields(parallel = block_len))]
    pub fn fft(&self, data: &mut [El<Complex64>], block_len: usize) {
        assert_eq!(block_len * 3usize.pow(self.log3_n as u32), data.len());
        for x in data.iter_mut() {
            <Complex64Base as CooleyTukeyRadix3Butterfly<Complex64Base>>::prepare_for_fft(CC.get_ring(), x);
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
                    CooleyTukeyRadix3Butterfly::butterfly(&CC.identity(), a, b, c, &z, &t, &zt_sqr);
                }
            }
        }
    }

    #[instrument(skip_all, fields(parallel = block_len))]
    pub fn ifft(&self, data: &mut [El<Complex64>], block_len: usize) {
        assert_eq!(block_len * 3usize.pow(self.log3_n as u32), data.len());
        for x in data.iter_mut() {
            <Complex64Base as CooleyTukeyRadix3Butterfly<Complex64Base>>::prepare_for_inv_fft(CC.get_ring(), x);
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
                    CooleyTukeyRadix3Butterfly::inv_butterfly(&CC.identity(), a, b, c, &z, &t, &t_sqr);
                }
            }
        }

        let inv_n = CC.from_f64(1. / 3usize.pow(self.log3_n as u32) as f64);
        for x in data.iter_mut() {
            CC.mul_assign(x, inv_n);
        }
    }
}

#[cfg(test)]
use feanor_math::ring_impls::zn::zn_64b::Zn64B;

#[cfg(test)]
use crate::ntt;

#[test]
fn test_same_as_zp() {
    let Zp = Zn64B::new(14503198313521);
    let field_descriptor = field_192();
    let ntt = ntt::SplitCRTTransform::new(Zp, &field_descriptor);
    let fft = ComplexCRTTransform::new(&field_descriptor);
    let n = field_descriptor.deg_L();

    let mut lhs_Zp = (0..n).map(|_| Zp.zero()).collect::<Vec<_>>();
    let mut lhs_CC = (0..n).map(|_| CC.zero()).collect::<Vec<_>>();
    lhs_Zp[0] = Zp.one();
    lhs_CC[0] = CC.one();
    ntt.crt_transform(&mut lhs_Zp);
    fft.crt_transform(&mut lhs_CC);

    for i in 0..n {
        let mut rhs_Zp = (0..n).map(|_| Zp.zero()).collect::<Vec<_>>();
        let mut rhs_CC = (0..n).map(|_| CC.zero()).collect::<Vec<_>>();
        rhs_Zp[i] = Zp.one();
        rhs_CC[i] = CC.one();
        let mut dst_Zp = (0..n).map(|_| Zp.zero()).collect::<Vec<_>>();
        let mut dst_CC = (0..n).map(|_| CC.zero()).collect::<Vec<_>>();

        ntt.crt_transform(&mut rhs_Zp);
        fft.crt_transform(&mut rhs_CC);
        for j in 0..n {
            dst_Zp[j] = Zp.mul(lhs_Zp[j], rhs_Zp[j]);
            dst_CC[j] = CC.mul(lhs_CC[j], rhs_CC[j]);
        }
        ntt.inv_crt_transform(&mut dst_Zp);
        fft.inv_crt_transform(&mut dst_CC);

        for j in 0..n {
            assert_eq!(
                Zp.smallest_lift(dst_Zp[j]),
                CC.get_ring().closest_gaussian_int(dst_CC[j]).0
            );
        }
    }
}
