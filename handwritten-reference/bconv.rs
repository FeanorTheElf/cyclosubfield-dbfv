use std::alloc::{Allocator, Global};
use std::array::from_fn;

use feanor_math::matrix::*;
use feanor_math::prelude::*;
use feanor_math::ring_impls::zn::ZnRingStore;
use feanor_math::ring_impls::zn::zn_64b::Zn64B;
use feanor_math::seq::VectorView;
use tracing::instrument;

use crate::matmul::{
    BLOCK_SIZE, matmul_4x4_i64_to_i128, pad_to_block, vecmatmul_4_i128_to_i128, vecmatmul_4_i128_to_i128_wrapping,
};

const ZZi128: StaticRing<i128> = StaticRing::RING;

/// Stores values for an almost exact conversion between RNS bases.
/// A complete conversion refers to the function
/// ```text
///   Z/QZ -> Z/Q'Z, x -> [lift(x)]
/// ```
/// In our case, the output of the function is allowed to have an error of `{ -Q, 0, Q }`,
/// unless the shortest lift of the input is bounded by `Q/4`, in which case the result
/// is always correct.
///
/// # Copied from fheanor
pub struct BConv<A = Global>
where
    A: Allocator + Clone,
{
    from_summands: Vec<Zn64B>,
    to_summands: Vec<Zn64B>,
    /// the values `q/Q mod q` for each RNS factor q dividing Q (ordered as `from_summands`)
    q_over_Q: Vec<El<Zn64B>>,
    /// shortest lifts of the values `Q/q mod q'` for each RNS factor q dividing Q (ordered
    /// as `from_summands`, mapped to col index) and q' dividing Q' (ordered as `to_summands`,
    /// mapped to row index)
    Q_over_q_mod: OwnedMatrix<i64>,
    /// the values `round( Q/q/gamma )` for each RNS factor `q` dividing `Q`; Unfortunately,
    /// these sometimes exceed 64 bits, thus cannot store them in the matrix `Q_over_q_mod`
    Q_over_q_downscaled: Vec<i128>,
    gamma: i128,
    /// `Q mod q'` for every RNS factor q' of Q' (ordered as `to_summands`)
    Q_mod_q: Vec<El<Zn64B>>,
    allocator: A,
}

// we currently use `any_lift()`; I haven't yet documented it anywhere, but in fact the largest
// output of `Zn64B_64::Zn64B::any_lift()` is currently `6 * modulus()`
const Zn64B_ANY_LIFT_FACTOR: i64 = 6;

impl BConv {
    pub fn new(in_rings: Vec<Zn64B>, out_rings: Vec<Zn64B>) -> Self {
        Self::new_with_alloc(in_rings, out_rings, Global)
    }
}

impl<A> BConv<A>
where
    A: Allocator + Clone,
{
    #[instrument(skip_all)]
    pub fn new_with_alloc(in_rings: Vec<Zn64B>, out_rings: Vec<Zn64B>, allocator: A) -> Self {
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64)));

        let max = |l, r| if ZZbig.is_geq(&l, &r) { l } else { r };
        let max_computation_result = ZZbig.prod(
            [
                in_rings
                    .iter()
                    .map(|ring| int_cast(*ring.modulus() * Zn64B_ANY_LIFT_FACTOR, ZZbig, ZZi64))
                    .reduce(max)
                    .unwrap_or(ZZbig.zero()),
                out_rings
                    .iter()
                    .map(|ring| int_cast(*ring.modulus(), ZZbig, ZZi64))
                    .reduce(max)
                    .unwrap_or(ZZbig.zero()),
                ZZbig.int_hom().map(in_rings.len() as i32),
            ]
            .into_iter(),
        );
        assert!(
            ZZbig.is_lt(&max_computation_result, &ZZbig.power_of_two(i128::BITS as usize - 1)),
            "temporarily unreduced modular lift sum will overflow"
        );

        // When computing the approximate lifted value, we can work with `gamma` in place of `Q`, where
        // `gamma >= 4 r max(q)` (`q` runs through the input factors)
        let log2_r = ZZi64.abs_log2_ceil(&(in_rings.len() as i64)).unwrap_or(0);
        let log2_qmax = ZZi64
            .abs_log2_ceil(
                &(0..in_rings.len())
                    .map(|i| *in_rings.at(i).modulus())
                    .max()
                    .unwrap_or(0),
            )
            .unwrap_or(0);
        let log2_any_lift_factor = ZZi64.abs_log2_ceil(&Zn64B_ANY_LIFT_FACTOR).unwrap_or(0);
        let gamma = ZZbig.power_of_two(log2_r + log2_qmax + log2_any_lift_factor + 2);
        // we compute a sum of `r` summands, each being a product of a lifted value (mod `q`, `q | Q`) and
        // `gamma/q`; this must not overflow
        assert!(
            ZZbig.abs_log2_ceil(&gamma).unwrap() + log2_r + log2_any_lift_factor + 1
                < ZZi128.get_ring().representable_bits().unwrap(),
            "correction computation will overflow"
        );
        let gamma_log2 = ZZbig.abs_log2_ceil(&gamma).unwrap();
        assert!(gamma_log2 == ZZbig.abs_log2_floor(&gamma).unwrap());

        let Q_over_q_mod = OwnedMatrix::from_fn_in(
            pad_to_block(out_rings.len()),
            pad_to_block(in_rings.len()),
            |i, j| {
                if i < out_rings.len() && j < in_rings.len() {
                    let ring = out_rings.at(i);
                    ring.smallest_lift(
                        ring.coerce(
                            &ZZbig,
                            ZZbig
                                .checked_div(&Q, &int_cast(*in_rings.at(j).modulus(), ZZbig, ZZi64))
                                .unwrap(),
                        ),
                    )
                } else {
                    0
                }
            },
            Global,
        );
        let Q_over_q_downscaled = (0..pad_to_block(in_rings.len()))
            .map(|j| {
                if j < in_rings.len() {
                    int_cast(
                        ZZbig.rounded_div(gamma.clone(), &int_cast(*in_rings.at(j).modulus(), ZZbig, ZZi64)),
                        ZZi128,
                        ZZbig,
                    )
                } else {
                    0
                }
            })
            .collect();
        let q_over_Q = (0..(in_rings.len()))
            .map(|i| {
                in_rings
                    .at(i)
                    .invert(
                        &in_rings.at(i).coerce(
                            &ZZbig,
                            ZZbig
                                .checked_div(&Q, &int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64))
                                .unwrap(),
                        ),
                    )
                    .unwrap()
            })
            .collect();

        Self {
            Q_over_q_mod,
            Q_over_q_downscaled,
            q_over_Q,
            Q_mod_q: (0..out_rings.len())
                .map(|i| out_rings.at(i).coerce(&ZZbig, Q.clone()))
                .collect(),
            gamma: ZZi128.power_of_two(gamma_log2),
            allocator: allocator.clone(),
            from_summands: in_rings,
            to_summands: out_rings,
        }
    }
}

impl<A> BConv<A>
where
    A: Allocator + Clone,
{
    pub fn input_rings<'a>(&'a self) -> &'a [Zn64B] { &self.from_summands }

    pub fn output_rings<'a>(&'a self) -> &'a [Zn64B] { &self.to_summands }

    /// Performs the (almost) exact RNS base conversion
    /// ```text
    ///   Z/QZ -> Z/Q'Z, x -> smallest_lift(x) + kQ mod Q'
    /// ```
    /// where `k in { -1, 0, 1 }`.
    ///
    /// Furthermore, if the shortest lift of the input is bounded by `Q/4`,
    /// then the result is guaranteed to be exact.
    #[instrument(skip_all)]
    pub fn bconv<V1, V2>(&self, input: Submatrix<V1, El<Zn64B>>, mut output: SubmatrixMut<V2, El<Zn64B>>)
    where
        V1: AsPointerToSlice<El<Zn64B>>,
        V2: AsPointerToSlice<El<Zn64B>>,
    {
        {
            assert_eq!(input.row_count(), self.input_rings().len());
            assert_eq!(output.row_count(), self.output_rings().len());
            assert_eq!(input.col_count(), output.col_count());

            let in_len = input.row_count();
            let out_len = output.row_count();
            let col_count = input.col_count();

            let int_to_homs = (0..self.output_rings().len())
                .map(|k| self.output_rings().at(k).can_hom(&ZZi128).unwrap())
                .collect::<Vec<_>>();

            let mut lifts = OwnedMatrix::from_fn_in(
                pad_to_block(in_len),
                pad_to_block(col_count),
                |_, _| 0,
                self.allocator.clone(),
            );
            let mut lifts = lifts.data_mut();

            for i in 0..in_len {
                for j in 0..col_count {
                    *lifts.at_mut(i, j) = self.from_summands[i]
                        .any_lift(self.from_summands[i].mul_ref(input.at(i, j), self.q_over_Q.at(i)));
                    debug_assert!(
                        *lifts.at(i, 0) >= 0
                            && *lifts.at(i, 0) as i128
                                <= Zn64B_ANY_LIFT_FACTOR as i128 * *self.from_summands[i].modulus() as i128
                    );
                }
            }

            let mut output_unreduced = OwnedMatrix::zero(pad_to_block(out_len), pad_to_block(col_count), ZZi128);

            assert_eq!(0, self.Q_over_q_mod.row_count() % BLOCK_SIZE);
            assert_eq!(0, self.Q_over_q_mod.col_count() % BLOCK_SIZE);
            assert_eq!(0, output_unreduced.row_count() % BLOCK_SIZE);
            assert_eq!(0, lifts.row_count() % BLOCK_SIZE);
            for (lhs_blocks, mut res_block) in self
                .Q_over_q_mod
                .data()
                .row_iter()
                .array_chunks::<BLOCK_SIZE>()
                .zip(output_unreduced.data_mut().row_iter().array_chunks::<BLOCK_SIZE>())
            {
                for (j, rhs_block) in lifts.as_const().row_iter().array_chunks::<BLOCK_SIZE>().enumerate() {
                    matmul_4x4_i64_to_i128(
                        from_fn(|i| &lhs_blocks[i].as_chunks::<BLOCK_SIZE>().0[j]),
                        rhs_block,
                        &mut res_block,
                    );
                }
            }

            let mut output_correction: Vec<i128> = (0..pad_to_block(col_count)).map(|_| 0).collect();
            assert_eq!(0, self.Q_over_q_downscaled.len() % BLOCK_SIZE);
            for (lhs_block, rhs_block) in self
                .Q_over_q_downscaled
                .iter()
                .copied()
                .array_chunks::<BLOCK_SIZE>()
                .zip(lifts.as_const().row_iter().array_chunks::<BLOCK_SIZE>())
            {
                vecmatmul_4_i128_to_i128(lhs_block, rhs_block, &mut output_correction);
            }

            for j in 0..col_count {
                let mut correction = *output_correction.at(j);
                correction = ZZi128.rounded_div(correction, &self.gamma);

                for i in 0..out_len {
                    *output.at_mut(i, j) = self.to_summands[i].sub(
                        int_to_homs.at(i).map_ref(output_unreduced.at(i, j)),
                        self.to_summands[i].mul_ref_snd(int_to_homs[i].map(correction), &self.Q_mod_q[i]),
                    );
                }
            }
        }
    }
}

pub struct BConvPow2<A = Global>
where
    A: Allocator + Clone,
{
    from_summands: Vec<Zn64B>,
    /// `Q' = 2^log2_to`
    log2_to: usize,
    /// the values `q/Q mod Q` for each RNS factor q dividing Q (ordered as `from_summands`)
    q_over_Q: Vec<El<Zn64B>>,
    /// shortest lifts of the values `Q/q mod Q'` for each RNS factor q dividing Q (ordered
    /// as `from_summands`)
    Q_over_q_mod: Vec<i128>,
    /// the values `round( Q/q/gamma )` for each RNS factor `q` dividing `Q`; Unfortunately,
    /// these sometimes exceed 64 bits, thus cannot store them in the matrix `Q_over_q_mod`
    Q_over_q_downscaled: Vec<i128>,
    gamma: i128,
    /// `Q mod Q'`
    Q_mod_target: i128,
    allocator: A,
}

impl<A> BConvPow2<A>
where
    A: Allocator + Clone,
{
    #[instrument(skip_all)]
    pub fn new_with_alloc(in_rings: Vec<Zn64B>, log2_to: usize, allocator: A) -> Self {
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64)));

        // we do not have to be afraid about overflow, since operations are wrapping
        assert!(log2_to < i128::BITS as usize);

        // When computing the approximate lifted value, we can work with `gamma` in place of `Q`, where
        // `gamma >= 4 r max(q)` (`q` runs through the input factors)
        let log2_r = ZZi64.abs_log2_ceil(&(in_rings.len() as i64)).unwrap_or(0);
        let log2_qmax = ZZi64
            .abs_log2_ceil(
                &(0..in_rings.len())
                    .map(|i| *in_rings.at(i).modulus())
                    .max()
                    .unwrap_or(0),
            )
            .unwrap_or(0);
        let log2_any_lift_factor = ZZi64.abs_log2_ceil(&Zn64B_ANY_LIFT_FACTOR).unwrap_or(0);
        let gamma = ZZbig.power_of_two(log2_r + log2_qmax + log2_any_lift_factor + 2);
        // we compute a sum of `r` summands, each being a product of a lifted value (mod `q`, `q | Q`) and
        // `gamma/q`; this must not overflow
        assert!(
            ZZbig.abs_log2_ceil(&gamma).unwrap() + log2_r + log2_any_lift_factor + 1
                < ZZi128.get_ring().representable_bits().unwrap(),
            "correction computation will overflow"
        );
        let gamma_log2 = ZZbig.abs_log2_ceil(&gamma).unwrap();
        assert!(gamma_log2 == ZZbig.abs_log2_floor(&gamma).unwrap());

        let mut Q_over_q_mod = (0..pad_to_block(in_rings.len())).map(|_| 0).collect::<Vec<_>>();
        let target = ZZbig.power_of_two(log2_to);
        for i in 0..in_rings.len() {
            Q_over_q_mod[i] = int_cast(
                ZZbig.euclidean_rem(
                    ZZbig
                        .checked_div(&Q, &int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64))
                        .unwrap(),
                    &target,
                ),
                ZZi128,
                ZZbig,
            );
        }
        let Q_over_q_downscaled = (0..pad_to_block(in_rings.len()))
            .map(|j| {
                if j < in_rings.len() {
                    int_cast(
                        ZZbig.rounded_div(gamma.clone(), &int_cast(*in_rings.at(j).modulus(), ZZbig, ZZi64)),
                        ZZi128,
                        ZZbig,
                    )
                } else {
                    0
                }
            })
            .collect();
        let q_over_Q = (0..(in_rings.len()))
            .map(|i| {
                in_rings
                    .at(i)
                    .invert(
                        &in_rings.at(i).coerce(
                            &ZZbig,
                            ZZbig
                                .checked_div(&Q, &int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64))
                                .unwrap(),
                        ),
                    )
                    .unwrap()
            })
            .collect();
        let Q_mod_target = int_cast(ZZbig.euclidean_rem(Q, &target), ZZi128, ZZbig);

        Self {
            Q_over_q_mod,
            Q_over_q_downscaled,
            q_over_Q,
            Q_mod_target,
            gamma: ZZi128.power_of_two(gamma_log2),
            allocator: allocator.clone(),
            from_summands: in_rings,
            log2_to,
        }
    }
}

impl<A> BConvPow2<A>
where
    A: Allocator + Clone,
{
    pub fn input_rings<'a>(&'a self) -> &'a [Zn64B] { &self.from_summands }

    /// Performs the (almost) exact RNS base conversion
    /// ```text
    ///   Z/QZ -> Z/Q'Z, x -> smallest_lift(x) + kQ mod Q'
    /// ```
    /// where `k in { -1, 0, 1 }`.
    ///
    /// Furthermore, if the shortest lift of the input is bounded by `Q/4`,
    /// then the result is guaranteed to be exact.
    #[instrument(skip_all)]
    pub fn bconv<V1>(&self, input: Submatrix<V1, El<Zn64B>>, output: &mut [i128])
    where
        V1: AsPointerToSlice<El<Zn64B>>,
    {
        {
            assert_eq!(input.row_count(), self.input_rings().len());
            assert_eq!(input.col_count(), output.len());

            let in_len = input.row_count();
            let col_count = input.col_count();

            let mut lifts = OwnedMatrix::from_fn_in(
                pad_to_block(in_len),
                pad_to_block(col_count),
                |_, _| 0,
                self.allocator.clone(),
            );
            let mut lifts = lifts.data_mut();

            for i in 0..in_len {
                for j in 0..col_count {
                    *lifts.at_mut(i, j) = self.from_summands[i]
                        .any_lift(self.from_summands[i].mul_ref(input.at(i, j), self.q_over_Q.at(i)));
                    debug_assert!(
                        *lifts.at(i, 0) >= 0
                            && *lifts.at(i, 0) as i128
                                <= Zn64B_ANY_LIFT_FACTOR as i128 * *self.from_summands[i].modulus() as i128
                    );
                }
            }

            let mut output_unreduced: Vec<i128> = (0..pad_to_block(col_count)).map(|_| 0).collect::<Vec<_>>();
            assert_eq!(0, self.Q_over_q_mod.len() % BLOCK_SIZE);
            assert_eq!(0, output_unreduced.len() % BLOCK_SIZE);
            assert_eq!(0, lifts.row_count() % BLOCK_SIZE);
            for (lhs_block, rhs_block) in self
                .Q_over_q_downscaled
                .iter()
                .copied()
                .array_chunks::<BLOCK_SIZE>()
                .zip(lifts.as_const().row_iter().array_chunks::<BLOCK_SIZE>())
            {
                vecmatmul_4_i128_to_i128_wrapping(lhs_block, rhs_block, &mut output_unreduced);
            }

            let mut output_correction: Vec<i128> = (0..pad_to_block(col_count)).map(|_| 0).collect();
            assert_eq!(0, self.Q_over_q_downscaled.len() % BLOCK_SIZE);
            for (lhs_block, rhs_block) in self
                .Q_over_q_downscaled
                .iter()
                .copied()
                .array_chunks::<BLOCK_SIZE>()
                .zip(lifts.as_const().row_iter().array_chunks::<BLOCK_SIZE>())
            {
                vecmatmul_4_i128_to_i128(lhs_block, rhs_block, &mut output_correction);
            }

            for j in 0..col_count {
                let mut correction = *output_correction.at(j);
                correction = ZZi128.rounded_div(correction, &self.gamma);
                output[j] = output_unreduced[j].wrapping_sub(correction.wrapping_mul(self.Q_mod_target));
            }
        }
    }
}
