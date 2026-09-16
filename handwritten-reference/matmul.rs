pub const BLOCK_SIZE_LOG2: usize = 2;
pub const BLOCK_SIZE: usize = 1 << BLOCK_SIZE_LOG2;

#[inline]
pub fn matmul_4x4_i64_to_i128(
    lhs: [&[i64; BLOCK_SIZE]; BLOCK_SIZE],
    rhs: [&[i64]; BLOCK_SIZE],
    out: &mut [&mut [i128]; BLOCK_SIZE],
) {
    let len = rhs[0].len();
    for i in 0..BLOCK_SIZE {
        assert_eq!(len, rhs[i].len());
        assert_eq!(len, out[i].len());
    }

    for (l_r, o_r) in lhs.into_iter().zip(out.into_iter()) {
        let o_r = &mut **o_r;
        for j in 0..len {
            let mut value = 0;
            for k in 0..BLOCK_SIZE {
                // safe since k < len
                debug_assert!(j < rhs[k].len());
                value += (unsafe { *rhs[k].get_unchecked(j) } as i128) * (l_r[k] as i128);
            }
            // safe since k < len
            debug_assert!(j < o_r.len());
            *unsafe { o_r.get_unchecked_mut(j) } += value;
        }
    }
}

#[inline]
pub fn vecmatmul_4_i128_to_i128(lhs: [i128; BLOCK_SIZE], rhs: [&[i64]; BLOCK_SIZE], out: &mut [i128]) {
    let len = out.len();
    for i in 0..BLOCK_SIZE {
        assert_eq!(len, rhs[i].len());
    }

    for j in 0..len {
        let mut value = 0;
        for k in 0..BLOCK_SIZE {
            // safe since k < len
            debug_assert!(j < rhs[k].len());
            value += (unsafe { *rhs[k].get_unchecked(j) } as i128) * (lhs[k] as i128);
        }
        // safe since k < len
        debug_assert!(j < out.len());
        *unsafe { out.get_unchecked_mut(j) } += value;
    }
}

#[inline]
pub fn vecmatmul_4_i128_to_i128_wrapping(lhs: [i128; BLOCK_SIZE], rhs: [&[i64]; BLOCK_SIZE], out: &mut [i128]) {
    let len = out.len();
    for i in 0..BLOCK_SIZE {
        assert_eq!(len, rhs[i].len());
    }

    for j in 0..len {
        let mut value = 0;
        for k in 0..BLOCK_SIZE {
            // safe since k < len
            debug_assert!(j < rhs[k].len());
            value += (unsafe { *rhs[k].get_unchecked(j) } as i128).wrapping_mul(lhs[k] as i128);
        }
        // safe since k < len
        debug_assert!(j < out.len());
        unsafe {
            *out.get_unchecked_mut(j) = out.get_unchecked_mut(j).wrapping_add(value);
        }
    }
}

pub fn pad_to_block(len: usize) -> usize {
    if len == 0 {
        0
    } else {
        ((len - 1) / (1 << BLOCK_SIZE_LOG2) + 1) * (1 << BLOCK_SIZE_LOG2)
    }
}
