//! `batch_matmul`, ported from cutile-examples/examples/batch_matmul.rs.
//!
//! The one thing this example exercises that the others do not is a rank-3
//! buffer. A tile is two-dimensional, so a rank-3 buffer is addressed as a
//! stack of planes: the launch index names a plane, a row and a column, and the
//! tile is the last two of them. CuTile spells that `pid.0` for the batch, and
//! the grid is `(batch, m / BM, n / BN)`. The host sizes are the original's:
//! four (128, 256) outputs over a contraction of 64, accumulated in f32. The
//! one number this port changes is the output tile: the original's 64x64 tile
//! compiles in seconds on the C and Metal targets but never finishes the
//! Metal-side shader compile under WGPU, so the tile here is 32x32.

use sile::{Device, Tensor};
use kernels::batch_matmul;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn batch_matmul<const BM: i64, const BN: i64, const BK: i64, const K: i64>(
        c: &mut Tensor<f32, { [1, BM, BN] }>,
        a: &Tensor<f32, { [-1, -1, K] }>,
        b: &Tensor<f32, { [-1, K, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let batch_idx = pid.0;
        let m_idx = pid.1;
        let n_idx = pid.2;

        let a_part = a.partition(shape![1, BM, BK]);
        let b_part = b.partition(shape![1, BK, BN]);

        let zero: f32 = convert_scalar(0i64);
        let mut acc = broadcast_scalar(zero, shape![BM, BN]);
        for k_idx in 0i64..(K / BK) {
            let a_tile = a_part.load([batch_idx, m_idx, k_idx]).reshape(shape![BM, BK]);
            let b_tile = b_part.load([batch_idx, k_idx, n_idx]).reshape(shape![BK, BN]);
            acc = mma(a_tile, b_tile, acc);
        }
        c.store(acc.reshape(shape![1, BM, BN]));
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let batch = 4i64;
    let (m, n) = (128i64, 256i64);
    const K: i64 = 64;
    const BM: i64 = 32;
    const BN: i64 = 32;
    const BK: i64 = 32;

    let a = Tensor::ones([batch, m, K], &device)?;
    let b = Tensor::ones([batch, K, n], &device)?;
    let mut c = Tensor::zeros([batch, m, n], &device)?;
    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;

    batch_matmul::<BM, BN, BK, K>(&mut c, &a, &b)
        .grid((batch as u32, (m / BM) as u32, (n / BN) as u32))
        .apply(&stream)?;

    let c_host = c.to_vec(&stream)?;
    let expected = common::numpy(
        "batch_matmul",
        &[
            batch.to_string(),
            m.to_string(),
            K.to_string(),
            n.to_string(),
        ],
        &[&a_host, &b_host],
    );
    let (batch, m, n, k) = (batch as usize, m as usize, n as usize, K as usize);
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let got = c_host[(bi * m + i) * n + j];
                let want = expected[(bi * m + i) * n + j];
                assert!(
                    (got - want).abs() <= 1e-3,
                    "mismatch at c[{bi}][{i}][{j}]: got {got}, expected {want}"
                );
            }
        }
    }
    println!("batch_matmul {batch}x{m}x{n}x{k} ok on {device:?}");
    Ok(())
}
