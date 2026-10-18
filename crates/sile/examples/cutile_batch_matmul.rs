//! Batched matrix multiplication written the way cutile-rs writes it.
//!
//! The one thing this example exercises that the others do not is a rank-3
//! buffer. A tile is two-dimensional, so a rank-3 buffer is addressed as a
//! stack of planes: the launch index names a plane, a row and a column, and the
//! tile is the last two of them. CuTile spells that `pid.0` for the batch, and
//! the grid is `(batch, m / BM, n / BN)`.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn batch_matmul<const BM: i64, const BN: i64, const BK: i64>(
    c: &mut Tensor<f32, { [1, BM, BN] }>,
    a: &Tensor<f32, { [-1, -1, -1] }>,
    b: &Tensor<f32, { [-1, -1, -1] }>,
) {
    let pid = get_tile_block_id();
    let batch_idx = pid.0;
    let m_idx = pid.1;
    let n_idx = pid.2;

    let a_part = a.partition(shape![1, BM, BK]);
    let b_part = b.partition(shape![1, BK, BN]);

    let zero: f32 = convert_scalar(0i64);
    let mut acc = broadcast_scalar(zero, shape![BM, BN]);
    let k_tiles: i64 = a.shape()[2] / BK;
    for k_idx in 0i64..k_tiles {
        let a_tile = a_part.load([batch_idx, m_idx, k_idx]).reshape(shape![BM, BK]);
        let b_tile = b_part.load([batch_idx, k_idx, n_idx]).reshape(shape![BK, BN]);
        acc = mma(a_tile, b_tile, acc);
    }
    c.store(acc.reshape(shape![1, BM, BN]));
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let batch = 4i64;
    let (m, n, k) = (8i64, 8i64, 4i64);
    const BM: i64 = 4;
    const BN: i64 = 4;
    const BK: i64 = 2;

    let a = Tensor::random([batch, m, k], &device)?;
    let b = Tensor::random([batch, k, n], &device)?;
    let mut c = Tensor::zeros([batch, m, n], &device)?;

    batch_matmul::<BM, BN, BK>(&mut c, &a, &b)
        .grid((batch as u32, (m / BM) as u32, (n / BN) as u32))
        .apply(&stream)?;

    let c_host = c.to_vec(&stream)?;
    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;
    let (batch, m, n, k) = (batch as usize, m as usize, n as usize, k as usize);
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_host[(bi * m + i) * k + l] * b_host[(bi * k + l) * n + j];
                }
                let got = c_host[(bi * m + i) * n + j];
                assert!(
                    (got - sum).abs() < 1e-3,
                    "mismatch at c[{bi}][{i}][{j}]: got {got}, expected {sum}"
                );
            }
        }
    }
    println!("cutile batch_matmul {batch}x{m}x{n}x{k} ok on {device:?}");
    Ok(())
}
