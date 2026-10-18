//! A two-layer MLP written the way cutile-rs writes it.
//!
//! Three kernels, three launches: a tiled gemm, a matvec that reduces along the
//! contraction axis, and an in-place ReLU. The host compares the result with a
//! plain Rust reference.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn gemm<const BM: i64, const BN: i64, const BK: i64>(
    z: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<f32, { [-1, -1] }>,
    y: &Tensor<f32, { [-1, -1] }>,
) {
    let part_x = x.partition(shape![BM, BK]);
    let part_y = y.partition(shape![BK, BN]);
    let pid = get_tile_block_id();
    let mut tile_z = load_tile_mut(z);
    for i in 0i64..(x.shape()[1] / BK) {
        let tile_x = part_x.load([pid.0, i]);
        let tile_y = part_y.load([i, pid.1]);
        tile_z = mma(tile_x, tile_y, tile_z);
    }
    z.store(tile_z);
}

#[sile::kernel]
fn matvec<const BM: i64, const BK: i64>(
    z: &mut Tensor<f32, { [BM] }>,
    x: &Tensor<f32, { [-1, -1] }>,
    y: &Tensor<f32, { [-1] }>,
) {
    let part_x = x.partition(shape![BM, BK]);
    let pid = get_tile_block_id();
    let mut acc = constant(0.0f32, const_shape![BM]);
    for k in 0i64..(x.shape()[1] / BK) {
        let tile_x = part_x.load([pid.0, k]);
        let tile_y = y.load_tile([BK], [k]);
        let tile_y = tile_y.broadcast(shape![BM, BK]);
        acc = acc + reduce_sum(tile_x * tile_y, 1i64);
    }
    z.store(acc);
}

#[sile::kernel]
fn relu<const D: i64>(z: &mut Tensor<f32, { [D] }>) {
    let zero = constant(0.0f32, const_shape![D]);
    let data = z.load();
    z.store(max_tile(zero, data));
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let dim: i64 = 64;
    const BM_G: i64 = 32;
    const BN_G: i64 = 32;
    const BK_G: i64 = 8;
    const BM_MV: i64 = 32;
    const BK_MV: i64 = 8;

    let w0 = Tensor::random([dim, dim], &device)?;
    let w1 = Tensor::random([dim], &device)?;
    let x = Tensor::random([dim, dim], &device)?;

    let mut out0 = Tensor::zeros([dim, dim], &device)?;
    gemm::<BM_G, BN_G, BK_G>(&mut out0, &x, &w0)
        .grid(((dim / BM_G) as u32, (dim / BN_G) as u32, 1))
        .apply(&stream)?;

    let mut out1 = Tensor::zeros([dim], &device)?;
    matvec::<BM_MV, BK_MV>(&mut out1, &out0, &w1)
        .grid(((dim / BM_MV) as u32, 1, 1))
        .apply(&stream)?;

    relu::<64>(&mut out1).grid((1, 1, 1)).apply(&stream)?;

    let got = out1.to_vec(&stream)?;

    // The reference reads the same host copies, so the comparison is against
    // the inputs the kernels actually saw.
    let w0_host = w0.to_vec(&stream)?;
    let w1_host = w1.to_vec(&stream)?;
    let x_host = x.to_vec(&stream)?;
    let dim = dim as usize;
    let mut mid = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0;
            for k in 0..dim {
                sum += x_host[i * dim + k] * w0_host[k * dim + j];
            }
            mid[i * dim + j] = sum;
        }
    }
    let mut want = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for k in 0..dim {
            sum += mid[i * dim + k] * w1_host[k];
        }
        want[i] = sum.max(0.0);
    }

    let mut max_diff = 0.0f32;
    for i in 0..dim {
        let diff = (got[i] - want[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "Mismatch at index {i}: GPU={}, CPU={}",
            got[i],
            want[i]
        );
    }
    println!("cutile mlp dim={dim} max difference = {max_diff:.2e} on {device:?}");
    Ok(())
}
