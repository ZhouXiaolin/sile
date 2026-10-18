//! `gemm_static`, ported from cutile-examples/examples/gemm_static.rs.
//!
//! The static-shape gemm: both inputs name every extent, `[M, K]` and `[K, N]`,
//! so the compiler sees the whole problem. Two things here are not in the other
//! ports. `num_tiles(&part_y, 0)` takes the loop bound from the buffer's
//! partition rather than from a constant, and the body ends with a `continue`.
//! The `continue` is where cutile injects its pipelining, so a port that
//! dropped it would not be the same program.
//!
//! The original scales the problem to `2^10` tiles per axis; this port keeps
//! the same program at a size the C, Metal and WGPU targets all run quickly,
//! and takes the reference from `numpy_ref.py matmul` over the whole output.

use sile::{Device, Tensor};
use kernels::gemm;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn gemm<
        const BM: i64,
        const BN: i64,
        const BK: i64,
        const M: i64,
        const N: i64,
        const K: i64,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [M, K] }>,
        y: &Tensor<f32, { [K, N] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let mut tile_z = load_tile_mut(z);
        let pid: (i64, i64, i64) = get_tile_block_id();
        let nk = num_tiles(&part_y, 0);
        for i in 0i64..nk {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const BM: i64 = 16;
    const BN: i64 = 16;
    const BK: i64 = 8;
    const M: i64 = 64;
    const N: i64 = 64;
    const K: i64 = 32;

    let mut z = Tensor::zeros([M, N], &device)?;
    let x = Tensor::ones([M, K], &device)?;
    let y = Tensor::ones([K, N], &device)?;
    let x_host = x.to_vec(&stream)?;
    let y_host = y.to_vec(&stream)?;
    let mut z_part = z.partition_mut([BM, BN]);

    gemm::<BM, BN, BK, M, N, K>(&mut z_part, &x, &y).apply(&stream)?;

    let z_host = z_part.unpartition().to_vec(&stream)?;
    let expected = common::numpy(
        "matmul",
        &[M.to_string(), K.to_string(), N.to_string()],
        &[&x_host, &y_host],
    );
    for (i, value) in z_host.iter().enumerate() {
        assert!(
            (value - expected[i]).abs() < 1e-3,
            "z[{i}] = {value}, want {}",
            expected[i]
        );
    }
    println!("gemm_static {M}x{N}x{K} ok on {device:?}");
    Ok(())
}
