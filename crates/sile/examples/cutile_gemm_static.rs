//! The static-shape gemm from cutile-rs, written the way cutile-rs writes it.
//!
//! Two things here are not in the other ports. `num_tiles(&part_y, 0)` takes the
//! loop bound from the buffer's partition rather than from a constant, and the
//! body ends with a `continue`. The `continue` is where cutile injects its
//! pipelining, so a port that dropped it would not be the same program.

use sile::{Device, Tensor};
use kernels::gemm;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn gemm<const BM: i64, const BN: i64, const BK: i64>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
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

    const BM: i64 = 32;
    const BN: i64 = 16;
    const BK: i64 = 8;
    const K: i64 = 64;
    let (m, n) = (128i64, 128i64);

    let mut z = Tensor::zeros([m, n], &device)?;
    let x = Tensor::ones([m, K], &device)?;
    let y = Tensor::ones([K, n], &device)?;
    let x_host = x.to_vec(&stream)?;
    let y_host = y.to_vec(&stream)?;

    gemm::<BM, BN, BK>(&mut z, &x, &y)
        .grid(((m / BM) as u32, (n / BN) as u32, 1))
        .apply(&stream)?;

    let z_host = z.to_vec(&stream)?;
    let expected = common::numpy(
        "matmul",
        &[m.to_string(), K.to_string(), n.to_string()],
        &[&x_host, &y_host],
    );
    for (i, value) in z_host.iter().enumerate() {
        assert!(
            (value - expected[i]).abs() < 1e-3,
            "z[{i}] = {value}, want {}",
            expected[i]
        );
    }
    println!("cutile gemm_static {m}x{n}x{K} ok on {device:?}");
    Ok(())
}
