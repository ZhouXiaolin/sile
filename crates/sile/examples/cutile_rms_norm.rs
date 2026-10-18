//! `rms_norm` written the way cutile-rs writes it.
//!
//! This is the widest single port: it writes a tiled output through
//! `partition_mut` and an indexed store, reads a one-element tile back as a
//! scalar, and takes a square root with an explicit rounding mode.
//!
//! Two things differ from the original. CuTile passes a host-partitioned output
//! to each program, so the program's row is already the base of its `out` and
//! the store names row `0`. Here the whole buffer arrives, so the store names
//! the row it owns. The `i32` annotations of the original are `i64` here,
//! because a kernel's const generics are `i64` in this front end.

use sile::{Device, Tensor};
use kernels::rms_norm;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn rms_norm<const N: i64, const BLOCK_SIZE: i64>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape = shape![1, BLOCK_SIZE];
        let num_tiles: i64 = N / BLOCK_SIZE;
        let pid: (i64, i64, i64) = get_tile_block_id();
        let row = pid.0;

        let x_part = x.partition(tile_shape);
        let mut rms = constant(0.0, tile_shape);
        for j in 0i64..num_tiles {
            let tx = x_part.load([row, j]);
            rms = rms + tx * tx;
        }
        let rms = reduce_sum(rms, 1i64);
        let rms = rms.reshape(shape![]);
        let rms = tile_to_scalar(rms);
        let n = convert_scalar(N);
        let rms = 1.0f32 / (rms / n + eps);
        let rms = sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
        let rms = tile_to_scalar(rms);
        let rms = rms.broadcast(tile_shape);

        let w_part = w.partition(shape![BLOCK_SIZE]);
        let mut out_part = out.partition_mut(tile_shape);
        for j in 0i64..num_tiles {
            let tx = x_part.load([row, j]);
            let tw = w_part.load([j]).reshape(tile_shape);
            let tout = tx * rms * tw;
            out_part.store(tout, [row, j]);
        }
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    const BLOCK_SIZE: i64 = 2;
    let eps = 1e-6f32;

    let x_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let w_data: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
    let x = Tensor::from_vec(x_data.clone(), [m, n], &device)?;
    let w = Tensor::from_vec(w_data.clone(), [n], &device)?;
    let mut out = Tensor::zeros([m, n], &device)?;

    rms_norm::<8, BLOCK_SIZE>(&x, &w, &mut out, eps)
        .grid((m as u32, 1, 1))
        .apply(&stream)?;

    let got = out.to_vec(&stream)?;
    let expected = common::numpy(
        "rms_norm",
        &[m.to_string(), n.to_string(), eps.to_string()],
        &[&x_data, &w_data],
    );
    for (i, (actual, expected)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "out[{i}] = {actual}, expected {expected}"
        );
    }
    println!("cutile rms_norm {m}x{n} ok on {device:?}");
    Ok(())
}
