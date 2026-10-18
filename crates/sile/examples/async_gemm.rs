//! `async_gemm`, ported from cutile-examples/examples/async_gemm.rs.
//!
//! The kernel is the original's at the sizes its host passes: `f16` inputs,
//! one loop of `mma` over the contraction axis, the launch grid derived from
//! the partition of `z`. This is the `f16` example: buffers carry `half`, the
//! loads widen them to `f32` tiles, the `mma` accumulates in `f32`, and the
//! store narrows back — no `f16` arithmetic anywhere, which is the original's
//! `f32`-accumulate semantics too. The reference is `numpy_ref.py matmul` over
//! the `f32` images of the quantized inputs.

use sile::{Device, DeviceOp, Tensor, op::to_host_vec};
use kernels::gemm;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;
    use sile::half::f16;

    #[sile::entry]
    pub fn gemm<const BM: i64, const BN: i64, const BK: i64, const K: i64>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f16, { [-1, K] }>,
        y: &Tensor<f16, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i64, i64, i64) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i64..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = sile::mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

/// One launch as an op: it owns the tensors, tiles z by (BM, BN) so the
/// partition count is the launch grid, and hands the tensors back.
fn gemm_op(
    z: Tensor<f32>,
    x: Tensor<half::f16>,
    y: Tensor<half::f16>,
) -> DeviceOp<(Tensor<f32>, Tensor<half::f16>, Tensor<half::f16>)> {
    DeviceOp::new(move |stream| {
        const BM: i64 = 16;
        const BN: i64 = 16;
        const BK: i64 = 8;
        const K: i64 = 16;
        let mut z = z;
        let mut z_part = z.partition_mut([BM, BN]);
        gemm::<BM, BN, BK, K>(&mut z_part, &x, &y).apply(stream)?;
        Ok((z, x, y))
    })
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n, k) = (64i64, 64i64, 16i64);
    let x = Tensor::<half::f16>::random([m, k], &device)?;
    let y = Tensor::<half::f16>::random([k, n], &device)?;
    let z = Tensor::zeros([m, n], &device)?;

    let z_host: Vec<f32> = pollster::block_on(async {
        gemm_op(z, x.clone(), y.clone())
            .then(|(z, _, _)| to_host_vec(z))
            .schedule(&stream)
            .await
    })?;

    let x_host = x.to_f32_vec();
    let y_host = y.to_f32_vec();
    let expected = common::numpy(
        "matmul",
        &[m.to_string(), k.to_string(), n.to_string()],
        &[&x_host, &y_host],
    );
    for (i, (value, want)) in z_host.iter().zip(&expected).enumerate() {
        assert!(
            (value - want).abs() < 1e-3,
            "z[{i}] = {value}, want {want}"
        );
    }
    println!("async_gemm {m}x{n}x{k} f16 ok on {device:?}");
    Ok(())
}
