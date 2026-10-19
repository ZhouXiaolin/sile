//! `async_saxpy`, ported from cutile-examples/examples/async_saxpy.rs.
//!
//! The kernel is the same saxpy as [`saxpy`](../saxpy.rs), but written for an
//! element type parameter rather than for `f32`: `T: ElementType` lets one
//! entry serve every float width. The host side matches the original:
//! `arange(32)` reshaped to (4, 8), `a = 2.0`, and the (2, 2) tile whose
//! partition count is the launch grid. What differs is only where the
//! reference comes from: the original asserts against host arithmetic, this
//! one against `numpy_ref.py saxpy`.
//!
//! The original also chains its launch through `cuda_async::DeviceOp`. This
//! port runs one launch, which is the part the kernel text decides.

use sile::{Device, Tensor};
use kernels::saxpy;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn saxpy<const BM: i64, const BN: i64>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        a: f32,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const BM: i64 = 2;
    const BN: i64 = 2;
    let (m, n) = (4i64, 8i64);
    let a = 2.0f32;

    let values: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(values.clone(), [m, n], &device)?;
    let mut y = Tensor::from_vec(values.clone(), [m, n], &device)?;
    let mut y_part = y.partition_mut([BM, BN]);

    let y_host = values.clone();

    saxpy::<BM, BN>(&mut y_part, a, &x).apply(&stream)?;

    let out = y_part.unpartition().to_vec(&stream)?;
    let expected = common::numpy("saxpy", &[a.to_string()], &[&values, &y_host]);
    for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "element {i}: {got}, want {want}"
        );
    }
    println!("async_saxpy {m}x{n} ok on {device:?}");
    Ok(())
}
