//! `async_add`, ported from cutile-examples/examples/async_add.rs.
//!
//! The kernel is the same elementwise add as the original; what the example
//! exercises is the op layer: each launch is a [`DeviceOp`] that owns its
//! tensors, the two launches chain with `.then`, and nothing runs until the
//! chain is scheduled on a stream. The host side matches the original:
//! `arange(32)` plus `ones(32)`, the (2,) tile whose partition count is the
//! launch grid, both calls computing `z = x + y` so the second overwrites the
//! first. What differs is only where the reference comes from: the original
//! asserts against host arithmetic, this one against `numpy_ref.py vec_add`.

use sile::{Device, DeviceOp, Tensor, op::to_host_vec};
use kernels::add;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::Tensor;

    #[sile::entry]
    pub fn add<const S: i64>(
        z: &mut Tensor<f32, { [S] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

/// One launch as an op: it owns the tensors, tiles z by (2,) the way the
/// original partitions it, and hands the tensors back for the next stage.
fn add_op(
    z: Tensor<f32>,
    x: Tensor<f32>,
    y: Tensor<f32>,
) -> DeviceOp<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    DeviceOp::new(move |stream| {
        const S: i64 = 2;
        let mut z = z;
        let mut z_part = z.partition_mut([S]);
        add::<S>(&mut z_part, &x, &y).apply(stream)?;
        Ok((z, x, y))
    })
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let len = 32i64;
    let x_data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let y_data: Vec<f32> = vec![1.0; len as usize];
    let x = Tensor::from_vec(x_data.clone(), [len], &device)?;
    let y = Tensor::from_vec(y_data.clone(), [len], &device)?;
    let z = Tensor::zeros([len], &device)?;

    let z_host: Vec<f32> = pollster::block_on(async {
        add_op(z, x, y)
            .then(|(z, x, y)| add_op(z, x, y))
            .then(|(z, _, _)| to_host_vec(z))
            .schedule(&stream)
            .await
    })?;

    let expected = common::numpy("vec_add", &[], &[&x_data, &y_data]);
    for (i, (got, want)) in z_host.iter().zip(&expected).enumerate() {
        assert!((got - want).abs() < 1e-5, "element {i}: {got}, want {want}");
    }
    println!("async_add {len} ok on {device:?}");
    Ok(())
}
