//! `async_and_then_example`, ported from
//! cutile-examples/examples/async_and_then_example.rs.
//!
//! The kernel is the saxpy of [`async_saxpy`](../async_saxpy.rs); what the
//! original exercises is the op layer: the kernel is compiled while the inputs
//! are still being fetched, then three launches chain with `.then` and nothing
//! runs until the chain is scheduled. Because the kernel updates `y` in place,
//! three launches apply saxpy three times.
//!
//! The host side matches the original: `arange(32)` reshaped to (4, 8),
//! `a = 2.0`, the (2, 2) tile whose partition count is the launch grid, and
//! three chained launches. What differs is only where the reference comes
//! from: the original asserts against host arithmetic, this one against
//! `numpy_ref.py saxpy` applied three times.

use sile::{Device, DeviceOp, Tensor, op::to_host_vec};
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

/// One launch as an op: it owns `y` and `x`, tiles `y` by (2, 2) the way the
/// original partitions it, and hands both back for the next stage.
fn saxpy_op(
    y: Tensor<f32>,
    a: f32,
    x: Tensor<f32>,
) -> DeviceOp<(Tensor<f32>, Tensor<f32>, f32)> {
    DeviceOp::new(move |stream| {
        const BM: i64 = 2;
        const BN: i64 = 2;
        let mut y = y;
        let mut y_part = y.partition_mut([BM, BN]);
        saxpy::<BM, BN>(&mut y_part, a, &x).apply(stream)?;
        Ok((y, x, a))
    })
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let a = 2.0f32;

    let values: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(values.clone(), [m, n], &device)?;
    let y = Tensor::from_vec(values.clone(), [m, n], &device)?;

    // Three launches, each computing `y = a * x + y` on the y the last one
    // left behind.
    let y_host: Vec<f32> = pollster::block_on(async {
        saxpy_op(y, a, x)
            .then(|(y, x, a)| saxpy_op(y, a, x))
            .then(|(y, x, a)| saxpy_op(y, a, x))
            .then(|(y, _, _)| to_host_vec(y))
            .schedule(&stream)
            .await
    })?;

    // The reference applies the same update three times, so it walks the
    // running value rather than reading `y` once.
    let mut running = values.clone();
    for round in 0..3 {
        let expected = common::numpy("saxpy", &[a.to_string()], &[&values, &running]);
        running = expected;
        if round == 2 {
            for (i, (got, want)) in y_host.iter().zip(&running).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "element {i}: {got}, want {want}"
                );
            }
        }
    }
    println!("async_and_then_example {m}x{n} ok on {device:?}");
    Ok(())
}
