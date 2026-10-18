//! `saxpy`, ported from cutile-examples/examples/saxpy.rs.
//!
//! The kernel takes the scalar `a` by value and reads it inside a tile
//! expression, so it exercises the scalar descriptor, the broadcast of a scalar
//! and both load spellings. The host side matches the original: `arange(32)`
//! reshaped to (4, 8), `a = 2.0`, and the (2, 2) tile that puts the launch on a
//! (2, 4) grid. What differs is only where the reference comes from: the
//! original asserts against host arithmetic, this one against
//! `numpy_ref.py saxpy`.

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

    // The kernel updates y in place, so the reference reads y as it is here,
    // before the launch.
    let y_host = values.clone();

    saxpy::<BM, BN>(&mut y, a, &x)
        .grid(((m / BM) as u32, (n / BN) as u32, 1))
        .apply(&stream)?;

    let out = y.to_vec(&stream)?;
    let expected = common::numpy("saxpy", &[a.to_string()], &[&values, &y_host]);
    for (i, (got, expected)) in out.iter().zip(&expected).enumerate() {
        let answer = a * values[i] + y_host[i];
        println!("{a} * {} + {} = {got}", values[i], y_host[i]);
        assert!(
            (got - expected).abs() < 1e-5,
            "element {i}: {got}, want {expected} (host {answer})"
        );
    }
    println!("saxpy {m}x{n} ok on {device:?}");
    Ok(())
}
