//! `saxpy` written the way cutile-rs writes it.
//!
//! The kernel takes the scalar `a` by value and reads it inside a tile
//! expression, so it exercises the scalar descriptor, the broadcast of a scalar
//! and both load spellings.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn saxpy<const BM: i64, const BN: i64>(
    y: &mut Tensor<f32, { [BM, BN] }>,
    a: f32,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_a = a.broadcast(y.shape());
    let tile_x = x.load_like(y);
    let tile_y = y.load();
    y.store(tile_a * tile_x + tile_y);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const BM: i64 = 2;
    const BN: i64 = 8;
    let (m, n) = (4i64, 8i64);
    let a = 2.0f32;

    let values: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(values.clone(), [m, n], &device)?;
    let mut y = Tensor::from_vec(values.clone(), [m, n], &device)?;

    saxpy::<BM, BN>(&mut y, a, &x)
        .grid(((m / BM) as u32, (n / BN) as u32, 1))
        .apply(&stream)?;

    let out = y.to_vec(&stream)?;
    for (i, got) in out.iter().enumerate().take(10) {
        let expected = a * values[i] + values[i];
        println!("{a} * {} + {} = {got} (expected {expected})", values[i], values[i]);
        assert_eq!(*got, expected, "element {i}");
    }
    println!("cutile saxpy {m}x{n} ok on {:?}", device);
    Ok(())
}
