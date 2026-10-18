//! `saxpy` with the shape written as a const array parameter.
//!
//! cutile-rs writes `const S: [i32; 2]` and names it in the parameter type:
//! `y: &mut Tensor<f32, S>`. Stable Rust has no const generic of array type, so
//! the macro expands the parameter into one integer per dimension and the call
//! site passes the extents.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn saxpy<const S: [i64; 2]>(
    y: &mut Tensor<f32, S>,
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

    let (m, n) = (4i64, 8i64);
    let a = 2.0f32;
    let values: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(values.clone(), [m, n], &device)?;
    let mut y = Tensor::from_vec(values.clone(), [m, n], &device)?;

    // `const S: [i64; 2]` becomes `S_0` and `S_1`, so the call passes the two
    // extents the tensor has.
    saxpy::<2, 8>(&mut y, a, &x)
        .grid(((m / 2) as u32, (n / 8) as u32, 1))
        .apply(&stream)?;

    let out = y.to_vec(&stream)?;
    for (i, got) in out.iter().enumerate().take(10) {
        let expected = a * values[i] + values[i];
        println!("{a} * {} + {} = {got} (expected {expected})", values[i], values[i]);
        assert_eq!(*got, expected, "element {i}");
    }
    println!("cutile saxpy with const shape {m}x{n} ok on {device:?}");
    Ok(())
}
