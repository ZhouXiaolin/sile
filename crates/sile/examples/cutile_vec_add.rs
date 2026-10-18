//! Element-wise addition written the way cutile-rs writes it.
//!
//! `program_id(axis)` is one scalar, which is what CuTile's spelling returns.

use sile::ct::*;
use sile::{Device, Tensor};

#[path = "common/mod.rs"]
mod common;

#[sile::kernel]
fn vec_add<const TILE: i64>(
    c: &mut Tensor<f32, { [TILE] }>,
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
) {
    let pid = program_id(0);
    let tile_a = a.load_tile([TILE], [pid]);
    let tile_b = b.load_tile([TILE], [pid]);
    c.store(tile_a + tile_b);
}

fn main() -> sile::Result<()> {
    const TILE: i64 = 2;
    let len = 16i64;
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let a = Tensor::from_vec((1..=len).map(|i| i as f32).collect(), [len], &device)?;
    let b = Tensor::from_vec(vec![2.0; len as usize], [len], &device)?;
    let mut c = Tensor::zeros([len], &device)?;

    vec_add::<TILE>(&mut c, &a, &b)
        .grid(((len / TILE) as u32, 1, 1))
        .apply(&stream)?;

    let got = c.to_vec(&stream)?;
    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;
    let expected = common::numpy("vec_add", &[], &[&a_host, &b_host]);
    for (i, value) in got.iter().enumerate() {
        assert_eq!(*value, expected[i], "element {i}");
    }
    println!("cutile vec_add {len} ok on {device:?}");
    Ok(())
}
