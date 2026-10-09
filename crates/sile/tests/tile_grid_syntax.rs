//! The CuTile-aligned grid vocabulary in kernel bodies.
//!
//! `program_id(axis)` is the canonical spelling; `id()` stays as a deprecated
//! alias so existing kernels keep compiling.

use sile::{Device, Tensor, tile};

#[sile::kernel]
fn add_via_program_id<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, { [-1] }>,
) {
    let pid = tile::program_id(0).0;
    let x = a.load_tile([TILE], [pid]);
    let y = b.load_tile([TILE], [pid]);
    c.store(x + y);
}

#[sile::kernel]
fn add_via_id_alias<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, { [-1] }>,
) {
    let pid = tile::id().0;
    let x = a.load_tile([TILE], [pid]);
    let y = b.load_tile([TILE], [pid]);
    c.store(x + y);
}

/// Both spellings must reach the same Tile IR lowering.
fn assert_add_matches<const TILE: i64>(len: i64) {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();
    let a = Tensor::from_vec((1..=len).map(|i| i as f32).collect(), [len], &device).unwrap();
    let b = Tensor::from_vec(vec![10.0; len as usize], [len], &device).unwrap();
    let expected: Vec<f32> = (1..=len).map(|i| i as f32 + 10.0).collect();

    let mut new = Tensor::zeros([len], &device).unwrap();
    add_via_program_id::<TILE>(&a, &b, &mut new)
        .grid(((len / TILE) as u32, 1, 1))
        .apply(&stream)
        .unwrap();
    assert_eq!(new.to_vec(&stream).unwrap(), expected);

    let mut old = Tensor::zeros([len], &device).unwrap();
    add_via_id_alias::<TILE>(&a, &b, &mut old)
        .grid(((len / TILE) as u32, 1, 1))
        .apply(&stream)
        .unwrap();
    assert_eq!(old.to_vec(&stream).unwrap(), expected);
}

#[test]
fn program_id_and_id_produce_the_same_result() {
    assert_add_matches::<4>(16);
}
