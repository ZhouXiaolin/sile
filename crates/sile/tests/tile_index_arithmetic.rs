//! A tile index must be any runtime expression, not just `program_id`.
//!
//! cutile-rs writes `part.load([pid_m, k])`, where the index is an arbitrary
//! scalar. This pins that down: the index is arithmetic over runtime values,
//! which is what grid-stride and swizzled tiles need.

use sile::{Device, Tensor, tile};

/// Tile `p` reads tile `p + 1`. Tile indexes are runtime values, so `src` is
/// the raw program id plus one.
#[sile::kernel]
fn shifted_vec_add<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, { [-1] }>,
) {
    // The whole point of the test: `src` is computed, not the raw tile id.
    let src = tile::program_id(0) + 1;
    let x = a.load_tile([TILE], [src.0]);
    let y = b.load_tile([TILE], [src.0]);
    c.store(x + y);
}

#[test]
fn tile_index_accepts_a_computed_runtime_expression() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();
    const TILE: i64 = 4;
    const TILES: i64 = 3;
    let out_len = TILE * TILES;

    // The inputs hold one tile more than the output: tile p reads tile p + 1,
    // so the last program still reads real data instead of running off the end.
    let a = Tensor::from_vec(
        (1..=out_len + TILE).map(|i| i as f32).collect(),
        [out_len + TILE],
        &device,
    )
    .unwrap();
    let b = Tensor::from_vec(vec![10.0; (out_len + TILE) as usize], [out_len + TILE], &device)
        .unwrap();
    let mut c = Tensor::zeros([out_len], &device).unwrap();

    shifted_vec_add::<TILE>(&a, &b, &mut c)
        .grid((TILES as u32, 1, 1))
        .apply(&stream)
        .unwrap();

    // Tile p sums tile (p + 1) of both inputs into slot p. Tile t of `a` starts
    // at t * TILE, and every element gets b's constant ten.
    let mut expected = Vec::new();
    for p in 0..TILES {
        let t = p + 1;
        expected.extend((t * TILE..(t + 1) * TILE).map(|i| i as f32 + 11.0));
    }

    let actual = c.to_vec(&stream).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "element {i}: got {got}, expected {want}; tile index arithmetic is being dropped"
        );
    }
}
