//! A parallel reduction written the way cutile-rs writes it.
//!
//! The reduction lowers one whole tile to a scalar, and the scalar is added
//! into the output atomically, so the answer is a sum over the launch rather
//! than over one program.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn parallel_reduce<const TILE: i64>(a: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [1] }>) {
    let pid = get_tile_block_id();
    let tile_a = a.load_tile([TILE], [pid.0]);
    let sum_tile = reduce_sum(tile_a, 0i64);
    out.atomic_add(0, sum_tile[0]);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const TILE: i64 = 100;
    const LEN: i64 = 10000;

    let a = Tensor::from_vec((1..=LEN).map(|i| i as f32).collect(), [LEN], &device)?;
    let mut out = Tensor::zeros([1], &device)?;

    parallel_reduce::<TILE>(&a, &mut out)
        .grid(((LEN / TILE) as u32, 1, 1))
        .apply(&stream)?;

    let result = out.to_vec(&stream)?[0];
    let expected = (LEN * (LEN + 1) / 2) as f32;
    // Each tile is summed exactly, then added f32-atomically into one
    // accumulator, so the error grows with the tile count rather than with the
    // data: 100 tiles lands about 32 units short in f32, which is 6.4e-7
    // relative. A 1e-5 relative bound is therefore honest and not a shrug.
    let tolerance = expected * 1e-5;
    println!("sum = {result}, expected = {expected}");
    assert!((result - expected).abs() < tolerance);
    println!("cutile parallel_reduce {LEN} ok on {device:?}");
    Ok(())
}
