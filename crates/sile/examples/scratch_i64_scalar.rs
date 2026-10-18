//! An integer scalar argument, used to divide a launch index.
//!
//! CuTile writes `query_group_size: i32` and divides a query head index by it.
//! A float division would round up to the same index and read the wrong row, so
//! the host checks a value the two divisions disagree on: with a group of two,
//! the integer division of row three is one and its product with the group is
//! two, while a float division would give three back.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn pick_row<const N: i64>(
    out: &mut Tensor<f32, { [1, N] }>,
    x: &Tensor<f32, { [-1, -1] }>,
    group: i64,
) {
    let pid = get_tile_block_id();
    // The group index of this row, and the row its group starts at.
    let group_of_row: i64 = pid.0 / group;
    let in_group: i64 = pid.0 % group;
    let row: i64 = group_of_row * group + in_group;

    let part = x.partition(shape![1, N]);
    out.store(part.load([row, 0i64]));
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (rows, n) = (4i64, 8i64);
    const N: i64 = 8;
    let group = 2i64;

    let x = Tensor::random([rows, n], &device)?;
    let mut out = Tensor::zeros([rows, n], &device)?;

    pick_row::<N>(&mut out, &x, group)
        .grid((rows as u32, 1, 1))
        .apply(&stream)?;

    let got = out.to_vec(&stream)?;
    let want = x.to_vec(&stream)?;
    // `(row / group) * group + row % group` is `row` when both are integers,
    // and the truncating step is what a float division would lose.
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "element {i}: got {g}, expected {w} (row {})",
            i / n as usize
        );
    }
    println!("integer scalar argument with group {group} ok on {device:?}");
    Ok(())
}
