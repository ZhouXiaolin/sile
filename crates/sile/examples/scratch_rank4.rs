//! A rank-4 buffer addressed through a rank-4 partition.
//!
//! A tile is two-dimensional, so every axis above the last two is a plane of
//! planes. A rank-4 buffer therefore needs two outer coordinates, and a load
//! that names only one of them lands in the wrong plane. This example names
//! both, on the way in and on the way out, and the host checks every element:
//! a wrong plane moves a value, and the check reads the moved value.
//!
//! The grid is `(b, h, m / BM)`, which is the grid CuTile's flash attention
//! uses, and the tile covers the whole of the last axis, so its column
//! coordinate is zero.

use sile::{Device, Tensor};
use kernels::scale_rank4;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn scale_rank4<const BM: i64, const D: i64>(
        out: &mut Tensor<f32, { [1, 1, BM, D] }>,
        x: &Tensor<f32, { [-1, -1, -1, -1] }>,
        factor: f32,
    ) {
        let pid = get_tile_block_id();
        let part = x.partition(shape![1, 1, BM, D]);
        let tile = part.load([pid.0, pid.1, pid.2, 0i64]).reshape(shape![BM, D]);
        let scale = factor.broadcast(shape![BM, D]);
        out.store((tile * scale).reshape(shape![1, 1, BM, D]));
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (b, h, m, d) = (2i64, 3i64, 4i64, 8i64);
    const BM: i64 = 2;
    const D: i64 = 8;
    let factor = 3.0f32;

    let x = Tensor::random([b, h, m, d], &device)?;
    let mut out = Tensor::zeros([b, h, m, d], &device)?;

    // The output declares the shape of one tile, and the buffer it writes is
    // the whole tensor, so the program names the plane it owns.
    scale_rank4::<BM, D>(&mut out, &x, factor)
        .grid((b as u32, h as u32, (m / BM) as u32))
        .apply(&stream)?;

    let got = out.to_vec(&stream)?;
    let want = x.to_vec(&stream)?;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let expected = w * factor;
        assert!(
            (*g - expected).abs() < 1e-6,
            "element {i}: got {g}, expected {expected} (from {w})"
        );
    }
    println!("rank-4 scale {b}x{h}x{m}x{d} ok on {device:?}");
    Ok(())
}
