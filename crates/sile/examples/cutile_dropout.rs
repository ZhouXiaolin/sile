//! `dropout` written the way cutile-rs writes it.
//!
//! The kernel exercises the comparison-and-select pair: `gt_tile` produces a
//! tile of `0.0`/`1.0` and `select` picks between two tiles element by element.

use sile::ct::*;
use sile::{Device, Tensor};

#[path = "common/mod.rs"]
mod common;

#[sile::kernel]
fn dropout<const BM: i64>(
    p: f32,
    x: &Tensor<f32, { [-1] }>,
    x_keep: &Tensor<f32, { [-1] }>,
    out: &mut Tensor<f32, { [BM] }>,
) {
    let zeros = constant(0.0, out.shape());
    let ones = constant(1.0, out.shape());
    let p_tile = p.broadcast(out.shape());
    let x_keep_tile = x_keep.load_like(out);
    let x_tile = x.load_like(out);
    // `x_keep` is the probability of keeping a value: a draw above `p` keeps it.
    let keep = gt_tile(x_keep_tile, p_tile);
    let out_tile = select(keep, x_tile, zeros);
    // Rescale the kept values so the expected value does not change.
    out.store(out_tile / (ones - p_tile));
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const BM: i64 = 4;
    let len = 16i64;
    let p = 0.4f32;

    let x_data: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
    let keep_data: Vec<f32> = (0..len).map(|i| ((i * 7) % 10) as f32 / 10.0).collect();
    let x = Tensor::from_vec(x_data.clone(), [len], &device)?;
    let keep = Tensor::from_vec(keep_data.clone(), [len], &device)?;
    let mut out = Tensor::zeros([len], &device)?;

    dropout::<BM>(p, &x, &keep, &mut out)
        .grid(((len / BM) as u32, 1, 1))
        .apply(&stream)?;

    let got = out.to_vec(&stream)?;
    let expected = common::numpy("dropout", &[p.to_string()], &[&x_data, &keep_data]);
    for i in 0..len as usize {
        assert!(
            (got[i] - expected[i]).abs() < 1e-5,
            "out[{i}] = {}, expected {}",
            got[i],
            expected[i]
        );
    }
    println!("cutile dropout {len} ok on {device:?}");
    Ok(())
}
