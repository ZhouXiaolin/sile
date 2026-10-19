//! `dropout`, ported from cutile-examples/examples/dropout.rs.
//!
//! The kernel exercises the comparison-and-select pair: `gt_tile` produces a
//! tile of `0.0`/`1.0` and `select` picks between two tiles element by element.
//! The original draws `x` from `randn` and `x_keep` from `rand`, both seeded on
//! the device; this port generates deterministic values on the host instead,
//! because the python reference must see the exact bytes the kernel reads.
//!
//! The kernel body matches the original line for line. `_x` is unused there and
//! is unused here: dropout keeps or zeroes the draws in `x_keep`, so the values
//! in `x` never reach the output.

use sile::{Device, Tensor};
use kernels::dropout;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn dropout<const BM: i64>(
        p: f32,
        _x: &Tensor<f32, { [-1] }>,
        x_keep: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, { [BM] }>,
    ) {
        let zeros = constant(0.0, out.shape());
        let ones = constant(1.0, out.shape());
        let p_tile = p.broadcast(out.shape());
        let x_keep_tile = x_keep.load_like(out);
        // x_keep_tile is the probability of keeping (higher is more likely).
        // p_tile is the probability of dropout (lower keeps more values).
        let out_tile = select(gt_tile(x_keep_tile, p_tile), x_keep_tile, zeros);
        // Rescale outputs based on dropout probability.
        let out_tile = out_tile / (ones - p_tile);
        out.store(out_tile);
    }
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
    let mut out_part = out.partition_mut([BM]);

    dropout::<BM>(p, &x, &keep, &mut out_part).apply(&stream)?;

    let got = out_part.unpartition().to_vec(&stream)?;
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
