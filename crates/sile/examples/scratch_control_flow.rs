//! A check of the control flow the CuTile surface names: `while`, `loop`,
//! `break`, a `continue` a branch holds, `step_by` over a range, and an early
//! `return`. Each kernel folds its loops into one tile the store writes, so
//! the three targets must agree exactly on the arithmetic the loop order
//! produces.

mod common;

use sile::{Device, Result, Tensor};
use kernels::{
    early_return, for_continue, for_steps, loop_break, nested_loops, while_acc, while_skip,
};

const TILE: i64 = 16;
const LEN: i64 = 32;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    /// A `while` that carries a scalar: the sum of the counts it walks through.
    #[sile::entry]
    pub fn while_acc(out: &mut Tensor<f32, { [16] }>) {
        let mut n = 5.0;
        let mut acc = 0.0;
        let mut i = 0.0;
        while i < n {
            acc = acc + i;
            i = i + 1.0;
        }
        let t = broadcast_scalar(acc, shape![16]);
        out.store(t);
    }

    /// A `continue` a branch holds: the passes past three contribute nothing, and
    /// the pass variable still moves, or the loop would never end.
    #[sile::entry]
    pub fn while_skip(out: &mut Tensor<f32, { [16] }>) {
        let mut acc = broadcast_scalar(0.0, shape![16]);
        let mut i = 0.0;
        let mut n = 6.0;
        while i < n {
            if i > 3.0 {
                i = i + 1.0;
                continue;
            }
            acc = acc + broadcast_scalar(i, shape![16]);
            i = i + 1.0;
        }
        out.store(acc);
    }

    /// A `loop` with a `break` at its end: the sum of 1..=4.
    #[sile::entry]
    pub fn loop_break(out: &mut Tensor<f32, { [16] }>) {
        let mut acc = broadcast_scalar(0.0, shape![16]);
        let mut i = 0.0;
        loop {
            acc = acc + broadcast_scalar(i, shape![16]);
            i = i + 1.0;
            if i >= 4.0 {
                break;
            }
        }
        out.store(acc);
    }

    /// A `for` in its three lowering shapes: a small stepped range unrolls, a
    /// range over a runtime bound lowers to a loop, and a stepped range too long
    /// to unroll lowers to a loop with the step in its back edge. The sum is
    /// 0+3+6, plus 0..=6, plus 7×(0+1+…+36).
    #[sile::entry]
    pub fn for_steps(out: &mut Tensor<f32, { [16] }>) {
        let mut acc = broadcast_scalar(0.0, shape![16]);
        for i in (0..8).step_by(3) {
            acc = acc + broadcast_scalar(i, shape![16]);
        }
        let mut n = 7;
        for j in 0..n {
            acc = acc + broadcast_scalar(j, shape![16]);
        }
        for k in (0..256).step_by(7) {
            acc = acc + broadcast_scalar(k, shape![16]);
        }
        out.store(acc);
    }

    /// An early `return` in a branch. One block runs, so the branch is never
    /// taken and the second store runs; the lowered control flow still has to
    /// hold both paths apart.
    #[sile::entry]
    pub fn early_return(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [16] }>) {
        let pid = get_tile_block_id().0;
        let t0 = x.load_tile([16], [0]);
        out.store(t0);
        if pid > 0 {
            return;
        }
        let t1 = x.load_tile([16], [1]);
        out.store(t1 + broadcast_scalar(1.0, shape![16]));
    }

    /// Loops nested in both directions: a `while` whose body holds a `for`, both
    /// carrying the same accumulator, and a branch the `while` carries past.
    #[sile::entry]
    pub fn nested_loops(out: &mut Tensor<f32, { [16] }>) {
        let mut acc = broadcast_scalar(0.0, shape![16]);
        let mut r = 0.0;
        let mut rounds = 3.0;
        while r < rounds {
            for j in 0..4 {
                acc = acc + broadcast_scalar(j, shape![16]);
            }
            r = r + 1.0;
            if r >= 2.0 {
                acc = acc + broadcast_scalar(100.0, shape![16]);
            }
        }
        out.store(acc);
    }

    /// A `continue` a branch holds inside a `for`: one pass out of eight adds
    /// nothing, and the loop still walks every value.
    #[sile::entry]
    pub fn for_continue(out: &mut Tensor<f32, { [16] }>) {
        let mut acc = broadcast_scalar(0.0, shape![16]);
        let mut n = 8;
        for i in 0..n {
            if i == 3 {
                continue;
            }
            acc = acc + broadcast_scalar(i, shape![16]);
        }
        out.store(acc);
    }
}

fn device() -> Result<(Device, sile::Stream)> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    Ok((device, stream))
}

fn compare_exact(got: &[f32], expected: &[f32], name: &str) {
    for (i, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(got, expected, "{name} element {i}");
    }
}

fn run(
    name: &str,
    out: &mut Tensor<f32>,
    apply: impl FnOnce(&mut Tensor<f32>, &sile::Stream) -> Result<()>,
    stream: &sile::Stream,
    x_data: &[f32],
) -> Result<()> {
    apply(out, stream)?;
    let expected = common::numpy(name, &[], &[x_data]);
    compare_exact(&out.to_vec(stream)?, &expected, name);
    println!("{name} ok");
    Ok(())
}

fn main() -> Result<()> {
    let (device, stream) = device()?;
    let x_data: Vec<f32> = (0..LEN)
        .map(|i| 0.5f32 + (i as f32) * 0.0625)
        .collect();
    let x = Tensor::from_vec(x_data.clone(), [LEN], &device)?;

    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_while_acc",
        &mut out,
        |out, s| while_acc(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_while_skip",
        &mut out,
        |out, s| while_skip(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_loop_break",
        &mut out,
        |out, s| loop_break(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_for_steps",
        &mut out,
        |out, s| for_steps(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_early_return",
        &mut out,
        |out, s| early_return(&x, out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_nested_loops",
        &mut out,
        |out, s| nested_loops(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;
    let mut out = Tensor::zeros([TILE], &device)?;
    run(
        "cf_for_continue",
        &mut out,
        |out, s| for_continue(out).grid((1, 1, 1)).apply(s),
        &stream,
        &x_data,
    )?;

    Ok(())
}
