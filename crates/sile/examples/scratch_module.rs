//! A check of the module surface: `#[sile::module]` with device functions the
//! entries call, transitive calls, an entry with every `#[sile::entry(...)]`
//! key spelled, and a device function that feeds two entries. Each entry
//! writes one result, so the three targets must agree exactly with numpy.

mod common;

use sile::{Device, Result, Tensor};
use scratch_module_kernels::{
    keyed_entry, plain_entry, reassign_entry, softmax_entry, transitive_entry,
};

#[sile::module]
mod scratch_module_kernels {
    use sile::ct::*;
    use sile::Tensor;

    /// A device function: no attribute, so its calls expand at the call site.
    fn scaled_row(x: &Tensor<f32, { [8] }>, factor: f32) -> Tile<f32> {
        let t = x.load();
        t + broadcast_scalar(factor, shape![8])
    }

    /// A device function that calls another device function.
    fn doubled(x: &Tensor<f32, { [8] }>, factor: f32) -> Tile<f32> {
        let once = scaled_row(x, factor);
        once + once
    }

    /// A bare `#[sile::entry]`: the inlined call reads one scaled row.
    #[sile::entry]
    pub fn plain_entry(x: &Tensor<f32, { [8] }>, out: &mut Tensor<f32, { [8] }>, factor: f32) {
        let t = scaled_row(x, factor);
        out.store(t);
    }

    /// A transitive device call: `doubled` calls `scaled_row` calls the tile
    /// ops, and every level expands before the kernel pipeline sees it.
    #[sile::entry]
    pub fn transitive_entry(x: &Tensor<f32, { [8] }>, out: &mut Tensor<f32, { [8] }>, factor: f32) {
        let t = doubled(x, factor);
        out.store(t);
    }

    /// One device function feeding two entries: each inlining is independent.
    #[sile::entry(
        print_ir = false,
        dump_mlir_dir = "",
        unchecked_accesses = false,
        deny_in_kernel_checks = false,
        preconditions = (dim(x, 0) == dim(out, 0)),
        optimization_hints = ()
    )]
    pub fn keyed_entry(x: &Tensor<f32, { [8] }>, out: &mut Tensor<f32, { [8] }>, factor: f32) {
        let t = scaled_row(x, factor * 2.0);
        out.store(t);
    }

    /// A device function the body reassigns its parameter in: the fresh
    /// binding keeps the caller's variable untouched.
    // The host keeps the loop's body and drops the condition, so the count
    // and the counter move only on the device: the two warns are that drop.
    #[allow(unused_variables, unused_assignments)]
    fn accumulate(t: Tile<f32>, times: f32) -> Tile<f32> {
        let mut acc = t;
        let mut n = 0.0_f32;
        while n < times {
            acc = acc + t;
            n = n + 1.0_f32;
        }
        acc
    }

    #[sile::entry]
    pub fn reassign_entry(x: &Tensor<f32, { [8] }>, out: &mut Tensor<f32, { [8] }>) {
        let base = broadcast_scalar(1.0, shape![8]);
        let t = accumulate(base, 3.0);
        out.store(t);
    }

    /// The softmax a `#[sile::kernel]` writes, as a module: the two helpers
    /// are device functions, and the entry reads like the cutile spelling.
    fn shifted_exp(x: Tile<f32>, peak: Tile<f32>) -> Tile<f32> {
        exp(x - peak)
    }

    fn row_sum(t: Tile<f32>) -> Tile<f32> {
        reduce_sum(t, 1i64)
    }

    #[sile::entry]
    pub fn softmax_entry(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 8] }>) {
        let tile_x = x.load();
        let peak = reduce_max(tile_x.clone(), 1i64);
        let peak = peak.reshape(shape![4, 1]).broadcast(out.shape());
        let num = shifted_exp(tile_x, peak);
        let denom = row_sum(num.clone());
        let denom = denom.reshape(shape![4, 1]).broadcast(out.shape());
        out.store(num / denom);
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
    inputs: &[&[f32]],
) -> Result<()> {
    apply(out, stream)?;
    let expected = common::numpy(name, &[], inputs);
    compare_exact(&out.to_vec(stream)?, &expected, name);
    println!("{name} ok");
    Ok(())
}

fn main() -> Result<()> {
    let (device, stream) = device()?;
    let x_data: Vec<f32> = (0..8).map(|i| 0.25f32 + i as f32 * 0.5).collect();
    let x = Tensor::from_vec(x_data.clone(), [8], &device)?;
    let factor = 1.5f32;

    let mut out = Tensor::zeros([8], &device)?;
    run(
        "mod_plain_entry",
        &mut out,
        |out, s| plain_entry(&x, out, factor).grid((1, 1, 1)).apply(s),
        &stream,
        &[&x_data],
    )?;

    let mut out = Tensor::zeros([8], &device)?;
    run(
        "mod_transitive_entry",
        &mut out,
        |out, s| transitive_entry(&x, out, factor).grid((1, 1, 1)).apply(s),
        &stream,
        &[&x_data],
    )?;

    let mut out = Tensor::zeros([8], &device)?;
    run(
        "mod_keyed_entry",
        &mut out,
        |out, s| keyed_entry(&x, out, factor).grid((1, 1, 1)).apply(s),
        &stream,
        &[&x_data],
    )?;

    let mut out = Tensor::zeros([8], &device)?;
    run(
        "mod_reassign_entry",
        &mut out,
        |out, s| reassign_entry(&x, out).grid((1, 1, 1)).apply(s),
        &stream,
        &[&x_data],
    )?;

    let x2_data: Vec<f32> = (0..32).map(|i| 1.0f32 + (i as f32 % 7.0) * 0.25).collect();
    let x2 = Tensor::from_vec(x2_data.clone(), [4, 8], &device)?;
    let mut out = Tensor::zeros([4, 8], &device)?;
    softmax_entry(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    // The reference the softmax example names is float64 numpy with a
    // tolerance: the fold crosses several transcendental steps, so the exact
    // agreement this file checks elsewhere is a three-target check, not a
    // numpy one.
    let expected = common::numpy(
        "softmax",
        &["4".to_string(), "8".to_string()],
        &[&x2_data],
    );
    let worst = out
        .to_vec(&stream)?
        .iter()
        .zip(&expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= 1e-5, "mod softmax differs from numpy by {worst}");
    println!("mod_softmax ok");

    Ok(())
}
