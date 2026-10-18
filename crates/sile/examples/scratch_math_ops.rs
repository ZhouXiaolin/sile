//! A check of the arithmetic and math surface the CuTile spellings name.
//!
//! Four kernels, one per family: the arithmetic aliases (bit-exact, since
//! they are adds and multiplies), the trigonometric functions, the
//! hyperbolic and rounding functions, and the count operations with the
//! comparison predicate. The transcendental families compare against numpy
//! within a tolerance, because three targets spell them with three libm
//! implementations; the arithmetic and count families must agree exactly.

mod common;

use sile::ct::*;
use sile::{Device, Result, Tensor};

const TILE: i64 = 16;
const LEN: i64 = 32;

#[sile::kernel]
fn math_arith(
    x: &Tensor<f32, { [32] }>,
    y: &Tensor<f32, { [32] }>,
    out: &mut Tensor<f32, { [16] }>,
) {
    let tx = load_tile(x, [16], [0]);
    let ty = y.load_tile([16], [0]);
    let a = addf(tx, ty);
    let b = subf(a, ty);
    let c = mulf(b, b);
    let d = divf(c, ty);
    let e = fma(d, ty, tx);
    let two = constant(2.0f32, shape![16]);
    let f = maxf(e, minf(ty, two));
    let g = absf(subf(f, constant(1.5f32, shape![16])));
    store_tile(out, g);
}

#[sile::kernel]
fn math_trig(
    x: &Tensor<f32, { [32] }>,
    y: &Tensor<f32, { [32] }>,
    out: &mut Tensor<f32, { [16] }>,
) {
    let tx = x.load_tile([16], [0]);
    let ty = y.load_tile([16], [0]);
    let s = sin(tx);
    let c = cos(tx);
    let t = tan(tx);
    let p = pow(tx, 3.0);
    let a = atan2(tx, ty);
    out.store(s + c + t + p + a);
}

#[sile::kernel]
fn math_hyp(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [16] }>) {
    let tx = x.load_tile([16], [0]);
    let h = sinh(tx) + cosh(tx);
    let th = tanh_with_rounding(tx);
    let r = exp_with_rounding(tx);
    let up = ceil(tx);
    let dn = floor(tx);
    let id = ftof(tx);
    out.store(h + th + r + up + dn + id);
}

#[sile::kernel]
fn math_int(
    x: &Tensor<f32, { [32] }>,
    y: &Tensor<f32, { [32] }>,
    out: &mut Tensor<f32, { [16] }>,
) {
    let grid_x = get_num_tile_blocks().0;
    let pid = get_tile_block_id().0;
    let base = muli(muli(pid, grid_x), 16);
    let shifted = addi(base, 3);
    let back = absi(subi(shifted, 20));
    let clamped = maxi(mini(back, 8), 0);
    let doubled = shri(shli(clamped, 4), 7);
    let tx = load_tile(x, [16], [doubled]);
    let ty = y.load_tile([16], [0]);
    let m = cmpf(tx, ty, predicate::GreaterThanOrEqual);
    let picked = select(m, tx, ty);
    out.store(picked);
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

fn compare_close(got: &[f32], expected: &[f32], tol: f32, name: &str) {
    let worst = got
        .iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= tol, "{name} differs from numpy by {worst}");
}

fn main() -> Result<()> {
    let (device, stream) = device()?;
    let x_data: Vec<f32> = (0..LEN)
        .map(|i| 0.5f32 + (i as f32) * 0.0625)
        .collect();
    let y_data: Vec<f32> = (0..LEN)
        .map(|i| 1.0f32 + (i as f32) * 0.0625)
        .collect();
    let x = Tensor::from_vec(x_data.clone(), [LEN], &device)?;
    let y = Tensor::from_vec(y_data.clone(), [LEN], &device)?;

    let mut out = Tensor::zeros([TILE], &device)?;
    math_arith(&x, &y, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("math_arith", &[], &[&x_data, &y_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "math_arith");

    let mut out = Tensor::zeros([TILE], &device)?;
    math_trig(&x, &y, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("math_trig", &[], &[&x_data, &y_data]);
    compare_close(&out.to_vec(&stream)?, &expected, 1e-5, "math_trig");

    let mut out = Tensor::zeros([TILE], &device)?;
    math_hyp(&x, &mut out).grid((1, 1, 1)).apply(&stream)?;
    let expected = common::numpy("math_hyp", &[], &[&x_data]);
    compare_close(&out.to_vec(&stream)?, &expected, 1e-4, "math_hyp");

    let mut out = Tensor::zeros([TILE], &device)?;
    math_int(&x, &y, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy(
        "math_int",
        &[TILE.to_string()],
        &[&x_data, &y_data],
    );
    compare_exact(&out.to_vec(&stream)?, &expected, "math_int");

    println!(
        "scratch math ops {TILE}x4 kernels ok on {:?}",
        Device::default()?
    );
    Ok(())
}
