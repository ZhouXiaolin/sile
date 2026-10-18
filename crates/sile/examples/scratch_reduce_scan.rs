//! A check of the folds the CuTile surface names: `reduce_prod`, and the
//! scans — `scan_sum` forward and reverse, a `scan` with a product combiner,
//! and a two-axis scan. Each kernel writes one fold's result, so the three
//! targets must agree exactly with the numpy order the fold walks.

mod common;

use sile::ct::*;
use sile::{Device, Result, Tensor};

/// The product of one 1D tile.
#[sile::kernel]
fn reduce_prod_1d(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [1] }>) {
    let t = x.load();
    let p = reduce_prod(t, 1i64);
    out.store(p);
}

/// The product down the columns of one 2D tile: four row products.
#[sile::kernel]
fn reduce_prod_2d(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4] }>) {
    let t = x.load();
    let p = reduce_prod(t, 1i64);
    out.store(p.reshape(shape![1, 4]));
}

/// The running sum forward: numpy's `cumsum` with a zero start.
#[sile::kernel]
fn scan_sum_fwd(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [32] }>) {
    let t = x.load_like(out);
    let s = scan_sum(t, 1i64, reverse::Forward, 0.0_f32);
    out.store(s);
}

/// The running sum from the back: the mirror of numpy's `cumsum`.
#[sile::kernel]
fn scan_sum_rev(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [32] }>) {
    let t = x.load_like(out);
    let s = scan_sum(t, 1i64, reverse::Reverse, 0.0_f32);
    out.store(s);
}

/// A `scan` whose combiner is a multiply: numpy's `cumprod`.
#[sile::kernel]
fn scan_prod_fwd(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [32] }>) {
    let t = x.load_like(out);
    let s = scan(t, 1i64, reverse::Forward, 1.0_f32, |a, b| a * b);
    out.store(s);
}

/// `reduce` with a closure: the same running fold, spelled the way CuTile
/// names it.
#[sile::kernel]
fn reduce_closure(x: &Tensor<f32, { [32] }>, out: &mut Tensor<f32, { [32] }>) {
    let t = x.load_like(out);
    let s = reduce(t, 1i64, 0.0_f32, |a, b| a + b);
    out.store(s);
}

/// The float power alias, spelled the CuTile way. The exponent is one, so
/// the value the check reads is the identity each target computes exactly —
/// this kernel proves the alias lowers to the power op, which the math
/// scratch checks numerically on every target.
#[sile::kernel]
fn alias_lanes(a: &Tensor<f32, { [8] }>, b: &Tensor<f32, { [8] }>, out: &mut Tensor<f32, { [8] }>) {
    let ta = a.load();
    let p = fpowf(ta, broadcast_scalar(1.0, shape![8]));
    out.store(p);
}

/// The running sum down the columns of a 2D tile, one row at a time.
#[sile::kernel]
fn scan_2d_axis1(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 8] }>) {
    let t = x.load_like(out);
    let s = scan_sum(t, 1i64, reverse::Forward, 0.0_f32);
    out.store(s);
}

/// The running sum across the rows of a 2D tile, one column at a time.
#[sile::kernel]
fn scan_2d_axis0(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 8] }>) {
    let t = x.load_like(out);
    let s = scan_sum(t, 0i64, reverse::Forward, 0.0_f32);
    out.store(s);
}

/// The two tiles joined along the column axis: eight columns from each side.
#[sile::kernel]
fn cat_cols(x: &Tensor<f32, { [4, 8] }>, y: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 16] }>) {
    let a = x.load();
    let b = y.load();
    out.store(cat(a, b, 1i64));
}

/// The two tiles joined along the row axis.
#[sile::kernel]
fn cat_rows(x: &Tensor<f32, { [4, 8] }>, y: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [8, 8] }>) {
    let a = x.load();
    let b = y.load();
    out.store(cat(a, b, 0i64));
}

/// The 2x4 subtile at block index (1, 1) of a 4x8 source.
#[sile::kernel]
fn extract_block(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [2, 4] }>) {
    let t = x.load();
    let i = 1;
    let j = 1;
    let sub = extract(t, [i, j], shape![2, 4]);
    out.store(sub);
}

/// A 4x8 destination with a 2x4 subtile written at block index (1, 1).
#[sile::kernel]
fn insert_block(x: &Tensor<f32, { [4, 8] }>, sub: &Tensor<f32, { [2, 4] }>, out: &mut Tensor<f32, { [4, 8] }>) {
    let t = x.load();
    let s = sub.load();
    let i = 1;
    let j = 1;
    out.store(insert(s, t, [i, j]));
}

/// The transpose spelling: `permute(t, [1, 0])` is `transpose(t)`.
#[sile::kernel]
fn permute_transpose(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [8, 4] }>) {
    let t = x.load();
    out.store(permute(t, [1, 0]));
}

/// The identity permutation keeps the tile as it is.
#[sile::kernel]
fn permute_identity(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 8] }>) {
    let t = x.load();
    out.store(permute(t, [0, 1]));
}

/// The assume family, one link per spelling: a promise about JIT-constant
/// predicates, and this lowering relies on none of it, so the lanes the
/// check reads are the source's own.
#[sile::kernel]
fn assume_lanes(x: &Tensor<f32, { [4, 8] }>, out: &mut Tensor<f32, { [4, 8] }>) {
    let t = x.load();
    let a = unsafe { assume_div_by::<_, 8>(t) };
    let b = unsafe { assume_div_by_raw(a, 8u64, None, None) };
    let c = unsafe { assume_div_by_every_along::<_, 2, 4, 1>(b) };
    let d = unsafe { assume_bounds::<_, 0, 32>(c) };
    let e = unsafe { assume_bounds_lower::<_, 0>(d) };
    let f = unsafe { assume_bounds_upper::<_, 32>(e) };
    let g = unsafe { assume_bounds_raw(f, Some(0), Some(32)) };
    let h = unsafe { assume_same_elements_1d::<_, 8>(g) };
    let i = unsafe { assume_same_elements_2d::<_, 2, 4>(h) };
    let j = unsafe { assume_same_elements_3d::<_, 2, 2, 2>(i) };
    let k = unsafe { assume_same_elements_4d::<_, 2, 2, 2, 1>(j) };
    let m = unsafe { assume_same_elements_raw(k, [2i64, 2i64]) };
    out.store(m);
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

fn main() -> Result<()> {
    let (device, stream) = device()?;
    let x_data: Vec<f32> = (0..32)
        .map(|i| 0.5f32 + (i as f32) * 0.0625)
        .collect();
    let x = Tensor::from_vec(x_data.clone(), [32], &device)?;
    let x2_data: Vec<f32> = (0..32)
        .map(|i| 1.0f32 + (i as f32 % 7.0) * 0.25)
        .collect();
    let x2 = Tensor::from_vec(x2_data.clone(), [4, 8], &device)?;

    let mut out = Tensor::zeros([1], &device)?;
    reduce_prod_1d(&x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_reduce_prod_1d", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_reduce_prod_1d");
    println!("rs_reduce_prod_1d ok");

    let mut out = Tensor::zeros([4], &device)?;
    reduce_prod_2d(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_reduce_prod_2d", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_reduce_prod_2d");
    println!("rs_reduce_prod_2d ok");

    let mut out = Tensor::zeros([32], &device)?;
    scan_sum_fwd(&x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_scan_sum_fwd", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_scan_sum_fwd");
    println!("rs_scan_sum_fwd ok");

    let mut out = Tensor::zeros([32], &device)?;
    scan_sum_rev(&x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_scan_sum_rev", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_scan_sum_rev");
    println!("rs_scan_sum_rev ok");

    let mut out = Tensor::zeros([32], &device)?;
    scan_prod_fwd(&x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_scan_prod_fwd", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_scan_prod_fwd");
    println!("rs_scan_prod_fwd ok");

    let mut out = Tensor::zeros([4, 8], &device)?;
    scan_2d_axis1(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_scan_2d_axis1", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_scan_2d_axis1");
    println!("rs_scan_2d_axis1 ok");

    let mut out = Tensor::zeros([4, 8], &device)?;
    scan_2d_axis0(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_scan_2d_axis0", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_scan_2d_axis0");
    println!("rs_scan_2d_axis0 ok");

    let mut out = Tensor::zeros([32], &device)?;
    reduce_closure(&x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_reduce_closure", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_reduce_closure");
    println!("rs_reduce_closure ok");

    let mut out = Tensor::zeros([8], &device)?;
    alias_lanes(&x, &x, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_alias_lanes", &[], &[&x_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_alias_lanes");
    println!("rs_alias_lanes ok");

    let mut out = Tensor::zeros([4, 16], &device)?;
    cat_cols(&x2, &x2, &mut out).grid((1, 1, 1)).apply(&stream)?;
    let expected = common::numpy("rs_cat_cols", &[], &[&x2_data, &x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_cat_cols");
    println!("rs_cat_cols ok");

    let mut out = Tensor::zeros([8, 8], &device)?;
    cat_rows(&x2, &x2, &mut out).grid((1, 1, 1)).apply(&stream)?;
    let expected = common::numpy("rs_cat_rows", &[], &[&x2_data, &x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_cat_rows");
    println!("rs_cat_rows ok");

    let mut out = Tensor::zeros([2, 4], &device)?;
    extract_block(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_extract_block", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_extract_block");
    println!("rs_extract_block ok");

    let sub_data: Vec<f32> = (0..8).map(|i| 100.0f32 + i as f32).collect();
    let sub = Tensor::from_vec(sub_data.clone(), [2, 4], &device)?;
    let mut out = Tensor::zeros([4, 8], &device)?;
    insert_block(&x2, &sub, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_insert_block", &[], &[&x2_data, &sub_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_insert_block");
    println!("rs_insert_block ok");

    let mut out = Tensor::zeros([8, 4], &device)?;
    permute_transpose(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_permute_transpose", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_permute_transpose");
    println!("rs_permute_transpose ok");

    let mut out = Tensor::zeros([4, 8], &device)?;
    permute_identity(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_permute_identity", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_permute_identity");
    println!("rs_permute_identity ok");

    let mut out = Tensor::zeros([4, 8], &device)?;
    assume_lanes(&x2, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;
    let expected = common::numpy("rs_assume_lanes", &[], &[&x2_data]);
    compare_exact(&out.to_vec(&stream)?, &expected, "rs_assume_lanes");
    println!("rs_assume_lanes ok");

    Ok(())
}
