//! `softmax` written the way cutile-rs writes it.
//!
//! The reductions name an `i32` axis, and the two broadcasts take their target
//! shape from the output buffer. Both are syntax the ported kernel keeps.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn softmax<const BM: i64, const BN: i64>(
    y: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_x = x.load_like(y);
    let tile_x_max = reduce_max(tile_x.clone(), 1i64);
    let tile_x_max = tile_x_max.reshape(shape![BM, 1]).broadcast(y.shape());
    let num = exp(tile_x - tile_x_max);
    let denom = reduce_sum(num.clone(), 1i64);
    let denom = denom.reshape(shape![BM, 1]).broadcast(y.shape());
    y.store(num / denom);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let (bm, bn) = (2i64, n);
    let data: Vec<f32> = (0..m * n).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;

    softmax::<2, 8>(&mut y, &x)
        .grid(((m / bm) as u32, (n / bn) as u32, 1))
        .apply(&stream)?;

    let y_host = y.to_vec(&stream)?;
    for i in 0..m as usize {
        let row = &y_host[i * n as usize..(i + 1) * n as usize];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{i}] = {sum}");
        assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
    }
    println!("cutile softmax {m}x{n} ok on {device:?}");
    Ok(())
}
