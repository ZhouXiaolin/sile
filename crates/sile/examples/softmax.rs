//! `softmax`, ported from cutile-examples/examples/softmax.rs.
//!
//! The reductions name an axis, and the two broadcasts take their target shape
//! from the output buffer. The host sizes match the original: (4, 8) input,
//! (2, 8) tile, so the launch is two programs, one per row block. The original
//! checks that each output row sums to one; this port compares every element
//! against `numpy_ref.py softmax`.

use sile::{Device, Tensor};
use kernels::softmax;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn softmax<const BM: i64, const BN: i64>(
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
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let (bm, bn) = (2i64, n);
    let data: Vec<f32> = (0..m * n).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data.clone(), [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;
    let mut y_part = y.partition_mut([bm, bn]);

    softmax::<2, 8>(&mut y_part, &x).apply(&stream)?;

    let y_host = y_part.unpartition().to_vec(&stream)?;
    for i in 0..m as usize {
        let row = &y_host[i * n as usize..(i + 1) * n as usize];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{i}] = {sum}");
        assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
    }
    let expected = common::numpy("softmax", &[m.to_string(), n.to_string()], &[&data]);
    let worst = y_host
        .iter()
        .zip(&expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= 1e-5, "softmax differs from numpy by {worst}");
    println!("cutile softmax {m}x{n} ok on {device:?}");
    Ok(())
}
