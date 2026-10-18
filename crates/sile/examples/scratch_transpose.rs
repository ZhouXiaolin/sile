//! A check that `transpose` exchanges the two tile axes, and that the result
//! still feeds a matrix multiply.

use sile::{Device, Tensor};
use kernels::at_b_transposed;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    /// `c = a @ b^T` where `b` is transposed on the device. The multiply reads the
    /// transposed tile, so a transpose that is wrong anywhere shows up in `c`.
    #[sile::entry]
    pub fn at_b_transposed<const BM: i64, const BN: i64, const BK: i64>(
        a: &Tensor<f32, { [BM, BK] }>,
        b: &Tensor<f32, { [BN, BK] }>,
        out: &mut Tensor<f32, { [BM, BN] }>,
    ) {
        let ta = a.load_like(out);
        let tb = b.load_like(out);
        let tb_t = tb.transpose();
        let acc = constant(0.0f32, shape![BM, BN]);
        out.store(mma(ta, tb_t, acc));
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (bm, bn, bk) = (4i64, 4i64, 4i64);
    let a_data: Vec<f32> = (0..bm * bk).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b_data: Vec<f32> = (0..bn * bk).map(|i| 1.0 + (i as f32) * 0.25).collect();
    let a = Tensor::from_vec(a_data.clone(), [bm, bk], &device)?;
    let b = Tensor::from_vec(b_data.clone(), [bn, bk], &device)?;
    let mut out = Tensor::zeros([bm, bn], &device)?;

    at_b_transposed::<4, 4, 4>(&a, &b, &mut out)
        .grid((1, 1, 1))
        .apply(&stream)?;

    let got = out.to_vec(&stream)?;
    for i in 0..bm as usize {
        for j in 0..bn as usize {
            let expected: f32 = (0..bk as usize)
                .map(|k| a_data[i * bk as usize + k] * b_data[j * bk as usize + k])
                .sum();
            let actual = got[i * bn as usize + j];
            assert!(
                (actual - expected).abs() < 1e-3,
                "out[{i}][{j}] = {actual}, expected {expected}"
            );
        }
    }
    println!("scratch transpose {bm}x{bn}x{bk} ok on {device:?}");
    Ok(())
}
