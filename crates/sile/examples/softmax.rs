use sile::{Device, Tensor};
use sile::load_tile_like_2d;
#[sile::kernel]
fn softmax<const BM: i64, const BN: i64>(
    x: &Tensor<f32, { [-1, -1] }>,
    y: &mut Tensor<f32, { [BM, BN] }>,
) {
    let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x,y);
    let tile_x_max: Tile<f32, { [BM] }> = sile::reduce_max(tile_x.clone(), 1i64);
    let tile_x_max_bcast: Tile<f32, { [BM, BN] }> =
        tile_x_max.reshape([BM, 1]).broadcast(&[BM, BN]);
    let num: Tile<f32, { [BM, BN] }> = sile::exp(tile_x - tile_x_max_bcast);
    let denom: Tile<f32, { [BM] }> = sile::reduce_sum(num.clone(), 1i64);
    let denom_bcast: Tile<f32, { [BM, BN] }> = denom.reshape([BM, 1]).broadcast(&[BM, BN]);
    y.store(num / denom_bcast);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let data: Vec<f32> = (0..(m * n) as i32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;
    softmax::<2, 8>(&x, &mut y).grid((2, 1, 1)).apply(&stream)?;

    let y_host = y.to_vec(&stream)?;
    for i in 0..m as usize {
        let row = &y_host[i * n as usize..(i + 1) * n as usize];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{i}] = {sum}");
        assert!((sum - 1.0).abs() < 1e-4);
    }
    Ok(())
}
