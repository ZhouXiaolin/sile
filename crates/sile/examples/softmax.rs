use sile::{Device, Tensor};

#[sile::kernel]
fn softmax(x: &Tensor<f32>, y: &mut Tensor<f32>) {
    let tile_x = x.load_tile_like_2d(y);
    let tile_x_max = reduce_max(tile_x, 1);
    let tile_x_max = tile_x_max.reshape([2, 1]).broadcast(y);
    let num = exp(tile_x - tile_x_max);
    let denom = reduce_sum(num, 1);
    let denom = denom.reshape([2, 1]).broadcast(y);
    y.store(num / denom);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let data: Vec<f32> = (0..(m * n) as i32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;

    softmax(&x, &mut y)
        .grid((2, 1, 1))
        .apply(&stream)?;

    let y_host = y.to_vec(&stream)?;
    for i in 0..m as usize {
        let row = &y_host[i * n as usize..(i + 1) * n as usize];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{i}] = {sum}");
        assert!((sum - 1.0).abs() < 1e-4);
    }
    Ok(())
}
