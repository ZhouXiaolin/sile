use sile::ops::{exp, reduce_max, reduce_sum};
use sile::{Device, Stream, Tensor};

#[sile::kernel]
fn softmax<const BM: i32, const BN: i32>(
    x: &Tensor<f32, { [-1, -1] }>,
    y: &mut Tensor<f32, { [BM, BN] }>,
) {
    let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x, y);
    let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
    let tile_x_max: Tile<f32, { [BM, BN] }> =
        tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
    let num: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
    let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1);
    let denom = denom.reshape(const_shape![BM, 1]).broadcast(y.shape());
    y.store(num / denom);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4, 8);
    let (bm, bn) = (2, n as i32);

    // 创建输入张量 x：值为 0..(m*n-1) 并 reshape 为 [m, n]
    let data: Vec<f32> = (0..(m * n) as i32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;

    // 创建输出张量 y：形状 [m, n]，划分为 [bm, bn] 的 tile
    let y = Tensor::zeros([m, n], &device).partition([bm, bn]);

    // 调用内核，设置网格（tile 块数量）
    softmax(&x, &mut y)
        .grid((m / bm, n / bn, 1))
        .apply(&stream)?;

    // 等待执行完成，合并 tile 并取回主机数据
    let y_host = y.unpartition().to_vec(&stream)?;

    // 验证每一行的 softmax 输出和为 1
    for i in 0..m {
        let start = i * n;
        let row = &y_host[start..start + n];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{}] = {}", i, sum);
        assert!((sum - 1.0).abs() < 1e-6);
    }

    Ok(())
}
