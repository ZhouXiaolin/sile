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

#[test]
fn cpu_backend_executes_softmax_rows_that_sum_to_one() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();
    let data: Vec<f32> = (0..32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [4, 8], &device).unwrap();
    let mut y = Tensor::zeros([4, 8], &device).unwrap();

    softmax(&x, &mut y)
        .grid((2, 1, 1))
        .apply(&stream)
        .unwrap();

    let out = y.to_vec(&stream).unwrap();
    for row in out.chunks_exact(8) {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax row sum = {sum}, expected 1.0"
        );
    }
}
