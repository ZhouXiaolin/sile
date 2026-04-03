use sile::{exp, reduce_max, reduce_sum, Device, Tensor};

#[sile::kernel]
fn softmax(x: &Tensor<f32>, y: &mut Tensor<f32>) {
    let tile_x = x.load_tile_like_2d(y);
    let tile_x_max = reduce_max(tile_x.clone(), 1);
    let tile_x_max = tile_x_max.reshape([2, 1]).broadcast(&[4, 8]);
    let num = exp(tile_x - tile_x_max);
    let denom = reduce_sum(num.clone(), 1);
    let denom = denom.reshape([2, 1]).broadcast(&[4, 8]);
    y.store(num / denom);
}

#[test]
fn softmax_macro_emits_reduce_and_broadcast_ops() {
    let device = Device::cpu();
    let x = Tensor::ones([4, 8], &device).unwrap();
    let mut y = Tensor::zeros([4, 8], &device).unwrap();

    let kernel = softmax(&x, &mut y).grid((2, 1, 1)).kernel();

    assert_eq!(kernel.name, "softmax");
    assert!(kernel.body.len() >= 6);
}
