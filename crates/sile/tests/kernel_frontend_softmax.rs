use sile::{exp, reduce_max, reduce_sum, Device, Tensor};

#[sile::kernel]
fn softmax<const BM: i64, const BN: i64>(
    x: &Tensor<f32, { [-1, -1] }>,
    y: &mut Tensor<f32, { [BM, BN] }>,
) {
    let tile_x = x.load_tile_like_2d(y);
    let tile_x_max = reduce_max(tile_x.clone(), 1);
    let tile_x_max = tile_x_max.reshape([BM, 1]).broadcast(&[BM, BN]);
    let num = exp(tile_x - tile_x_max);
    let denom = reduce_sum(num.clone(), 1);
    let denom = denom.reshape([BM, 1]).broadcast(&[BM, BN]);
    y.store(num / denom);
}

#[test]
fn softmax_macro_emits_reduce_and_broadcast_ops() {
    let device = Device::cpu();
    let x = Tensor::ones([4, 8], &device).unwrap();
    let mut y = Tensor::zeros([4, 8], &device).unwrap();

    let kernel = softmax::<2, 8>(&x, &mut y).grid((2, 1, 1)).kernel();

    assert_eq!(kernel.name, "softmax");
    assert!(kernel.body.len() >= 6);
}
