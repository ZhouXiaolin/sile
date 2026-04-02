use sile::ssa::lower_typed_kernel_to_ssa;
use sile::typeck::check_kernel;
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
fn softmax_lowers_to_reduce_reshape_broadcast_ssa() {
    let device = Device::cpu();
    let x = Tensor::ones([4, 8], &device).unwrap();
    let mut y = Tensor::zeros([4, 8], &device).unwrap();
    let typed = check_kernel(softmax(&x, &mut y).kernel()).unwrap();
    let ssa = lower_typed_kernel_to_ssa(&typed);

    let opcodes = ssa
        .instructions
        .iter()
        .map(|inst| inst.opcode_name())
        .collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        vec![
            "load_tile_like_2d",
            "reduce_max",
            "reshape",
            "broadcast",
            "sub",
            "exp",
            "reduce_sum",
            "reshape",
            "broadcast",
            "div",
            "store",
        ]
    );
}
