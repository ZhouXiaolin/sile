use sile::ssa::lower_typed_kernel_to_ssa;
use sile::typeck::check_kernel;
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
fn softmax_lowers_to_reduce_reshape_broadcast_ssa() {
    let device = Device::cpu();
    let x = Tensor::ones([4, 8], &device).unwrap();
    let mut y = Tensor::zeros([4, 8], &device).unwrap();
    let typed = check_kernel(softmax::<2, 8>(&x, &mut y).kernel()).unwrap();
    let ssa = lower_typed_kernel_to_ssa(&typed);

    let opcodes = ssa
        .instructions
        .iter()
        .map(|inst| inst.opcode_name())
        .collect::<Vec<_>>();
    assert!(opcodes.iter().filter(|op| **op == "program_id").count() >= 1);
    assert!(opcodes.iter().filter(|op| **op == "shape_dim").count() >= 2);
    let expected = [
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
    ];
    let mut cursor = 0usize;
    for opcode in opcodes {
        if cursor < expected.len() && opcode == expected[cursor] {
            cursor += 1;
        }
    }
    assert_eq!(cursor, expected.len());
}
