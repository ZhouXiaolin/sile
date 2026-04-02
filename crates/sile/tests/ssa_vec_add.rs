use sile::ssa::lower_typed_kernel_to_ssa;
use sile::typeck::check_kernel;
use sile::{Device, Tensor};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = sile::tile::id().0;
    let tile_a = a.load_tile([4], [tid]);
    let tile_b = b.load_tile([4], [tid]);
    c.store(tile_a + tile_b);
}

#[test]
fn vec_add_lowers_to_three_ssa_values_before_store() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();
    let kernel = vec_add(&a, &b, &mut c).kernel().clone();

    let typed = check_kernel(&kernel).unwrap();
    let ssa = lower_typed_kernel_to_ssa(&typed);

    assert_eq!(ssa.instructions.len(), 4);
    assert_eq!(ssa.instructions[0].opcode_name(), "program_id");
    assert_eq!(ssa.instructions[3].opcode_name(), "store");
}
