use sile::{Device, Tensor};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = sile::tile::id().0;
    let tile_a = a.load_tile([4], [tid]);
    let tile_b = b.load_tile([4], [tid]);
    c.store(tile_a + tile_b);
}

#[test]
fn vec_add_macro_emits_hir_kernel() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();

    let launcher = vec_add(&a, &b, &mut c).grid((4, 1, 1));
    let kernel = launcher.kernel();

    assert_eq!(kernel.name, "vec_add");
    assert_eq!(kernel.params.len(), 3);
    assert!(kernel.body.len() >= 3);
}
