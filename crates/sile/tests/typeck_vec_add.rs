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
fn vec_add_typeck_infers_tile_shapes() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();
    let kernel = vec_add(&a, &b, &mut c).kernel().clone();

    let typed = check_kernel(&kernel).unwrap();

    assert_eq!(typed.locals["tile_a"].category(), sile::hir::ValueCategory::Tile);
    assert_eq!(typed.locals["tile_b"].category(), sile::hir::ValueCategory::Tile);
}
