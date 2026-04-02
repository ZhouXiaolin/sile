use sile::{tile, Device, Tensor};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = tile::id().0;
    let x = a.load_tile([4], [tid]);
    let y = b.load_tile([4], [tid]);
    c.store(x + y);
}

#[test]
fn macro_builds_a_launcher_with_the_expected_spec() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();

    let launcher = vec_add(&a, &b, &mut c).grid((4, 1, 1));
    assert_eq!(launcher.spec_ref().name, "vec_add");
    assert_eq!(launcher.spec_ref().nodes.len(), 3);
}
