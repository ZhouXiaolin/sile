use sile::{Device, Tensor, tile};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = tile::id().0;
    let x = a.load_tile([4], [tid]);
    let y = b.load_tile([4], [tid]);
    c.store(x + y);
}

#[test]
fn cpu_backend_executes_vec_add() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();
    let a = Tensor::from_vec(vec![1.0; 16], [16], &device).unwrap();
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();

    vec_add(&a, &b, &mut c)
        .grid((4, 1, 1))
        .apply(&stream)
        .unwrap();

    assert_eq!(c.to_vec(&stream).unwrap(), vec![3.0; 16]);
}
