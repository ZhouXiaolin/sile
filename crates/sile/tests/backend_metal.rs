#![cfg(target_os = "macos")]

use sile::{Device, Tensor};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = sile::tile::id().0;
    let tile_a = a.load_tile([4], [tid]);
    let tile_b = b.load_tile([4], [tid]);
    c.store(tile_a + tile_b);
}

#[sile::kernel]
fn matmul<const BM: i64, const BN: i64, const BK: i64>(
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
    c: &mut Tensor<f32, { [BM, BN] }>,
) {
    let m_idx = sile::tile::id().0;
    let n_idx = sile::tile::id().1;

    let mut acc = sile::constant(0.0, [BM, BN]);
    for k_idx in 0..(a.shape()[1] / BK) {
        let a_tile = a.load_tile([BM, BK], [m_idx, k_idx]);
        let b_tile = b.load_tile([BK, BN], [k_idx, n_idx]);
        acc = sile::mma(a_tile, b_tile, acc.clone());
    }
    c.store(acc);
}

#[test]
fn metal_backend_executes_vec_add_through_compiler_pipeline() {
    let device = Device::metal();
    let stream = device.create_stream().unwrap();
    let a = Tensor::from_vec(vec![1.0; 16], [16], &device).unwrap();
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();

    let run = vec_add(&a, &b, &mut c).grid((4, 1, 1)).apply(&stream);
    if matches!(&run, Err(sile::Error::UnsupportedBackend(_))) {
        return;
    }
    run.unwrap();

    assert_eq!(c.to_vec(&stream).unwrap(), vec![3.0; 16]);
}

#[test]
fn metal_backend_executes_dynamic_k_matmul() {
    let device = Device::metal();
    let stream = device.create_stream().unwrap();

    const BM: i64 = 2;
    const BN: i64 = 2;
    const BK: i64 = 2;

    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
        [4, 4],
        &device,
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![
            1.0, 0.0, 2.0, 1.0, //
            0.0, 1.0, 3.0, 2.0, //
            1.0, 0.0, 4.0, 3.0, //
            0.0, 1.0, 5.0, 4.0,
        ],
        [4, 4],
        &device,
    )
    .unwrap();
    let mut c = Tensor::zeros([4, 4], &device).unwrap();

    let run = matmul::<BM, BN, BK>(&a, &b, &mut c)
        .grid((4, 1, 1))
        .apply(&stream);
    if matches!(&run, Err(sile::Error::UnsupportedBackend(_))) {
        return;
    }
    run.unwrap();

    let actual = c.to_vec(&stream).unwrap();
    let expected = vec![
        4.0, 6.0, 40.0, 30.0, //
        12.0, 14.0, 96.0, 70.0, //
        20.0, 22.0, 152.0, 110.0, //
        28.0, 30.0, 208.0, 150.0,
    ];
    assert_eq!(actual, expected);
}
