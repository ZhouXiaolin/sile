use sile::{Device, Tensor};

#[sile::kernel]
fn matmul<const BM: i64, const BN: i64, const BK: i64, const K_BLOCKS: i64>(
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
    c: &mut Tensor<f32, { [BM, BN] }>,
) {
    let m_idx = sile::tile::id().0;
    let n_idx = sile::tile::id().1;

    let mut acc = sile::constant(0.0, [BM, BN]);
    for k_idx in 0..K_BLOCKS {
        let a_tile = a.load_tile([BM, BK], [m_idx, k_idx]);
        let b_tile = b.load_tile([BK, BN], [k_idx, n_idx]);
        acc = sile::mma(a_tile, b_tile, acc.clone());
    }
    c.store(acc);
}

#[test]
fn cpu_backend_executes_matmul_through_tile_pipeline() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();

    const BM: i64 = 2;
    const BN: i64 = 2;
    const BK: i64 = 2;
    const K_BLOCKS: i64 = 2;

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

    matmul::<BM, BN, BK, K_BLOCKS>(&a, &b, &mut c)
        .grid((4, 1, 1))
        .apply(&stream)
        .unwrap();

    let actual = c.to_vec(&stream).unwrap();
    let expected = vec![
        4.0, 6.0, 40.0, 30.0, //
        12.0, 14.0, 96.0, 70.0, //
        20.0, 22.0, 152.0, 110.0, //
        28.0, 30.0, 208.0, 150.0,
    ];

    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-4,
            "mismatch at {idx}: got {lhs}, expected {rhs}"
        );
    }
}
