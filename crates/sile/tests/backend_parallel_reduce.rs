use sile::{Device, Tensor, tile};

#[sile::kernel]
fn parallel_reduce<const TILE: i64>(a: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [1] }>) {
    let pid = tile::id().0;
    let tile_a = a.load_tile([TILE], [pid]);
    let sum_tile = sile::reduce_sum(tile_a, 0);
    out.atomic_add(0, sum_tile[0]);
}

#[test]
fn cpu_backend_executes_parallel_reduce_with_atomic_add() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();

    const TILE: i64 = 1000;
    const LEN: i64 = 100_000;

    let a = Tensor::from_vec(vec![1.0; LEN as usize], [LEN], &device).unwrap();
    let mut out = Tensor::zeros([1], &device).unwrap();

    parallel_reduce::<TILE>(&a, &mut out)
        .grid(((LEN / TILE) as u32, 1, 1))
        .apply(&stream)
        .unwrap();

    let result = out.to_vec(&stream).unwrap()[0];
    assert!((result - LEN as f32).abs() < 1e-4);
}
