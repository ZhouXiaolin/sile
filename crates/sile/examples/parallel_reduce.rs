use sile::{Device, Tensor, tile};

#[sile::kernel]
fn parallel_reduce<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    out: &mut Tensor<f32, { [1] }>,
) {
    let pid = tile::id().0;
    let tile_a = a.load_tile([TILE], [pid]);
    let sum_tile = sile::reduce_sum(tile_a, 0);
    out.atomic_add(0, sum_tile[0]);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const TILE: i64 = 100;
    const LEN: i64 = 10000;

    let a = Tensor::from_vec((1..=LEN).map(|i| i as f32).collect(), [LEN], &device)?;
    let mut out = Tensor::zeros([1], &device)?;

    parallel_reduce::<TILE>(&a, &mut out)
        .grid(((LEN / TILE) as u32, 1, 1))
        .apply(&stream)?;

    let result = out.to_vec(&stream)?[0];
    let expected = (LEN * (LEN + 1) / 2) as f32;
    let tolerance = expected * 1e-5;
    println!("sum = {}, expected = {}", result, expected);
    assert!((result - expected).abs() < tolerance);
    Ok(())
}
