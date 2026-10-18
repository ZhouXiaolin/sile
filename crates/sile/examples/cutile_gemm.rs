// The gemm kernel from cutile-examples/examples/gemm.rs, written with the same
// syntax so the front end is exercised by the real vocabulary: `shape![..]`,
// `get_tile_block_id()`, `x.partition(shape)` and `part.load([i, j])`.

use sile::ct::*;
use sile::{Device, Tensor};

#[sile::kernel]
fn gemm<const BM: i64, const BN: i64, const BK: i64, const K: i64>(
    z: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<f32, { [-1, K] }>,
    y: &Tensor<f32, { [-1, -1] }>,
) {
    let part_x = x.partition(shape![BM, BK]);
    let part_y = y.partition(shape![BK, BN]);
    let pid: (i64, i64, i64) = get_tile_block_id();
    let mut tile_z = load_tile_mut(z);
    for i in 0i64..(K / BK) {
        let tile_x = part_x.load([pid.0, i]);
        let tile_y = part_y.load([i, pid.1]);
        tile_z = sile::mma(tile_x, tile_y, tile_z);
    }
    z.store(tile_z);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const BM: i64 = 32;
    const BN: i64 = 16;
    const BK: i64 = 8;
    const K: i64 = 64;
    let (m, n) = (128i64, 128i64);

    let mut z = Tensor::zeros([m, n], &device)?;
    let x = Tensor::ones([m, K], &device)?;
    let y = Tensor::ones([K, n], &device)?;

    gemm::<BM, BN, BK, K>(&mut z, &x, &y)
        .grid(((m / BM) as u32, (n / BN) as u32, 1))
        .apply(&stream)?;

    let z_host = z.to_vec(&stream)?;
    for (i, value) in z_host.iter().enumerate().take(10) {
        println!("z_host[{i}] = {value} answer = {K}");
        assert!((value - K as f32).abs() < 1e-3, "z[{i}] = {value}, want {K}");
    }
    println!("cutile gemm {m}x{n}x{K} ok on {:?}", device);
    Ok(())
}
