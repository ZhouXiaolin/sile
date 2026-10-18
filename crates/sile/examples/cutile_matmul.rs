//! Matrix multiplication written the way cutile-rs writes it.

use sile::{Device, Tensor};
use kernels::matmul;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn matmul<const BM: i64, const BN: i64, const BK: i64>(
        c: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f32, { [-1, -1] }>,
        b: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part_a = a.partition(shape![BM, BK]);
        let part_b = b.partition(shape![BK, BN]);
        let pid = get_tile_block_id();
        let mut tile_c = load_tile_mut(c);
        for k in 0i64..(a.shape()[1] / BK) {
            let tile_a = part_a.load([pid.0, k]);
            let tile_b = part_b.load([k, pid.1]);
            tile_c = mma(tile_a, tile_b, tile_c);
        }
        c.store(tile_c);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n, k) = (128i64, 128i64, 64i64);
    const BM: i64 = 32;
    const BN: i64 = 16;
    const BK: i64 = 8;

    let a = Tensor::random([m, k], &device)?;
    let b = Tensor::random([k, n], &device)?;
    let mut c = Tensor::zeros([m, n], &device)?;

    matmul::<BM, BN, BK>(&mut c, &a, &b)
        .grid(((m / BM) as u32, (n / BN) as u32, 1))
        .apply(&stream)?;

    let c_host = c.to_vec(&stream)?;
    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;
    let expected = common::numpy(
        "matmul",
        &[m.to_string(), k.to_string(), n.to_string()],
        &[&a_host, &b_host],
    );
    let (m, n, k) = (m as usize, n as usize, k as usize);
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c_host[i * n + j] - expected[i * n + j]).abs() < 1e-2,
                "mismatch at c[{i}][{j}]: got {}, expected {}",
                c_host[i * n + j],
                expected[i * n + j]
            );
        }
    }
    println!("cutile matmul {m}x{n}x{k} ok on {device:?}");
    Ok(())
}
