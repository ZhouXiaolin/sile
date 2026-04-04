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

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n, k) = (128i64, 128i64, 64i64);
    const BM: i64 = 32;
    const BN: i64 = 16;
    const BK: i64 = 16;
    const K_BLOCKS: i64 = 4;

    let a = Tensor::random([m, k], &device)?;
    let b = Tensor::random([k, n], &device)?;
    let mut c = Tensor::zeros([m, n], &device)?;

    let grid = (((m / BM) * (n / BN)) as u32, 1u32, 1u32);

    matmul::<BM, BN, BK, K_BLOCKS>(&a, &b, &mut c)
        .grid(grid)
        .apply(&stream)?;

    let c_host = c.to_vec(&stream)?;

    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for l_idx in 0..k as usize {
                sum += a_host[i * k as usize + l_idx] * b_host[l_idx * n as usize + j];
            }
            assert!(
                (c_host[i * n as usize + j] - sum).abs() < 1e-2,
                "mismatch at c[{i}][{j}]: got {}, expected {}",
                c_host[i * n as usize + j],
                sum
            );
        }
    }

    Ok(())
}
