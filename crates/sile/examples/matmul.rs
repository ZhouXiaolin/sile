use sile::{Device, Tensor};

#[sile::kernel]
fn matmul<const BM: i32, const BN: i32, const BK: i32>(
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
    c: &mut Tensor<f32, { [BM, BN] }>,
) {
    let m_idx = sile::tile::id().0;
    let n_idx = sile::tile::id().1;

    let acc = sile::constant(0.0, [BM, BN]);

    let k_blocks = a.dim(1) / BK as i64;
    for k_idx in 0..k_blocks {
        let a_tile = a.load_tile([BM, BK], [m_idx, k_idx]);
        let b_tile = b.load_tile([BK, BN], [k_idx, n_idx]);
        let new_acc = sile::mma(a_tile, b_tile, acc);
        c.store(new_acc);
    }

    c.store(acc);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n, k) = (128i64, 256i64, 64i64);
    const BM: i32 = 64;
    const BN: i32 = 64;
    const BK: i32 = 32;

    let a = Tensor::random([m, k], &device)?;
    let b = Tensor::random([k, n], &device)?;
    let mut c = Tensor::zeros([m, n], &device)?;

    let grid = ((m / BM as i64) as u32, (n / BN as i64) as u32, 1);

    matmul::<BM, BN, BK>(&a, &b, &mut c)
        .grid(grid)
        .apply(&stream)?;

    let c_host = c.to_vec(&stream)?;

    // Host-side reference matmul for verify
    let a_host = a.to_vec(&stream)?;
    let b_host = b.to_vec(&stream)?;
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for l in 0..k as usize {
                sum += a_host[(i * k + l) as usize] * b_host[(l * n + j) as usize];
            }
            assert!(
                (c_host[(i * n + j) as usize] - sum).abs() < 1e-3,
                "mismatch at c[{i}][{j}]: got {}, expected {}",
                c_host[(i * n + j) as usize],
                sum
            );
        }
    }

    Ok(())
}
