use sile::{Device, Tensor, tile};

// ----------------------------------------------
// 1. 矩阵乘法 (Gemm) - 第一层全连接
// ----------------------------------------------
#[sile::kernel]
fn gemm<const BM: i64, const BN: i64, const BK: i64>(
    x: &Tensor<f32, { [-1, -1] }>, // [M, K]
    y: &Tensor<f32, { [-1, -1] }>, // [K, N]
    z: &mut Tensor<f32, { [BM, BN] }>,
) {
    let m_idx = tile::id().0;
    let n_idx = tile::id().1;
    let mut acc = sile::constant(0.0, [BM, BN]);

    let k_tiles = x.shape()[1] / BK;
    for k_idx in 0..k_tiles {
        let tile_x = x.load_tile([BM, BK], [m_idx, k_idx]);
        let tile_y = y.load_tile([BK, BN], [k_idx, n_idx]);
        acc = sile::mma(tile_x, tile_y, acc.clone());
    }
    z.store(acc);
}

// ----------------------------------------------
// 2. 矩阵-向量乘 (Matvec) - 第二层全连接
// ----------------------------------------------
#[sile::kernel]
fn matvec<const BM: i64, const BK: i64>(
    x: &Tensor<f32, { [-1, -1] }>, // [M, K]
    y: &Tensor<f32, { [-1] }>,     // [K]
    z: &mut Tensor<f32, { [BM] }>,
) {
    let m_idx = tile::id().0;
    let mut acc = sile::constant(0.0, [BM]);

    let k_tiles = x.shape()[1] / BK;
    for k_idx in 0..k_tiles {
        let tile_x = x.load_tile([BM, BK], [m_idx, k_idx]); // [BM, BK]
        let tile_y = y.load_tile([BK], [k_idx]).broadcast(&[BM, BK]); // [BM, BK]
        let partial = sile::reduce_sum(tile_x * tile_y, 1); // [BM]
        acc = acc + partial;
    }
    z.store(acc);
}

// ----------------------------------------------
// 3. ReLU 激活函数
// ----------------------------------------------
#[sile::kernel]
fn relu<const D: i64>(input_output: &mut Tensor<f32, { [D] }>) {
    let zero = sile::constant(0.0, [D]);
    let data = input_output.load_tile([D], [0]);
    input_output.store(sile::max_tile(zero, data));
}

// ----------------------------------------------
// 主函数：构造 MLP 并验证
// ----------------------------------------------
fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    // 超参数 (确保所有维度能被分块大小整除)
    let dim = 64; // 输入/隐藏/输出维度
    const BM_G: i64 = 32; // gemm 行分块
    const BN_G: i64 = 32; // gemm 列分块
    const BK_G: i64 = 8; // gemm 内部分块
    const BM_MV: i64 = 32; // matvec 行分块
    const BK_MV: i64 = 8; // matvec 内部分块

    // 1. 创建随机权重和输入
    let w0 = Tensor::random([dim, dim], &device)?; // 第一层权重 [dim, dim]
    let w1 = Tensor::random([dim], &device)?; // 第二层权重 [dim]
    let x_input = Tensor::random([dim, dim], &device)?; // 输入 [dim, dim]

    // 2. 第一层: out0 = x_input @ w0   (形状 [dim, dim])
    let mut out0 = Tensor::zeros([dim, dim], &device)?;
    let grid_gemm = ((dim / BM_G) as u32, (dim / BN_G) as u32, 1);
    gemm::<BM_G, BN_G, BK_G>(&x_input, &w0, &mut out0)
        .grid(grid_gemm)
        .apply(&stream)?;

    // 3. 第二层: out1 = out0 @ w1     (形状 [dim])
    let mut out1 = Tensor::zeros([dim], &device)?;
    let grid_matvec = ((dim / BM_MV) as u32, 1, 1);
    matvec::<BM_MV, BK_MV>(&out0, &w1, &mut out1)
        .grid(grid_matvec)
        .apply(&stream)?;

    // 4. ReLU 激活 (in-place)
    relu::<64>(&mut out1).grid((1, 1, 1)).apply(&stream)?;

    // 5. 将结果拷贝到主机
    let gpu_result = out1.to_vec(&stream)?;

    // 6. CPU 参考计算 (使用相同的随机权重和输入)
    let w0_host = w0.to_vec(&stream)?;
    let w1_host = w1.to_vec(&stream)?;
    let x_host = x_input.to_vec(&stream)?;

    // 6.1 第一层: out0_cpu = x_input @ w0
    let mut out0_cpu = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0;
            for k in 0..dim {
                sum += x_host[(i * dim + k) as usize] * w0_host[(k * dim + j) as usize];
            }
            out0_cpu[(i * dim + j) as usize] = sum;
        }
    }

    // 6.2 第二层: out1_cpu = out0_cpu @ w1
    let mut out1_cpu = vec![0.0f32; dim as usize];
    for i in 0..dim {
        let mut sum = 0.0;
        for k in 0..dim {
            sum += out0_cpu[(i * dim + k) as usize] * w1_host[k as usize];
        }
        out1_cpu[i as usize] = sum;
    }

    // 6.3 ReLU
    for val in &mut out1_cpu {
        if *val < 0.0 {
            *val = 0.0;
        }
    }

    // 7. 比较 GPU 和 CPU 结果
    let tolerance = 1e-3;
    let mut max_diff = 0.0;
    for i in 0..dim {
        let diff = (gpu_result[i as usize] - out1_cpu[i as usize]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < tolerance,
            "Mismatch at index {}: GPU={}, CPU={}",
            i,
            gpu_result[i as usize],
            out1_cpu[i as usize]
        );
    }
    println!(
        "MLP forward pass successful! Max difference = {:.2e}",
        max_diff
    );
    println!("First 5 outputs: {:?}", &gpu_result[..5]);

    Ok(())
}
