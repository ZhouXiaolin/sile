//! Flash attention, ported from cutile-examples/examples/flash_attention.rs.
//!
//! This is the widest single port: a rank-4 input addressed through a rank-4
//! partition, a transposed tile feeding a matrix multiply, a rank-zero tile
//! read back as a scalar, and a softmax that runs in the exponential's base-two
//! form.
//!
//! Three things differ from the original. CuTile's `i32` annotations are `i64`
//! here, because a kernel's const generics are `i64` in this front end. The
//! original takes a host-partitioned output; here the whole buffer arrives and
//! the program names the plane it owns. And the scale is a kernel argument
//! rather than a device-side square root, which is where the original gets it.

use sile::{Device, Tensor};
use kernels::fmha;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn fmha<const BM: i64, const BN: i64, const D: i64>(
        out: &mut Tensor<f32, { [1, BM, D] }>, // (b*h, m, d)
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, h, m, d)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
        qk_scale: f32,
        query_group_size: i64,
    ) {
        let pid = get_tile_block_id(); // (b*h, m/bm, 1)
        let h = q.shape()[1];
        let batch_idx = pid.0 / h; // in [0, b)
        let q_head_idx = pid.0 % h; // in [0, h)
        let q_m_idx = pid.1; // in [0, m/bm)
        let kv_head_idx = q_head_idx / query_group_size;

        // This lets us use exp2 vs exp.
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(shape![BM, BN]);

        // m and l are for softmax.
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        // This is the output tile.
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        // We load just one query block per process.
        let q_part = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [1, 1, BM, D] }> = q_part.load([batch_idx, q_head_idx, q_m_idx, 0i64]);
        let tq: Tile<f32, { [BM, D] }> = tq.reshape(shape![BM, D]);

        let n: i64 = k.shape()[2];
        let num_tiles: i64 = ceil_div(n, BN);

        let k_part = k.partition(shape![1, 1, BN, D]); // permuted after loading.
        let v_part = v.partition(shape![1, 1, BN, D]);

        // j corresponds to tile index along key / value seq len dim.
        for j in 0i64..num_tiles {
            // Compute q @ k^T.
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, kv_head_idx, j, 0i64])
                .reshape(shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);

            // Apply scale(q @ k^T).
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            // Recenter before softmax.
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM, BN]);

            // Apply softmax(mask(scale(q @ k^T))).
            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = fma(l_i, alpha, l_ij);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(shape![BM, D]);
            acc = acc * alpha;

            // Compute softmax(mask(scale(q @ k^T))) @ v.
            let v_tile: Tile<f32, { [1, 1, BN, D] }> = v_part.load([batch_idx, kv_head_idx, j, 0i64]);
            let v_tile: Tile<f32, { [BN, D] }> = v_tile.reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        acc = true_div(acc, l_i.broadcast(shape![BM, D]));
        let acc = acc.reshape(shape![1, BM, D]);
        out.store(acc);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (b, h, m, d) = (1i64, 2i64, 8i64, 8i64);
    let hkv = 1i64;
    let (bm, _bn) = (4i64, 4i64);
    let qk_scale = 1.0f32 / (d as f32).sqrt();

    let sample = |count: i64, salt: f32| -> Vec<f32> {
        (0..count)
            .map(|i| ((i as f32 + salt) * 0.37).sin() * 0.5)
            .collect()
    };
    let q_data = sample(b * h * m * d, 0.0);
    let k_data = sample(b * hkv * m * d, 1.0);
    let v_data = sample(b * hkv * m * d, 2.0);

    let q = Tensor::from_vec(q_data.clone(), [b, h, m, d], &device)?;
    let k = Tensor::from_vec(k_data.clone(), [b, hkv, m, d], &device)?;
    let v = Tensor::from_vec(v_data.clone(), [b, hkv, m, d], &device)?;
    let mut out = Tensor::zeros([b * h, m, d], &device)?;
    let mut out_part = out.partition_mut([1, bm, d]);

    fmha::<4, 4, 8>(&mut out_part, &q, &k, &v, qk_scale, h / hkv).apply(&stream)?;

    let got = out_part.unpartition().to_vec(&stream)?;
    // The GQA group comes from the array sizes on the python side: the q plane
    // count over the kv plane count, with the batch folded into the planes.
    let expected = common::numpy(
        "flash_attention",
        &[
            m.to_string(),
            d.to_string(),
            qk_scale.to_string(),
        ],
        &[&q_data, &k_data, &v_data],
    );
    let worst = got
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= 1e-4, "flash attention differs by {worst}");
    println!("cutile flash_attention {b}x{h}x{m}x{d} ok on {device:?}");
    Ok(())
}
