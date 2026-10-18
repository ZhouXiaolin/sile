//! Flash attention with a causal mask, written the way cutile-rs writes it.
//!
//! This port adds four things to the plain attention example. A branch on a
//! const generic picks the causal path at compile time. A comparison of two
//! counts builds the mask. `iota` builds the coordinate tiles the comparison
//! reads. And `&` combines two masks into one.
//!
//! The mask decides two separate things, which is why the branch appears twice.
//! `mask_start` and `tc` decide how many key tiles this block reads, and the
//! per-element comparison decides which scores inside a tile survive. A tile
//! that is entirely masked away costs nothing to skip, so the block skips it.

use sile::ct::*;
use sile::{Device, Tensor};

#[path = "common/mod.rs"]
mod common;

#[sile::kernel]
fn fmha<
    const BM: i64,
    const BN: i64,
    const D: i64,
    const H: i64,
    const CAUSAL: i64,
    const EVEN_K: i64,
>(
    out: &mut Tensor<f32, { [1, BM, D] }>, // (b*h, m, d)
    q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, h, m, d)
    k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
    v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
    qk_scale: f32,
    input_pos: i64,
    query_group_size: i64,
) {
    let pid = get_tile_block_id(); // (b*h, m/BM, 1)
    let batch_idx = pid.0 / H;
    let q_head_idx = pid.0 % H;
    let q_m_idx = pid.1;
    let kv_head_idx = q_head_idx / query_group_size;

    // This lets us use exp2 vs exp.
    let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
    let log2: f32 = tile_to_scalar(log(two));
    let qk_scale: Tile<f32, { [BM, BN] }> = broadcast_scalar(qk_scale / log2, shape![BM, BN]);

    // m and l are for softmax.
    let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
    let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
    // This is the output tile.
    let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

    // We load just one query block per process.
    let q_part = q.partition(shape![1, 1, BM, D]);
    let tq: Tile<f32, { [BM, D] }> = q_part
        .load([batch_idx, q_head_idx, q_m_idx, 0i64])
        .reshape(shape![BM, D]);

    // The last query row of this block, in absolute position. Rows past this one
    // are the ones the causal mask hides.
    let k_seqlen: i64 = k.shape()[2];
    let m_end: i64 = input_pos + (q_m_idx + 1i64) * BM;
    // Without a mask every key tile is read; the causal path narrows the range.
    let mut mask_start: i64 = k_seqlen / BN;
    let mut tc: i64 = ceil_div(k_seqlen, BN);
    if CAUSAL == 1i64 {
        mask_start = (input_pos + q_m_idx * BM) / BN;
        let k_seqlen_tiles = k_seqlen / BN;
        mask_start = min(mask_start, k_seqlen_tiles);
        tc = ceil_div(min(m_end, k_seqlen), BN);
    }

    let k_part = k.partition(shape![1, 1, BN, D]);
    let v_part = v.partition(shape![1, 1, BN, D]);

    // The key coordinate of each column, and the query coordinate of each row.
    let offs_n_tile: Tile<i32, { [BN] }> = iota(shape![BN]);
    let offs_n_tile: Tile<i32, { [BM, BN] }> =
        offs_n_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);

    let offs_m_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
    let offs_m_iota = offs_m_iota.reshape(shape![BM, 1]);
    let offs_m: Tile<i32, { [BM, 1] }> =
        broadcast_scalar(q_m_idx * BM + input_pos, shape![BM, 1]) + offs_m_iota;
    let offs_m: Tile<i32, { [BM, BN] }> = offs_m.broadcast(shape![BM, BN]);
    let k_seqlen_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(k_seqlen, shape![BM, BN]);
    let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
    let mask_false: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);

    // j corresponds to tile index along key / value seq len dim.
    for j in 0i64..tc {
        // Compute q @ k^T.
        let k_tile: Tile<f32, { [BN, D] }> = k_part
            .load([batch_idx, kv_head_idx, j, 0i64])
            .reshape(shape![BN, D]);
        let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
        let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        qk = mma(tq, k_tile_trans, qk);

        // A key tile that lies entirely at or before the mask start is either
        // all visible or all hidden, so only the boundary tile needs a mask.
        if (CAUSAL == 1i64 || EVEN_K == 0i64) && j >= mask_start {
            let offs_n: Tile<i32, { [BM, BN] }> =
                broadcast_scalar(j * BN, shape![BM, BN]) + offs_n_tile;
            let mut mask: Tile<bool, { [BM, BN] }> = constant(true, shape![BM, BN]);
            // A short key sequence leaves columns past its end with no key.
            if EVEN_K == 0i64 {
                let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, k_seqlen_tile);
                mask = mask & lt_res;
            }
            // A key is visible when it is not later than the query.
            if CAUSAL == 1i64 {
                let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                mask = mask & ge_res;
            }
            qk = qk + select(mask, mask_true, mask_false);
        }

        // Apply scale(q @ k^T).
        qk = qk * qk_scale;

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
        l_i = l_i * alpha + l_ij;
        acc = acc * alpha.broadcast(shape![BM, D]);

        // Compute softmax(mask(scale(q @ k^T))) @ v.
        let v_tile: Tile<f32, { [BN, D] }> = v_part
            .load([batch_idx, kv_head_idx, j, 0i64])
            .reshape(shape![BN, D]);
        acc = mma(p, v_tile, acc);
        m_i = m_ij;
    }

    out.store(true_div(acc, l_i.broadcast(shape![BM, D])).reshape(shape![1, BM, D]));
}

/// One head of one batch, on the host: the same arithmetic the kernel performs,
/// without the tiling and without the mask.
#[allow(clippy::too_many_arguments)]
fn run(causal: bool, device: &Device) -> sile::Result<()> {
    let stream = device.create_stream()?;

    let (b, h, m, d) = (2i64, 8i64, 64i64, 32i64);
    let hkv = 8i64;
    let bm = 32i64;
    let input_pos = if causal { 5i64 } else { 0i64 };
    // The sequence length divides the key tile, so the short-tile mask path
    // (EVEN_K = 0) is never taken; the instantiations below pass 1.
    let qk_scale = 1.0f32 / (d as f32).sqrt();

    let sample = |count: i64, salt: f32| -> Vec<f32> {
        (0..count)
            .map(|i| ((i as f32 + salt) * 0.37).sin() * 0.5)
            .collect()
    };
    let q_data = sample(b * h * m * d, 0.0);
    let k_data = sample(b * hkv * m * d, 1.0);
    let v_data = sample(b * hkv * m * d, 2.0);

    let q = Tensor::from_vec(q_data.clone(), [b, h, m, d], device)?;
    let k = Tensor::from_vec(k_data.clone(), [b, hkv, m, d], device)?;
    let v = Tensor::from_vec(v_data.clone(), [b, hkv, m, d], device)?;
    let mut out = Tensor::zeros([b * h, m, d], device)?;

    let grid = ((b * h) as u32, (m / bm) as u32, 1);
    let launcher = if causal {
        fmha::<32, 32, 32, 8, 1, 1>(&mut out, &q, &k, &v, qk_scale, input_pos, h / hkv)
    } else {
        fmha::<32, 32, 32, 8, 0, 1>(&mut out, &q, &k, &v, qk_scale, input_pos, h / hkv)
    };
    launcher.grid(grid).apply(&stream)?;

    let got = out.to_vec(&stream)?;
    let op = if causal {
        "flash_attention_causal"
    } else {
        "flash_attention"
    };
    let expected = common::numpy(
        op,
        &[
            m.to_string(),
            d.to_string(),
            qk_scale.to_string(),
            input_pos.to_string(),
        ],
        &[&q_data, &k_data, &v_data],
    );
    let worst = got
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    if worst > 1e-5 {
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            if (a - e).abs() > 1e-5 {
                let d = d as usize;
                eprintln!(
                    "first mismatch at {i}: got {a} expected {e} (plane {}, row {}, col {})",
                    i / (m as usize * d),
                    (i / d) % m as usize,
                    i % d
                );
                break;
            }
        }
    }
    assert!(worst <= 1e-5, "flash attention (causal={causal}) differs by {worst}");
    println!(
        "cutile flash_attention_causal ({}) {b}x{h}x{m}x{d} ok on {device:?}",
        if causal { "causal" } else { "plain" }
    );
    Ok(())
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    run(false, &device)?;
    run(true, &device)?;
    Ok(())
}
