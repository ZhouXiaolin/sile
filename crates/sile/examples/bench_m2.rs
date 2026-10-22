//! The M2 benchmark: the same four kernels at the same sizes as the
//! handwritten references in `handwritten/`.
//!
//! `docs/BACKEND_PLAN.md` M2 asks for saxpy, softmax and rms_norm at 80% of
//! hand-written C. That comparison only means something when both sides run the
//! same problem, so this example uses the reference's own sizes — 16M elements
//! for saxpy, 4096x1024 for softmax and rms_norm, 1024x1024x1024 for gemm —
//! and reports the median of 100 launches, which is how `handwritten/common.h`
//! times its side. The count is high because the first ~20 omp regions of any
//! process on this box run under a flat cold-start transient; the median has
//! to outvote them.
//!
//! The gemm arm takes `SILE_GEMM_M`/`SILE_GEMM_N`/`SILE_GEMM_K` (default
//! 1024) and `SILE_GEMM_ONES` for an all-ones fill. The accumulator seeds
//! from a zero tile, not from z: a benchmark relaunches the kernel, and a
//! z-seeded accumulator would stack every launch's product.
//!
//! Usage: `cargo run --release --example bench_m2 -- [saxpy|softmax|rms_norm|gemm]`

use std::time::Instant;

use sile::{Device, Tensor};

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn saxpy<const BM: i64, const BN: i64>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        a: f32,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }

    #[sile::entry]
    pub fn softmax<const BM: i64, const BN: i64>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(y);
        let tile_x_max = reduce_max(tile_x.clone(), 1i64);
        let tile_x_max = tile_x_max.reshape(shape![BM, 1]).broadcast(y.shape());
        let num = exp(tile_x - tile_x_max);
        let denom = reduce_sum(num.clone(), 1i64);
        let denom = denom.reshape(shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }

    #[sile::entry]
    pub fn gemm<const BM: i64, const BN: i64, const BK: i64, const K: i64>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i64, i64, i64) = get_tile_block_id();
        // The seed is a zero tile, not z's current content: a benchmark
        // relaunches the kernel many times, and a z-seeded accumulator would
        // add every launch's product onto the last.
        let zero: f32 = convert_scalar(0i64);
        let mut tile_z = broadcast_scalar(zero, shape![BM, BN]);
        for i in 0i64..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = sile::mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    #[sile::entry]
    pub fn rms_norm<const N: i64, const BLOCK_SIZE: i64>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape = shape![1, BLOCK_SIZE];
        let num_tiles: i64 = N / BLOCK_SIZE;
        let pid: (i64, i64, i64) = get_tile_block_id();
        let row = pid.0;

        let x_part = x.partition(tile_shape);
        let mut rms = constant(0.0, tile_shape);
        for j in 0i64..num_tiles {
            let tx = x_part.load([row, j]);
            rms = fma(tx, tx, rms);
        }
        let rms = reduce_sum(rms, 1i64);
        let rms = rms.reshape(shape![]);
        let rms = tile_to_scalar(rms);
        let n = convert_scalar(N);
        let rms = 1.0f32 / (rms / n + eps);
        let rms = sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
        let rms = tile_to_scalar(rms);
        let rms = rms.broadcast(tile_shape);

        let w_part = w.partition(shape![BLOCK_SIZE]);
        let mut out_part = out.partition_mut(tile_shape);
        for j in 0i64..num_tiles {
            let tx = x_part.load([row, j]);
            let tw = w_part.load([j]).reshape(tile_shape);
            let tout = tx * rms * tw;
            out_part.store(tout, [row, j]);
        }
    }
}

/// The median of `reps` launches, in milliseconds: the same statistic the
/// handwritten references report. The median needs more than half the samples
/// disturbed to move; a best-of lets one stray clock reading become the
/// number, and `clock_gettime` produces those here.
fn best_of<F: FnMut()>(reps: usize, mut run: F) -> f64 {
    // The first launch compiles and loads the kernel, which costs more than the
    // kernel itself; timing it would measure the compiler.
    run();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        run();
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
    samples[reps / 2]
}

fn report(kernel: &str, ms: f64, work: f64, unit: &str) {
    println!(
        "BENCH {kernel:<12} {ms:>10.3} ms  {:>8.2} {unit}",
        work / (ms * 1e-3) / 1e9
    );
}

fn run_saxpy(device: &Device) -> sile::Result<()> {
    let stream = device.create_stream()?;
    // 16M elements, the reference's size. The tile is small because the
    // lowering materialises every intermediate tile in stack storage: a
    // 1M-element tile is 4 MB, and five of them overflow the stack.
    const BM: i64 = 16;
    const BN: i64 = 1024;
    let n = 16 * 1024 * 1024i64;
    let (m, n) = (n / BN, BN);
    let x = Tensor::from_vec(vec![1.0f32; (m * n) as usize], [m, n], device)?;
    let mut y = Tensor::zeros([m, n], device)?;
    let mut y_part = y.partition_mut([BM, BN]);
    // `apply` consumes the launcher, so each repetition builds a fresh one:
    // what is timed is the launch, not the kernel's construction.
    let ms = best_of(100, || {
        kernels::saxpy::<BM, BN>(&mut y_part, 2.0, &x)
            .apply(&stream)
            .expect("saxpy launch")
    });
    // Two buffers read and one written: 3 * 4 bytes per element.
    report("saxpy", ms, (m * n * 3 * 4) as f64, "GB/s");
    Ok(())
}

fn run_softmax(device: &Device) -> sile::Result<()> {
    let stream = device.create_stream()?;
    let (rows, cols) = (4096i64, 1024i64);
    // One row per tile, the reference's own decomposition: a wider tile puts
    // two 16 KB workgroup tiles beside the reduce scratch, and a workgroup
    // that big pins the occupancy at one workgroup per compute unit — the
    // kernel then runs at a fraction of the hardware, not a fraction of 80%.
    const BM: i64 = 1;
    let x = Tensor::from_vec(vec![0.5f32; (rows * cols) as usize], [rows, cols], device)?;
    let mut y = Tensor::zeros([rows, cols], device)?;
    let mut y_part = y.partition_mut([BM, cols]);
    let ms = best_of(100, || {
        kernels::softmax::<BM, 1024>(&mut y_part, &x)
            .apply(&stream)
            .expect("softmax launch")
    });
    // The byte count must match the handwritten reference's own convention,
    // or the GB/s ratio compares two different denominators: softmax is
    // counted as a read of x, a read of y, a write of y, and the final read
    // for the checksum, 4 flows of 4 bytes.
    report("softmax", ms, (rows * cols * 4 * 4) as f64, "GB/s");
    Ok(())
}

fn run_rms_norm(device: &Device) -> sile::Result<()> {
    let stream = device.create_stream()?;
    let (rows, cols) = (4096i64, 1024i64);
    let x = Tensor::from_vec(vec![0.5f32; (rows * cols) as usize], [rows, cols], device)?;
    let mut w = Tensor::from_vec(vec![1.0f32; cols as usize], [cols], device)?;
    let mut out = Tensor::zeros([rows, cols], device)?;
    let mut out_part = out.partition_mut([1, cols]);
    let ms = best_of(100, || {
        kernels::rms_norm::<1024, 64>(&x, &mut w, &mut out_part, 1e-6)
            .apply(&stream)
            .expect("rms_norm launch")
    });
    // The row is read twice and written once.
    report("rms_norm", ms, (rows * cols * 3 * 4) as f64, "GB/s");
    Ok(())
}

fn run_gemm(device: &Device) -> sile::Result<()> {
    let stream = device.create_stream()?;
    let env_num = |name: &str, default: i64| {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let (m, n, k) = (
        env_num("SILE_GEMM_M", 1024),
        env_num("SILE_GEMM_N", 1024),
        env_num("SILE_GEMM_K", 1024),
    );
    // The reference's decomposition: a 32x16 output tile per workgroup, the
    // contraction walked in 16-deep panel steps.
    const BM: i64 = 32;
    const BN: i64 = 16;
    const BK: i64 = 16;

    // The same deterministic fill the reference harness uses, so the oracle
    // checks the product of the same inputs on both sides.
    let mut seed = 1u64;
    let mut fill = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    };
    let ones_mode = std::env::var_os("SILE_GEMM_ONES").is_some();
    let (xf, yf) = if ones_mode {
        (vec![1.0f32; (m * k) as usize], vec![1.0f32; (k * n) as usize])
    } else {
        (fill((m * k) as usize), fill((k * n) as usize))
    };
    let x = Tensor::from_vec(xf, [m, k], device)?;
    let y = Tensor::from_vec(yf, [k, n], device)?;
    let mut z = Tensor::zeros([m, n], device)?;
    let mut z_part = z.partition_mut([BM, BN]);
    let ms = best_of(100, || {
                if k == 64 {
            kernels::gemm::<BM, BN, BK, 64>(&mut z_part, &x, &y)
        } else if k == 256 {
            kernels::gemm::<BM, BN, BK, 256>(&mut z_part, &x, &y)
        } else {
            kernels::gemm::<BM, BN, BK, 1024>(&mut z_part, &x, &y)
        }
            .apply(&stream)
            .expect("gemm launch")
    });
    // The oracle on a corner of the output: the full product in f64 at 1024^3
    // is a minute of work for a check, and the reference checks the corner.
    let x_host = x.to_vec(&stream)?;
    let y_host = y.to_vec(&stream)?;
    let z_host = z.to_vec(&stream)?;
    let mut worst = 0.0f32;
    for i in 0..8usize {
        for j in 0..8usize {
            let mut sum = 0.0f64;
            for p in 0..k as usize {
                sum += x_host[i * k as usize + p] as f64 * y_host[p * n as usize + j] as f64;
            }
            worst = worst.max((z_host[i * n as usize + j] - sum as f32).abs());
        }
    }
    assert!(worst < 1e-2, "gemm corner error {worst}");
    report("gemm", ms, 2.0 * (m * n * k) as f64, "GFLOP/s");
    Ok(())
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let only: Vec<String> = std::env::args().skip(1).collect();
    let wanted = |name: &str| only.is_empty() || only.iter().any(|arg| arg == name);

    if wanted("saxpy") {
        run_saxpy(&device)?;
    }
    if wanted("softmax") {
        run_softmax(&device)?;
    }
    if wanted("rms_norm") {
        run_rms_norm(&device)?;
    }
    if wanted("gemm") {
        run_gemm(&device)?;
    }
    Ok(())
}
