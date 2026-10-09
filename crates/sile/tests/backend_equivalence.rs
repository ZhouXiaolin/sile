#![cfg(target_os = "macos")]

//! Do all three targets compute the same number from the same IR?
//!
//! The shared emitter exists so a kernel is written once, so this is the
//! property that makes the third target worth adding at all: not that WGPU
//! emits something that parses, but that C, Metal and WGPU agree on the answer.
//! A disagreement here is a bug in one dialect, and this is the only place that
//! catches it without a hand-written table copied between targets.

use sile::{Device, Tensor, tile};

#[sile::kernel]
fn equivalence_vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = tile::id().0;
    let x = a.load_tile([4], [tid]);
    let y = b.load_tile([4], [tid]);
    c.store(x + y);
}

#[sile::kernel]
fn equivalence_matmul<const BM: i64, const BN: i64, const BK: i64>(
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
    c: &mut Tensor<f32, { [BM, BN] }>,
) {
    let m_idx = tile::id().0;
    let n_idx = tile::id().1;

    let mut acc = sile::constant(0.0, [BM, BN]);
    for k_idx in 0..(a.shape()[1] / BK) {
        let a_tile = a.load_tile([BM, BK], [m_idx, k_idx]);
        let b_tile = b.load_tile([BK, BN], [k_idx, n_idx]);
        acc = sile::mma(a_tile, b_tile, acc.clone());
    }
    c.store(acc);
}

#[sile::kernel]
fn equivalence_reduce<const TILE: i64>(a: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [1] }>) {
    let pid = tile::id().0;
    let tile_a = a.load_tile([TILE], [pid]);
    let sum_tile = sile::reduce_sum(tile_a, 0);
    out.atomic_add(0, sum_tile[0]);
}

/// Every device that can actually run a kernel here, in one order.
///
/// A device with no driver on this machine is skipped rather than failed, so
/// the comparison holds on any machine and the count assertion below still
/// catches a target that silently stops running.
fn devices() -> Vec<(&'static str, Device)> {
    let mut all: Vec<(&'static str, Device)> = Vec::new();
    for (name, device) in [
        ("cpu", Device::cpu()),
        ("metal", Device::metal()),
        ("wgpu", Device::wgpu()),
    ] {
        let stream = device.create_stream().unwrap();
        // The smallest kernel there is: one tile, four elements. If the target
        // cannot run that, it cannot be compared, so it is left out.
        let probe = Tensor::zeros([4], &device).unwrap();
        let mut sink = Tensor::zeros([4], &device).unwrap();
        if equivalence_vec_add(&probe, &probe, &mut sink)
            .grid((1, 1, 1))
            .apply(&stream)
            .is_err()
        {
            continue;
        }
        all.push((name, device));
    }
    all
}

/// The names that actually ran, so a silent skip cannot look like agreement.
fn assert_all_targets_ran(results: &[(&str, impl std::fmt::Debug)]) {
    assert_eq!(
        results.len(),
        3,
        "expected cpu, metal and wgpu to all run; got {:?}",
        results.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );
}

#[test]
fn three_targets_agree_on_vec_add() {
    let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, device) in devices() {
        let stream = device.create_stream().unwrap();
        let a = Tensor::from_vec(vec![1.0; 16], [16], &device).unwrap();
        let b = Tensor::from_vec(vec![2.0; 16], [16], &device).unwrap();
        let mut c = Tensor::zeros([16], &device).unwrap();
        equivalence_vec_add(&a, &b, &mut c)
            .grid((4, 1, 1))
            .apply(&stream)
            .unwrap();
        results.push((name, c.to_vec(&stream).unwrap()));
    }
    assert_all_targets_ran(&results);
    for (name, values) in &results[1..] {
        assert_eq!(values, &results[0].1, "{name} disagrees with cpu on vec_add");
    }
}

#[test]
fn three_targets_agree_on_dynamic_k_matmul() {
    const BM: i64 = 2;
    const BN: i64 = 2;
    const BK: i64 = 2;

    let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, device) in devices() {
        let stream = device.create_stream().unwrap();
        let a = Tensor::from_vec((1..=16).map(|i| i as f32).collect::<Vec<f32>>(), [4, 4], &device)
            .unwrap();
        let b = Tensor::from_vec(
            vec![
                1.0, 0.0, 2.0, 1.0, //
                0.0, 1.0, 3.0, 2.0, //
                1.0, 0.0, 4.0, 3.0, //
                0.0, 1.0, 5.0, 4.0,
            ],
            [4, 4],
            &device,
        )
        .unwrap();
        let mut c = Tensor::zeros([4, 4], &device).unwrap();
        equivalence_matmul::<BM, BN, BK>(&a, &b, &mut c)
            .grid((4, 1, 1))
            .apply(&stream)
            .unwrap();
        results.push((name, c.to_vec(&stream).unwrap()));
    }
    assert_all_targets_ran(&results);
    for (name, values) in &results[1..] {
        assert_eq!(values, &results[0].1, "{name} disagrees with cpu on matmul");
    }
}

/// The atomic case, which is the one where the three dialects cannot agree by
/// construction: WGSL spells the float add as a compare exchange over bit
/// patterns, and the other two use a native float atomic.
///
/// The comparison is exact rather than tolerant, and the data is chosen so it
/// can be. Every tile sums to 1000 and every partial sum is an integer no
/// larger than 100000, which f32 represents exactly, so any reduction order
/// reaches the same bits. That makes the three-way agreement a claim about the
/// lowering, not a claim that the differences stayed under some bound.
#[test]
fn three_targets_agree_on_parallel_reduce_exactly() {
    const TILE: i64 = 1000;
    const LEN: i64 = 100_000;

    let mut results: Vec<(&str, f32)> = Vec::new();
    for (name, device) in devices() {
        let stream = device.create_stream().unwrap();
        let a = Tensor::from_vec(vec![1.0; LEN as usize], [LEN], &device).unwrap();
        let mut out = Tensor::zeros([1], &device).unwrap();
        equivalence_reduce::<TILE>(&a, &mut out)
            .grid(((LEN / TILE) as u32, 1, 1))
            .apply(&stream)
            .unwrap();
        results.push((name, out.to_vec(&stream).unwrap()[0]));
    }

    assert_all_targets_ran(&results);
    let (_, first) = results[0];
    assert_eq!(first, LEN as f32, "cpu lost or corrupted the atomic sum");
    for (name, value) in &results[1..] {
        assert_eq!(*value, first, "{name} disagrees with cpu on the atomic sum");
    }
}
