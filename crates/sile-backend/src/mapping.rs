//! The per-target mapping decisions.
//!
//! The mapping family in `sile-lsir` is policy-driven: the transforms are
//! general, and this module is the one place that decides what a target asks
//! of them. The decision is a pure function of the target and the function,
//! so the compile path, the emitter and the dump can each recompute it and
//! cannot disagree.

use sile_lsir as lsir;
use sile_lsir::launch::LaunchShape;
use sile_lsir::mapping::{MappingPolicy, MmaPolicy, ReducePolicy, VectorPolicy};

use crate::CodegenTarget;

/// The largest workgroup a WGSL kernel asks for.
///
/// Every adapter allows at least this along the x axis, and the kernels map
/// one tile to one workgroup, so a larger count only idles threads: the tile
/// sizes the lowering picks stay far below it.
const MAX_WORKGROUP: i64 = 256;

/// The mapping policy `target` runs `func` under.
///
/// C, Metal and PTX keep the serial policy — the mapping is identity for
/// them, which is what keeps their output byte-identical to the pre-mapping
/// pipeline. C's `omp simd` hint is a mapping judgment that survives a serial
/// mapping: the pragma belongs to the spelling, and the policy only says
/// whether the target wants it.
pub fn policy_for(target: CodegenTarget, func: &lsir::Function) -> MappingPolicy {
    // Debug switch: `SILE_SERIAL=1` forces the serial mapping so a numeric
    // mismatch can be pinned on the parallel mapping or the spelling alone.
    if target == CodegenTarget::Wgsl && std::env::var_os("SILE_SERIAL").is_some() {
        let mut serial = MappingPolicy::serial();
        serial.vector = wgsl_policy(func).vector;
        return serial;
    }
    match target {
        CodegenTarget::C => MappingPolicy {
            launch: LaunchShape::SERIAL,
            reduce: ReducePolicy::Serial,
            mma: MmaPolicy::Serial,
            vector: VectorPolicy::OmpSimd,
            shared_cap: 0,
        },
        CodegenTarget::Metal | CodegenTarget::Ptx => MappingPolicy::serial(),
        CodegenTarget::Wgsl => wgsl_policy(func),
    }
}

/// The serial policy that keeps `vector`, for a kernel the parallel mapping
/// rejected: the spelling judgment survives, only the threading is given up.
pub fn policy_with_vector(vector: lsir::VectorPolicy) -> MappingPolicy {
    MappingPolicy {
        launch: LaunchShape::SERIAL,
        reduce: ReducePolicy::Serial,
        mma: MmaPolicy::Serial,
        vector,
        shared_cap: 0,
    }
}

/// WGSL maps one tile to one workgroup: the threads cooperate on the tile's
/// element runs, its folds and its multiply.
fn wgsl_policy(func: &lsir::Function) -> MappingPolicy {
    // The output tile is the extent the threads share. A kernel without one
    // has no work to divide, and a one-element tile has no parallelism worth
    // two barriers; both stay serial.
    let Some((rows, cols)) = lsir::mapping::output_tile(func) else {
        return MappingPolicy::serial();
    };
    let tile_len = (rows * cols) as i64;
    if tile_len < 2 {
        return MappingPolicy::serial();
    }
    // A thread past every plumbing loop's extent would idle through all of
    // them and still pay for the barrier rounds, so the extent caps the count.
    let ceiling = lsir::mapping::plumbing_extent(func)
        .map(|extent| extent.min(tile_len))
        .unwrap_or(tile_len)
        .min(MAX_WORKGROUP);
    // The tree reduction walks its strides by halving, so the count is a
    // power of two: round down to the nearest one.
    let threads = 1i64 << ceiling.ilog2();
    if threads < 2 {
        return MappingPolicy::serial();
    }
    MappingPolicy {
        launch: LaunchShape::flat(threads as u32),
        reduce: ReducePolicy::WorkgroupTree,
        mma: MmaPolicy::ThreadStrips,
        vector: VectorPolicy::None,
        // The WGSL spec's smallest guaranteed workgroup storage. A tile at or
        // under it moves to `var<workgroup>`; a larger one — the saxpy bench's
        // 64 KB strip — stays private, correct but spilled.
        shared_cap: 16384,
    }
}
