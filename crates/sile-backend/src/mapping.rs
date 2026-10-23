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

/// The largest workgroup a GPU kernel asks for.
///
/// Every adapter allows at least this along the x axis, and the kernels map
/// one tile to one workgroup, so a larger count only idles threads: the tile
/// sizes the lowering picks stay far below it.
const MAX_WORKGROUP: i64 = 256;

/// The mapping policy `target` runs `func` under.
///
/// C and PTX keep the serial policy — the mapping is identity for them. Metal
/// and WGSL map one tile to one workgroup and share the flat spread: the
/// threads cooperate on the tile's element runs, its folds and its multiply.
pub fn policy_for(target: CodegenTarget, func: &lsir::Function) -> MappingPolicy {
    // Debug switch: `SILE_SERIAL=1` forces the serial mapping on the parallel
    // targets so a numeric mismatch can be pinned on the parallel mapping or
    // the spelling alone.
    if matches!(target, CodegenTarget::Metal | CodegenTarget::Wgsl)
        && std::env::var_os("SILE_SERIAL").is_some()
    {
        let mut serial = MappingPolicy::serial();
        serial.vector = flat_tile_policy(func).vector;
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
        CodegenTarget::Metal => metal_tile_policy(func),
        CodegenTarget::Wgsl => flat_tile_policy(func),
        CodegenTarget::Ptx => MappingPolicy::serial(),
    }
}

/// The Metal flat spread: the shared tile-to-workgroup decision with the two
/// SIMD-group tiers on top. The fold runs on the hardware `simd_*`
/// primitives — feasible wherever the shared tree is, since a partial
/// SIMD-group folds its active lanes — and the multiply on
/// `simdgroup_matrix` fragments when the device and every site can. Where
/// either says no, the ladder's next rung is the shared tree and strips, the
/// decision `flat_tile_policy` already made.
fn metal_tile_policy(func: &lsir::Function) -> MappingPolicy {
    let mut policy = flat_tile_policy(func);
    if !policy.is_parallel() {
        return policy;
    }
    policy.reduce = ReducePolicy::SimdGroupTree;
    let threads = policy.launch.count() as i64;
    if simdgroup_matrix_supported()
        && lsir::mapping::simd_tiles_feasible(func, threads, policy.shared_cap)
    {
        policy.mma = MmaPolicy::SimdGroupTiles;
    }
    policy
}

/// Whether this machine's GPU lowers the `simdgroup_matrix` fragment
/// program. The fragment ops are an Apple7-family feature: elsewhere the
/// front end accepts them and the pipeline compile fails with `SC
/// compilation failure: there is a call to an undefined label` — an AMD
/// Radeon Pro 5500M (Mac2) reproduces exactly that, while the `simd_*`
/// reduction primitives lower fine on the same device. The capability is a
/// per-process constant, so one probe feeds every policy decision; on a
/// device without it the mapping keeps the ThreadStrips rung, which is the
/// degrade-when-absent rule the ladder spells.
#[cfg(target_os = "macos")]
fn simdgroup_matrix_supported() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        metal::Device::system_default()
            .map(|device| device.supports_family(metal::MTLGPUFamily::Apple7))
            .unwrap_or(false)
    })
}

#[cfg(not(target_os = "macos"))]
fn simdgroup_matrix_supported() -> bool {
    false
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

/// One tile to one workgroup: the threads cooperate on the tile's
/// element runs, its folds and its multiply. The GPU targets whose flat
/// spread is identical share this decision, so a kernel maps the same way
/// wherever it runs.
fn flat_tile_policy(func: &lsir::Function) -> MappingPolicy {
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
        // The smallest workgroup storage either dialect guarantees (WGSL's
        // spec floor; MSL guarantees 32 KB, so the WGSL number covers both).
        // A tile at or under it moves to workgroup storage; a larger one —
        // the saxpy bench's 64 KB strip — stays private, correct but spilled.
        shared_cap: 16384,
    }
}
