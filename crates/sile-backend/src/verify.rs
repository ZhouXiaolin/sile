use sile_core::{Error, Result};
use sile_lsir::Function as LsirFunction;

use crate::emit::{find_alloca_for_loops, is_loop_header};
use crate::CodegenTarget;

pub(crate) fn for_target(
    lsir: &LsirFunction,
    target: CodegenTarget,
    stage: &str,
) -> Result<()> {
    verify_shared_contract(lsir, stage)?;
    match target {
        CodegenTarget::C => Ok(()),
        CodegenTarget::Metal | CodegenTarget::Wgsl | CodegenTarget::Ptx => {
            verify_gpu_contract(lsir, stage)
        }
    }
}

fn verify_shared_contract(lsir: &LsirFunction, stage: &str) -> Result<()> {
    for block in &lsir.blocks {
        if matches!(block.terminator, sile_lsir::Terminator::Switch { .. }) {
            return Err(Error::Compile(format!(
                "{stage}: backend emit does not support switch terminators (block `{}`)",
                block.name
            )));
        }
        for inst in &block.insts {
            if let sile_lsir::InstOp::Call { func, .. } = &inst.op
                && func.starts_with("tile_")
            {
                return Err(Error::Compile(format!(
                    "{stage}: backend emit does not allow tile helper calls in LSIR, got `{func}` (block `{}`)",
                    block.name
                )));
            }
            // The SIMD-group intrinsics are the Metal mapping's plants: the
            // lowering never produces them, so one in the input means the
            // pipeline changed under the mapping. Checked before the target
            // split so even the CPU target's input says no.
            if let sile_lsir::InstOp::Intrinsic { intrinsic, .. } = &inst.op
                && matches!(
                    intrinsic,
                    sile_lsir::Intrinsic::SimdReduce { .. }
                        | sile_lsir::Intrinsic::SimdLaneId
                        | sile_lsir::Intrinsic::SimdGroupId
                        | sile_lsir::Intrinsic::SimdGroupMma { .. }
                )
            {
                return Err(Error::Compile(format!(
                    "{stage}: the lowering never produces the SIMD-group intrinsics — only the mapping plants them (block `{}`)",
                    block.name
                )));
            }
        }
    }
    verify_structured_shapes(lsir, stage)
}

/// The emitter spells three shapes — straight line, counted `for`, if/else —
/// and a loop header outside the counted shape would silently fall back to a
/// `while (true)` loop: correct text, but no vectorization hint and a contract
/// violation nobody sees. This check makes that fallback loud instead: every
/// loop header must match the alloca-form counted loop the lowering emits.
fn verify_structured_shapes(lsir: &LsirFunction, stage: &str) -> Result<()> {
    let (for_loops, _) = find_alloca_for_loops(lsir);
    let dominators = sile_lsir::passes::cfg::dominators(lsir);
    for block in &lsir.blocks {
        if matches!(block.terminator, sile_lsir::Terminator::CondBr { .. })
            && is_loop_header(lsir, block, &dominators)
            && !for_loops.contains_key(&block.id)
        {
            return Err(Error::Compile(format!(
                "{stage}: loop header `{}` is not a counted loop (index alloca with an entry init, a header load+compare, and a latch step), so backend emit cannot spell it as `for`",
                block.name
            )));
        }
    }
    Ok(())
}

/// What both GPU targets reject, named for both.
///
/// WGSL and MSL both reject a call instruction and an unsupported intrinsic
/// axis, and both spell `thread_id` and `barrier` differently enough that the
/// shared rejection lives here while each dialect decides what it does support.
///
/// A `thread_id` or a `barrier` is not rejected outright: the WGSL mapping
/// plants both in every parallel kernel. The input contract still rejects
/// them — the lowering never produces them, so one in the input means the
/// pipeline changed under the mapping — and [`parallel_contract`] accepts
/// them only where the mapping that introduces them also proved they pair.
fn verify_gpu_contract(lsir: &LsirFunction, stage: &str) -> Result<()> {
    for block in &lsir.blocks {
        for inst in &block.insts {
            match &inst.op {
                sile_lsir::InstOp::Intrinsic { intrinsic, .. } => match intrinsic {
                    sile_lsir::Intrinsic::ThreadId { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: GPU backend does not support thread_id intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::Barrier { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: GPU backend does not support barrier intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::BlockId { dim } if *dim > 2 => {
                        return Err(Error::Compile(format!(
                            "{stage}: GPU backend only supports block_id dimensions 0..=2, got {dim} (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::GridDim { dim } if *dim > 2 => {
                        return Err(Error::Compile(format!(
                            "{stage}: GPU backend only supports grid_dim dimensions 0..=2, got {dim} (block `{}`)",
                            block.name
                        )));
                    }
                    _ => {}
                },
                sile_lsir::InstOp::Call { func, .. } => {
                    return Err(Error::Compile(format!(
                        "{stage}: GPU backend does not support LSIR call instructions, got `{func}` (block `{}`)",
                        block.name
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// The contract a mapped parallel kernel owes its speller.
///
/// The mapping replaced the tile intrinsics and the vector relays with the
/// strips, folds and barriers that run them; whatever survives is something
/// the mapping declined to touch, and emitting it would spell the tile form
/// under a workgroup — a race, not a slowdown. Shared storage and barriers
/// must pair: a shared write without a barrier between the write and the
/// read across threads is exactly the race the mapping exists to order.
///
/// The SIMD-group intrinsics are the one mapped surface that varies by
/// target: only the Metal mapping plants them, so their appearance on any
/// other target's kernel is the same pipeline-changed shape a surviving tile
/// intrinsic is.
pub(crate) fn parallel_contract(
    lsir: &LsirFunction,
    target: CodegenTarget,
    stage: &str,
) -> Result<()> {
    let mut has_shared = false;
    let mut has_barrier = false;
    for block in &lsir.blocks {
        for inst in &block.insts {
            if let sile_lsir::InstOp::Alloca { addr_space, .. } = &inst.op
                && *addr_space == sile_lsir::AddressSpace::Shared
            {
                has_shared = true;
            }
            match &inst.op {
                sile_lsir::InstOp::Intrinsic { intrinsic, .. } => match intrinsic {
                    sile_lsir::Intrinsic::TileReduce { .. } | sile_lsir::Intrinsic::TileMma { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: a mapped parallel kernel must have its tile intrinsic materialized, got `{intrinsic:?}` (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::SimdReduce { .. }
                    | sile_lsir::Intrinsic::SimdLaneId
                    | sile_lsir::Intrinsic::SimdGroupId
                    | sile_lsir::Intrinsic::SimdGroupMma { .. }
                        if target != CodegenTarget::Metal =>
                    {
                        return Err(Error::Compile(format!(
                            "{stage}: only the Metal mapping plants the SIMD-group intrinsics, got `{intrinsic:?}` (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::VecLoad { .. } | sile_lsir::Intrinsic::VecStore { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: a mapped parallel kernel must have its vector relays scalarized (block `{}`)",
                            block.name
                        )));
                    }
                    sile_lsir::Intrinsic::Barrier { .. } => has_barrier = true,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    if has_shared != has_barrier {
        return Err(Error::Compile(format!(
            "{stage}: a mapped parallel kernel has {} but no {}, and the two only work paired",
            if has_shared { "shared storage" } else { "barriers" },
            if has_shared { "barriers" } else { "shared storage" },
        )));
    }
    Ok(())
}
