pub mod canonicalize_shape;
pub mod canonicalize_tile_ops;
pub mod dce;
pub mod lowering;
pub mod tile_expr_cse;
pub mod verify;

use crate::MirFunction;
pub use lowering::{
    ACTIVE_LLIR_LOWERING_PIPELINE, LlirLoweringPassKind, RECOMMENDED_LLIR_LOWERING_PIPELINE,
    lower_mir_to_llir, run_default_llir_lowering_pipeline, run_llir_lowering_pipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirPassKind {
    VerifyInput,
    CanonicalizeShape,
    CanonicalizeTileOps,
    TileExprCse,
    Dce,
    VerifyOutput,
}

/// Planned MIR pipeline.
///
/// MIR keeps tile/broadcast/reduce/mma semantics, so passes here should stay
/// semantic and shape-aware rather than reason about low-level CFG details.
pub const RECOMMENDED_PIPELINE: &[MirPassKind] = &[
    MirPassKind::VerifyInput,
    MirPassKind::CanonicalizeShape,
    MirPassKind::CanonicalizeTileOps,
    MirPassKind::TileExprCse,
    MirPassKind::Dce,
    MirPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[MirPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(
    mut func: MirFunction,
    pipeline: &[MirPassKind],
) -> Result<MirFunction, String> {
    for pass in pipeline {
        func = match pass {
            MirPassKind::VerifyInput | MirPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            MirPassKind::CanonicalizeShape => canonicalize_shape::run(func),
            MirPassKind::CanonicalizeTileOps => canonicalize_tile_ops::run(func),
            MirPassKind::TileExprCse => tile_expr_cse::run(func),
            MirPassKind::Dce => dce::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: MirFunction) -> Result<MirFunction, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
