pub mod dce;
pub mod llir_plan;

use crate::MirFunction;
pub use llir_plan::{LlirLoweringPlan, build_llir_lowering_plan};

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
pub const ACTIVE_PIPELINE: &[MirPassKind] = &[MirPassKind::Dce];

pub fn run_pipeline(mut func: MirFunction, pipeline: &[MirPassKind]) -> MirFunction {
    for pass in pipeline {
        func = match pass {
            MirPassKind::Dce => dce::run(func),
            MirPassKind::VerifyInput
            | MirPassKind::CanonicalizeShape
            | MirPassKind::CanonicalizeTileOps
            | MirPassKind::TileExprCse
            | MirPassKind::VerifyOutput => func,
        };
    }
    func
}

pub fn run_default_pipeline(func: MirFunction) -> MirFunction {
    run_pipeline(func, ACTIVE_PIPELINE)
}
