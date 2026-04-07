pub mod canonicalize_shape;
pub mod canonicalize_tile_ops;
pub mod dce;
pub mod lowering;
pub mod tile_expr_cse;
pub mod verify;

use crate::TileIrFunction;
pub use lowering::{
    ACTIVE_LLVM_IR_LOWERING_PIPELINE, LlvmIrLoweringPassKind,
    RECOMMENDED_LLVM_IR_LOWERING_PIPELINE, lower_tile_ir_to_llvm_ir,
    run_default_llvm_ir_lowering_pipeline, run_llvm_ir_lowering_pipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIrPassKind {
    VerifyInput,
    CanonicalizeShape,
    CanonicalizeTileOps,
    TileExprCse,
    Dce,
    VerifyOutput,
}

/// Planned Tile IR pipeline.
///
/// Tile IR keeps tile/broadcast/reduce/mma semantics, so passes here should stay
/// semantic and shape-aware rather than reason about low-level CFG details.
pub const RECOMMENDED_PIPELINE: &[TileIrPassKind] = &[
    TileIrPassKind::VerifyInput,
    TileIrPassKind::CanonicalizeShape,
    TileIrPassKind::CanonicalizeTileOps,
    TileIrPassKind::TileExprCse,
    TileIrPassKind::Dce,
    TileIrPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[TileIrPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(
    mut func: TileIrFunction,
    pipeline: &[TileIrPassKind],
) -> Result<TileIrFunction, String> {
    for pass in pipeline {
        func = match pass {
            TileIrPassKind::VerifyInput | TileIrPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            TileIrPassKind::CanonicalizeShape => canonicalize_shape::run(func),
            TileIrPassKind::CanonicalizeTileOps => canonicalize_tile_ops::run(func),
            TileIrPassKind::TileExprCse => tile_expr_cse::run(func),
            TileIrPassKind::Dce => dce::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: TileIrFunction) -> Result<TileIrFunction, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
