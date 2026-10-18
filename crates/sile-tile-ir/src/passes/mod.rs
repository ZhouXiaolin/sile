pub mod dce;
pub mod pointwise_fusion;
pub mod tile_expr_cse;
pub mod verify;

use crate::TileIrFunction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIrPassKind {
    VerifyInput,
    PointwiseFusion,
    TileExprCse,
    Dce,
    VerifyOutput,
}

/// Planned Tile IR pipeline.
///
/// A pass belongs here only while it changes the IR of at least one real kernel.
/// Removal is what measures that, on the printed IR,
/// so a pass that turns into a no-op fails the suite and gets removed.
///
/// Tile IR keeps tile/broadcast/reduce/mma semantics, so passes here stay
/// semantic and shape-aware rather than reason about low-level CFG details.
pub const RECOMMENDED_PIPELINE: &[TileIrPassKind] = &[
    TileIrPassKind::VerifyInput,
    TileIrPassKind::PointwiseFusion,
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
            TileIrPassKind::PointwiseFusion => pointwise_fusion::run(func),
            TileIrPassKind::TileExprCse => tile_expr_cse::run(func),
            TileIrPassKind::Dce => dce::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: TileIrFunction) -> Result<TileIrFunction, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
