pub mod ir;
pub mod lower;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_tile_ir;
pub use passes::{
    ACTIVE_PIPELINE as ACTIVE_TILE_IR_PIPELINE, RECOMMENDED_PIPELINE as RECOMMENDED_TILE_IR_PIPELINE,
    TileIrPassKind, run_default_pipeline as run_tile_ir_passes, run_pipeline as run_tile_ir_pipeline,
};
pub use print::format_tile_ir;
