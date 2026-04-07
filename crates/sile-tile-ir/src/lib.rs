pub mod ir;
pub mod lower;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_tile_ir;
pub use passes::{
    ACTIVE_LLVM_IR_LOWERING_PIPELINE, ACTIVE_PIPELINE as ACTIVE_TILE_IR_PIPELINE,
    LlvmIrLoweringPassKind, RECOMMENDED_LLVM_IR_LOWERING_PIPELINE,
    RECOMMENDED_PIPELINE as RECOMMENDED_TILE_IR_PIPELINE, TileIrPassKind, dce,
    lower_tile_ir_to_llvm_ir, run_default_llvm_ir_lowering_pipeline,
    run_default_pipeline as run_tile_ir_passes, run_llvm_ir_lowering_pipeline,
    run_pipeline as run_tile_ir_pipeline,
};
pub use print::format_tile_ir;
