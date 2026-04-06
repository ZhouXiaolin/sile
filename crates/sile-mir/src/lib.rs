pub mod ir;
pub mod lower;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_mir;
pub use passes::{
    ACTIVE_LLIR_LOWERING_PIPELINE, LlirLoweringPassKind, RECOMMENDED_LLIR_LOWERING_PIPELINE, dce,
    lower_mir_to_llir, run_default_llir_lowering_pipeline, run_llir_lowering_pipeline,
};
