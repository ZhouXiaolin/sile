use sile_core::{Error, Result};
use sile_hir::typeck::TypedKernel;
use sile_llir::Function as LlirFunction;
use sile_mir::MirFunction;

pub use sile_llir::{
    ACTIVE_LLIR_PIPELINE, LlirPassKind, RECOMMENDED_LLIR_PIPELINE, run_llir_passes,
    run_llir_pipeline,
};
pub use sile_mir::passes::{
    ACTIVE_PIPELINE as ACTIVE_MIR_PIPELINE, MirPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_MIR_PIPELINE, build_llir_lowering_plan,
    run_default_pipeline as run_mir_passes, run_pipeline as run_mir_pipeline,
};
pub use sile_mir::{
    dce, lower_mir_to_llir, lower_mir_to_llir_raw, lower_mir_to_llir_raw_with_plan, lower_to_mir,
};

pub fn compile_to_llir(typed: &TypedKernel) -> Result<(MirFunction, LlirFunction)> {
    let mir = run_mir_passes(lower_to_mir(typed));
    let llir_plan = build_llir_lowering_plan(&mir);
    let llir = lower_mir_to_llir_raw_with_plan(&mir, typed, &llir_plan);
    let llir = run_llir_passes(llir).map_err(Error::Shape)?;
    Ok((mir, llir))
}
