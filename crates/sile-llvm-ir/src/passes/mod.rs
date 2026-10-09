pub mod canonicalize;
pub mod cse;
pub mod loop_simplify;
pub mod phi_alias;
pub mod verify;

use crate::Function;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmIrPassKind {
    VerifyInput,
    Canonicalize,
    LoopSimplify,
    PhiAlias,
    Cse,
    VerifyOutput,
}

/// Planned LLVM IR pipeline.
///
/// A pass belongs here only while removing it changes the IR of at least one real
/// kernel. `crates/sile/tests/pass_ablation_e2e.rs` measures exactly that, so a
/// pass that turns into ballast fails the suite and gets removed. Note the test
/// removes passes rather than running them alone: `Canonicalize` looks redundant
/// on its own in one order and indispensable in another, and only removal
/// reports the truth.
///
/// LLVM IR passes work on explicit SSA/CFG/memory semantics and stay
/// backend-independent. Target-specific legalizations run after this stage in
/// backend-owned code.
pub const RECOMMENDED_PIPELINE: &[LlvmIrPassKind] = &[
    LlvmIrPassKind::VerifyInput,
    LlvmIrPassKind::Canonicalize,
    LlvmIrPassKind::LoopSimplify,
    LlvmIrPassKind::PhiAlias,
    LlvmIrPassKind::Cse,
    LlvmIrPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[LlvmIrPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(mut func: Function, pipeline: &[LlvmIrPassKind]) -> Result<Function, String> {
    for pass in pipeline {
        func = match pass {
            LlvmIrPassKind::VerifyInput | LlvmIrPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            LlvmIrPassKind::Canonicalize => canonicalize::run(func),
            LlvmIrPassKind::LoopSimplify => loop_simplify::run(func),
            LlvmIrPassKind::PhiAlias => phi_alias::run(func),
            LlvmIrPassKind::Cse => cse::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: Function) -> Result<Function, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
