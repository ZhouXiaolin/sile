pub mod alloca_simplify;
pub mod canonicalize;
pub mod cse;
pub mod dce;
pub mod licm;
pub mod loop_induction_canonicalize;
pub mod loop_simplify;
pub mod phi_alias;
pub mod simplify_cfg;
pub mod verify;

use crate::Function;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmIrPassKind {
    VerifyInput,
    Canonicalize,
    SimplifyCfg,
    LoopSimplify,
    PhiAlias,
    LoopInductionCanonicalize,
    Licm,
    Cse,
    Dce,
    AllocaSimplify,
    VerifyOutput,
}

/// Planned LLVM IR pipeline.
///
/// LLVM IR passes should work on explicit SSA/CFG/memory semantics and remain
/// backend-independent. Target-specific legalizations should run after this
/// stage in backend-owned code.
pub const RECOMMENDED_PIPELINE: &[LlvmIrPassKind] = &[
    LlvmIrPassKind::VerifyInput,
    LlvmIrPassKind::Canonicalize,
    LlvmIrPassKind::SimplifyCfg,
    LlvmIrPassKind::LoopSimplify,
    LlvmIrPassKind::PhiAlias,
    LlvmIrPassKind::LoopInductionCanonicalize,
    LlvmIrPassKind::Licm,
    LlvmIrPassKind::Cse,
    LlvmIrPassKind::Dce,
    LlvmIrPassKind::AllocaSimplify,
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
            LlvmIrPassKind::SimplifyCfg => simplify_cfg::run(func),
            LlvmIrPassKind::LoopSimplify => loop_simplify::run(func),
            LlvmIrPassKind::PhiAlias => phi_alias::run(func),
            LlvmIrPassKind::LoopInductionCanonicalize => loop_induction_canonicalize::run(func),
            LlvmIrPassKind::Licm => licm::run(func),
            LlvmIrPassKind::Cse => cse::run(func),
            LlvmIrPassKind::Dce => dce::run(func),
            LlvmIrPassKind::AllocaSimplify => alloca_simplify::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: Function) -> Result<Function, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
