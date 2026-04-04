pub mod canonicalize;
pub mod verify;

use crate::Function;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlirPassKind {
    VerifyInput,
    Canonicalize,
    SimplifyCfg,
    Cse,
    Dce,
    LoopSimplify,
    VerifyOutput,
}

/// Planned LLIR pipeline.
///
/// LLIR passes should work on explicit SSA/CFG/memory semantics and remain
/// backend-independent. Target-specific legalizations should run after this
/// stage in backend-owned code.
pub const RECOMMENDED_PIPELINE: &[LlirPassKind] = &[
    LlirPassKind::VerifyInput,
    LlirPassKind::Canonicalize,
    LlirPassKind::SimplifyCfg,
    LlirPassKind::Cse,
    LlirPassKind::Dce,
    LlirPassKind::LoopSimplify,
    LlirPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[LlirPassKind] = &[
    LlirPassKind::VerifyInput,
    LlirPassKind::Canonicalize,
    LlirPassKind::VerifyOutput,
];

pub fn run_pipeline(mut func: Function, pipeline: &[LlirPassKind]) -> Result<Function, String> {
    for pass in pipeline {
        func = match pass {
            LlirPassKind::VerifyInput | LlirPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            LlirPassKind::Canonicalize => canonicalize::run(func),
            LlirPassKind::SimplifyCfg
            | LlirPassKind::Cse
            | LlirPassKind::Dce
            | LlirPassKind::LoopSimplify => func,
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: Function) -> Result<Function, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
