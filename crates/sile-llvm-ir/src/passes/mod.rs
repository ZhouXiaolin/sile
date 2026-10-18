pub mod canonicalize;
mod cfg;
pub mod cse;
pub mod instcombine;
pub mod licm;
pub mod loop_simplify;
pub mod loop_unroll;
pub mod phi_alias;
pub mod sroa;
pub mod verify;

use crate::Function;

/// The LLVM IR passes this pipeline can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmIrPassKind {
    VerifyInput,
    Canonicalize,
    LoopSimplify,
    PhiAlias,
    Cse,
    InstCombine,
    LoopUnroll,
    Licm,
    Sroa,
    Dce,
    VerifyOutput,
}

/// Planned LLVM IR pipeline.
///
/// A pass belongs here only while removing it changes the IR of at least one
/// real kernel. A pass is judged by removal, not by running it alone:
/// `Canonicalize` looks redundant on its own in one order and indispensable in
/// another, and only removal reports the truth.
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
    LlvmIrPassKind::InstCombine,
    LlvmIrPassKind::LoopUnroll,
    LlvmIrPassKind::InstCombine,
    LlvmIrPassKind::Licm,
    LlvmIrPassKind::Sroa,
    LlvmIrPassKind::Dce,
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
            LlvmIrPassKind::InstCombine => instcombine::run(func),
            LlvmIrPassKind::LoopUnroll => loop_unroll::run(func),
            LlvmIrPassKind::Licm => licm::run(func),
            LlvmIrPassKind::Sroa => sroa::run(func),
            LlvmIrPassKind::Dce => {
                canonicalize::eliminate_dead_insts(&mut func);
                func
            }
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: Function) -> Result<Function, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
