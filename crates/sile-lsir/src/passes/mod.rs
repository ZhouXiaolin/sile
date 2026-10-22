pub mod canonicalize;
pub mod cfg;
pub mod cse;
pub mod instcombine;
pub mod licm;
pub mod loop_simplify;
pub mod loop_unroll;
pub mod phi_alias;
pub mod scalar_promote;
pub mod sroa;
pub mod storage;
pub mod verify;

use crate::Function;

/// The LSIR passes this pipeline can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LsirPassKind {
    VerifyInput,
    Canonicalize,
    LoopSimplify,
    PhiAlias,
    Cse,
    InstCombine,
    LoopUnroll,
    Licm,
    Sroa,
    ScalarPromote,
    Dce,
    VerifyOutput,
}

/// Planned LSIR pipeline.
///
/// A pass belongs here only while removing it changes the IR of at least one
/// real kernel. A pass is judged by removal, not by running it alone:
/// `Canonicalize` looks redundant on its own in one order and indispensable in
/// another, and only removal reports the truth.
///
/// LSIR passes work on explicit SSA/CFG/memory semantics and stay
/// backend-independent. Target-specific legalizations run after this stage in
/// backend-owned code.
pub const RECOMMENDED_PIPELINE: &[LsirPassKind] = &[
    LsirPassKind::VerifyInput,
    LsirPassKind::Canonicalize,
    LsirPassKind::LoopSimplify,
    LsirPassKind::PhiAlias,
    LsirPassKind::Cse,
    LsirPassKind::InstCombine,
    LsirPassKind::LoopUnroll,
    LsirPassKind::InstCombine,
    LsirPassKind::Licm,
    LsirPassKind::Sroa,
    LsirPassKind::ScalarPromote,
    LsirPassKind::Dce,
    LsirPassKind::Cse,
    LsirPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[LsirPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(mut func: Function, pipeline: &[LsirPassKind]) -> Result<Function, String> {
    for pass in pipeline {
        func = match pass {
            LsirPassKind::VerifyInput | LsirPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            LsirPassKind::Canonicalize => canonicalize::run(func),
            LsirPassKind::LoopSimplify => loop_simplify::run(func),
            LsirPassKind::PhiAlias => phi_alias::run(func),
            LsirPassKind::Cse => cse::run(func),
            LsirPassKind::InstCombine => instcombine::run(func),
            LsirPassKind::LoopUnroll => loop_unroll::run(func),
            LsirPassKind::Licm => licm::run(func),
            LsirPassKind::Sroa => sroa::run(func),
            LsirPassKind::ScalarPromote => scalar_promote::run(func),
            LsirPassKind::Dce => {
                // Dead instructions are only the tail of it: the relay copies
                // between tile storages die as whole allocations.
                storage::run(&mut func);
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
