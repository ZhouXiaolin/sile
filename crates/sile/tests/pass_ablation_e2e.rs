//! Which LLVM IR passes earn their place in the active pipeline?
//!
//! The judgment is removal, not single-pass reach: a pass that reshapes the CFG
//! without deleting instructions is still doing work, and a pass that is silent
//! on its own can still be the only thing holding the pipeline together.
//! `SimplifyCfg`, `AllocaSimplify` and `Dce` all failed this ablation and were
//! removed; the four that survived are asserted here.
//!
//! `crates/sile-llvm-ir/tests/pass_ablation.rs` proves the opposite property
//! with hand-built fixtures: that a pass can move its own canonical input.
//!
//! `sile-llvm-ir` cannot depend on `sile` (that would be a dependency cycle), so
//! this fixture-level ablation lives next to the kernels.

use sile::compiler::{
    ACTIVE_LLVM_IR_PIPELINE, LlvmIrPassKind, lower_to_tile_ir, lower_tile_ir_to_llvm_ir,
    run_llvm_ir_pipeline,
};
use sile::fixtures::{build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel};
use sile::typeck;

type Kernel = &'static sile::hir::Kernel;

fn fixtures() -> Vec<Kernel> {
    vec![
        Box::leak(Box::new(build_vec_add_kernel())),
        Box::leak(Box::new(build_softmax_kernel())),
        Box::leak(Box::new(build_dynamic_k_matmul_kernel())),
    ]
}

/// Passes that stay in the active pipeline: each must be unremovable.
///
/// Removal is the test that matters, not single-pass reach. `Dce` deleted nothing
/// on its own for `vec_add`, yet removing it from the pipeline left instructions
/// behind, because `Cse` had just created the duplicate it consumes. A pass can
/// be silent alone and still be load-bearing.
const WORKING: [LlvmIrPassKind; 4] = [
    LlvmIrPassKind::Canonicalize,
    LlvmIrPassKind::LoopSimplify,
    LlvmIrPassKind::PhiAlias,
    LlvmIrPassKind::Cse,
];

#[test]
fn every_working_pass_is_unremovable_from_the_pipeline() {
    for pass in WORKING {
        let mut costs = Vec::new();
        for kernel in fixtures() {
            let typed = typeck::check_kernel(kernel).expect("typecheck");
            let mk = || lower_tile_ir_to_llvm_ir(&lower_to_tile_ir(&typed), &typed);
            let full = run_llvm_ir_pipeline(mk(), ACTIVE_LLVM_IR_PIPELINE).expect("full");
            let trimmed: Vec<_> = ACTIVE_LLVM_IR_PIPELINE
                .iter()
                .copied()
                .filter(|kind| *kind != pass)
                .collect();
            let without = run_llvm_ir_pipeline(mk(), &trimmed).expect("pipeline without one pass");
            if without != full {
                costs.push(kernel.name.clone());
            }
        }
        assert!(
            !costs.is_empty(),
            "{pass:?} can be removed with no effect on any kernel: it no longer earns its place"
        );
    }
}

#[test]
fn the_pipeline_still_shrinks_every_kernel() {
    for kernel in fixtures() {
        let typed = typeck::check_kernel(kernel).expect("typecheck");
        let mk = || lower_tile_ir_to_llvm_ir(&lower_to_tile_ir(&typed), &typed);
        let raw = mk();
        let optimized =
            run_llvm_ir_pipeline(mk(), ACTIVE_LLVM_IR_PIPELINE).expect("active pipeline");
        let count = |f: &sile::llvmir::Function| f.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
        assert!(
            count(&optimized) < count(&raw),
            "{}: the active pipeline removed nothing, so it no longer earns its cost",
            kernel.name
        );
    }
}
