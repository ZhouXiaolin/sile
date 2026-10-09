//! Guard the "no-op on real kernels" decision for the inactive LLVM IR passes.
//!
//! `sile-llvm-ir` cannot depend on `sile` (that would be a dependency cycle), so
//! this fixture-level ablation lives next to the kernels instead.

use sile::compiler::{
    ACTIVE_LLVM_IR_PIPELINE, LlvmIrPassKind, lower_to_tile_ir, lower_tile_ir_to_llvm_ir,
    run_llvm_ir_pipeline,
};
use sile::fixtures::{build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel};
use sile::typeck;

#[test]
fn licm_and_inline_preheader_stay_no_ops_on_the_kernel_fixtures() {
    for kernel in [
        build_vec_add_kernel(),
        build_softmax_kernel(),
        build_dynamic_k_matmul_kernel(),
    ] {
        let typed = typeck::check_kernel(&kernel).expect("typecheck");
        let tile_ir = lower_to_tile_ir(&typed);
        let base = lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
        let baseline =
            run_llvm_ir_pipeline(base, ACTIVE_LLVM_IR_PIPELINE).expect("active pipeline");

        for (label, extra) in [
            ("licm", vec![LlvmIrPassKind::Licm]),
            (
                "inline_preheader",
                vec![LlvmIrPassKind::InlinePreheaderConstants],
            ),
            (
                "loop_induction",
                vec![LlvmIrPassKind::LoopInductionCanonicalize],
            ),
        ] {
            let mut pipeline: Vec<_> = ACTIVE_LLVM_IR_PIPELINE
                .iter()
                .copied()
                .filter(|kind| *kind != LlvmIrPassKind::VerifyOutput)
                .collect();
            pipeline.extend(extra);
            pipeline.push(LlvmIrPassKind::VerifyOutput);
            let with_extra = run_llvm_ir_pipeline(
                lower_tile_ir_to_llvm_ir(&lower_to_tile_ir(&typed), &typed),
                &pipeline,
            )
            .expect("pipeline with extra pass");
            assert_eq!(
                baseline, with_extra,
                "{label} now changes {}: the pass may belong back in ACTIVE_LLVM_IR_PIPELINE",
                kernel.name
            );
        }
    }
}


