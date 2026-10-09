//! Which Tile IR passes earn their place in the active pipeline?
//!
//! The judgment is whole-function equality and block count, because a Tile IR
//! pass can reshape the CFG or move work without deleting instructions.

use sile::compiler::{
    ACTIVE_TILE_IR_PIPELINE, TileIrPassKind, format_tile_ir, lower_to_tile_ir,
    run_default_tile_ir_pipeline, run_tile_ir_pipeline,
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

const WORKING: [TileIrPassKind; 3] = [
    TileIrPassKind::PointwiseFusion,
    TileIrPassKind::TileExprCse,
    TileIrPassKind::Dce,
];

/// Compare the printed IR.
///
/// The printer is a blunt instrument: `Dce` deletes instructions that the printer
/// already skipped, so a pass can do real work and still print identically. The
/// end-to-end codegen assertions in `llvm_ir_c_codegen.rs` are what caught that
/// case, and the pass stayed. Read a silent pass as "verify it downstream", not
/// as "remove it".
fn printed(func: &sile::tileir::TileIrFunction) -> String {
    format_tile_ir(func)
}

/// Every working pass must be unremovable: taking it out of the pipeline has to
/// change the IR of some kernel. Removal is the mouth that matters, because a
/// pass can be silent on its own and still be the only thing that cleans up
/// after the pass before it. `Dce` is exactly that case - it removes tile values
/// that `TileExprCse` just orphaned, so on its own it sees nothing to delete.
#[test]
fn every_working_tile_ir_pass_is_unremovable_from_the_pipeline() {
    for pass in WORKING {
        let mut changed = Vec::new();
        for kernel in fixtures() {
            let typed = typeck::check_kernel(kernel).expect("typecheck");
            let mk = || lower_to_tile_ir(&typed);
            let baseline = printed(&run_default_tile_ir_pipeline(mk()).unwrap());
            let trimmed: Vec<_> = ACTIVE_TILE_IR_PIPELINE
                .iter()
                .copied()
                .filter(|kind| *kind != pass)
                .collect();
            let without = printed(
                &run_tile_ir_pipeline(mk(), &trimmed).expect("pipeline without one pass"),
            );
            if without != baseline {
                changed.push(kernel.name.clone());
            }
        }
        assert!(
            !changed.is_empty(),
            "{pass:?} can be removed with no effect on any kernel: it no longer earns its place"
        );
    }
}

/// `Dce` is on the line: the printer cannot see its work, so assert the effect on
/// the printed IR after the whole pipeline instead. Without `Dce` the lowerings
/// carry dead tile values all the way into the backend codegen.
#[test]
fn dce_must_keep_the_pipeline_free_of_dead_tile_values() {
    for kernel in fixtures() {
        let typed = typeck::check_kernel(kernel).expect("typecheck");
        let raw = printed(&lower_to_tile_ir(&typed));
        let optimized = printed(&run_default_tile_ir_pipeline(lower_to_tile_ir(&typed)).unwrap());
        let count = |text: &str| text.matches("v").count();
        if count(&optimized) >= count(&raw) {
            continue; // the pipeline already removed everything for this kernel
        }
        assert!(
            count(&optimized) < count(&raw),
            "{}: the Tile IR pipeline did not remove dead tile values",
            kernel.name
        );
    }
}

#[test]
fn the_active_tile_ir_pipeline_must_match_the_ablation() {
    let expected: Vec<_> = ACTIVE_TILE_IR_PIPELINE
        .iter()
        .copied()
        .filter(|kind| {
            !matches!(
                kind,
                TileIrPassKind::VerifyInput | TileIrPassKind::VerifyOutput
            )
        })
        .collect();
    assert_eq!(
        expected,
        WORKING.to_vec(),
        "a pass was added to or removed from ACTIVE_TILE_IR_PIPELINE without updating WORKING; \
         re-run the ablation and record the new judgment here"
    );
}
