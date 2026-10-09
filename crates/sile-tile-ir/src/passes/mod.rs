pub mod lowering;
pub mod dce;
pub mod pointwise_fusion;
pub mod tile_expr_cse;
pub mod verify;

use crate::TileIrFunction;

/// A retired pass must be deleted, not left in the tree.
///
/// A `.rs` file with no `mod` declaration does not compile, so no warning and
/// no test can see it. This alarm catches that drift instead of relying on
/// someone remembering to delete the file.
#[cfg(test)]
mod no_orphan_files {
    #[test]
    fn every_rs_file_in_passes_is_declared() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/passes");
        let declared = std::fs::read_to_string(dir.join("mod.rs"))
            .expect("read src/passes/mod.rs");
        let mut orphans = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("read src/passes directory") {
            let path = entry.expect("dir entry").path();
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if stem == "mod" {
                continue;
            }
            let wants = format!("pub mod {stem};");
            if !declared.contains(&wants) {
                orphans.push(format!("{stem}.rs is not declared as `{wants}`"));
            }
        }
        assert!(
            orphans.is_empty(),
            "these files are not compiled, so nothing else can see them:\n{}",
            orphans.join("\n")
        );
    }
}
pub use lowering::{
    ACTIVE_LLVM_IR_LOWERING_PIPELINE, LlvmIrLoweringPassKind,
    RECOMMENDED_LLVM_IR_LOWERING_PIPELINE, lower_tile_ir_to_llvm_ir,
    run_default_llvm_ir_lowering_pipeline, run_llvm_ir_lowering_pipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIrPassKind {
    VerifyInput,
    PointwiseFusion,
    TileExprCse,
    Dce,
    VerifyOutput,
}

/// Planned Tile IR pipeline.
///
/// A pass belongs here only while it changes the IR of at least one real kernel.
/// `crates/sile/tests/tile_ir_pass_ablation.rs` measures that on the printed IR,
/// so a pass that turns into a no-op fails the suite and gets removed.
///
/// Tile IR keeps tile/broadcast/reduce/mma semantics, so passes here stay
/// semantic and shape-aware rather than reason about low-level CFG details.
pub const RECOMMENDED_PIPELINE: &[TileIrPassKind] = &[
    TileIrPassKind::VerifyInput,
    TileIrPassKind::PointwiseFusion,
    TileIrPassKind::TileExprCse,
    TileIrPassKind::Dce,
    TileIrPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[TileIrPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(
    mut func: TileIrFunction,
    pipeline: &[TileIrPassKind],
) -> Result<TileIrFunction, String> {
    for pass in pipeline {
        func = match pass {
            TileIrPassKind::VerifyInput | TileIrPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            TileIrPassKind::PointwiseFusion => pointwise_fusion::run(func),
            TileIrPassKind::TileExprCse => tile_expr_cse::run(func),
            TileIrPassKind::Dce => dce::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: TileIrFunction) -> Result<TileIrFunction, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
