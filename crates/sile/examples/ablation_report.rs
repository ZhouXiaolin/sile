//! Report what each pass costs, so a retirement is a decision and not a guess.
//!
//! Runs the active pipeline on every kernel fixture and reports, for each
//! pass, what removing it changes: the IR, the instruction count, and whether
// any of the three targets still produces the same arithmetic.

use sile::compiler::{
    ACTIVE_LLVM_IR_PIPELINE, ACTIVE_TILE_IR_PIPELINE, CodegenTarget, LlvmIrPassKind,
    TileIrPassKind, lower_to_tile_ir, lower_tile_ir_to_llvm_ir, run_llvm_ir_pipeline,
    run_tile_ir_pipeline,
};
use sile::fixtures::{build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel};
use sile::typeck;

fn kernels() -> Vec<&'static sile::hir::Kernel> {
    vec![
        Box::leak(Box::new(build_vec_add_kernel())),
        Box::leak(Box::new(build_softmax_kernel())),
        Box::leak(Box::new(build_dynamic_k_matmul_kernel())),
    ]
}

fn main() {
    println!("== Tile IR pipeline: what removal changes ==");
    let working: Vec<TileIrPassKind> = ACTIVE_TILE_IR_PIPELINE
        .iter()
        .copied()
        .filter(|k| !matches!(k, TileIrPassKind::VerifyInput | TileIrPassKind::VerifyOutput))
        .collect();
    for pass in &working {
        let mut changed = Vec::new();
        for kernel in kernels() {
            let typed = typeck::check_kernel(kernel).unwrap();
            let mk = || lower_to_tile_ir(&typed);
            let base = sile::tileir::format_tile_ir(&run_tile_ir_pipeline(mk(), ACTIVE_TILE_IR_PIPELINE).unwrap());
            let trimmed: Vec<_> = ACTIVE_TILE_IR_PIPELINE
                .iter()
                .copied()
                .filter(|k| k != pass)
                .collect();
            let without = sile::tileir::format_tile_ir(&run_tile_ir_pipeline(mk(), &trimmed).unwrap());
            if without != base {
                changed.push(format!("{} ({} -> {} lines)", kernel.name, base.lines().count(), without.lines().count()));
            }
        }
        println!("{pass:?}: {}", if changed.is_empty() { "REMOVABLE".to_string() } else { changed.join(", ") });
    }

    println!();
    println!("== LLVM IR pipeline: what removal changes ==");
    let working: Vec<LlvmIrPassKind> = ACTIVE_LLVM_IR_PIPELINE
        .iter()
        .copied()
        .filter(|k| !matches!(k, LlvmIrPassKind::VerifyInput | LlvmIrPassKind::VerifyOutput))
        .collect();
    for pass in &working {
        let mut changed = Vec::new();
        for kernel in kernels() {
            let typed = typeck::check_kernel(kernel).unwrap();
            let mk = || lower_tile_ir_to_llvm_ir(&lower_to_tile_ir(&typed), &typed);
            let full = run_llvm_ir_pipeline(mk(), ACTIVE_LLVM_IR_PIPELINE).unwrap();
            let trimmed: Vec<_> = ACTIVE_LLVM_IR_PIPELINE
                .iter()
                .copied()
                .filter(|k| k != pass)
                .collect();
            let without = run_llvm_ir_pipeline(mk(), &trimmed).unwrap();
            if without != full {
                let count = |f: &sile::llvmir::Function| f.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                let blocks = |f: &sile::llvmir::Function| f.blocks.len();
                changed.push(format!(
                    "{} ({} -> {} insts, {} -> {} blocks)",
                    kernel.name,
                    count(&without),
                    count(&full),
                    blocks(&without),
                    blocks(&full),
                ));
            }
        }
        println!("{pass:?}: {}", if changed.is_empty() { "REMOVABLE".to_string() } else { changed.join(", ") });
    }

    println!();
    println!("== Backend text after removing each pass: does any target change? ==");
    for pass in &working {
        let mut changes = Vec::new();
        for kernel in kernels() {
            let typed = typeck::check_kernel(kernel).unwrap();
            let let_llir = |pipeline: &[LlvmIrPassKind]| {
                let tile_ir = lower_to_tile_ir(&typed);
                lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
                run_llvm_ir_pipeline(
                    lower_tile_ir_to_llvm_ir(&lower_to_tile_ir(&typed), &typed),
                    pipeline,
                )
                .unwrap()
            };
            let full = let_llir(ACTIVE_LLVM_IR_PIPELINE);
            let trimmed: Vec<_> = ACTIVE_LLVM_IR_PIPELINE
                .iter()
                .copied()
                .filter(|k| k != pass)
                .collect();
            let without = let_llir(&trimmed);
            let mut per_target = Vec::new();
            for (name, text) in [
                ("c", sile::codegen::llvmir_c::generate(&without).ok()),
                ("metal", sile::codegen::llvmir_metal::generate(&without).ok()),
                ("wgsl", sile::codegen::llvmir_wgsl::generate(&without).ok()),
            ] {
                let full_text = match name {
                    "c" => sile::codegen::llvmir_c::generate(&full).ok(),
                    "metal" => sile::codegen::llvmir_metal::generate(&full).ok(),
                    _ => sile::codegen::llvmir_wgsl::generate(&full).ok(),
                };
                if text != full_text {
                    per_target.push(name.to_string());
                }
            }
            if !per_target.is_empty() {
                changes.push(format!("{} via {}", kernel.name, per_target.join("+")));
            }
        }
        println!(
            "{pass:?}: {}",
            if changes.is_empty() {
                "no target changes its text".to_string()
            } else {
                changes.join(", ")
            }
        );
    }

    println!();
    println!("== Backend text: same IR, three targets ==");
    for kernel in kernels() {
        let typed = typeck::check_kernel(kernel).unwrap();
        let tile_ir = run_tile_ir_pipeline(lower_to_tile_ir(&typed), ACTIVE_TILE_IR_PIPELINE).unwrap();
        let llir = run_llvm_ir_pipeline(lower_tile_ir_to_llvm_ir(&tile_ir, &typed), ACTIVE_LLVM_IR_PIPELINE).unwrap();
        let mut line = format!("{}: ", kernel.name);
        for target in [CodegenTarget::C, CodegenTarget::Metal, CodegenTarget::Wgsl] {
            let text = sile::codegen::llvmir_c::generate(&llir)
                .ok();
            let _ = target;
            let _ = text;
            match target {
                CodegenTarget::C => line.push_str(&format!("c={} ", sile::codegen::llvmir_c::generate(&llir).map(|t| t.lines().count()).unwrap_or(0))),
                CodegenTarget::Metal => line.push_str(&format!("metal={} ", sile::codegen::llvmir_metal::generate(&llir).map(|t| t.lines().count()).unwrap_or(0))),
                CodegenTarget::Wgsl => line.push_str(&format!("wgsl={} ", sile::codegen::llvmir_wgsl::generate(&llir).map(|t| t.lines().count()).unwrap_or(0))),
            }
        }
        println!("{line}");
    }
}
