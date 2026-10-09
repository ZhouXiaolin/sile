//! Ablation report: how much each pipeline stage changes the generated code.
//!
//! For each kernel we print the LLVM IR and generated C at every stage of the
//! LLVM IR pipeline, plus instruction/alloca/block counts, so the cost and
//! benefit of each pass can be compared directly.

use sile::codegen::llvmir_c;
use sile::compiler::{
    self, lower_to_tile_ir, lower_tile_ir_to_llvm_ir, run_llvm_ir_pipeline,
};
use sile::hir::Kernel;
use sile::typeck;

fn main() {
    for kernel in kernels() {
        let typed = typeck::check_kernel(&kernel).expect("typecheck");
        let tile_ir = lower_to_tile_ir(&typed);
        let tile_ir = sile::compiler::run_tile_ir_pipeline(
            tile_ir,
            sile::compiler::ACTIVE_TILE_IR_PIPELINE,
        )
        .expect("tile ir pipeline");
        let base = lower_tile_ir_to_llvm_ir(&tile_ir, &typed);

        println!("=== {} ===", kernel.name);
        report("lowered", &base);

        let mut current = base.clone();
        for kind in compiler::ACTIVE_LLVM_IR_PIPELINE {
            let next = match run_llvm_ir_pipeline(current.clone(), &[*kind]) {
                Ok(f) => f,
                Err(e) => {
                    println!("  {kind:?}: ERROR {e}");
                    break;
                }
            };
            report(&format!("{kind:?}"), &next);
            current = next;
        }

        // Full pipeline result.
        match run_llvm_ir_pipeline(base.clone(), compiler::ACTIVE_LLVM_IR_PIPELINE) {
            Ok(final_ir) => match llvmir_c::generate(&final_ir) {
                Ok(c) => {
                    println!("  final C: {} lines", c.lines().count());
                    println!("  C allocas: {}", c.matches("int64_t v").count());
                    println!("  C nodes: {}", count_nodes(&c));
                }
                Err(e) => println!("  C codegen ERROR: {e}"),
            },
            Err(e) => println!("  pipeline ERROR: {e}"),
        }
    }
}

fn report(label: &str, func: &sile::llvmir::Function) {
    let insts: usize = func.blocks.iter().map(|b| b.insts.len()).sum();
    let allocas = func
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i.op, sile::llvmir::InstOp::Alloca { .. }))
        .count();
    let loops = func
        .blocks
        .iter()
        .filter(|b| matches!(b.terminator, sile::llvmir::Terminator::CondBr { .. }))
        .count();
    println!(
        "  {label:<26} blocks={:<3} insts={:<4} allocas={:<3} condbr={:<3}",
        func.blocks.len(),
        insts,
        allocas,
        loops
    );
}

/// Count distinct operations emitted in the C source as a rough complexity proxy.
fn count_nodes(c: &str) -> usize {
    c.lines().filter(|l| l.contains("=")).count()
}

fn kernels() -> Vec<Kernel> {
    vec![
        sile::fixtures::build_vec_add_kernel(),
        sile::fixtures::build_softmax_kernel(),
        sile::fixtures::build_dynamic_k_matmul_kernel(),
    ]
}
