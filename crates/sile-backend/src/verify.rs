use sile_core::{Error, Result};
use sile_llvm_ir::Function as LlvmIrFunction;

use crate::CodegenTarget;

pub(crate) fn for_target(
    llvm_ir: &LlvmIrFunction,
    target: CodegenTarget,
    stage: &str,
) -> Result<()> {
    verify_shared_contract(llvm_ir, stage)?;
    match target {
        CodegenTarget::C => Ok(()),
        #[cfg(target_os = "macos")]
        CodegenTarget::Metal => verify_metal_contract(llvm_ir, stage),
        #[cfg(not(target_os = "macos"))]
        CodegenTarget::Metal => Ok(()),
    }
}

fn verify_shared_contract(llvm_ir: &LlvmIrFunction, stage: &str) -> Result<()> {
    for block in &llvm_ir.blocks {
        if matches!(block.terminator, sile_llvm_ir::Terminator::Switch { .. }) {
            return Err(Error::Compile(format!(
                "{stage}: backend emit does not support switch terminators (block `{}`)",
                block.name
            )));
        }
        for inst in &block.insts {
            if let sile_llvm_ir::InstOp::Call { func, .. } = &inst.op
                && func.starts_with("tile_")
            {
                return Err(Error::Compile(format!(
                    "{stage}: backend emit does not allow tile helper calls in LLVM IR, got `{func}` (block `{}`)",
                    block.name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn verify_metal_contract(llvm_ir: &LlvmIrFunction, stage: &str) -> Result<()> {
    for block in &llvm_ir.blocks {
        for inst in &block.insts {
            match &inst.op {
                sile_llvm_ir::InstOp::Intrinsic { intrinsic, .. } => match intrinsic {
                    sile_llvm_ir::Intrinsic::ThreadId { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend does not support thread_id intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_llvm_ir::Intrinsic::Barrier { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend does not support barrier intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_llvm_ir::Intrinsic::BlockId { dim } if *dim > 2 => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend only supports block_id dimensions 0..=2, got {dim} (block `{}`)",
                            block.name
                        )));
                    }
                    _ => {}
                },
                sile_llvm_ir::InstOp::Call { func, .. } => {
                    return Err(Error::Compile(format!(
                        "{stage}: Metal backend does not support LLVM IR call instructions, got `{func}` (block `{}`)",
                        block.name
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(())
}
