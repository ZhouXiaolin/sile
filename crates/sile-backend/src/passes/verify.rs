use sile_core::{Error, Result};
use sile_llir::Function as LlirFunction;

use crate::CodegenTarget;

pub fn run(llir: &LlirFunction, stage: &str) -> Result<()> {
    verify_shared_backend_contract(llir, stage)
}

pub fn run_for_target(llir: &LlirFunction, target: CodegenTarget, stage: &str) -> Result<()> {
    run(llir, stage)?;
    match target {
        CodegenTarget::C => Ok(()),
        CodegenTarget::Metal => verify_metal_backend_contract(llir, stage),
    }
}

fn verify_shared_backend_contract(llir: &LlirFunction, stage: &str) -> Result<()> {
    for block in &llir.blocks {
        if matches!(block.terminator, sile_llir::Terminator::Switch { .. }) {
            return Err(Error::Compile(format!(
                "{stage}: backend emit does not support switch terminators (block `{}`)",
                block.name
            )));
        }
    }
    Ok(())
}

fn verify_metal_backend_contract(llir: &LlirFunction, stage: &str) -> Result<()> {
    for block in &llir.blocks {
        for inst in &block.insts {
            match &inst.op {
                sile_llir::InstOp::Intrinsic { intrinsic, .. } => match intrinsic {
                    sile_llir::Intrinsic::ThreadId { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend does not support thread_id intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_llir::Intrinsic::Barrier { .. } => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend does not support barrier intrinsic (block `{}`)",
                            block.name
                        )));
                    }
                    sile_llir::Intrinsic::BlockId { dim } if *dim > 2 => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend only supports block_id dimensions 0..=2, got {dim} (block `{}`)",
                            block.name
                        )));
                    }
                    _ => {}
                },
                sile_llir::InstOp::Call { func, args } => match func.as_str() {
                    "tile_load_2d_f32" | "tile_store_2d_f32" => {
                        if args.len() != 7 {
                            return Err(Error::Compile(format!(
                                "{stage}: Metal backend expects helper `{func}` to have 7 arguments, got {} (block `{}`)",
                                args.len(),
                                block.name
                            )));
                        }
                    }
                    _ => {
                        return Err(Error::Compile(format!(
                            "{stage}: Metal backend only supports helper calls `tile_load_2d_f32` and `tile_store_2d_f32`, got `{func}` (block `{}`)",
                            block.name
                        )));
                    }
                },
                _ => {}
            }
        }
    }
    Ok(())
}
