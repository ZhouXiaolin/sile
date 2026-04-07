use sile_core::{Error, Result};
use sile_hir::{Kernel, typeck::TypedKernel};
use sile_llvm_ir::Function as LlvmIrFunction;

pub use sile_backend::{BackendArtifact, CodegenTarget, compile as compile_backend};

pub use sile_hir::{
    ACTIVE_HIR_PIPELINE as ACTIVE_HIR_TYPECK_PIPELINE, HirPassKind as HirAnalysisPassKind,
    RECOMMENDED_HIR_PIPELINE as RECOMMENDED_HIR_TYPECK_PIPELINE,
    run_hir_passes as run_hir_analysis_passes, run_hir_pipeline as run_hir_analysis_pipeline,
    verify_typed_kernel as verify_hir_typed_kernel,
};

pub use sile_llvm_ir::{
    ACTIVE_LLVM_IR_PIPELINE, LlvmIrPassKind, RECOMMENDED_LLVM_IR_PIPELINE, run_llvm_ir_passes,
    run_llvm_ir_pipeline,
};
pub use sile_tile_ir::passes::{
    ACTIVE_PIPELINE as ACTIVE_TILE_IR_PIPELINE,
    RECOMMENDED_PIPELINE as RECOMMENDED_TILE_IR_PIPELINE, TileIrPassKind,
    run_default_pipeline as run_tile_ir_passes, run_pipeline as run_tile_ir_pipeline,
};
pub use sile_tile_ir::{
    ACTIVE_LLVM_IR_LOWERING_PIPELINE, LlvmIrLoweringPassKind as LlvmLoweringPassKind,
    RECOMMENDED_LLVM_IR_LOWERING_PIPELINE, TileIrFunction, dce, format_tile_ir,
    lower_tile_ir_to_llvm_ir, lower_to_tile_ir,
    run_default_llvm_ir_lowering_pipeline as run_default_llvm_lowering_pipeline,
    run_llvm_ir_lowering_pipeline as run_llvm_lowering_pipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirToTileIrPassKind {
    LowerToTileIr,
}

pub const ACTIVE_HIR_TO_TILE_IR_PIPELINE: &[HirToTileIrPassKind] =
    &[HirToTileIrPassKind::LowerToTileIr];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirPassKind {
    TypeCheck,
    LowerToTileIr,
}

pub const ACTIVE_HIR_PIPELINE: &[HirPassKind] =
    &[HirPassKind::TypeCheck, HirPassKind::LowerToTileIr];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIrToLlvmIrPassKind {
    TileIr(TileIrPassKind),
    LowerTileIrToLlvmIr,
}

pub fn compose_tile_ir_to_llvm_ir_pipeline(
    tile_ir_pipeline: &[TileIrPassKind],
) -> Vec<TileIrToLlvmIrPassKind> {
    let mut pipeline = tile_ir_pipeline
        .iter()
        .copied()
        .map(TileIrToLlvmIrPassKind::TileIr)
        .collect::<Vec<_>>();
    pipeline.push(TileIrToLlvmIrPassKind::LowerTileIrToLlvmIr);
    pipeline
}

pub fn run_hir_to_tile_ir_pipeline(
    typed: &TypedKernel,
    pipeline: &[HirToTileIrPassKind],
) -> Result<TileIrFunction> {
    verify_typed_kernel(typed, "HIR->Tile IR input")?;

    let mut tile_ir = None;
    for pass in pipeline {
        match pass {
            HirToTileIrPassKind::LowerToTileIr => {
                tile_ir = Some(lower_to_tile_ir(typed));
            }
        }
    }

    let tile_ir = tile_ir
        .ok_or_else(|| Error::Compile("HIR->Tile IR pipeline did not produce Tile IR".into()))?;
    verify_tile_ir(&tile_ir, "HIR->Tile IR output")?;
    Ok(tile_ir)
}

pub fn run_tile_ir_to_llvm_ir_pipeline(
    typed: &TypedKernel,
    mut tile_ir: TileIrFunction,
    pipeline: &[TileIrToLlvmIrPassKind],
) -> Result<(TileIrFunction, LlvmIrFunction)> {
    verify_typed_kernel(typed, "Tile IR->LLVM IR typed input")?;

    let mut llvm_ir = None;

    for pass in pipeline {
        match pass {
            TileIrToLlvmIrPassKind::TileIr(kind) => {
                tile_ir = run_tile_ir_pipeline(tile_ir, &[*kind]).map_err(Error::Shape)?;
            }
            TileIrToLlvmIrPassKind::LowerTileIrToLlvmIr => {
                llvm_ir = Some(
                    run_default_llvm_lowering_pipeline(&tile_ir, typed).map_err(Error::Compile)?,
                );
            }
        }
    }

    let llvm_ir = llvm_ir.ok_or_else(|| {
        Error::Compile("Tile IR->LLVM IR pipeline did not produce LLVM IR".into())
    })?;
    Ok((tile_ir, llvm_ir))
}

pub fn compile_to_llvm_ir(typed: &TypedKernel) -> Result<(TileIrFunction, LlvmIrFunction)> {
    let tile_ir = run_hir_to_tile_ir_pipeline(typed, ACTIVE_HIR_TO_TILE_IR_PIPELINE)?;
    let tile_to_llvm = compose_tile_ir_to_llvm_ir_pipeline(ACTIVE_TILE_IR_PIPELINE);
    let (tile_ir, llvm_ir) = run_tile_ir_to_llvm_ir_pipeline(typed, tile_ir, &tile_to_llvm)?;
    let llvm_ir = finalize_llvm_ir_for_backend(llvm_ir)?;
    Ok((tile_ir, llvm_ir))
}

pub fn compile_kernel_to_llvm_ir(
    kernel: &'static Kernel,
) -> Result<(TypedKernel, TileIrFunction, LlvmIrFunction)> {
    let (typed, tile_ir) = run_hir_pipeline(kernel, ACTIVE_HIR_PIPELINE)?;
    let tile_to_llvm = compose_tile_ir_to_llvm_ir_pipeline(ACTIVE_TILE_IR_PIPELINE);
    let (tile_ir, llvm_ir) = run_tile_ir_to_llvm_ir_pipeline(&typed, tile_ir, &tile_to_llvm)?;
    let llvm_ir = finalize_llvm_ir_for_backend(llvm_ir)?;
    Ok((typed, tile_ir, llvm_ir))
}

pub fn run_hir_pipeline(
    kernel: &'static Kernel,
    pipeline: &[HirPassKind],
) -> Result<(TypedKernel, TileIrFunction)> {
    let mut typed = None;
    let mut tile_ir = None;

    for pass in pipeline {
        match pass {
            HirPassKind::TypeCheck => {
                typed = Some(
                    run_hir_analysis_pipeline(
                        kernel,
                        &[
                            HirAnalysisPassKind::VerifyInput,
                            HirAnalysisPassKind::TypeCheck,
                            HirAnalysisPassKind::VerifyOutput,
                        ],
                    )
                    .map_err(|e| Error::Shape(e.to_string()))?,
                );
            }
            HirPassKind::LowerToTileIr => {
                let typed_ref = typed.as_ref().ok_or_else(|| {
                    Error::Compile("LowerToTileIr requires TypeCheck earlier in pipeline".into())
                })?;
                tile_ir = Some(lower_to_tile_ir(typed_ref));
            }
        }
    }

    let typed =
        typed.ok_or_else(|| Error::Compile("HIR pipeline did not produce TypedKernel".into()))?;
    let tile_ir =
        tile_ir.ok_or_else(|| Error::Compile("HIR pipeline did not produce Tile IR".into()))?;
    verify_tile_ir(&tile_ir, "HIR output Tile IR")?;
    Ok((typed, tile_ir))
}

fn verify_typed_kernel(typed: &TypedKernel, stage: &str) -> Result<()> {
    verify_hir_typed_kernel(typed).map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn verify_tile_ir(tile_ir: &TileIrFunction, stage: &str) -> Result<()> {
    run_tile_ir_pipeline(tile_ir.clone(), &[TileIrPassKind::VerifyInput])
        .map(|_| ())
        .map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn finalize_llvm_ir_for_backend(llvm_ir: LlvmIrFunction) -> Result<LlvmIrFunction> {
    run_llvm_ir_pipeline(llvm_ir, ACTIVE_LLVM_IR_PIPELINE).map_err(Error::Shape)
}

pub fn compile_to_backend_source(
    typed: &TypedKernel,
    target: CodegenTarget,
) -> Result<(TileIrFunction, LlvmIrFunction, BackendArtifact)> {
    let tile_ir = run_hir_to_tile_ir_pipeline(typed, ACTIVE_HIR_TO_TILE_IR_PIPELINE)?;
    let tile_to_llvm = compose_tile_ir_to_llvm_ir_pipeline(ACTIVE_TILE_IR_PIPELINE);
    let (tile_ir, llvm_ir) = run_tile_ir_to_llvm_ir_pipeline(typed, tile_ir, &tile_to_llvm)?;
    let llvm_ir = finalize_llvm_ir_for_backend(llvm_ir)?;
    let artifact = compile_backend(&llvm_ir, target)?;
    Ok((tile_ir, llvm_ir, artifact))
}

pub fn compile_kernel_to_backend_source(
    kernel: &'static Kernel,
    target: CodegenTarget,
) -> Result<(TypedKernel, TileIrFunction, LlvmIrFunction, BackendArtifact)> {
    let (typed, tile_ir) = run_hir_pipeline(kernel, ACTIVE_HIR_PIPELINE)?;
    let tile_to_llvm = compose_tile_ir_to_llvm_ir_pipeline(ACTIVE_TILE_IR_PIPELINE);
    let (tile_ir, llvm_ir) = run_tile_ir_to_llvm_ir_pipeline(&typed, tile_ir, &tile_to_llvm)?;
    let llvm_ir = finalize_llvm_ir_for_backend(llvm_ir)?;
    let artifact = compile_backend(&llvm_ir, target)?;
    Ok((typed, tile_ir, llvm_ir, artifact))
}
