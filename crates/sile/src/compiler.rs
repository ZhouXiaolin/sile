use sile_core::{Error, Result};
use sile_hir::{Kernel, typeck::TypedKernel};
use sile_llir::Function as LlirFunction;
use sile_mir::MirFunction;

use sile_backend_cpu::codegen_llir_c::generate_kernel as generate_llir_c_source;
use sile_backend_metal::codegen_llir_metal::generate as generate_llir_metal_source;

pub use sile_llir::{
    ACTIVE_LLIR_PIPELINE, LlirPassKind, RECOMMENDED_LLIR_PIPELINE, run_llir_passes,
    run_llir_pipeline,
};
pub use sile_mir::passes::{
    ACTIVE_PIPELINE as ACTIVE_MIR_PIPELINE, MirPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_MIR_PIPELINE, run_default_pipeline as run_mir_passes,
    run_pipeline as run_mir_pipeline,
};
pub use sile_mir::{
    ACTIVE_LLIR_LOWERING_PIPELINE, LlirLoweringPassKind, RECOMMENDED_LLIR_LOWERING_PIPELINE, dce,
    lower_mir_to_llir, lower_to_mir, run_default_llir_lowering_pipeline,
    run_llir_lowering_pipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirToMirPassKind {
    LowerToMir,
}

pub const ACTIVE_HIR_TO_MIR_PIPELINE: &[HirToMirPassKind] = &[HirToMirPassKind::LowerToMir];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirPassKind {
    TypeCheck,
    LowerToMir,
}

pub const ACTIVE_HIR_PIPELINE: &[HirPassKind] = &[HirPassKind::TypeCheck, HirPassKind::LowerToMir];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirToLirPassKind {
    Mir(MirPassKind),
    LowerMirToLir,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTarget {
    C,
    Metal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendArtifact {
    CSource(String),
    MetalSource(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LirToBackendPassKind {
    Llir(LlirPassKind),
    Codegen(CodegenTarget),
}

pub fn compose_mir_to_lir_pipeline(mir_pipeline: &[MirPassKind]) -> Vec<MirToLirPassKind> {
    let mut pipeline = mir_pipeline
        .iter()
        .copied()
        .map(MirToLirPassKind::Mir)
        .collect::<Vec<_>>();
    pipeline.push(MirToLirPassKind::LowerMirToLir);
    pipeline
}

pub fn compose_lir_to_backend_pipeline(
    llir_pipeline: &[LlirPassKind],
    codegen: Option<CodegenTarget>,
) -> Vec<LirToBackendPassKind> {
    let mut pipeline = llir_pipeline
        .iter()
        .copied()
        .map(LirToBackendPassKind::Llir)
        .collect::<Vec<_>>();
    if let Some(target) = codegen {
        pipeline.push(LirToBackendPassKind::Codegen(target));
    }
    pipeline
}

pub fn run_hir_to_mir_pipeline(
    typed: &TypedKernel,
    pipeline: &[HirToMirPassKind],
) -> Result<MirFunction> {
    let mut mir = None;
    for pass in pipeline {
        match pass {
            HirToMirPassKind::LowerToMir => {
                mir = Some(lower_to_mir(typed));
            }
        }
    }

    mir.ok_or_else(|| Error::Compile("HIR->MIR pipeline did not produce MIR".into()))
}

pub fn run_mir_to_lir_pipeline(
    typed: &TypedKernel,
    mut mir: MirFunction,
    pipeline: &[MirToLirPassKind],
) -> Result<(MirFunction, LlirFunction)> {
    let mut llir = None;

    for pass in pipeline {
        match pass {
            MirToLirPassKind::Mir(kind) => {
                mir = run_mir_pipeline(mir, &[*kind]).map_err(Error::Shape)?;
            }
            MirToLirPassKind::LowerMirToLir => {
                llir =
                    Some(run_default_llir_lowering_pipeline(&mir, typed).map_err(Error::Compile)?);
            }
        }
    }

    let llir =
        llir.ok_or_else(|| Error::Compile("MIR->LIR pipeline did not produce LIR".into()))?;
    Ok((mir, llir))
}

pub fn run_hir_pipeline(
    kernel: &'static Kernel,
    pipeline: &[HirPassKind],
) -> Result<(TypedKernel, MirFunction)> {
    let mut typed = None;
    let mut mir = None;

    for pass in pipeline {
        match pass {
            HirPassKind::TypeCheck => {
                typed = Some(
                    sile_hir::typeck::check_kernel(kernel)
                        .map_err(|e| Error::Shape(e.to_string()))?,
                );
            }
            HirPassKind::LowerToMir => {
                let typed_ref = typed.as_ref().ok_or_else(|| {
                    Error::Compile("LowerToMir requires TypeCheck earlier in pipeline".into())
                })?;
                mir = Some(lower_to_mir(typed_ref));
            }
        }
    }

    let typed =
        typed.ok_or_else(|| Error::Compile("HIR pipeline did not produce TypedKernel".into()))?;
    let mir = mir.ok_or_else(|| Error::Compile("HIR pipeline did not produce MIR".into()))?;
    Ok((typed, mir))
}

pub fn run_lir_to_backend_pipeline(
    mut llir: LlirFunction,
    pipeline: &[LirToBackendPassKind],
) -> Result<(LlirFunction, Option<BackendArtifact>)> {
    let mut artifact = None;

    for pass in pipeline {
        match pass {
            LirToBackendPassKind::Llir(kind) => {
                llir = run_llir_pipeline(llir, &[*kind]).map_err(Error::Shape)?;
            }
            LirToBackendPassKind::Codegen(target) => {
                artifact = Some(match target {
                    CodegenTarget::C => BackendArtifact::CSource(generate_llir_c_source(&llir)?),
                    CodegenTarget::Metal => {
                        BackendArtifact::MetalSource(generate_llir_metal_source(&llir)?)
                    }
                });
            }
        }
    }

    Ok((llir, artifact))
}

pub fn compile_to_llir(typed: &TypedKernel) -> Result<(MirFunction, LlirFunction)> {
    let mir = run_hir_to_mir_pipeline(typed, ACTIVE_HIR_TO_MIR_PIPELINE)?;
    let mir_to_lir = compose_mir_to_lir_pipeline(ACTIVE_MIR_PIPELINE);
    let (mir, llir) = run_mir_to_lir_pipeline(typed, mir, &mir_to_lir)?;
    let lir_to_backend = compose_lir_to_backend_pipeline(ACTIVE_LLIR_PIPELINE, None);
    let (llir, _) = run_lir_to_backend_pipeline(llir, &lir_to_backend)?;
    Ok((mir, llir))
}

pub fn compile_kernel_to_llir(
    kernel: &'static Kernel,
) -> Result<(TypedKernel, MirFunction, LlirFunction)> {
    let (typed, mir) = run_hir_pipeline(kernel, ACTIVE_HIR_PIPELINE)?;
    let mir_to_lir = compose_mir_to_lir_pipeline(ACTIVE_MIR_PIPELINE);
    let (mir, llir) = run_mir_to_lir_pipeline(&typed, mir, &mir_to_lir)?;
    let lir_to_backend = compose_lir_to_backend_pipeline(ACTIVE_LLIR_PIPELINE, None);
    let (llir, _) = run_lir_to_backend_pipeline(llir, &lir_to_backend)?;
    Ok((typed, mir, llir))
}

pub fn compile_to_backend_source(
    typed: &TypedKernel,
    target: CodegenTarget,
) -> Result<(MirFunction, LlirFunction, BackendArtifact)> {
    let mir = run_hir_to_mir_pipeline(typed, ACTIVE_HIR_TO_MIR_PIPELINE)?;
    let mir_to_lir = compose_mir_to_lir_pipeline(ACTIVE_MIR_PIPELINE);
    let (mir, llir) = run_mir_to_lir_pipeline(typed, mir, &mir_to_lir)?;
    let lir_to_backend = compose_lir_to_backend_pipeline(ACTIVE_LLIR_PIPELINE, Some(target));
    let (llir, artifact) = run_lir_to_backend_pipeline(llir, &lir_to_backend)?;
    let artifact = artifact.ok_or_else(|| {
        Error::Compile("LIR->Backend pipeline did not produce backend artifact".into())
    })?;
    Ok((mir, llir, artifact))
}

pub fn compile_kernel_to_backend_source(
    kernel: &'static Kernel,
    target: CodegenTarget,
) -> Result<(TypedKernel, MirFunction, LlirFunction, BackendArtifact)> {
    let (typed, mir) = run_hir_pipeline(kernel, ACTIVE_HIR_PIPELINE)?;
    let mir_to_lir = compose_mir_to_lir_pipeline(ACTIVE_MIR_PIPELINE);
    let (mir, llir) = run_mir_to_lir_pipeline(&typed, mir, &mir_to_lir)?;
    let lir_to_backend = compose_lir_to_backend_pipeline(ACTIVE_LLIR_PIPELINE, Some(target));
    let (llir, artifact) = run_lir_to_backend_pipeline(llir, &lir_to_backend)?;
    let artifact = artifact.ok_or_else(|| {
        Error::Compile("LIR->Backend pipeline did not produce backend artifact".into())
    })?;
    Ok((typed, mir, llir, artifact))
}
