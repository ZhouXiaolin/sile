use sile_core::{Error, Result};
use sile_hir::{Kernel, typeck::TypedKernel};
use sile_llir::Function as LlirFunction;
use sile_mir::MirFunction;

pub use sile_backend::{
    BackendArtifact, BackendPassKind, CodegenTarget, compose_backend_pipeline, run_backend_pass,
    run_backend_pipeline,
};

pub use sile_hir::{
    ACTIVE_HIR_PIPELINE as ACTIVE_HIR_TYPECK_PIPELINE, HirPassKind as HirAnalysisPassKind,
    RECOMMENDED_HIR_PIPELINE as RECOMMENDED_HIR_TYPECK_PIPELINE,
    run_hir_passes as run_hir_analysis_passes, run_hir_pipeline as run_hir_analysis_pipeline,
    verify_typed_kernel as verify_hir_typed_kernel,
};

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
pub enum LirToBackendPassKind {
    Llir(LlirPassKind),
    Backend(BackendPassKind),
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
        pipeline.extend(
            compose_backend_pipeline(target)
                .into_iter()
                .map(LirToBackendPassKind::Backend),
        );
    }
    pipeline
}

pub fn run_hir_to_mir_pipeline(
    typed: &TypedKernel,
    pipeline: &[HirToMirPassKind],
) -> Result<MirFunction> {
    verify_typed_kernel(typed, "HIR->MIR input")?;

    let mut mir = None;
    for pass in pipeline {
        match pass {
            HirToMirPassKind::LowerToMir => {
                mir = Some(lower_to_mir(typed));
            }
        }
    }

    let mir = mir.ok_or_else(|| Error::Compile("HIR->MIR pipeline did not produce MIR".into()))?;
    verify_mir_via_pipeline(&mir, "HIR->MIR output")?;
    Ok(mir)
}

pub fn run_mir_to_lir_pipeline(
    typed: &TypedKernel,
    mut mir: MirFunction,
    pipeline: &[MirToLirPassKind],
) -> Result<(MirFunction, LlirFunction)> {
    verify_typed_kernel(typed, "MIR->LIR typed input")?;
    ensure_mir_to_lir_pipeline_contract(pipeline)?;

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
    verify_mir_via_pipeline(&mir, "HIR output MIR")?;
    Ok((typed, mir))
}

pub fn run_lir_to_backend_pipeline(
    mut llir: LlirFunction,
    pipeline: &[LirToBackendPassKind],
) -> Result<(LlirFunction, Option<BackendArtifact>)> {
    ensure_lir_to_backend_pipeline_contract(pipeline)?;

    let mut artifact = None;

    for pass in pipeline {
        match pass {
            LirToBackendPassKind::Llir(kind) => {
                llir = run_llir_pipeline(llir, &[*kind]).map_err(Error::Shape)?;
            }
            LirToBackendPassKind::Backend(pass) => {
                let (next_llir, maybe_artifact) = run_backend_pass(llir, *pass)?;
                llir = next_llir;
                if let Some(next) = maybe_artifact {
                    artifact = Some(next);
                }
            }
        }
    }

    Ok((llir, artifact))
}

fn verify_typed_kernel(typed: &TypedKernel, stage: &str) -> Result<()> {
    verify_hir_typed_kernel(typed).map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn verify_mir_function(mir: &MirFunction, stage: &str) -> Result<()> {
    run_mir_pipeline(mir.clone(), &[MirPassKind::VerifyInput])
        .map(|_| ())
        .map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn verify_mir_via_pipeline(mir: &MirFunction, stage: &str) -> Result<()> {
    verify_mir_function(mir, stage)
}

fn ensure_mir_to_lir_pipeline_contract(pipeline: &[MirToLirPassKind]) -> Result<()> {
    let has_verify_input = pipeline
        .iter()
        .any(|pass| matches!(pass, MirToLirPassKind::Mir(MirPassKind::VerifyInput)));
    let has_verify_output = pipeline
        .iter()
        .any(|pass| matches!(pass, MirToLirPassKind::Mir(MirPassKind::VerifyOutput)));
    let has_lower = pipeline
        .iter()
        .any(|pass| matches!(pass, MirToLirPassKind::LowerMirToLir));

    if !has_lower {
        return Err(Error::Compile(
            "MIR->LIR pipeline must include LowerMirToLir".into(),
        ));
    }
    if !has_verify_input || !has_verify_output {
        return Err(Error::Compile(
            "MIR->LIR pipeline must include MIR VerifyInput and VerifyOutput passes".into(),
        ));
    }
    Ok(())
}

fn ensure_lir_to_backend_pipeline_contract(pipeline: &[LirToBackendPassKind]) -> Result<()> {
    let has_llir_verify_input = pipeline
        .iter()
        .any(|pass| matches!(pass, LirToBackendPassKind::Llir(LlirPassKind::VerifyInput)));
    let has_llir_verify_output = pipeline
        .iter()
        .any(|pass| matches!(pass, LirToBackendPassKind::Llir(LlirPassKind::VerifyOutput)));

    if !has_llir_verify_input || !has_llir_verify_output {
        return Err(Error::Compile(
            "LIR->Backend pipeline must include LLIR VerifyInput and VerifyOutput passes".into(),
        ));
    }

    let has_backend = pipeline
        .iter()
        .any(|pass| matches!(pass, LirToBackendPassKind::Backend(_)));
    if has_backend {
        let has_backend_verify_input = pipeline.iter().any(|pass| {
            matches!(
                pass,
                LirToBackendPassKind::Backend(BackendPassKind::VerifyInput)
            )
        });
        let has_backend_verify_output = pipeline.iter().any(|pass| {
            matches!(
                pass,
                LirToBackendPassKind::Backend(BackendPassKind::VerifyOutput)
            )
        });
        if !has_backend_verify_input || !has_backend_verify_output {
            return Err(Error::Compile(
                "LIR->Backend pipeline must include Backend VerifyInput and VerifyOutput passes when backend stage is enabled".into(),
            ));
        }
    }

    Ok(())
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
