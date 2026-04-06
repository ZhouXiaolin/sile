use std::collections::HashSet;

use sile_core::{Error, Result};
use sile_hir::{Kernel, typeck::TypedKernel};
use sile_llir::Function as LlirFunction;
use sile_mir::MirFunction;

pub use sile_backend::{
    BackendArtifact, BackendPassKind, CodegenTarget, compose_backend_pipeline, run_backend_pass,
    run_backend_pipeline,
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
    verify_mir_function(&mir, "HIR->MIR output")?;
    Ok(mir)
}

pub fn run_mir_to_lir_pipeline(
    typed: &TypedKernel,
    mut mir: MirFunction,
    pipeline: &[MirToLirPassKind],
) -> Result<(MirFunction, LlirFunction)> {
    verify_typed_kernel(typed, "MIR->LIR typed input")?;
    verify_mir_function(&mir, "MIR->LIR input")?;

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
    verify_mir_function(&mir, "MIR->LIR MIR output")?;
    verify_llir_function(&llir, "MIR->LIR LIR output")?;
    Ok((mir, llir))
}

pub fn run_hir_pipeline(
    kernel: &'static Kernel,
    pipeline: &[HirPassKind],
) -> Result<(TypedKernel, MirFunction)> {
    verify_hir_kernel(kernel, "HIR input")?;

    let mut typed = None;
    let mut mir = None;

    for pass in pipeline {
        match pass {
            HirPassKind::TypeCheck => {
                typed = Some(
                    sile_hir::typeck::check_kernel(kernel)
                        .map_err(|e| Error::Shape(e.to_string()))?,
                );
                verify_typed_kernel(
                    typed
                        .as_ref()
                        .expect("typed kernel must exist right after typecheck"),
                    "HIR typecheck output",
                )?;
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
    verify_typed_kernel(&typed, "HIR output typed kernel")?;
    verify_mir_function(&mir, "HIR output MIR")?;
    Ok((typed, mir))
}

pub fn run_lir_to_backend_pipeline(
    mut llir: LlirFunction,
    pipeline: &[LirToBackendPassKind],
) -> Result<(LlirFunction, Option<BackendArtifact>)> {
    verify_llir_function(&llir, "LIR->Backend input")?;

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

    verify_llir_function(&llir, "LIR->Backend output")?;
    Ok((llir, artifact))
}

fn verify_hir_kernel(kernel: &Kernel, stage: &str) -> Result<()> {
    if kernel.name.trim().is_empty() {
        return Err(Error::Compile(format!(
            "{stage}: kernel name must not be empty"
        )));
    }

    let mut seen_params = HashSet::new();
    for param in &kernel.params {
        if param.name.trim().is_empty() {
            return Err(Error::Compile(format!(
                "{stage}: kernel param name must not be empty"
            )));
        }
        if !seen_params.insert(param.name.as_str()) {
            return Err(Error::Compile(format!(
                "{stage}: duplicate kernel param name `{}`",
                param.name
            )));
        }
    }

    let mut seen_consts = HashSet::new();
    for (name, _) in &kernel.const_params {
        if name.trim().is_empty() {
            return Err(Error::Compile(format!(
                "{stage}: const param name must not be empty"
            )));
        }
        if !seen_consts.insert(name.as_str()) {
            return Err(Error::Compile(format!(
                "{stage}: duplicate const param name `{name}`"
            )));
        }
    }

    Ok(())
}

fn verify_typed_kernel(typed: &TypedKernel, stage: &str) -> Result<()> {
    verify_hir_kernel(&typed.kernel, stage)?;
    for local in typed.locals.keys() {
        if local.trim().is_empty() {
            return Err(Error::Compile(format!(
                "{stage}: typed local name must not be empty"
            )));
        }
    }
    Ok(())
}

fn verify_mir_function(mir: &MirFunction, stage: &str) -> Result<()> {
    sile_mir::passes::verify::verify_function(mir)
        .map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn verify_llir_function(llir: &LlirFunction, stage: &str) -> Result<()> {
    sile_llir::passes::verify::verify_function(llir)
        .map_err(|err| Error::Compile(format!("{stage}: {err}")))
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
