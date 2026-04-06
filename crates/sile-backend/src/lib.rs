use sile_core::Result;
use sile_llir::Function as LlirFunction;

pub mod cpu;
pub mod metal;
pub mod passes;

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
pub enum BackendPassKind {
    VerifyInput,
    Legalize,
    Canonicalize,
    VerifyOutput,
    Emit(CodegenTarget),
}

pub const SHARED_BACKEND_PIPELINE: &[BackendPassKind] = &[
    BackendPassKind::VerifyInput,
    BackendPassKind::Legalize,
    BackendPassKind::Canonicalize,
    BackendPassKind::VerifyOutput,
];

pub fn compose_backend_pipeline(target: CodegenTarget) -> Vec<BackendPassKind> {
    let mut pipeline = SHARED_BACKEND_PIPELINE.to_vec();
    pipeline.push(BackendPassKind::Emit(target));
    pipeline
}

pub fn run_backend_pipeline(
    mut llir: LlirFunction,
    pipeline: &[BackendPassKind],
) -> Result<(LlirFunction, Option<BackendArtifact>)> {
    let mut artifact = None;
    for pass in pipeline {
        let (next_llir, next_artifact) = run_backend_pass(llir, *pass)?;
        llir = next_llir;
        if let Some(next_artifact) = next_artifact {
            artifact = Some(next_artifact);
        }
    }
    Ok((llir, artifact))
}

pub fn run_backend_pass(
    llir: LlirFunction,
    pass: BackendPassKind,
) -> Result<(LlirFunction, Option<BackendArtifact>)> {
    match pass {
        BackendPassKind::VerifyInput | BackendPassKind::VerifyOutput => {
            passes::verify::run(&llir, "Backend verify pass")?;
            Ok((llir, None))
        }
        BackendPassKind::Legalize => Ok((passes::legalize::run(llir), None)),
        BackendPassKind::Canonicalize => Ok((passes::canonicalize::run(llir), None)),
        BackendPassKind::Emit(target) => {
            let artifact = passes::emit::run(&llir, target)?;
            Ok((llir, Some(artifact)))
        }
    }
}
