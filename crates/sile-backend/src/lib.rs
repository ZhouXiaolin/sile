use sile_core::Result;
use sile_llir::Function as LlirFunction;

mod emit;
mod verify;

pub mod cpu;
pub mod metal;

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

pub fn compile(llir: &LlirFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    verify::for_target(llir, target, "Backend compile input")?;
    emit::run(llir, target)
}
