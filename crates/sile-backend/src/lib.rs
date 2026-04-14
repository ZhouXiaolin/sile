use sile_core::Result;
use sile_llvm_ir::Function as LlvmIrFunction;

mod emit;
mod verify;

pub mod cpu;
#[cfg(target_os = "macos")]
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

pub fn compile(llvm_ir: &LlvmIrFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    verify::for_target(llvm_ir, target, "Backend compile input")?;
    emit::run(llvm_ir, target)
}
