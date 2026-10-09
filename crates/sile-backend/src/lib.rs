use sile_core::Result;
use sile_llvm_ir::Function as LlvmIrFunction;

mod emit;
mod verify;

pub mod cpu;
pub mod metal;
pub mod wgsl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTarget {
    C,
    Metal,
    Wgsl,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendArtifact {
    CSource(String),
    MetalSource(String),
    WgslSource(String),
}

pub fn compile(llvm_ir: &LlvmIrFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    verify::for_target(llvm_ir, target, "Backend compile input")?;
    let artifact = emit::run(llvm_ir, target)?;
    // WGSL is the one target with a real parser and validator in the tree, so
    // the IR to source step is checked here rather than trusting the emitter.
    // Without this, a kernel that only the GPU driver would reject ships as a
    // string and the first sign of trouble is a runtime failure.
    if let BackendArtifact::WgslSource(source) = &artifact {
        wgsl::validate(source)?;
    }
    Ok(artifact)
}
