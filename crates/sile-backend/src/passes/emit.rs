use sile_core::Result;
use sile_llir::Function as LlirFunction;

use crate::cpu::codegen_llir_c::generate_kernel as generate_llir_c_source;
use crate::metal::codegen_llir_metal::generate as generate_llir_metal_source;
use crate::{BackendArtifact, CodegenTarget};

pub(crate) mod shared;

pub fn run(llir: &LlirFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    let artifact = match target {
        CodegenTarget::C => BackendArtifact::CSource(generate_llir_c_source(llir)?),
        CodegenTarget::Metal => BackendArtifact::MetalSource(generate_llir_metal_source(llir)?),
    };
    Ok(artifact)
}
