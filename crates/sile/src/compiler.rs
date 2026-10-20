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

/// The only path from a kernel to a backend's input.
///
/// The arena `Module` is the whole Tile IR, so there is no second IR to keep in
/// step and no lowering pipeline to compose. The typed kernel is carried
/// alongside because the host side still reads the launch checks the compiler
/// moved out of the kernel.
pub fn compile_kernel_to_module(
    kernel: &Kernel,
) -> Result<(TypedKernel, sile_tile_ir::ir::Module, LlvmIrFunction)> {
    let typed = run_hir_analysis_pipeline(kernel, ACTIVE_HIR_TYPECK_PIPELINE)
        .map_err(|e| Error::Shape(e.to_string()))?;
    verify_typed_kernel(&typed, "HIR output")?;
    let (module, _checks) = sile_tile_ir::cutile::emit::emit(&typed).map_err(Error::Shape)?;
    module
        .verify_dominance()
        .map_err(|e| Error::Compile(format!("Tile IR dominance: {e}")))?;
    let llvm_ir = sile_llvm_ir::lower_module_to_llvm_ir(&module).map_err(Error::Compile)?;
    if std::env::var_os("SILE_PRINT_LLVMIR").is_some() {
        eprintln!("{}", sile_llvm_ir::format_llvm_ir(&llvm_ir));
    }
    let llvm_ir = finalize_llvm_ir_for_backend(llvm_ir)?;
    Ok((typed, module, llvm_ir))
}

/// The same compilation, also returning the backend's source text.
pub fn compile_kernel_to_backend_source(
    kernel: &Kernel,
    target: CodegenTarget,
) -> Result<(TypedKernel, sile_tile_ir::ir::Module, LlvmIrFunction, BackendArtifact)> {
    let (typed, module, llvm_ir) = compile_kernel_to_module(kernel)?;
    let artifact = compile_backend(&llvm_ir, target)?;
    Ok((typed, module, llvm_ir, artifact))
}

fn verify_typed_kernel(typed: &TypedKernel, stage: &str) -> Result<()> {
    verify_hir_typed_kernel(typed).map_err(|err| Error::Compile(format!("{stage}: {err}")))
}

fn finalize_llvm_ir_for_backend(llvm_ir: LlvmIrFunction) -> Result<LlvmIrFunction> {
    run_llvm_ir_pipeline(llvm_ir, ACTIVE_LLVM_IR_PIPELINE).map_err(Error::Shape)
}
