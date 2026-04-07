use sile_backend::{BackendArtifact, cpu::CpuBackend, metal::MetalBackend};
use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_llvm_ir::format_llvm_ir;
use sile_tile_ir::format_tile_ir;

use crate::compiler::{CodegenTarget, compile_backend, compile_kernel_to_llvm_ir};

const PRINT_TILE_IR_ENV: &str = "SILE_PRINT_TILEIR";
const PRINT_LLVM_IR_ENV: &str = "SILE_PRINT_LLVMIR";
const PRINT_SOURCE_ENV: &str = "SILE_PRINT_SOURCE";

pub struct KernelLauncher<'a> {
    kernel: &'static Kernel,
    args: Vec<KernelArg<'a>>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(kernel: &'static Kernel, args: Vec<KernelArg<'a>>) -> Self {
        Self {
            kernel,
            args,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn kernel(&self) -> &'static Kernel {
        self.kernel
    }

    pub fn apply(self, stream: &Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self
                .grid
                .ok_or_else(|| sile_core::Error::Shape("grid not set".into()))?,
        };

        let (typed, tile_ir_func, llvm_ir_func) = compile_kernel_to_llvm_ir(self.kernel)?;

        if should_print(PRINT_TILE_IR_ENV) {
            eprintln!("{}", format_tile_ir(&tile_ir_func));
        }
        if should_print(PRINT_LLVM_IR_ENV) {
            eprintln!("{}", format_llvm_ir(&llvm_ir_func));
        }

        match stream.device() {
            Device::Cpu(_) => {
                maybe_print_backend_source(&llvm_ir_func, CodegenTarget::C)?;
                let backend = CpuBackend::new();
                backend.execute_llir(&llvm_ir_func, &self.args, &launch, stream)
            }
            Device::Metal(_) => {
                maybe_print_backend_source(&llvm_ir_func, CodegenTarget::Metal)?;
                let backend = MetalBackend::new()?;
                let param_kinds: Vec<_> =
                    typed.kernel.params.iter().map(|param| param.kind).collect();
                backend.execute_llir(&llvm_ir_func, &param_kinds, &self.args, &launch, stream)
            }
            _ => Err(sile_core::Error::UnsupportedBackend(
                "backend not implemented",
            )),
        }
    }
}

fn should_print(env: &str) -> bool {
    std::env::var_os(env).is_some()
}

fn maybe_print_backend_source(
    llvm_ir_func: &sile_llvm_ir::Function,
    target: CodegenTarget,
) -> Result<()> {
    if !should_print(PRINT_SOURCE_ENV) {
        return Ok(());
    }

    let artifact = compile_backend(llvm_ir_func, target)?;
    match artifact {
        BackendArtifact::CSource(source) | BackendArtifact::MetalSource(source) => {
            eprintln!("{}", source);
        }
    }
    Ok(())
}
