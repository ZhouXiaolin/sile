use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use sile_backend::{
    BackendArtifact, cpu::CpuBackend, metal::MetalBackend, wgsl::runtime::WgpuBackend,
};
use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_llvm_ir::format_llvm_ir;
use sile_tile_ir::format_tile_ir;

use crate::compiler::{CodegenTarget, compile_backend, compile_kernel_to_llvm_ir};

const PRINT_TILE_IR_ENV: &str = "SILE_PRINT_TILEIR";
const PRINT_LLVM_IR_ENV: &str = "SILE_PRINT_LLVMIR";
const PRINT_SOURCE_ENV: &str = "SILE_PRINT_SOURCE";

/// One compiled kernel, shared by every launch of the same source.
///
/// Compilation is a pure function of the HIR: the device only chooses which
/// backend consumes the LLVM IR, never what the IR is. So the result is cached
/// once per distinct kernel and every target reuses it. Without this, a launch
/// inside a loop re-runs the whole front end and both lowering stages on every
/// iteration.
type CompiledParts = (sile_hir::typeck::TypedKernel, sile_tile_ir::TileIrFunction, sile_llvm_ir::Function);

fn compile_cache() -> &'static Mutex<HashMap<String, Arc<CompiledParts>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<CompiledParts>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cache key. The HIR is the compilation unit, so its full debug form is
/// the identity: two kernels with the same body and constants compile to the
/// same IR, whatever they are named.
fn cache_key(kernel: &Kernel) -> String {
    format!("{kernel:?}")
}

fn compiled_parts(kernel: &Kernel) -> Result<Arc<CompiledParts>> {
    let key = cache_key(kernel);
    if let Some(hit) = compile_cache().lock().unwrap().get(&key) {
        return Ok(Arc::clone(hit));
    }

    let parts = Arc::new(compile_kernel_to_llvm_ir(kernel)?);
    compile_cache()
        .lock()
        .unwrap()
        .insert(key, Arc::clone(&parts));
    Ok(parts)
}

pub struct KernelLauncher<'a> {
    kernel: Kernel,
    args: Vec<KernelArg<'a>>,
    /// Float scalar arguments, in kernel parameter order.
    ///
    /// A scalar travels beside the buffers rather than inside them, so the
    /// launch keeps the lists apart all the way to the backend. An integer
    /// scalar needs a list of its own: the two do not share a width.
    scalars: Vec<f32>,
    scalars_i64: Vec<i64>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(
        kernel: Kernel,
        args: Vec<KernelArg<'a>>,
        scalars: Vec<f32>,
        scalars_i64: Vec<i64>,
    ) -> Self {
        Self {
            kernel,
            args,
            scalars,
            scalars_i64,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn apply(self, stream: &Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self
                .grid
                .ok_or_else(|| sile_core::Error::Shape("grid not set".into()))?,
        };

        let parts = compiled_parts(&self.kernel)?;
        let (typed, tile_ir_func, llvm_ir_func) = parts.as_ref();

        if should_print(PRINT_TILE_IR_ENV) {
            eprintln!("{}", format_tile_ir(tile_ir_func));
        }
        if should_print(PRINT_LLVM_IR_ENV) {
            eprintln!("{}", format_llvm_ir(llvm_ir_func));
        }

        match stream.device() {
            Device::Cpu(_) => {
                maybe_print_backend_source(llvm_ir_func, CodegenTarget::C)?;
                let backend = CpuBackend::new();
                backend.execute_llir(
                    llvm_ir_func,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
            }
            Device::Metal(_) => {
                maybe_print_backend_source(llvm_ir_func, CodegenTarget::Metal)?;
                let backend = MetalBackend::new()?;
                // `args` holds the buffers only, so the kinds list has to line
                // up with it: a scalar parameter is not in `args` and would
                // shift every output that follows it.
                let param_kinds: Vec<_> = typed
                    .kernel
                    .params
                    .iter()
                    .filter(|param| !matches!(param.ty, sile_hir::Type::Scalar(_)))
                    .map(|param| param.kind)
                    .collect();
                backend.execute_llir(
                    llvm_ir_func,
                    &param_kinds,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
            }
            Device::Wgpu(_) => {
                maybe_print_backend_source(llvm_ir_func, CodegenTarget::Wgsl)?;
                let backend = WgpuBackend::new()?;
                backend.execute_llir(
                    llvm_ir_func,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
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
        BackendArtifact::CSource(source)
        | BackendArtifact::MetalSource(source)
        | BackendArtifact::WgslSource(source) => {
            eprintln!("{}", source);
        }
    }
    Ok(())
}
