use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_lir::Backend;

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

        let typed = sile_hir::typeck::check_kernel(self.kernel)
            .map_err(|e| sile_core::Error::Shape(e.to_string()))?;

        let lir_func = sile_compiler::compile(&typed);

        match stream.device() {
            Device::Cpu(_) => {
                let backend = sile_backend_cpu::CpuBackend::new();
                backend.execute(&lir_func, self.kernel, &self.args, &launch, stream)
            }
            Device::Metal(_) => {
                let backend = sile_backend_metal::MetalBackend::new()?;
                backend.execute(&lir_func, self.kernel, &self.args, &launch, stream)
            }
            _ => Err(sile_core::Error::UnsupportedBackend(
                "backend not implemented",
            )),
        }
    }
}
