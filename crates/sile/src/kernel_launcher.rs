use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_lir::{Backend, print::format_executable_kernel};
use sile_llir::format_function as format_llir_function;

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
        let use_llir_cpu = std::env::var_os("SILE_USE_LLIR_CPU").is_some();
        let use_llir_metal = std::env::var_os("SILE_USE_LLIR_METAL").is_some();

        let typed = sile_hir::typeck::check_kernel(self.kernel)
            .map_err(|e| sile_core::Error::Shape(e.to_string()))?;

        let executable = sile_mir::lower_to_mir(&typed);
        let executable = sile_mir::dce::run(executable);
        let llir_func =
            if std::env::var_os("SILE_PRINT_LLIR").is_some() || use_llir_metal || use_llir_cpu {
                Some(sile_mir::lower_mir_to_llir(&executable, &typed))
            } else {
                None
            };

        if let Some(llir_func) = llir_func
            .as_ref()
            .filter(|_| std::env::var_os("SILE_PRINT_LLIR").is_some())
        {
            eprintln!("{}", format_llir_function(llir_func));
        }

        let executable = sile_mir::lower_mir_to_lir(&executable, &typed);

        if std::env::var_os("SILE_PRINT_LIR").is_some() {
            eprintln!("{}", format_executable_kernel(&executable));
        }

        match stream.device() {
            Device::Cpu(_) => {
                let backend = sile_backend_cpu::CpuBackend::new();
                if use_llir_cpu {
                    let llir_func = llir_func.as_ref().ok_or_else(|| {
                        sile_core::Error::Compile(
                            "LLIR CPU path requested but LLIR function was not generated".into(),
                        )
                    })?;
                    backend.execute_llir(llir_func, &self.args, &launch, stream)
                } else {
                    backend.execute(&executable, &self.args, &launch, stream)
                }
            }
            Device::Metal(_) => {
                let backend = sile_backend_metal::MetalBackend::new()?;
                if use_llir_metal {
                    let llir_func = llir_func.as_ref().ok_or_else(|| {
                        sile_core::Error::Compile(
                            "LLIR Metal path requested but LLIR function was not generated".into(),
                        )
                    })?;
                    let param_kinds: Vec<_> =
                        typed.kernel.params.iter().map(|param| param.kind).collect();
                    backend.execute_llir(llir_func, &param_kinds, &self.args, &launch, stream)
                } else {
                    backend.execute(&executable, &self.args, &launch, stream)
                }
            }
            _ => Err(sile_core::Error::UnsupportedBackend(
                "backend not implemented",
            )),
        }
    }
}
