pub mod codegen;

use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use sile_llir::Function as LlirFunction;
use tempfile::tempdir;

use self::codegen::generate as generate_llir_kernel;
use sile_core::{KernelArg, LaunchConfig, Result, Stream};

type KernelFn = unsafe extern "C" fn(*const *const c_void, i64, i64, *const i64, i64);

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    fn compiler() -> Result<&'static str> {
        if cfg!(target_os = "macos") {
            if Command::new("/usr/local/opt/llvm@20/bin/clang")
                .arg("--version")
                .output()
                .is_ok()
            {
                return Ok("/usr/local/opt/llvm@20/bin/clang");
            }
            if Command::new("clang").arg("--version").output().is_ok() {
                return Ok("clang");
            }
        }
        for candidate in ["cc", "clang", "gcc"] {
            if Command::new(candidate).arg("--version").output().is_ok() {
                return Ok(candidate);
            }
        }
        Err(sile_core::Error::UnsupportedBackend(
            "no C compiler found".into(),
        ))
    }

    fn compile_llir_kernel(func: &LlirFunction) -> Result<String> {
        generate_llir_kernel(func)
    }

    fn execute_compiled_source(
        &self,
        kernel_name: &str,
        c_code: &str,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        llir_mode: bool,
    ) -> Result<()> {
        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", kernel_name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let so_path = dir.path().join(format!("lib{}.{}", kernel_name, ext));

        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let mut cmd = Command::new(compiler);

        if cfg!(target_os = "macos") {
            cmd.args([
                "-shared",
                "-fPIC",
                "-O2",
                "-Xpreprocessor",
                "-fopenmp",
                "-I/usr/local/opt/libomp/include",
                "-L/usr/local/opt/libomp/lib",
                "-lomp",
                "-o",
            ]);
        } else {
            cmd.args([
                "-shared",
                "-fPIC",
                "-O2",
                "-Xpreprocessor",
                "-fopenmp",
                "-lomp",
                "-lm",
                "-o",
            ]);
        }

        let output = cmd.arg(&so_path).arg(&c_path).output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(sile_core::Error::Compile(format!(
                "C compiler failed:\nstderr:\n{}\nstdout:\n{}\nGenerated C:\n{}",
                stderr, stdout, c_code
            )));
        }

        unsafe {
            let library =
                Library::new(&so_path).map_err(|e| sile_core::Error::Compile(e.to_string()))?;
            let symbol_name = format!("sile_kernel_{}", kernel_name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| sile_core::Error::Compile(e.to_string()))?;

            let buffers: Vec<*const c_void> =
                args.iter().map(|a| a.mut_ptr as *const c_void).collect();

            let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();

            let num_threadgroups = if llir_mode {
                launch.grid.iter().map(|dim| *dim as i64).product::<i64>()
            } else {
                launch.grid[0] as i64
            };
            let threads_per_group = if llir_mode {
                1
            } else if args.is_empty() || args[0].shape.is_empty() {
                256
            } else {
                args[0].shape[0] / num_threadgroups
            };

            func(
                buffers.as_ptr(),
                num_threadgroups,
                threads_per_group,
                shapes.as_ptr(),
                shapes.len() as i64,
            );
        }

        Ok(())
    }

    pub fn execute_llir(
        &self,
        func: &LlirFunction,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let c_code = Self::compile_llir_kernel(func)?;
        self.execute_compiled_source(&func.name, &c_code, args, launch, true)
    }
}
