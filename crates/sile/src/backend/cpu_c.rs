use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    codegen::c::{BufferKind, KernelGenInfo},
    kernel::{KernelArg, LaunchConfig},
    lir, scheduling, Result, Stream,
};

type KernelFn = unsafe extern "C" fn(*const *const c_void, i64, i64, *const i64, i64);

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    fn compiler() -> Result<&'static str> {
        for candidate in ["cc", "clang", "gcc"] {
            if Command::new(candidate).arg("--version").output().is_ok() {
                return Ok(candidate);
            }
        }
        Err(crate::Error::UnsupportedBackend("no C compiler found"))
    }

    fn compile_kernel_from_hir(kernel: &crate::hir::Kernel) -> Result<String> {
        let typed = crate::typeck::check_kernel(kernel)
            .map_err(|err| crate::Error::Shape(err.to_string()))?;
        let ssa = crate::passes::canonicalize::run(crate::ssa::lower_typed_kernel_to_ssa(&typed));
        let ssa = crate::passes::dce::run(ssa);
        let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);
        let annotations = scheduling::annotate(&lir_func);

        let info = KernelGenInfo {
            name: kernel.name.clone(),
            num_buffers: kernel.params.len(),
            buffer_kinds: kernel
                .params
                .iter()
                .map(|p| match p.kind {
                    crate::hir::ParamKind::Input => BufferKind::Input,
                    crate::hir::ParamKind::Output => BufferKind::Output,
                })
                .collect(),
            num_shapes: 1,
        };

        let code = crate::codegen::c::generate(&lir_func, &annotations, &info)?;
        Ok(code)
    }
}

impl crate::backend::Backend for CpuBackend {
    fn launch_kernel(
        &self,
        kernel: &crate::hir::Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", kernel.name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let so_path = dir.path().join(format!("lib{}.{}", kernel.name, ext));

        let c_code = Self::compile_kernel_from_hir(kernel)?;
        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC", "-O3", "-Xpreprocessor", "-fopenmp"])
            .arg("-lomp")
            .arg("-lm")
            .arg(&c_path)
            .arg("-o")
            .arg(&so_path)
            .output()?;
        if !output.status.success() {
            return Err(crate::Error::Compile(
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ));
        }

        unsafe {
            let library =
                Library::new(&so_path).map_err(|e| crate::Error::Compile(e.to_string()))?;
            let symbol_name = format!("sile_kernel_{}", kernel.name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| crate::Error::Compile(e.to_string()))?;

            let buffers: Vec<*const c_void> =
                args.iter().map(|a| a.mut_ptr as *const c_void).collect();

            let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();

            let num_threadgroups = launch.grid[0] as i64;
            let threads_per_group = if args.is_empty() || args[0].shape.is_empty() {
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
}
