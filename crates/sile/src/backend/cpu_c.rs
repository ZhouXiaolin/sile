use std::{fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    backend_ir::{self, ir::BackendOp},
    kernel::{KernelArg, LaunchConfig},
    ssa, Result, Stream,
};

type KernelFn1D = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, i64);

type KernelFnSoftmax = unsafe extern "C" fn(*const f32, *mut f32, i64, i64, i64, i64);

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

    fn compile_kernel_from_hir(kernel: &crate::hir::Kernel) -> Result<(String, BackendOp)> {
        let typed = crate::typeck::check_kernel(kernel)
            .map_err(|err| crate::Error::Shape(err.to_string()))?;
        let ssa = crate::passes::canonicalize::run(ssa::lower_typed_kernel_to_ssa(&typed));
        let ssa = crate::passes::dce::run(ssa);
        let backend = backend_ir::lower::lower_ssa_to_backend_ir(&ssa);
        let op = backend.op;
        let code = crate::codegen::c::generate(&backend)?;
        Ok((code, op))
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

        let (c_code, backend_op) = Self::compile_kernel_from_hir(kernel)?;
        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC", "-O2"])
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

            let tile_size = args[0].shape[0] / launch.grid[0] as i64;

            match backend_op {
                BackendOp::Softmax2D => {
                    let func: libloading::Symbol<KernelFnSoftmax> = library
                        .get(symbol_name.as_bytes())
                        .map_err(|e| crate::Error::Compile(e.to_string()))?;

                    let x_ptr = args[0].mut_ptr;
                    let y_ptr = args[1].mut_ptr;
                    let n = args[0].shape[1];
                    let bm = tile_size;

                    for gx in 0..launch.grid[0] {
                        func(x_ptr, y_ptr, gx as i64, bm, n, n);
                    }
                }
                BackendOp::VecAdd1D => {
                    let func: libloading::Symbol<KernelFn1D> = library
                        .get(symbol_name.as_bytes())
                        .map_err(|e| crate::Error::Compile(e.to_string()))?;

                    let a_ptr = args[0].mut_ptr;
                    let b_ptr = args[1].mut_ptr;
                    let c_ptr = args[2].mut_ptr;

                    for gx in 0..launch.grid[0] {
                        func(a_ptr, b_ptr, c_ptr, gx as i64, tile_size);
                    }
                }
            }
        }

        Ok(())
    }
}
