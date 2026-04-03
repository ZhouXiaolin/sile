use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    codegen::c::{BufferKind, KernelGenInfo},
    hir::Type,
    kernel::{KernelArg, LaunchConfig},
    lir, Result, Stream,
};

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
        Err(crate::Error::UnsupportedBackend("no C compiler found"))
    }

    fn compile_kernel_from_hir(kernel: &crate::hir::Kernel) -> Result<String> {
        let typed = crate::typeck::check_kernel(kernel)
            .map_err(|err| crate::Error::Shape(err.to_string()))?;
        let ssa = crate::passes::canonicalize::run(crate::ssa::lower_typed_kernel_to_ssa(&typed));
        let ssa = crate::passes::dce::run(ssa);
        let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);

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
            num_shapes: kernel
                .params
                .iter()
                .map(|param| match &param.ty {
                    Type::Tensor { shape, .. } | Type::Tile { shape, .. } => shape.rank(),
                    Type::Shape | Type::Scalar(_) => 0,
                })
                .sum(),
            param_ranks: kernel
                .params
                .iter()
                .map(|param| match &param.ty {
                    Type::Tensor { shape, .. } | Type::Tile { shape, .. } => shape.rank(),
                    Type::Shape | Type::Scalar(_) => 0,
                })
                .collect(),
            shape_offsets: {
                let mut offsets = Vec::with_capacity(kernel.params.len());
                let mut next = 0usize;
                for param in &kernel.params {
                    offsets.push(next);
                    next += match &param.ty {
                        Type::Tensor { shape, .. } | Type::Tile { shape, .. } => shape.rank(),
                        Type::Shape | Type::Scalar(_) => 0,
                    };
                }
                offsets
            },
        };

        let code = crate::codegen::c::generate(&lir_func, &info)?;
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
        fs::write(&c_path, &c_code)?;

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
            return Err(crate::Error::Compile(format!(
                "C compiler failed:\nstderr:\n{}\nstdout:\n{}\nGenerated C:\n{}",
                stderr, stdout, c_code
            )));
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
