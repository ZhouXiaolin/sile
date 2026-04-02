use std::{fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    kernel::{KernelArg, LaunchConfig},
    Result, Stream,
};

type KernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, i64);

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

        let tile_size = args[0].shape[0] / launch.grid[0] as i64;
        let c_code = format!(
            "#include <stdint.h>\n#include <stddef.h>\n\n\
             void sile_kernel_{name}(float* a, float* b, float* c, int64_t pid, int64_t tile_size) {{\n\
             \x20   int64_t base = pid * tile_size;\n\
             \x20   for (int64_t i = 0; i < tile_size; ++i) {{\n\
             \x20       c[base + i] = a[base + i] + b[base + i];\n\
             \x20   }}\\
             }}\n",
            name = kernel.name,
        );
        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC", "-O2"])
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

            let a_ptr = args[0].mut_ptr;
            let b_ptr = args[1].mut_ptr;
            let c_ptr = args[2].mut_ptr;

            for gx in 0..launch.grid[0] {
                func(a_ptr, b_ptr, c_ptr, gx as i64, tile_size);
            }
        }

        Ok(())
    }
}
