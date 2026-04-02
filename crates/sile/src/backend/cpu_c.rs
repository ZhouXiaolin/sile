use std::{fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    codegen,
    kernel::{KernelArg, LaunchConfig},
    KernelSpec, Result, Stream,
};

type KernelFn =
    unsafe extern "C" fn(*const SileTensorArg, usize, *const SileLaunch, *const i64);

#[repr(C)]
struct SileTensorArg {
    data: *mut core::ffi::c_void,
    dtype: i32,
    rank: i32,
    shape: *const i64,
    strides: *const i64,
}

#[repr(C)]
struct SileLaunch {
    grid: [i64; 3],
    tile_shape: [i64; 4],
    tile_rank: i32,
}

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
    fn launch_spec(
        &self,
        spec: &KernelSpec,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", spec.name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let so_path = dir.path().join(format!("lib{}.{}", spec.name, ext));
        fs::write(&c_path, codegen::c::generate(spec)?)?;

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
            let symbol_name = format!("sile_kernel_{}", spec.name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| crate::Error::Compile(e.to_string()))?;

            let packed_args: Vec<SileTensorArg> = args
                .iter()
                .map(|arg| SileTensorArg {
                    data: arg.mut_ptr.cast(),
                    dtype: 0,
                    rank: arg.shape.len() as i32,
                    shape: arg.shape.as_ptr(),
                    strides: core::ptr::null(),
                })
                .collect();

            let tile_size = spec.tile_size()?;
            let launch_arg = SileLaunch {
                grid: [
                    launch.grid[0] as i64,
                    launch.grid[1] as i64,
                    launch.grid[2] as i64,
                ],
                tile_shape: [tile_size, 0, 0, 0],
                tile_rank: 1,
            };

            for gx in 0..launch.grid[0] {
                let tile_id = [gx as i64, 0, 0];
                func(
                    packed_args.as_ptr(),
                    packed_args.len(),
                    &launch_arg,
                    tile_id.as_ptr(),
                );
            }
        }

        Ok(())
    }
}
