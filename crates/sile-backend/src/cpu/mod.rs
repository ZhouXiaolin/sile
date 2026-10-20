pub mod codegen;

use std::{
    collections::HashMap,
    ffi::c_void,
    fs,
    path::PathBuf,
    process::Command,
    sync::{Mutex, OnceLock},
};

use libloading::Library;
use sile_llvm_ir::Function as LlvmIrFunction;
use tempfile::tempdir;

use self::codegen::generate as generate_llvm_ir_kernel;
use sile_core::{KernelArg, LaunchConfig, Result, Stream};

type KernelFn = unsafe extern "C" fn(
    *const *const c_void,
    i64,
    i64,
    *const i64,
    i64,
    *const f32,
    *const i64,
    *const i64,
);

/// Compiled kernels, keyed by the source they were built from.
///
/// `cc -O2` costs tens of milliseconds, which dwarfs a short kernel's own
/// runtime: a 16M-element saxpy runs in 2.7 ms. Keeping the loaded library
/// means the second launch of a kernel measures the kernel, not the compiler.
fn loaded_libraries() -> &'static Mutex<HashMap<String, &'static Library>> {
    static LIBRARIES: OnceLock<Mutex<HashMap<String, &'static Library>>> = OnceLock::new();
    LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Generated C source, keyed by the address of the IR it was lowered from.
///
/// The address is a plain `usize` so the map can live in a `static`: the
/// function outlives the cache entry, which is only read back for that same
/// function.
fn generated_sources() -> &'static Mutex<HashMap<usize, String>> {
    static SOURCES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
    SOURCES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The temporary directory each cached library was built in. It has to outlive
/// the library: the `.so` is mapped from it.
fn kept_directories() -> &'static Mutex<Vec<PathBuf>> {
    static DIRECTORIES: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    DIRECTORIES.get_or_init(|| Mutex::new(Vec::new()))
}

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

    fn compile_llvm_ir_kernel(func: &LlvmIrFunction) -> Result<String> {
        generate_llvm_ir_kernel(func)
    }

    fn execute_compiled_source(
        &self,
        kernel_name: &str,
        c_code: &str,
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        llvm_ir_mode: bool,
    ) -> Result<()> {
        // A cached library serves every launch of the same source.
        let library = {
            let libraries = loaded_libraries().lock().unwrap();
            libraries.get(c_code).copied()
        };
        if let Some(library) = library {
            return Self::launch(library, kernel_name, args, scalars, scalars_i64, launch, llvm_ir_mode);
        }

        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", kernel_name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "windows") {
            "dll"
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
        } else if cfg!(target_os = "windows") {
            // clang on Windows targets MSVC: no `-fPIC`, and the runtime has no
            // `libm` to link. Its OpenMP library is LLVM's own, which
            // `-fopenmp=libomp` names instead of the `omp.lib` the default
            // spelling asks for. A shared library exports nothing by default,
            // so the one entry point the host looks up is named explicitly.
            // `_Float16` conversions live in compiler-rt, which the MSVC target
            // does not link by default.
            let export = format!("-Wl,/EXPORT:sile_kernel_{kernel_name}");
            cmd.args([
                "-shared",
                "-O2",
                "-fopenmp=libomp",
                "-rtlib=compiler-rt",
            ]);
            cmd.arg(export);
            cmd.arg("-o");
        } else {
            // gcc's OpenMP runtime is libgomp, clang's is libomp. Each driver
            // links its own when it sees `-fopenmp`; naming `-lomp` by hand
            // picks a library half the Linux machines do not have.
            //
            // `nodelete` keeps the library mapped after `dlclose`. libgomp's
            // worker threads are parked inside this library and the runtime
            // tears them down lazily, so unloading it under them segfaults.
            cmd.args([
                "-shared",
                "-fPIC",
                "-O2",
                "-fopenmp",
                "-Wl,-z,nodelete",
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

        // The library is leaked on purpose: it is cached by source text and
        // outlives every launch, so its directory must too.
        let kept = dir.keep();
        let library: &'static Library = Box::leak(Box::new(
            unsafe { Library::new(&so_path) }.map_err(|e| sile_core::Error::Compile(e.to_string()))?,
        ));
        kept_directories().lock().unwrap().push(kept);
        loaded_libraries()
            .lock()
            .unwrap()
            .insert(c_code.to_string(), library);
        Self::launch(library, kernel_name, args, scalars, scalars_i64, launch, llvm_ir_mode)
    }

    /// Call one entry point in an already-loaded kernel library.
    fn launch(
        library: &'static Library,
        kernel_name: &str,
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        llvm_ir_mode: bool,
    ) -> Result<()> {
        unsafe {
            let symbol_name = format!("sile_kernel_{}", kernel_name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| sile_core::Error::Compile(e.to_string()))?;

            let buffers: Vec<*const c_void> =
                args.iter().map(|a| a.mut_ptr as *const c_void).collect();

            let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();

            let num_threadgroups = if llvm_ir_mode {
                launch.grid.iter().map(|dim| *dim as i64).product::<i64>()
            } else {
                launch.grid[0] as i64
            };
            let threads_per_group = if llvm_ir_mode {
                1
            } else if args.is_empty() || args[0].shape.is_empty() {
                256
            } else {
                args[0].shape[0] / num_threadgroups
            };

            let grid: [i64; 3] = [
                launch.grid[0] as i64,
                launch.grid[1] as i64,
                launch.grid[2] as i64,
            ];

            func(
                buffers.as_ptr(),
                num_threadgroups,
                threads_per_group,
                shapes.as_ptr(),
                shapes.len() as i64,
                scalars.as_ptr(),
                scalars_i64.as_ptr(),
                grid.as_ptr(),
            );
        }

        Ok(())
    }

    pub fn execute_llir(
        &self,
        func: &LlvmIrFunction,
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        // Generating the C source costs more than a short kernel runs for, and
        // the same function is lowered on every launch. The IR value is stable
        // for a compiled kernel, so its address identifies the source.
        let c_code = {
            let mut sources = generated_sources().lock().unwrap();
            match sources.get(&(func as *const LlvmIrFunction as usize)) {
                Some(cached) => cached.clone(),
                None => {
                    let generated = Self::compile_llvm_ir_kernel(func)?;
                    sources.insert(func as *const LlvmIrFunction as usize, generated.clone());
                    generated
                }
            }
        };
        self.execute_compiled_source(
            &func.name,
            &c_code,
            args,
            scalars,
            scalars_i64,
            launch,
            true,
        )
    }
}
