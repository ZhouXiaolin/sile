//! The PTX backend: source generation plus the CUDA driver runtime.
//!
//! There is no CUDA toolkit and no `nvcc` in the loop. The driver that ships
//! with the GPU carries a PTX JIT compiler, so a module loaded with
//! `cuModuleLoadDataEx` is compiled by the driver itself. This module binds
//! that API directly out of `nvcuda.dll` (or `libcuda.so.1`) with `libloading`.
//!
//! One launch:
//!
//! 1. load the driver and read the device's compute capability, which selects
//!    the `.target sm_XY` line;
//! 2. create a context, generate the PTX module and JIT it;
//! 3. allocate one device buffer per kernel argument and upload the host bytes,
//!    plus the shapes, float-scalar and integer-scalar descriptors;
//! 4. launch one thread per tile (the grid the C wrapper flattens);
//! 5. synchronize and copy every buffer back.

pub mod codegen;

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::OnceLock;

use libloading::Library;
use sile_core::{Error, KernelArg, LaunchConfig, Result, Stream};
use sile_llvm_ir::Function as LlvmIrFunction;

use crate::emit::{flatten_grid, is_scalars_i64_param, is_scalars_param, is_shapes_param};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;

/// `CUDA_SUCCESS`.
const CUDA_SUCCESS: CuResult = 0;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
const COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CU_JIT_ERROR_LOG_BUFFER`.
const JIT_ERROR_LOG_BUFFER: c_int = 5;
/// `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`.
const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
/// How much of the driver's JIT diagnostic to keep for an error message.
const JIT_ERROR_LOG_BYTES: usize = 16 * 1024;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type FnCuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    *mut CuModule,
    *const c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type FnCuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type FnCuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type FnCuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

/// The driver entry points this backend uses, bound once.
struct Driver {
    _library: Library,
    cu_init: FnCuInit,
    cu_device_get: FnCuDeviceGet,
    cu_device_get_attribute: FnCuDeviceGetAttribute,
    cu_ctx_create: FnCuCtxCreate,
    cu_ctx_destroy: FnCuCtxDestroy,
    cu_module_load_data_ex: FnCuModuleLoadDataEx,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_module_unload: FnCuModuleUnload,
    cu_mem_alloc: FnCuMemAlloc,
    cu_mem_free: FnCuMemFree,
    cu_memcpy_htod: FnCuMemcpyHtoD,
    cu_memcpy_dtoh: FnCuMemcpyDtoH,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    cu_get_error_string: Option<FnCuGetErrorString>,
}

/// The driver library name for this platform.
fn driver_library_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "nvcuda.dll"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "libcuda.so.1"
    }
}

/// Bind one symbol, preferring the versioned spelling when the driver has it.
fn symbol<T: Copy>(library: &Library, names: &[&[u8]]) -> std::result::Result<T, String> {
    for name in names {
        // SAFETY: the handle owns the library for as long as the symbol type is
        // used, and the pointer is copied out here.
        if let Ok(loaded) = unsafe { library.get::<T>(*name) } {
            return Ok(*loaded);
        }
    }
    Err(format!(
        "CUDA driver symbol `{}` is missing",
        String::from_utf8_lossy(names[0])
    ))
}

impl Driver {
    fn load() -> std::result::Result<Self, String> {
        let name = driver_library_name();
        // SAFETY: the library is kept alive in the returned value.
        let library =
            unsafe { Library::new(name) }.map_err(|error| format!("cannot load {name}: {error}"))?;
        // 64-bit drivers export the `_v2` spellings; the unversioned names are
        // kept as a fallback for older libraries.
        let cu_mem_alloc = symbol(&library, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"])?;
        let cu_mem_free = symbol(&library, &[b"cuMemFree_v2\0", b"cuMemFree\0"])?;
        let cu_memcpy_htod = symbol(&library, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"])?;
        let cu_memcpy_dtoh = symbol(&library, &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"])?;
        let cu_ctx_create = symbol(&library, &[b"cuCtxCreate_v2\0", b"cuCtxCreate\0"])?;
        let cu_ctx_destroy = symbol(&library, &[b"cuCtxDestroy_v2\0", b"cuCtxDestroy\0"])?;
        let driver = Self {
            cu_init: symbol(&library, &[b"cuInit\0"])?,
            cu_device_get: symbol(&library, &[b"cuDeviceGet\0"])?,
            cu_device_get_attribute: symbol(&library, &[b"cuDeviceGetAttribute\0"])?,
            cu_ctx_create,
            cu_ctx_destroy,
            cu_module_load_data_ex: symbol(&library, &[b"cuModuleLoadDataEx\0"])?,
            cu_module_get_function: symbol(&library, &[b"cuModuleGetFunction\0"])?,
            cu_module_unload: symbol(&library, &[b"cuModuleUnload\0"])?,
            cu_mem_alloc,
            cu_mem_free,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cu_launch_kernel: symbol(&library, &[b"cuLaunchKernel\0"])?,
            cu_ctx_synchronize: symbol(&library, &[b"cuCtxSynchronize\0"])?,
            // Optional: an error code stands on its own when the lookup fails.
            cu_get_error_string: symbol(&library, &[b"cuGetErrorString\0"]).ok(),
            _library: library,
        };
        Ok(driver)
    }

    fn error_string(&self, code: CuResult) -> String {
        let Some(get_error_string) = self.cu_get_error_string else {
            return "no driver error text".into();
        };
        let mut text: *const c_char = null();
        // SAFETY: the driver writes a static string pointer into `text`.
        if unsafe { get_error_string(code, &mut text) } == CUDA_SUCCESS && !text.is_null() {
            // SAFETY: the pointer is a NUL-terminated static driver string.
            unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
        } else {
            "no driver error text".into()
        }
    }

    fn check(&self, code: CuResult, what: &str) -> Result<()> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::Compile(format!(
                "CUDA driver {what} failed (CUresult={code}: {})",
                self.error_string(code)
            )))
        }
    }

    fn initialize(&self) -> Result<()> {
        // SAFETY: `cuInit` takes the flags word; zero means "no flags".
        self.check(unsafe { (self.cu_init)(0) }, "cuInit")
    }

    fn device(&self, index: c_int) -> Result<CuDevice> {
        let mut device: CuDevice = 0;
        // SAFETY: `device` is a valid out-parameter.
        self.check(unsafe { (self.cu_device_get)(&mut device, index) }, "cuDeviceGet")?;
        Ok(device)
    }

    fn attribute(&self, attribute: c_int, device: CuDevice) -> Result<c_int> {
        let mut value: c_int = 0;
        // SAFETY: `value` is a valid out-parameter.
        self.check(
            unsafe { (self.cu_device_get_attribute)(&mut value, attribute, device) },
            "cuDeviceGetAttribute",
        )?;
        Ok(value)
    }

    /// The compute capability as `sm_XY` writes it: major 6, minor 1 -> `61`.
    fn compute_capability(&self, device: CuDevice) -> Result<u32> {
        let major = self.attribute(COMPUTE_CAPABILITY_MAJOR, device)? as u32;
        let minor = self.attribute(COMPUTE_CAPABILITY_MINOR, device)? as u32;
        Ok(major * 10 + minor)
    }

    fn create_context(&self, device: CuDevice) -> Result<CuContext> {
        let mut context: CuContext = null_mut();
        // SAFETY: the flags word is zero and `context` is a valid out-parameter.
        self.check(
            unsafe { (self.cu_ctx_create)(&mut context, 0, device) },
            "cuCtxCreate",
        )?;
        Ok(context)
    }

    fn destroy_context(&self, context: CuContext) -> Result<()> {
        // SAFETY: the context came from `cuCtxCreate` and is destroyed once.
        self.check(unsafe { (self.cu_ctx_destroy)(context) }, "cuCtxDestroy")
    }
}

fn driver() -> Result<&'static Driver> {
    static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();
    match DRIVER.get_or_init(Driver::load) {
        Ok(driver) => Ok(driver),
        Err(message) => Err(Error::UnsupportedBackend(Box::leak(
            message.clone().into_boxed_str(),
        ))),
    }
}

/// Device memory the launch owns, freed when the launch ends.
struct DeviceAllocations<'a> {
    driver: &'a Driver,
    module: CuModule,
    allocations: Vec<CuDevicePtr>,
    released: bool,
}

impl DeviceAllocations<'_> {
    fn allocate(&mut self, bytes: usize) -> Result<CuDevicePtr> {
        let size = bytes.max(1);
        let mut pointer: CuDevicePtr = 0;
        // SAFETY: `pointer` is a valid out-parameter and the size is non-zero.
        self.driver
            .check(unsafe { (self.driver.cu_mem_alloc)(&mut pointer, size) }, "cuMemAlloc")?;
        self.allocations.push(pointer);
        Ok(pointer)
    }

    fn upload(&self, pointer: CuDevicePtr, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: `pointer` owns at least `bytes.len()` bytes, and `bytes`
        // stays alive for the duration of the copy.
        self.driver.check(
            unsafe {
                (self.driver.cu_memcpy_htod)(
                    pointer,
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                )
            },
            "cuMemcpyHtoD",
        )
    }

    fn read_back(&self, pointer: CuDevicePtr, destination: *mut u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: the caller passes a host buffer of `bytes` bytes, and
        // `pointer` owns at least that many device bytes.
        self.driver.check(
            unsafe {
                (self.driver.cu_memcpy_dtoh)(destination as *mut c_void, pointer, bytes)
            },
            "cuMemcpyDtoH",
        )
    }

    /// Unload the module and free every allocation, including on an error path.
    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        for pointer in self.allocations.drain(..) {
            // SAFETY: every pointer was allocated here and is freed once.
            unsafe {
                (self.driver.cu_mem_free)(pointer);
            }
        }
        // SAFETY: the module came from `cuModuleLoadDataEx` and is unloaded
        // once.
        unsafe {
            (self.driver.cu_module_unload)(self.module);
        }
    }
}

impl Drop for DeviceAllocations<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

/// The bytes of a typed slice, for the driver's byte-oriented copies.
fn as_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: the slice is reinterpreted as bytes of the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// The PTX backend runtime.
pub struct PtxBackend;

impl PtxBackend {
    pub fn new() -> Result<Self> {
        // Fail here rather than at the first launch when there is no driver.
        driver()?;
        Ok(Self)
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
        let driver = driver()?;
        driver.initialize()?;
        let device = driver.device(0)?;
        let arch = driver.compute_capability(device)?;
        let context = driver.create_context(device)?;
        let outcome = (|| -> Result<()> {
            let ptx = codegen::generate_for_arch(func, arch)?;
            let source = CString::new(ptx.as_bytes())
                .map_err(|_| Error::Compile("generated PTX contains a NUL byte".into()))?;
            let module = load_module(driver, &ptx, &source)?;
            let mut allocations = DeviceAllocations {
                driver,
                module,
                allocations: Vec::new(),
                released: false,
            };
            run_launch(
                driver,
                func,
                args,
                scalars,
                scalars_i64,
                launch,
                &mut allocations,
            )
        })();
        let destroyed = driver.destroy_context(context);
        match (outcome, destroyed) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
}

/// JIT one PTX module through the driver and return its handle.
fn load_module(driver: &'static Driver, ptx: &str, source: &CString) -> Result<CuModule> {
    let mut options = [JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
    let mut log = vec![0u8; JIT_ERROR_LOG_BYTES];
    let mut values: [*mut c_void; 2] = [
        log.as_mut_ptr() as *mut c_void,
        log.len() as *mut c_void,
    ];
    let mut module: CuModule = null_mut();
    // SAFETY: the image is a live NUL-terminated string, the option arrays match
    // their declared count, and `module` is a valid out-parameter.
    let code = unsafe {
        (driver.cu_module_load_data_ex)(
            &mut module,
            source.as_ptr() as *const c_void,
            2,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    if code != CUDA_SUCCESS {
        let log = String::from_utf8_lossy(&log);
        let log = log.trim_end_matches('\0').trim();
        let log = if log.is_empty() {
            String::new()
        } else {
            format!("\nJIT log:\n{log}")
        };
        return Err(Error::Compile(format!(
            "CUDA driver rejected the generated PTX (CUresult={code}: {}):\n{ptx}{log}",
            driver.error_string(code)
        )));
    }
    Ok(module)
}

/// Allocate, upload, launch and read back one kernel.
fn run_launch(
    driver: &'static Driver,
    func: &LlvmIrFunction,
    args: &[KernelArg<'_>],
    scalars: &[f32],
    scalars_i64: &[i64],
    launch: &LaunchConfig,
    allocations: &mut DeviceAllocations<'_>,
) -> Result<()> {
    let name = CString::new(codegen::entry_name(&func.name))
        .map_err(|_| Error::Compile("kernel name contains a NUL byte".into()))?;
    let mut function: CuFunction = null_mut();
    // SAFETY: `allocations.module` is the loaded module and `name` is a
    // NUL-terminated entry point name.
    driver.check(
        unsafe {
            (driver.cu_module_get_function)(&mut function, allocations.module, name.as_ptr())
        },
        "cuModuleGetFunction",
    )?;

    let mut buffer_pointers: Vec<CuDevicePtr> = Vec::with_capacity(args.len());
    for arg in args {
        // SAFETY: the host buffer is `len_in_bytes()` bytes as its shape and
        // element size say.
        let bytes = unsafe { std::slice::from_raw_parts(arg.ptr, arg.len_in_bytes()) };
        let pointer = allocations.allocate(bytes.len())?;
        allocations.upload(pointer, bytes)?;
        buffer_pointers.push(pointer);
    }

    let shapes: Vec<i64> = args.iter().flat_map(|arg| arg.shape.iter().copied()).collect();
    let shapes_pointer = allocations.allocate(shapes.len() * std::mem::size_of::<i64>())?;
    allocations.upload(shapes_pointer, as_bytes(&shapes))?;

    let scalars_pointer = allocations.allocate(scalars.len() * std::mem::size_of::<f32>())?;
    allocations.upload(scalars_pointer, as_bytes(scalars))?;

    let scalars_i64_pointer =
        allocations.allocate(scalars_i64.len() * std::mem::size_of::<i64>())?;
    allocations.upload(scalars_i64_pointer, as_bytes(scalars_i64))?;

    let mut values: Vec<u64> = Vec::with_capacity(func.params.len());
    let mut buffer_index = 0usize;
    for param in &func.params {
        let value = if is_shapes_param(param) {
            shapes_pointer
        } else if is_scalars_param(param) {
            scalars_pointer
        } else if is_scalars_i64_param(param) {
            scalars_i64_pointer
        } else {
            let pointer = buffer_pointers[buffer_index];
            buffer_index += 1;
            pointer
        };
        values.push(value);
    }
    // `cuLaunchKernel` wants one pointer per argument, each pointing at that
    // argument's value. The vector does not move after this point.
    let mut parameter_pointers: Vec<*mut c_void> = values
        .iter_mut()
        .map(|value| value as *mut u64 as *mut c_void)
        .collect();

    let grid = flatten_grid(func, launch);
    let grid = [grid[0].max(1), grid[1].max(1), grid[2].max(1)];
    // One CUDA thread runs one tile or program, which is what the codegen's
    // index decode expects.
    // SAFETY: every pointer in `parameter_pointers` stays valid for the call
    // and describes a kernel parameter of the generated signature.
    driver.check(
        unsafe {
            (driver.cu_launch_kernel)(
                function,
                grid[0],
                grid[1],
                grid[2],
                1,
                1,
                1,
                0,
                null_mut(),
                parameter_pointers.as_mut_ptr(),
                null_mut(),
            )
        },
        "cuLaunchKernel",
    )?;
    // SAFETY: no arguments; it synchronizes the current context.
    driver.check(unsafe { (driver.cu_ctx_synchronize)() }, "cuCtxSynchronize")?;

    for (index, arg) in args.iter().enumerate() {
        allocations.read_back(buffer_pointers[index], arg.mut_ptr, arg.len_in_bytes())?;
    }
    Ok(())
}
