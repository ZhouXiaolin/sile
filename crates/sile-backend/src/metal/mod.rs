pub mod codegen;

#[cfg(target_os = "macos")]
use metal::{CommandQueue, ComputePipelineState, Device, Library};
#[cfg(target_os = "macos")]
use sile_core::{KernelArg, LaunchConfig, Result};
#[cfg(target_os = "macos")]
use sile_lsir::Function as LsirFunction;

#[cfg(target_os = "macos")]
pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| sile_core::Error::UnsupportedBackend("no Metal device found"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    fn compile_shader(&self, source: &str) -> Result<Library> {
        // Note on fast-math: AIR's default fast-math folds `x + (-INFINITY)`
        // into `x`, which once swallowed the causal mask whole (see
        // examples/probe_metal.rs). The defense lives in the IR — instcombine
        // rewrites `x + select(mask, 0, -inf)` into `select(mask, x, -inf)`,
        // a shape fast-math leaves alone — so the compiler options stay at
        // their fast defaults.
        let options = metal::CompileOptions::new();
        self.device
            .new_library_with_source(source, &options)
            .map_err(|e| sile_core::Error::Compile(e.to_string()))
    }

    fn get_function(&self, library: &Library, name: &str) -> Result<metal::Function> {
        library
            .get_function(name, None)
            .map_err(|e| sile_core::Error::Compile(e.to_string()))
    }

    fn create_pipeline(&self, function: &metal::Function) -> Result<ComputePipelineState> {
        let descriptor = metal::ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(function));
        self.device
            .new_compute_pipeline_state(&descriptor)
            .map_err(|e| sile_core::Error::Compile(e.to_string()))
    }

    fn upload_bytes(&self, bytes: &[u8]) -> metal::Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len() as u64, metal::MTLResourceOptions::StorageModeShared);
        let ptr = buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        }
        buffer
    }

    fn download_bytes(&self, buffer: &metal::Buffer, out: *mut u8, len: usize) {
        let ptr = buffer.contents() as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out, len);
        }
    }

    /// Dispatch one compiled kernel.
    ///
    /// The source is the artifact [`crate::compile`] produced — the mapping
    /// has already run, and `threads` is the threadgroup width its policy
    /// decided, serial fallback included. One threadgroup owns one tile, so
    /// the dispatch owes the kernel exactly that count.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_source(
        &self,
        func: &LsirFunction,
        source: &str,
        outputs: &[bool],
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        threads: u32,
    ) -> Result<()> {
        let launch = LaunchConfig {
            // The kernel reads the grid per axis (`threadgroup_position_in_grid`),
            // so the dispatch keeps the shape the launcher resolved.
            grid: launch.grid,
        };
        self.dispatch(
            &func.name,
            source,
            outputs,
            args,
            scalars,
            scalars_i64,
            &launch,
            threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        kernel_name: &str,
        source: &str,
        outputs: &[bool],
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        threads: u32,
    ) -> Result<()> {
        let library = self.compile_shader(source)?;
        let fn_name = format!("sile_kernel_{}", kernel_name);
        let function = self.get_function(&library, &fn_name)?;
        let pipeline = self.create_pipeline(&function)?;
        // Fail closed on an adapter that cannot fit the mapped threadgroup:
        // dispatching fewer threads than the kernel's strips assume would
        // silently drop the elements those threads own.
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        if u64::from(threads) > max_threads {
            return Err(sile_core::Error::Compile(format!(
                "the mapping wants {threads} threads per threadgroup but this \
                 adapter allows at most {max_threads}"
            )));
        }

        let mut mtl_buffers: Vec<metal::Buffer> = Vec::new();
        for arg in args {
            // The bytes travel as the host holds them, so a half buffer is
            // half the bytes and the kernel reads its own element width.
            let bytes =
                unsafe { std::slice::from_raw_parts(arg.ptr, arg.len_in_bytes()) };
            let buf = self.upload_bytes(bytes);
            mtl_buffers.push(buf);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        for (i, buffer) in mtl_buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();
        let shapes_buf = self.device.new_buffer(
            (shapes.len() * std::mem::size_of::<i64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let shapes_ptr = shapes_buf.contents() as *mut i64;
        unsafe {
            std::ptr::copy_nonoverlapping(shapes.as_ptr(), shapes_ptr, shapes.len());
        }
        encoder.set_buffer(args.len() as u64, Some(&shapes_buf), 0);

        // The scalar descriptor binds after the shape descriptor, matching the
        // order the LSIR signature declares them. A kernel with no scalar
        // argument still carries the parameter, so the slot is bound to an
        // empty buffer rather than skipped.
        let scalars_buf = self.device.new_buffer(
            (scalars.len().max(1) * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        if !scalars.is_empty() {
            let scalars_ptr = scalars_buf.contents() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(scalars.as_ptr(), scalars_ptr, scalars.len());
            }
        }
        encoder.set_buffer((args.len() + 1) as u64, Some(&scalars_buf), 0);

        // The integer scalar descriptor binds after the float one. A kernel
        // that declares no integer scalar does not read the slot, and a bound
        // buffer nothing reads costs a few bytes and no branch.
        let scalars_i64_buf = self.device.new_buffer(
            (scalars_i64.len().max(1) * std::mem::size_of::<i64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        if !scalars_i64.is_empty() {
            let ptr = scalars_i64_buf.contents() as *mut i64;
            unsafe {
                std::ptr::copy_nonoverlapping(scalars_i64.as_ptr(), ptr, scalars_i64.len());
            }
        }
        encoder.set_buffer((args.len() + 2) as u64, Some(&scalars_i64_buf), 0);

        let grid_x = launch.grid[0] as u64;
        let grid_y = launch.grid[1] as u64;
        let grid_z = launch.grid[2] as u64;
        // One threadgroup owns one tile; the flat spread runs along x, the
        // axis the mapping's strips and barriers are written against.
        let threads_per_threadgroup = metal::MTLSize::new(u64::from(threads), 1, 1);

        encoder.dispatch_thread_groups(
            metal::MTLSize::new(grid_x, grid_y, grid_z),
            threads_per_threadgroup,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        for (i, arg) in args.iter().enumerate() {
            let is_output = outputs.get(i).copied().unwrap_or(false);
            if is_output {
                self.download_bytes(&mtl_buffers[i], arg.mut_ptr, arg.len_in_bytes());
            }
        }

        Ok(())
    }
}
