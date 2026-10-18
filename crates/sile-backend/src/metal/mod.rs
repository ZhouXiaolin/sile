pub mod codegen;

use metal::{CommandQueue, ComputePipelineState, Device, Library};
use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_hir::ParamKind;
use sile_llvm_ir::Function as LlvmIrFunction;

use self::codegen::generate as generate_llvm_ir_metal;

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| sile_core::Error::UnsupportedBackend("no Metal device found"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    fn compile_shader(&self, source: &str) -> Result<Library> {
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

    pub fn execute_llir(
        &self,
        func: &LlvmIrFunction,
        param_kinds: &[ParamKind],
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let source = generate_llvm_ir_metal(func)?;
        let launch = self.normalize_llvm_ir_launch(func, launch);
        self.execute_source(
            &func.name,
            &source,
            param_kinds,
            args,
            scalars,
            scalars_i64,
            &launch,
        )
    }

    fn normalize_llvm_ir_launch(
        &self,
        func: &LlvmIrFunction,
        launch: &LaunchConfig,
    ) -> LaunchConfig {
        LaunchConfig {
            grid: crate::emit::flatten_grid(func, launch),
        }
    }

    fn execute_source(
        &self,
        kernel_name: &str,
        source: &str,
        param_kinds: &[ParamKind],
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
    ) -> Result<()> {
        let library = self.compile_shader(source)?;
        let fn_name = format!("sile_kernel_{}", kernel_name);
        let function = self.get_function(&library, &fn_name)?;
        let pipeline = self.create_pipeline(&function)?;

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
        // order the LLVM IR signature declares them. A kernel with no scalar
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
        let threads_per_threadgroup = metal::MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(
            metal::MTLSize::new(grid_x, grid_y, grid_z),
            threads_per_threadgroup,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        for (i, arg) in args.iter().enumerate() {
            let is_output = matches!(param_kinds.get(i), Some(ParamKind::Output));
            if is_output {
                self.download_bytes(&mtl_buffers[i], arg.mut_ptr, arg.len_in_bytes());
            }
        }

        Ok(())
    }
}
