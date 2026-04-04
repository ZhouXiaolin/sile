pub mod codegen_llir_metal;
pub mod codegen_metal;

use metal::{CommandQueue, ComputePipelineState, Device, Library};
use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_hir::ParamKind;
use sile_lir::ExecutableKernel;
use sile_llir::Function as LlirFunction;

use crate::codegen_llir_metal::generate as generate_llir_metal;
use crate::codegen_metal::generate;

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

    fn upload_to_device(&self, data: &[f32]) -> metal::Buffer {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device
            .new_buffer(size, metal::MTLResourceOptions::StorageModeShared)
    }

    fn upload_data_to_buffer(&self, buffer: &metal::Buffer, data: &[f32]) {
        let ptr = buffer.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn download_from_device(&self, buffer: &metal::Buffer) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        let len = (buffer.length() / std::mem::size_of::<f32>() as u64) as usize;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    pub fn execute_llir(
        &self,
        func: &LlirFunction,
        param_kinds: &[ParamKind],
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let source = generate_llir_metal(func)?;
        self.execute_source(&func.name, &source, param_kinds, args, launch)
    }

    fn execute_source(
        &self,
        kernel_name: &str,
        source: &str,
        param_kinds: &[ParamKind],
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
    ) -> Result<()> {
        let library = self.compile_shader(source)?;
        let fn_name = format!("sile_kernel_{}", kernel_name);
        let function = self.get_function(&library, &fn_name)?;
        let pipeline = self.create_pipeline(&function)?;

        let mut mtl_buffers: Vec<metal::Buffer> = Vec::new();
        for arg in args {
            let data = unsafe {
                std::slice::from_raw_parts(arg.mut_ptr, arg.shape.iter().product::<i64>() as usize)
            };
            let buf = self.upload_to_device(data);
            self.upload_data_to_buffer(&buf, data);
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
                let result = self.download_from_device(&mtl_buffers[i]);
                let len = arg.shape.iter().product::<i64>() as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(result.as_ptr(), arg.mut_ptr, len);
                }
            }
        }

        Ok(())
    }
}

impl sile_lir::Backend for MetalBackend {
    fn execute(
        &self,
        kernel: &ExecutableKernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let source = generate(&kernel.func, &kernel.abi, &kernel.value_info)?;
        let param_kinds: Vec<_> = kernel.abi.params.iter().map(|param| param.kind).collect();
        self.execute_source(&kernel.name, &source, &param_kinds, args, launch)
    }
}
