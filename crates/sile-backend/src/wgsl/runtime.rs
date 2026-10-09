//! The WGPU device runtime: one compute dispatch over the same LLVM IR.
//!
//! C and Metal each need a driver and a compiler. WGPU needs an instance, an
//! adapter, a device, a queue, one buffer per kernel argument, one bind group
//! and one compute pass. The shapes descriptor is bound last, which is the
//! order `codegen` gives its bindings, and the launch grid is written directly
//! to `dispatch_workgroups`.

use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_llvm_ir::Function as LlvmIrFunction;

/// Bytes of a typed slice, which is the only way `write_buffer` takes data.
fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    }
}

/// `request_device` reports its failure as one enum, and the sile error type
/// takes a `&str`, so a stable description is needed.
fn request_device_error(error: impl std::fmt::Display) -> &'static str {
    Box::leak(error.to_string().into_boxed_str())
}

/// Read a typed vector back out of mapped bytes.
fn from_bytes<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / std::mem::size_of::<T>();
    let mut out = vec![T::default(); count];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

use super::codegen::generate as generate_wgsl;

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        })
        .ok_or_else(|| {
            sile_core::Error::UnsupportedBackend("no WGPU adapter found")
        })?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sile wgpu device"),
                // The lowering makes every index an i64, which is an optional
                // WGSL capability rather than a default. Metal, Vulkan and DX12
                // all have it, so the kernel asks for it instead of narrowing
                // every index to 32 bits.
                required_features: adapter
                    .features()
                    .intersection(wgpu::Features::SHADER_INT64),
                ..Default::default()
            },
            None,
        ))
        .map_err(|e| sile_core::Error::UnsupportedBackend(request_device_error(e)))?;
        Ok(Self { device, queue })
    }

    pub fn execute_llir(
        &self,
        func: &LlvmIrFunction,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()> {
        let source = generate_wgsl(func)?;
        super::validate(&source)?;

        // Every kernel buffer becomes one storage buffer, uploaded after the
        // host leaves its data in shared storage. The shapes descriptor is the
        // last binding, exactly as `codegen` numbered them.
        let mut buffers = Vec::with_capacity(args.len());
        for arg in args {
            let data = unsafe {
                std::slice::from_raw_parts(arg.ptr, arg.shape.iter().product::<i64>() as usize)
            };
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sile kernel buffer"),
                size: (data.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, as_bytes(data));
            buffers.push(buffer);
        }

        let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();
        let shapes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sile shapes"),
            size: (shapes.len() * std::mem::size_of::<i64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&shapes_buffer, 0, as_bytes(&shapes));

        // The shape descriptor is the one read-only binding, and the kernel
        // buffers are the read-write ones. `codegen` puts the descriptor last,
        // so its binding number is the count of kernel buffers.
        let read_write = |index: usize| wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: index == args.len(),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sile bind group layout"),
                entries: &(0..=args.len()).map(read_write).collect::<Vec<_>>(),
            });
        let entries = (0..=args.len())
            .map(|index| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: if index == args.len() {
                    shapes_buffer.as_entire_binding()
                } else {
                    buffers[index].as_entire_binding()
                },
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sile bind group"),
            layout: &layout,
            entries: &entries,
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sile pipeline layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sile wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sile compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(&format!("sile_kernel_{}", func.name)),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sile command encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sile compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                launch.grid[0].max(1),
                launch.grid[1].max(1),
                launch.grid[2].max(1),
            );
        }
        self.queue.submit(Some(encoder.finish()));

        // A kernel writes its output in place, so the host reads the buffer
        // back. The stream names the device the launch went to, and today a
        // stream is one device, so the read-back is the whole synchronization.
        let _ = stream;
        for (index, arg) in args.iter().enumerate() {
            let len = arg.shape.iter().product::<i64>() as usize;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sile readback"),
                size: (len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut copy = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sile copy encoder"),
                });
            copy.copy_buffer_to_buffer(
                &buffers[index],
                0,
                &staging,
                0,
                (len * std::mem::size_of::<f32>()) as u64,
            );
            self.queue.submit(Some(copy.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::Maintain::Wait);
            let mapped = slice.get_mapped_range();
            let values: Vec<f32> = from_bytes(&mapped);
            drop(mapped);
            staging.unmap();
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), arg.mut_ptr, len);
            }
        }

        Ok(())
    }
}
