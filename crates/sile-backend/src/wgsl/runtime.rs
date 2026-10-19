//! The WGPU device runtime: one compute dispatch over the same LLVM IR.
//!
//! C and Metal each need a driver and a compiler. WGPU needs an instance, an
//! adapter, a device, a queue, one buffer per kernel argument, one bind group
//! and one compute pass. The shapes descriptor is bound last, which is the
//! order `codegen` gives its bindings, and the launch grid is written directly
//! to `dispatch_workgroups`.

use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_llvm_ir::Function as LlvmIrFunction;

use crate::emit::{flatten_grid, is_descriptor_param};

/// Bytes of a typed slice, which is the only way `write_buffer` takes data.
fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// `request_device` reports its failure as one enum, and the sile error type
/// takes a `&str`, so a stable description is needed.
fn request_device_error(error: impl std::fmt::Display) -> &'static str {
    Box::leak(error.to_string().into_boxed_str())
}


use super::codegen::generate as generate_wgsl;

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The optional features the device was created with, so the runtime can
    /// reject a kernel the device cannot run instead of letting wgpu panic.
    features: wgpu::Features,
    /// The device's limits, so a kernel too large for this adapter is named
    /// here rather than failing inside the driver.
    limits: wgpu::Limits,
}

/// True when the type is, contains, or points at an `f16`.
fn contains_f16(ty: &sile_llvm_ir::Type) -> bool {
    use sile_llvm_ir::Type;
    match ty {
        Type::F16 => true,
        Type::Ptr { pointee, .. } => contains_f16(pointee),
        Type::Vector { elem, .. } | Type::Array { elem, .. } => contains_f16(elem),
        _ => false,
    }
}

/// True when any value or parameter of the function is an `f16`.
fn uses_f16(func: &LlvmIrFunction) -> bool {
    func.params.iter().any(|param| contains_f16(&param.ty))
        || func.blocks.iter().any(|block| {
            block.params.iter().any(|param| contains_f16(&param.ty))
                || block.insts.iter().any(|inst| contains_f16(&inst.ty))
        })
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        })
        .map_err(|_| sile_core::Error::UnsupportedBackend("no WGPU adapter found"))?;
        // The lowering makes every index an i64, so a device without 64-bit
        // shader integers cannot run any kernel. Asking for the intersection of
        // optional features would silently drop this one, and the shader would
        // fail validation later as if the kernel were at fault.
        let (device, queue) = {
            let available = adapter.features();
            if !available.contains(wgpu::Features::SHADER_INT64) {
                return Err(sile_core::Error::UnsupportedBackend(
                    "the WGPU adapter has no SHADER_INT64 feature, which every kernel needs for its i64 indices",
                ));
            }
            // The lowering makes every index an i64, which is an optional
            // WGSL capability rather than a default, and a half buffer
            // is the same kind of ask. Take what the adapter offers here;
            // `execute_llir` rejects an f16 kernel when this one is missing.
            let required_features =
                available.intersection(wgpu::Features::SHADER_INT64 | wgpu::Features::SHADER_F16);
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("sile wgpu device"),
                required_features,
                // The device starts at conservative defaults, which understate
                // what the adapter can do: a kernel that fits the hardware
                // would be rejected by the software limit instead. Ask for what
                // the adapter reports.
                required_limits: adapter.limits(),
                ..Default::default()
            }))
            .map_err(|e| sile_core::Error::UnsupportedBackend(request_device_error(e)))?
        };
        let features = device.features();
        let limits = device.limits();
        Ok(Self {
            device,
            queue,
            features,
            limits,
        })
    }

    pub fn execute_llir(
        &self,
        func: &LlvmIrFunction,
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()> {
        // An adapter without `f16` support rejects the shader inside wgpu's
        // validation, which panics on the stream worker. Name the real cause
        // here instead: the kernel is fine, the hardware cannot run it.
        if uses_f16(func) && !self.features.contains(wgpu::Features::SHADER_F16) {
            return Err(sile_core::Error::UnsupportedBackend(
                "this kernel uses f16 buffers, but the WGPU adapter has no SHADER_F16 feature",
            ));
        }
        let source = generate_wgsl(func)?;
        super::validate(&source)?;

        // A function-scope `var` array lives in the workgroup's storage, and
        // the driver refuses a pipeline over the adapter's limit without
        // saying so — the dispatch fails later as `Encoder is invalid`. Name
        // the real cause here, the way the f16 check above does.
        let local_bytes = super::local_array_bytes(&source);
        let limit = self.limits.max_compute_workgroup_storage_size;
        if local_bytes > limit {
            return Err(sile_core::Error::UnsupportedBackend(Box::leak(
                format!(
                    "this kernel needs {local_bytes} bytes of workgroup storage, \
                     but the WGPU adapter allows {limit}"
                )
                .into_boxed_str(),
            )));
        }

        // Every kernel buffer becomes one storage buffer, uploaded after the
        // host leaves its data in shared storage. The bytes travel as the host
        // holds them, so a half buffer is half the bytes and the kernel reads
        // its own element width. The shapes descriptor is the last binding,
        // exactly as `codegen` numbered them.
        let mut buffers = Vec::with_capacity(args.len());
        for arg in args {
            let bytes = unsafe { std::slice::from_raw_parts(arg.ptr, arg.len_in_bytes()) };
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sile kernel buffer"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, bytes);
            buffers.push(buffer);
        }

        // A kernel declares the integer scalar descriptor only when it takes
        // an integer scalar, so the host asks the shader rather than assuming.
        let has_scalars_i64 = func
            .params
            .iter()
            .any(|param| crate::emit::is_scalars_i64_param(param));

        let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();
        let shapes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sile shapes"),
            size: (shapes.len().max(1) * std::mem::size_of::<i64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&shapes_buffer, 0, as_bytes(&shapes));

        // The scalar descriptor binds after the shape descriptor, matching the
        // order `codegen` numbered the bindings in. A kernel with no scalar
        // argument still carries the parameter, so the buffer exists and holds
        // one unused element.
        let scalars_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sile scalars"),
            size: (scalars.len().max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !scalars.is_empty() {
            self.queue
                .write_buffer(&scalars_buffer, 0, as_bytes(scalars));
        }

        // The integer scalar descriptor binds after the float one, and only a
        // kernel that declares one gets a binding: a bind group entry without
        // a binding in the shader is an error, not an unused slot.
        let scalars_i64_buffer = if has_scalars_i64 {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sile scalars i64"),
                size: (scalars_i64.len().max(1) * std::mem::size_of::<i64>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !scalars_i64.is_empty() {
                self.queue.write_buffer(&buffer, 0, as_bytes(scalars_i64));
            }
            Some(buffer)
        } else {
            None
        };

        // A float atomic add binds the same storage a second time as an integer
        // atomic, so the alias bindings come after the kernel buffers and the
        // shape descriptor and cover the buffers `codegen` names.
        let aliases = super::codegen::atomic_alias_bindings(func);

        // The two descriptors are the read-only bindings, and the kernel
        // buffers and the atomic aliases are the read-write ones. `codegen`
        // puts the descriptors first among the trailing bindings, shapes then
        // scalars, so their numbers are the count of kernel buffers and the one
        // after it, and the aliases follow both.
        // The descriptors are the trailing bindings and the aliases come after
        // them, so every number after the kernel buffers shifts with the count
        // of descriptors the kernel declares.
        let descriptor_count = func
            .params
            .iter()
            .filter(|param| is_descriptor_param(param))
            .count();
        let read_write = |index: usize| wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: index >= args.len() && index < args.len() + descriptor_count,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let binding_count = args.len() + descriptor_count + aliases.len();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sile bind group layout"),
                entries: &(0..binding_count).map(read_write).collect::<Vec<_>>(),
            });
        let entries = (0..binding_count)
            .map(|index| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: if index < args.len() {
                    buffers[index].as_entire_binding()
                } else if index == args.len() {
                    shapes_buffer.as_entire_binding()
                } else if index == args.len() + 1 {
                    scalars_buffer.as_entire_binding()
                } else if index < args.len() + descriptor_count {
                    scalars_i64_buffer
                        .as_ref()
                        .expect("an integer scalar binding needs its buffer")
                        .as_entire_binding()
                } else {
                    // The alias of one kernel buffer, bound over the same
                    // storage. `codegen` numbers these after the descriptors in
                    // the order it declared the buffers, which is the order of
                    // `aliases`.
                    let buffer = aliases[index - args.len() - descriptor_count];
                    buffers[buffer].as_entire_binding()
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
        // A driver that rejects the module reports the reason on the device's
        // error scope; without this the pipeline is silently invalid and the
        // failure surfaces as `Encoder is invalid` at submit time.
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sile wgsl"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.device.poll(wgpu::PollType::Wait);
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(sile_core::Error::Compile(format!(
                "the WGPU device rejected the shader: {error}"
            )));
        }
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
        self.device.poll(wgpu::PollType::Wait);
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(sile_core::Error::Compile(format!(
                "the WGPU device rejected the pipeline: {error}"
            )));
        }

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
            let grid = flatten_grid(func, launch);
            pass.dispatch_workgroups(grid[0].max(1), grid[1].max(1), grid[2].max(1));
        }
        self.queue.submit(Some(encoder.finish()));

        // A kernel writes its output in place, so the host reads the buffer
        // back. The read-back is a byte copy: the device holds the same
        // little-endian elements the host laid down, f32 or half alike. The
        // stream names the device the launch went to, and today a stream is
        // one device, so the read-back is the whole synchronization.
        let _ = stream;
        for (index, arg) in args.iter().enumerate() {
            let byte_len = arg.len_in_bytes();
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sile readback"),
                size: byte_len as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut copy = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sile copy encoder"),
                });
            copy.copy_buffer_to_buffer(&buffers[index], 0, &staging, 0, byte_len as u64);
            self.queue.submit(Some(copy.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait);
            let mapped = slice.get_mapped_range();
            unsafe {
                std::ptr::copy_nonoverlapping(mapped.as_ptr(), arg.mut_ptr, byte_len);
            }
            drop(mapped);
            staging.unmap();
        }

        Ok(())
    }
}
