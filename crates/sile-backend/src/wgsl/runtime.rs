//! The WGPU device runtime: one compute dispatch over the same LSIR.
//!
//! C and Metal each need a driver and a compiler. WGPU needs an instance, an
//! adapter, a device, a queue, one buffer per kernel argument, one bind group
//! and one compute pass. The shapes descriptor is bound last, which is the
//! order `codegen` gives its bindings, and the launch grid is written directly
//! to `dispatch_workgroups`.

use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_lsir::Function as LsirFunction;

use crate::emit::is_descriptor_param;

/// Bytes of a typed slice, which is the only way `write_buffer` takes data.
fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// `request_device` reports its failure as one enum, and the sile error type
/// takes a `&str`, so a stable description is needed.
fn request_device_error(error: impl std::fmt::Display) -> &'static str {
    Box::leak(error.to_string().into_boxed_str())
}


use super::codegen::atomic_alias_bindings;
use crate::{BackendArtifact, CodegenTarget, compile};

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The optional features the device was created with, so the runtime can
    /// reject a kernel the device cannot run instead of letting wgpu panic.
    features: wgpu::Features,
    /// The device's limits, so a kernel too large for this adapter is named
    /// here rather than failing inside the driver.
    limits: wgpu::Limits,
    /// Compiled kernels keyed by the WGSL source that produced them. The same
    /// kernel launches repeatedly, and the driver-side compile is launch cost,
    /// not kernel work: a bench of repeated launches would otherwise time the
    /// compiler a hundred times. The layout goes with the pipeline, because
    /// every launch builds its bind group over it.
    pipelines: std::sync::Mutex<std::collections::HashMap<String, (wgpu::BindGroupLayout, wgpu::ComputePipeline)>>,
    /// Device storage keyed by the host range it mirrors, so a repeated launch
    /// reuses its buffers instead of reallocating — each allocation is
    /// zero-initialized, and the next launch throws it away.
    buffers: std::sync::Mutex<std::collections::HashMap<(usize, usize), (wgpu::Buffer, bool)>>,
    /// MAP_READ staging keyed by the host range, for the same reason.
    staging: std::sync::Mutex<std::collections::HashMap<(usize, usize), wgpu::Buffer>>,
    descriptors: std::sync::Mutex<std::collections::HashMap<u64, wgpu::Buffer>>,
}

/// True when the type is, contains, or points at an `f16`.
fn contains_f16(ty: &sile_lsir::Type) -> bool {
    use sile_lsir::Type;
    match ty {
        Type::F16 => true,
        Type::Ptr { pointee, .. } => contains_f16(pointee),
        Type::Vector { elem, .. } | Type::Array { elem, .. } => contains_f16(elem),
        _ => false,
    }
}

/// The workgroup size the source's entry point declares, read back from the
/// text.
///
/// The declaration is the one fact the dispatch and the limits check both
/// need, and the spelled source is where it lives once the mapping has run.
fn declared_workgroup_size(source: &str) -> Option<u32> {
    let line = source
        .lines()
        .find(|line| line.contains("@workgroup_size("))?;
    let (_, rest) = line.split_once("@workgroup_size(")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// True when any value or parameter of the function is an `f16`.
fn uses_f16(func: &LsirFunction) -> bool {
    func.params.iter().any(|param| contains_f16(&param.ty))
        || func.blocks.iter().any(|block| {
            block.params.iter().any(|param| contains_f16(&param.ty))
                || block.insts.iter().any(|inst| contains_f16(&inst.ty))
        })
}

/// The adapter a WGSL stream runs on.
///
/// The integrated GPU is the default: on one vendor a shader the driver
/// chokes on takes the kernel down with a full-chip reset, and a stream that
/// just wants to run a kernel has no business asking the discrete card for
/// it. `SILE_WGPU_ADAPTER` names a card by substring (`nvidia`, `AMD`, a
/// model name) when the discrete one is the ask, and a name that matches
/// nothing is an error rather than a silent fallback.
fn pick_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if let Ok(want) = std::env::var("SILE_WGPU_ADAPTER") {
        let want = want.to_lowercase();
        return adapters
            .iter()
            .find(|adapter| adapter.get_info().name.to_lowercase().contains(&want))
            .cloned()
            .ok_or_else(|| {
                sile_core::Error::UnsupportedBackend(
                    "no WGPU adapter name matches SILE_WGPU_ADAPTER",
                )
            });
    }
    adapters
        .iter()
        .find(|adapter| adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
        .cloned()
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()
        })
        .ok_or_else(|| sile_core::Error::UnsupportedBackend("no WGPU adapter found"))
}

impl WgpuBackend {
    /// The process's one backend. A launch builds its launcher fresh, and a
    /// fresh `WgpuBackend` per launch would rebuild the whole device —
    /// instance, adapter, `request_device` — and drop the pipeline cache with
    /// it. The device options are a unit type, so one shared instance is the
    /// whole table.
    pub fn shared() -> Result<std::sync::Arc<Self>> {
        static SHARED: std::sync::OnceLock<
            std::sync::Mutex<Option<std::sync::Arc<WgpuBackend>>>,
        > = std::sync::OnceLock::new();
        let cell = SHARED.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = cell
            .lock()
            .expect("the shared WGPU backend lock is poisoned");
        if let Some(backend) = guard.as_ref() {
            return Ok(backend.clone());
        }
        let backend = std::sync::Arc::new(Self::new()?);
        *guard = Some(backend.clone());
        Ok(backend)
    }

    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pick_adapter(&instance)?;
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
            pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
            buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
            staging: std::sync::Mutex::new(std::collections::HashMap::new()),
            descriptors: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }
    /// The device mirror of the host range `(ptr, len)`, allocated on first
    /// use, with whether the mirror has ever been filled from the host.
    ///
    /// A written argument's mirror needs exactly one fill: the kernel
    /// overwrites it and the launch reads the result back, which leaves the
    /// host and the device holding the same bytes — so the next launch's
    /// upload would move megabytes to rewrite them with what is already
    /// there. A read-only argument has no such sync point, and its upload
    /// stays per launch.
    fn device_buffer_entry(&self, ptr: usize, len: usize) -> (wgpu::Buffer, bool) {
        let mut cache = self
            .buffers
            .lock()
            .expect("the device buffer cache lock is poisoned");
        if let Some((buffer, uploaded)) = cache.get(&(ptr, len)) {
            return (buffer.clone(), *uploaded);
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sile kernel buffer"),
            size: len as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cache.insert((ptr, len), (buffer.clone(), false));
        (buffer, false)
    }

    /// Record that the device mirror of `(ptr, len)` now holds the host bytes.
    fn mark_buffer_uploaded(&self, ptr: usize, len: usize) {
        let mut cache = self
            .buffers
            .lock()
            .expect("the device buffer cache lock is poisoned");
        if let Some((_, uploaded)) = cache.get_mut(&(ptr, len)) {
            *uploaded = true;
        }
    }

    /// The MAP_READ staging mirror of the host range `(ptr, len)`.
    fn staging_buffer(&self, ptr: usize, len: usize) -> wgpu::Buffer {
        let mut cache = self
            .staging
            .lock()
            .expect("the staging cache lock is poisoned");
        if let Some(buffer) = cache.get(&(ptr, len)) {
            return buffer.clone();
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sile readback"),
            size: len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cache.insert((ptr, len), buffer.clone());
        buffer
    }

    /// A reusable descriptor buffer of `size` bytes, filled per launch.
    ///
    /// The shape and scalar descriptors change content every launch but never
    /// size, and an allocation per launch costs the launch a device-memory
    /// allocation for a few dozen bytes.
    fn descriptor_buffer(&self, size: u64, label: &str) -> wgpu::Buffer {
        let mut cache = self
            .descriptors
            .lock()
            .expect("the descriptor buffer cache lock is poisoned");
        if let Some(buffer) = cache.get(&size) {
            return buffer.clone();
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cache.insert(size, buffer.clone());
        buffer
    }

    pub fn execute_llir(
        &self,
        func: &LsirFunction,
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
        // The compile entry owns the mapping and both contract checks, so the
        // runtime never spells a kernel the pipeline did not verify.
        let source = match compile(func, CodegenTarget::Wgsl)? {
            BackendArtifact::WgslSource(source) => source,
            _ => unreachable!("the WGSL target spells WGSL source"),
        };
        self.execute_source(&source, func, args, scalars, scalars_i64, launch, stream)
    }

    /// Dispatch a kernel from its spelled WGSL and the function it came from.
    ///
    /// The launcher holds the source of one kernel once — the source is a pure
    /// function of the kernel's LSIR — so a launch loop pays for the mapping,
    /// the emitter and the validation a single time and calls this per
    /// iteration.
    pub fn execute_source(
        &self,
        source: &str,
        func: &LsirFunction,
        args: &[KernelArg<'_>],
        scalars: &[f32],
        scalars_i64: &[i64],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()> {
        // The binding shapes come from the function alone, so they are settled
        // before the cache: both the miss path's layout and every launch's
        // bind group read them.
        let has_scalars_i64 = func
            .params
            .iter()
            .any(|param| crate::emit::is_scalars_i64_param(param));
        // A float atomic add binds the same storage a second time as an integer
        // atomic, so the alias bindings come after the kernel buffers and the
        // shape descriptor and cover the buffers `codegen` names.
        let aliases = atomic_alias_bindings(func);
        // The two descriptors are the read-only bindings, and the kernel
        // buffers and the atomic aliases are the read-write ones. `codegen`
        // puts the descriptors first among the trailing bindings, shapes then
        // scalars, so their numbers are the count of kernel buffers and the one
        // after it, and the aliases follow both.
        let descriptor_count = func
            .params
            .iter()
            .filter(|param| is_descriptor_param(param))
            .count();

        // One source names one kernel: the validation, the limit checks and
        // the driver-side compile all read the source alone, so the cache
        // turns a repeated launch into the dispatch it is. A first launch
        // runs the whole gate once; the same source again takes the pipeline.
        let (layout, pipeline) = {
            let mut cache = self
                .pipelines
                .lock()
                .expect("the pipeline cache lock is poisoned");
            match cache.get(source) {
                Some((layout, pipeline)) => (layout.clone(), pipeline.clone()),
                None => {
                    super::validate(source)?;

                    // The entry point declares its workgroup size from the mapping
                    // policy, and an adapter caps it along every axis. The driver would
                    // refuse the pipeline without naming the cause, so name it here.
                    if let Some(threads) = declared_workgroup_size(source)
                        && threads > self.limits.max_compute_workgroup_size_x
                    {
                        return Err(sile_core::Error::UnsupportedBackend(Box::leak(
                            format!(
                                "this kernel maps {} threads to a workgroup, \
                                 but the WGPU adapter allows {}",
                                threads, self.limits.max_compute_workgroup_size_x
                            )
                            .into_boxed_str(),
                        )));
                    }

                    // A `var<workgroup>` array lives in the workgroup's shared storage,
                    // and the driver refuses a pipeline over the adapter's limit without
                    // saying so — the dispatch fails later as `Encoder is invalid`. Name
                    // the real cause here, the way the f16 check above does.
                    let local_bytes = super::local_array_bytes(source);
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

                    let read_write = |index: usize| wgpu::BindGroupLayoutEntry {
                        binding: index as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: index >= args.len()
                                    && index < args.len() + descriptor_count,
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
                            source: wgpu::ShaderSource::Wgsl(source.to_string().into()),
                        });
                    let _ = self.device.poll(wgpu::PollType::Wait);
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
                    let _ = self.device.poll(wgpu::PollType::Wait);
                    if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
                        return Err(sile_core::Error::Compile(format!(
                            "the WGPU device rejected the pipeline: {error}"
                        )));
                    }
                    cache.insert(source.to_string(), (layout.clone(), pipeline.clone()));
                    (layout, pipeline)
                }
            }
        };

        // Every kernel buffer becomes one storage buffer, uploaded after the
        // host leaves its data in shared storage. The bytes travel as the host
        // holds them, so a half buffer is half the bytes and the kernel reads
        // its own element width. The shapes descriptor is the last binding,
        // exactly as `codegen` numbered them.
        // Kernel buffers live across launches, keyed by the host range they
        // mirror: a repeated launch of the same kernel hands the same host
        // buffers back, and reallocating device storage per launch — each
        // allocation zero-initialized — would put allocation, not the kernel,
        // in the timing loop. `write_buffer` still runs per launch: the host
        // bytes may have changed since.
        let mut buffers = Vec::with_capacity(args.len());
        for arg in args {
            let bytes = unsafe { std::slice::from_raw_parts(arg.ptr, arg.len_in_bytes()) };
            let (buffer, uploaded) = self.device_buffer_entry(arg.ptr as usize, bytes.len());
            // A written argument's mirror was filled once and every launch
            // since has read it back, so the mirror holds the host bytes; an
            // untouched mirror, like a read-only argument's, needs the upload.
            if !uploaded || !arg.written {
                self.queue.write_buffer(&buffer, 0, bytes);
                self.mark_buffer_uploaded(arg.ptr as usize, bytes.len());
            }
            buffers.push(buffer);
        }

        // A kernel declares the integer scalar descriptor only when it takes
        // an integer scalar, so the host asks the shader rather than assuming.

        let shapes: Vec<i64> = args.iter().flat_map(|a| a.shape.iter().copied()).collect();
        let shapes_buffer = self.descriptor_buffer(
            (shapes.len().max(1) * std::mem::size_of::<i64>()) as u64,
            "sile shapes",
        );
        self.queue
            .write_buffer(&shapes_buffer, 0, as_bytes(&shapes));

        // The scalar descriptor binds after the shape descriptor, matching the
        // order `codegen` numbered the bindings in. A kernel with no scalar
        // argument still carries the parameter, so the buffer exists and holds
        // one unused element.
        let scalars_buffer = self.descriptor_buffer(
            (scalars.len().max(1) * std::mem::size_of::<f32>()) as u64,
            "sile scalars",
        );
        if !scalars.is_empty() {
            self.queue
                .write_buffer(&scalars_buffer, 0, as_bytes(scalars));
        }

        // The integer scalar descriptor binds after the float one, and only a
        // kernel that declares one gets a binding: a bind group entry without
        // a binding in the shader is an error, not an unused slot.
        let scalars_i64_buffer = if has_scalars_i64 {
            let buffer = self.descriptor_buffer(
                (scalars_i64.len().max(1) * std::mem::size_of::<i64>()) as u64,
                "sile scalars i64",
            );
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
        let binding_count = args.len() + descriptor_count + aliases.len();
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
            // The grid dispatches as the caller wrote it, one workgroup axis
            // per program axis: the kernel reads `workgroup_id` by axis, the
            // same terms the CPU spells its `gid` in.
            pass.dispatch_workgroups(
                launch.grid[0].max(1),
                launch.grid[1].max(1),
                launch.grid[2].max(1),
            );
        }
        // The read-back rides the same submission as the kernel: one command
        // buffer, one queue flush. A second encoder per launch would pay a
        // second submit for a copy that must wait for this kernel anyway.
        // Only the buffers the kernel writes come home — a read-only
        // argument's bytes are already on the host, which uploaded them.
        let readback: Vec<(usize, usize)> = args
            .iter()
            .enumerate()
            .filter(|(_, arg)| arg.written)
            .map(|(index, arg)| (index, arg.len_in_bytes()))
            .collect();
        for (index, byte_len) in &readback {
            let staging = self.staging_buffer(arg_ptr(args, *index), *byte_len);
            encoder.copy_buffer_to_buffer(&buffers[*index], 0, &staging, 0, *byte_len as u64);
        }
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);

        // A kernel writes its output in place, so the host reads the buffer
        // back. The read-back is a byte copy: the device holds the same
        // little-endian elements the host laid down, f32 or half alike. The
        // stream names the device the launch went to, and today a stream is
        // one device, so the read-back is the whole synchronization.
        let _ = stream;
        for (index, byte_len) in readback {
            // One staging buffer per host range, reused across launches: a
            // fresh MAP_READ allocation per launch is zero-initialized memory
            // the next launch throws away.
            let staging = self.staging_buffer(arg_ptr(args, index), byte_len);
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait);
            let mapped = slice.get_mapped_range();
            let arg = &args[index];
            unsafe {
                std::ptr::copy_nonoverlapping(mapped.as_ptr(), arg.mut_ptr, byte_len);
            }
            drop(mapped);
            staging.unmap();
        }

        Ok(())
    }
}

/// The host address of one kernel argument, for the staging lookups.
fn arg_ptr(args: &[KernelArg<'_>], index: usize) -> usize {
    args[index].ptr as usize
}
