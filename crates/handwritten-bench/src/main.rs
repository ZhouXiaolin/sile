//! Runs the handwritten WGSL references so they can be checked and timed on
//! this machine, on the same wgpu stack the WGPU backend uses.
//!
//! The references are `handwritten/*.wgsl`. Each one takes a uniform `Params`
//! (dims, scalars) and storage buffers in a fixed order, so one runner covers
//! all of them: build the inputs, dispatch, read the output back, compare
//! against a double-precision CPU oracle, then time the dispatch.
//!
//! Usage: `cargo run --release -p handwritten-bench -- saxpy|softmax|rms_norm|gemm`

use std::time::Instant;

const WGSL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../handwritten");

/// A xorshift32 stream, the same generator the C references use.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as i32) as f32 / 2147483648.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no WGPU adapter");
        eprintln!("adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("handwritten"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("no WGPU device");
        Self { device, queue }
    }
}

/// One compiled reference plus the buffers it reads.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffers: Vec<wgpu::Buffer>,
}

impl Gpu {
    /// Compile `source` and bind `params` and the `storage` buffers after it.
    fn build(&self, name: &str, source: &str, params: &[u8], storage: &[&[f32]]) -> Kernel {
        naga::front::wgsl::Frontend::new()
            .parse(source)
            .unwrap_or_else(|e| panic!("{name}.wgsl does not parse: {e}"));
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: params.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform, 0, params);

        let mut buffers = vec![uniform];
        for (index, data) in storage.iter().enumerate() {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(*data))
            };
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("storage"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, bytes);
            buffers.push(buffer);
            let _ = index;
        }

        let entries = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layout"),
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(index, _)| wgpu::BindGroupLayoutEntry {
                        binding: index as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if index == 0 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage { read_only: false }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &layout,
            entries: &entries,
        });
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pipeline layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Kernel {
            pipeline,
            bind_group,
            buffers,
        }
    }

    fn dispatch(&self, kernel: &Kernel, grid: [u32; 3]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &kernel.bind_group, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Copy buffer `index` back to the host.
    fn read(&self, kernel: &Kernel, index: usize, len: usize) -> Vec<f32> {
        let bytes = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&kernel.buffers[index], 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let mapped = slice.get_mapped_range();
        let out = mapped
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        drop(mapped);
        staging.unmap();
        out
    }
}

/// dims and scalars, the uniform every reference shares.
fn params(dims: [u32; 4], scalars: [f32; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for d in dims {
        bytes.extend_from_slice(&d.to_le_bytes());
    }
    for s in scalars {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

fn source(name: &str) -> String {
    std::fs::read_to_string(format!("{WGSL_DIR}/{name}.wgsl")).expect("read the reference")
}

fn worst(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let (w, at) = worst(got, want);
    if w > tol {
        println!(
            "FAIL {name:<24} worst={w:.3e} at {at} (got {} want {})",
            got[at], want[at]
        );
        std::process::exit(1);
    }
    println!("PASS {name:<24} worst={w:.3e} (tol {tol:.0e})");
}

fn bench(name: &str, gpu: &Gpu, kernel: &Kernel, grid: [u32; 3], reps: usize) -> f64 {
    // One warm-up dispatch, then the fastest of `reps` wall-clock runs.
    gpu.dispatch(kernel, grid);
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    let mut best = f64::MAX;
    for _ in 0..reps {
        let started = Instant::now();
        gpu.dispatch(kernel, grid);
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        best = best.min(started.elapsed().as_secs_f64() * 1e3);
    }
    println!("BENCH {name:<12} {best:10.3} ms");
    best
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "saxpy".to_string());
    let gpu = Gpu::new();

    match which.as_str() {
        "saxpy" => {
            let n = 1usize << 22;
            let mut rng = Rng(1);
            let x = rng.fill(n);
            let y = rng.fill(n);
            let out = vec![0.0f32; n];
            let a = 2.0f32;
            let kernel = gpu.build(
                "saxpy",
                &source("saxpy"),
                &params([n as u32, 0, 0, 0], [a, 0.0, 0.0, 0.0]),
                &[&x, &y, &out],
            );
            let grid = [n.div_ceil(256) as u32, 1, 1];
            gpu.dispatch(&kernel, grid);
            let got = gpu.read(&kernel, 3, n);
            let want: Vec<f32> = (0..n).map(|i| a * x[i] + y[i]).collect();
            check("saxpy 4M", &got, &want, 1e-5);
            let ms = bench("saxpy", &gpu, &kernel, grid, 7);
            println!(
                "       {:.2} GB/s",
                (n as f64 * 3.0 * 4.0) / (ms * 1e-3) / 1e9
            );
        }
        "softmax" => {
            let (rows, cols) = (4096usize, 1024usize);
            let mut rng = Rng(1);
            let x = rng.fill(rows * cols);
            let y = vec![0.0f32; rows * cols];
            let out = vec![0.0f32; rows * cols];
            let kernel = gpu.build(
                "softmax",
                &source("softmax"),
                &params([rows as u32, cols as u32, 0, 0], [0.0; 4]),
                &[&x, &y, &out],
            );
            let grid = [rows as u32, 1, 1];
            gpu.dispatch(&kernel, grid);
            let got = gpu.read(&kernel, 3, rows * cols);
            // The oracle: the same formula in f64.
            let mut want = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                let max = row.iter().copied().fold(f64::MIN, |a, v| a.max(v as f64));
                let exps: Vec<f64> = row.iter().map(|v| ((*v as f64) - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for c in 0..cols {
                    want[r * cols + c] = (exps[c] / sum) as f32;
                }
            }
            check("softmax 4096x1024", &got, &want, 1e-5);
            let ms = bench("softmax", &gpu, &kernel, grid, 7);
            println!(
                "       {:.2} GB/s",
                (rows * cols * 4 * 4) as f64 / (ms * 1e-3) / 1e9
            );
        }
        "rms_norm" => {
            let (rows, cols) = (4096usize, 1024usize);
            let eps = 1e-8f32;
            let mut rng = Rng(1);
            let x = rng.fill(rows * cols);
            let w = rng.fill(cols);
            let out = vec![0.0f32; rows * cols];
            let kernel = gpu.build(
                "rms_norm",
                &source("rms_norm"),
                &params([rows as u32, cols as u32, 0, 0], [0.0, eps, 0.0, 0.0]),
                &[&x, &w, &out],
            );
            let grid = [rows as u32, 1, 1];
            gpu.dispatch(&kernel, grid);
            let got = gpu.read(&kernel, 3, rows * cols);
            let mut want = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                let sum: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let inv = 1.0 / (sum / cols as f64 + eps as f64).sqrt();
                for c in 0..cols {
                    want[r * cols + c] = (row[c] as f64 * inv * w[c] as f64) as f32;
                }
            }
            check("rms_norm 4096x1024", &got, &want, 1e-5);
            let ms = bench("rms_norm", &gpu, &kernel, grid, 7);
            println!(
                "       {:.2} GB/s",
                (rows * cols * 3 * 4) as f64 / (ms * 1e-3) / 1e9
            );
        }
        "gemm" => {
            let (m, n, k) = (1024usize, 1024usize, 1024usize);
            let mut rng = Rng(1);
            let x = rng.fill(m * k);
            let y = rng.fill(k * n);
            let out = vec![0.0f32; m * n];
            let kernel = gpu.build(
                "gemm",
                &source("gemm"),
                &params([m as u32, n as u32, k as u32, 0], [0.0; 4]),
                &[&x, &y, &out],
            );
            // 32x16 tiles: columns along x, rows along y.
            let grid = [n.div_ceil(16) as u32, m.div_ceil(32) as u32, 1];
            gpu.dispatch(&kernel, grid);
            let got = gpu.read(&kernel, 3, m * n);
            // The oracle on a corner of the output: the full product in f64 at
            // 1024^3 is a minute of work for a check.
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += x[i * k + p] as f64 * y[p * n + j] as f64;
                    }
                    want[i * n + j] = sum as f32;
                }
            }
            check("gemm 1024^3", &got, &want, 1e-2);
            let ms = bench("gemm", &gpu, &kernel, grid, 5);
            println!(
                "       {:.2} GFLOP/s",
                (2.0 * (m * n * k) as f64) / (ms * 1e-3) / 1e9
            );
        }
        other => {
            eprintln!("unknown reference: {other}");
            std::process::exit(2);
        }
    }
}
