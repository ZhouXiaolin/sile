# METAL 后端实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SILE 添加 METAL 后端，支持 vec_add、matmul、softmax 三个 example 在 Apple GPU 上运行

**Architecture:** 
- 创建 `sile-backend-metal` crate，实现 `Backend` trait
- 生成 Metal shader 代码，运行时编译并执行
- 修改 kernel_launcher 根据 Device 类型选择 backend

**Tech Stack:** metal crate (metal-rs), MTLDevice, MTLLibrary

---

## Task 1: 创建 sile-backend-metal crate

**Files:**
- Create: `crates/sile-backend-metal/Cargo.toml`
- Create: `crates/sile-backend-metal/src/lib.rs`
- Modify: `Cargo.toml:2` (添加 workspace member)

- [ ] **Step 1: 创建 Cargo.toml**

```toml
[package]
name = "sile-backend-metal"
version.workspace = true
edition.workspace = true

[dependencies]
sile-core = { path = "../sile-core" }
sile-hir = { path = "../sile-hir" }
sile-lir = { path = "../sile-lir" }
metal = "0.29"
```

- [ ] **Step 2: 创建 lib.rs 空壳**

```rust
pub mod codegen_metal;

use std::ffi::c_void;

use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_lir::ir::Function;

use crate::codegen_metal::generate;

pub struct MetalBackend;

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }
}

impl sile_lir::Backend for MetalBackend {
    fn execute(
        &self,
        func: &Function,
        kernel: &Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        todo!()
    }
}
```

- [ ] **Step 3: 创建 codegen_metal.rs 空壳**

```rust
use sile_lir::ir::*;

pub fn generate(func: &Function, name: &str) -> sile_core::Result<String> {
    todo!()
}
```

- [ ] **Step 4: 添加 workspace member**

Modify: `Cargo.toml:2` - 在 members 数组中添加 `"crates/sile-backend-metal"`

- [ ] **Step 5: 验证编译**

Run: `cargo check -p sile-backend-metal`
Expected: PASS

---

## Task 2: 实现 Metal 代码生成

**Files:**
- Modify: `crates/sile-backend-metal/src/codegen_metal.rs`

- [ ] **Step 1: 实现基础框架**

在 `codegen_metal.rs` 中实现:
- `MetalCodegen` 结构体
- `generate()` 函数
- `emit_prologue()` - Metal shader 头部
- `emit_kernel_signature()` - kernel 参数
- `emit_instruction()` - 指令生成

需要支持的 LIR 指令与 CPU 后端相同:
- `Add`, `Sub`, `Mul`, `Div`, `Exp`, `FNeg`, `FMax`, `FMin`
- `TileAlloc`, `TileLoad2D`, `TileMma`, `TileReduceMax`, `TileReduceSum`
- `TileBroadcast`, `TileStore2D`, `GetTileCoord`

Metal 特有:
- `threadgroup` 内存
- `simdgroup` 操作
- `[[threadgroup]]`, `[[thread_position_in_grid]]` 等属性

- [ ] **Step 2: 验证 codegen 输出**

Run: `cargo build -p sile-backend-metal`
Expected: PASS

---

## Task 3: 实现 Metal 执行器

**Files:**
- Modify: `crates/sile-backend-metal/src/lib.rs`

- [ ] **Step 1: 实现 Metal runtime**

在 `lib.rs` 中实现:
- 获取 `MTLDevice`
- 编译 shader (使用 `device.new_library_with_source`)
- 创建 `MTLComputePipelineState`
- 编码 command buffer
- 执行 kernel

```rust
impl MetalBackend {
    fn get_device() -> sile_core::Result<metal::Device> {
        metal::Device::all()
            .next()
            .ok_or_else(|| sile_core::Error::UnsupportedBackend("no Metal device"))
    }

    fn compile_shader(&self, source: &str) -> sile_core::Result<metal::Library> {
        let device = Self::get_device()?;
        device.new_library_with_source(source, &{})
            .map_err(|e| sile_core::Error::Compile(e.to_string()))
    }

    fn execute_kernel(&self, ... /* params */) -> sile_core::Result<()> {
        // 1. 获取 device
        // 2. 编译 shader
        // 3. 创建 pipeline
        // 4. 创建 command buffer
        // 5. 设置 buffer 参数
        // 6. dispatch
        // 7. wait
    }
}
```

- [ ] **Step 2: 运行 vec_add example**

测试是否能在 Metal 上运行

---

## Task 4: 集成到 KernelLauncher

**Files:**
- Modify: `crates/sile/src/kernel_launcher.rs`

- [ ] **Step 1: 根据 Device 选择 Backend**

```rust
pub fn apply(self, stream: &Stream) -> Result<()> {
    let launch = LaunchConfig { ... };
    let typed = ...;
    let lir_func = sile_compiler::compile(&typed);

    match stream.device() {
        Device::Cpu(_) => {
            let backend = sile_backend_cpu::CpuBackend::new();
            backend.execute(...)
        }
        Device::Metal(_) => {
            let backend = sile_backend_cpu::MetalBackend::new();
            backend.execute(...)
        }
        _ => Err(sile_core::Error::UnsupportedBackend("...")),
    }
}
```

- [ ] **Step 2: 验证编译**

Run: `cargo check -p sile`
Expected: PASS

---

## Task 5: 支持 matmul

**Files:**
- Modify: `crates/sile-backend-metal/src/codegen_metal.rs`

- [ ] **Step 1: 实现 TileMma**

在 Metal 中实现矩阵乘法的 tile 级操作:
```metal
// 简单实现: 嵌套循环
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

- [ ] **Step 2: 测试 matmul**

Run example 并验证结果

---

## Task 6: 支持 softmax

**Files:**
- Modify: `crates/sile-backend-metal/src/codegen_metal.rs`

- [ ] **Step 1: 实现 reduce 和 exp**

- `TileReduceMax`: 使用 simdgroup reduce 或简单循环
- `TileReduceSum`: 同上
- `Exp`: 使用 Metal 内置 `metal::exp`

- [ ] **Step 2: 测试 softmax**

Run example 并验证结果

---

## Task 7: 最终验证

- [ ] **Step 1: 运行 vec_add**

```bash
SILE_DEVICE=METAL cargo run --example vec_add
```

- [ ] **Step 2: 运行 matmul**

```bash
SILE_DEVICE=METAL cargo run --example matmul
```

- [ ] **Step 3: 运行 softmax**

```bash
SILE_DEVICE=METAL cargo run --example softmax
```
