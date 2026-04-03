# METAL 后端设计

## 概述

为 SILE 添加 METAL 后端（Apple GPU），用于 Apple 设备的 GPU 加速。

## 目标

支持三个 example 能在 METAL 后端上编译运行：
- `vec_add` - 向量加法
- `matmul` - 矩阵乘法
- `softmax` - Softmax

## 架构

```
┌─────────────────────────────────────────────┐
│                   SILE                      │
│  (vec_add, matmul, softmax 等 kernel)       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              sile-lir                       │
│              (Function IR)                 │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────┴────────┐
         │   Backend trait │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐
│ CPU   │   │ METAL   │   │  ...    │
│Backend│   │ Backend │   │         │
└───────┘   └─────────┘   └─────────┘
```

## 实现方案

### 1. 项目结构

创建 `crates/sile-backend-metal/`:

```
crates/sile-backend-metal/
├── Cargo.toml
└── src/
    ├── lib.rs          # Backend impl + Metal runtime
    └── codegen_metal.rs # LIR → Metal shader
```

### 2. 代码生成 (codegen_metal.rs)

将 LIR 指令转换为 Metal shader 代码。

**支持的指令：**

| LIR 指令 | Metal 实现 |
|----------|------------|
| `Add`, `Sub`, `Mul`, `Div` | element-wise ops |
| `Exp` | `metal::exp` |
| `FNeg` | unary minus |
| `TileAlloc` | stack array `float[N][M]` |
| `TileLoad2D` | threadgroup memory |
| `TileMma` | nested loop multiply-accumulate |
| `TileReduceMax` | simdgroup reduce |
| `TileReduceSum` | simdgroup reduce |
| `TileBroadcast` | copy to all elements |
| `TileStore2D` | write to device memory |

**Metal kernel 结构：**

```metal
kernel void sile_kernel_${name}(
    device float* buf0 [[buffer(0)]],
    device float* buf1 [[buffer(1)]],
    // ...
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // tile operations
}
```

### 3. 编译策略

- **运行时编译**: 使用 `metal-rs` 调用 `MTLDevice` 创建 `MTLLibrary`
- 或者使用 `xcrun metal` 预编译 shader（与 CPU 后端类似）

### 4. 执行流程

1. 生成 Metal shader 源码
2. 编译成 `MTLLibrary`
3. 从 library 获取 `MTLFunction`
4. 创建 `MTLComputePipelineState`
5. 编码 command buffer，设置 buffer 参数
6. dispatch compute pipeline
7. waitUntilCompleted

### 5. Buffer 管理

- Input: `MTLBuffer` (device)
- Output: `MTLBuffer` (device)
- 参数通过 `setBuffer` 传入 kernel

### 6. 错误处理

- Metal device unavailable: `Error::UnsupportedBackend`
- Shader 编译失败: `Error::Compile`
- 执行失败: `Error::Execute`

## 测试计划

1. 先让 `vec_add` 能跑通（最简单的 tile load/store）
2. 再支持 `matmul`（需要 MMA、constant）
3. 最后支持 `softmax`（需要 reduce、broadcast、exp）

## 里程碑

- [ ] 创建 `sile-backend-metal` crate
- [ ] 实现基本的 Metal codegen
- [ ] 实现 Buffer 管理和执行
- [ ] 运行 vec_add example
- [ ] 支持 matmul 所需指令
- [ ] 运行 matmul example
- [ ] 支持 softmax 所需指令
- [ ] 运行 softmax example
