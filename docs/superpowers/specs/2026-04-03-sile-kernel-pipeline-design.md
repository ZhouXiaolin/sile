# Sile Kernel Pipeline — Full Link-Through Design

## Goal

打通 `#[kernel]` 宏到 C 代码生成的完整管线，使 `vec_add` 和 `softmax` 两个 example 能编译运行并输出正确结果。

## Current State

架构正确但实现不完整。6 个断裂点：
1. `parse` 只识别 `let` 和 `store`，丢弃用户表达式
2. `lower` 硬编码 ProgramId/tmp，丢失 Rank 信息
3. `partition` API 签名与使用不匹配
4. `Tile` 类型是空壳
5. `SSA lower` 硬编码固定指令序列
6. `codegen/c` 只有两个硬编码 C 模板

## Approach

方案 A：渐进式修复。在现有架构上逐个修复断裂点，保持 `parse → lower → HIR → typeck → SSA → passes → backend_ir → codegen` 管线不变。

---

## Phase 1: 宏解析增强 — 提取真实表达式

### 1.1 AST 扩展 (`frontend/ast.rs`)

新增 `KernelExpr` 枚举覆盖所有 kernel 内表达式：

```rust
pub enum KernelExpr {
    Var(syn::Ident),
    Const(syn::Lit),
    MethodCall {
        receiver: Box<KernelExpr>,
        method: syn::Ident,
        args: Vec<KernelExpr>,
    },
    BinaryOp {
        left: Box<KernelExpr>,
        op: syn::BinOp,
        right: Box<KernelExpr>,
    },
    Array(syn::ExprArray),      // 用于 shape 参数如 [BM, BN]
    ConstBlock(syn::Expr),      // 用于 const_shape![...]
}
```

`KernelStmt::Let` 和 `KernelStmt::Store` 的 `expr`/`value` 字段改为 `KernelExpr`。

### 1.2 递归解析 (`frontend/parse.rs`)

新增 `parse_expr(syn::Expr) -> KernelExpr`：
- `syn::Expr::MethodCall` → `KernelExpr::MethodCall`，递归解析 receiver 和 args
- `syn::Expr::Binary` → `KernelExpr::BinaryOp`，递归解析 left/right
- `syn::Expr::Path` → `KernelExpr::Var`
- `syn::Expr::Lit` → `KernelExpr::Const`
- `syn::Expr::Array` → `KernelExpr::Array`

### 1.3 验证

- `parse_kernel` 能正确解析 vec_add 和 softmax 的完整函数体
- 测试：新增 `parse_vec_add_expressions` 和 `parse_softmax_expressions`

---

## Phase 2: Lower 保真 — 传递 Rank 和表达式

### 2.1 表达式翻译 (`frontend/lower.rs`)

递归遍历 `KernelExpr`，映射到 `hir::Expr`：

| KernelExpr | hir::Expr |
|------------|-----------|
| `MethodCall { method: "load_tile", ... }` | `Builtin { op: LoadTile, args: [...] }` |
| `MethodCall { method: "reduce_max", ... }` | `Builtin { op: ReduceMax, args: [...] }` |
| `MethodCall { method: "exp", ... }` | `Builtin { op: Exp, args: [...] }` |
| `MethodCall { method: "reshape", ... }` | `Builtin { op: Reshape, args: [...] }` |
| `MethodCall { method: "broadcast", ... }` | `Builtin { op: Broadcast, args: [...] }` |
| `MethodCall { method: "store", ... }` | (生成 `Stmt::Store`) |
| `BinaryOp { op: Add, ... }` | `Builtin { op: Add, args: [left, right] }` |
| `BinaryOp { op: Sub, ... }` | `Builtin { op: Sub, args: [left, right] }` |
| `Var(name)` | `Expr::Var(name)` |

### 2.2 Rank 信息传递

从函数签名提取 const 泛型参数和类型注解：
- `const BM: i32` → `ShapeExpr::Symbol("BM")`
- `{ [-1, -1] }` → `ShapeExpr::Tuple([Dynamic, Dynamic])`
- `{ [BM, BN] }` → `ShapeExpr::Tuple([Symbol("BM"), Symbol("BN")])`

参数类型从 `ShapeExpr::dynamic()` 改为从类型注解中提取的真实 shape。

### 2.3 Let 语句保真

不再硬编码 `ProgramId`，而是翻译 `KernelStmt::Let` 中的真实 `expr`：
```rust
KernelStmt::Let { name, expr } → hir::Stmt::Let {
    name: name.to_string(),
    ty: None,  // 由 typeck 推断
    expr: translate_expr(expr),
}
```

### 2.4 Store 语句保真

不再硬编码 `Var("tmp")`，而是翻译真实 value：
```rust
KernelStmt::Store { target, value } → hir::Stmt::Store {
    target: target.to_string(),
    value: translate_expr(value),
}
```

### 2.5 验证

- `kernel_frontend_vec_add` 测试验证 HIR body 包含 `ProgramId`, `LoadTile`, `Add`, `Store`
- `kernel_frontend_softmax` 测试验证 HIR body 包含 `LoadTileLike2D`, `ReduceMax`, `Reshape`, `Broadcast`, `Sub`, `Exp`, `ReduceSum`, `Div`, `Store`

---

## Phase 3: Partition API 修复

### 3.1 签名修正 (`tensor.rs`)

```rust
// 之前
pub fn partition(&self, _axis: usize, _count: usize) -> Partition<Self>

// 之后
pub fn partition(&self, tile_shape: impl Into<Vec<i64>>) -> Partition<Self>
```

### 3.2 Partition 增强

```rust
pub struct Partition<T> {
    pub parts: Vec<T>,
    pub tile_shape: Vec<i64>,     // 新增：记录分块维度
    pub grid_shape: Vec<i64>,     // 新增：记录网格维度
}
```

### 3.3 Unpartition 修正

改为 `Partition` 的方法，支持多维 concat：
```rust
impl<T: Clone> Partition<Tensor<T, DListNil>> {
    pub fn unpartition(self) -> Tensor<T, DListNil> {
        // 按 tile_shape 维度正确合并数据
    }
}
```

### 3.4 Kernel 参数适配

`Partition` 增加 `as_kernel_arg` 方法，提取第一个 part 的指针和 shape 传给 kernel。

### 3.5 验证

- `examples/vec_add.rs` 和 `examples/softmax.rs` 编译通过
- `partition([4])` 和 `partition([bm, bn])` 都能正确工作

---

## Phase 4: Tile 类型充实

### 4.1 数据结构 (`tile.rs`)

```rust
pub struct Tile<T, R: Rank = DListNil> {
    pub shape: Vec<i64>,
    pub _elem: std::marker::PhantomData<T>,
    pub _rank: std::marker::PhantomData<R>,
}
```

Tile 在运行时不持有数据（数据在 kernel 执行时由 GPU/CPU 后端管理），但需要 shape 信息用于类型检查。

### 4.2 算术运算

实现 `Add`, `Sub`, `Mul`, `Div` trait，返回新 Tile，shape 保持与操作数一致（broadcast 除外）。

### 4.3 Kernel 内方法

```rust
impl<T, R: Rank> Tile<T, R> {
    pub fn reduce_max(&self, axis: i32) -> Tile<T, /* reduced rank */> { ... }
    pub fn reduce_sum(&self, axis: i32) -> Tile<T, /* reduced rank */> { ... }
    pub fn reshape<const NewShape>(&self, shape: /* */) -> Tile<T, NewRank> { ... }
    pub fn broadcast(&self, target_shape: &[i64]) -> Tile<T, NewRank> { ... }
    pub fn exp(&self) -> Tile<T, R> { ... }
}
```

### 4.4 load_tile 自由函数

```rust
pub fn load_tile<T, R: Rank>(tensor: &Tensor<T, R>, tile_shape: /* */, indices: /* */) -> Tile<T, R> {
    // 返回 Tile 实例，shape 由 tile_shape 决定
}
```

### 4.5 验证

- `examples/softmax.rs` 中所有 Tile 类型注解编译通过
- `tile::id()` 返回正确的 TileId

---

## Phase 5: SSA Lower 通用化

### 5.1 SSA IR 增强 (`ssa/ir.rs`)

```rust
pub struct SsaInstruction {
    pub def: SsaValue,           // 定义的值
    pub opcode: SsaOpcode,
    pub uses: Vec<SsaValue>,     // 使用的值
    pub immediates: Vec<i64>,    // 立即数（如 reduce axis）
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaValue {
    Param(usize),    // 参数索引
    Local(usize),    // 局部变量索引
    Const(i64),      // 常量
}
```

### 5.2 通用 Lower (`ssa/lower.rs`)

遍历 `TypedKernel` 的 body，维护 `name → SsaValue` 映射：

```rust
pub fn lower_typed_kernel_to_ssa(typed: &TypedKernel) -> SsaProgram {
    let mut locals = HashMap::new();  // name → SsaValue
    let mut instructions = Vec::new();
    let mut next_local = 0;

    for stmt in &typed.kernel.body {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let value = lower_expr(expr, &mut instructions, &locals);
                locals.insert(name.clone(), value);
            }
            Stmt::Store { target, value } => {
                let val = lower_expr(value, &mut instructions, &locals);
                instructions.push(SsaInstruction {
                    def: SsaValue::Local(next_local),
                    opcode: SsaOpcode::Store,
                    uses: vec![val],
                    immediates: vec![],
                });
                next_local += 1;
            }
        }
    }

    SsaProgram { instructions }
}
```

`lower_expr` 递归翻译 `hir::Expr`：
- `Expr::Var(name)` → 查表得到 `SsaValue`
- `Expr::Builtin { op, args }` → 递归 lower args，生成对应 `SsaOpcode`
- `Expr::ScalarI32(v)` → `SsaValue::Const(v as i64)`
- `Expr::Shape(shape)` → 编码为 immediates

### 5.3 验证

- `ssa_vec_add` 测试验证指令数、opcode 顺序
- `ssa_softmax` 测试验证 reduce/reshape/broadcast 序列

---

## Phase 6: Codegen 通用化

### 6.1 Backend IR 增强 (`backend_ir/ir.rs`)

```rust
pub struct BackendKernel {
    pub op: BackendOp,
    pub tile_rank: usize,
    pub tile_shape_symbols: Vec<String>,
    pub instructions: Vec<BackendInstruction>,  // 新增
}

pub enum BackendInstruction {
    Load { dest: String, src: String, indices: Vec<String> },
    Compute { dest: String, op: String, args: Vec<String> },
    Reduce { dest: String, src: String, axis: i64, kind: ReduceKind },
    Store { src: String, dest: String },
}
```

### 6.2 SSA → Backend IR (`backend_ir/lower.rs`)

基于 SSA opcode 序列生成真正的 backend IR：
- `ProgramId` → 循环变量
- `LoadTile` → `BackendInstruction::Load`
- `Add/Sub/Mul/Div/Exp` → `BackendInstruction::Compute`
- `ReduceMax/ReduceSum` → `BackendInstruction::Reduce`
- `Reshape/Broadcast` → 元数据，不生成代码
- `Store` → `BackendInstruction::Store`

### 6.3 通用 C 代码生成 (`codegen/c.rs`)

遍历 Backend IR 指令，生成 C 代码：

```rust
pub fn generate(kernel: &BackendKernel) -> Result<String> {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <math.h>\n\n");

    // 函数签名
    out.push_str(&generate_signature(kernel));

    // 循环嵌套（基于 tile_rank）
    for dim in 0..kernel.tile_rank {
        out.push_str(&format!("  for (int64_t i{dim} = 0; i{dim} < {sym}; ++i{dim}) {{\n",
            dim = dim, sym = kernel.tile_shape_symbols[dim]));
    }

    // 指令翻译
    for inst in &kernel.instructions {
        out.push_str(&generate_instruction(inst, kernel));
    }

    // 关闭循环
    for _ in 0..kernel.tile_rank {
        out.push_str("  }\n");
    }

    out.push_str("}\n");
    Ok(out)
}
```

### 6.4 验证

- `c_codegen` 测试验证生成的 C 代码包含正确函数名和逻辑
- `backend_vec_add` / `backend_softmax` 测试验证编译和运行
- 最终 `cargo run --example vec_add` 输出 `[3.0, 4.0, 5.0, ...]`
- 最终 `cargo run --example softmax` 输出每行 sum ≈ 1.0

---

## Test Matrix

| 测试 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 |
|------|---------|---------|---------|---------|---------|---------|
| `kernel_macro_smoke` | ✅ | ✅ | - | - | - | - |
| `kernel_frontend_vec_add` | ✅ | ✅ | - | - | - | - |
| `kernel_frontend_softmax` | ✅ | ✅ | - | - | - | - |
| `typeck_vec_add` | - | ✅ | - | ✅ | - | - |
| `typeck_softmax` | - | ✅ | - | ✅ | - | - |
| `ssa_vec_add` | - | - | - | - | ✅ | - |
| `ssa_softmax` | - | - | - | - | ✅ | - |
| `c_codegen` | - | - | - | - | - | ✅ |
| `backend_vec_add` | - | - | - | - | - | ✅ |
| `backend_softmax` | - | - | - | - | - | ✅ |
| `example vec_add` | - | - | ✅ | ✅ | - | ✅ |
| `example softmax` | - | - | ✅ | ✅ | - | ✅ |

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| 方法链解析复杂度高 | 先支持单层方法调用，再扩展到链式 |
| Rank 类型推导困难 | 先在 HIR 层用 ShapeExpr 符号表示，不做强类型检查 |
| Tile 运行时数据管理 | Tile 在 kernel 函数内只记录操作，实际数据由后端管理 |
| SSA φ 节点处理 | 当前 kernel 无控制流，暂不需要 φ 节点 |
| C 代码生成复杂度 | 先支持 1D/2D 简单循环，后续再优化 |
