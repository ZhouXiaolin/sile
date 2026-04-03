# Executable Kernel LIR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the compiler produce a single printable `ExecutableKernel` object that contains LIR `Function`, backend ABI metadata, and value/type-shape metadata so CPU and Metal backends no longer consume HIR `Kernel` directly.

**Architecture:** Keep `sile_hir::Kernel` as the frontend/typecheck input, but move execution-facing metadata into a new `sile_lir::ExecutableKernel` package. The compiler will return `ExecutableKernel`, the runtime launcher will print or inspect it before execution, and both backends will consume it as their only IR input. Add a printer that emits an LLVM-IR-like text view of the executable package, and replace duplicated backend-local `KernelGenInfo` construction with compiler-owned `KernelAbi` and `ValueInfoTable`.

**Tech Stack:** Rust workspace, `sile-hir`, `sile-lir`, `sile-compiler`, CPU C codegen backend, Metal backend, `cargo test`, `cargo run`.

---

## File Structure

**Create:**
- `crates/sile-lir/src/executable.rs` — execution-boundary structs: `ExecutableKernel`, `KernelAbi`, `KernelParamAbi`, `ShapeLayout`, `LaunchSemantics`, `ValueInfo`, `ValueInfoTable`
- `crates/sile-lir/src/print.rs` — LLVM-IR-like printer for `ExecutableKernel`
- `crates/sile-lir/tests/executable_kernel.rs` — unit tests for ABI and value metadata data structures
- `crates/sile-lir/tests/print.rs` — unit tests for printer output formatting
- `crates/sile-compiler/tests/executable_kernel.rs` — compiler tests that verify `compile()` returns complete executable metadata

**Modify:**
- `crates/sile-lir/src/lib.rs` — export the new executable and print modules
- `crates/sile-lir/src/backend.rs` — change backend trait to accept `ExecutableKernel`
- `crates/sile-compiler/src/lib.rs` — return `ExecutableKernel` instead of bare `Function`
- `crates/sile-compiler/src/lower_lir.rs` — produce `Function` plus `ValueInfoTable`
- `crates/sile/src/kernel_launcher.rs` — consume `ExecutableKernel`, print/debug it, and pass it into backends
- `crates/sile-backend-cpu/src/lib.rs` — remove `Kernel` dependency from execute path, read `KernelAbi`
- `crates/sile-backend-cpu/src/codegen_c.rs` — consume `KernelAbi` and `ValueInfoTable` instead of local shape analysis
- `crates/sile-backend-metal/src/lib.rs` — same as CPU backend
- `crates/sile-backend-metal/src/codegen_metal.rs` — same as CPU backend

**Test and Verify:**
- `crates/sile/tests/backend_vec_add.rs`
- `crates/sile/tests/backend_softmax.rs`
- `crates/sile/tests/backend_matmul.rs`
- `crates/sile/tests/backend_metal.rs`
- `crates/sile/examples/vec_add.rs`
- `crates/sile/examples/softmax.rs`
- `crates/sile/examples/matmul.rs`

## Task 1: Add Executable LIR Types And Exports

**Files:**
- Create: `crates/sile-lir/src/executable.rs`
- Modify: `crates/sile-lir/src/lib.rs`
- Test: `crates/sile-lir/tests/executable_kernel.rs`

- [ ] **Step 1: Write the failing `sile-lir` metadata tests**

Create `crates/sile-lir/tests/executable_kernel.rs` with these tests:

```rust
use sile_hir::ParamKind;
use sile_lir::{
    ElemType, ExecutableKernel, FloatType, Function, KernelAbi, KernelParamAbi, LaunchSemantics,
    Param, ParamPassing, ShapeLayout, Type, ValueInfo, ValueInfoTable,
};

#[test]
fn executable_kernel_keeps_abi_and_value_info_together() {
    let kernel = ExecutableKernel {
        name: "vec_add".into(),
        abi: KernelAbi {
            params: vec![
                KernelParamAbi {
                    index: 0,
                    name: "a".into(),
                    kind: ParamKind::Input,
                    elem: ElemType::F32,
                    rank: 1,
                    passing: ParamPassing::Buffer,
                },
                KernelParamAbi {
                    index: 1,
                    name: "c".into(),
                    kind: ParamKind::Output,
                    elem: ElemType::F32,
                    rank: 1,
                    passing: ParamPassing::Buffer,
                },
            ],
            shape_layout: ShapeLayout {
                total_dims: 2,
                offsets: vec![0, 1],
            },
            launch: LaunchSemantics { program_id_dims: 1 },
        },
        func: Function::new(
            "vec_add",
            vec![Param {
                name: "a".into(),
                ty: Type::ptr(Type::Float(FloatType::F32)),
            }],
            Type::Void,
        ),
        value_info: ValueInfoTable {
            params: vec![
                ValueInfo::Buffer {
                    elem: ElemType::F32,
                    rank: 1,
                },
                ValueInfo::Buffer {
                    elem: ElemType::F32,
                    rank: 1,
                },
            ],
            instructions: vec![ValueInfo::Tile {
                elem: ElemType::F32,
                rows: 1,
                cols: 4,
            }],
        },
    };

    assert_eq!(kernel.abi.shape_layout.total_dims, 2);
    assert_eq!(kernel.abi.launch.program_id_dims, 1);
    assert!(matches!(
        kernel.value_info.instructions[0],
        ValueInfo::Tile {
            elem: ElemType::F32,
            rows: 1,
            cols: 4
        }
    ));
}

#[test]
fn shape_layout_offsets_match_param_order() {
    let layout = ShapeLayout {
        total_dims: 5,
        offsets: vec![0, 2, 4],
    };

    assert_eq!(layout.offsets, vec![0, 2, 4]);
    assert_eq!(layout.total_dims, 5);
}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-lir --test executable_kernel -- --nocapture
```

Expected: FAIL with unresolved imports such as `ExecutableKernel`, `KernelAbi`, `ParamPassing`, or missing `ElemType` re-export from `sile-lir`.

- [ ] **Step 3: Add the new execution-boundary types**

Create `crates/sile-lir/src/executable.rs`:

```rust
use sile_hir::{types::ElemType, ParamKind};

use crate::ir::Function;

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableKernel {
    pub name: String,
    pub abi: KernelAbi,
    pub func: Function,
    pub value_info: ValueInfoTable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    pub params: Vec<KernelParamAbi>,
    pub shape_layout: ShapeLayout,
    pub launch: LaunchSemantics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelParamAbi {
    pub index: usize,
    pub name: String,
    pub kind: ParamKind,
    pub elem: ElemType,
    pub rank: usize,
    pub passing: ParamPassing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamPassing {
    Buffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeLayout {
    pub total_dims: usize,
    pub offsets: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchSemantics {
    pub program_id_dims: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueInfoTable {
    pub params: Vec<ValueInfo>,
    pub instructions: Vec<ValueInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueInfo {
    Buffer { elem: ElemType, rank: usize },
    Scalar { elem: ElemType },
    Index,
    Shape,
    Tile { elem: ElemType, rows: i64, cols: i64 },
    Void,
}
```

- [ ] **Step 4: Export the new module from `sile-lir`**

Update `crates/sile-lir/src/lib.rs` to:

```rust
pub mod backend;
pub mod builder;
pub mod executable;
pub mod ir;
pub mod print;

pub use backend::Backend;
pub use executable::{
    ExecutableKernel, KernelAbi, KernelParamAbi, LaunchSemantics, ParamPassing, ShapeLayout,
    ValueInfo, ValueInfoTable,
};
pub use ir::{
    BasicBlock, CmpOp, Constant, FloatType, Function, GlobalVariable, Instruction, IntegerType,
    Param, PhiNode, Program, Terminator, Type, Value,
};
pub use sile_hir::types::ElemType;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-lir --test executable_kernel -- --nocapture
```

Expected: PASS with `2 passed`.

- [ ] **Step 6: Commit the metadata layer**

Run:

```bash
rtk git add crates/sile-lir/src/executable.rs crates/sile-lir/src/lib.rs crates/sile-lir/tests/executable_kernel.rs
rtk git commit -m "feat: add executable kernel metadata types"
```

Expected: commit succeeds and `git status --short` no longer lists those files.

## Task 2: Make The Compiler Return `ExecutableKernel`

**Files:**
- Modify: `crates/sile-compiler/src/lib.rs`
- Modify: `crates/sile-compiler/src/lower_lir.rs`
- Create: `crates/sile-compiler/tests/executable_kernel.rs`
- Test: `crates/sile-compiler/tests/executable_kernel.rs`

- [ ] **Step 1: Write the failing compiler test for executable output**

Create `crates/sile-compiler/tests/executable_kernel.rs`:

```rust
use sile_compiler::compile;
use sile_hir::{typeck::check_kernel, Kernel, Param, ParamKind, Stmt, Type};
use sile_hir::types::{ElemType, ShapeExpr};
use sile_lir::ValueInfo;

fn vec_add_kernel() -> Kernel {
    Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
        ],
        vec![Stmt::Store {
            target: "c".into(),
            value: sile_hir::Expr::Var("a".into()),
        }],
    )
}

#[test]
fn compile_returns_executable_kernel_with_abi() {
    let typed = check_kernel(&vec_add_kernel()).expect("typed kernel");
    let executable = compile(&typed);

    assert_eq!(executable.name, "vec_add");
    assert_eq!(executable.abi.params.len(), 3);
    assert_eq!(executable.abi.shape_layout.offsets, vec![0, 1, 2]);
    assert_eq!(executable.abi.launch.program_id_dims, 1);
    assert!(matches!(
        executable.value_info.params[2],
        ValueInfo::Buffer { rank: 1, .. }
    ));
}
```

- [ ] **Step 2: Run the compiler test to verify it fails**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-compiler --test executable_kernel -- --nocapture
```

Expected: FAIL because `compile()` returns `Function` and therefore has no `abi` or `value_info`.

- [ ] **Step 3: Change `compile()` to return `ExecutableKernel`**

Update `crates/sile-compiler/src/lib.rs` to:

```rust
pub mod lower_hir;
pub mod lower_lir;
pub mod mir;
pub mod passes;

pub use lower_hir::lower_typed_kernel_to_ssa;
pub use mir::ir::{SsaOpcode, SsaProgram, SsaValue};

use sile_hir::typeck::TypedKernel;
use sile_lir::ExecutableKernel;

pub fn compile(typed: &TypedKernel) -> ExecutableKernel {
    let ssa = lower_hir::lower_typed_kernel_to_ssa(typed);
    let ssa = passes::canonicalize::run(ssa);
    let ssa = passes::dce::run(ssa);
    lower_lir::lower_ssa_to_lir(&ssa, typed)
}
```

- [ ] **Step 4: Return `Function + ABI + ValueInfoTable` from lowering**

Replace the top-level shape of `crates/sile-compiler/src/lower_lir.rs` with this structure:

```rust
use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_hir::types::ElemType;
use sile_lir::builder::LirBuilder;
use sile_lir::{
    Constant, ExecutableKernel, Function, Instruction, KernelAbi, KernelParamAbi,
    LaunchSemantics, Param, ParamPassing, ShapeLayout, Type, Value, ValueInfo, ValueInfoTable,
};

use crate::mir::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};

pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> ExecutableKernel {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);
    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let mut opcode_map: HashMap<usize, SsaOpcode> = HashMap::new();
    let mut value_info = ValueInfoTable {
        params: typed
            .kernel
            .params
            .iter()
            .map(|param| ValueInfo::Buffer {
                elem: ElemType::F32,
                rank: rank_of_param(param),
            })
            .collect(),
        instructions: Vec::new(),
    };
    let mut max_program_id_dim = 0usize;

    for (i, _) in typed.kernel.params.iter().enumerate() {
        value_map.insert(i, Value::Param(i));
    }

    let body = builder.append_block("body");
    builder.switch_to_block(&body);

    for inst in &ssa.instructions {
        lower_ssa_instruction(
            inst,
            &mut builder,
            &mut value_map,
            &mut opcode_map,
            &mut value_info,
            &mut max_program_id_dim,
        );
    }

    builder.ret(None);
    let func = builder.finish();
    let abi = build_kernel_abi(typed, max_program_id_dim + 1);

    ExecutableKernel {
        name: typed.kernel.name.clone(),
        abi,
        func,
        value_info,
    }
}
```

Add these helpers at the bottom of the same file:

```rust
fn build_kernel_abi(typed: &TypedKernel, program_id_dims: usize) -> KernelAbi {
    let mut offsets = Vec::with_capacity(typed.kernel.params.len());
    let mut next = 0usize;

    let params = typed
        .kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let rank = rank_of_param(param);
            offsets.push(next);
            next += rank;
            KernelParamAbi {
                index,
                name: param.name.clone(),
                kind: param.kind,
                elem: ElemType::F32,
                rank,
                passing: ParamPassing::Buffer,
            }
        })
        .collect();

    KernelAbi {
        params,
        shape_layout: ShapeLayout {
            total_dims: next,
            offsets,
        },
        launch: LaunchSemantics { program_id_dims },
    }
}

fn rank_of_param(param: &sile_hir::Param) -> usize {
    match &param.ty {
        sile_hir::Type::Tensor { shape, .. } | sile_hir::Type::Tile { shape, .. } => shape.rank(),
        sile_hir::Type::Shape | sile_hir::Type::Scalar(_) => 0,
    }
}
```

- [ ] **Step 5: Record `ValueInfo` directly while lowering instructions**

In `lower_ssa_instruction`, replace the old `tile_shapes` bookkeeping with direct `ValueInfo` writes like this:

```rust
fn push_instruction_info(value_info: &mut ValueInfoTable, def_idx: usize, info: ValueInfo) {
    if value_info.instructions.len() <= def_idx {
        value_info
            .instructions
            .resize(def_idx + 1, ValueInfo::Void);
    }
    value_info.instructions[def_idx] = info;
}

fn info_for_binary(
    lhs: &SsaValue,
    rhs: &SsaValue,
    table: &ValueInfoTable,
) -> ValueInfo {
    info_for_value(lhs, table)
        .or_else(|| info_for_value(rhs, table))
        .unwrap_or(ValueInfo::Scalar { elem: ElemType::F32 })
}

fn info_for_value(value: &SsaValue, table: &ValueInfoTable) -> Option<ValueInfo> {
    match value {
        SsaValue::Param(i) => table.params.get(*i).cloned(),
        SsaValue::Local(i) => table.instructions.get(*i).cloned(),
        SsaValue::Const(_) => Some(ValueInfo::Scalar { elem: ElemType::F32 }),
    }
}
```

Then use these rules in the opcode match:

```rust
SsaOpcode::ProgramId => {
    push_instruction_info(value_info, def_idx, ValueInfo::Index);
    Some(builder.const_int(0))
}
SsaOpcode::ShapeDim => {
    push_instruction_info(value_info, def_idx, ValueInfo::Shape);
    // existing value lowering remains
}
SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
    push_instruction_info(
        value_info,
        def_idx,
        ValueInfo::Tile {
            elem: ElemType::F32,
            rows,
            cols,
        },
    );
    // existing tile load builder call remains
}
SsaOpcode::Add | SsaOpcode::Sub | SsaOpcode::Mul | SsaOpcode::Div | SsaOpcode::Exp => {
    push_instruction_info(value_info, def_idx, info_for_binary(&inst.uses[0], &inst.uses[1], value_info));
    // existing builder call remains
}
SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
    push_instruction_info(
        value_info,
        def_idx,
        if axis == 1 {
            ValueInfo::Tile {
                elem: ElemType::F32,
                rows,
                cols: 1,
            }
        } else {
            ValueInfo::Tile {
                elem: ElemType::F32,
                rows: 1,
                cols,
            }
        },
    );
    // existing builder call remains
}
SsaOpcode::Broadcast => {
    push_instruction_info(
        value_info,
        def_idx,
        ValueInfo::Tile {
            elem: ElemType::F32,
            rows,
            cols,
        },
    );
    // existing builder call remains
}
SsaOpcode::Store => {
    push_instruction_info(value_info, def_idx, ValueInfo::Void);
    // existing store emission remains
}
```

Use the same pattern for `Constant` and `Mma`, and update `max_program_id_dim` whenever `builder.get_tile_coord(dim)` is emitted.

- [ ] **Step 6: Run the compiler test to verify it passes**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-compiler --test executable_kernel -- --nocapture
```

Expected: PASS with `1 passed`.

- [ ] **Step 7: Commit the compiler output change**

Run:

```bash
rtk git add crates/sile-compiler/src/lib.rs crates/sile-compiler/src/lower_lir.rs crates/sile-compiler/tests/executable_kernel.rs
rtk git commit -m "feat: compile kernels into executable lir"
```

Expected: commit succeeds.

## Task 3: Add An LLVM-Style Printer For `ExecutableKernel`

**Files:**
- Create: `crates/sile-lir/src/print.rs`
- Create: `crates/sile-lir/tests/print.rs`
- Modify: `crates/sile-lir/src/lib.rs`
- Test: `crates/sile-lir/tests/print.rs`

- [ ] **Step 1: Write the failing printer test**

Create `crates/sile-lir/tests/print.rs`:

```rust
use sile_hir::ParamKind;
use sile_lir::{
    print::format_executable_kernel, ElemType, ExecutableKernel, FloatType, Function, KernelAbi,
    KernelParamAbi, LaunchSemantics, Param, ParamPassing, ShapeLayout, Type, ValueInfo,
    ValueInfoTable,
};

#[test]
fn printer_includes_abi_and_function_sections() {
    let kernel = ExecutableKernel {
        name: "softmax".into(),
        abi: KernelAbi {
            params: vec![KernelParamAbi {
                index: 0,
                name: "x".into(),
                kind: ParamKind::Input,
                elem: ElemType::F32,
                rank: 2,
                passing: ParamPassing::Buffer,
            }],
            shape_layout: ShapeLayout {
                total_dims: 2,
                offsets: vec![0],
            },
            launch: LaunchSemantics { program_id_dims: 2 },
        },
        func: Function::new(
            "softmax",
            vec![Param {
                name: "x".into(),
                ty: Type::ptr(Type::Float(FloatType::F32)),
            }],
            Type::Void,
        ),
        value_info: ValueInfoTable {
            params: vec![ValueInfo::Buffer {
                elem: ElemType::F32,
                rank: 2,
            }],
            instructions: vec![ValueInfo::Index],
        },
    };

    let text = format_executable_kernel(&kernel);
    assert!(text.contains("kernel @softmax"));
    assert!(text.contains("abi:"));
    assert!(text.contains("arg0 input buffer<f32, rank=2>"));
    assert!(text.contains("launch program_id_dims=2"));
    assert!(text.contains("func:"));
}
```

- [ ] **Step 2: Run the printer test to verify it fails**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-lir --test print -- --nocapture
```

Expected: FAIL because `print::format_executable_kernel` does not exist yet.

- [ ] **Step 3: Add the printer module**

Create `crates/sile-lir/src/print.rs`:

```rust
use std::fmt::Write;

use crate::{ExecutableKernel, ParamPassing, ValueInfo};
use sile_hir::ParamKind;

pub fn format_executable_kernel(kernel: &ExecutableKernel) -> String {
    let mut out = String::new();
    writeln!(&mut out, "kernel @{}", kernel.name).unwrap();
    writeln!(&mut out, "abi:").unwrap();

    for param in &kernel.abi.params {
        let kind = match param.kind {
            ParamKind::Input => "input",
            ParamKind::Output => "output",
        };
        let passing = match param.passing {
            ParamPassing::Buffer => "buffer",
        };
        let offset = kernel.abi.shape_layout.offsets[param.index];
        writeln!(
            &mut out,
            "  arg{} {} {}<f32, rank={}> shape_offset={}",
            param.index, kind, passing, param.rank, offset
        )
        .unwrap();
    }

    writeln!(
        &mut out,
        "  total_dims={}",
        kernel.abi.shape_layout.total_dims
    )
    .unwrap();
    writeln!(
        &mut out,
        "  launch program_id_dims={}",
        kernel.abi.launch.program_id_dims
    )
    .unwrap();
    writeln!(&mut out, "func:").unwrap();

    for block in &kernel.func.blocks {
        writeln!(&mut out, "{}:", block.label).unwrap();
        for (idx, inst) in block.instructions.iter().enumerate() {
            let info = kernel
                .value_info
                .instructions
                .get(idx)
                .map(format_value_info)
                .unwrap_or_else(|| "void".into());
            writeln!(&mut out, "  %{}:{} = {:?}", idx, info, inst).unwrap();
        }
        writeln!(&mut out, "  {:?}", block.terminator).unwrap();
    }

    out
}

fn format_value_info(info: &ValueInfo) -> String {
    match info {
        ValueInfo::Buffer { rank, .. } => format!("buffer<f32, rank={rank}>"),
        ValueInfo::Scalar { .. } => "scalar<f32>".into(),
        ValueInfo::Index => "index".into(),
        ValueInfo::Shape => "shape".into(),
        ValueInfo::Tile { rows, cols, .. } => format!("tile<f32, {rows}x{cols}>"),
        ValueInfo::Void => "void".into(),
    }
}
```

- [ ] **Step 4: Re-export the printer**

Ensure `crates/sile-lir/src/lib.rs` still contains:

```rust
pub mod print;
```

No additional code needed here if Task 1 already added it.

- [ ] **Step 5: Run the printer test to verify it passes**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile-lir --test print -- --nocapture
```

Expected: PASS with `1 passed`.

- [ ] **Step 6: Commit the printer**

Run:

```bash
rtk git add crates/sile-lir/src/print.rs crates/sile-lir/tests/print.rs crates/sile-lir/src/lib.rs
rtk git commit -m "feat: print executable kernels"
```

Expected: commit succeeds.

## Task 4: Switch Runtime And Backends To `ExecutableKernel`

**Files:**
- Modify: `crates/sile-lir/src/backend.rs`
- Modify: `crates/sile/src/kernel_launcher.rs`
- Modify: `crates/sile-backend-cpu/src/lib.rs`
- Modify: `crates/sile-backend-cpu/src/codegen_c.rs`
- Modify: `crates/sile-backend-metal/src/lib.rs`
- Modify: `crates/sile-backend-metal/src/codegen_metal.rs`
- Test: `crates/sile/tests/backend_vec_add.rs`
- Test: `crates/sile/tests/backend_softmax.rs`
- Test: `crates/sile/tests/backend_matmul.rs`
- Test: `crates/sile/tests/backend_metal.rs`

- [ ] **Step 1: Write a failing launcher integration test**

Append this test to `crates/sile/tests/backend_vec_add.rs`:

```rust
#[test]
fn compiler_output_is_executable_kernel_for_cpu_backend() {
    let typed = sile_hir::typeck::check_kernel(vec_add::kernel()).expect("typed kernel");
    let executable = sile_compiler::compile(&typed);

    assert_eq!(executable.name, "vec_add");
    assert_eq!(executable.abi.params.len(), 3);
    assert_eq!(executable.abi.launch.program_id_dims, 1);
}
```

If `vec_add::kernel()` is not available from the macro output, put the same assertion in a fresh test file `crates/sile/tests/executable_kernel_runtime.rs` and use a tiny local `#[sile::kernel] fn vec_add(...) { ... }` definition. The exact body can match the existing `backend_vec_add.rs` kernel.

- [ ] **Step 2: Run the targeted integration test to verify it fails**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile compiler_output_is_executable_kernel_for_cpu_backend -- --nocapture
```

Expected: FAIL because `backend_vec_add` still receives a bare `Function` and runtime plumbing has not been updated.

- [ ] **Step 3: Change the backend trait to accept `ExecutableKernel`**

Update `crates/sile-lir/src/backend.rs` to:

```rust
use crate::ExecutableKernel;
use sile_core::{KernelArg, LaunchConfig, Result, Stream};

pub trait Backend: Send + Sync {
    fn execute(
        &self,
        kernel: &ExecutableKernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()>;
}
```

- [ ] **Step 4: Update the runtime launcher**

Update `crates/sile/src/kernel_launcher.rs` to:

```rust
use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_lir::{print::format_executable_kernel, Backend};

pub struct KernelLauncher<'a> {
    kernel: &'static Kernel,
    args: Vec<KernelArg<'a>>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(kernel: &'static Kernel, args: Vec<KernelArg<'a>>) -> Self {
        Self {
            kernel,
            args,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn kernel(&self) -> &'static Kernel {
        self.kernel
    }

    pub fn apply(self, stream: &Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self
                .grid
                .ok_or_else(|| sile_core::Error::Shape("grid not set".into()))?,
        };

        let typed = sile_hir::typeck::check_kernel(self.kernel)
            .map_err(|e| sile_core::Error::Shape(e.to_string()))?;
        let executable = sile_compiler::compile(&typed);

        if std::env::var_os("SILE_PRINT_LIR").is_some() {
            eprintln!("{}", format_executable_kernel(&executable));
        }

        match stream.device() {
            Device::Cpu(_) => {
                let backend = sile_backend_cpu::CpuBackend::new();
                backend.execute(&executable, &self.args, &launch, stream)
            }
            Device::Metal(_) => {
                let backend = sile_backend_metal::MetalBackend::new()?;
                backend.execute(&executable, &self.args, &launch, stream)
            }
            _ => Err(sile_core::Error::UnsupportedBackend(
                "backend not implemented",
            )),
        }
    }
}
```

- [ ] **Step 5: Replace CPU backend-local metadata with `KernelAbi`**

Update `crates/sile-backend-cpu/src/lib.rs` so the execute signature is:

```rust
impl sile_lir::Backend for CpuBackend {
    fn execute(
        &self,
        kernel: &sile_lir::ExecutableKernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let c_code = generate(&kernel.func, &kernel.abi, &kernel.value_info)?;
        // existing compile-and-run path stays the same
    }
}
```

Delete the local `KernelGenInfo` builder and move ABI access into `codegen_c.rs`. Change the generator signature there to:

```rust
pub fn generate(
    func: &Function,
    abi: &KernelAbi,
    value_info: &ValueInfoTable,
) -> sile_core::Result<String>
```

Inside `codegen_c.rs`, replace:
- `info.num_buffers` with `abi.params.len()`
- `info.buffer_kinds` with `abi.params[i].kind`
- `info.param_ranks` with `abi.params[i].rank`
- `info.shape_offsets` with `abi.shape_layout.offsets`
- `analyze_instruction_shapes(func)` with `value_info.instructions`

- [ ] **Step 6: Make the same backend change for Metal**

Apply the same pattern to:
- `crates/sile-backend-metal/src/lib.rs`
- `crates/sile-backend-metal/src/codegen_metal.rs`

Use this signature:

```rust
pub fn generate(
    func: &Function,
    abi: &KernelAbi,
    value_info: &ValueInfoTable,
) -> sile_core::Result<String>
```

Then replace the same five categories of local metadata reads as in CPU codegen.

- [ ] **Step 7: Run the runtime and backend tests**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test -p sile backend_vec_add -- --nocapture
RUSTC_WRAPPER= rtk cargo test -p sile backend_softmax -- --nocapture
RUSTC_WRAPPER= rtk cargo test -p sile backend_matmul -- --nocapture
RUSTC_WRAPPER= rtk cargo test -p sile --test backend_metal -- --nocapture
```

Expected:
- CPU tests PASS
- `backend_metal` PASS on macOS with Metal available

- [ ] **Step 8: Commit the runtime/backend migration**

Run:

```bash
rtk git add crates/sile-lir/src/backend.rs crates/sile/src/kernel_launcher.rs crates/sile-backend-cpu/src/lib.rs crates/sile-backend-cpu/src/codegen_c.rs crates/sile-backend-metal/src/lib.rs crates/sile-backend-metal/src/codegen_metal.rs crates/sile/tests/backend_vec_add.rs
rtk git commit -m "refactor: route backends through executable lir"
```

Expected: commit succeeds.

## Task 5: Full Verification And LIR Print Validation

**Files:**
- Verify only; no new source files required
- Test: `crates/sile/examples/vec_add.rs`
- Test: `crates/sile/examples/softmax.rs`
- Test: `crates/sile/examples/matmul.rs`

- [ ] **Step 1: Verify the full workspace test suite**

Run:

```bash
RUSTC_WRAPPER= rtk cargo test
```

Expected: PASS with no failing suites.

- [ ] **Step 2: Verify LIR printing on CPU example**

Run:

```bash
RUSTC_WRAPPER= SILE_PRINT_LIR=1 rtk cargo run -p sile --example vec_add
```

Expected:
- Program prints a `kernel @vec_add` section
- Output contains `abi:`
- Output contains `func:`
- Program still finishes successfully

- [ ] **Step 3: Verify printed LIR on Metal example**

Run:

```bash
RUSTC_WRAPPER= SILE_DEVICE=METAL SILE_PRINT_LIR=1 rtk cargo run -p sile --example softmax
```

Expected:
- Printed IR includes `launch program_id_dims=2`
- Program still prints row sums equal to `1`

- [ ] **Step 4: Verify all three Metal examples still pass**

Run:

```bash
RUSTC_WRAPPER= SILE_DEVICE=METAL rtk cargo run -p sile --example vec_add
RUSTC_WRAPPER= SILE_DEVICE=METAL rtk cargo run -p sile --example softmax
RUSTC_WRAPPER= SILE_DEVICE=METAL rtk cargo run -p sile --example matmul
```

Expected:
- `vec_add` prints the correct result vector
- `softmax` prints row sums of `1`
- `matmul` exits successfully without assertion failures

- [ ] **Step 5: Commit the verification-only cleanup if needed**

If verification required tiny non-functional fixes such as import cleanup or printer formatting adjustments, run:

```bash
rtk git add crates/sile-lir/src/print.rs crates/sile/src/kernel_launcher.rs crates/sile-backend-cpu/src/codegen_c.rs crates/sile-backend-metal/src/codegen_metal.rs
rtk git commit -m "chore: polish executable lir output"
```

If no code changed during verification, skip this step and leave the checkbox unchecked with note `No changes needed`.

## Self-Review

**Spec coverage:** This plan covers the requested execution-boundary redesign:
- Introduces a printable `ExecutableKernel`
- Moves backend ABI metadata out of backend-local `KernelGenInfo`
- Keeps `HIR Kernel` for frontend/typeck only
- Changes compiler output and backend trait to consume a single executable package
- Adds an LLVM-style printer and runtime print hook
- Verifies CPU and Metal backends with tests and examples

**Placeholder scan:** No `TODO`, `TBD`, “add appropriate handling”, or “similar to previous task” placeholders remain. All code steps include concrete snippets, concrete files, and concrete commands.

**Type consistency:** The same names are used throughout:
- `ExecutableKernel`
- `KernelAbi`
- `KernelParamAbi`
- `ShapeLayout`
- `LaunchSemantics`
- `ValueInfo`
- `ValueInfoTable`
- `ParamPassing::Buffer`

These names are used consistently in `sile-lir`, `sile-compiler`, runtime, and backends.
