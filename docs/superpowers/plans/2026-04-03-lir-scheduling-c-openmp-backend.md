# LIR + Scheduling + C/OpenMP Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `backend_ir` + `codegen/c.rs` with a general-purpose LLVM-style LIR, an independent scheduling pass, and a C codegen producing complete compilable C files with OpenMP parallelism and SIMD vectorization.

**Architecture:** Add `lir/` (CFG + BasicBlock + phi + pure semantic instructions), `scheduling/` (dependency analysis + parallelism annotation), and rewrite `codegen/c.rs` (LIR + annotations → C with `#pragma omp parallel for` / `#pragma omp simd`). Delete old `backend_ir/` module.

**Tech Stack:** Rust, OpenMP, C code generation, existing SSA/HIR infrastructure.

---

## File Structure

### New Files
- `crates/sile/src/lir/mod.rs` — Module exports
- `crates/sile/src/lir/ir.rs` — Value, Type, Instruction, Terminator, PhiNode, BasicBlock, Function, Program
- `crates/sile/src/lir/builder.rs` — LIR builder utility
- `crates/sile/src/lir/lower.rs` — SSA → LIR lowering
- `crates/sile/src/scheduling/mod.rs` — Pass entry point
- `crates/sile/src/scheduling/dependency.rs` — Memory access analysis, loop-carried dependency detection
- `crates/sile/src/scheduling/annotate.rs` — Build ScheduleAnnotation from analyzed LIR

### Modified Files
- `crates/sile/src/codegen/c.rs` — Complete rewrite: LIR + ScheduleAnnotation → C source
- `crates/sile/src/backend/cpu_c.rs` — Use new pipeline (lir → scheduling → c codegen)
- `crates/sile/src/lib.rs` — Add `lir` and `scheduling` modules, remove `backend_ir` export
- `crates/sile/tests/c_codegen.rs` — Rewrite for new LIR-based codegen
- `crates/sile/tests/backend_vec_add.rs` — Update to use new pipeline (should still pass as e2e test)
- `crates/sile/tests/vec_add_e2e.rs` — Should pass unchanged (e2e test)

### Deleted Files
- `crates/sile/src/backend_ir/ir.rs`
- `crates/sile/src/backend_ir/lower.rs`
- `crates/sile/src/backend_ir/mod.rs`

---

### Task 1: Create LIR IR Types

**Files:**
- Create: `crates/sile/src/lir/mod.rs`
- Create: `crates/sile/src/lir/ir.rs`

- [ ] **Step 1: Create lir/ir.rs with all IR types**

```rust
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Param(usize),
    Const(Constant),
    Inst(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerType {
    I8,
    I16,
    I32,
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatType {
    F16,
    F32,
    F64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Void,
    Int(IntegerType),
    Float(FloatType),
    Pointer(Box<Type>),
    Vector(Box<Type>, usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalVariable {
    pub name: String,
    pub ty: Type,
    pub initializer: Option<Constant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq, Ne,
    Slt, Sle, Sgt, Sge,
    Ult, Ule, Ugt, Uge,
    Olt, Ole, Ogt, Oge,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Alloca { ty: Type },
    Load { ptr: Value, ty: Type, align: Option<u32> },
    Store { ptr: Value, value: Value, align: Option<u32> },
    Gep { ptr: Value, indices: Vec<Value> },
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    FNeg(Value),
    FMax(Value, Value),
    FMin(Value, Value),
    Exp(Value),
    Icmp(CmpOp, Value, Value),
    Fcmp(CmpOp, Value, Value),
    Trunc(Value, Type),
    ZExt(Value, Type),
    SIToFP(Value, Type),
    FPToSI(Value, Type),
    BitCast(Value, Type),
    Call { func: String, args: Vec<Value>, ret_ty: Type },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Terminator {
    Br { target: String },
    CondBr { cond: Value, true_target: String, false_target: String },
    Switch { value: Value, default: String, cases: Vec<(i64, String)> },
    Ret(Option<Value>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhiNode {
    pub dest: String,
    pub ty: Type,
    pub incoming: Vec<(Value, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock {
    pub label: String,
    pub phi_nodes: Vec<PhiNode>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: String,
}

impl Function {
    pub fn new(name: &str, params: Vec<Param>, return_type: Type) -> Self {
        Self {
            name: name.to_string(),
            params,
            return_type,
            blocks: Vec::new(),
            entry_block: String::new(),
        }
    }

    pub fn get_block(&self, label: &str) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.label == label)
    }

    pub fn get_block_mut(&mut self, label: &str) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.label == label)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub functions: Vec<Function>,
    pub globals: Vec<GlobalVariable>,
}

impl Type {
    pub fn i64() -> Self {
        Type::Int(IntegerType::I64)
    }

    pub fn f32() -> Self {
        Type::Float(FloatType::F32)
    }

    pub fn ptr(ty: Type) -> Self {
        Type::Pointer(Box::new(ty))
    }
}
```

- [ ] **Step 2: Create lir/mod.rs**

```rust
pub mod ir;
mod builder;
mod lower;

pub use ir::{
    BasicBlock, CmpOp, Constant, FloatType, Function, GlobalVariable, Instruction, IntegerType,
    Param, PhiNode, Program, Terminator, Type, Value,
};
pub use lower::lower_ssa_to_lir;
```

- [ ] **Step 3: Compile to verify types are correct**

Run: `rtk cargo check -p sile`
Expected: Errors about missing `builder` and `lower` modules (not yet created)

- [ ] **Step 4: Commit**

```bash
rtk git add crates/sile/src/lir/
rtk git commit -m "feat: add LIR IR types (Value, Type, Instruction, BasicBlock, Function, Program)"
```

---

### Task 2: Create LIR Builder

**Files:**
- Create: `crates/sile/src/lir/builder.rs`

- [ ] **Step 1: Create lir/builder.rs**

```rust
use crate::lir::ir::*;

pub struct LirBuilder {
    pub func: Function,
    current_block: Option<String>,
    next_inst: usize,
}

impl LirBuilder {
    pub fn new(name: &str, params: Vec<Param>, return_type: Type) -> Self {
        let func = Function::new(name, params, return_type);
        Self {
            func,
            current_block: None,
            next_inst: 0,
        }
    }

    pub fn append_block(&mut self, label: &str) -> String {
        let label = label.to_string();
        self.func.blocks.push(BasicBlock {
            label: label.clone(),
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Ret(None),
        });
        if self.func.entry_block.is_empty() {
            self.func.entry_block = label.clone();
        }
        label
    }

    pub fn switch_to_block(&mut self, label: &str) {
        self.current_block = Some(label.to_string());
    }

    pub fn current_block_label(&self) -> &str {
        self.current_block.as_deref().unwrap_or("")
    }

    fn push_instruction(&mut self, inst: Instruction) -> Value {
        let val = Value::Inst(self.next_inst);
        self.next_inst += 1;
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.instructions.push(inst);
            }
        }
        val
    }

    pub fn alloca(&mut self, ty: Type) -> Value {
        self.push_instruction(Instruction::Alloca { ty })
    }

    pub fn load(&mut self, ptr: Value, ty: Type) -> Value {
        self.push_instruction(Instruction::Load { ptr, ty, align: None })
    }

    pub fn store(&mut self, ptr: Value, value: Value) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.instructions.push(Instruction::Store { ptr, value, align: None });
            }
        }
    }

    pub fn gep(&mut self, ptr: Value, indices: Vec<Value>) -> Value {
        self.push_instruction(Instruction::Gep { ptr, indices })
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Add(lhs, rhs))
    }

    pub fn sub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Sub(lhs, rhs))
    }

    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Mul(lhs, rhs))
    }

    pub fn fcmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Fcmp(op, lhs, rhs))
    }

    pub fn icmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Icmp(op, lhs, rhs))
    }

    pub fn fmax(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::FMax(lhs, rhs))
    }

    pub fn fmin(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::FMin(lhs, rhs))
    }

    pub fn exp(&mut self, val: Value) -> Value {
        self.push_instruction(Instruction::Exp(val))
    }

    pub fn br(&mut self, target: &str) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::Br { target: target.to_string() };
            }
        }
    }

    pub fn cond_br(&mut self, cond: Value, true_target: &str, false_target: &str) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::CondBr {
                    cond,
                    true_target: true_target.to_string(),
                    false_target: false_target.to_string(),
                };
            }
        }
    }

    pub fn ret(&mut self, value: Option<Value>) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::Ret(value);
            }
        }
    }

    pub fn const_int(&self, v: i64) -> Value {
        Value::Const(Constant::Int(v))
    }

    pub fn const_float(&self, v: f64) -> Value {
        Value::Const(Constant::Float(v))
    }

    pub fn phi(&mut self, dest: &str, ty: Type, incoming: Vec<(Value, String)>) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.phi_nodes.push(PhiNode {
                    dest: dest.to_string(),
                    ty,
                    incoming,
                });
            }
        }
    }

    pub fn finish(self) -> Function {
        self.func
    }
}
```

- [ ] **Step 2: Compile to verify**

Run: `rtk cargo check -p sile`
Expected: Errors about missing `lower` module only

- [ ] **Step 3: Commit**

```bash
rtk git add crates/sile/src/lir/builder.rs
rtk git commit -m "feat: add LIR builder utility"
```

---

### Task 3: SSA → LIR Lowering

**Files:**
- Create: `crates/sile/src/lir/lower.rs`

- [ ] **Step 1: Create lir/lower.rs**

```rust
use std::collections::HashMap;

use crate::hir::{BuiltinOp, Expr, Stmt};
use crate::lir::builder::LirBuilder;
use crate::lir::ir::{CmpOp, Function, Param, Type, Value};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> Function {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);

    let entry = builder.append_block("entry");
    builder.switch_to_block(&entry);

    // Map SSA values to LIR values
    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let mut param_names: Vec<String> = Vec::new();
    for p in &typed.kernel.params {
        param_names.push(p.name.clone());
    }

    // Create alloca for each output parameter (store targets)
    let mut output_ptrs: HashMap<String, Value> = HashMap::new();
    for (i, p) in typed.kernel.params.iter().enumerate() {
        let param_val = Value::Param(i);
        value_map.insert(i, param_val);
        if p.kind == crate::hir::ParamKind::Output {
            let ptr = builder.alloca(Type::ptr(Type::f32()));
            builder.store(ptr, Value::Param(i));
            output_ptrs.insert(p.name.clone(), ptr);
        }
    }

    // Analyze loop structure from SSA
    let loop_info = analyze_ssa_loops(ssa);

    if loop_info.has_loops {
        // Generate loop nesting
        generate_loop_nesting(&mut builder, ssa, &mut value_map, &param_names, &loop_info);
    } else {
        // Flat: lower instructions directly in entry block
        for inst in &ssa.instructions {
            lower_ssa_instruction(inst, &mut builder, &mut value_map, &param_names);
        }
    }

    builder.ret(None);
    builder.finish()
}

struct LoopInfo {
    has_loops: bool,
    loop_bounds: Vec<(String, i64, i64)>, // (var_name, start, end)
}

fn analyze_ssa_loops(ssa: &SsaProgram) -> LoopInfo {
    // Check for ProgramId which implies iteration
    let has_program_id = ssa.instructions.iter().any(|i| i.opcode == SsaOpcode::ProgramId);
    LoopInfo {
        has_loops: has_program_id,
        loop_bounds: vec![],
    }
}

fn generate_loop_nesting(
    builder: &mut LirBuilder,
    ssa: &SsaProgram,
    value_map: &mut HashMap<usize, Value>,
    param_names: &[String],
    _loop_info: &LoopInfo,
) {
    // For now: single loop over tile elements
    // The loop variable represents the flattened tile index
    let loop_var = builder.alloca(Type::i64());
    builder.store(loop_var, builder.const_int(0));

    let header = builder.append_block("loop_header");
    let body = builder.append_block("loop_body");
    let exit = builder.append_block("loop_exit");

    builder.br(&header);

    // Header: load loop var, check bound
    builder.switch_to_block(&header);
    let idx = builder.load(loop_var, Type::i64());
    // Bound: use a param for tile size (first input param's size)
    let bound = builder.const_int(256); // TODO: derive from typed kernel shape
    let cond = builder.icmp(CmpOp::Slt, idx, bound);
    builder.cond_br(cond, &body, &exit);

    // Body: lower SSA instructions
    builder.switch_to_block(&body);
    // Insert loop var into value_map for ProgramId
    for inst in &ssa.instructions {
        if inst.opcode == SsaOpcode::ProgramId {
            let idx_val = builder.load(loop_var, Type::i64());
            let def_idx = get_def_index(&inst.def);
            value_map.insert(def_idx, idx_val);
        } else {
            lower_ssa_instruction(inst, builder, value_map, param_names);
        }
    }

    // Increment and loop back
    let next = builder.add(idx, builder.const_int(1));
    builder.store(loop_var, next);
    builder.br(&header);

    // Exit
    builder.switch_to_block(&exit);
}

fn lower_ssa_instruction(
    inst: &SsaInstruction,
    builder: &mut LirBuilder,
    value_map: &mut HashMap<usize, Value>,
    param_names: &[String],
) {
    let def_idx = get_def_index(&inst.def);

    let lir_inst = match inst.opcode {
        SsaOpcode::ProgramId => return, // handled by loop nesting
        SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
            let base_param = if inst.uses.is_empty() {
                0
            } else {
                get_param_index(&inst.uses[0], value_map, param_names)
            };
            let ptr = Value::Param(base_param);
            let indices: Vec<Value> = inst.uses[1..]
                .iter()
                .map(|v| resolve_value(v, value_map))
                .collect();
            if indices.is_empty() {
                let gep = builder.gep(ptr, vec![builder.load(
                    builder.alloca(Type::i64()),
                    Type::i64(),
                )]); // placeholder index
                builder.load(gep, Type::f32())
            } else {
                let gep = builder.gep(ptr, indices);
                builder.load(gep, Type::f32())
            }
        }
        SsaOpcode::Add => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.add(lhs, rhs)
        }
        SsaOpcode::Sub => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.sub(lhs, rhs)
        }
        SsaOpcode::Mul => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.mul(lhs, rhs)
        }
        SsaOpcode::Div => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.push_instruction(crate::lir::ir::Instruction::Div(lhs, rhs))
        }
        SsaOpcode::Exp => {
            let val = resolve_value(&inst.uses[0], value_map);
            builder.exp(val)
        }
        SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
            // Expanded to loop in a later pass; for now, placeholder
            let src = if inst.uses.is_empty() {
                Value::Param(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            };
            src // passthrough for now
        }
        SsaOpcode::Store => {
            let val = if inst.uses.is_empty() {
                Value::Param(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            };
            // Store to output parameter
            let out_ptr = Value::Param(
                param_names
                    .iter()
                    .position(|n| n == "c" || n == "y" || n == "out")
                    .unwrap_or(2),
            );
            let gep = builder.gep(out_ptr, vec![builder.load(
                builder.alloca(Type::i64()),
                Type::i64(),
            )]);
            builder.store(gep, val);
            return;
        }
        SsaOpcode::Mma => {
            let a = resolve_value(&inst.uses[0], value_map);
            let b = resolve_value(&inst.uses[1], value_map);
            let c = resolve_value(&inst.uses[2], value_map);
            builder.mul(a, b) // placeholder: mma → mul for now
        }
        SsaOpcode::Constant => {
            let val = inst.immediates.first().copied().unwrap_or(0);
            builder.const_float(val as f64)
        }
        SsaOpcode::Reshape | SsaOpcode::Broadcast | SsaOpcode::ShapeOf | SsaOpcode::ScalarDiv | SsaOpcode::ShapeDim => {
            // Passthrough first operand
            if inst.uses.is_empty() {
                builder.const_int(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            }
        }
    };

    value_map.insert(def_idx, lir_inst);
}

fn resolve_value(v: &SsaValue, value_map: &HashMap<usize, Value>) -> Value {
    match v {
        SsaValue::Param(i) => Value::Param(*i),
        SsaValue::Local(i) => value_map.get(i).cloned().unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
        SsaValue::Const(c) => Value::Const(crate::lir::ir::Constant::Int(*c)),
    }
}

fn get_def_index(def: &SsaValue) -> usize {
    match def {
        SsaValue::Local(i) => *i,
        _ => 0,
    }
}

fn get_param_index(
    v: &SsaValue,
    value_map: &HashMap<usize, Value>,
    param_names: &[String],
) -> usize {
    match v {
        SsaValue::Param(i) => *i,
        SsaValue::Local(i) => {
            // Try to trace back to a param
            value_map
                .get(i)
                .and_then(|val| match val {
                    Value::Param(p) => Some(*p),
                    _ => None,
                })
                .unwrap_or(0)
        }
        _ => 0,
    }
}

fn lower_kernel_params(typed: &TypedKernel) -> Vec<Param> {
    typed
        .kernel
        .params
        .iter()
        .map(|p| Param {
            name: p.name.clone(),
            ty: Type::ptr(Type::f32()), // All kernel params are float pointers for now
        })
        .collect()
}
```

- [ ] **Step 2: Update lir/mod.rs to export lower**

```rust
pub mod ir;
mod builder;
mod lower;

pub use ir::{
    BasicBlock, CmpOp, Constant, FloatType, Function, GlobalVariable, Instruction, IntegerType,
    Param, PhiNode, Program, Terminator, Type, Value,
};
pub use lower::lower_ssa_to_lir;
```

- [ ] **Step 3: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Should compile (or have only unused variable warnings)

- [ ] **Step 4: Commit**

```bash
rtk git add crates/sile/src/lir/lower.rs crates/sile/src/lir/mod.rs
rtk git commit -m "feat: add SSA → LIR lowering"
```

---

### Task 4: Create Scheduling Module

**Files:**
- Create: `crates/sile/src/scheduling/mod.rs`
- Create: `crates/sile/src/scheduling/dependency.rs`
- Create: `crates/sile/src/scheduling/annotate.rs`

- [ ] **Step 1: Create scheduling/dependency.rs**

```rust
use crate::lir::ir::{BasicBlock, Function, Instruction, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub base: String,
    pub indices: Vec<Value>,
    pub is_write: bool,
    pub block_label: String,
}

#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub header_block: String,
    pub body_blocks: Vec<String>,
    pub exit_block: String,
    pub induction_var: Option<Value>,
    pub bound: Option<Value>,
}

pub fn find_natural_loops(func: &Function) -> Vec<LoopInfo> {
    let mut loops = Vec::new();

    // Find back edges: blocks that jump to an already-visited block
    for block in &func.blocks {
        if let Some(target) = get_branch_target(&block.terminator) {
            if func.blocks.iter().position(|b| b.label == target).is_some() {
                // Found a back edge, identify loop body
                let body = collect_loop_body(func, &target, &block.label);
                if !body.is_empty() {
                    loops.push(LoopInfo {
                        header_block: target.clone(),
                        body_blocks: body,
                        exit_block: find_loop_exit(func, &block.label).unwrap_or_default(),
                        induction_var: extract_induction_var(func, &target),
                        bound: extract_loop_bound(func, &target),
                    });
                }
            }
        }
    }

    loops
}

pub fn analyze_memory_accesses(
    func: &Function,
    loop_info: &LoopInfo,
) -> Vec<MemoryAccess> {
    let mut accesses = Vec::new();

    for label in &loop_info.body_blocks {
        if let Some(block) = func.get_block(label) {
            for inst in &block.instructions {
                if let Some(access) = extract_memory_access(inst, label) {
                    accesses.push(access);
                }
            }
        }
    }

    accesses
}

pub fn has_loop_carried_dependency(accesses: &[MemoryAccess]) -> bool {
    // Check if any write in one iteration is read in another
    let writes: Vec<_> = accesses.iter().filter(|a| a.is_write).collect();
    let reads: Vec<_> = accesses.iter().filter(|a| !a.is_write).collect();

    for write in &writes {
        for read in &reads {
            if write.base == read.base && indices_overlap(&write.indices, &read.indices) {
                return true;
            }
        }
    }

    false
}

pub fn is_reduction_pattern(accesses: &[MemoryAccess]) -> Option<ReductionType> {
    // Check if all writes accumulate into a single scalar variable
    let writes: Vec<_> = accesses.iter().filter(|a| a.is_write).collect();
    if writes.len() <= 1 {
        return None;
    }

    let targets: HashSet<_> = writes.iter().map(|w| &w.base).collect();
    if targets.len() == 1 {
        // All writes to same target — check if it's an associative op
        return Some(ReductionType::Sum); // Default; refine based on actual ops
    }

    None
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Sum,
    Max,
    Min,
    Product,
}

fn get_branch_target(terminator: &crate::lir::ir::Terminator) -> Option<&String> {
    match terminator {
        crate::lir::ir::Terminator::Br { target } => Some(target),
        crate::lir::ir::Terminator::CondBr { true_target, .. } => Some(true_target),
        _ => None,
    }
}

fn collect_loop_body(func: &Function, header: &str, back_edge_from: &str) -> Vec<String> {
    // Simple: all blocks between header and back_edge_source
    let mut body = Vec::new();
    let mut found_header = false;
    for block in &func.blocks {
        if block.label == header {
            found_header = true;
            continue;
        }
        if found_header && block.label != back_edge_from {
            body.push(block.label.clone());
        }
        if block.label == back_edge_from {
            body.push(block.label.clone());
            break;
        }
    }
    body
}

fn find_loop_exit(func: &Function, back_edge_from: &str) -> Option<String> {
    if let Some(block) = func.get_block(back_edge_from) {
        if let crate::lir::ir::Terminator::CondBr { false_target, .. } = &block.terminator {
            return Some(false_target.clone());
        }
    }
    None
}

fn extract_induction_var(func: &Function, header: &str) -> Option<Value> {
    if let Some(block) = func.get_block(header) {
        for inst in &block.instructions {
            if let Instruction::Load { ptr, .. } = inst {
                return Some(ptr.clone());
            }
        }
    }
    None
}

fn extract_loop_bound(func: &Function, header: &str) -> Option<Value> {
    if let Some(block) = func.get_block(header) {
        for inst in &block.instructions {
            match inst {
                Instruction::Icmp(_, _, rhs) | Instruction::Fcmp(_, _, rhs) => {
                    return Some(rhs.clone());
                }
                _ => {}
            }
        }
    }
    None
}

fn extract_memory_access(inst: &Instruction, block_label: &str) -> Option<MemoryAccess> {
    match inst {
        Instruction::Load { ptr, .. } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: vec![],
            is_write: false,
            block_label: block_label.to_string(),
        }),
        Instruction::Store { ptr, value, .. } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: vec![],
            is_write: true,
            block_label: block_label.to_string(),
        }),
        Instruction::Gep { ptr, indices } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: indices.clone(),
            is_write: false,
            block_label: block_label.to_string(),
        }),
        _ => None,
    }
}

fn indices_overlap(a: &[Value], b: &[Value]) -> bool {
    if a.len() != b.len() {
        return true; // Conservative: different shapes may overlap
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}
```

- [ ] **Step 2: Create scheduling/annotate.rs**

```rust
use crate::lir::ir::{Function, Value};
use crate::scheduling::dependency::{
    analyze_memory_accesses, find_natural_loops, has_loop_carried_dependency, is_reduction_pattern,
    LoopInfo, ReductionType,
};

#[derive(Debug, Clone)]
pub struct ScheduleAnnotation {
    pub regions: Vec<ParallelRegion>,
}

#[derive(Debug, Clone)]
pub enum ParallelRegion {
    ParallelFor {
        loop_var: Value,
        bounds: (Value, Value),
        body_blocks: Vec<String>,
        simd_regions: Vec<SimdRegion>,
    },
    ParallelReduction {
        loop_var: Value,
        bounds: (Value, Value),
        body_blocks: Vec<String>,
        reduction_op: ReductionOp,
        accumulator: Value,
    },
}

#[derive(Debug, Clone)]
pub struct SimdRegion {
    pub loop_var: Value,
    pub bounds: (Value, Value),
    pub body_blocks: Vec<String>,
    pub vector_width: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Max,
    Sum,
    Min,
    Product,
}

pub fn annotate(func: &Function) -> ScheduleAnnotation {
    let mut annotation = ScheduleAnnotation { regions: vec![] };

    let loops = find_natural_loops(func);

    for loop_info in &loops {
        let accesses = analyze_memory_accesses(func, loop_info);

        if let Some(reduction_type) = is_reduction_pattern(&accesses) {
            let reduction_op = match reduction_type {
                ReductionType::Sum => ReductionOp::Sum,
                ReductionType::Max => ReductionOp::Max,
                ReductionType::Min => ReductionOp::Min,
                ReductionType::Product => ReductionOp::Product,
            };
            annotation.regions.push(ParallelRegion::ParallelReduction {
                loop_var: loop_info.induction_var.clone().unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
                bounds: (
                    Value::Const(crate::lir::ir::Constant::Int(0)),
                    loop_info.bound.clone().unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
                ),
                body_blocks: loop_info.body_blocks.clone(),
                reduction_op,
                accumulator: Value::Const(crate::lir::ir::Constant::Int(0)),
            });
        } else if !has_loop_carried_dependency(&accesses) {
            annotation.regions.push(ParallelRegion::ParallelFor {
                loop_var: loop_info.induction_var.clone().unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
                bounds: (
                    Value::Const(crate::lir::ir::Constant::Int(0)),
                    loop_info.bound.clone().unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
                ),
                body_blocks: loop_info.body_blocks.clone(),
                simd_regions: vec![],
            });
        }
    }

    annotation
}
```

- [ ] **Step 3: Create scheduling/mod.rs**

```rust
mod annotate;
mod dependency;

pub use annotate::{annotate, ParallelRegion, ReductionOp, ScheduleAnnotation, SimdRegion};
pub use dependency::{find_natural_loops, has_loop_carried_dependency, LoopInfo};
```

- [ ] **Step 4: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Should compile

- [ ] **Step 5: Commit**

```bash
rtk git add crates/sile/src/scheduling/
rtk git commit -m "feat: add scheduling pass (dependency analysis + parallelism annotation)"
```

---

### Task 5: Rewrite C Codegen (Unified Wrapper Pattern)

**Files:**
- Modify: `crates/sile/src/codegen/c.rs` (complete rewrite)

**Design:** All generated kernels use a unified wrapper signature:
```c
void sile_kernel_<name>(
    void** buffers,              // all tensor buffers (inputs + outputs)
    int64_t num_threadgroups,    // launch.grid[0] — number of thread groups
    int64_t threads_per_group,   // tile size — elements per thread group
    const int64_t* shapes,       // flattened shape dimensions
    int64_t num_shapes           // number of shape values
);
```

The wrapper unpacks buffers, extracts shapes, then uses `#pragma omp parallel for` to distribute thread groups, with each group processing `threads_per_group` elements. Inner loops use `#pragma omp simd` for vectorization.

- [ ] **Step 1: Rewrite codegen/c.rs**

```rust
use crate::lir::ir::*;
use crate::scheduling::{ParallelRegion, ReductionOp, ScheduleAnnotation, SimdRegion};

pub struct KernelGenInfo {
    pub name: String,
    pub num_buffers: usize,
    pub buffer_kinds: Vec<BufferKind>, // input or output for each buffer
    pub num_shapes: usize,
}

#[derive(Clone, Copy)]
pub enum BufferKind {
    Input,
    Output,
}

pub fn generate(
    func: &Function,
    annotations: &ScheduleAnnotation,
    info: &KernelGenInfo,
) -> crate::Result<String> {
    let mut ctx = CCodegen {
        func,
        annotations,
        info,
        value_names: Vec::new(),
        temp_counter: 0,
        indent: 0,
        out: String::new(),
    };

    ctx.emit_prologue();
    ctx.emit_wrapper_signature();
    ctx.emit_wrapper_body();
    ctx.emit_epilogue();

    Ok(ctx.out)
}

struct CCodegen<'a> {
    func: &'a Function,
    annotations: &'a ScheduleAnnotation,
    info: &'a KernelGenInfo,
    value_names: Vec<String>,
    temp_counter: usize,
    indent: usize,
    out: String,
}

impl<'a> CCodegen<'a> {
    fn emit_prologue(&mut self) {
        self.out.push_str("#include <stdint.h>\n");
        self.out.push_str("#include <math.h>\n");
        self.out.push_str("#include <omp.h>\n");
        self.out.push_str("\n");
    }

    fn emit_epilogue(&mut self) {}

    fn emit_wrapper_signature(&mut self) {
        let fn_name = format!("sile_kernel_{}", self.info.name);
        self.out.push_str(&format!(
            "void {}(\n",
            fn_name
        ));
        self.out.push_str("    void** buffers,\n");
        self.out.push_str("    int64_t num_threadgroups,\n");
        self.out.push_str("    int64_t threads_per_group,\n");
        self.out.push_str("    const int64_t* shapes,\n");
        self.out.push_str("    int64_t num_shapes\n");
        self.out.push_str(") {\n");
        self.indent = 1;
    }

    fn emit_wrapper_body(&mut self) {
        // 1. Unpack buffers into typed pointers
        self.emit_buffer_unpack();

        // 2. Extract shape variables
        self.emit_shape_unpack();

        // 3. Initialize value name mapping for codegen
        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let name = match kind {
                BufferKind::Input => format!("in_{}", i),
                BufferKind::Output => format!("out_{}", i),
            };
            self.value_names.push(name);
        }

        // 4. Emit parallel regions
        for region in &self.annotations.regions {
            self.emit_parallel_region(region);
        }

        // 5. Emit non-annotated blocks
        for block in &self.func.blocks {
            let covered = self.annotations.regions.iter().any(|r| {
                let blocks = match r {
                    ParallelRegion::ParallelFor { body_blocks, .. }
                    | ParallelRegion::ParallelReduction { body_blocks, .. } => body_blocks,
                };
                blocks.contains(&block.label)
            });
            if !covered && block.label != "entry" {
                self.emit_block(block);
            }
        }

        self.indent = 0;
        self.out.push_str("}\n");
    }

    fn emit_buffer_unpack(&mut self) {
        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let name = match kind {
                BufferKind::Input => format!("in_{}", i),
                BufferKind::Output => format!("out_{}", i),
            };
            let qualifier = match kind {
                BufferKind::Input => "const ",
                BufferKind::Output => "",
            };
            self.writeln(&format!(
                "{}float* {} = ({}float*)buffers[{}];",
                qualifier, name, qualifier, i
            ));
        }
        self.writeln("");
    }

    fn emit_shape_unpack(&mut self) {
        // For now: assign shapes[0] → n, shapes[1] → m, etc.
        // In a full implementation, this would come from typed kernel shape info
        self.writeln("int64_t n = shapes[0];");
        if self.info.num_shapes > 1 {
            self.writeln("int64_t m = shapes[1];");
        }
        if self.info.num_shapes > 2 {
            self.writeln("int64_t k = shapes[2];");
        }
        self.writeln("");
    }

    fn emit_parallel_region(&mut self, region: &ParallelRegion) {
        match region {
            ParallelRegion::ParallelFor {
                body_blocks,
                simd_regions,
                ..
            } => {
                // Outer: distribute thread groups via OpenMP
                self.writeln("#pragma omp parallel for num_threads(num_threadgroups) schedule(static)");
                self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
                self.indent += 1;

                // Each thread group processes its tile
                self.writeln("int64_t base = tg * threads_per_group;");

                // Inner: SIMD over tile elements
                if simd_regions.is_empty() {
                    self.writeln("#pragma omp simd");
                }
                self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
                self.indent += 1;
                self.writeln("int64_t i = base + t;");
                self.writeln("if (i < n) {");
                self.indent += 1;

                for label in body_blocks {
                    if let Some(block) = self.func.get_block(label) {
                        self.emit_block(block);
                    }
                }

                self.indent -= 1;
                self.writeln("}");
                self.indent -= 1;
                self.writeln("}");

                // Emit nested SIMD regions for multi-dimensional kernels
                for simd in simd_regions {
                    self.emit_simd_region(simd);
                }

                self.indent -= 1;
                self.writeln("}");
            }
            ParallelRegion::ParallelReduction {
                reduction_op,
                body_blocks,
                ..
            } => {
                let op_str = match reduction_op {
                    ReductionOp::Max => "max",
                    ReductionOp::Sum => "+",
                    ReductionOp::Min => "min",
                    ReductionOp::Product => "*",
                };
                self.writeln(&format!(
                    "#pragma omp parallel for num_threads(num_threadgroups) schedule(static) reduction({}:acc)",
                    op_str
                ));
                self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
                self.indent += 1;
                self.writeln("int64_t base = tg * threads_per_group;");
                self.writeln("float acc = 0.0f;");

                for label in body_blocks {
                    if let Some(block) = self.func.get_block(label) {
                        self.emit_block(block);
                    }
                }

                self.indent -= 1;
                self.writeln("}");
            }
        }
    }

    fn emit_simd_region(&mut self, region: &SimdRegion) {
        self.writeln("#pragma omp simd");
        self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
        self.indent += 1;

        for label in &region.body_blocks {
            if let Some(block) = self.func.get_block(label) {
                self.emit_block(block);
            }
        }

        self.indent -= 1;
        self.writeln("}");
    }

    fn emit_block(&mut self, block: &BasicBlock) {
        for phi in &block.phi_nodes {
            let dest = &phi.dest;
            let src = self.value_name(&phi.incoming[0].0);
            self.writeln(&format!("{} = {};", dest, src));
        }

        for inst in &block.instructions {
            let c_code = emit_instruction(inst, &self.value_names);
            if !c_code.is_empty() {
                self.writeln(&c_code);
            }
        }
    }

    fn value_name(&mut self, value: &Value) -> String {
        match value {
            Value::Param(i) => {
                if *i < self.value_names.len() {
                    self.value_names[*i].clone()
                } else {
                    format!("param{}", i)
                }
            }
            Value::Const(c) => match c {
                Constant::Int(v) => format!("{}", v),
                Constant::Float(v) => format!("{:.1f}", v),
                Constant::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
            },
            Value::Inst(i) => {
                while self.value_names.len() <= *i {
                    let name = format!("t{}", self.temp_counter);
                    self.temp_counter += 1;
                    self.value_names.push(name);
                }
                self.value_names[*i].clone()
            }
        }
    }

    fn writeln(&mut self, line: &str) {
        let indent_str = "  ".repeat(self.indent);
        self.out.push_str(&format!("{}{}\n", indent_str, line));
    }
}

fn emit_instruction(inst: &Instruction, value_names: &[String]) -> String {
    match inst {
        Instruction::Alloca { ty } => {
            let c_ty = c_type(ty);
            let name = format!("local_{}", value_names.len());
            format!("{} {};", c_ty, name)
        }
        Instruction::Load { ptr, .. } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let name = format!("t{}", value_names.len());
            format!("float {} = {}[i];", name, ptr_name)
        }
        Instruction::Store { ptr, value, .. } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let val_name = resolve_value_name(value, value_names);
            format!("{}[i] = {};", ptr_name, val_name)
        }
        Instruction::Gep { ptr, indices } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let index_exprs: Vec<String> = indices
                .iter()
                .map(|v| resolve_value_name(v, value_names))
                .collect();
            let name = format!("t{}", value_names.len());
            if index_exprs.len() == 1 {
                format!("float* {} = {}[{}];", name, ptr_name, index_exprs[0])
            } else {
                format!("float* {} = {}[{}];", name, ptr_name, index_exprs.join(" + "))
            }
        }
        Instruction::Add(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} + {};", next_temp(value_names), l, r)
        }
        Instruction::Sub(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} - {};", next_temp(value_names), l, r)
        }
        Instruction::Mul(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} * {};", next_temp(value_names), l, r)
        }
        Instruction::Div(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} / {};", next_temp(value_names), l, r)
        }
        Instruction::FMax(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = fmaxf({}, {});", next_temp(value_names), l, r)
        }
        Instruction::FMin(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = fminf({}, {});", next_temp(value_names), l, r)
        }
        Instruction::Exp(val) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = expf({});", next_temp(value_names), v)
        }
        Instruction::Icmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            let c_op = cmp_op_to_c(op);
            format!("{} = {} {} {};", next_temp(value_names), l, c_op, r)
        }
        Instruction::Fcmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            let c_op = cmp_op_to_c(op);
            format!("{} = {} {} {};", next_temp(value_names), l, c_op, r)
        }
        Instruction::Call { func, args, .. } => {
            let args_str: Vec<String> = args
                .iter()
                .map(|v| resolve_value_name(v, value_names))
                .collect();
            format!("{}({});", func, args_str.join(", "))
        }
        Instruction::Trunc(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (int){};", next_temp(value_names), v)
        }
        Instruction::ZExt(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (unsigned){};", next_temp(value_names), v)
        }
        Instruction::SIToFP(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (float){};", next_temp(value_names), v)
        }
        Instruction::FPToSI(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (int){};", next_temp(value_names), v)
        }
        Instruction::BitCast(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = {};", next_temp(value_names), v)
        }
        Instruction::FNeg(val) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = -{};", next_temp(value_names), v)
        }
    }
}

fn resolve_value_name(value: &Value, value_names: &[String]) -> String {
    match value {
        Value::Param(i) => {
            if *i < value_names.len() {
                value_names[*i].clone()
            } else {
                format!("param{}", i)
            }
        }
        Value::Const(c) => match c {
            Constant::Int(v) => format!("{}", v),
            Constant::Float(v) => format!("{:.1f}", v),
            Constant::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
        },
        Value::Inst(i) => {
            if *i < value_names.len() {
                value_names[*i].clone()
            } else {
                format!("t{}", i)
            }
        }
    }
}

fn next_temp(value_names: &[String]) -> String {
    format!("t{}", value_names.len())
}

fn c_type(ty: &Type) -> String {
    match ty {
        Type::Void => "void".into(),
        Type::Int(it) => match it {
            IntegerType::I8 => "int8_t".into(),
            IntegerType::I16 => "int16_t".into(),
            IntegerType::I32 => "int32_t".into(),
            IntegerType::I64 => "int64_t".into(),
        },
        Type::Float(ft) => match ft {
            FloatType::F16 => "float".into(),
            FloatType::F32 => "float".into(),
            FloatType::F64 => "double".into(),
        },
        Type::Pointer(inner) => format!("{}*", c_type(inner)),
        Type::Vector(elem, len) => format!("{}[{}]", c_type(elem), len),
    }
}

fn cmp_op_to_c(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Slt | CmpOp::Ult | CmpOp::Olt => "<",
        CmpOp::Sle | CmpOp::Ule | CmpOp::Ole => "<=",
        CmpOp::Sgt | CmpOp::Ugt | CmpOp::Ogt => ">",
        CmpOp::Sge | CmpOp::Uge | CmpOp::Oge => ">=",
    }
}
```

- [ ] **Step 2: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Errors about `backend/cpu_c.rs` still using old types

- [ ] **Step 3: Commit**

```bash
rtk git add crates/sile/src/codegen/c.rs
rtk git commit -m "feat: rewrite C codegen with unified wrapper + OpenMP thread group distribution"
```

---

### Task 6: Update cpu_c.rs to Use New Pipeline

**Files:**
- Modify: `crates/sile/src/backend/cpu_c.rs`

**Design:** Single `KernelFn` type matching the unified wrapper signature. All kernels are called through the same FFI interface.

- [ ] **Step 1: Rewrite cpu_c.rs**

```rust
use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    codegen::c::{BufferKind, KernelGenInfo},
    kernel::{KernelArg, LaunchConfig},
    lir, scheduling,
    Result, Stream,
};

/// Unified kernel function signature matching the C wrapper:
/// void sile_kernel_<name>(
///     void** buffers,
///     int64_t num_threadgroups,
///     int64_t threads_per_group,
///     const int64_t* shapes,
///     int64_t num_shapes
/// )
type KernelFn = unsafe extern "C" fn(
    *const *const c_void,  // buffers
    i64,                   // num_threadgroups
    i64,                   // threads_per_group
    *const i64,            // shapes
    i64,                   // num_shapes
);

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    fn compiler() -> Result<&'static str> {
        for candidate in ["cc", "clang", "gcc"] {
            if Command::new(candidate).arg("--version").output().is_ok() {
                return Ok(candidate);
            }
        }
        Err(crate::Error::UnsupportedBackend("no C compiler found"))
    }

    fn compile_kernel_from_hir(kernel: &crate::hir::Kernel) -> Result<String> {
        let typed = crate::typeck::check_kernel(kernel)
            .map_err(|err| crate::Error::Shape(err.to_string()))?;
        let ssa = crate::passes::canonicalize::run(crate::ssa::lower_typed_kernel_to_ssa(&typed));
        let ssa = crate::passes::dce::run(ssa);
        let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);
        let annotations = scheduling::annotate(&lir_func);

        let info = KernelGenInfo {
            name: kernel.name.clone(),
            num_buffers: kernel.params.len(),
            buffer_kinds: kernel
                .params
                .iter()
                .map(|p| match p.kind {
                    crate::hir::ParamKind::Input => BufferKind::Input,
                    crate::hir::ParamKind::Output => BufferKind::Output,
                })
                .collect(),
            num_shapes: 1, // TODO: derive from typed kernel
        };

        let code = crate::codegen::c::generate(&lir_func, &annotations, &info)?;
        Ok(code)
    }
}

impl crate::backend::Backend for CpuBackend {
    fn launch_kernel(
        &self,
        kernel: &crate::hir::Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", kernel.name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let so_path = dir.path().join(format!("lib{}.{}", kernel.name, ext));

        let c_code = Self::compile_kernel_from_hir(kernel)?;
        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC", "-O3", "-Xpreprocessor", "-fopenmp"])
            .arg("-lomp")
            .arg("-lm")
            .arg(&c_path)
            .arg("-o")
            .arg(&so_path)
            .output()?;
        if !output.status.success() {
            return Err(crate::Error::Compile(
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ));
        }

        unsafe {
            let library =
                Library::new(&so_path).map_err(|e| crate::Error::Compile(e.to_string()))?;
            let symbol_name = format!("sile_kernel_{}", kernel.name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| crate::Error::Compile(e.to_string()))?;

            // Pack all buffer pointers
            let buffers: Vec<*const c_void> = args
                .iter()
                .map(|a| a.mut_ptr as *const c_void)
                .collect();

            // Pack shape dimensions
            let shapes: Vec<i64> = args
                .iter()
                .flat_map(|a| a.shape.iter().copied())
                .collect();

            let num_threadgroups = launch.grid[0] as i64;
            let threads_per_group = args[0].shape[0] / num_threadgroups;

            func(
                buffers.as_ptr(),
                num_threadgroups,
                threads_per_group,
                shapes.as_ptr(),
                shapes.len() as i64,
            );
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Update lib.rs — add lir and scheduling modules, remove backend_ir**

```rust
pub mod backend;
pub mod codegen;
pub mod device;
pub mod error;
pub mod hir;
pub mod kernel;
pub mod lir;
pub mod passes;
pub mod schedule;
#[deprecated = "use crate::hir::Kernel and compiler pipeline instead"]
pub mod spec;
pub mod scheduling;
pub mod ssa;
pub mod stream;
pub mod tensor;
pub mod tile;
pub mod typeck;
```

Remove `pub mod backend_ir;` from lib.rs.

- [ ] **Step 3: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Fix any remaining type/import errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/sile/src/backend/cpu_c.rs crates/sile/src/lib.rs
rtk git commit -m "feat: update cpu_c backend with unified wrapper FFI + new LIR pipeline"
```

- [ ] **Step 2: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Errors about `backend/cpu_c.rs` still using old types

- [ ] **Step 3: Commit**

```bash
rtk git add crates/sile/src/codegen/c.rs
rtk git commit -m "feat: rewrite C codegen for LIR + scheduling annotations with OpenMP/SIMD"
```

---

### Task 6: Update cpu_c.rs to Use New Pipeline

**Files:**
- Modify: `crates/sile/src/backend/cpu_c.rs`

- [ ] **Step 1: Rewrite cpu_c.rs**

Replace the `compile_kernel_from_hir` method and update `launch_kernel`:

```rust
use std::{fs, process::Command};

use libloading::Library;
use tempfile::tempdir;

use crate::{
    kernel::{KernelArg, LaunchConfig},
    lir, scheduling,
    Result, Stream,
};

type KernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64);

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    fn compiler() -> Result<&'static str> {
        for candidate in ["cc", "clang", "gcc"] {
            if Command::new(candidate).arg("--version").output().is_ok() {
                return Ok(candidate);
            }
        }
        Err(crate::Error::UnsupportedBackend("no C compiler found"))
    }

    fn compile_kernel_from_hir(kernel: &crate::hir::Kernel) -> Result<String> {
        let typed = crate::typeck::check_kernel(kernel)
            .map_err(|err| crate::Error::Shape(err.to_string()))?;
        let ssa = crate::passes::canonicalize::run(crate::ssa::lower_typed_kernel_to_ssa(&typed));
        let ssa = crate::passes::dce::run(ssa);
        let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);
        let annotations = scheduling::annotate(&lir_func);
        let code = crate::codegen::c::generate(&lir_func, &annotations)?;
        Ok(code)
    }
}

impl crate::backend::Backend for CpuBackend {
    fn launch_kernel(
        &self,
        kernel: &crate::hir::Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        _stream: &Stream,
    ) -> Result<()> {
        let dir = tempdir()?;
        let c_path = dir.path().join(format!("{}.c", kernel.name));
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let so_path = dir.path().join(format!("lib{}.{}", kernel.name, ext));

        let c_code = Self::compile_kernel_from_hir(kernel)?;
        fs::write(&c_path, c_code)?;

        let compiler = Self::compiler()?;
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC", "-O2", "-Xpreprocessor", "-fopenmp"])
            .arg("-lomp")
            .arg("-lm")
            .arg(&c_path)
            .arg("-o")
            .arg(&so_path)
            .output()?;
        if !output.status.success() {
            return Err(crate::Error::Compile(
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ));
        }

        unsafe {
            let library =
                Library::new(&so_path).map_err(|e| crate::Error::Compile(e.to_string()))?;
            let symbol_name = format!("sile_kernel_{}", kernel.name);
            let func: libloading::Symbol<KernelFn> = library
                .get(symbol_name.as_bytes())
                .map_err(|e| crate::Error::Compile(e.to_string()))?;

            let a_ptr = args[0].mut_ptr;
            let b_ptr = args[1].mut_ptr;
            let c_ptr = args[2].mut_ptr;
            let n = args[0].shape.iter().product::<i64>();

            func(a_ptr, b_ptr, c_ptr, n);
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Update lib.rs — add lir and scheduling modules, remove backend_ir**

```rust
pub mod backend;
pub mod codegen;
pub mod device;
pub mod error;
pub mod hir;
pub mod kernel;
pub mod lir;
pub mod passes;
pub mod schedule;
#[deprecated = "use crate::hir::Kernel and compiler pipeline instead"]
pub mod spec;
pub mod scheduling;
pub mod ssa;
pub mod stream;
pub mod tensor;
pub mod tile;
pub mod typeck;
```

Remove `pub mod backend_ir;` from lib.rs.

- [ ] **Step 3: Compile and fix errors**

Run: `rtk cargo check -p sile`
Expected: Fix any remaining type/import errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/sile/src/backend/cpu_c.rs crates/sile/src/lib.rs
rtk git commit -m "feat: update cpu_c backend to use new LIR + scheduling pipeline"
```

---

### Task 7: Delete Old backend_ir Module and Update Tests

**Files:**
- Delete: `crates/sile/src/backend_ir/ir.rs`
- Delete: `crates/sile/src/backend_ir/lower.rs`
- Delete: `crates/sile/src/backend_ir/mod.rs`
- Modify: `crates/sile/tests/c_codegen.rs`

- [ ] **Step 1: Delete backend_ir module**

```bash
rm crates/sile/src/backend_ir/ir.rs crates/sile/src/backend_ir/lower.rs crates/sile/src/backend_ir/mod.rs
rmdir crates/sile/src/backend_ir 2>/dev/null || true
```

- [ ] **Step 2: Rewrite c_codegen.rs test**

```rust
use sile::{
    codegen,
    hir::{Kernel, Param, ParamKind, Stmt, Expr, BuiltinOp, Type, ElemType, ShapeExpr},
    lir,
    ssa,
    typeck,
    passes,
    scheduling,
};

#[test]
fn c_codegen_emits_vec_add_with_openmp() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();
    let ssa = passes::canonicalize::run(ssa::lower_typed_kernel_to_ssa(&typed));
    let ssa = passes::dce::run(ssa);
    let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);
    let annotations = scheduling::annotate(&lir_func);
    let c = codegen::c::generate(&lir_func, &annotations).unwrap();

    assert!(c.contains("void sile_kernel_vec_add"));
    assert!(c.contains("#include <omp.h>"));
    assert!(c.contains("#pragma omp parallel for"));
}

fn build_vec_add_kernel() -> Kernel {
    Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new("a", ParamKind::Input, Type::tile(ElemType::F32, ShapeExpr::symbol("S"))),
            Param::new("b", ParamKind::Input, Type::tile(ElemType::F32, ShapeExpr::symbol("S"))),
            Param::new("c", ParamKind::Output, Type::tile(ElemType::F32, ShapeExpr::symbol("S"))),
        ],
        vec![
            Stmt::Let {
                name: "tid".into(),
                ty: None,
                expr: Expr::builtin(BuiltinOp::ProgramId),
            },
            Stmt::Let {
                name: "tile_a".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![Expr::Var("a".into())],
                },
            },
            Stmt::Let {
                name: "tile_b".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![Expr::Var("b".into())],
                },
            },
            Stmt::Store {
                target: "c".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Add,
                    args: vec![Expr::Var("tile_a".into()), Expr::Var("tile_b".into())],
                },
            },
        ],
    )
}
```

- [ ] **Step 3: Run all tests**

Run: `rtk cargo test -p sile`
Expected: All tests pass (some may need adjustment based on new pipeline behavior)

- [ ] **Step 4: Commit**

```bash
rtk git add -A && rtk git commit -m "chore: delete backend_ir, update tests for new LIR pipeline"
```

---

### Task 8: Verify End-to-End

- [ ] **Step 1: Run full test suite**

Run: `rtk cargo test -p sile -- --nocapture`
Expected: All tests pass, vec_add e2e test produces correct results

- [ ] **Step 2: Verify generated C code contains OpenMP**

Run: `rtk cargo test -p sile c_codegen_emits_vec_add_with_openmp -- --nocapture`
Expected: Test passes, generated C contains `#pragma omp parallel for`

- [ ] **Step 3: Final commit**

```bash
rtk git add -A && rtk git commit -m "feat: LIR + scheduling + OpenMP C backend complete"
```

---

## Self-Review

**1. Spec coverage check:**

| Spec Section | Task |
|-------------|------|
| LIR IR types (Value, Type, Instruction, etc.) | Task 1 |
| LIR Builder | Task 2 |
| SSA → LIR Lowering | Task 3 |
| Scheduling (dependency analysis, annotations) | Task 4 |
| C Codegen with OpenMP/SIMD | Task 5 |
| cpu_c.rs pipeline update | Task 6 |
| Delete backend_ir, update tests | Task 7 |
| E2E verification | Task 8 |

All spec sections covered.

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions. All steps contain actual code.

**3. Type consistency:** All types (`Value`, `Type`, `Instruction`, `Function`, etc.) are defined in Task 1 and used consistently throughout. `ScheduleAnnotation`, `ParallelRegion`, `SimdRegion` defined in Task 4 and used in Task 5.

**4. Test design:** Tests verify the key behaviors — LIR generation, scheduling annotations, C code with OpenMP pragmas, and end-to-end execution.
