# LIR + Scheduling + C/OpenMP Backend Design

## Summary

Replace the current hardcoded `backend_ir` + `codegen/c.rs` with a general-purpose LLVM-style LIR (Low-level IR), an independent scheduling pass for parallelism analysis, and a C codegen that produces complete compilable C files with OpenMP thread-level parallelism and SIMD vectorization.

## Pipeline

```
Macro → HIR → TypeCheck → SSA → Passes → LIR → Scheduling → C Codegen
```

The old `backend_ir::lower` and `codegen::c` are deleted. The new LIR is backend-agnostic; the scheduling pass and C codegen are specific to the CPU target.

## LIR IR Design

### Value Reference

```rust
enum Value {
    Param(usize),
    Const(Constant),
    Inst(usize),
}

enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}
```

### Type System

```rust
enum IntegerType { I8, I16, I32, I64 }
enum FloatType { F16, F32, F64 }

enum Type {
    Void,
    Int(IntegerType),
    Float(FloatType),
    Pointer(Box<Type>),
    Vector(Box<Type>, usize),
}

struct Param {
    name: String,
    ty: Type,
}

struct GlobalVariable {
    name: String,
    ty: Type,
    initializer: Option<Constant>,
}
```

### Instructions (pure semantics, no parallelism info)

```rust
enum Instruction {
    // Stack
    Alloca { ty: Type },

    // Memory
    Load { ptr: Value, ty: Type, align: Option<u32> },
    Store { ptr: Value, value: Value, align: Option<u32> },
    Gep { ptr: Value, indices: Vec<Value> },

    // Arithmetic
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    FNeg(Value),
    FMax(Value, Value),
    FMin(Value, Value),
    Exp(Value),

    // Comparison
    Icmp(CmpOp, Value, Value),
    Fcmp(CmpOp, Value, Value),

    // Type conversion
    Trunc(Value, Type),
    ZExt(Value, Type),
    SIToFP(Value, Type),
    FPToSI(Value, Type),
    BitCast(Value, Type),

    // Call
    Call { func: String, args: Vec<Value>, ret_ty: Type },
}
```

### Control Flow

```rust
struct BasicBlock {
    label: String,
    phi_nodes: Vec<PhiNode>,
    instructions: Vec<Instruction>,
    terminator: Terminator,
}

enum Terminator {
    Br { target: String },
    CondBr { cond: Value, true_target: String, false_target: String },
    Switch { value: Value, default: String, cases: Vec<(i64, String)> },
    Ret(Option<Value>),
}

struct PhiNode {
    dest: String,
    ty: Type,
    incoming: Vec<(Value, String)>,
}
```

### Function & Program

```rust
struct Function {
    name: String,
    params: Vec<Param>,
    return_type: Type,
    blocks: IndexMap<String, BasicBlock>,
    entry_block: String,
}

struct Program {
    functions: Vec<Function>,
    globals: Vec<GlobalVariable>,
}
```

## SSA → LIR Lowering

### Strategy

1. Create function signature from typed kernel params
2. Analyze implicit loop structure from SSA `ProgramId` and tile shapes
3. Generate explicit loop nesting with header/body/exit blocks
4. Lower SSA instructions to LIR within loop scopes
5. Insert phi nodes at block merge points

### Key Mappings

| SSA Opcode | LIR |
|-----------|-----|
| `ProgramId` | Loop variable + bound check |
| `LoadTile` / `LoadTileLike2D` | `Gep` → `Load` |
| `Add/Sub/Mul/Div/Exp` | Corresponding LIR arithmetic or `Call` |
| `ReduceMax/Sum` | Expanded to loop with accumulator |
| `Store` | `Gep` → `Store` |
| `Mma` | Nested loops or runtime call |
| `Constant` | Inline `Const` |

### Reduce Lowering

Reduce operations are expanded into explicit loops:

```
%acc = alloca f32
store %acc, -inf
br %reduce_header

%reduce_header:
  %i = load %var
  %cond = icmp slt %i, %bound
  cond_br %cond, %reduce_body, %reduce_exit

%reduce_body:
  %val = load %src[%i]
  %acc_val = load %acc
  %new = fmax %acc_val, %val
  store %acc, %new
  store %var, add %i, 1
  br %reduce_header

%reduce_exit:
  %result = load %acc
```

## Scheduling Pass

### Annotations

```rust
struct ScheduleAnnotation {
    regions: Vec<ParallelRegion>,
}

enum ParallelRegion {
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

struct SimdRegion {
    loop_var: Value,
    bounds: (Value, Value),
    body_blocks: Vec<String>,
    vector_width: Option<usize>,
}

enum ReductionOp { Max, Sum, Min, Product }
```

### Dependency Analysis

Detect loop-carried dependencies by analyzing memory access patterns:

- **Parallelizable**: Loop body only reads/writes `base[i]` where `i` is the loop variable, no cross-iteration access
- **Not parallelizable**: Write to `base[i]` and read from `base[i+1]` (or any cross-iteration dependency)
- **Parallel reduction**: All iterations accumulate into a single variable with an associative operator

### Algorithm

1. Find natural loops via back-edge detection in CFG
2. For each loop, collect all memory accesses
3. Check for loop-carried dependencies
4. If no dependencies → mark as `ParallelFor`
5. If reduction pattern → mark as `ParallelReduction`
6. Within parallel loops, check inner loops for SIMD opportunities

## C Codegen

### Output Structure

Every generated C file includes:
- Required headers (`<stdint.h>`, `<math.h>`, `<omp.h>`)
- Single kernel function with `restrict` pointers
- OpenMP pragmas for parallelism and vectorization

### Parallel Region Translation

```
ParallelFor → #pragma omp parallel for schedule(static)
SimdRegion  → #pragma omp simd
ParallelReduction → #pragma omp parallel for reduction(op:acc)
```

### Instruction Translation

| LIR | C |
|-----|---|
| `Load` | `ptr[i]` or `*(type*)ptr` |
| `Store` | `ptr[i] = val` |
| `Gep` | `ptr + idx0*stride0 + idx1*stride1` (row-major, strides derived from shape params) |
| `Add/Sub/Mul/Div` | `a + b` etc. |
| `FMax/FMin` | `fmaxf(a, b)` / `fminf(a, b)` |
| `Exp` | `expf(v)` |
| `Icmp/Fcmp` | `a < b` etc. in conditionals |
| `Call` | `func(args)` |

### Example: VecAdd

```c
#include <stdint.h>
#include <omp.h>

void sile_kernel_vec_add(
    void** buffers,
    int64_t num_threadgroups,
    int64_t threads_per_group,
    const int64_t* shapes,
    int64_t num_shapes
) {
    const float* in_0 = (const float*)buffers[0];
    const float* in_1 = (const float*)buffers[1];
    float* out_0 = (float*)buffers[2];

    int64_t n = shapes[0];

    #pragma omp parallel for num_threads(num_threadgroups) schedule(static)
    for (int64_t tg = 0; tg < num_threadgroups; ++tg) {
        int64_t base = tg * threads_per_group;
        #pragma omp simd
        for (int64_t t = 0; t < threads_per_group; ++t) {
            int64_t i = base + t;
            if (i < n) {
                out_0[i] = in_0[i] + in_1[i];
            }
        }
    }
}
```

### Example: Softmax

```c
#include <stdint.h>
#include <math.h>
#include <omp.h>

void sile_kernel_softmax(
    void** buffers,
    int64_t num_threadgroups,
    int64_t threads_per_group,
    const int64_t* shapes,
    int64_t num_shapes
) {
    const float* in_0 = (const float*)buffers[0];
    float* out_0 = (float*)buffers[1];

    int64_t m = shapes[0];
    int64_t n = shapes[1];

    #pragma omp parallel for num_threads(num_threadgroups) schedule(static)
    for (int64_t tg = 0; tg < num_threadgroups; ++tg) {
        int64_t row_base = tg * threads_per_group;
        for (int64_t row = 0; row < threads_per_group; ++row) {
            int64_t r = row_base + row;
            if (r >= m) break;

            // Phase 1: find max (reduction, SIMD-safe)
            float max_val = in_0[r * n];
            #pragma omp simd reduction(max:max_val)
            for (int64_t col = 1; col < n; ++col) {
                max_val = fmaxf(max_val, in_0[r * n + col]);
            }

            // Phase 2: exp and sum
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int64_t col = 0; col < n; ++col) {
                float e = expf(in_0[r * n + col] - max_val);
                out_0[r * n + col] = e;
                sum += e;
            }

            // Phase 3: normalize
            #pragma omp simd
            for (int64_t col = 0; col < n; ++col) {
                out_0[r * n + col] /= sum;
            }
        }
    }
}
```

### Example: MatMul

```c
#include <stdint.h>
#include <omp.h>

void sile_kernel_matmul(
    void** buffers,
    int64_t num_threadgroups,
    int64_t threads_per_group,
    const int64_t* shapes,
    int64_t num_shapes
) {
    const float* in_0 = (const float*)buffers[0];
    const float* in_1 = (const float*)buffers[1];
    float* out_0 = (float*)buffers[2];

    int64_t m = shapes[0];
    int64_t n = shapes[1];
    int64_t k = shapes[2];

    #pragma omp parallel for num_threads(num_threadgroups) schedule(static) collapse(2)
    for (int64_t tg_m = 0; tg_m < num_threadgroups; ++tg_m) {
        for (int64_t tg_n = 0; tg_n < num_threadgroups; ++tg_n) {
            int64_t row_base = tg_m * threads_per_group;
            int64_t col_base = tg_n * threads_per_group;
            for (int64_t i = 0; i < threads_per_group; ++i) {
                for (int64_t j = 0; j < threads_per_group; ++j) {
                    int64_t row = row_base + i;
                    int64_t col = col_base + j;
                    if (row >= m || col >= n) continue;
                    float acc = 0.0f;
                    #pragma omp simd reduction(+:acc)
                    for (int64_t l = 0; l < k; ++l) {
                        acc += in_0[row * k + l] * in_1[l * n + col];
                    }
                    out_0[row * n + col] = acc;
                }
            }
        }
    }
}
```

## Module Structure

```
crates/sile/src/
  lir/
    mod.rs          # Module exports
    ir.rs           # Value, Type, Instruction, Terminator, PhiNode, BasicBlock, Function, Program
    builder.rs      # LIR builder utility
  scheduling/
    mod.rs          # Pass entry point
    dependency.rs   # Memory access analysis, loop-carried dependency detection
    annotate.rs     # Build ScheduleAnnotation from analyzed LIR
  codegen/
    c.rs            # LIR + ScheduleAnnotation → C source
  backend/
    cpu_c.rs        # Updated: uses new pipeline (lir → scheduling → c codegen)
```

## Migration Plan

1. Add `lir/` module with IR types
2. Add `lir/builder.rs`
3. Rewrite `ssa/lower.rs` → `lir/lower.rs` (SSA → LIR)
4. Add `scheduling/` module
5. Rewrite `codegen/c.rs` (LIR → C)
6. Update `backend/cpu_c.rs` to use new pipeline
7. Delete old `backend_ir/` module
8. Update all tests to use new pipeline

## Future: Metal/wgpu Backends

The LIR is backend-agnostic. Adding Metal or wgpu requires:
- A target-specific scheduling pass (e.g., mapping parallel_for to threadgroup/thread dispatch)
- A target-specific codegen (LIR + annotations → Metal shader source or WGSL)

No changes to LIR itself are needed.
