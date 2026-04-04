# LLIR Redesign For CPU And Metal

## Goal

Replace the current mixed-purpose `LIR` with a two-layer execution pipeline:

`KernelIR -> LLIR -> Backend`

The new `LLIR` should be low-level, SSA-based, explicit about control flow and memory, and close in spirit to LLVM IR. It must be shared by both CPU and Metal backends without embedding target-specific tile semantics into the IR itself.

## Problem

The current `LIR` is not low enough.

It still contains domain-specific operations such as:

- `TileAlloc`
- `TileLoad2D`
- `TileMma`
- `TileReduceMax`
- `TileReduceSum`
- `TileBroadcast`

These operations are already partially shaped around backend code generation, but they are still too semantic to map naturally to different hardware models.

This causes two classes of problems:

1. CPU and Metal need different control-flow shapes.
CPU C codegen can tolerate label-style CFG lowering. Metal cannot use `goto` or labels and needs structured control flow.

2. Memory spaces are implicit.
Metal must distinguish `device`, `threadgroup`, and `thread` memory. CPU mostly maps everything to host memory or stack allocations. The current IR does not model this explicitly.

As a result, the compiler is forced to solve backend-specific problems too late, after high-level semantics and low-level execution details have already been mixed together.

## Target Architecture

### Layer 1: KernelIR

`KernelIR` is the last high-level kernel representation.

Responsibilities:

- represent tile-level intent
- preserve algorithm structure
- carry shape and layout knowledge
- express semantic ops like tile load, tile store, mma, reduction, broadcast

Examples of KernelIR ops:

- `tile.load`
- `tile.store`
- `tile.splat`
- `tile.mma`
- `tile.reduce`
- `tile.broadcast`
- `shape.dim`
- `program.id`

This layer is where matmul/softmax logic remains readable.

### Layer 2: LLIR

`LLIR` is the execution IR.

Responsibilities:

- explicit SSA values
- explicit basic blocks
- explicit block params / phi semantics
- explicit memory operations
- explicit address spaces
- explicit loops and branches
- target-independent intrinsics

This layer must not contain domain-specific tile ops.

`tile.mma` should already have been lowered into one of:

- loops over scalar/vector operations
- backend-independent matrix intrinsics
- backend-selected intrinsic placeholders

## LLIR Design

### Values

```text
ValueId
BlockId
```

All values are SSA.

Block arguments replace ad-hoc loop-carried value hacks.

### Types

Core types:

- `void`
- `i1`
- `i32`
- `i64`
- `f16`
- `f32`
- `f64`
- `ptr<addrspace, elem>`
- `vec<N, T>`
- `array<N, T>`
- optional small aggregates for backend ABI

Do not encode tensor semantics in the type system at this layer.

### Address Spaces

At minimum:

- `generic`
- `global`
- `constant`
- `shared`
- `private`

Mapping:

- CPU C/OpenMP:
  - `global` -> buffer pointers
  - `shared` -> stack or temporary local arrays inside worker scope
  - `private` -> scalar locals
- Metal:
  - `global` -> `device`
  - `constant` -> `constant`
  - `shared` -> `threadgroup`
  - `private` -> `thread`

### Instructions

#### Control Flow

- `br target(args...)`
- `condbr cond, true(args...), false(args...)`
- `switch value, default, cases`
- `ret`

#### Memory

- `alloca type, addrspace`
- `gep base, indices`
- `load ptr`
- `store ptr, value`
- `memcpy dst, src, size`

#### Arithmetic

- integer binary ops
- float binary ops
- compare ops
- cast ops
- select

#### Vector Ops

- `extractelement`
- `insertelement`
- `shufflevector`

#### Calls And Intrinsics

- `call fn(args...)`
- `intrinsic.thread_id(dim)`
- `intrinsic.block_id(dim)`
- `intrinsic.barrier(scope)`
- `intrinsic.matmul_fragment(...)`
- `intrinsic.reduce_add(...)`
- `intrinsic.reduce_max(...)`

These intrinsics are target-independent names. Each backend decides whether to:

- lower them to scalar loops
- map them to vector operations
- map them to target-native APIs

### Metadata

Optimization and scheduling hints must be metadata, not core IR semantics.

Examples:

- loop parallel hint
- reduction hint
- vectorize width
- unroll factor
- alignment
- noalias
- readonly / writeonly

This is how CPU gets OpenMP lowering and Metal gets SIMDGROUP / threadgroup decisions without hardcoding those choices into LLIR itself.

## Lowering Responsibilities

### HIR -> MIR

No major change in responsibility.

This stage still handles:

- frontend syntax
- typing
- shape reasoning
- loop-carried variables

### MIR -> KernelIR

This stage makes tile semantics explicit and regular.

Examples:

- normalize tile loads/stores
- normalize broadcast / reshape
- canonicalize reductions
- canonicalize mma patterns

### KernelIR -> LLIR

This is the key new lowering boundary.

Responsibilities:

1. convert tile-shaped computation into explicit memory + loop structure
2. materialize address spaces
3. materialize block params and control flow
4. choose whether an operation becomes:
   - scalar loop
   - vector sequence
   - target-independent intrinsic

Examples:

#### Dynamic-K matmul

Current form:

```text
acc = tile.constant(0)
for k_idx in 0 .. shape(a, 1) / BK:
  a_tile = tile.load(a, [BM, BK], [m_idx, k_idx])
  b_tile = tile.load(b, [BK, BN], [k_idx, n_idx])
  acc = tile.mma(a_tile, b_tile, acc)
tile.store(c, acc)
```

LLIR shape:

```text
bb.entry:
  %m_idx = intrinsic.block_id(0)
  %n_idx = intrinsic.block_id(1)
  %acc0 = alloca array[BM][BN], private
  init %acc0 with zero
  br bb.loop_header(%k0=0, %acc_phi=%acc0)

bb.loop_header(%k, %acc):
  %k_end = load shape dim
  %cond = icmp %k < %k_end
  condbr %cond, bb.loop_body(%k, %acc), bb.exit(%acc)

bb.loop_body(%k, %acc):
  %a_tile = ...
  %b_tile = ...
  %acc_next = call/intrinsic/or loop-expanded mma(%a_tile, %b_tile, %acc)
  %k_next = add %k, 1
  br bb.loop_header(%k_next, %acc_next)

bb.exit(%acc_final):
  store output tile
  ret
```

This shape is shared.
Only the backend-specific structurization or codegen is different.

## Backend Contracts

### CPU Backend

Input:

- `LLIR`
- metadata

Responsibilities:

1. structurize or directly emit C CFG
2. lower `shared/private/global` into valid C storage
3. map loop metadata into:
   - `#pragma omp parallel for`
   - `#pragma omp simd`
   - reduction clauses
4. lower vector operations either to:
   - plain scalar C
   - compiler-vectorizable loops
   - explicit vector intrinsics later

Important point:
OpenMP is not part of LLIR semantics. It is a backend optimization choice driven by metadata.

### Metal Backend

Input:

- `LLIR`
- metadata

Responsibilities:

1. structurize CFG into legal MSL control flow
2. map address spaces correctly
3. lower intrinsics into:
   - scalar loops
   - SIMD-group matrix intrinsics
   - threadgroup barriers
4. generate valid MSL storage declarations for shared temporaries

Important point:
Metal cannot accept arbitrary CFG syntax. Therefore the backend must run a dedicated structurization pass before printing MSL.

## Structurization Pass

This pass belongs between LLIR and text codegen.

Purpose:

- convert reducible CFG into structured `if`, `while`, `for`
- preserve SSA / phi semantics by introducing structured assignments

This pass is mandatory for Metal.
It is optional for CPU if direct CFG-to-C remains acceptable.

## Migration Plan

### Phase 1

Add `KernelIR` and `LLIR` side-by-side with existing `LIR`.

Do not delete current code yet.

### Phase 2

Make `dynamic-K matmul` the first full migration path:

- HIR
- MIR
- KernelIR
- LLIR
- CPU backend
- Metal backend

This should become the reference path.

### Phase 3

Migrate:

- `vec_add`
- `softmax`
- reduction kernels

### Phase 4

Switch both backends to consume only `LLIR`.

### Phase 5

Delete high-level tile opcodes from old `LIR`.

## Acceptance Criteria

The redesign is complete when:

1. no backend consumes tile-specific high-level execution ops directly
2. CPU and Metal both consume the same `LLIR`
3. dynamic-K matmul lowers without target-specific hacks in the shared IR
4. OpenMP decisions come from metadata, not IR opcodes
5. Metal control flow is produced by a structurization pass, not ad-hoc codegen hacks
6. tests exist for:
   - LLIR text snapshots
   - CPU generated C
   - Metal generated MSL
   - dynamic-K matmul correctness on CPU
   - dynamic-K matmul codegen and runtime on Metal when device is available

## Immediate Next Step

The next implementation task should be:

`Write the concrete LLIR data structures and textual printer, then migrate dynamic-K matmul lowering to LLIR as the first vertical slice.`
