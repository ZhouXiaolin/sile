# MIR To LLIR Redesign For CPU And Metal

## Goal

Replace the current mixed-purpose `LIR` with a three-layer compiler pipeline:

`HIR -> MIR -> LLIR -> Backend`

There is no separate `KernelIR`.

`MIR` is the canonical kernel-semantics IR.
`LLIR` is the execution IR.

The hard boundary is:

- `MIR` answers: what does the kernel compute
- `LLIR` answers: how will the machine execute it

Both CPU and Metal must consume the same `LLIR`.

## Problem

The current execution path is too mixed.

It still blends together:

- tile-level algorithm semantics
- loop and memory execution details
- backend legality workarounds
- target capability choices

This causes three concrete problems:

1. the shared IR is not low enough
2. backend-specific concerns leak too early
3. CPU and Metal are forced to rediscover meaning that should already have been lowered

The redesign goal is not "more layers." The goal is a clean semantic-to-execution boundary.

## Three-Layer Architecture

### HIR

`HIR` is the frontend kernel representation.

Responsibilities:

- preserve user-written kernel structure
- carry frontend typing and shape syntax
- expose source-level tile operations
- remain readable and close to the user model

`HIR` may contain:

- source-level loops and assignments
- symbolic and dynamic shape expressions
- semantic tile operations
- frontend convenience constructs

`HIR` must not contain:

- SSA values
- explicit basic blocks
- explicit address spaces
- backend intrinsics
- backend legality workarounds

### MIR

`MIR` is the canonical kernel-semantics IR.

Responsibilities:

- remove syntax sugar
- normalize tile operations into a stable semantic form
- make loop-carried values and kernel dataflow explicit enough for lowering
- preserve algorithm semantics without committing to a specific execution strategy

`MIR` may contain:

- `ProgramId`
- `ShapeDim`
- `TileConstant`
- `TileLoad`
- `TileStore`
- `TileBroadcast`
- `TileReduce`
- `TileMma`
- structured loops and branches
- shape and tile metadata needed to lower semantics correctly

`MIR` must not contain:

- `alloca`
- `gep`
- explicit pointer arithmetic
- explicit address spaces
- block params / phi-style execution IR constructs
- backend-specific matrix intrinsics
- C- or Metal-specific code generation constraints

In other words:

- `MIR` is where kernel semantics live
- `MIR` is the old planned `KernelIR`, just without introducing another abstraction layer

### LLIR

`LLIR` is the execution IR.

Responsibilities:

- explicit SSA values
- explicit basic blocks
- explicit block params / phi semantics
- explicit memory operations
- explicit address spaces
- explicit loops and branches
- low-level target-independent intrinsics
- metadata-driven optimization hints

`LLIR` may contain:

- `br`
- `condbr`
- `switch`
- `ret`
- `alloca`
- `gep`
- `load`
- `store`
- `memcpy`
- scalar and vector arithmetic
- compare, cast, select
- `call`
- low-level intrinsics such as thread id, block id, barrier, and optional matrix-fragment intrinsics

`LLIR` must not contain:

- `tile.load`
- `tile.store`
- `tile.broadcast`
- `tile.reduce`
- semantic `tile.mma`
- shape-language constructs
- backend-specific execution hacks

In other words:

- `LLIR` is where execution lives
- if an operation is still describing algorithm meaning, it is too high-level for `LLIR`

## Formal Boundary Rules

### HIR -> MIR

This boundary is responsible for semantic normalization.

Required outcomes:

- source-level syntax sugar is removed
- tile operations are put into a canonical form
- loop structure is normalized
- shape access becomes explicit and regular

Forbidden outcomes:

- introducing explicit address spaces
- choosing hardware intrinsics
- introducing backend legality constraints
- lowering semantic tile ops into execution memory ops

### MIR -> LLIR

This is the semantic-to-execution boundary.

Required outcomes:

- semantic tile operations are lowered away
- explicit storage and memory traffic are introduced
- loop-carried values become block params / phi-style dataflow
- control flow becomes explicit CFG
- address spaces are materialized
- backend-independent low-level intrinsic choices are made

Allowed decisions at this boundary:

- lower a semantic op into scalar loops
- lower a semantic op into vector code
- lower a semantic op into a low-level intrinsic placeholder

Forbidden outcomes:

- keeping semantic tile ops in `LLIR`
- introducing Metal-only or CPU-only IR opcodes
- hiding memory movement inside semantic helpers

### LLIR -> Backend

This boundary is responsible only for target realization.

CPU backend responsibilities:

- print legal C/OpenMP
- use metadata for parallel/vectorization decisions
- map address spaces to host representations
- realize low-level intrinsics as loops or target facilities

Metal backend responsibilities:

- structurize reducible CFG into legal MSL control flow
- map address spaces to `device` / `threadgroup` / `thread`
- realize low-level intrinsics using scalar code or Metal capabilities
- emit legal MSL declarations and barriers

Forbidden backend work:

- reconstructing high-level kernel semantics
- guessing whether an op is semantic or low-level
- compensating for missing MIR -> LLIR lowering decisions

## MMA Rule

`mma` must be split into two different concepts.

### Semantic MMA

`TileMma` in `MIR` means:

- perform tile-level matrix multiply-accumulate semantics
- preserve algorithm intent
- make no promise about hardware support

This is a semantic op and belongs in `MIR`.

### Execution MMA

Low-level matrix intrinsics in `LLIR` mean:

- use a low-level matrix fragment or hardware-like operation
- represent an execution choice, not an algorithm description
- be valid to lower either to native hardware support or to explicit loops

This belongs in `LLIR` only if it is represented as a low-level intrinsic such as:

- `intrinsic.matmul_fragment`

### Lowering Rule For MMA

`MIR::TileMma` must lower into exactly one of:

- explicit scalar loops in `LLIR`
- vectorized loop structure in `LLIR`
- low-level `LLIR` matrix intrinsic

`MIR::TileMma` must never survive unchanged into `LLIR`.

The choice is made during `MIR -> LLIR`, based on lowering policy and target capability.

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

These intrinsics are target-independent low-level names.
Each backend decides whether to:

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

This is how CPU gets OpenMP lowering and Metal gets SIMD-group / threadgroup decisions without hardcoding those choices into LLIR itself.

## Lowering Responsibilities

### HIR -> MIR

This stage handles:

- frontend syntax cleanup
- typing-driven normalization
- shape reasoning
- loop-carried semantic normalization

### MIR -> LLIR

This is the key lowering boundary.

Responsibilities:

1. convert tile-shaped computation into explicit memory + loop structure
2. materialize address spaces
3. materialize block params and control flow
4. choose whether a semantic operation becomes:
   - scalar loop
   - vector sequence
   - target-independent intrinsic

Example:

#### Dynamic-K matmul

Semantic `MIR` form:

```text
acc = tile.constant(0)
for k_idx in 0 .. shape(a, 1) / BK:
  a_tile = tile.load(a, [BM, BK], [m_idx, k_idx])
  b_tile = tile.load(b, [BK, BN], [k_idx, n_idx])
  acc = tile.mma(a_tile, b_tile, acc)
tile.store(c, acc)
```

Lowered `LLIR` shape:

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
  %acc_next = loops or intrinsic(%a_tile, %b_tile, %acc)
  %k_next = add %k, 1
  br bb.loop_header(%k_next, %acc_next)

bb.exit(%acc_final):
  store output tile
  ret
```

This shape is shared.
Only the backend-specific structurization or text codegen is different.

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
OpenMP is not part of LLIR semantics.
It is a backend optimization choice driven by metadata.

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
Metal cannot accept arbitrary CFG syntax.
Therefore the backend must run a dedicated structurization pass before printing MSL.

## Structurization Pass

This pass belongs between `LLIR` and text codegen.

Purpose:

- convert reducible CFG into structured `if`, `while`, `for`
- preserve SSA / phi semantics by introducing structured assignments

This pass is mandatory for Metal.
It is optional for CPU if direct CFG-to-C remains acceptable.

## Migration Plan

### Phase 1

Keep existing `HIR -> MIR -> LIR` working while introducing the new `LLIR` path.

Do not delete current code yet.

### Phase 2

Redefine `MIR` as the canonical kernel-semantics IR.

Tasks:

- document what `MIR` may contain
- prohibit execution-only constructs from leaking into `MIR`
- document what `LLIR` may contain
- prohibit semantic tile ops from surviving into `LLIR`

### Phase 3

Make `dynamic-K matmul` the first full reference migration path:

- HIR
- MIR
- LLIR
- CPU backend
- Metal backend

### Phase 4

Migrate:

- `vec_add`
- `softmax`
- reduction kernels

### Phase 5

Switch both backends to consume only `LLIR`.

### Phase 6

Delete high-level tile opcodes from old `LIR`.

## Acceptance Criteria

The redesign is complete when:

1. no backend consumes tile-specific high-level execution ops directly
2. `MIR` is the only semantic kernel IR between HIR and LLIR
3. CPU and Metal both consume the same `LLIR`
4. dynamic-K matmul lowers without target-specific hacks in the shared IR
5. OpenMP decisions come from metadata, not IR opcodes
6. Metal control flow is produced by a structurization pass, not ad-hoc codegen hacks
7. `mma` exists as a semantic op only in `MIR`, and as a low-level intrinsic only in `LLIR`
8. tests exist for:
   - LLIR text snapshots
   - CPU generated C
   - Metal generated MSL
   - dynamic-K matmul correctness on CPU
   - dynamic-K matmul codegen and runtime on Metal when device is available

## Current Opcode Audit

This section describes the current codebase state, not the target state.

### MIR Audit

Current `MIR` ops and intended status:

- `ProgramId`: keep in `MIR`
- `ShapeDim`: keep in `MIR`
- `ConstI64`: keep in `MIR`
- `ConstF64`: keep in `MIR`
- `IBinary`: keep in `MIR`
- `ICmp`: keep in `MIR`
- `TileConstant`: keep in `MIR`
- `TileLoad`: keep in `MIR`
- `TileStore`: keep in `MIR`
- `TileBinary`: keep in `MIR`
- `TileUnary`: keep in `MIR`
- `TileBroadcast`: keep in `MIR`
- `TileReduce`: keep in `MIR`
- `TileMma`: keep in `MIR`

Conclusion:

- the current `MIR` opcode set is already close to the intended semantic boundary
- the redesign work is primarily about fixing what gets emitted into `LLIR`

### LLIR Audit

Current `LLIR` core execution ops and intended status:

- `Alloca`: keep in `LLIR`
- `Gep`: keep in `LLIR`
- `Load`: keep in `LLIR`
- `Store`: keep in `LLIR`
- `Memcpy`: keep in `LLIR`
- `Bin`: keep in `LLIR`
- `Cmp`: keep in `LLIR`
- `Cast`: keep in `LLIR`
- `Select`: keep in `LLIR`
- `Br`: keep in `LLIR`
- `CondBr`: keep in `LLIR`
- `Switch`: keep in `LLIR`
- `Ret`: keep in `LLIR`
- `Intrinsic::ThreadId`: keep in `LLIR`
- `Intrinsic::BlockId`: keep in `LLIR`
- `Intrinsic::Barrier`: keep in `LLIR`

Current `LLIR` transitional semantic leftovers and intended disposition:

- `call @shape_dim(...)`
  - current status: transitional helper call
  - problem: still hides runtime shape semantics behind a helper name
  - target: explicit shape-buffer access or a narrowly defined low-level builtin

- `call @tile_splat_f32(...)`
  - current status: semantic helper
  - problem: still describes tile initialization as a domain helper
  - target: explicit loops plus stores

- `call @tile_load_2d_f32(...)`
  - current status: semantic helper
  - problem: still hides tile memory movement and indexing policy
  - target: explicit `gep` plus `load` loop nests

- `call @tile_store_2d_f32(...)`
  - current status: semantic helper
  - problem: still hides tile writeback semantics
  - target: explicit `gep` plus `store` loop nests

- `call @tile_add_f32(...)`
  - current status: semantic helper
  - problem: still hides elementwise tile execution
  - target: explicit loops plus scalar arithmetic

- `call @tile_sub_f32(...)`
  - current status: semantic helper
  - problem: same issue as tile add
  - target: explicit loops plus scalar arithmetic

- `call @tile_mul_f32(...)`
  - current status: semantic helper
  - problem: same issue as tile add
  - target: explicit loops plus scalar arithmetic

- `call @tile_div_f32(...)`
  - current status: semantic helper
  - problem: same issue as tile add
  - target: explicit loops plus scalar arithmetic

- `call @tile_exp_f32(...)`
  - current status: semantic helper
  - problem: hides loop structure
  - target: explicit loops plus scalar math call

- `call @tile_neg_f32(...)`
  - current status: semantic helper
  - problem: hides loop structure
  - target: explicit loops plus scalar negation

- `call @tile_broadcast_f32(...)`
  - current status: semantic helper
  - problem: hides indexing and replication semantics
  - target: explicit loops plus scalar copy pattern

- `intrinsic.reduce_add(...)`
  - current status: transitional
  - problem: currently stands for full tile reduction semantics, not a clearly low-level primitive
  - target: either lower to explicit loops or redefine as a true low-level subgroup/workgroup primitive

- `intrinsic.reduce_max(...)`
  - current status: transitional
  - problem: same issue as reduce add
  - target: either lower to explicit loops or redefine as a true low-level subgroup/workgroup primitive

- `intrinsic.matmul_fragment(...)`
  - current status: conditionally acceptable
  - problem: acceptable only if treated as a low-level capability intrinsic, not as semantic `tile.mma`
  - target: keep only as an execution choice produced by `MIR -> LLIR`; otherwise lower to explicit loops

### Priority Order For Cleanup

The recommended cleanup order is:

1. replace `shape_dim` helper-style lowering with an explicit low-level form
2. replace `tile_load_2d_f32` and `tile_store_2d_f32` with explicit loop-plus-memory lowering
3. replace tile elementwise helpers, unary helpers, and broadcast helpers with explicit loop lowering
4. decide whether tile reductions become explicit loops or true low-level collectives
5. make `TileMma` lower either to loops or to `intrinsic.matmul_fragment` based on lowering policy

### Practical Reading Of The Current State

Today’s `LLIR` is a usable transitional execution layer.

It is already good enough to:

- preserve explicit CFG
- preserve address spaces
- drive CPU execution
- drive Metal code generation

But it is not yet a clean final `LLIR`, because it still carries semantic helper calls that belong on the `MIR` side of the boundary.

## Immediate Next Step

The next implementation tasks should be:

1. audit current `LLIR` ops and mark which ones are still semantic leftovers
2. move semantic leftovers back to `MIR`
3. make `MIR -> LLIR` lower `TileMma` either to explicit loops or to low-level matrix intrinsics
4. add a dedicated LLIR structurization pass for Metal instead of embedding structurization logic in codegen
