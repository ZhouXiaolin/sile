# Sile Kernel Compiler Design

## Background

`crates/sile/examples/vec_add.rs` and `crates/sile/examples/softmax.rs` define the target user experience for `sile`.

They are not ordinary Rust helper examples. They describe a tile-centric kernel DSL embedded in Rust syntax:

- Users describe the behavior of one tile block, not the full tensor traversal.
- Shapes are part of the language and participate in type checking.
- Kernel code primarily manipulates `Tile` values, not host-side `Tensor` containers.
- `reduce_*`, `reshape`, `broadcast`, `exp`, and tile loading are compiler-known operations.
- `partition` and `unpartition` are runtime or scheduling concerns that should eventually be inferred and handled automatically rather than forming part of the kernel author's core mental model.

This design replaces the current MVP direction based on a narrow `KernelSpec` and ad-hoc code generation. The new direction is a proper compiler pipeline:

- frontend parsing
- HIR construction
- type and shape checking
- SSA conversion
- canonicalization and optimization passes
- backend lowering
- runtime launch integration

## Goals

1. Preserve the language shape expressed by the examples.
2. Make tile programming the primary abstraction, similar in spirit to Triton.
3. Support compile-time and runtime shape reasoning in one system.
4. Generate correct backend code from a typed IR rather than from hardcoded macro expansion.
5. Absorb partitioning, tile offsets, and launch mapping into compiler and runtime infrastructure instead of forcing users to reason about them directly.

## Non-Goals

1. Preserve the current MVP `KernelSpec` as the core internal representation.
2. Optimize for peak performance in the first implementation phase.
3. Support arbitrary Rust syntax inside kernels.
4. Support implicit broadcasting in the first version.
5. Design every future backend up front. The first backend only needs to be good enough to validate the pipeline.

## Language Model

### Kernel Author Mental Model

A kernel describes what one tile block does:

1. Determine the current tile block id.
2. Load one or more tiles from global tensors.
3. Perform tile-local elementwise and reduction operations.
4. Store the resulting tile into an output tensor window.

The user does not write global loops over the tensor. The global launch grid determines how many blocks exist, and the compiler/runtime map each block to its region in the input and output tensors.

### Core Value Categories

- `Tensor`: global logical tensor view passed into the kernel.
- `Tile`: tile-local value produced by loads and transformed by kernel operations.
- `Shape`: shape-level value used in type checking and shape transforms.
- `ProgramId`: current tile block id in the launch grid.

### Shape Semantics

The examples imply a shape language with these forms:

- dynamic dimension: `-1`
- symbolic dimension: `BM`, `BN`, `S`
- explicit tuple shape: `[BM, BN]`
- const shape macro form: `const_shape![BM, 1]`

Shape expressions must participate in checking and lowering, not remain as opaque tokens.

### Required Builtins

The examples require these language-level builtins:

- `get_tile_block_id()`
- `load_tile`
- `load_tile_like_2d`
- `store`
- `reduce_sum`
- `reduce_max`
- `reshape`
- `broadcast`
- `exp`
- arithmetic operators on `Tile`
- `shape()` on values whose shape is visible to the compiler

These are not ordinary helper functions. They are semantic operations that must lower into HIR and later IR nodes.

## Frontend

### Responsibility

The frontend receives the kernel definition and constructs a parsed representation for the DSL. It is responsible for:

- collecting kernel name and attributes
- collecting const shape parameters
- parsing tensor and tile type annotations
- parsing statements and expressions in the restricted kernel body
- resolving builtin operations and special forms
- producing user-facing syntax diagnostics

### Scope

The kernel body will support a restricted Rust-like subset:

- `let` bindings
- explicit type annotations on `let`
- simple reassignment where needed
- builtin and method-call syntax for recognized operations
- arithmetic operators
- final `store`

General Rust control flow is out of scope initially.

## HIR

### Purpose

HIR should preserve language intent and remain backend-independent. It must still know about:

- tensor vs tile values
- shape expressions
- axes
- tile block ids
- builtin operation kinds

### Core HIR Structures

Recommended top-level structures:

- `Kernel`
- `Param`
- `Type`
- `ShapeExpr`
- `Stmt`
- `Expr`
- `BuiltinOp`

Recommended value and operation categories:

- `TensorType(elem, shape)`
- `TileType(elem, shape)`
- `ShapeType`
- `ProgramId`
- `LoadTile`
- `LoadTileLike2D`
- `StoreTile`
- `Unary`
- `Binary`
- `Reduce`
- `Reshape`
- `Broadcast`
- `ShapeOf`

### Example HIR Intent for `vec_add`

The `vec_add` example describes:

1. one symbolic tile shape parameter `S`
2. two dynamic 1D input tensors
3. one output view with tile shape `S`
4. current block id acquisition
5. tile loads from `a` and `b` using shape `S` and offset `[pid]`
6. elementwise add
7. tile store into `c`

### Example HIR Intent for `softmax`

The `softmax` example adds:

1. symbolic 2D tile shape parameters `BM` and `BN`
2. 2D input tensor and 2D output tile view
3. load based on output tile shape
4. reduction along axis `1`
5. reshape from `[BM]` to `[BM, 1]`
6. broadcast from `[BM, 1]` to `[BM, BN]`
7. elementwise subtraction, exponentiation, summation, division
8. final tile store

## Type and Shape Checking

### Purpose

This phase assigns a checked type and checked shape to every value. It is the semantic backbone of the compiler.

### Rules

1. `Tensor<f32, { [-1] }>` means a runtime 1D tensor with one dynamic extent.
2. `Tensor<f32, { [-1, -1] }>` means a runtime 2D tensor with dynamic extents.
3. `Tensor<f32, { [BM, BN] }>` in a kernel parameter denotes the tile-visible shape contract for the current program block.
4. `load_tile(shape=S, offset=[pid])` returns `Tile<f32, S>`.
5. `load_tile_like_2d(x, y)` returns a tile whose shape matches `y`.
6. `reduce_max(Tile<[BM, BN]>, axis=1)` returns `Tile<[BM]>`.
7. `reduce_sum(Tile<[BM, BN]>, axis=1)` returns `Tile<[BM]>`.
8. `reshape(Tile<[BM]>, [BM, 1])` requires equal element counts and returns `Tile<[BM, 1]>`.
9. `broadcast(Tile<[BM, 1]>, [BM, BN])` requires standard broadcast compatibility and returns `Tile<[BM, BN]>`.
10. Binary arithmetic requires equal shapes in the first implementation.
11. `store` requires source tile shape to match the destination tile view shape.

### Diagnostics

Diagnostics must report:

- invalid shape syntax
- unresolved names
- invalid builtin usage
- invalid reduction axis
- incompatible reshape element counts
- invalid broadcast
- store shape mismatch

Errors should point back to the kernel source, not only to internal IR.

## SSA IR

### Purpose

After HIR is type-checked, it should lower into a typed SSA form where:

- every intermediate value has one definition
- every value carries element type and shape
- operations are explicit and canonical
- optimization and lowering become predictable

### Required Properties

1. Every tile-producing operation yields a typed SSA value.
2. Shape information is explicit on values or instructions.
3. Program id access is explicit.
4. Loads and stores are explicit memory boundary operations.
5. High-level shape operations remain explicit until lowering decides how to implement them.

### Canonical SSA Sketch for `softmax`

```text
%0  = program_id axis=0
%1  = load_tile_like_2d %x, %y        : tile<f32, [BM, BN]>
%2  = reduce_max %1 axis=1            : tile<f32, [BM]>
%3  = reshape %2 to [BM, 1]           : tile<f32, [BM, 1]>
%4  = broadcast %3 to [BM, BN]        : tile<f32, [BM, BN]>
%5  = sub %1, %4                      : tile<f32, [BM, BN]>
%6  = exp %5                          : tile<f32, [BM, BN]>
%7  = reduce_sum %6 axis=1            : tile<f32, [BM]>
%8  = reshape %7 to [BM, 1]           : tile<f32, [BM, 1]>
%9  = broadcast %8 to [BM, BN]        : tile<f32, [BM, BN]>
%10 = div %6, %9                      : tile<f32, [BM, BN]>
store_tile %y, %10
```

This is the kind of structure later passes and backends should consume.

## Pass Pipeline

### Mandatory First-Phase Passes

1. `NameResolution`
Maps identifiers to params, const shape symbols, locals, and builtins.

2. `TypeAndShapeCheck`
Infers and validates every type and shape.

3. `LowerToTypedSSA`
Produces explicit SSA values and canonical operations.

4. `Canonicalize`
Normalizes equivalent forms and simplifies builtins into a smaller core set.

5. `FoldConstants`
Folds shape constants and simple scalar constants where legal.

6. `NormalizeShapeOps`
Normalizes `reshape` and `broadcast` patterns into backend-friendly form.

7. `DCE`
Removes unused SSA values.

8. `BackendLowering`
Converts canonical typed SSA into backend IR.

### Deferred Passes

These are important later but should not block the first implementation:

- common subexpression elimination
- fusion
- layout-aware rewrites
- local memory promotion
- vectorization
- backend-specific performance tuning

## Scheduling and Automatic Partitioning

### Principle

Kernel authors should not have to manually partition tensors in order for the kernel to be semantically correct.

The compiler and runtime together should decide:

1. how a launch grid maps to tile coordinates
2. how tile coordinates map to tensor windows
3. how output views correspond to the current program block
4. how partial boundary tiles are handled

### Implication

`partition` and `unpartition` are not core language semantics. They may exist temporarily at the runtime API layer, but the long-term design is:

- host tensors remain global logical tensors
- kernel parameter shapes describe tile-local contracts
- scheduling maps global tensors to tile-local accesses automatically

### Boundary Policy

The initial implementation should pick one explicit policy and encode it consistently:

1. require divisibility for the first phase, or
2. support masks and partial tiles as a later extension

The first phase should prefer requiring divisibility to keep the pipeline smaller and more predictable.

## Backend IR

### Purpose

Backend IR bridges typed SSA and concrete code generation. It should be lower-level than HIR and SSA but still independent of any one textual backend.

It should explicitly represent:

- loop structure over tile dimensions
- access offsets into global tensors
- temporary tile buffers
- scalar temporaries for reduction
- shape extents known at compile time vs runtime

This avoids binding HIR directly to C code shape.

## Backend Strategy

### First Backend

The first backend should be CPU through generated C, because it is sufficient to validate:

- correctness of parsing
- correctness of type and shape checking
- correctness of SSA and pass structure
- correctness of scheduling and tile mapping

### First-Phase Required Capabilities

1. 1D and 2D tile addressing
2. elementwise arithmetic
3. unary `exp`
4. reduction along one axis
5. reshape
6. explicit broadcast
7. final store to output window

Performance is secondary to correctness.

## Runtime Integration

The runtime should keep responsibility for:

- device and stream management
- host tensor allocation
- launch submission
- passing runtime shapes and launch parameters into compiled kernels

The runtime should not be responsible for understanding kernel semantics beyond what is needed to execute compiled code.

## Testing Strategy

### Language-Level Tests

1. parser tests for kernel signatures and shape syntax
2. HIR construction tests for builtin recognition
3. type and shape tests for valid and invalid programs
4. SSA snapshot tests for canonicalized `vec_add`
5. SSA snapshot tests for canonicalized `softmax`

### End-to-End Tests

1. `vec_add` executes correctly on CPU backend
2. `softmax` executes correctly on CPU backend
3. invalid shape programs produce source-facing diagnostics
4. invalid reduction and broadcast cases fail before backend lowering

## Migration Strategy

### Phase 1

Introduce new compiler modules without forcing an immediate rewrite of all runtime code.

### Phase 2

Route kernel compilation through the new frontend, HIR, type checker, SSA pipeline, and backend lowering.

### Phase 3

Reduce the old MVP `KernelSpec` to a compatibility layer or remove it once the new compiler path fully replaces it.

## Recommended Repository Structure

```text
crates/sile-macros/
  src/
    lib.rs
    frontend/
    diagnostics/

crates/sile/
  src/
    hir/
    typeck/
    ssa/
    passes/
    schedule/
    backend_ir/
    backend/
    runtime/
```

Exact module names can change, but the phase boundaries should remain.

## Implementation Milestones

1. Build the new frontend and HIR around `vec_add`.
2. Add type and shape checking for `vec_add`.
3. Lower `vec_add` to typed SSA and generate working CPU code.
4. Extend the type system and SSA to support `softmax` operations.
5. Add `reduce`, `reshape`, `broadcast`, and `exp` lowering.
6. Move tile mapping and partitioning semantics into scheduling/runtime integration.
7. Replace the old MVP-centered path with the new compiler path.

## Open Decisions

These decisions still need to be fixed before implementation begins:

1. whether kernel entry syntax remains an attribute form or moves to a function-like macro form
2. whether `shape()` is always compile-time visible or may sometimes lower to runtime shape queries
3. whether the first implementation supports only divisible shapes or introduces masked tails
4. how much ordinary Rust syntax is accepted inside kernel bodies before explicit rejection

## Recommendation

Proceed with a full compiler architecture rather than incrementally extending the MVP `KernelSpec`.

The first implementation target should be:

- `vec_add` end-to-end through the new compiler pipeline
- then `softmax` as the feature-completeness test for reductions, shape transforms, and tile-local semantics

This preserves the language direction expressed by the examples and avoids spending engineering effort on a representation that will be discarded.
