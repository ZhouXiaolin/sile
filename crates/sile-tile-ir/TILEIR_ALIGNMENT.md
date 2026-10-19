# Tile IR alignment with cutile-rs

`sile` must print the same Tile IR text as `cutile-rs` for the same kernel. The
fixtures in `crates/sile/tests/fixtures/tileir/` are that text, taken from
`cutile-rs/cutile/tests/tileir_alignment.rs`. `scripts/check_tileir_alignment.sh`
diffs both sides and reports the first line that differs.

`diff = 0` is byte for byte, so the emission order and the redundant
instructions are part of the contract. This note records the rules that order
follows, each read off real cutile output rather than guessed.

## The kernel source travels in the HIR

cutile expands each argument into several entry parameters and then emits the
view, token and view-set-up code from the source of the kernel. `sile` copies
that: `sile-hir` carries the module name, the source of the module and the
index of the entry inside it, and the Tile IR lowering reads them back. This
keeps the "`sile` prints what `cutile` prints" rule honest, and it is the only
way to land on the same SSA numbering, which depends on how many constants
each source expression asks for.

## Module and entry names

Both sides hand their kernels to a module named `kernels`, so the printed
`cuda_tile.module @kernels` matches. The entry is `<fn>_entry`, and it carries
`di_name = "<fn>"`.

## Entry parameters

A tensor parameter becomes `1 + 2*rank` entry parameters, in this order:

1. the pointer, `tile<ptr<E>>`;
2. `rank` shape extents, `tile<i32>`;
3. `rank` strides, `tile<i32>`;
4. for a mutable parameter only, `rank` partition shape extents and `rank`
   partition strides, again `tile<i32>`.

A scalar parameter stays one `tile<E>` parameter. The entry is named
`<fn>_entry` and carries `di_name = "<fn>"`.

## Prologue

One block per tensor parameter, in parameter order. The tokens and the `%n`
numbering come from the order below, so nothing here is optional.

1. One `constant <i32: v>` per const generic, in declaration order, even when
   the generic is never used.
2. For each tensor parameter:
   1. `assume bounded<0, ?>` on each shape argument that is written `-1` in the
      declared shape. A shape extent that is a const generic prints as the
      constant itself.
   2. `make_token`.
   3. `make_tensor_view`, either
      `%p, shape = [%a, %b], strides = [+, +] : tile<i32> -> tensor_view<...>`
      when a shape argument is dynamic, or without the `: tile<i32> ->` part
      when the shape is fully static.

## Constant fetches

Each source expression that needs a shape reads it again: one `constant` per
axis, in source order, and every one of them is fresh — never a reuse, never a
fold. Measured counts (`cargo test -p cutile --test ir_probe`):

| source | constants emitted |
| --- | --- |
| `a.broadcast(shape![2, 4])` over a scalar | `[2, 4]` `[2, 4]` `[1, 1]` `[1, 1]` before the `reshape`, `[1, 1]` `[2, 4]` after it |
| `a.broadcast(shape![4])` over a scalar | `[4]` `[4]`, then `reshape`, then `[1]` `[4]` |
| `a.broadcast(x.shape())`, rank 1, static | four `[4]`, then `[1]` `[1]`, `reshape`, `[1]` `[4]` |

The broadcast of a scalar is always spelled as three instructions: `reshape`
the scalar tile to all-ones, then `broadcast` to the wanted shape. The
constants around it are what the shape arguments of `reshape` and `broadcast`
cost, and they print as `<i32: v> : tile<i32>` even where the operation reads
them as a shape.

This is the part of the contract that cannot be derived from the HIR: it is a
property of cutile's front end. To land on the same numbering, `sile-hir` must
carry the kernel source and the lowering must read it back, which is why the
section above says so.

## Memory access

`load` emits, in order:

1. the shape constants for the buffer and for the tile;
2. `get_tile_block_id`, which returns three `tile<i32>` results, followed by
   three `assume bounded<0, ?>`;
3. `make_partition_view` over the parameter's tensor view — `padding_value =
   zero` for a read view, absent for a write view;
4. `load_view_tko weak %view[%i, ...] token = %tok`, which returns the tile and
   a token.

A mutable parameter's tensor view is the one built from its full tensor shape,
which is why a store can index it by the tile block id.

`if` and `for` carry their region inline, and `for` takes the carried values as
`iter_values(... = %v)` with a matching `continue %v` at the end of the body.

## What this asks of `sile`

`sile-hir` keeps its own shape, because it must stay expressible for the C,
WGPU and PTX backends. The lowering to Tile IR is where the cutile emission
order lives, and that lowering is a separate emitter from the one the other
backends consume.

## The emitter

`sile-tile-ir` gains `cutile::Module` / `cutile::Op` / `cutile::Type`: a
cutile-shaped IR whose only job is to print the same text. It is separate from
`sile-tile-ir::ir` because that one carries what the C, WGPU and PTX lowerings
read (shape descriptors, buffer ranks, `stride_shape_idx`), and those fields
have no place in cutile's text.

The pass order is one walk, not four:

1. `cutile::Module` is built from `Kernel::origin` — the source text and the
   entry index — replaying the walk cutile's front end makes: expand each
   parameter into entry parameters, then prologue, then the body expression by
   expression, fetching each shape argument as a fresh `constant`.
2. The body needs typed values while it is built, so the walk runs against
   `sile-hir::typeck::TypedKernel` and uses its types; it does not consult
   `sile-tile-ir::ir` at all.
3. `format` prints it.

Consequences to keep in mind:

- The value ids are allocated in the walk's own order, which is what makes the
  numbering match. Nothing may be reordered to "clean up" the output.
- Every construct the examples use is already enumerated in
  `fixtures/tileir/*.tileir`: 34 opcode names, listed by
  `rg -o '^\s+(%[0-9]+, )?%?[0-9]*[a-z_]+' fixtures/tileir`. Anything outside
  that set is out of scope until an example needs it.
- A kernel without an origin (`KernelOrigin::default()`) cannot be replayed.
  The emitter reports that rather than printing something plausible.

## Print formats, measured

The 34 opcodes the eleven fixtures use, with the exact text each prints. `%t`
is the result, `%a`/`%b`/`%c` are operands. Read this instead of re-probing.

```
%t = constant <i32: 2> : tile<i32>                 # also <f32: 0.5>, <i1: 1>
%t = assume bounded<0, ?>, %a : tile<i32>          # second bound is `bounded<0, 7>`
%t = make_token : token
%t = make_tensor_view %p, shape = [%a, %b], strides = [8, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[8,1]>
%t = make_tensor_view %p, shape = [4], strides = [1] : tensor_view<4xf32, strides=[1]>   # no `->` when static
%t = make_partition_view %v : partition_view<tile=(2x2), padding_value = zero, tensor_view<?x?xf32, strides=[8,1]>>
%t = make_partition_view %v : partition_view<tile=(32x16), tensor_view<?x?xf32, strides=[128,1]>>   # write view: no padding
%t, %u = load_view_tko weak %v[%i, %j] token = %k : partition_view<...>, tile<i32> -> tile<2x2xf32>, token
%t = store_view_tko weak %val, %v[%i, %j] token = %k : tile<2x2xf32>, partition_view<...>, tile<i32> -> token
%t, %u, %w = get_tile_block_id : tile<i32>
%t, %u = get_index_space_shape %v : partition_view<...> -> tile<i32>
%t = addi %a, %b : tile<i32>                       # subb/muli/addf/subf/mulf/divf: no suffix
%t = divi %a, %b signed : tile<i32>                # divi/remi/mini/maxi take ` signed`
%t = andi %a, %b : tile<i1>
%t = cmpi less_than %a, %b, signed : tile<i32> -> tile<i1>
%t = cmpf greater_than ordered %a, %b : tile<4xf32> -> tile<4xi1>
%t = select %m, %a, %b : tile<4xi1>, tile<4xf32>
%t = mmaf %a, %b, %c : tile<32x8xf32>, tile<8x16xf32>, tile<32x16xf32>
%t = reduce %a dim=1 identities=[-inf] : tile<32x32xf32> -> tile<32xf32> { ^bb0(%x, %y): ... yield %z : ... }
%t = reshape %a : tile<32x32xf32> -> tile<1x32x32xf32>
%t = broadcast %a : tile<1x1xf32> -> tile<32x32xf32>
%t = permute %a [1, 0] : tile<4x8xf32> -> tile<8x4xf32>
%t = iota : tile<32xi32>
%t = fma %a, %b, %c : tile<4x1xf32>
%t = exp %a : tile<2x8xf32>                        # exp2/log/sqrt/... likewise; sqrt adds rounding<negative_inf>
%t = maxf %a, %b {rounding_mode = 0} : tile<32x1xf32>
%t = itof %a signed : tile<i32> -> tile<f32>
%t = for %i in (%lo to %hi, step %s) : tile<i32> iter_values(%p = %v, ...) -> (tile<...>, ...) {
      continue %a, %b : tile<...>, tile<...>
    }
%t0, %t1 = for ... { yield %a, %b : tile<...>, tile<...> }        # when the loop is an expression
%t0, %t1 = if %c -> (tile<...>, tile<...>) { ... yield %a, %b : ... } else { ... yield ... }
assert %c, "partition access out of bounds: dim 0, block index >= ceil(?/1) or index < 0" : tile<i1>
return
```

Two rules the table cannot show: `, signed` appears only on the integer ops
that carry a signedness attribute, and an empty attribute dictionary is
omitted entirely rather than printed as `{}`.

## Access checks: the proof ladder

cutile decides an access check before it emits anything, and only falls back to
a device `assert` when every proof source fails (`cutile-compiler/src/compiler/
checks/mod.rs`). The rungs are tried top-down and the first one that decides the
axis ends it:

1. **Provenance.** The index is bounded by construction: a `tile_block_id(k)`
   component is bounded by the launch grid, and a partition-axis iterand is
   bounded by its axis's tile count. Both goals hold, so no check is emitted.
2. **Static fold.** The index interval and the tile grid are both known, so
   `0 <= start && end < num_tiles` decides the axis. This rung is a decision,
   not a check: a violation is a **compile error**.
3. **Declared-fact entailment.** The goal follows at JIT time from a
   launch-verified precondition.
4. **Launch hoist.** The operands are launch-known, so the check moves to the
   host.
5. **Residual placement.** Nothing decided it, so the check becomes an
   in-kernel `assert`, hoisted to the outermost loop preheader whose frame
   decides it.

The emitter implements rungs 1, 2 and 5. Rungs 3 and 4 need a host-side
predicate language that sile does not have yet, so an undecided goal goes
straight to rung 5 — which is what cutile does too whenever its own rungs 3 and
4 find nothing to work with.

### What each rung needs

Rung 1 reads a fact carried by the value: which axis, if any, bounds it.
`Builder::bounded_loop_vars` holds the names a `for` assumed non-negative, and
`is_block_id_component` recognises the grid components. A value *derived* from
either has lost the bound, which is why `i + 1` is not rung-1 material.

Rung 2 needs two things, and the second is the one that used to be missing:

- the **interval** of the index, from `bounds.rs` via `bounds_of`. A `for`
  variable contributes `[0, end]` because its `assume bounded<0, n>` says so.
- the **static extent**, which `declared_extents` already resolved from the
  parameter's declaration. A literal extent means the tile grid is known; a
  `-1` or a dynamic symbol means it is not, and rung 2 abstains.

So `fold_static` reads the interval, computes `num_tiles = ceil(extent / tile)`,
and either accepts the axis silently or records a compile error. The message is
the one cutile prints, because it is the kernel author's message:

```
Bounds check failed: 0 <= 1 && 40 < 32
```

That error leaves `emit` as an `Err`, not as an `assert` in the text. This is
the one place sile rejects a kernel that it used to accept, and it is the
correct behaviour: a statically violated access is not a program.

### Why the fixtures keep their asserts

`gemm` carries one `assert` and `flash_attention` carries eight, and both match
cutile byte for byte. That is not a gap: those kernels index by a block id
whose extent is `?`, so rung 2 has no static extent and abstains, and rungs 3
and 4 have nothing to prove with. cutile emits the same assert for the same
reason. The check is the honest answer, not a fallback sile failed to improve
on.

`static_extent` is the fixture that shows the difference. Its input declares
`[64]`, so the extent is static, the index is a loop counter plus one, and
rung 2 proves `1 <= i + 1 <= 4` inside `ceil(64 / 2) = 32` tiles. No assert is
emitted. The same kernel with `0..40` is a compile error on both sides.

## Status

**All fifteen fixtures print byte for byte.**
`./scripts/check_tileir_alignment.sh` reports `15/15 PASS`: `saxpy`,
`async_add`, `async_gemm`, `batch_matmul`, `dropout`, `flash_attention`,
`flash_attention_causal`, `gemm`, `gemm_static`, `rms_norm`, `softmax`,
`async_saxpy`, `hello_world`, `jit_custom_store`, `static_extent`. The
comparison needs no GPU and no `tileiras`: cutile compiles with
`KernelCompiler::compile()` and reads `CompileArtifacts::ir_text()`, sile dumps
`tileir-00-lowered` with `SILE_DUMP_TILEIR_ONLY=1`.

The per-construct value counts and the shape-fetch groups are in
`TILEIR_MEASUREMENTS.md`. They are a regression list, not the specification:
this document states why the text is what it is.

### Shape fetches

A tensor's extents become SSA values once, when its view is built. A later
`broadcast` or `load_like` that needs them again pays for them again, because
cutile reads the shape out of the tensor's type metadata and only the first
asker pays for materialising it. Two facts decide the extra round:
`shape_materialised` records the buffers whose extents have already become
values, and `shape_derived` marks the bindings whose value came out of a shape.

### Control flow

A region is written after the ops it contains, because its result ids follow
everything they dominate. An `if` yields the values the enclosing `for` carries
first, then the names the branch assigns. A branch that changes a carried value
prints an `else` even when the kernel has none, holding the value from before
the branch.

### Element types

`constant(true, shape)` is an `i1` tile, not `1.0`: the front end keeps a boolean
literal as `Expr::ScalarBool`, so the emitter can name the mask's element type.
Arithmetic follows the operands: integer tiles print `addi`/`subi`/`muli` and
integer comparisons print `cmpi … signed`, float ones `addf`/`cmpf … ordered`.

### Program id and print

Two spellings read the same three grid components and cost different ids, so
they are separate ops. `get_tile_block_id().k` reads the triple once and every
later component reuses it; `program_id(axis)` is CuTile's helper, written as a
plain Rust function with an `if` chain, so each call reads the triple again and
tests the axis. Every test names its candidate constant and yields a constant
predicate, and the compared id between them is allocated and dropped. Only the
`else` arm carries the assertion, and it tests the predicate the last comparison
already produced.

`print_tko` yields a token and never states it: the operand types already end
the line. Its format string keeps the kernel's text, with each `{}` rewritten to
`%` plus the operand's element letter, which is why the print op has to be a HIR
statement carrying both the text and the operands.
