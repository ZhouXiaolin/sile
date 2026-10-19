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

## Measured value counts, per construct

Every body op of every fixture has `Location::Unknown`, so a constant cannot be
attributed to its source position (`cutile/tests/ir_locate.rs`). The counts
below come from compiling one statement at a time and diffing the results
(`cutile/tests/ir_probe3.rs`, `ir_probe5.rs`, `ir_probe6.rs`, `ir_probe7.rs`,
`ir_probe8.rs`).

Most of the shape constants are **dead**: `Broadcast` and `Reshape` take their
shape as an attribute and name only one operand, so the constants their shape
arguments cost are printed but never consumed. cutile does not run DCE before
printing, and neither may this emitter.

| construct | values | breakdown |
| --- | --- | --- |
| `constant` `assume` `make_token` `make_tensor_view` `make_partition_view` `store_view_tko` arithmetic `reshape` `broadcast` | 1 | — |
| `get_tile_block_id` | 3 | plus 3 `assume`, so 6 in the body |
| `load_view_tko` | 2 | tile and token |
| `reduce` | 4 | `^bb0` args 2, body 1, result 1 |
| `a.broadcast(literal S)` over a scalar, rank `r` | 2r + (r+1) + 1 + r + r + 1 | `2r` target extents, `r+1` ones, `reshape`, `r` ones, `r` target, `broadcast` |
| `a.broadcast(x.shape())` over a scalar, rank `r` | 4r + (r+1) + 1 + r + r + 1 | `x.shape()` costs **4r**, twice what a literal shape costs |

The scalar broadcast is always three instructions — `reshape` the scalar to
all-ones, then `broadcast` — and the target shape is materialised twice before
the reshape and once more around the broadcast. Rank 1, rank 3 and rank 4 all
follow the same formula; it was checked at each rank.

## Verified: saxpy prints byte for byte

`scripts/check_tileir_alignment.sh` reports `saxpy PASS`. The emitter covers the
five constructs saxpy uses: a scalar broadcast, `load_like`, `load`, `store`, and
one binary arithmetic op.

## The shape-fetch groups, and why they do not generalise

A tile read is groups of `constant` fetches around one `get_tile_block_id`
(`ir_probe3`, `ir_probe10`, `ir_probe12`, each compiled with distinct extents so
a group boundary is unambiguous):

```
<generics>          one constant per const generic, once
group group ...     before the block id
get_tile_block_id   3 results, then one assume each
group group ...     after the block id
make_partition_view
load_view_tko
```

One group is `rank` constants — one shape expression's worth. The counts:

| case | rank | groups before | after |
| --- | --- | --- | --- |
| `z.load()`, `z.store(t)` | 2 | 2 | 1 |
| `x.load_like(z)`, all static | 2 | 5 | 2 |
| `x.load_like(z)`, source dynamic | 2 | 5 | 2 |
| `x.load_like(z)`, all static | 1 | 4 | 2 |
| saxpy `x.load_like(y)` | 2 | 4 | 2 |
| saxpy `y.load()` | 2 | 2 | 1 |

Static and dynamic give the same answer, and a plain `load` gives the same
answer twice over. Rank does not: rank 1 costs 4 groups and rank 2 costs 5. And
two rank-2 `load_like` calls disagree with each other (5 versus saxpy's 4).

### Where the counts come from (read off cutile's source)

Three mechanisms, all confirmed in `cutile-rs`:

1. **`compile_binary_op.rs:668`** — cutile allocates the id for an integer
   operation, runs `value_facts::transfer`, and if the interval is a single
   value it returns a fresh `constant` *without appending the operation*. The
   discarded id is never printed. This is the whole reason the fixtures have
   gaps in their numbering (gemm's `%74`, async_gemm's `%73`, batch_matmul's
   `%88`, rms_norm's `%31`, flash_attention_causal's `%163/304/322/325`).
   `crates/sile-tile-ir/src/cutile/bounds.rs` is the interval arithmetic, and
   `Builder::discard` reproduces the gap.
2. **`compile_cuda_tile_op.rs:792`** — `make_partition_view` *compiles* its
   shape argument (emitting that shape's `constant` fetches) but then takes the
   tile shape from the argument's static *type* (`type_instance.shape`) and
   throws the compiled value away. So a shape argument is printed and dropped.
3. **`cutile/src/_core.rs:4141`** — `load_tile_like` is ordinary generic Rust;
   the JIT inlines it. Every `shape()` call and every `make_partition_view`
   shape argument inside that body costs its own `2r` fetches, so the group
   count follows the *library body*, not the kernel.

Point 3 is the wall. `load_tile_like` is written once per rank, and how many
times its `y.shape()` and tile-shape argument get materialised depends on the
bounds cutile infers for each (mechanism 1 changing the control flow of the
inlined body). Two rank-2 `load_like` calls with identical strides and generics
land on 6 and 7 groups respectively, and the difference tracks whether the
*source* view's extents are static.

### Measured, from cutile's own output (rank 1, one tensor parameter)

Constant counts read off probe kernels, deltas over an empty body:

| body | constants | delta |
| --- | --- | --- |
| empty, rank 1 | 2 | — (the prologue's own `2 2`) |
| `z.shape()` | 3 | **1** = rank |
| `z.shape()` twice | 4 | +1 |
| `z.shape()` three times | 5 | +1 each |
| `z.shape()` rank 2 | 6 | 4 = 2 × rank for the prologue, +2 for the call |
| `x.load_like(z)` | 8 | **6** = 3 × rank × 2 |
| `x.load_like(z)` twice | 14 | +6 |
| `x.load_like(z)` ×3 | 20 | +6 each |
| `x.load_like(z)`, rank 2 | 16 | **14** = 7 × rank |

So `shape()` costs exactly `rank` constants every time it is called — it is
**not** memoised. `load_like` costs `3 × rank × 2` for a rank-1 target and
`7 × rank` for rank 2, i.e. its body names the target's extents three times
(two rounds before the block id, one after) and the source's extents the same,
which is the `read_tile` shape already implemented.

### Refuted: a shape memo

An earlier turn hypothesised that `get_tensor_shape_meta` memoises the shape, so
the second `load_like` on the same target would be cheaper. Two measurements
kill it:

* `z.shape()` called three times in one body costs `1, 1, 1` — no saving.
* `async_add`'s second `load_like` is short by exactly one constant, and a memo
  keyed on the *target* cannot explain that while leaving `saxpy` intact: adding
  it regressed `saxpy` from PASS to DIFF, because `saxpy` calls
  `a.broadcast(y.shape())` first and would then double-count.

The one-constant difference in `async_add` is therefore **not** a memo hit. It
is still unexplained, and it is the only thing standing between the current
emitter and a second passing fixture.

### The decisive fact: the same expression is not always the same cost

In `async_add` the golden text contains two `load_like` calls on the same
target tensor `z` and the same rank:

```
x.load_like(z)   costs 5 shape fetches before the block id:  2 2 -1 2 -1
y.load_like(z)   costs 2 shape fetches before the block id:  2 2
```

Same library function, same `z`, same rank, same generics. The second call is
cheap because `get_tensor_shape_meta` is a `return_type_meta_field` compiler op
(`compile_intrinsic.rs:2197`) that hands back the shape value **already stored
in the tensor's type metadata** — it does not emit a new `constant` at all.
`label_param_extents` (`_function.rs:394`) then annotates that shared value with
the parameter's extent atoms, in place.

So the cost of a shape fetch is a property of **whether that shape has been
materialised yet in this compilation**, not of the expression that asks for it.
An emitter that walks expressions independently cannot reproduce it; it needs
the same memo of "this tensor's shape is already a value".

### Why this is not a table either

The remaining ten fixtures cannot be finished by extending a formula. Both are rank-2
`load_like` over an output whose view is whole-buffer; the only difference is
that one source is fully dynamic and the other declared `[-1,-1]` against a
`[2,2]` tile. Same shape, same strides, same generics count, different cost.

The counts come from cutile's generic instantiation and bounds inference
(`generics.rs`, 138K; `compile_cuda_tile_op.rs`, 191K), which substitute the
bounds each `const_shape!` argument resolves to. Reproducing them means
reimplementing that substitution, not extending a formula.

## Status

**All eleven fixtures print byte for byte.** `./scripts/check_tileir_alignment.sh`
reports `11/11 PASS`: `saxpy`, `async_add`, `async_gemm`, `batch_matmul`,
`dropout`, `flash_attention`, `flash_attention_causal`, `gemm`, `gemm_static`,
`rms_norm`, `softmax`. The comparison needs no GPU and no `tileiras`: cutile
compiles with `KernelCompiler::compile()` and reads `CompileArtifacts::ir_text()`,
sile dumps `tileir-00-lowered` with `SILE_DUMP_TILEIR_ONLY=1`.

The two pieces the last section called for are now in the emitter, but in the
form the measurements supported rather than as a general memo and a general
interval analysis:

- **Shape fetches are counted per construct.** `shape_materialised` records the
  buffers whose extents have already become values, and `shape_derived` marks the
  bindings whose value came out of a shape. Those two flags decide the extra
  round a `broadcast` or a `load_like` pays. A general per-tensor memo was not
  needed: the only place the reference reuses a stored shape is a shape the
  kernel bound to a name.
- **Folding is decided per comparison, not by interval arithmetic.** A comparison
  between a const generic and a literal folds while compiling: the compared value
  is named, `Builder::discard` reproduces the id the dropped comparison took, and
  the branch is pasted in place of a region. `bounds.rs` states the rules; the
  emitter does not yet need a general interval domain for these fixtures.

### Control flow

A region is written after the ops it contains, because its result ids follow
everything they dominate. An `if` yields the values the enclosing `for` carries
first, then the names the branch assigns. A branch that changes a carried value
prints an `else` even when the kernel has none, holding the value from before the
branch.

### Element types

`constant(true, shape)` is an `i1` tile, not `1.0`: the front end keeps a boolean
literal as `Expr::ScalarBool`, so the emitter can name the mask's element type.
Arithmetic follows the operands: integer tiles print `addi`/`subi`/`muli` and
integer comparisons print `cmpi … signed`, float ones `addf`/`cmpf … ordered`.
