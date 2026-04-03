# Simplify Tensor, Add MMA, Run Matmul

## Goal

Remove type-level `Rank` from runtime `Tensor`/`Tile`. The `#[kernel]` macro parses `{[...]}` shape syntax for the HIR compiler pipeline, but rewrites types to simple `Tensor<f32>` for runtime. Unify all dimension values to `i64`. Add `Tensor::random()`. Make the matmul example run end-to-end with correct results.

## Changes

### 1. Remove `Rank` from runtime types (`tensor.rs`, `tile.rs`)

**Delete:** `Rank` trait, `DList<V, R>`, `DListNil`

**`Tensor<T>`:**
```rust
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
}
```

**`Tile<T>`:**
```rust
pub struct Tile<T> {
    pub shape: Vec<i64>,
    pub _elem: PhantomData<T>,
}
```

Remove all `R: Rank` bounds from impl blocks and method signatures. Methods like `load_tile`, `store`, `dim`, `exp`, `reduce_*` use runtime `Vec<i64>` shape only.

**`Tensor::dim()` must actually return the shape value** (currently a stub returning 0):
```rust
pub fn dim(&self, idx: usize) -> i64 {
    self.shape[idx]
}
```

**Remove `#![feature(generic_const_exprs)]` and `#![allow(incomplete_features)]`** from `lib.rs`. These existed solely for `DList<const V: i32, R>` const generics.

### 2. Macro parses shape, writes to HIR

In `sile-macros/src/frontend/ast.rs`, add shape field to `KernelParam`:
```rust
pub struct KernelParam {
    pub name: syn::Ident,
    pub is_mut: bool,
    pub shape: Option<Vec<i64>>,  // extracted from {[...]} syntax, -1 for dynamic
}
```

In `sile-macros/src/frontend/parse.rs`, `parse_param` extracts the `{[m,n,k]}` shape from `Tensor<f32, {[...]}>` type annotations by matching through `syn::Type::Reference -> syn::Type::Path -> AngleBracketed args -> const expr -> ExprArray`.

In `sile-macros/src/frontend/lower.rs`, the extracted shape is lowered to `ShapeExpr` and passed into `Type::tensor(F32, shape_expr)` instead of always using `ShapeExpr::dynamic()`. Parse `i64` for `ShapeExpr::Constant` values.

In `sile-macros/src/lib.rs`:
- **Delete `rewrite_shape_expr` function** entirely (no longer generates DList types)
- **Rewrite `rewrite_type`** to produce `Tensor<f32>` with single type arg (drops rank param completely)

### 3. Unify to `i64`

- `ShapeExpr::Constant(i64)` (was `i32`)
- `Tensor::dim()` returns `i64`, takes `usize` index
- `Tensor::load_tile` shape params use `i64`
- `constant()` free function: `_shape: [i64; N]` (was `[i32; N]`)
- Free functions in `lib.rs` lose `R: Rank` bounds and use `Tile<f32>`:
  - `load_tile_like_2d(x: &Tensor<f32>, y: &Tensor<f32>) -> Tile<f32>`
  - `reduce_max(tile: Tile<f32>, axis: i32) -> Tile<f32>`
  - `reduce_sum(tile: Tile<f32>, axis: i32) -> Tile<f32>`
  - `exp(tile: Tile<f32>) -> Tile<f32>`
  - `constant<const N: usize>(value: f32, shape: [i64; N]) -> Tile<f32>`
  - `mma(a: Tile<f32>, b: Tile<f32>, c: Tile<f32>) -> Tile<f32>`
- `Expr::ScalarI32` stays as `i32` (for scalar values, not dimensions)
- SSA `immediates` already `i64`

### 4. Add `Tensor::random()`

Add `rand` dependency to `crates/sile/Cargo.toml`. New method on `Tensor<f32>`:
```rust
pub fn random(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self>
```

Uses `rand::Rng` to fill with random `f32` values in `[0.0, 1.0)`.

### 5. Fix matmul C codegen and CPU backend

The C codegen for `MatMul2D` (`codegen/c.rs`) uses hardcoded `acc[64][64]`. Fix to use VLA with runtime `bm`/`bn` parameters (C99 VLA is supported by `cc`/`clang`/`gcc` which are already required).

The current 3D branch generates monolithic C code ignoring `kernel.instructions`. Fix to use instruction-driven codegen via `generate_matmul_instruction`, or keep the template approach but parameterize it correctly.

Ensure `cpu_c.rs` passes correct shape parameters (including `bk`) to the generated kernel function.

### 6. Update matmul example

Use `Tensor::random()` for `a` and `b`. Compute a host-side reference matmul to verify correctness.

## Files Changed

| File | Change |
|------|--------|
| `crates/sile/src/tensor.rs` | Remove Rank/DList/DListNil, simplify Tensor, implement dim(), add random() |
| `crates/sile/src/tile.rs` | Remove Rank, simplify Tile |
| `crates/sile/src/lib.rs` | Remove feature flags, update exports, simplify free functions (remove R: Rank) |
| `crates/sile/Cargo.toml` | Add rand dependency |
| `crates/sile-macros/src/lib.rs` | Delete rewrite_shape_expr, rewrite rewrite_type to drop rank param |
| `crates/sile-macros/src/frontend/ast.rs` | Add shape field to KernelParam |
| `crates/sile-macros/src/frontend/parse.rs` | Extract {[...]} shape from Tensor type |
| `crates/sile-macros/src/frontend/lower.rs` | Use extracted shape in HIR, parse i64 for ShapeExpr::Constant |
| `crates/sile/src/hir/types.rs` | ShapeExpr::Constant(i64) |
| `crates/sile/src/codegen/c.rs` | Parameterized matmul C generation with VLA |
| `crates/sile/src/backend/cpu_c.rs` | Fix matmul argument passing |
| `crates/sile/examples/matmul.rs` | Use random, verify results with host reference |
| `crates/sile/tests/hir_shape_model.rs` | Update ShapeExpr::constant calls for i64 |
