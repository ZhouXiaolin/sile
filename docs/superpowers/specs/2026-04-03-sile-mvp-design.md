# Sile MVP Design Spec

## Goal

Build a vertical slice that gets `vec_add` working end-to-end: `#[sile::kernel]` macro parses the function, builds HIR, generates C code with OMP, compiles with clang, dynamically loads via libloading, and executes.

## Project Structure

```
sile/                          # Cargo workspace
├── Cargo.toml
├── crates/
│   ├── sile/                  # library crate (runtime + HIR + C codegen)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         # re-exports macro + runtime types
│   │       ├── tensor.rs      # Tensor, TensorArg trait
│   │       ├── device.rs      # Device enum, Backend trait
│   │       ├── stream.rs      # Stream (lightweight for MVP)
│   │       ├── hir.rs         # HirGraph, HirNode, ValueId
│   │       ├── codegen_c.rs   # HIR -> C code generation (with OMP)
│   │       ├── loader.rs      # CpuBackend: clang compile + libloading
│   │       └── error.rs       # Error type
│   └── sile-macros/           # proc-macro crate
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs         # #[sile::kernel] attribute macro
└── examples/
    └── vec_add.rs
```

## Error Type

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("compile failed: {0}")]
    Compile(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

## HIR (Minimal for vec_add)

```rust
type ValueId = u32;

struct HirGraph {
    nodes: Vec<HirNode>,
}

enum HirNode {
    Param { index: usize, ty: ScalarType, shape: Shape },
    LoadTile { param: usize, tile_id: TileIdExpr, shape: Shape },
    StoreTile { param: usize, tile_id: TileIdExpr, value: ValueId },
    BinaryOp { op: BinOp, lhs: ValueId, rhs: ValueId },
    Const { value: f64, shape: Shape },
}

enum BinOp { Add, Sub, Mul, Div }
enum ScalarType { F32 }  // MVP: f32 only
type Shape = Vec<i64>;   // MVP: all dimensions are positive integers (no -1)

struct TileIdExpr {
    components: Vec<TileIdComponent>,
}
enum TileIdComponent {
    Fixed(i64),
    GridAxis(usize),  // 0=x, 1=y, 2=z
}
```

Notes:
- `ScalarType` has only `F32` for MVP. Future: `F64`, `I32`, `I64`.
- Shape dimensions are always positive in MVP. Dynamic dimensions (`-1`) are a future feature.
- The HIR is embedded as a `const` in the generated code. Backends compile from HIR at runtime, leaving room for future backends (CUDA, WGPU, Metal).

## Tile Intrinsics Module

The `sile` crate provides a `tile` module with intrinsic functions/types. These are **not** real functions — the macro recognizes them by path and converts them to HIR nodes. However, they must exist as real Rust items so the function body type-checks before the macro rewrites it.

```rust
// sile::tile module (real items for type-checking, replaced by macro during HIR construction)

pub struct TileId(pub i64);

pub fn id() -> TileId {
    TileId(0)  // stub, never executed — macro consumes the call
}
```

The macro recognizes:
- `tile::id()` → captures the grid axis reference (maps to `GridAxis(0)` in MVP, i.e. x-axis)
- `a.load_tile(shape_expr, [tid_expr])` → `LoadTile` with param index derived from `a`'s position, tile_id from `tid_expr`, shape from `shape_expr`
- `c.store(value_expr)` → `StoreTile` with param index from `c`, tile_id inferred as **same tile_id as the last `tile::id()` call in scope**, value from `value_expr`

The tile_id for `store` is implicit: it matches the `tile::id()` captured earlier in the same kernel body. MVP kernels have a single tile ID per tile invocation (1D tiles mapped to grid x-axis).

## Macro Frontend (`#[sile::kernel]`)

Uses `darling` for attribute parsing and `syn` for function parsing.

**Parsing flow:**
1. `darling` parses attribute parameters (none for MVP, but extensible)
2. `syn` parses function signature: extract param names and types (`&Tensor<T>`, `&mut Tensor<T>`)
3. Walk function body, recognize primitives:
   - `tile::id()` → record current tile_id context
   - `<param>.load_tile(shape, [tid])` → `LoadTile` (param index from function signature position)
   - `<param>.store(value)` → `StoreTile` (tile_id from current context)
   - `a + b` → `BinaryOp(Add, ...)`
   - literal constants → `Const`
4. Build `HirGraph`
5. Generate Rust output with **fully-qualified paths** (e.g., `::sile::KernelLauncher`) to avoid namespace issues

**Generated code shape:**

```rust
pub fn vec_add<'a>(
    a: &'a ::sile::Tensor<f32>,
    b: &'a ::sile::Tensor<f32>,
    c: &'a mut ::sile::Tensor<f32>,
) -> ::sile::KernelLauncher<'a> {
    static HIR: ::sile::HirGraph = ::sile::HirGraph { /* const-evaluable */ };
    ::sile::KernelLauncher::new(
        &HIR,
        "vec_add",
        a,
        b,
        c,
    )
}
```

**KernelLauncher:**

```rust
pub struct KernelLauncher<'a> {
    hir: &'a HirGraph,
    name: &'static str,
    args: Vec<&'a dyn TensorArg>,
    grid: Option<(u32, u32, u32)>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(hir: &'a HirGraph, name: &'static str, args: Vec<&'a dyn TensorArg>) -> Self;
    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self;
    pub fn apply(self, stream: &Stream) -> Result<()> {
        let grid = self.grid.ok_or(Error::Backend("grid not set"))?;
        let backend = stream.device().backend();
        let kernel = backend.compile(self.name, self.hir, grid)?;
        backend.launch(&kernel, &self.args)
    }
}
```

Key design: `KernelLauncher` does **not** hold a `&dyn Backend`. It obtains the backend from `Stream::device()` at `apply()` time. This avoids lifetime issues and lets the user choose the backend at invocation time.

## Backend Abstraction

```rust
trait Backend: Send + Sync {
    type CompiledKernel;
    fn compile(&self, name: &str, hir: &HirGraph, grid: (u32, u32, u32)) -> Result<Self::CompiledKernel>;
    fn launch(&self, kernel: &Self::CompiledKernel, args: &[&dyn TensorArg]) -> Result<()>;
}
```

MVP implements only `CpuBackend`. The trait allows future `CudaBackend`, `WgpuBackend`, `MetalBackend` without changing `KernelLauncher`.

## C Codegen + Loader

**codegen_c.rs** generates a C function per kernel:

```c
#include <math.h>
#include <omp.h>
#include <string.h>

void sile_kernel_<name>(
    const float* restrict arg0,
    const float* restrict arg1,
    float* restrict arg2,
    const int64_t grid_x, const int64_t grid_y, const int64_t grid_z,
    const int64_t tile_size
) {
    #pragma omp parallel for collapse(3)
    for (int64_t gz = 0; gz < grid_z; gz++)
    for (int64_t gy = 0; gy < grid_y; gy++)
    for (int64_t gx = 0; gx < grid_x; gx++) {
        int64_t tid = gx + gy * grid_x + gz * grid_x * grid_y;
        // LoadTile -> memcpy from global + tid * tile_size to stack array
        // BinaryOp -> loop over tile elements
        // StoreTile -> memcpy from stack array to global + tid * tile_size
    }
}
```

**tile_size derivation**: extracted from the `LoadTile` node's `shape` field in the HIR. For vec_add, `shape = [4]`, so `tile_size = 4`. MVP assumes all tiles in a kernel have the same tile shape. This is validated during HIR construction.

**C type selection**: MVP generates `const float*` / `float*` for all params. The HIR `ScalarType::F32` confirms this. Future backends will map `ScalarType` to the appropriate C type.

**loader.rs** (`CpuBackend`):

```rust
struct CpuCompiledKernel {
    lib: Library,  // libloading::Library — keeps .so loaded
    // Type-erased function pointer: C function has variable arg count,
    // so we store raw pointer and reconstruct the call in launch().
    func_ptr: unsafe extern "C" fn(),
}

impl Backend for CpuBackend {
    type CompiledKernel = CpuCompiledKernel;

    fn compile(&self, name: &str, hir: &HirGraph, grid: (u32, u32, u32)) -> Result<CpuCompiledKernel> {
        let c_code = codegen_c::generate(name, hir);
        let hash = compute_hash(name, hir);  // SHA-256 of (name + serialized HIR)
        // 1. Check cache_dir/<hash>.so
        // 2. If not cached: write .c -> clang -shared -fPIC -O3 -march=native -fopenmp -o <hash>.so <hash>.c -lm
        // 3. libloading::Library::new -> get symbol "sile_kernel_<name>"
    }

    fn launch(&self, kernel: &CpuCompiledKernel, args: &[&dyn TensorArg]) -> Result<()> {
        // Build argument array: [arg0.ptr, arg1.ptr, ..., argN.ptr, grid_x, grid_y, grid_z, tile_size]
        // Cast func_ptr to the correct signature and call
        // unsafe { transmute::<_, unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, i64, i64, i64)>(kernel.func_ptr)(...) }
    }
}
```

**Cache hash**: computed from `name` string + serialized `HirGraph` bytes (via `sha2::Sha256`). Does not include grid — the same kernel with different grids reuses the cached `.so` (grid is passed as a runtime parameter to the C function). This means one `.so` per kernel definition.

## Runtime

**Device:**

```rust
#[derive(Clone)]
pub enum Device {
    Cpu {
        cache_dir: PathBuf,
        #[cfg(hidden)]
        backend: CpuBackend,  // internal, constructed lazily
    },
    // Future: Cuda, Wgpu, Metal
}

impl Device {
    pub fn create_stream(&self) -> Result<Stream>;
    pub fn backend(&self) -> &dyn Backend;  // returns &self.internal_backend
}
```

`CpuBackend` is owned inside `Device::Cpu` and constructed when the variant is created. `Device` derives `Clone` — `CpuBackend` is cheaply cloneable (it only holds `cache_dir: PathBuf` and a `HashMap` for cached kernels wrapped in `Arc<Mutex<>>`).

**Tensor:**

```rust
pub struct Tensor<T> {
    data: Arc<Vec<T>>,
    shape: Vec<i64>,
}

impl Tensor<f32> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self>;
    pub fn ones(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self>;
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<i64>>) -> Result<Self>;
    pub fn to_vec(&self, _stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.data.clone().unwrap_or_else(|| {
            self.data.iter().copied().collect()
        }))
    }
}
```

`Tensor<T>` stores `Arc<Vec<T>>` — simple and correct. `T` is used by the `Vec<T>`. No unsafe pointer management needed for MVP. The `device` parameter on `zeros`/`ones` is accepted for API consistency but CPU tensors are just heap-allocated `Vec<f32>`.

**TensorArg trait** (for Backend::launch):

```rust
pub trait TensorArg {
    fn ptr(&self) -> *const u8;
    fn ptr_mut(&mut self) -> *mut u8;
    fn element_size(&self) -> usize;
    fn len(&self) -> usize;
}

impl TensorArg for Tensor<f32> {
    fn ptr(&self) -> *const u8 { self.data.as_ptr() as *const u8 }
    fn ptr_mut(&mut self) -> *mut u8 { self.data.as_ptr() as *mut u8 }  // Arc<Vec> needs interior mutability for mut refs
    fn element_size(&self) -> usize { 4 }
    fn len(&self) -> usize { self.data.len() }
}
```

Note: `TensorArg` for `&Tensor` and `&mut Tensor` will be implemented separately to handle the shared/mutable reference distinction.

**Stream:**

```rust
pub struct Stream {
    device: Device,
}

impl Stream {
    pub fn device(&self) -> &Device { &self.device }
    pub fn synchronize(&self) -> Result<()> { Ok(()) }  // C function is synchronous; OMP handles parallelism internally
}
```

## vec_add Example

```rust
// examples/vec_add.rs
use sile::{Tensor, Device, Stream};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = tile::id().0;
    let tile_a = a.load_tile([4], [tid]);
    let tile_b = b.load_tile([4], [tid]);
    c.store(tile_a + tile_b);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::Cpu { cache_dir: "/tmp/sile_cache".into() };
    let stream = device.create_stream()?;

    let a = Tensor::from_vec(vec![1.0f32; 16], [16])?;
    let b = Tensor::from_vec(vec![2.0f32; 16], [16])?;
    let mut c = Tensor::zeros([16], &device)?;

    vec_add(&a, &b, &mut c)
        .grid((4, 1, 1))
        .apply(&stream)?;

    let result = c.to_vec(&stream)?;
    assert_eq!(result, vec![3.0f32; 16]);
    Ok(())
}
```

## Execution Flow

1. **Compile time** — `#[sile::kernel]` macro parses `vec_add`, builds HIR (Param x3 + LoadTile x2 + BinaryOp(Add) + StoreTile), embeds as `static HirGraph`
2. **First call** — `apply()` -> `stream.device().backend()` gets `CpuBackend` -> `CpuBackend::compile()` -> `codegen_c` generates C code -> `clang -shared -fPIC -O3 -fopenmp` -> `libloading` loads -> caches `.so` (keyed by SHA-256 of name + HIR)
3. **Subsequent calls** — cache hit, skip compilation
4. **Execute** — `CpuBackend::launch()` -> calls C function pointer with raw pointers to a/b/c data + grid(4,1,1) + tile_size=4 -> OMP parallelizes 4 tiles
5. **Verify** — `to_vec()` returns data as `Vec<f32>`, assert correctness

## MVP Scope — Excluded

These are deferred to future iterations:
- Multi-dimensional tensors / strides / partition
- Reduce, Broadcast, Reshape, Select, UnaryOp HIR nodes
- softmax / dropout examples
- CUDA / WGPU / Metal backends
- Shape inference / optimization passes (CSE, DCE, constant folding)
- Generic scalar types beyond f32
- Dynamic shapes (`-1` dimension)

## Dependencies

- `sile-macros`: `syn`, `quote`, `proc-macro2`, `darling`
- `sile`: `libloading`, `thiserror`, `sha2` (for cache hashing)
