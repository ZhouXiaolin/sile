pub use sile_core as core;
pub use sile_hir as hir;
pub use sile_macros::kernel;

pub use sile_core::{Device, Error, KernelArg, LaunchConfig, Result, Stream};
pub use sile_hir::{
    BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type, ValueCategory,
};

pub mod tensor;
pub mod tile;

pub use tensor::{Partition, Tensor};
pub use tile::Tile;

/// The CuTile core vocabulary, re-exported under one name.
///
/// A kernel written against CuTile imports `cutile::core::*`. This module is
/// the same vocabulary under this crate's name, so a ported kernel changes the
/// import and nothing else.
pub mod ct {
    pub use crate::tile::{
        IntoElement, ScalarBroadcast, TileView, get_num_tile_blocks, get_tile_block_id,
        load_tile_mut, num_programs, num_tiles, program_id, Tile,
    };
    pub use crate::{
        broadcast_scalar, ceil_div, constant, convert_scalar, eq_tile, exp, exp2, ge_tile, gt_tile,
        iota, le_tile, log, log2, lt_tile, mma, ne_tile, reduce_max, reduce_min, reduce_sum,
        rsqrt, scalar_to_tile, select, sqrt, tile_to_scalar, true_div,
    };
    // CuTile writes a tile operation with a `_tile` suffix and the scalar one
    // without. Both names reach the same operation here.
    pub use crate::{max, max as max_tile, min, min as min_tile};
    // The arithmetic and math spellings CuTile names, each the operation its
    // plain name has; the mode arguments they carry drop in the parser.
    pub use crate::{
        absf, absi, addf, addi, andi, atan2, ceil, cmpf, cmpi, cos, cosh, divf, divi, exp_with_rounding,
        fma, floor, ftof, get_shape_dim, load_tile, maxf, maxi, minf, mini, mulf, muli, negf,
        negi, noti, ori, pow, remf, remi, shli, shri, sin, sinh, store_tile, subf, subi, tan,
        tanh, tanh_with_rounding, xori,
    };
    // The policy namespaces a kernel names modes and predicates with. The
    // parser drops a value from one of these, so the types only have to exist.
    pub mod ftz {
        pub struct Disabled;
        pub struct Enabled;
    }
    pub mod rounding {
        pub struct Zero;
        pub struct Nearest;
    }
    pub mod ordering {
        pub struct Weak;
        pub struct Strong;
    }
    pub mod scope {
        pub struct TileBlock;
        pub struct Cta;
        pub struct Gpu;
    }
    pub mod nan {
        pub struct Enabled;
        pub struct Disabled;
    }
    pub mod fast_acc {
        pub struct Enabled;
        pub struct Disabled;
    }
    pub mod saturating {
        pub struct Enabled;
        pub struct Disabled;
    }
    pub mod overflow {
        pub struct None;
        pub struct NoSignedWrap;
        pub struct NoUnsignedWrap;
        pub struct NoWrap;
    }
    pub mod predicate {
        pub struct Equal;
        pub struct NotEqual;
        pub struct LessThan;
        pub struct LessThanOrEqual;
        pub struct GreaterThan;
        pub struct GreaterThanOrEqual;
    }
    pub mod cmp_ordering {
        pub struct Unordered;
        pub struct Ordered;
    }
    pub mod reverse {
        pub struct Forward;
        pub struct Reverse;
    }
    pub mod signedness {
        pub struct Signed;
        pub struct Unsigned;
    }
    pub use crate::{const_shape, shape};
}

pub mod typeck {
    pub use sile_hir::typeck::*;
}
pub mod tileir {
    pub use sile_tile_ir::ir::*;
    pub use sile_tile_ir::{
        TileIrBlock, TileIrFunction, TileIrInst, TileIrOp, TileIrParam, TileIrTerminator,
        TileIrType, format_tile_ir, lower_to_tile_ir,
    };
}
pub mod llvmir {
    pub use sile_llvm_ir::{
        AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CastOp, CmpPred, Constant, Function,
        Inst, InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, SyncScope, Terminator, Type,
        ValueId, format_llvm_ir,
    };
}
pub mod compiler;
pub mod codegen {
    pub mod llvmir_c {
        pub use sile_backend::cpu::codegen::*;
    }
    pub mod llvmir_metal {
        pub use sile_backend::metal::codegen::*;
    }
    pub mod llvmir_wgsl {
        pub use sile_backend::wgsl::codegen::*;
    }
}
pub mod schedule {
    pub fn require_divisible(total: i64, tile: i64) -> sile_core::Result<()> {
        if total % tile == 0 {
            Ok(())
        } else {
            Err(sile_core::Error::Shape(format!(
                "shape {total} is not divisible by tile {tile}"
            )))
        }
    }
}

pub mod kernel_launcher;
pub use kernel_launcher::KernelLauncher;

pub fn reduce_max(tile: Tile<f32>, axis: i64) -> Tile<f32> {
    tile.reduced_shape(axis)
}

pub fn reduce_sum(tile: Tile<f32>, axis: i64) -> Tile<f32> {
    tile.reduced_shape(axis)
}

pub fn reduce_min(tile: Tile<f32>, axis: i64) -> Tile<f32> {
    tile.reduced_shape(axis)
}

pub fn sqrt(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn rsqrt(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn exp2(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn log(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn log2(tile: Tile<f32>) -> Tile<f32> {
    tile
}

/// `tile_to_scalar(t)`: the one element a one-element tile holds.
pub fn tile_to_scalar(_tile: Tile<f32>) -> f32 {
    0.0
}

/// `scalar_to_tile(s)`: a rank-zero tile holding `s`.
pub fn scalar_to_tile(_value: f32) -> Tile<f32> {
    Tile::new([0i64; 0])
}

/// `convert_scalar(x)`: a scalar constant as an element value.
pub fn convert_scalar(value: i64) -> f32 {
    value as f32
}

/// `broadcast_scalar(s, shape)`: the free-function spelling of `.broadcast`.
///
/// The scalar may be an element or a count: `broadcast_scalar(j * BN, shape)`
/// splats a coordinate across the tile it indexes.
pub fn broadcast_scalar<V: crate::tile::IntoElement, const N: usize>(
    _value: V,
    _shape: [i64; N],
) -> Tile<f32> {
    Tile::new(_shape)
}

/// `iota(shape)`: the sequence `0, 1, …, N-1` across a rank-one shape.
///
/// The host body reads only the shape, because the sequence itself is the
/// coordinate each element is written at and no value it computes is read.
pub fn iota<const N: usize>(shape: [i64; N]) -> Tile<f32> {
    Tile::new(shape)
}

/// `gt_tile(a, b)`: a tile of `0.0`/`1.0` from an elementwise comparison.
pub fn gt_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn ge_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn lt_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn le_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn eq_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn ne_tile(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

/// `select(cond, a, b)`: `a` where `cond` is non-zero, else `b`.
pub fn select(_cond: Tile<f32>, on_true: Tile<f32>, _on_false: Tile<f32>) -> Tile<f32> {
    on_true
}

/// `max(a, b)` and `max_tile(a, b)`: the larger of two values.
///
/// CuTile spells the scalar and the tile operation apart. One name serves both
/// here, because the operands decide which one it is.
pub fn max<A: crate::tile::Extremum>(a: A, b: A) -> A {
    a.max_of(b)
}

/// `min(a, b)` and `min_tile(a, b)`: the smaller of two values.
pub fn min<A: crate::tile::Extremum>(a: A, b: A) -> A {
    a.min_of(b)
}

/// `ceil_div(a, b)`: `a` over `b`, rounded up. CuTile writes it for counts.
pub fn ceil_div<A>(a: A, b: A) -> A
where
    A: Copy + From<u8> + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + std::ops::Div<Output = A>,
{
    (a + b - A::from(1)) / b
}

pub fn exp(tile: Tile<f32>) -> Tile<f32> {
    tile.exp()
}

/// `constant(v, shape)`: a tile of `v`. The shape may be an array literal or
/// another tile's `.shape()`, so it takes anything that reads as `[i64]`.
pub fn constant<V: crate::tile::IntoElement>(_value: V, _shape: impl AsRef<[i64]>) -> Tile<f32> {
    Tile::new(_shape)
}

/// `true_div(a, b)`: a division that does not truncate.
///
/// Generic over what divides, because CuTile writes it for a tile and for a
/// scalar alike. The kernel parser turns the call into the same division the
/// `/` operator produces, so this only has to exist for the host body.
pub fn true_div<A: std::ops::Div<Output = A>>(a: A, b: A) -> A {
    a / b
}

/// `mma(a, b, c)`: the accumulator's shape, since that is what it extends.
pub fn mma(_a: Tile<f32>, _b: Tile<f32>, c: Tile<f32>) -> Tile<f32> {
    c
}

// The CuTile arithmetic spellings. Each one names the operation its plain name
// has, and the mode arguments it carries (`rounding::`, `ftz::`, `overflow::`)
// are policies this IR does not model: the parser drops them, and the operands
// decide whether the operation is a tile one or a scalar one.

/// `addf(a, b)`, and `addi` in its scalar spelling: one addition.
pub fn addf<A: std::ops::Add<Output = A>>(a: A, b: A) -> A {
    a + b
}

pub fn subf<A: std::ops::Sub<Output = A>>(a: A, b: A) -> A {
    a - b
}

pub fn mulf<A: std::ops::Mul<Output = A>>(a: A, b: A) -> A {
    a * b
}

pub fn divf<A: std::ops::Div<Output = A>>(a: A, b: A) -> A {
    a / b
}

pub fn remf<A: std::ops::Rem<Output = A>>(a: A, b: A) -> A {
    a % b
}

pub use addf as addi;
pub use subf as subi;
pub use mulf as muli;
pub use divf as divi;
pub use remf as remi;

/// `negi(x)`: one negation of a count.
pub fn negi<A: std::ops::Neg<Output = A>>(a: A) -> A {
    -a
}

/// `fma(a, b, c)`: one multiply and one add. The fused form is not spelled the
/// same way on every target this backend writes, so the two steps are what
/// keeps the three targets bit-identical.
pub fn fma<A: std::ops::Mul<Output = A> + std::ops::Add<Output = A>>(a: A, b: A, c: A) -> A {
    a * b + c
}

/// `maxf`/`minf` and the integer `maxi`/`mini`: the extremes `max` and `min`
/// already name, over tiles and scalars alike.
pub use crate::{max as maxf, max as maxi, min as minf, min as mini};

/// `absi(x)`: the larger of `x` and `-x`. The comparison form keeps the count
/// kind of its operand; the float absolute value is `absf`.
pub fn absi<A: Copy + PartialOrd + std::ops::Neg<Output = A>>(a: A) -> A {
    let negated = -a;
    if a < negated {
        negated
    } else {
        a
    }
}

/// The count bitwise operations. A mask has no exclusive or and no shift in
/// this IR, so these are scalar operations; tiles combine with `&` instead.
pub fn andi(a: i64, b: i64) -> i64 {
    a & b
}

pub fn ori(a: i64, b: i64) -> i64 {
    a | b
}

pub fn xori(a: i64, b: i64) -> i64 {
    a ^ b
}

/// `noti(x)`: the complement, which is the exclusive or with all ones.
pub fn noti(a: i64) -> i64 {
    !a
}

pub fn shli(a: i64, b: i64) -> i64 {
    a << b
}

pub fn shri(a: i64, b: i64) -> i64 {
    a >> b
}

/// `cmpi`/`cmpf(lhs, rhs, predicate::…)` names its comparison with a predicate
/// value the parser reads; the host body keeps only the operands.
pub fn cmpi(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

pub fn cmpf(lhs: Tile<f32>, _rhs: Tile<f32>) -> Tile<f32> {
    lhs
}

// The float functions. Each lowers to one intrinsic the three targets spell,
// so the host body only has to make the name exist and hand the tile on.
//
// A tile is a value of the element the kernel computes, and the host body
// never reads the values it holds, so returning the tile unchanged is what
// every identity stub here does.

pub fn absf(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn negf(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn ceil(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn floor(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn sin(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn cos(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn tan(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn sinh(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn cosh(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub fn tanh(tile: Tile<f32>) -> Tile<f32> {
    tile
}

pub use crate::{exp as exp_with_rounding, tanh as tanh_with_rounding};

/// `pow(a, b)`, elementwise. The exponent is the scalar a kernel writes; a
/// tile exponent lowers to the same broadcast the scalar one does, and the
/// host body only has to make the name exist.
pub fn pow(a: Tile<f32>, _b: f32) -> Tile<f32> {
    a
}

/// `atan2(y, x)`, elementwise.
pub fn atan2(y: Tile<f32>, _x: Tile<f32>) -> Tile<f32> {
    y
}

/// `ftof(x)`: a float-to-float conversion, which this IR's one element type
/// makes a no-op the name carries for CuTile's spelling.
pub fn ftof(tile: Tile<f32>) -> Tile<f32> {
    tile
}

/// `get_shape_dim(shape, dim)`: one extent of a shape.
pub fn get_shape_dim(shape: impl AsRef<[i64]>, dim: i64) -> i64 {
    shape.as_ref()[dim as usize]
}

/// `load_tile(x, shape, idx)`: the free-function spelling of the tensor form.
pub fn load_tile<const N: usize, const M: usize>(
    tensor: &Tensor<f32>,
    tile_shape: [i64; N],
    indices: [i64; M],
) -> Tile<f32> {
    tensor.load_tile(tile_shape, indices)
}

/// `store_tile(y, result)`: the free-function spelling of `y.store(result)`.
pub fn store_tile(tensor: &mut Tensor<f32>, value: Tile<f32>) {
    tensor.store(value)
}
