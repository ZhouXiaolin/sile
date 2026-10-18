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
        IntoElement, ScalarBroadcast, TileView, get_tile_block_id, load_tile_mut, num_programs,
        num_tiles, program_id, Tile,
    };
    pub use crate::{
        broadcast_scalar, ceil_div, constant, convert_scalar, eq_tile, exp, exp2, ge_tile, gt_tile,
        iota, le_tile, log, log2, lt_tile, mma, ne_tile, reduce_max, reduce_min, reduce_sum,
        rsqrt, scalar_to_tile, select, sqrt, tile_to_scalar, true_div,
    };
    // CuTile writes a tile operation with a `_tile` suffix and the scalar one
    // without. Both names reach the same operation here.
    pub use crate::{max, max as max_tile, min, min as min_tile};
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
