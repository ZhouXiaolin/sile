#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod backend;
pub mod backend_ir;
pub mod codegen;
pub mod device;
pub mod error;
pub mod hir;
pub mod kernel;
pub mod passes;
pub mod schedule;
#[deprecated = "use crate::hir::Kernel and compiler pipeline instead"]
pub mod spec;
pub mod ssa;
pub mod stream;
pub mod tensor;
pub mod tile;
pub mod typeck;

pub use device::Device;
pub use error::{Error, Result};
pub use hir::{
    BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type, ValueCategory,
};
pub use kernel::{KernelArg, KernelLauncher, LaunchConfig};
pub use sile_macros::kernel;
pub use stream::Stream;
pub use tensor::{unpartition, DList, DListNil, Partition, Rank, Tensor};
pub use tile::Tile;

// Free functions for kernel context (softmax example)
pub fn load_tile_like_2d<R: Rank>(x: &Tensor<f32>, y: &Tensor<f32, R>) -> Tile<f32, R> {
    Tile::new(y.shape().to_vec())
}

pub fn reduce_max<R: Rank>(tile: Tile<f32, R>, _axis: i32) -> Tile<f32, DListNil> {
    Tile::new(vec![tile.shape[0]])
}

pub fn reduce_sum<R: Rank>(tile: Tile<f32, R>, _axis: i32) -> Tile<f32, DListNil> {
    Tile::new(vec![tile.shape[0]])
}

pub fn exp<R: Rank>(tile: Tile<f32, R>) -> Tile<f32, R> {
    tile.exp()
}
