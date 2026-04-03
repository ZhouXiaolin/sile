pub mod backend;
pub mod codegen;
pub mod device;
pub mod error;
pub mod hir;
pub mod kernel;
pub mod lir;
pub mod passes;
pub mod schedule;
pub mod scheduling;
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
pub use tensor::Tensor;
pub use tile::Tile;

// Free functions for kernel context
pub fn load_tile_like_2d(x: &Tensor<f32>, y: &Tensor<f32>) -> Tile<f32> {
    Tile::new(y.shape().to_vec())
}

pub fn reduce_max(tile: Tile<f32>, _axis: i64) -> Tile<f32> {
    Tile::new(vec![tile.shape[0]])
}

pub fn reduce_sum(tile: Tile<f32>, _axis: i64) -> Tile<f32> {
    Tile::new(vec![tile.shape[0]])
}

pub fn exp(tile: Tile<f32>) -> Tile<f32> {
    tile.exp()
}

pub fn constant<const N: usize>(_value: f32, _shape: [i64; N]) -> Tile<f32> {
    Tile::new(_shape.iter().copied().collect())
}

pub fn mma(_a: Tile<f32>, _b: Tile<f32>, _c: Tile<f32>) -> Tile<f32> {
    Tile::new(vec![])
}
