pub use sile_core as core;
pub use sile_hir as hir;
pub use sile_macros::kernel;

pub use sile_core::{Device, Error, KernelArg, LaunchConfig, Result, Stream};
pub use sile_hir::{
    BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type, ValueCategory,
};

pub mod tensor;
pub mod tile;

pub use tensor::Tensor;
pub use tile::Tile;

pub mod typeck {
    pub use sile_hir::typeck::*;
}
pub mod mir {
    pub use sile_mir::ir::*;
    pub use sile_mir::lower::lower_to_mir;
    pub use sile_mir::print::format_mir;
}
pub mod llir {
    pub use sile_llir::*;
}
pub mod compiler;
pub mod codegen {
    pub mod llir_c {
        pub use sile_backend::cpu::codegen_llir_c::*;
    }
    pub mod llir_metal {
        pub use sile_backend::metal::codegen_llir_metal::*;
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

pub fn load_tile_like_2d(_x: &Tensor<f32>, y: &Tensor<f32>) -> Tile<f32> {
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
