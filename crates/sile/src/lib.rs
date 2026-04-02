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
pub use hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type, ValueCategory};
pub use kernel::{KernelArg, KernelLauncher, LaunchConfig};
pub use sile_macros::kernel;
pub use stream::Stream;
pub use tensor::Tensor;
