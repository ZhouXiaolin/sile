pub mod backend;
pub mod backend_ir;
pub mod codegen;
pub mod device;
pub mod error;
pub mod hir;
pub mod kernel;
pub mod passes;
pub mod schedule;
pub mod spec;
pub mod ssa;
pub mod stream;
pub mod tensor;
pub mod tile;
pub mod typeck;

pub use device::Device;
pub use error::{Error, Result};
pub use kernel::{KernelArg, KernelLauncher, LaunchConfig};
pub use sile_macros::kernel;
pub use spec::{
    BinaryOp, KernelSpec, KernelSpecRef, Node, NodeRef, Param, ParamKind, ParamRef, ScalarType,
    Shape, Store, StoreRef, TileExpr,
};
pub use stream::Stream;
pub use tensor::Tensor;
