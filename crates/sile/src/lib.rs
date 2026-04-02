pub mod backend;
pub mod hir;
pub mod codegen;
pub mod device;
pub mod error;
pub mod kernel;
pub mod spec;
pub mod stream;
pub mod tensor;
pub mod tile;

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
