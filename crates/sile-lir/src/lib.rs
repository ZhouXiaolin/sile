pub mod backend;
pub mod builder;
pub mod executable;
pub mod ir;

pub use backend::Backend;
pub use executable::{
    ExecutableKernel, KernelAbi, KernelParamAbi, LaunchSemantics, ParamPassing, ShapeLayout, ValueInfo,
    ValueInfoTable,
};
pub use ir::{
    BasicBlock, CmpOp, Constant, FloatType, Function, GlobalVariable, Instruction, IntegerType,
    Param, PhiNode, Program, Terminator, Type, Value,
};
pub use sile_hir::types::ElemType;
