pub mod builder;
pub mod ir;
pub mod lower;

pub use ir::{
    BasicBlock, CmpOp, Constant, FloatType, Function, GlobalVariable, Instruction, IntegerType,
    Param, PhiNode, Program, Terminator, Type, Value,
};
pub use lower::lower_ssa_to_lir;
