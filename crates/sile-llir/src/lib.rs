pub mod ir;
pub mod print;

pub use ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CastOp, CmpPred, Constant, Function,
    Inst, InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, SyncScope, Terminator, Type,
    ValueId,
};
pub use print::format_function;
