pub mod ir;
pub mod passes;
pub mod print;

pub use ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CastOp, CmpPred, Constant, Function,
    Inst, InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, SyncScope, Terminator, Type,
    ValueId,
};
pub use passes::{
    ACTIVE_PIPELINE as ACTIVE_LLVM_IR_PIPELINE, LlvmIrPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_LLVM_IR_PIPELINE,
    run_default_pipeline as run_llvm_ir_passes, run_pipeline as run_llvm_ir_pipeline,
};
pub use print::{format_function, format_llvm_ir};
