pub mod ir;
pub mod lower;
pub mod passes;
pub mod print;

pub use ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CastOp, CmpPred, Constant, Function,
    Inst, InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, SCALARS_I64_PARAM_NAME,
    SCALARS_PARAM_NAME, SHAPES_PARAM_NAME, SyncScope, Terminator, TileReduceOp, Type, ValueId,
};
pub use lower::lower_module_to_llvm_ir;
pub use passes::{
    ACTIVE_PIPELINE as ACTIVE_LLVM_IR_PIPELINE, LlvmIrPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_LLVM_IR_PIPELINE,
    run_default_pipeline as run_llvm_ir_passes, run_pipeline as run_llvm_ir_pipeline,
};
pub use print::{format_function, format_llvm_ir};
