pub mod ir;
pub mod passes;
pub mod print;

pub use ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CastOp, CmpPred, Constant, Function,
    Inst, InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, SyncScope, Terminator, Type,
    ValueId,
};
pub use passes::{
    ACTIVE_PIPELINE as ACTIVE_LLIR_PIPELINE, LlirPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_LLIR_PIPELINE, run_default_pipeline as run_llir_passes,
    run_pipeline as run_llir_pipeline,
};
pub use print::format_function;
