use sile_llvm_ir as llvm_ir;

use super::core::{LowerLlvmIrCtx, llvm_ir_block, resolve_operand};
use crate::ir::*;

pub(crate) fn lower_terminator(
    term: &TileIrTerminator,
    ctx: &LowerLlvmIrCtx,
) -> llvm_ir::Terminator {
    match term {
        TileIrTerminator::Jump { target, args } => llvm_ir::Terminator::Br {
            target: llvm_ir_block(*target),
            args: args.iter().map(|arg| resolve_operand(*arg, ctx)).collect(),
        },
        TileIrTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => llvm_ir::Terminator::CondBr {
            cond: resolve_operand(*cond, ctx),
            true_target: llvm_ir_block(*true_target),
            true_args: true_args
                .iter()
                .map(|arg| resolve_operand(*arg, ctx))
                .collect(),
            false_target: llvm_ir_block(*false_target),
            false_args: false_args
                .iter()
                .map(|arg| resolve_operand(*arg, ctx))
                .collect(),
        },
        TileIrTerminator::Return => llvm_ir::Terminator::Ret { value: None },
    }
}
