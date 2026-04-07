use sile_llvm_ir as llvm_ir;

use crate::TileIrFunction;
use crate::passes::lowering::core::block::{BlockLowerer, LowerLlvmIrCtx};
use crate::passes::lowering::core::emit::insts::{const_i64, emit_bin, emit_cmp};

pub(super) fn lower_single_col_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    cols: i64,
    mut body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand,
        llvm_ir::Operand,
    ),
) {
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![("loop_col", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![("loop_col", llvm_ir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(header);
    let col = llvm_ir::Operand::Value(header_params[0].id);
    let header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            col.clone(),
            const_i64(cols),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![col.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_col = llvm_ir::Operand::Value(body_params[0].id);
    let body_term = builder.with_current_insts(|ctx, tile_ir, out| {
        body(ctx, tile_ir, out, const_i64(0), body_col.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_col.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_col],
        }
    });
    builder.set_current_terminator(body_term);
    builder.switch_to(continue_block);
}

pub(super) fn lower_single_row_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    mut body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand,
        llvm_ir::Operand,
    ),
) {
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(header);
    let row = llvm_ir::Operand::Value(header_params[0].id);
    let header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            row.clone(),
            const_i64(rows),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![row.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let body_term = builder.with_current_insts(|ctx, tile_ir, out| {
        body(ctx, tile_ir, out, body_row.clone(), const_i64(0));
        let next_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_row.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(body_term);
    builder.switch_to(continue_block);
}
