use sile_llvm_ir as llvm_ir;

use crate::TileIrFunction;
use crate::passes::lowering::core::block::{BlockLowerer, LowerLlvmIrCtx};
use crate::passes::lowering::core::emit::insts::{const_i64, emit_bin, emit_cmp};

pub(super) fn lower_full_2d_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    mut body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand,
        llvm_ir::Operand,
    ),
) {
    let (row_header, row_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (col_header, col_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (row_latch, row_latch_params) = builder.create_block(
        &format!("{prefix}_row_latch"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row = llvm_ir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
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
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(col_header);
    let col_row = llvm_ir::Operand::Value(col_params[0].id);
    let col = llvm_ir::Operand::Value(col_params[1].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
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
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let body_col = llvm_ir::Operand::Value(body_params[1].id);
    let body_term = builder.with_current_insts(|ctx, tile_ir, out| {
        body(ctx, tile_ir, out, body_row.clone(), body_col.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_col.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: col_header,
            args: vec![body_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(body_term);

    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}
