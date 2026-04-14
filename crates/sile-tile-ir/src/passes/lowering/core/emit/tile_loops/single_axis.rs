use sile_llvm_ir as llvm_ir;

use crate::TileIrFunction;
use crate::passes::lowering::core::block::{BlockLowerer, LowerLlvmIrCtx};
use crate::passes::lowering::core::emit::insts::{const_i64, emit_bin, emit_cmp};

/// Threshold above which reduce extent uses an LLVMIR loop instead of full unrolling.
pub(crate) const REDUCE_UNROLL_THRESHOLD: i64 = 16;

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

/// Emits an LLVMIR loop that iterates over the reduce dimension with a loop-carried accumulator.
///
/// The loop structure is:
///   current → header(loop_idx, acc) → [cond_br] → body(loop_idx, acc) → header
///                                                 → exit(acc) → (back to caller)
///
/// The `body` closure receives the current loop index and the accumulator value,
/// and must return the updated accumulator.
pub(crate) fn lower_reduce_extent_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    reduce_extent: i64,
    start_idx: i64,
    init_acc: llvm_ir::Operand,
    mut body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand, // current loop idx
        llvm_ir::Operand, // current acc
    ) -> llvm_ir::Operand, // returns updated acc
) -> llvm_ir::Operand {
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_header"),
        vec![
            ("reduce_idx", llvm_ir::Type::I64),
            ("reduce_acc", llvm_ir::Type::F32),
        ],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![
            ("reduce_idx", llvm_ir::Type::I64),
            ("reduce_acc", llvm_ir::Type::F32),
        ],
    );
    let (exit_block, exit_params) = builder.create_block(
        &format!("{prefix}_exit"),
        vec![("final_acc", llvm_ir::Type::F32)],
    );

    // Jump from current block into the loop header with start_idx and initial acc
    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: header,
        args: vec![const_i64(start_idx), init_acc],
    });

    // Header: check idx < reduce_extent
    builder.switch_to(header);
    let header_idx = llvm_ir::Operand::Value(header_params[0].id);
    let header_acc = llvm_ir::Operand::Value(header_params[1].id);
    let header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            header_idx.clone(),
            const_i64(reduce_extent),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![header_idx.clone(), header_acc.clone()],
            false_target: exit_block,
            false_args: vec![header_acc],
        }
    });
    builder.set_current_terminator(header_term);

    // Body: call the closure with current idx and acc, then increment idx and loop back
    builder.switch_to(body_block);
    let body_idx = llvm_ir::Operand::Value(body_params[0].id);
    let body_acc = llvm_ir::Operand::Value(body_params[1].id);
    let latch_term = builder.with_current_insts(|ctx, tile_ir, out| {
        let new_acc = body(ctx, tile_ir, out, body_idx.clone(), body_acc);
        let next_idx = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_idx,
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_idx, new_acc],
        }
    });
    builder.set_current_terminator(latch_term);

    // Exit: the final accumulator value
    builder.switch_to(exit_block);
    llvm_ir::Operand::Value(exit_params[0].id)
}

/// Emits an LLVMIR loop with a configurable step over the reduce dimension.
///
/// Identical to [`lower_reduce_extent_loop`] except the increment is `step`
/// instead of 1. The loop runs from `start_idx` to `reduce_extent` (exclusive)
/// with step `step`. Each iteration calls `body(idx, acc)` → updated acc.
pub(crate) fn lower_reduce_extent_loop_step(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    reduce_extent: i64,
    start_idx: i64,
    step: i64,
    init_acc: llvm_ir::Operand,
    mut body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand, // current loop idx
        llvm_ir::Operand, // current acc
    ) -> llvm_ir::Operand, // returns updated acc
) -> llvm_ir::Operand {
    debug_assert!(step >= 1);
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_header"),
        vec![
            ("reduce_idx", llvm_ir::Type::I64),
            ("reduce_acc", llvm_ir::Type::F32),
        ],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![
            ("reduce_idx", llvm_ir::Type::I64),
            ("reduce_acc", llvm_ir::Type::F32),
        ],
    );
    let (exit_block, exit_params) = builder.create_block(
        &format!("{prefix}_exit"),
        vec![("final_acc", llvm_ir::Type::F32)],
    );

    // Jump from current block into the loop header with start_idx and initial acc
    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: header,
        args: vec![const_i64(start_idx), init_acc],
    });

    // Header: check idx < reduce_extent
    builder.switch_to(header);
    let header_idx = llvm_ir::Operand::Value(header_params[0].id);
    let header_acc = llvm_ir::Operand::Value(header_params[1].id);
    let header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            header_idx.clone(),
            const_i64(reduce_extent),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![header_idx.clone(), header_acc.clone()],
            false_target: exit_block,
            false_args: vec![header_acc],
        }
    });
    builder.set_current_terminator(header_term);

    // Body: call the closure with current idx and acc, then increment idx by step and loop back
    builder.switch_to(body_block);
    let body_idx = llvm_ir::Operand::Value(body_params[0].id);
    let body_acc = llvm_ir::Operand::Value(body_params[1].id);
    let latch_term = builder.with_current_insts(|ctx, tile_ir, out| {
        let new_acc = body(ctx, tile_ir, out, body_idx.clone(), body_acc);
        let next_idx = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_idx,
            const_i64(step),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_idx, new_acc],
        }
    });
    builder.set_current_terminator(latch_term);

    // Exit: the final accumulator value
    builder.switch_to(exit_block);
    llvm_ir::Operand::Value(exit_params[0].id)
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
