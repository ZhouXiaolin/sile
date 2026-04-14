use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, const_i64, emit_bin, emit_cmp, emit_gep,
    emit_select, emit_store, load_tile_scalar_dynamic, lower_nested_tile_loop,
    lower_reduce_extent_loop, resolve_operand, REDUCE_UNROLL_THRESHOLD,
};

pub(crate) fn reduce_combine(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    acc: llvm_ir::Operand,
    next: llvm_ir::Operand,
    is_max: bool,
) -> llvm_ir::Operand {
    if is_max {
        let is_gt = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Ogt,
            next.clone(),
            acc.clone(),
            llvm_ir::Type::I1,
        );
        emit_select(ctx, out, is_gt, next, acc, llvm_ir::Type::F32)
    } else {
        emit_bin(ctx, out, llvm_ir::BinOp::Add, acc, next, llvm_ir::Type::F32)
    }
}

pub(crate) fn lower_tile_reduce_inst(
    result: ValueId,
    value: ValueId,
    is_max: bool,
    axis: i64,
    in_rows: i64,
    in_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if axis == 1 {
        (in_rows, 1)
    } else {
        (1, in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(value, builder.ctx());
    let reduce_extent = if axis == 1 { in_cols } else { in_rows };
    let prefix = format!("tile_reduce_{}", result.0);

    if reduce_extent <= REDUCE_UNROLL_THRESHOLD {
        // Small extent: fully unroll (original behavior)
        lower_nested_tile_loop(
            builder,
            prefix.as_str(),
            out_rows,
            out_cols,
            move |ctx, _, out, row, col| {
                let mut acc = load_tile_scalar_dynamic(
                    ctx,
                    out,
                    src_tile.clone(),
                    if axis == 1 { row.clone() } else { const_i64(0) },
                    if axis == 1 { const_i64(0) } else { col.clone() },
                );

                for idx in 1..reduce_extent {
                    let next = load_tile_scalar_dynamic(
                        ctx,
                        out,
                        src_tile.clone(),
                        if axis == 1 {
                            row.clone()
                        } else {
                            const_i64(idx)
                        },
                        if axis == 1 {
                            const_i64(idx)
                        } else {
                            col.clone()
                        },
                    );
                    acc = reduce_combine(ctx, out, acc, next, is_max);
                }

                let dst_ptr = emit_gep(
                    ctx,
                    out,
                    dst_tile.clone(),
                    vec![row, col],
                    llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
                );
                emit_store(out, dst_ptr, acc);
            },
        );
    } else {
        // Large extent: use LLVMIR loop over the reduce dimension.
        // We build everything manually with BlockLowerer because lower_nested_tile_loop's
        // body closure cannot create new blocks.
        lower_tile_reduce_loop(
            builder,
            &prefix,
            out_rows,
            out_cols,
            reduce_extent,
            axis,
            is_max,
            src_tile,
            dst_tile,
        );
    }
}

/// Builds a complete tile loop with an inner reduce LLVMIR loop.
/// Used when reduce_extent > REDUCE_UNROLL_THRESHOLD.
fn lower_tile_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    out_rows: i64,
    out_cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    src_tile: llvm_ir::Operand,
    dst_tile: llvm_ir::Operand,
) {
    // Build a single-axis or 2D tile loop manually, with reduce loop inside body.
    if out_rows == 1 {
        lower_single_col_reduce_loop(
            builder,
            prefix,
            out_cols,
            reduce_extent,
            axis,
            is_max,
            src_tile,
            dst_tile,
        );
    } else if out_cols == 1 {
        lower_single_row_reduce_loop(
            builder,
            prefix,
            out_rows,
            reduce_extent,
            axis,
            is_max,
            src_tile,
            dst_tile,
        );
    } else {
        lower_full_2d_reduce_loop(
            builder,
            prefix,
            out_rows,
            out_cols,
            reduce_extent,
            axis,
            is_max,
            src_tile,
            dst_tile,
        );
    }
}

/// Single col loop (rows=1) with inner reduce LLVMIR loop.
fn lower_single_col_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    src_tile: llvm_ir::Operand,
    dst_tile: llvm_ir::Operand,
) {
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![("loop_col", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_col_body"),
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
        let cond = emit_cmp(ctx, out, llvm_ir::CmpPred::Slt, col.clone(), const_i64(cols), llvm_ir::Type::I1);
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
    let row = const_i64(0);
    let col = body_col.clone();

    // Emit initial load for idx=0
    let init_acc = builder.with_current_insts(|ctx, _, out| {
        load_tile_scalar_dynamic(
            ctx,
            out,
            src_tile.clone(),
            if axis == 1 { row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { col.clone() },
        )
    });

    // Inner reduce loop (starts at idx=1 since idx=0 is already in init_acc)
    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1
        init_acc,
        |ctx, _, out, reduce_idx, acc| {
            let next = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                if axis == 1 { row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { col.clone() },
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    // Store result
    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_col],
        }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

/// Single row loop (cols=1) with inner reduce LLVMIR loop.
fn lower_single_row_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    src_tile: llvm_ir::Operand,
    dst_tile: llvm_ir::Operand,
) {
    let (header, header_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_row_body"),
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
        let cond = emit_cmp(ctx, out, llvm_ir::CmpPred::Slt, row.clone(), const_i64(rows), llvm_ir::Type::I1);
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
    let row = body_row.clone();
    let col = const_i64(0);

    // Emit initial load for idx=0
    let init_acc = builder.with_current_insts(|ctx, _, out| {
        load_tile_scalar_dynamic(
            ctx,
            out,
            src_tile.clone(),
            if axis == 1 { row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { col.clone() },
        )
    });

    // Inner reduce loop (starts at idx=1 since idx=0 is already in init_acc)
    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1
        init_acc,
        |ctx, _, out, reduce_idx, acc| {
            let next = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                if axis == 1 { row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { col.clone() },
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    // Store result
    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_row, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br {
            target: header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

/// Full 2D tile loop with inner reduce LLVMIR loop.
fn lower_full_2d_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    src_tile: llvm_ir::Operand,
    dst_tile: llvm_ir::Operand,
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

    // Row header
    builder.switch_to(row_header);
    let row = llvm_ir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(ctx, out, llvm_ir::CmpPred::Slt, row.clone(), const_i64(rows), llvm_ir::Type::I1);
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    // Col header
    builder.switch_to(col_header);
    let col_row = llvm_ir::Operand::Value(col_params[0].id);
    let col = llvm_ir::Operand::Value(col_params[1].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(ctx, out, llvm_ir::CmpPred::Slt, col.clone(), const_i64(cols), llvm_ir::Type::I1);
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    // Body: load initial element, run reduce loop, store result
    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let body_col = llvm_ir::Operand::Value(body_params[1].id);

    let init_acc = builder.with_current_insts(|ctx, _, out| {
        load_tile_scalar_dynamic(
            ctx,
            out,
            src_tile.clone(),
            if axis == 1 { body_row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { body_col.clone() },
        )
    });

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1
        init_acc,
        |ctx, _, out, reduce_idx, acc| {
            let next = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                if axis == 1 { body_row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { body_col.clone() },
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    let body_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![body_row.clone(), body_col.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br {
            target: col_header,
            args: vec![body_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(body_term);

    // Row latch
    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, latch_row.clone(), const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}
