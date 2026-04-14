use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::TileMapExpr;
use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_f32, const_i64, emit_bin,
    emit_cmp, emit_gep, emit_load, emit_store, lower_nested_tile_loop,
    lower_reduce_extent_loop, lower_reduce_extent_loop_step, resolve_operand, REDUCE_UNROLL_THRESHOLD,
};
use crate::passes::lowering::tile_compute::reduce::reduce_combine;
use crate::passes::lowering::tile_expr::{collect_map_load_bases, eval_map_expr_scalar};
use crate::passes::lowering::tile_memory::{BufferIndexBase, build_buffer_index_base, emit_buffer_element_ptr, emit_buffer_linear_index};

pub(crate) struct LoadReduceFusion {
    pub(crate) buf: ValueId,
    pub(crate) row_coord: ValueId,
    pub(crate) col_coord: ValueId,
    pub(crate) src_rows: i64,
    pub(crate) src_cols: i64,
    pub(crate) stride_shape_idx: usize,
    pub(crate) is_max: bool,
    pub(crate) axis: i64,
    pub(crate) in_rows: i64,
    pub(crate) in_cols: i64,
}

pub(crate) fn lower_fused_load_reduce_inst(
    result: ValueId,
    fusion: &LoadReduceFusion,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if fusion.axis == 1 {
        (fusion.in_rows, 1)
    } else {
        (1, fusion.in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);

    let buf_operand = resolve_operand(fusion.buf, builder.ctx());
    let row_operand = resolve_operand(fusion.row_coord, builder.ctx());
    let col_operand = resolve_operand(fusion.col_coord, builder.ctx());
    let rank = buffer_rank_of(fusion.buf, builder.tile_ir());
    let index_base = builder.with_current_insts(|ctx, _, out| {
        build_buffer_index_base(
            ctx,
            out,
            buf_operand.clone(),
            row_operand.clone(),
            col_operand.clone(),
            fusion.src_rows,
            fusion.src_cols,
            fusion.stride_shape_idx,
            rank,
        )
    });

    let reduce_extent = if fusion.axis == 1 {
        fusion.src_cols
    } else {
        fusion.src_rows
    };
    let prefix = format!("tile_fused_load_reduce_{}", result.0);

    if reduce_extent <= REDUCE_UNROLL_THRESHOLD {
        // Small extent: fully unroll (original behavior)
        lower_nested_tile_loop(
            builder,
            prefix.as_str(),
            out_rows,
            out_cols,
            {
                let buf_operand = buf_operand.clone();
                let index_base = index_base.clone();
                let dst_tile = dst_tile.clone();
                let axis = fusion.axis;
                let is_max = fusion.is_max;

                move |ctx, _, out, row, col| {
                    // Use vectorized path for sum-reduce along axis=1
                    // (contiguous memory), when extent >= 4.
                    let use_vec = !is_max && axis == 1 && reduce_extent >= 4;

                    let acc = if use_vec {
                        let vec_chunks = (reduce_extent / 4) as usize;
                        let base_offset = emit_buffer_linear_index(
                            ctx, out, &index_base, row.clone(), const_i64(0),
                        );
                        let mut a = emit_vec_sum_reduce(
                            ctx, out, buf_operand.clone(), base_offset.clone(), vec_chunks,
                        );
                        // Scalar remainder
                        for idx in (vec_chunks * 4)..reduce_extent as usize {
                            let r_offset = emit_bin(
                                ctx, out, llvm_ir::BinOp::Add,
                                base_offset.clone(), const_i64(idx as i64),
                                llvm_ir::Type::I64,
                            );
                            let ptr = emit_gep(
                                ctx, out, buf_operand.clone(), vec![r_offset],
                                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
                            );
                            let next = emit_load(ctx, out, ptr, llvm_ir::Type::F32);
                            a = emit_bin(ctx, out, llvm_ir::BinOp::Add, a, next, llvm_ir::Type::F32);
                        }
                        a
                    } else {
                        let first_r = if axis == 1 { row.clone() } else { const_i64(0) };
                        let first_c = if axis == 1 { const_i64(0) } else { col.clone() };
                        let first_ptr = emit_buffer_element_ptr(
                            ctx, out, buf_operand.clone(), index_base.clone(), first_r, first_c,
                        );
                        let mut a = emit_load(ctx, out, first_ptr, llvm_ir::Type::F32);

                        for idx in 1..reduce_extent {
                            let next_r = if axis == 1 { row.clone() } else { const_i64(idx) };
                            let next_c = if axis == 1 { const_i64(idx) } else { col.clone() };
                            let next_ptr = emit_buffer_element_ptr(
                                ctx, out, buf_operand.clone(), index_base.clone(), next_r, next_c,
                            );
                            let next = emit_load(ctx, out, next_ptr, llvm_ir::Type::F32);
                            a = reduce_combine(ctx, out, a, next, is_max);
                        }
                        a
                    };

                    let dst_ptr = emit_gep(
                        ctx, out, dst_tile.clone(), vec![row, col],
                        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
                    );
                    emit_store(out, dst_ptr, acc);
                }
            },
        );
    } else {
        // Large extent: LLVMIR loop
        lower_fused_load_reduce_loop(
            builder,
            &prefix,
            out_rows,
            out_cols,
            reduce_extent,
            fusion.axis,
            fusion.is_max,
            buf_operand,
            index_base,
            dst_tile,
        );
    }
}

pub(crate) struct MapReduceFusion {
    pub(crate) expr: TileMapExpr,
    pub(crate) src_rows: i64,
    pub(crate) src_cols: i64,
    pub(crate) is_max: bool,
    pub(crate) axis: i64,
    pub(crate) in_rows: i64,
    pub(crate) in_cols: i64,
}

pub(crate) fn lower_fused_map_reduce_inst(
    result: ValueId,
    fusion: &MapReduceFusion,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if fusion.axis == 1 {
        (fusion.in_rows, 1)
    } else {
        (1, fusion.in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);

    let fused_load_bases = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_map_load_bases(&fusion.expr, ctx, tile_ir, out, &mut bases, &mut coord_bases);
        bases
    });

    let reduce_extent = if fusion.axis == 1 {
        fusion.src_cols
    } else {
        fusion.src_rows
    };
    let prefix = format!("tile_fused_map_reduce_{}", result.0);

    if reduce_extent <= REDUCE_UNROLL_THRESHOLD {
        // Small extent: fully unroll (original behavior)
        lower_nested_tile_loop(
            builder,
            prefix.as_str(),
            out_rows,
            out_cols,
            {
                let expr = fusion.expr.clone();
                let dst_tile = dst_tile.clone();
                let axis = fusion.axis;
                let is_max = fusion.is_max;
                let fused_load_bases = fused_load_bases.clone();

                move |ctx, tile_ir, out, row, col| {
                    let first_r = if axis == 1 { row.clone() } else { const_i64(0) };
                    let first_c = if axis == 1 { const_i64(0) } else { col.clone() };
                    let mut acc = eval_map_expr_scalar(
                        &expr, first_r, first_c, ctx, tile_ir, &fused_load_bases, out,
                    );

                    for idx in 1..reduce_extent {
                        let next_r = if axis == 1 { row.clone() } else { const_i64(idx) };
                        let next_c = if axis == 1 { const_i64(idx) } else { col.clone() };
                        let next = eval_map_expr_scalar(
                            &expr, next_r, next_c, ctx, tile_ir, &fused_load_bases, out,
                        );
                        acc = reduce_combine(ctx, out, acc, next, is_max);
                    }

                    let dst_ptr = emit_gep(
                        ctx, out, dst_tile.clone(), vec![row, col],
                        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
                    );
                    emit_store(out, dst_ptr, acc);
                }
            },
        );
    } else {
        // Large extent: LLVMIR loop
        lower_fused_map_reduce_loop(
            builder,
            &prefix,
            out_rows,
            out_cols,
            reduce_extent,
            fusion.axis,
            fusion.is_max,
            &fusion.expr,
            &fused_load_bases,
            dst_tile,
        );
    }
}

/// Helper: emit tile loop + inner reduce LLVMIR loop for fused load+reduce.
fn lower_fused_load_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    out_rows: i64,
    out_cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    buf_operand: llvm_ir::Operand,
    index_base: BufferIndexBase,
    dst_tile: llvm_ir::Operand,
) {
    if out_rows == 1 {
        lower_single_col_fused_load_reduce_loop(
            builder, prefix, out_cols, reduce_extent, axis, is_max,
            buf_operand, index_base, dst_tile,
        );
    } else if out_cols == 1 {
        lower_single_row_fused_load_reduce_loop(
            builder, prefix, out_rows, reduce_extent, axis, is_max,
            buf_operand, index_base, dst_tile,
        );
    } else {
        lower_full_2d_fused_load_reduce_loop(
            builder, prefix, out_rows, out_cols, reduce_extent, axis, is_max,
            buf_operand, index_base, dst_tile,
        );
    }
}

fn lower_single_col_fused_load_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    buf_operand: llvm_ir::Operand,
    index_base: BufferIndexBase,
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
            true_args: vec![col],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_col = llvm_ir::Operand::Value(body_params[0].id);
    let row = const_i64(0);
    let col = body_col.clone();

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = if let Some(acc) = emit_vec_reduce_loop(
        builder, &reduce_prefix, reduce_extent, axis, is_max,
        &buf_operand, &index_base, row.clone(),
    ) {
        acc
    } else {
        // Scalar fallback
        let init_acc = builder.with_current_insts(|ctx, _, out| {
            let ptr = emit_buffer_element_ptr(
                ctx, out, buf_operand.clone(), index_base.clone(),
                if axis == 1 { row.clone() } else { const_i64(0) },
                if axis == 1 { const_i64(0) } else { col.clone() },
            );
            emit_load(ctx, out, ptr, llvm_ir::Type::F32)
        });

        lower_reduce_extent_loop(
            builder,
            &reduce_prefix,
            reduce_extent,
            1,
            init_acc,
            |ctx, _, out, reduce_idx, acc| {
                let ptr = emit_buffer_element_ptr(
                    ctx, out, buf_operand.clone(), index_base.clone(),
                    if axis == 1 { row.clone() } else { reduce_idx.clone() },
                    if axis == 1 { reduce_idx } else { col.clone() },
                );
                let next = emit_load(ctx, out, ptr, llvm_ir::Type::F32);
                reduce_combine(ctx, out, acc, next, is_max)
            },
        )
    };

    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: header, args: vec![next_col] }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

fn lower_single_row_fused_load_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    buf_operand: llvm_ir::Operand,
    index_base: BufferIndexBase,
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
            true_args: vec![row],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let row = body_row.clone();
    let col = const_i64(0);

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = if let Some(acc) = emit_vec_reduce_loop(
        builder, &reduce_prefix, reduce_extent, axis, is_max,
        &buf_operand, &index_base, row.clone(),
    ) {
        acc
    } else {
        // Scalar fallback
        let init_acc = builder.with_current_insts(|ctx, _, out| {
            let ptr = emit_buffer_element_ptr(
                ctx, out, buf_operand.clone(), index_base.clone(),
                if axis == 1 { row.clone() } else { const_i64(0) },
                if axis == 1 { const_i64(0) } else { col.clone() },
            );
            emit_load(ctx, out, ptr, llvm_ir::Type::F32)
        });

        lower_reduce_extent_loop(
            builder,
            &reduce_prefix,
            reduce_extent,
            1, // start from idx=1 since idx=0 is already in init_acc
            init_acc,
            |ctx, _, out, reduce_idx, acc| {
                let ptr = emit_buffer_element_ptr(
                    ctx, out, buf_operand.clone(), index_base.clone(),
                    if axis == 1 { row.clone() } else { reduce_idx.clone() },
                    if axis == 1 { reduce_idx } else { col.clone() },
                );
                let next = emit_load(ctx, out, ptr, llvm_ir::Type::F32);
                reduce_combine(ctx, out, acc, next, is_max)
            },
        )
    };

    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_row, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: header, args: vec![next_row] }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

fn lower_full_2d_fused_load_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    buf_operand: llvm_ir::Operand,
    index_base: BufferIndexBase,
    dst_tile: llvm_ir::Operand,
) {
    let (row_header, row_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (col_header, col_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![("loop_row", llvm_ir::Type::I64), ("loop_col", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![("loop_row", llvm_ir::Type::I64), ("loop_col", llvm_ir::Type::I64)],
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
            true_args: vec![row, const_i64(0)],
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
            false_args: vec![col_row],
        }
    });
    builder.set_current_terminator(col_term);

    // Body
    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let body_col = llvm_ir::Operand::Value(body_params[1].id);

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = if let Some(acc) = emit_vec_reduce_loop(
        builder, &reduce_prefix, reduce_extent, axis, is_max,
        &buf_operand, &index_base,
        if axis == 1 { body_row.clone() } else { const_i64(0) },
    ) {
        acc
    } else {
        // Scalar fallback
        let init_acc = builder.with_current_insts(|ctx, _, out| {
            let ptr = emit_buffer_element_ptr(
                ctx, out, buf_operand.clone(), index_base.clone(),
                if axis == 1 { body_row.clone() } else { const_i64(0) },
                if axis == 1 { const_i64(0) } else { body_col.clone() },
            );
            emit_load(ctx, out, ptr, llvm_ir::Type::F32)
        });

        lower_reduce_extent_loop(
            builder,
            &reduce_prefix,
            reduce_extent,
            1,
            init_acc,
            |ctx, _, out, reduce_idx, acc| {
                let ptr = emit_buffer_element_ptr(
                    ctx, out, buf_operand.clone(), index_base.clone(),
                    if axis == 1 { body_row.clone() } else { reduce_idx.clone() },
                    if axis == 1 { reduce_idx } else { body_col.clone() },
                );
                let next = emit_load(ctx, out, ptr, llvm_ir::Type::F32);
                reduce_combine(ctx, out, acc, next, is_max)
            },
        )
    };

    let body_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![body_row.clone(), body_col.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: col_header, args: vec![body_row, next_col] }
    });
    builder.set_current_terminator(body_term);

    // Row latch
    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, latch_row, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: row_header, args: vec![next_row] }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}

/// Helper: emit tile loop + inner reduce LLVMIR loop for fused map+reduce.
fn lower_fused_map_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    out_rows: i64,
    out_cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    expr: &TileMapExpr,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    dst_tile: llvm_ir::Operand,
) {
    if out_rows == 1 {
        lower_single_col_fused_map_reduce_loop(
            builder, prefix, out_cols, reduce_extent, axis, is_max,
            expr, fused_load_bases, dst_tile,
        );
    } else if out_cols == 1 {
        lower_single_row_fused_map_reduce_loop(
            builder, prefix, out_rows, reduce_extent, axis, is_max,
            expr, fused_load_bases, dst_tile,
        );
    } else {
        lower_full_2d_fused_map_reduce_loop(
            builder, prefix, out_rows, out_cols, reduce_extent, axis, is_max,
            expr, fused_load_bases, dst_tile,
        );
    }
}

fn lower_single_col_fused_map_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    expr: &TileMapExpr,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
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
            true_args: vec![col],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_col = llvm_ir::Operand::Value(body_params[0].id);
    let row = const_i64(0);
    let col = body_col.clone();

    let init_acc = builder.with_current_insts(|ctx, tile_ir, out| {
        eval_map_expr_scalar(
            expr,
            if axis == 1 { row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { col.clone() },
            ctx, tile_ir, fused_load_bases, out,
        )
    });

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1 since idx=0 is already in init_acc
        init_acc,
        |ctx, tile_ir, out, reduce_idx, acc| {
            let next = eval_map_expr_scalar(
                expr,
                if axis == 1 { row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { col.clone() },
                ctx, tile_ir, fused_load_bases, out,
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: header, args: vec![next_col] }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

fn lower_single_row_fused_map_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    expr: &TileMapExpr,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
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
            true_args: vec![row],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let row = body_row.clone();
    let col = const_i64(0);

    let init_acc = builder.with_current_insts(|ctx, tile_ir, out| {
        eval_map_expr_scalar(
            expr,
            if axis == 1 { row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { col.clone() },
            ctx, tile_ir, fused_load_bases, out,
        )
    });

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1 since idx=0 is already in init_acc
        init_acc,
        |ctx, tile_ir, out, reduce_idx, acc| {
            let next = eval_map_expr_scalar(
                expr,
                if axis == 1 { row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { col.clone() },
                ctx, tile_ir, fused_load_bases, out,
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    let latch_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![row, col],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_row, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: header, args: vec![next_row] }
    });
    builder.set_current_terminator(latch_term);
    builder.switch_to(continue_block);
}

// ---------------------------------------------------------------------------
// Vectorized load + reduce helpers (VecLoad4 + VecReduceAdd)
// ---------------------------------------------------------------------------

/// Emit a `VecLoad { len: 4 }` intrinsic: load 4 contiguous f32 from `buf` at
/// `element_offset`. Returns the `ValueId` of the resulting `<4 x f32>`.
fn emit_vec_load4(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    element_offset: llvm_ir::Operand,
) -> llvm_ir::ValueId {
    let (id, name) = ctx.fresh_value("vec");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty: llvm_ir::Type::vector(4, llvm_ir::Type::F32),
        op: llvm_ir::InstOp::Intrinsic {
            intrinsic: llvm_ir::Intrinsic::VecLoad { len: 4 },
            args: vec![buf, element_offset],
        },
        metadata: Vec::new(),
    });
    id
}

/// Emit a `VecReduceAdd` intrinsic: horizontally sum a `<4 x f32>` into f32.
fn emit_vec_reduce_add(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    vector: llvm_ir::ValueId,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("vsum");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty: llvm_ir::Type::F32,
        op: llvm_ir::InstOp::Intrinsic {
            intrinsic: llvm_ir::Intrinsic::VecReduceAdd,
            args: vec![llvm_ir::Operand::Value(vector)],
        },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

/// Emit vectorized sum-reduce of `vec_count` groups of 4 contiguous elements.
///
/// Given `base_offset` (= linear index of element 0), loads `vec_count`
/// float4 vectors, reduces each to a scalar with VecReduceAdd, and
/// accumulates them, returning the scalar sum.
fn emit_vec_sum_reduce(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    base_offset: llvm_ir::Operand,
    vec_count: usize,
) -> llvm_ir::Operand {
    let vec0 = emit_vec_load4(ctx, out, buf.clone(), base_offset.clone());
    let mut acc = emit_vec_reduce_add(ctx, out, vec0);
    for chunk in 1..vec_count {
        let chunk_offset = emit_bin(
            ctx, out, llvm_ir::BinOp::Add,
            base_offset.clone(), const_i64((chunk * 4) as i64),
            llvm_ir::Type::I64,
        );
        let vec = emit_vec_load4(ctx, out, buf.clone(), chunk_offset);
        let reduced = emit_vec_reduce_add(ctx, out, vec);
        acc = emit_bin(ctx, out, llvm_ir::BinOp::Add, acc, reduced, llvm_ir::Type::F32);
    }
    acc
}

/// Emit a vectorized sum-reduce loop for a fused load+reduce path.
///
/// When `!is_max && axis == 1 && reduce_extent >= 4`, this generates:
/// 1. A step-4 reduce loop that loads 4 contiguous f32 via VecLoad4,
///    reduces each to scalar with VecReduceAdd, and accumulates.
/// 2. A scalar reduce loop for the remaining tail elements.
///
/// Returns `Some(final_acc)` when vectorization was applied, `None` otherwise.
fn emit_vec_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    buf_operand: &llvm_ir::Operand,
    index_base: &BufferIndexBase,
    row: llvm_ir::Operand,
) -> Option<llvm_ir::Operand> {
    // Only vectorize: sum-reduce (!is_max), axis==1 (contiguous), extent >= 4
    if is_max || axis != 1 || reduce_extent < 4 {
        return None;
    }

    let vec_count = (reduce_extent / 4) as usize;
    let scalar_start = (vec_count * 4) as i64;

    // Step 1: vectorized loop (step = 4)
    //   For each iteration idx (0, 4, 8, ...):
    //     base_offset = emit_buffer_linear_index(ctx, out, index_base, row, idx)
    //     vec = VecLoad4(buf, base_offset)
    //     reduced = VecReduceAdd(vec)
    //     acc += reduced
    let vec_prefix = format!("{prefix}_vec");
    let vec_acc = lower_reduce_extent_loop_step(
        builder,
        &vec_prefix,
        reduce_extent,
        0,
        4,
        const_f32(0.0),
        |ctx, _, out, reduce_idx, acc| {
            let base_offset = emit_buffer_linear_index(
                ctx, out, index_base, row.clone(), reduce_idx,
            );
            let vec = emit_vec_load4(ctx, out, buf_operand.clone(), base_offset);
            let reduced = emit_vec_reduce_add(ctx, out, vec);
            emit_bin(ctx, out, llvm_ir::BinOp::Add, acc, reduced, llvm_ir::Type::F32)
        },
    );

    // Step 2: scalar tail loop (if any remainder)
    if scalar_start < reduce_extent {
        let tail_prefix = format!("{prefix}_tail");
        lower_reduce_extent_loop(
            builder,
            &tail_prefix,
            reduce_extent,
            scalar_start,
            vec_acc,
            |ctx, _, out, reduce_idx, acc| {
                let ptr = emit_buffer_element_ptr(
                    ctx, out, buf_operand.clone(), index_base.clone(),
                    row.clone(), reduce_idx,
                );
                let next = emit_load(ctx, out, ptr, llvm_ir::Type::F32);
                emit_bin(ctx, out, llvm_ir::BinOp::Add, acc, next, llvm_ir::Type::F32)
            },
        )
    } else {
        vec_acc
    }
    .into()
}

fn lower_full_2d_fused_map_reduce_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    reduce_extent: i64,
    axis: i64,
    is_max: bool,
    expr: &TileMapExpr,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    dst_tile: llvm_ir::Operand,
) {
    let (row_header, row_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (col_header, col_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![("loop_row", llvm_ir::Type::I64), ("loop_col", llvm_ir::Type::I64)],
    );
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![("loop_row", llvm_ir::Type::I64), ("loop_col", llvm_ir::Type::I64)],
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
            true_args: vec![row, const_i64(0)],
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
            false_args: vec![col_row],
        }
    });
    builder.set_current_terminator(col_term);

    // Body
    builder.switch_to(body_block);
    let body_row = llvm_ir::Operand::Value(body_params[0].id);
    let body_col = llvm_ir::Operand::Value(body_params[1].id);

    let init_acc = builder.with_current_insts(|ctx, tile_ir, out| {
        eval_map_expr_scalar(
            expr,
            if axis == 1 { body_row.clone() } else { const_i64(0) },
            if axis == 1 { const_i64(0) } else { body_col.clone() },
            ctx, tile_ir, fused_load_bases, out,
        )
    });

    let reduce_prefix = format!("{prefix}_reduce");
    let final_acc = lower_reduce_extent_loop(
        builder,
        &reduce_prefix,
        reduce_extent,
        1, // start from idx=1 since idx=0 is already in init_acc
        init_acc,
        |ctx, tile_ir, out, reduce_idx, acc| {
            let next = eval_map_expr_scalar(
                expr,
                if axis == 1 { body_row.clone() } else { reduce_idx.clone() },
                if axis == 1 { reduce_idx } else { body_col.clone() },
                ctx, tile_ir, fused_load_bases, out,
            );
            reduce_combine(ctx, out, acc, next, is_max)
        },
    );

    let body_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx, out, dst_tile.clone(), vec![body_row.clone(), body_col.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, final_acc);
        let next_col = emit_bin(ctx, out, llvm_ir::BinOp::Add, body_col, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: col_header, args: vec![body_row, next_col] }
    });
    builder.set_current_terminator(body_term);

    // Row latch
    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(ctx, out, llvm_ir::BinOp::Add, latch_row, const_i64(1), llvm_ir::Type::I64);
        llvm_ir::Terminator::Br { target: row_header, args: vec![next_row] }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}
