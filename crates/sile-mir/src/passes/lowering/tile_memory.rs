use sile_llir as llir;

use super::tile_loops::{lower_tile_load_rank2_loop, lower_tile_store_rank2_loop};
use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_f32, const_i64, emit_bin, emit_gep,
    emit_load, emit_shape_dim, emit_store, lower_1d_tile_coord, lower_nested_tile_loop,
    resolve_operand,
};

pub(crate) fn lower_tile_constant_inst(
    result: ValueId,
    value: f64,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    lower_nested_tile_loop(
        builder,
        "tile_const_loop",
        rows,
        cols,
        move |ctx, _, out, row, col| {
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, const_f32(value));
        },
    );
}

pub(crate) fn lower_tile_load_inst(
    result: ValueId,
    buf: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let buf_operand = resolve_operand(buf, builder.ctx());
    let row_operand = resolve_operand(row_coord, builder.ctx());
    let col_operand = resolve_operand(col_coord, builder.ctx());
    let rank = buffer_rank_of(buf, builder.mir());

    let (tile_base, row_base, col_base, stride) = builder.with_current_insts(|ctx, _, out| {
        if rank <= 1 {
            let tile_coord =
                lower_1d_tile_coord(ctx, out, row_operand.clone(), col_operand.clone());
            let base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                tile_coord,
                const_i64(cols),
                llir::Type::I64,
            );
            (Some(base), None, None, None)
        } else {
            let row_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                row_operand.clone(),
                const_i64(rows),
                llir::Type::I64,
            );
            let col_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                col_operand.clone(),
                const_i64(cols),
                llir::Type::I64,
            );
            let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
            (None, Some(row_base), Some(col_base), Some(stride))
        }
    });

    if rank > 1 {
        lower_tile_load_rank2_loop(
            builder,
            dst_tile,
            buf_operand,
            row_base.expect("row base"),
            col_base.expect("col base"),
            stride.expect("stride"),
            rows,
            cols,
        );
        return;
    }

    lower_nested_tile_loop(
        builder,
        "tile_load_loop",
        rows,
        cols,
        move |ctx, _, out, local_row, local_col| {
            let linear_index = if let Some(tile_base) = tile_base.clone() {
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    tile_base,
                    local_col.clone(),
                    llir::Type::I64,
                )
            } else {
                let src_row = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_base.clone().expect("row base"),
                    local_row.clone(),
                    llir::Type::I64,
                );
                let src_col = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    col_base.clone().expect("col base"),
                    local_col.clone(),
                    llir::Type::I64,
                );
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    src_row,
                    stride.clone().expect("stride"),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_offset,
                    src_col,
                    llir::Type::I64,
                )
            };
            let src_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
            );
            let loaded = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![local_row, local_col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, loaded);
        },
    );
}

pub(crate) fn lower_tile_store_inst(
    buf: ValueId,
    value: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let buf_operand = resolve_operand(buf, builder.ctx());
    let value_operand = resolve_operand(value, builder.ctx());
    let row_operand = resolve_operand(row_coord, builder.ctx());
    let col_operand = resolve_operand(col_coord, builder.ctx());
    let rank = buffer_rank_of(buf, builder.mir());

    let (tile_base, row_base, col_base, stride) = builder.with_current_insts(|ctx, _, out| {
        if rank <= 1 {
            let tile_coord =
                lower_1d_tile_coord(ctx, out, row_operand.clone(), col_operand.clone());
            let base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                tile_coord,
                const_i64(rows * cols),
                llir::Type::I64,
            );
            (Some(base), None, None, None)
        } else {
            let row_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                row_operand.clone(),
                const_i64(rows),
                llir::Type::I64,
            );
            let col_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                col_operand.clone(),
                const_i64(cols),
                llir::Type::I64,
            );
            let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
            (None, Some(row_base), Some(col_base), Some(stride))
        }
    });

    if rank > 1 {
        lower_tile_store_rank2_loop(
            builder,
            buf_operand,
            value_operand,
            row_base.expect("row base"),
            col_base.expect("col base"),
            stride.expect("stride"),
            rows,
            cols,
        );
        return;
    }

    lower_nested_tile_loop(
        builder,
        "tile_store_loop",
        rows,
        cols,
        move |ctx, _, out, local_row, local_col| {
            let src_ptr = emit_gep(
                ctx,
                out,
                value_operand.clone(),
                vec![local_row.clone(), local_col.clone()],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let scalar = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let linear_index = if let Some(tile_base) = tile_base.clone() {
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    tile_base,
                    local_col.clone(),
                    llir::Type::I64,
                )
            } else {
                let dst_row = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_base.clone().expect("row base"),
                    local_row,
                    llir::Type::I64,
                );
                let dst_col = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    col_base.clone().expect("col base"),
                    local_col,
                    llir::Type::I64,
                );
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    dst_row,
                    stride.clone().expect("stride"),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_offset,
                    dst_col,
                    llir::Type::I64,
                )
            };
            let dst_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}
