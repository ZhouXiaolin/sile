use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_f32, const_i64, emit_bin, emit_gep,
    emit_load, emit_shape_dim, emit_store, load_tile_scalar_dynamic, lower_1d_tile_coord,
    lower_nested_tile_loop, resolve_operand,
};

pub(crate) fn lower_tile_constant_inst(
    result: ValueId,
    value: f64,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let prefix = format!("tile_fill_{}", result.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, _, out, row, col| {
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
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
    let rank = buffer_rank_of(buf, builder.tile_ir());
    let prefix = format!("tile_load_{}", result.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, _, out, row, col| {
            let src_ptr = emit_buffer_element_ptr(
                ctx,
                out,
                buf_operand.clone(),
                row_operand.clone(),
                col_operand.clone(),
                row.clone(),
                col.clone(),
                rows,
                cols,
                stride_shape_idx,
                rank,
            );
            let value = emit_load(ctx, out, src_ptr, llvm_ir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, value);
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
    let rank = buffer_rank_of(buf, builder.tile_ir());
    let prefix = format!("tile_store_{}", value.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, _, out, row, col| {
            let src =
                load_tile_scalar_dynamic(ctx, out, value_operand.clone(), row.clone(), col.clone());
            let dst_ptr = emit_buffer_element_ptr(
                ctx,
                out,
                buf_operand.clone(),
                row_operand.clone(),
                col_operand.clone(),
                row,
                col,
                rows,
                cols,
                stride_shape_idx,
                rank,
            );
            emit_store(out, dst_ptr, src);
        },
    );
}

fn emit_buffer_element_ptr(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    row_coord: llvm_ir::Operand,
    col_coord: llvm_ir::Operand,
    tile_row: llvm_ir::Operand,
    tile_col: llvm_ir::Operand,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    rank: usize,
) -> llvm_ir::Operand {
    let linear_index = if rank <= 1 {
        let tile_coord = lower_1d_tile_coord(ctx, out, row_coord, col_coord);
        let tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            tile_coord,
            const_i64(rows * cols),
            llvm_ir::Type::I64,
        );
        let row_offset = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            tile_row,
            const_i64(cols),
            llvm_ir::Type::I64,
        );
        let element_offset = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            row_offset,
            tile_col,
            llvm_ir::Type::I64,
        );
        emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            tile_base,
            element_offset,
            llvm_ir::Type::I64,
        )
    } else {
        let row_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_coord,
            const_i64(rows),
            llvm_ir::Type::I64,
        );
        let col_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            col_coord,
            const_i64(cols),
            llvm_ir::Type::I64,
        );
        let src_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            row_base,
            tile_row,
            llvm_ir::Type::I64,
        );
        let src_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            col_base,
            tile_col,
            llvm_ir::Type::I64,
        );
        let stride = emit_shape_dim(ctx, out, buf.clone(), stride_shape_idx);
        let row_offset = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            src_row,
            stride,
            llvm_ir::Type::I64,
        );
        emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            row_offset,
            src_col,
            llvm_ir::Type::I64,
        )
    };

    emit_gep(
        ctx,
        out,
        buf,
        vec![linear_index],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
    )
}
