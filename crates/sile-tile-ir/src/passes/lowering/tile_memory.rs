use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_f32, const_i64, emit_bin, emit_gep,
    emit_load, emit_shape_dim, emit_store, load_tile_scalar_dynamic, lower_1d_tile_coord,
    lower_nested_tile_loop, resolve_operand,
};

#[derive(Clone)]
pub(crate) enum BufferIndexBase {
    Linear {
        tile_base: llvm_ir::Operand,
        tile_cols: i64,
    },
    Strided2D {
        tile_origin: llvm_ir::Operand,
        stride: llvm_ir::Operand,
    },
}

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

pub(crate) fn lower_tile_uninit_inst(
    result: ValueId,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let _ = alloc_tile_result(builder, result, rows, cols);
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
    let index_base = builder.with_current_insts(|ctx, _, out| {
        build_buffer_index_base(
            ctx,
            out,
            buf_operand.clone(),
            row_operand.clone(),
            col_operand.clone(),
            rows,
            cols,
            stride_shape_idx,
            rank,
        )
    });
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
                index_base.clone(),
                row.clone(),
                col.clone(),
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
    let index_base = builder.with_current_insts(|ctx, _, out| {
        build_buffer_index_base(
            ctx,
            out,
            buf_operand.clone(),
            row_operand.clone(),
            col_operand.clone(),
            rows,
            cols,
            stride_shape_idx,
            rank,
        )
    });
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
                index_base.clone(),
                row,
                col,
            );
            emit_store(out, dst_ptr, src);
        },
    );
}

pub(crate) fn build_buffer_index_base(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    row_coord: llvm_ir::Operand,
    col_coord: llvm_ir::Operand,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    rank: usize,
) -> BufferIndexBase {
    if rank <= 1 {
        let tile_coord = lower_1d_tile_coord(ctx, out, row_coord, col_coord);
        let tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            tile_coord,
            const_i64(rows * cols),
            llvm_ir::Type::I64,
        );
        BufferIndexBase::Linear {
            tile_base,
            tile_cols: cols,
        }
    } else {
        let row_tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_coord,
            const_i64(rows),
            llvm_ir::Type::I64,
        );
        let col_tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            col_coord,
            const_i64(cols),
            llvm_ir::Type::I64,
        );
        let stride = emit_shape_dim(ctx, out, buf, stride_shape_idx);
        let row_origin = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_tile_base,
            stride.clone(),
            llvm_ir::Type::I64,
        );
        let tile_origin = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            row_origin,
            col_tile_base,
            llvm_ir::Type::I64,
        );
        BufferIndexBase::Strided2D {
            tile_origin,
            stride,
        }
    }
}

/// Compute the flat element index (`tile_origin + row * stride + col` or
/// `tile_base + row * tile_cols + col`) without performing the GEP.
pub(crate) fn emit_buffer_linear_index(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    index_base: &BufferIndexBase,
    tile_row: llvm_ir::Operand,
    tile_col: llvm_ir::Operand,
) -> llvm_ir::Operand {
    match index_base {
        BufferIndexBase::Linear {
            tile_base,
            tile_cols,
        } => {
            let row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                tile_row,
                const_i64(*tile_cols),
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
                tile_base.clone(),
                element_offset,
                llvm_ir::Type::I64,
            )
        }
        BufferIndexBase::Strided2D {
            tile_origin,
            stride,
        } => {
            let row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                tile_row,
                stride.clone(),
                llvm_ir::Type::I64,
            );
            let with_row = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                tile_origin.clone(),
                row_offset,
                llvm_ir::Type::I64,
            );
            emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                with_row,
                tile_col,
                llvm_ir::Type::I64,
            )
        }
    }
}

pub(crate) fn emit_buffer_element_ptr(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    index_base: BufferIndexBase,
    tile_row: llvm_ir::Operand,
    tile_col: llvm_ir::Operand,
) -> llvm_ir::Operand {
    let linear_index = emit_buffer_linear_index(ctx, out, &index_base, tile_row, tile_col);

    emit_gep(
        ctx,
        out,
        buf,
        vec![linear_index],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
    )
}
