use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_f32, const_i64, emit_bin, emit_gep,
    emit_load, emit_shape_dim, emit_store, load_tile_scalar_dynamic, lower_1d_tile_coord,
    lower_nested_tile_loop, resolve_operand, tile_dims_of,
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

/// Lower `reshape`: a fresh tile walked in the new extent, each element read
/// from the operand at the row-major linear position.
pub(crate) fn lower_tile_reshape_inst(
    result: ValueId,
    value: ValueId,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let (_, src_cols) = tile_dims_of(value, builder.tile_ir()).unwrap_or((rows, cols));
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let src_tile = resolve_operand(value, builder.ctx());
    let elem_ptr_ty = llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32);
    let prefix = format!("tile_reshape_{}", result.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, _, out, row, col| {
            let row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                row.clone(),
                const_i64(cols),
                llvm_ir::Type::I64,
            );
            let linear = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                row_offset,
                col.clone(),
                llvm_ir::Type::I64,
            );
            let src_row = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Div,
                linear.clone(),
                const_i64(src_cols),
                llvm_ir::Type::I64,
            );
            let src_col = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Rem,
                linear,
                const_i64(src_cols),
                llvm_ir::Type::I64,
            );
            let src_ptr = emit_gep(
                ctx,
                out,
                src_tile.clone(),
                vec![src_row, src_col],
                elem_ptr_ty.clone(),
            );
            let elem = emit_load(ctx, out, src_ptr, llvm_ir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                elem_ptr_ty.clone(),
            );
            emit_store(out, dst_ptr, elem);
        },
    );
}

pub(crate) fn lower_tile_load_inst(
    result: ValueId,
    buf: ValueId,
    outer_coords: &[ValueId],
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
            outer_coords,
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
    outer_coords: &[ValueId],
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
            outer_coords,
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

/// The flat index of one element of a buffer with two or more axes, and the
/// row stride it used.
///
/// `src_row` and `src_col` already carry the tile origin. A tile covers the
/// last two axes, so every axis above them is a plane of planes: its
/// coordinate picks one, and its stride is the product of the extents below
/// it. The list of outer coordinates is outermost first, so walking it
/// backwards walks the axes from the tile outward and each step is one
/// multiplication. Every strided access in the lowering goes through here, so
/// the outer axes are added in one place.
///
/// A tile covers the whole of an outer axis, because a partition's shape has
/// extent one above the last two axes, so an outer coordinate is a tile index
/// and needs no scaling.
pub(crate) fn emit_strided_element_index(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    outer_coords: &[llvm_ir::Operand],
    stride_shape_idx: usize,
    src_row: llvm_ir::Operand,
    src_col: llvm_ir::Operand,
) -> (llvm_ir::Operand, llvm_ir::Operand) {
    let stride = emit_shape_dim(ctx, out, buf.clone(), stride_shape_idx);
    let row_offset = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        src_row,
        stride.clone(),
        llvm_ir::Type::I64,
    );
    let mut index = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Add,
        row_offset,
        src_col,
        llvm_ir::Type::I64,
    );
    // The stride of the row axis is `stride`, so the stride of the axis just
    // above it is the row extent times `stride`, and so on outward.
    let mut running = stride.clone();
    for i in (0..outer_coords.len()).rev() {
        let inner_extent = emit_shape_dim(ctx, out, buf.clone(), i + 1);
        running = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            running,
            inner_extent,
            llvm_ir::Type::I64,
        );
        let origin = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            outer_coords[i].clone(),
            running.clone(),
            llvm_ir::Type::I64,
        );
        index = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            origin,
            index,
            llvm_ir::Type::I64,
        );
    }
    (index, stride)
}

pub(crate) fn build_buffer_index_base(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    outer_coords: &[ValueId],
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
        let outer_operands: Vec<llvm_ir::Operand> = outer_coords
            .iter()
            .map(|coord| resolve_operand(*coord, ctx))
            .collect();
        let (tile_origin, stride) = emit_strided_element_index(
            ctx,
            out,
            buf,
            &outer_operands,
            stride_shape_idx,
            row_tile_base,
            col_tile_base,
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
