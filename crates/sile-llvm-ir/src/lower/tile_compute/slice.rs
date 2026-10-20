use crate as llvm_ir;

use sile_tile_ir::function_ir::ValueId;
use crate::lower::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, const_i64, emit_bin, emit_gep, emit_store,
    load_tile_scalar_dynamic, lower_nested_tile_loop, resolve_operand,
};

/// The tile store a copy writes through: a private-memory pointer to one
/// element of the result.
fn store_tile_element(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    dst_tile: &llvm_ir::Operand,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
    value: llvm_ir::Operand,
) {
    let dst_ptr = emit_gep(
        ctx,
        out,
        dst_tile.clone(),
        vec![row, col],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
    );
    emit_store(out, dst_ptr, value);
}

/// The shift a block index applies to a source coordinate: the index times the
/// subtile extent on that axis.
fn block_base(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    idx: &llvm_ir::Operand,
    extent: i64,
) -> llvm_ir::Operand {
    emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        idx.clone(),
        const_i64(extent),
        llvm_ir::Type::I64,
    )
}

/// Cat: the lhs extent is copied in place, then the rhs extent behind it, so
/// each source element moves once and the order along the joined axis is the
/// lhs one first.
pub(crate) fn lower_tile_cat_inst(
    result: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    axis: i64,
    lhs_rows: i64,
    lhs_cols: i64,
    rhs_rows: i64,
    rhs_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if axis == 1 {
        (lhs_rows, lhs_cols + rhs_cols)
    } else {
        (lhs_rows + rhs_rows, lhs_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let lhs_tile = resolve_operand(lhs, builder.ctx());
    let rhs_tile = resolve_operand(rhs, builder.ctx());
    let prefix = format!("tile_cat_{}", result.0);

    lower_nested_tile_loop(
        builder,
        &format!("{prefix}_lhs"),
        lhs_rows,
        lhs_cols,
        |ctx, _, out, row, col| {
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                lhs_tile.clone(),
                row.clone(),
                col.clone(),
            );
            store_tile_element(ctx, out, &dst_tile, row, col, value);
        },
    );
    lower_nested_tile_loop(
        builder,
        &format!("{prefix}_rhs"),
        rhs_rows,
        rhs_cols,
        |ctx, _, out, row, col| {
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                rhs_tile.clone(),
                row.clone(),
                col.clone(),
            );
            let (dst_row, dst_col) = if axis == 1 {
                (
                    row,
                    emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        col,
                        const_i64(lhs_cols),
                        llvm_ir::Type::I64,
                    ),
                )
            } else {
                (
                    emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        row,
                        const_i64(lhs_rows),
                        llvm_ir::Type::I64,
                    ),
                    col,
                )
            };
            store_tile_element(ctx, out, &dst_tile, dst_row, dst_col, value);
        },
    );
}

/// Extract: every element of the subtile reads the source at the block base
/// plus its own coordinate, so one pass over the subtile extent fills the
/// result.
pub(crate) fn lower_tile_extract_slice_inst(
    result: ValueId,
    source: ValueId,
    row_idx: ValueId,
    col_idx: ValueId,
    out_rows: i64,
    out_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(source, builder.ctx());
    let row_base = resolve_operand(row_idx, builder.ctx());
    let col_base = resolve_operand(col_idx, builder.ctx());
    let prefix = format!("tile_extract_{}", result.0);

    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        out_rows,
        out_cols,
        |ctx, _, out, row, col| {
            let row_offset = block_base(ctx, out, &row_base, out_rows);
            let src_row = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                row_offset,
                row.clone(),
                llvm_ir::Type::I64,
            );
            let col_offset = block_base(ctx, out, &col_base, out_cols);
            let src_col = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                col_offset,
                col.clone(),
                llvm_ir::Type::I64,
            );
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                src_row,
                src_col,
            );
            store_tile_element(ctx, out, &dst_tile, row, col, value);
        },
    );
}

/// Insert: the destination is copied whole, then the subtile is written over
/// its block, so the elements the subtile covers end up with the subtile's
/// values and every other element keeps the destination's.
pub(crate) fn lower_tile_insert_slice_inst(
    result: ValueId,
    source: ValueId,
    destination: ValueId,
    row_idx: ValueId,
    col_idx: ValueId,
    in_rows: i64,
    in_cols: i64,
    out_rows: i64,
    out_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(source, builder.ctx());
    let dst_src_tile = resolve_operand(destination, builder.ctx());
    let row_base = resolve_operand(row_idx, builder.ctx());
    let col_base = resolve_operand(col_idx, builder.ctx());
    let prefix = format!("tile_insert_{}", result.0);

    lower_nested_tile_loop(
        builder,
        &format!("{prefix}_copy"),
        out_rows,
        out_cols,
        |ctx, _, out, row, col| {
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                dst_src_tile.clone(),
                row.clone(),
                col.clone(),
            );
            store_tile_element(ctx, out, &dst_tile, row, col, value);
        },
    );
    lower_nested_tile_loop(
        builder,
        &format!("{prefix}_write"),
        in_rows,
        in_cols,
        |ctx, _, out, row, col| {
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                row.clone(),
                col.clone(),
            );
            let row_offset = block_base(ctx, out, &row_base, in_rows);
            let src_row = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                row_offset,
                row.clone(),
                llvm_ir::Type::I64,
            );
            let col_offset = block_base(ctx, out, &col_base, in_cols);
            let src_col = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                col_offset,
                col.clone(),
                llvm_ir::Type::I64,
            );
            store_tile_element(ctx, out, &dst_tile, src_row, src_col, value);
        },
    );
}
