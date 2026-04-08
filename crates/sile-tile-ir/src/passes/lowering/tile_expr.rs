use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::passes::lowering::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, buffer_rank_of, const_f32, const_i64,
    emit_bin, emit_gep, emit_intrinsic, emit_load, emit_shape_dim, emit_store,
    load_tile_scalar_dynamic, lower_1d_tile_coord, lower_bin_op, lower_nested_tile_loop,
    resolve_operand, tile_dims_of,
};
use crate::{TileIrFunction, TileIrOp, TileMapExpr, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CoordOperandKey {
    Value(llvm_ir::ValueId),
    ConstI64(i64),
}

pub(crate) type PointwiseTileDefs = HashMap<ValueId, TileIrOp>;

pub(crate) fn build_pointwise_tile_defs(
    root: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    tile_ir: &TileIrFunction,
    use_counts: &HashMap<ValueId, usize>,
    require_single_use_root: bool,
) -> Option<PointwiseTileDefs> {
    let mut pending_tiles = HashMap::new();
    collect_pointwise_tile_defs(
        root,
        root,
        defs,
        tile_ir,
        use_counts,
        require_single_use_root,
        &mut pending_tiles,
    )
    .then_some(pending_tiles)
}

pub(crate) fn lower_tile_expr_inst(
    result: ValueId,
    rows: i64,
    cols: i64,
    pending_tiles: PointwiseTileDefs,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let fused_load_bases = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_fused_load_bases(
            result,
            &pending_tiles,
            ctx,
            tile_ir,
            out,
            &mut bases,
            &mut coord_bases,
        );
        bases
    });

    lower_nested_tile_loop(
        builder,
        "tile_expr_loop",
        rows,
        cols,
        move |ctx, tile_ir, out, row, col| {
            let scalar = eval_tile_scalar(
                result,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                &pending_tiles,
                &fused_load_bases,
                out,
            );
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}

pub(crate) fn lower_pointwise_rank1_store_inst(
    buf: ValueId,
    value: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    pending_tiles: PointwiseTileDefs,
    builder: &mut BlockLowerer<'_>,
) {
    let buf_operand = resolve_operand(buf, builder.ctx());
    let (fused_load_bases, store_base) = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_fused_load_bases(
            value,
            &pending_tiles,
            ctx,
            tile_ir,
            out,
            &mut bases,
            &mut coord_bases,
        );
        let store_base = get_or_create_1d_tile_base(
            row_coord,
            col_coord,
            rows * cols,
            ctx,
            out,
            &mut coord_bases,
        );
        (bases, store_base)
    });

    let prefix = format!("tile_pointwise_store_{}", value.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, tile_ir, out, row, col| {
            let scalar = eval_tile_scalar(
                value,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                &pending_tiles,
                &fused_load_bases,
                out,
            );
            let element_offset = if rows == 1 {
                col
            } else {
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Mul,
                    row,
                    const_i64(cols),
                    llvm_ir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Add,
                    row_offset,
                    col,
                    llvm_ir::Type::I64,
                )
            };
            let linear_index = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                store_base.clone(),
                element_offset,
                llvm_ir::Type::I64,
            );
            let dst_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}

pub(crate) fn lower_tile_map_inst(
    result: ValueId,
    expr: TileMapExpr,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let fused_load_bases = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_map_load_bases(&expr, ctx, tile_ir, out, &mut bases, &mut coord_bases);
        bases
    });

    lower_nested_tile_loop(
        builder,
        "tile_map_loop",
        rows,
        cols,
        move |ctx, tile_ir, out, row, col| {
            let scalar = eval_map_expr_scalar(
                &expr,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                &fused_load_bases,
                out,
            );
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}

pub(crate) fn lower_pointwise_rank1_store_map_inst(
    buf: ValueId,
    expr: TileMapExpr,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let buf_operand = resolve_operand(buf, builder.ctx());
    let (fused_load_bases, store_base) = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_map_load_bases(&expr, ctx, tile_ir, out, &mut bases, &mut coord_bases);
        let store_base = get_or_create_1d_tile_base(
            row_coord,
            col_coord,
            rows * cols,
            ctx,
            out,
            &mut coord_bases,
        );
        (bases, store_base)
    });

    let prefix = format!("tile_map_store_{}", buf.0);
    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        rows,
        cols,
        move |ctx, tile_ir, out, row, col| {
            let scalar = eval_map_expr_scalar(
                &expr,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                &fused_load_bases,
                out,
            );
            let element_offset = if rows == 1 {
                col
            } else {
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Mul,
                    row,
                    const_i64(cols),
                    llvm_ir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Add,
                    row_offset,
                    col,
                    llvm_ir::Type::I64,
                )
            };
            let linear_index = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                store_base.clone(),
                element_offset,
                llvm_ir::Type::I64,
            );
            let dst_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}

fn eval_tile_scalar(
    value: ValueId,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    pending_tiles: &HashMap<ValueId, TileIrOp>,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    out: &mut Vec<llvm_ir::Inst>,
) -> llvm_ir::Operand {
    if let Some(op) = pending_tiles.get(&value).cloned() {
        match op {
            TileIrOp::Splat { value, .. } => return const_f32(value),
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
            } => {
                let buf_operand = resolve_operand(buf, ctx);
                let rank = buffer_rank_of(buf, tile_ir);
                let linear_index = if rank <= 1 {
                    let tile_base = fused_load_bases.get(&value).cloned().unwrap_or_else(|| {
                        let mut coord_bases = HashMap::new();
                        get_or_create_1d_tile_base(
                            row_coord,
                            col_coord,
                            rows * cols,
                            ctx,
                            out,
                            &mut coord_bases,
                        )
                    });
                    let element_offset = if rows == 1 {
                        col
                    } else {
                        let row_offset = emit_bin(
                            ctx,
                            out,
                            llvm_ir::BinOp::Mul,
                            row,
                            const_i64(cols),
                            llvm_ir::Type::I64,
                        );
                        emit_bin(
                            ctx,
                            out,
                            llvm_ir::BinOp::Add,
                            row_offset,
                            col,
                            llvm_ir::Type::I64,
                        )
                    };
                    emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        tile_base,
                        element_offset,
                        llvm_ir::Type::I64,
                    )
                } else {
                    let row_operand = resolve_operand(row_coord, ctx);
                    let col_operand = resolve_operand(col_coord, ctx);
                    let row_base = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Mul,
                        row_operand,
                        const_i64(rows),
                        llvm_ir::Type::I64,
                    );
                    let col_base = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Mul,
                        col_operand,
                        const_i64(cols),
                        llvm_ir::Type::I64,
                    );
                    let src_row = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        row_base,
                        row,
                        llvm_ir::Type::I64,
                    );
                    let src_col = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        col_base,
                        col,
                        llvm_ir::Type::I64,
                    );
                    let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
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
                let src_ptr = emit_gep(
                    ctx,
                    out,
                    buf_operand,
                    vec![linear_index],
                    llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
                );
                return emit_load(ctx, out, src_ptr, llvm_ir::Type::F32);
            }
            TileIrOp::AddF { lhs, rhs, .. }
            | TileIrOp::SubF { lhs, rhs, .. }
            | TileIrOp::MulF { lhs, rhs, .. }
            | TileIrOp::DivF { lhs, rhs, .. } => {
                let lhs = eval_tile_scalar(
                    lhs,
                    row.clone(),
                    col.clone(),
                    ctx,
                    tile_ir,
                    pending_tiles,
                    fused_load_bases,
                    out,
                );
                let rhs = eval_tile_scalar(
                    rhs,
                    row,
                    col,
                    ctx,
                    tile_ir,
                    pending_tiles,
                    fused_load_bases,
                    out,
                );
                let bin_op = match op {
                    TileIrOp::AddF { .. } => crate::BinOp::Add,
                    TileIrOp::SubF { .. } => crate::BinOp::Sub,
                    TileIrOp::MulF { .. } => crate::BinOp::Mul,
                    TileIrOp::DivF { .. } => crate::BinOp::Div,
                    _ => unreachable!(),
                };
                return emit_bin(ctx, out, lower_bin_op(bin_op), lhs, rhs, llvm_ir::Type::F32);
            }
            TileIrOp::NegF { operand, .. } => {
                let src = eval_tile_scalar(
                    operand,
                    row.clone(),
                    col.clone(),
                    ctx,
                    tile_ir,
                    pending_tiles,
                    fused_load_bases,
                    out,
                );
                return emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Sub,
                    const_f32(0.0),
                    src,
                    llvm_ir::Type::F32,
                );
            }
            TileIrOp::Exp { operand, .. } => {
                let src = eval_tile_scalar(
                    operand,
                    row.clone(),
                    col.clone(),
                    ctx,
                    tile_ir,
                    pending_tiles,
                    fused_load_bases,
                    out,
                );
                return emit_intrinsic(
                    ctx,
                    out,
                    llvm_ir::Intrinsic::Exp,
                    vec![src],
                    llvm_ir::Type::F32,
                );
            }
            TileIrOp::Broadcast { value: src, .. } => {
                let (src_rows, src_cols) = tile_dims_of(src, tile_ir).unwrap_or((1, 1));
                let src_row = if src_rows == 1 { const_i64(0) } else { row };
                let src_col = if src_cols == 1 { const_i64(0) } else { col };
                return eval_tile_scalar(
                    src,
                    src_row,
                    src_col,
                    ctx,
                    tile_ir,
                    pending_tiles,
                    fused_load_bases,
                    out,
                );
            }
            _ => {}
        }
    }

    load_tile_scalar_dynamic(ctx, out, resolve_operand(value, ctx), row, col)
}

fn collect_fused_load_bases(
    value: ValueId,
    pending_tiles: &HashMap<ValueId, TileIrOp>,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    out: &mut Vec<llvm_ir::Inst>,
    bases: &mut HashMap<ValueId, llvm_ir::Operand>,
    coord_bases: &mut HashMap<(CoordOperandKey, CoordOperandKey, i64), llvm_ir::Operand>,
) {
    let Some(op) = pending_tiles.get(&value).cloned() else {
        return;
    };
    match op {
        TileIrOp::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            ..
        } => {
            if buffer_rank_of(buf, tile_ir) <= 1 && !bases.contains_key(&value) {
                let tile_base = get_or_create_1d_tile_base(
                    row_coord,
                    col_coord,
                    rows * cols,
                    ctx,
                    out,
                    coord_bases,
                );
                bases.insert(value, tile_base);
            }
        }
        TileIrOp::AddF { lhs, rhs, .. }
        | TileIrOp::SubF { lhs, rhs, .. }
        | TileIrOp::MulF { lhs, rhs, .. }
        | TileIrOp::DivF { lhs, rhs, .. } => {
            collect_fused_load_bases(lhs, pending_tiles, ctx, tile_ir, out, bases, coord_bases);
            collect_fused_load_bases(rhs, pending_tiles, ctx, tile_ir, out, bases, coord_bases);
        }
        TileIrOp::NegF { operand, .. } | TileIrOp::Exp { operand, .. } => {
            collect_fused_load_bases(
                operand,
                pending_tiles,
                ctx,
                tile_ir,
                out,
                bases,
                coord_bases,
            );
        }
        TileIrOp::Broadcast { value, .. } => {
            collect_fused_load_bases(value, pending_tiles, ctx, tile_ir, out, bases, coord_bases);
        }
        _ => {}
    }
}

fn collect_map_load_bases(
    expr: &TileMapExpr,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    out: &mut Vec<llvm_ir::Inst>,
    bases: &mut HashMap<ValueId, llvm_ir::Operand>,
    coord_bases: &mut HashMap<(CoordOperandKey, CoordOperandKey, i64), llvm_ir::Operand>,
) {
    match expr {
        TileMapExpr::Value(_) | TileMapExpr::Splat { .. } => {}
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            ..
        } => {
            if buffer_rank_of(*buf, tile_ir) <= 1 {
                let tile_key = ValueId(
                    (buf.0 << 24)
                        ^ (row_coord.0 << 16)
                        ^ (col_coord.0 << 8)
                        ^ ((*rows as u32) << 4)
                        ^ (*cols as u32),
                );
                let tile_base = get_or_create_1d_tile_base(
                    *row_coord,
                    *col_coord,
                    rows * cols,
                    ctx,
                    out,
                    coord_bases,
                );
                bases.insert(tile_key, tile_base);
            }
        }
        TileMapExpr::Add { lhs, rhs }
        | TileMapExpr::Sub { lhs, rhs }
        | TileMapExpr::Mul { lhs, rhs }
        | TileMapExpr::Div { lhs, rhs } => {
            collect_map_load_bases(lhs, ctx, tile_ir, out, bases, coord_bases);
            collect_map_load_bases(rhs, ctx, tile_ir, out, bases, coord_bases);
        }
        TileMapExpr::Neg { operand } | TileMapExpr::Exp { operand } => {
            collect_map_load_bases(operand, ctx, tile_ir, out, bases, coord_bases);
        }
        TileMapExpr::Broadcast { value, .. } => {
            collect_map_load_bases(value, ctx, tile_ir, out, bases, coord_bases);
        }
    }
}

fn eval_map_expr_scalar(
    expr: &TileMapExpr,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    out: &mut Vec<llvm_ir::Inst>,
) -> llvm_ir::Operand {
    match expr {
        TileMapExpr::Value(value) => {
            load_tile_scalar_dynamic(ctx, out, resolve_operand(*value, ctx), row, col)
        }
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => emit_map_load_scalar(
            *buf,
            *row_coord,
            *col_coord,
            *rows,
            *cols,
            *stride_shape_idx,
            row,
            col,
            ctx,
            tile_ir,
            fused_load_bases,
            out,
        ),
        TileMapExpr::Splat { value } => const_f32(*value),
        TileMapExpr::Add { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_bin(ctx, out, llvm_ir::BinOp::Add, lhs, rhs, llvm_ir::Type::F32)
        }
        TileMapExpr::Sub { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_bin(ctx, out, llvm_ir::BinOp::Sub, lhs, rhs, llvm_ir::Type::F32)
        }
        TileMapExpr::Mul { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_bin(ctx, out, llvm_ir::BinOp::Mul, lhs, rhs, llvm_ir::Type::F32)
        }
        TileMapExpr::Div { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_bin(ctx, out, llvm_ir::BinOp::Div, lhs, rhs, llvm_ir::Type::F32)
        }
        TileMapExpr::Neg { operand } => {
            let operand =
                eval_map_expr_scalar(operand, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Sub,
                const_f32(0.0),
                operand,
                llvm_ir::Type::F32,
            )
        }
        TileMapExpr::Exp { operand } => {
            let operand =
                eval_map_expr_scalar(operand, row, col, ctx, tile_ir, fused_load_bases, out);
            emit_intrinsic(
                ctx,
                out,
                llvm_ir::Intrinsic::Exp,
                vec![operand],
                llvm_ir::Type::F32,
            )
        }
        TileMapExpr::Broadcast {
            value,
            src_rows,
            src_cols,
        } => {
            let src_row = if *src_rows == 1 { const_i64(0) } else { row };
            let src_col = if *src_cols == 1 { const_i64(0) } else { col };
            eval_map_expr_scalar(value, src_row, src_col, ctx, tile_ir, fused_load_bases, out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_map_load_scalar(
    buf: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    out: &mut Vec<llvm_ir::Inst>,
) -> llvm_ir::Operand {
    let buf_operand = resolve_operand(buf, ctx);
    let rank = buffer_rank_of(buf, tile_ir);
    let linear_index = if rank <= 1 {
        let tile_key = ValueId(
            (buf.0 << 24)
                ^ (row_coord.0 << 16)
                ^ (col_coord.0 << 8)
                ^ ((rows as u32) << 4)
                ^ (cols as u32),
        );
        let tile_base = fused_load_bases.get(&tile_key).cloned().unwrap_or_else(|| {
            let mut coord_bases = HashMap::new();
            get_or_create_1d_tile_base(
                row_coord,
                col_coord,
                rows * cols,
                ctx,
                out,
                &mut coord_bases,
            )
        });
        let element_offset = if rows == 1 {
            col
        } else {
            let row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                row,
                const_i64(cols),
                llvm_ir::Type::I64,
            );
            emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                row_offset,
                col,
                llvm_ir::Type::I64,
            )
        };
        emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            tile_base,
            element_offset,
            llvm_ir::Type::I64,
        )
    } else {
        let row_operand = resolve_operand(row_coord, ctx);
        let col_operand = resolve_operand(col_coord, ctx);
        let row_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_operand,
            const_i64(rows),
            llvm_ir::Type::I64,
        );
        let col_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            col_operand,
            const_i64(cols),
            llvm_ir::Type::I64,
        );
        let src_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            row_base,
            row,
            llvm_ir::Type::I64,
        );
        let src_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            col_base,
            col,
            llvm_ir::Type::I64,
        );
        let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
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
    let src_ptr = emit_gep(
        ctx,
        out,
        buf_operand,
        vec![linear_index],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
    );
    emit_load(ctx, out, src_ptr, llvm_ir::Type::F32)
}

fn get_or_create_1d_tile_base(
    row_coord: ValueId,
    col_coord: ValueId,
    tile_size: i64,
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    bases: &mut HashMap<(CoordOperandKey, CoordOperandKey, i64), llvm_ir::Operand>,
) -> llvm_ir::Operand {
    let row_operand = resolve_operand(row_coord, ctx);
    let col_operand = resolve_operand(col_coord, ctx);
    let cache_key = operand_coord_key(&row_operand)
        .zip(operand_coord_key(&col_operand))
        .map(|(row_key, col_key)| (row_key, col_key, tile_size));
    if let Some(base) = cache_key.as_ref().and_then(|key| bases.get(key).cloned()) {
        return base;
    }

    let tile_coord = lower_1d_tile_coord(ctx, out, row_operand, col_operand);
    let tile_base = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        tile_coord,
        const_i64(tile_size),
        llvm_ir::Type::I64,
    );
    if let Some(cache_key) = cache_key {
        bases.insert(cache_key, tile_base.clone());
    }
    tile_base
}

fn operand_coord_key(operand: &llvm_ir::Operand) -> Option<CoordOperandKey> {
    match operand {
        llvm_ir::Operand::Value(id) => Some(CoordOperandKey::Value(*id)),
        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value)) => {
            Some(CoordOperandKey::ConstI64(*value))
        }
        _ => None,
    }
}

fn collect_pointwise_tile_defs(
    root: ValueId,
    value: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    tile_ir: &TileIrFunction,
    use_counts: &HashMap<ValueId, usize>,
    require_single_use_root: bool,
    pending_tiles: &mut PointwiseTileDefs,
) -> bool {
    if pending_tiles.contains_key(&value) {
        return true;
    }

    let Some(op) = defs.get(&value).cloned() else {
        return false;
    };

    let use_count = use_counts.get(&value).copied().unwrap_or(0);
    let requires_single_use = if value == root {
        require_single_use_root
    } else {
        true
    };
    if requires_single_use && use_count != 1 {
        return false;
    }

    match &op {
        TileIrOp::LoadPtrTko { buf, .. } => {
            if buffer_rank_of(*buf, tile_ir) == 0 {
                return false;
            }
        }
        TileIrOp::Splat { .. } => {}
        TileIrOp::AddF { lhs, rhs, .. }
        | TileIrOp::SubF { lhs, rhs, .. }
        | TileIrOp::MulF { lhs, rhs, .. }
        | TileIrOp::DivF { lhs, rhs, .. } => {
            if !collect_pointwise_tile_defs(
                root,
                *lhs,
                defs,
                tile_ir,
                use_counts,
                require_single_use_root,
                pending_tiles,
            ) || !collect_pointwise_tile_defs(
                root,
                *rhs,
                defs,
                tile_ir,
                use_counts,
                require_single_use_root,
                pending_tiles,
            ) {
                return false;
            }
        }
        TileIrOp::NegF { operand, .. } | TileIrOp::Exp { operand, .. } => {
            if !collect_pointwise_tile_defs(
                root,
                *operand,
                defs,
                tile_ir,
                use_counts,
                require_single_use_root,
                pending_tiles,
            ) {
                return false;
            }
        }
        TileIrOp::Broadcast { value, .. } => {
            if !collect_pointwise_tile_defs(
                root,
                *value,
                defs,
                tile_ir,
                use_counts,
                require_single_use_root,
                pending_tiles,
            ) {
                return false;
            }
        }
        _ => return false,
    }

    pending_tiles.insert(value, op);
    true
}
