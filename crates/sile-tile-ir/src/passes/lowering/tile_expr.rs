use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::passes::lowering::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, buffer_rank_of, const_f32, const_i64,
    emit_bin, emit_cast, emit_cmp, emit_gep, emit_intrinsic, emit_load, emit_select, emit_store,
    load_tile_scalar_dynamic, lower_1d_tile_coord, lower_bin_op, lower_cmp_pred,
    lower_float_cmp_pred, lower_nested_tile_loop, resolve_operand, tile_dims_of,
};

use super::tile_memory::emit_strided_element_index;
use crate::{CmpOp, TileIrFunction, TileIrOp, TileIrType, TileMapExpr, UnaryOp, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CoordOperandKey {
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
    // A map over a tile fills private storage, one element per iteration. A map
    // whose operands are all scalars is a single value: `min(a, b)` of two
    // scalars is one comparison and one choice. The result type says which of
    // the two this map is, and the scalar one needs neither storage nor a loop.
    let result_ty = builder.tile_ir().types.get(&result).cloned();
    if !matches!(result_ty, Some(crate::TileIrType::Tile { .. })) {
        // A map that produces one value produces a count or an element, and the
        // result type says which: `min(mask_start, k_seqlen_tiles)` compares two
        // counts and yields a count, which is what the division after it needs.
        let count = matches!(result_ty, Some(crate::TileIrType::I64));
        let zero = const_i64(0);
        builder.with_current_insts(|ctx, tile_ir, out| {
            let mut bases = HashMap::new();
            let mut coord_bases = HashMap::new();
            collect_map_load_bases(&expr, ctx, tile_ir, out, &mut bases, &mut coord_bases);
            let value =
                eval_map_expr_scalar(&expr, zero.clone(), zero, ctx, tile_ir, &bases, out, count);
            ctx.operands.insert(result, value);
        });
        return;
    }
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
                false,
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
                false,
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
                outer_coords,
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
                    let outer_operands: Vec<llvm_ir::Operand> = outer_coords
                        .iter()
                        .map(|coord| resolve_operand(*coord, ctx))
                        .collect();
                    emit_strided_element_index(
                        ctx,
                        out,
                        buf_operand.clone(),
                        &outer_operands,
                        stride_shape_idx,
                        src_row,
                        src_col,
                    )
                    .0
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
            TileIrOp::UnaryF { op, operand, .. } => {
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
                return eval_unary_scalar(op, src, ctx, out, false);
            }
            TileIrOp::Broadcast { value: src, .. } => {
                // A broadcast of a scalar has no tile to index, so the value is
                // the scalar itself. `a.broadcast(y.shape())` is the way CuTile
                // splats one argument across a tile.
                if tile_dims_of(src, tile_ir).is_none() {
                    return element_operand(src, ctx, tile_ir, out);
                }
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

    // A value with no tile type is a scalar, not a private tile: the operand is
    // the element itself. Indexing it would read through a float as a pointer,
    // which is how a broadcast scalar became `&(a[0][0])` in the generated C.
    if tile_dims_of(value, tile_ir).is_none() {
        return element_operand(value, ctx, tile_ir, out);
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
        TileIrOp::UnaryF { operand, .. } => {
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
        // A broadcast of a scalar has no load to hoist, and the recursive call
        // returns immediately when the source is not a pending tile.
        TileIrOp::Broadcast { value, .. } => {
            collect_fused_load_bases(value, pending_tiles, ctx, tile_ir, out, bases, coord_bases);
        }
        _ => {}
    }
}

pub(crate) fn collect_map_load_bases(
    expr: &TileMapExpr,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    out: &mut Vec<llvm_ir::Inst>,
    bases: &mut HashMap<ValueId, llvm_ir::Operand>,
    coord_bases: &mut HashMap<(CoordOperandKey, CoordOperandKey, i64), llvm_ir::Operand>,
) {
    match expr {
        TileMapExpr::Value(_) | TileMapExpr::Splat { .. } | TileMapExpr::Iota => {}
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
        TileMapExpr::Cmp { lhs, rhs, .. }
        | TileMapExpr::Add { lhs, rhs }
        | TileMapExpr::Sub { lhs, rhs }
        | TileMapExpr::Mul { lhs, rhs }
        | TileMapExpr::Div { lhs, rhs }
        | TileMapExpr::Rem { lhs, rhs }
        | TileMapExpr::Pow { lhs, rhs }
        | TileMapExpr::Atan2 { lhs, rhs } => {
            collect_map_load_bases(lhs, ctx, tile_ir, out, bases, coord_bases);
            collect_map_load_bases(rhs, ctx, tile_ir, out, bases, coord_bases);
        }
        TileMapExpr::Unary { operand, .. } => {
            collect_map_load_bases(operand, ctx, tile_ir, out, bases, coord_bases);
        }
        TileMapExpr::Select {
            cond,
            on_true,
            on_false,
        } => {
            collect_map_load_bases(cond, ctx, tile_ir, out, bases, coord_bases);
            collect_map_load_bases(on_true, ctx, tile_ir, out, bases, coord_bases);
            collect_map_load_bases(on_false, ctx, tile_ir, out, bases, coord_bases);
        }
        TileMapExpr::Broadcast { value, .. } | TileMapExpr::Transpose { value } => {
            collect_map_load_bases(value, ctx, tile_ir, out, bases, coord_bases);
        }
    }
}

/// One unary float operation on one value.
///
/// A negation is a subtraction from zero and a reciprocal square root is one
/// over the square root: neither is an instruction any target spells the way
/// this IR prints the others, and the two steps are what they mean. The zero
/// carries the operand's type, because an integer negation and a float one are
/// different subtractions to every target.
fn eval_unary_scalar(
    op: UnaryOp,
    operand: llvm_ir::Operand,
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    count: bool,
) -> llvm_ir::Operand {
    match op {
        UnaryOp::Neg if count => emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Sub,
            const_i64(0),
            operand,
            llvm_ir::Type::I64,
        ),
        UnaryOp::Neg => emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Sub,
            const_f32(0.0),
            operand,
            llvm_ir::Type::F32,
        ),
        UnaryOp::Rsqrt => {
            let root = emit_intrinsic(
                ctx,
                out,
                llvm_ir::Intrinsic::Sqrt,
                vec![operand],
                llvm_ir::Type::F32,
            );
            emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Div,
                const_f32(1.0),
                root,
                llvm_ir::Type::F32,
            )
        }
        UnaryOp::Exp
        | UnaryOp::Exp2
        | UnaryOp::Log
        | UnaryOp::Log2
        | UnaryOp::Sqrt
        | UnaryOp::Abs
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Tan
        | UnaryOp::Sinh
        | UnaryOp::Cosh
        | UnaryOp::Tanh
        | UnaryOp::Ceil
        | UnaryOp::Floor => {
            let intrinsic = match op {
                UnaryOp::Exp => llvm_ir::Intrinsic::Exp,
                UnaryOp::Exp2 => llvm_ir::Intrinsic::Exp2,
                UnaryOp::Log => llvm_ir::Intrinsic::Log,
                UnaryOp::Log2 => llvm_ir::Intrinsic::Log2,
                UnaryOp::Sqrt => llvm_ir::Intrinsic::Sqrt,
                UnaryOp::Abs => llvm_ir::Intrinsic::Abs,
                UnaryOp::Sin => llvm_ir::Intrinsic::Sin,
                UnaryOp::Cos => llvm_ir::Intrinsic::Cos,
                UnaryOp::Tan => llvm_ir::Intrinsic::Tan,
                UnaryOp::Sinh => llvm_ir::Intrinsic::Sinh,
                UnaryOp::Cosh => llvm_ir::Intrinsic::Cosh,
                UnaryOp::Tanh => llvm_ir::Intrinsic::Tanh,
                UnaryOp::Ceil => llvm_ir::Intrinsic::Ceil,
                UnaryOp::Floor => llvm_ir::Intrinsic::Floor,
                _ => unreachable!("Neg and Rsqrt are handled above"),
            };
            emit_intrinsic(ctx, out, intrinsic, vec![operand], llvm_ir::Type::F32)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_map_expr_scalar(
    expr: &TileMapExpr,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    fused_load_bases: &HashMap<ValueId, llvm_ir::Operand>,
    out: &mut Vec<llvm_ir::Inst>,
    count: bool,
) -> llvm_ir::Operand {
    match expr {
        TileMapExpr::Value(value) => {
            // A value with no tile type is a scalar operand, so the map reads
            // it directly instead of indexing a private tile. An element value
            // is a float, so a count that reaches one is converted here.
            if tile_dims_of(*value, tile_ir).is_none() {
                if count {
                    resolve_operand(*value, ctx)
                } else {
                    element_operand(*value, ctx, tile_ir, out)
                }
            } else {
                load_tile_scalar_dynamic(ctx, out, resolve_operand(*value, ctx), row, col)
            }
        }
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => emit_map_load_scalar(
            *buf,
            outer_coords,
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
        TileMapExpr::Cmp { op, lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            // The comparison yields a mask, and a mask is a value like any
            // other: one where it holds, zero where it does not. Keeping the
            // `i1` here would put a boolean in a tile of floats, and a mask of
            // counts is a count.
            let mask = emit_cmp(
                ctx,
                out,
                comparison_pred(*op, count),
                lhs,
                rhs,
                llvm_ir::Type::I1,
            );
            let (on_true, on_false) = mask_values(count);
            emit_select(ctx, out, mask, on_true, on_false, scalar_ty(count))
        }
        TileMapExpr::Add { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_bin(ctx, out, llvm_ir::BinOp::Add, lhs, rhs, scalar_ty(count))
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
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_bin(ctx, out, llvm_ir::BinOp::Sub, lhs, rhs, scalar_ty(count))
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
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_bin(ctx, out, llvm_ir::BinOp::Mul, lhs, rhs, scalar_ty(count))
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
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_bin(ctx, out, llvm_ir::BinOp::Div, lhs, rhs, scalar_ty(count))
        }
        // Pow and atan2 are the two binary functions the targets spell as
        // library calls, so they lower to intrinsics like a unary does.
        TileMapExpr::Pow { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_intrinsic(
                ctx,
                out,
                llvm_ir::Intrinsic::Pow,
                vec![lhs, rhs],
                llvm_ir::Type::F32,
            )
        }
        TileMapExpr::Atan2 { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_intrinsic(
                ctx,
                out,
                llvm_ir::Intrinsic::Atan2,
                vec![lhs, rhs],
                llvm_ir::Type::F32,
            )
        }
        TileMapExpr::Unary { op, operand } => {
            let operand =
                eval_map_expr_scalar(operand, row, col, ctx, tile_ir, fused_load_bases, out, count);
            eval_unary_scalar(*op, operand, ctx, out, count)
        }
        TileMapExpr::Rem { lhs, rhs } => {
            let lhs = eval_map_expr_scalar(
                lhs,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let rhs = eval_map_expr_scalar(rhs, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_bin(ctx, out, llvm_ir::BinOp::Rem, lhs, rhs, scalar_ty(count))
        }
        TileMapExpr::Select {
            cond,
            on_true,
            on_false,
        } => {
            // A comparison used as a condition is already a test. Reading the
            // mask it would produce and then testing the mask costs two
            // instructions and says the same thing, so the two sides of the
            // comparison select directly.
            let is_set = match cond.as_ref() {
                TileMapExpr::Cmp { op, lhs, rhs } => {
                    let cmp_count = map_expr_is_count(lhs, tile_ir);
                    let lhs = eval_map_expr_scalar(
                        lhs,
                        row.clone(),
                        col.clone(),
                        ctx,
                        tile_ir,
                        fused_load_bases,
                        out,
                        count,
                    );
                    let rhs =
                        eval_map_expr_scalar(rhs, row.clone(), col.clone(), ctx, tile_ir, fused_load_bases, out, count);
                    emit_cmp(
                        ctx,
                        out,
                        comparison_pred(*op, cmp_count),
                        lhs,
                        rhs,
                        llvm_ir::Type::I1,
                    )
                }
                cond => {
                    // A comparison lowers to a mask of ones and zeroes, the way
                    // `gt_tile` documents it. `select` tests it, so the test is
                    // "greater than zero": a NaN mask is not set, and neither is
                    // a cleared element.
                    let (pred, zero) = mask_test(map_expr_is_count(cond, tile_ir));
                    let cond = eval_map_expr_scalar(
                        cond,
                        row.clone(),
                        col.clone(),
                        ctx,
                        tile_ir,
                        fused_load_bases,
                        out,
                        count,
                    );
                    emit_cmp(ctx, out, pred, cond, zero, llvm_ir::Type::I1)
                }
            };
            let on_true = eval_map_expr_scalar(
                on_true,
                row.clone(),
                col.clone(),
                ctx,
                tile_ir,
                fused_load_bases,
                out,
                count,
            );
            let on_false =
                eval_map_expr_scalar(on_false, row, col, ctx, tile_ir, fused_load_bases, out, count);
            emit_select(ctx, out, is_set, on_true, on_false, scalar_ty(count))
        }
        TileMapExpr::Broadcast {
            value,
            src_rows,
            src_cols,
        } => {
            let src_row = if *src_rows == 1 { const_i64(0) } else { row };
            let src_col = if *src_cols == 1 { const_i64(0) } else { col };
            eval_map_expr_scalar(value, src_row, src_col, ctx, tile_ir, fused_load_bases, out, count)
        }
        // The transpose reads its operand at the coordinates of the element
        // this walk is writing, exchanged.
        TileMapExpr::Transpose { value } => {
            eval_map_expr_scalar(value, col, row, ctx, tile_ir, fused_load_bases, out, count)
        }
        // A sequence is the coordinate the walk stands on. CuTile names it for
        // a rank-one shape, which is one row of N columns here, so the column is
        // the element and the conversion makes it an element value.
        TileMapExpr::Iota => emit_cast(
            ctx,
            out,
            llvm_ir::CastOp::Sitofp,
            col,
            llvm_ir::Type::F32,
        ),
    }
}

/// A scalar operand read as an element value.
///
/// An element value is a float, so a count that reaches element arithmetic is
/// converted here: `broadcast_scalar(q_m_idx * BM, shape![BM, 1])` splats a
/// count across a tile of elements, and the tile holds floats.
fn element_operand(
    value: ValueId,
    ctx: &mut LowerLlvmIrCtx,
    tile_ir: &TileIrFunction,
    out: &mut Vec<llvm_ir::Inst>,
) -> llvm_ir::Operand {
    let is_count = matches!(tile_ir.types.get(&value), Some(TileIrType::I64));
    let operand = resolve_operand(value, ctx);
    if is_count {
        return emit_cast(ctx, out, llvm_ir::CastOp::Sitofp, operand, llvm_ir::Type::F32);
    }
    operand
}

/// The type an element value is carried in: a count is `i64` and everything
/// else, a tile element and a float scalar alike, is `float`.
fn scalar_ty(count: bool) -> llvm_ir::Type {
    if count {
        llvm_ir::Type::I64
    } else {
        llvm_ir::Type::F32
    }
}

/// Whether every value a map expression is built from is a count.
///
/// The tree carries no types of its own, so this reads the operand table. One
/// float in the tree makes the result a float: `qk * scale` multiplies a tile
/// of elements by a scalar, and the product is an element.
fn map_expr_is_count(expr: &TileMapExpr, tile_ir: &TileIrFunction) -> bool {
    match expr {
        TileMapExpr::Value(value) => matches!(tile_ir.types.get(value), Some(TileIrType::I64)),
        // A comparison of two counts is a count, so the left operand's kind is
        // the comparison's.
        TileMapExpr::Cmp { lhs, .. } => map_expr_is_count(lhs, tile_ir),
        TileMapExpr::Add { lhs, rhs }
        | TileMapExpr::Sub { lhs, rhs }
        | TileMapExpr::Mul { lhs, rhs }
        | TileMapExpr::Div { lhs, rhs }
        | TileMapExpr::Rem { lhs, rhs } => {
            map_expr_is_count(lhs, tile_ir) && map_expr_is_count(rhs, tile_ir)
        }
        TileMapExpr::Select {
            on_true, on_false, ..
        } => map_expr_is_count(on_true, tile_ir) && map_expr_is_count(on_false, tile_ir),
        // A sequence, a splat, a transcendental, a binary float function, a
        // load and a broadcast of one are all element values.
        TileMapExpr::Splat { .. }
        | TileMapExpr::Unary { .. }
        | TileMapExpr::Pow { .. }
        | TileMapExpr::Atan2 { .. }
        | TileMapExpr::Iota
        | TileMapExpr::Broadcast { .. }
        | TileMapExpr::Transpose { .. }
        | TileMapExpr::LoadPtrTko { .. } => false,
    }
}

/// The predicate a comparison of this type uses.
///
/// LLVM has one comparison instruction for integers and another for floats, and
/// the operands say which this is.
fn comparison_pred(op: CmpOp, count: bool) -> llvm_ir::CmpPred {
    if count {
        lower_cmp_pred(op)
    } else {
        lower_float_cmp_pred(op)
    }
}

/// The two values a mask holds, as the type it is carried in.
fn mask_values(count: bool) -> (llvm_ir::Operand, llvm_ir::Operand) {
    if count {
        (const_i64(1), const_i64(0))
    } else {
        (const_f32(1.0), const_f32(0.0))
    }
}

/// The test that reads a value as a mask, and the zero it tests against.
fn mask_test(count: bool) -> (llvm_ir::CmpPred, llvm_ir::Operand) {
    if count {
        (llvm_ir::CmpPred::Ne, const_i64(0))
    } else {
        (llvm_ir::CmpPred::Ogt, const_f32(0.0))
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_map_load_scalar(
    buf: ValueId,
    outer_coords: &[ValueId],
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
        let outer_operands: Vec<llvm_ir::Operand> = outer_coords
            .iter()
            .map(|coord| resolve_operand(*coord, ctx))
            .collect();
        emit_strided_element_index(
            ctx,
            out,
            buf_operand.clone(),
            &outer_operands,
            stride_shape_idx,
            src_row,
            src_col,
        )
        .0
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
        TileIrOp::UnaryF { operand, .. } => {
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
            // A broadcast of a scalar has nothing to collect: the source is the
            // value itself, and the map reads it as an operand.
            if defs.contains_key(value)
                && !collect_pointwise_tile_defs(
                    root,
                    *value,
                    defs,
                    tile_ir,
                    use_counts,
                    require_single_use_root,
                    pending_tiles,
                )
            {
                return false;
            }
        }
        _ => return false,
    }

    pending_tiles.insert(value, op);
    true
}
