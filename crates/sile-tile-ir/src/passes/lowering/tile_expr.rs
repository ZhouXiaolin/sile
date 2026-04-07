use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::passes::lowering::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, buffer_rank_of, const_f32, const_i64,
    emit_bin, emit_gep, emit_intrinsic, emit_load, emit_shape_dim, emit_store,
    load_tile_scalar_dynamic, lower_1d_tile_coord, lower_bin_op, lower_nested_tile_loop,
    resolve_operand, tile_dims_of,
};
use crate::{TileIrFunction, TileIrOp, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CoordOperandKey {
    Value(llvm_ir::ValueId),
    ConstI64(i64),
}

pub(crate) fn lower_tile_expr_inst(
    result: ValueId,
    root_op: TileIrOp,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);

    let pending_tiles = HashMap::from([(result, root_op)]);
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
                    let row_offset = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Mul,
                        row,
                        const_i64(cols),
                        llvm_ir::Type::I64,
                    );
                    let element_offset = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        row_offset,
                        col,
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
