use std::collections::{HashMap, HashSet};

use crate::{TileIrFunction, TileIrInst, TileIrOp, TileIrType, TileMapExpr, ValueId};

pub fn run(mut func: TileIrFunction) -> TileIrFunction {
    let use_counts = build_use_counts(&func);
    let types = &func.types;

    for block in &mut func.blocks {
        let defs = block
            .insts
            .iter()
            .map(|inst| (inst.result, inst.op.clone()))
            .collect::<HashMap<_, _>>();
        let mut fused_values = HashSet::new();
        let mut new_insts = Vec::with_capacity(block.insts.len());

        for inst in &block.insts {
            if fused_values.contains(&inst.result) {
                continue;
            }
            if !is_pointwise_root(&inst.op) {
                new_insts.push(inst.clone());
                continue;
            }

            if let Some((expr, newly_fused)) =
                build_pointwise_map_expr(inst.result, &defs, types, &use_counts, true)
            {
                fused_values.extend(newly_fused);
                let (rows, cols) = tile_shape_for_op(&inst.op);
                new_insts.push(TileIrInst {
                    result: inst.result,
                    op: TileIrOp::Map { expr, rows, cols },
                });
                continue;
            }

            new_insts.push(inst.clone());
        }

        block.insts = new_insts;
    }

    func
}

fn is_pointwise_root(op: &TileIrOp) -> bool {
    matches!(
        op,
        TileIrOp::AddF { .. }
            | TileIrOp::SubF { .. }
            | TileIrOp::MulF { .. }
            | TileIrOp::DivF { .. }
            | TileIrOp::UnaryF { .. }
            | TileIrOp::Broadcast { .. }
    )
}

fn build_use_counts(func: &TileIrFunction) -> HashMap<ValueId, usize> {
    let mut counts = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            for used in TileIrFunction::inst_uses(&inst.op) {
                *counts.entry(used).or_insert(0) += 1;
            }
        }
        for used in TileIrFunction::terminator_uses(&block.terminator) {
            *counts.entry(used).or_insert(0) += 1;
        }
    }
    counts
}

fn build_pointwise_map_expr(
    value: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    types: &HashMap<ValueId, TileIrType>,
    use_counts: &HashMap<ValueId, usize>,
    is_root: bool,
) -> Option<(TileMapExpr, HashSet<ValueId>)> {
    let op = defs.get(&value)?.clone();
    if !is_root && use_counts.get(&value).copied().unwrap_or(0) != 1 {
        return None;
    }

    match op {
        TileIrOp::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => Some((
            TileMapExpr::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
            },
            HashSet::new(),
        )),
        TileIrOp::Splat { value, .. } => Some((TileMapExpr::Splat { value }, HashSet::new())),
        TileIrOp::AddF { lhs, rhs, .. } => {
            build_binary_expr(value, lhs, rhs, defs, types, use_counts, |lhs, rhs| {
                TileMapExpr::Add { lhs, rhs }
            })
        }
        TileIrOp::SubF { lhs, rhs, .. } => {
            build_binary_expr(value, lhs, rhs, defs, types, use_counts, |lhs, rhs| {
                TileMapExpr::Sub { lhs, rhs }
            })
        }
        TileIrOp::MulF { lhs, rhs, .. } => {
            build_binary_expr(value, lhs, rhs, defs, types, use_counts, |lhs, rhs| {
                TileMapExpr::Mul { lhs, rhs }
            })
        }
        TileIrOp::DivF { lhs, rhs, .. } => {
            build_binary_expr(value, lhs, rhs, defs, types, use_counts, |lhs, rhs| {
                TileMapExpr::Div { lhs, rhs }
            })
        }
        TileIrOp::UnaryF { op, operand, .. } => {
            let (operand_expr, fused) = build_map_child_expr(operand, defs, types, use_counts);
            let mut fused_values = fused;
            if !matches!(operand_expr, TileMapExpr::Value(_)) {
                fused_values.insert(operand);
            }
            Some((
                TileMapExpr::Unary {
                    op,
                    operand: Box::new(operand_expr),
                },
                fused_values,
            ))
        }
        TileIrOp::Broadcast { value: src, .. } => {
            let (src_expr, fused) = build_map_child_expr(src, defs, types, use_counts);
            let mut fused_values = fused;
            if !matches!(src_expr, TileMapExpr::Value(_)) {
                fused_values.insert(src);
            }
            let (src_rows, src_cols) = tile_shape_for_value(src, defs, types);
            Some((
                TileMapExpr::Broadcast {
                    value: Box::new(src_expr),
                    src_rows,
                    src_cols,
                },
                fused_values,
            ))
        }
        _ => None,
    }
}

fn build_binary_expr(
    value: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    types: &HashMap<ValueId, TileIrType>,
    use_counts: &HashMap<ValueId, usize>,
    ctor: impl Fn(Box<TileMapExpr>, Box<TileMapExpr>) -> TileMapExpr,
) -> Option<(TileMapExpr, HashSet<ValueId>)> {
    let (lhs_expr, lhs_fused) = build_map_child_expr(lhs, defs, types, use_counts);
    let (rhs_expr, rhs_fused) = build_map_child_expr(rhs, defs, types, use_counts);
    let mut fused_values = lhs_fused;
    fused_values.extend(rhs_fused);
    if !matches!(lhs_expr, TileMapExpr::Value(_)) {
        fused_values.insert(lhs);
    }
    if !matches!(rhs_expr, TileMapExpr::Value(_)) {
        fused_values.insert(rhs);
    }
    fused_values.remove(&value);
    Some((ctor(Box::new(lhs_expr), Box::new(rhs_expr)), fused_values))
}

fn build_map_child_expr(
    value: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    types: &HashMap<ValueId, TileIrType>,
    use_counts: &HashMap<ValueId, usize>,
) -> (TileMapExpr, HashSet<ValueId>) {
    build_pointwise_map_expr(value, defs, types, use_counts, false)
        .unwrap_or_else(|| (TileMapExpr::Value(value), HashSet::new()))
}

/// The shape a map reads `value` at.
///
/// A value this block defines reports the shape of its own op. A block
/// parameter has no op here, so the function's type table is the only record of
/// its shape. Assuming a scalar instead would read element zero of every row,
/// which is what a loop-carried operand used to do.
fn tile_shape_for_value(
    value: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    types: &HashMap<ValueId, TileIrType>,
) -> (i64, i64) {
    if let Some(op) = defs.get(&value) {
        return tile_shape_for_op(op);
    }
    match types.get(&value) {
        Some(TileIrType::Tile { rows, cols }) => (*rows, *cols),
        _ => (1, 1),
    }
}

fn tile_shape_for_op(op: &TileIrOp) -> (i64, i64) {
    match op {
        TileIrOp::LoadPtrTko { rows, cols, .. }
        | TileIrOp::Splat { rows, cols, .. }
        | TileIrOp::AddF { rows, cols, .. }
        | TileIrOp::SubF { rows, cols, .. }
        | TileIrOp::MulF { rows, cols, .. }
        | TileIrOp::DivF { rows, cols, .. }
        | TileIrOp::UnaryF { rows, cols, .. }
        | TileIrOp::Broadcast { rows, cols, .. }
        | TileIrOp::Map { rows, cols, .. } => (*rows, *cols),
        TileIrOp::ReduceSum {
            axis,
            in_rows,
            in_cols,
            ..
        }
        | TileIrOp::ReduceMax {
            axis,
            in_rows,
            in_cols,
            ..
        } => {
            if *axis == 0 {
                (1, *in_cols)
            } else {
                (*in_rows, 1)
            }
        }
        _ => (1, 1),
    }
}
