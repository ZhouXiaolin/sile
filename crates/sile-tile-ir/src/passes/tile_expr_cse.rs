use std::collections::HashMap;

use crate::{BinOp, TileIrFunction, TileIrInst, TileIrOp, TileIrTerminator, ValueId};

pub fn run(mut func: TileIrFunction) -> TileIrFunction {
    // Two phases, because a value one block computes may be a duplicate of a
    // value an earlier block computes. A use of the duplicate can sit in a third
    // block, or inside a map expression, and the replacement has to reach it.
    // Rewriting as the scan goes would miss both.
    let mut replacements = HashMap::<ValueId, ValueId>::new();
    for block in &func.blocks {
        // A duplicate is only folded into a value of its own block: the two are
        // then in the same block and the first one dominates every use of the
        // second. Folding across blocks would need a dominator check.
        let mut expr_to_value = HashMap::<String, ValueId>::new();
        for inst in &block.insts {
            // The key reads the operands the instruction will have, so two
            // instructions that differ only in a duplicate operand match.
            let op = rewrite_op(inst.op.clone(), &replacements);
            if let Some(key) = expr_key(&op) {
                if let Some(existing) = expr_to_value.get(&key).copied() {
                    replacements.insert(inst.result, existing);
                    continue;
                }
                expr_to_value.insert(key, inst.result);
            }
        }
    }

    for block in &mut func.blocks {
        let mut new_insts = Vec::with_capacity(block.insts.len());
        for inst in &block.insts {
            if replacements.contains_key(&inst.result) {
                continue;
            }
            new_insts.push(TileIrInst {
                result: inst.result,
                op: rewrite_op(inst.op.clone(), &replacements),
            });
        }
        block.insts = new_insts;
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }

    let all_values = collect_defined_values(&func);
    func.types.retain(|value, _| all_values.contains_key(value));
    func
}

fn expr_key(op: &TileIrOp) -> Option<String> {
    match op {
        TileIrOp::ConstI64(v) => Some(format!("const_i64:{v}")),
        TileIrOp::ConstF64(v) => Some(format!("const_f64:{}", v.to_bits())),
        TileIrOp::IBinary { op, lhs, rhs } => {
            let (lhs, rhs) = canonical_pair(*op, *lhs, *rhs);
            Some(format!("ibinary:{op:?}:{}:{}", lhs.0, rhs.0))
        }
        TileIrOp::FBinary { op, lhs, rhs } => {
            let (lhs, rhs) = canonical_pair(*op, *lhs, *rhs);
            Some(format!("fbinary:{op:?}:{}:{}", lhs.0, rhs.0))
        }
        TileIrOp::ICmp { op, lhs, rhs } => Some(format!("icmp:{op:?}:{}:{}", lhs.0, rhs.0)),
        TileIrOp::Splat { value, rows, cols } => {
            Some(format!("splat:{}:{rows}:{cols}", value.to_bits()))
        }
        TileIrOp::UnaryF {
            op,
            operand,
            rows,
            cols,
        } => Some(format!("unary:{op:?}:{}:{rows}:{cols}", operand.0)),
        TileIrOp::AddF {
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let (lhs, rhs) = canonical_pair(BinOp::Add, *lhs, *rhs);
            Some(format!("addf:{}:{}:{rows}:{cols}", lhs.0, rhs.0))
        }
        TileIrOp::MulF {
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let (lhs, rhs) = canonical_pair(BinOp::Mul, *lhs, *rhs);
            Some(format!("mulf:{}:{}:{rows}:{cols}", lhs.0, rhs.0))
        }
        TileIrOp::SubF {
            lhs,
            rhs,
            rows,
            cols,
        } => Some(format!("subf:{}:{}:{rows}:{cols}", lhs.0, rhs.0)),
        TileIrOp::DivF {
            lhs,
            rhs,
            rows,
            cols,
        } => Some(format!("divf:{}:{}:{rows}:{cols}", lhs.0, rhs.0)),
        TileIrOp::Broadcast { value, rows, cols } => {
            Some(format!("broadcast:{}:{rows}:{cols}", value.0))
        }
        TileIrOp::Fma {
            lhs,
            rhs,
            acc,
            rows,
            cols,
        } => Some(format!("fma:{}:{}:{}:{rows}:{cols}", lhs.0, rhs.0, acc.0)),
        // maxf and minf are commutative, so the pair sorts the way an addf's
        // does.
        TileIrOp::MaxF { lhs, rhs, rows, cols } => {
            let (lhs, rhs) = if lhs.0 > rhs.0 { (*rhs, *lhs) } else { (*lhs, *rhs) };
            Some(format!("maxf:{}:{}:{rows}:{cols}", lhs.0, rhs.0))
        }
        TileIrOp::MinF { lhs, rhs, rows, cols } => {
            let (lhs, rhs) = if lhs.0 > rhs.0 { (*rhs, *lhs) } else { (*lhs, *rhs) };
            Some(format!("minf:{}:{}:{rows}:{cols}", lhs.0, rhs.0))
        }
        TileIrOp::Select {
            cond,
            on_true,
            on_false,
            rows,
            cols,
        } => Some(format!(
            "select:{}:{}:{}:{rows}:{cols}",
            cond.0, on_true.0, on_false.0
        )),
        TileIrOp::Reshape { value, rows, cols } => {
            Some(format!("reshape:{}:{rows}:{cols}", value.0))
        }
        TileIrOp::Map { .. } => None,
        TileIrOp::ShapeDim { shape, dim } => Some(format!("shape_dim:{}:{dim}", shape.0)),
        TileIrOp::GridDim { dim } => Some(format!("grid_dim:{dim}")),
        TileIrOp::NumTiles {
            shape,
            dim,
            tile_extent,
        } => Some(format!("num_tiles:{}:{dim}:{tile_extent}", shape.0)),
        TileIrOp::Extract {
            tile,
            row_coord,
            col_coord,
        } => Some(format!(
            "tile_extract:{}:{}:{}",
            tile.0, row_coord.0, col_coord.0
        )),
        TileIrOp::LoadPtrTko { .. }
        | TileIrOp::StorePtrTko { .. }
        | TileIrOp::SileAtomicAdd { .. }
        | TileIrOp::MmaF { .. }
        | TileIrOp::ReduceSum { .. }
        | TileIrOp::ReduceMax { .. }
        | TileIrOp::ReduceMin { .. }
        | TileIrOp::ReduceProd { .. }
        | TileIrOp::Scan { .. }
        | TileIrOp::Cat { .. }
        | TileIrOp::ExtractSlice { .. }
        | TileIrOp::InsertSlice { .. } => None,
    }
}

fn canonical_pair(op: BinOp, lhs: ValueId, rhs: ValueId) -> (ValueId, ValueId) {
    if matches!(op, BinOp::Add | BinOp::Mul) && lhs.0 > rhs.0 {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn rewrite_op(op: TileIrOp, replacements: &HashMap<ValueId, ValueId>) -> TileIrOp {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match op {
        TileIrOp::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => TileIrOp::LoadPtrTko {
            buf: rewrite(buf),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            outer_coords: outer_coords.iter().map(|c| rewrite(*c)).collect(),
            rows,
            cols,
            stride_shape_idx,
        },
        TileIrOp::StorePtrTko {
            buf,
            value,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => TileIrOp::StorePtrTko {
            buf: rewrite(buf),
            value: rewrite(value),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            outer_coords: outer_coords.iter().map(|c| rewrite(*c)).collect(),
            rows,
            cols,
            stride_shape_idx,
        },
        TileIrOp::SileAtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            outer_coords,
            stride_shape_idx,
        } => TileIrOp::SileAtomicAdd {
            buf: rewrite(buf),
            value: rewrite(value),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            outer_coords: outer_coords.iter().map(|c| rewrite(*c)).collect(),
            stride_shape_idx,
        },
        TileIrOp::Splat { value, rows, cols } => TileIrOp::Splat { value, rows, cols },
        TileIrOp::AddF {
            lhs,
            rhs,
            rows,
            cols,
        } => TileIrOp::AddF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::SubF {
            lhs,
            rhs,
            rows,
            cols,
        } => TileIrOp::SubF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::MulF {
            lhs,
            rhs,
            rows,
            cols,
        } => TileIrOp::MulF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::DivF {
            lhs,
            rhs,
            rows,
            cols,
        } => TileIrOp::DivF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::UnaryF {
            op,
            operand,
            rows,
            cols,
        } => TileIrOp::UnaryF {
            op,
            operand: rewrite(operand),
            rows,
            cols,
        },
        TileIrOp::Fma {
            lhs,
            rhs,
            acc,
            rows,
            cols,
        } => TileIrOp::Fma {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            acc: rewrite(acc),
            rows,
            cols,
        },
        TileIrOp::MaxF { lhs, rhs, rows, cols } => TileIrOp::MaxF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::MinF { lhs, rhs, rows, cols } => TileIrOp::MinF {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        TileIrOp::Select {
            cond,
            on_true,
            on_false,
            rows,
            cols,
        } => TileIrOp::Select {
            cond: rewrite(cond),
            on_true: rewrite(on_true),
            on_false: rewrite(on_false),
            rows,
            cols,
        },
        TileIrOp::MmaF {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => TileIrOp::MmaF {
            a: rewrite(a),
            b: rewrite(b),
            acc: rewrite(acc),
            tile_m,
            tile_n,
            tile_k,
        },
        TileIrOp::ReduceSum {
            value,
            axis,
            in_rows,
            in_cols,
        } => TileIrOp::ReduceSum {
            value: rewrite(value),
            axis,
            in_rows,
            in_cols,
        },
        TileIrOp::ReduceMax {
            value,
            axis,
            in_rows,
            in_cols,
        } => TileIrOp::ReduceMax {
            value: rewrite(value),
            axis,
            in_rows,
            in_cols,
        },
        TileIrOp::ReduceMin {
            value,
            axis,
            in_rows,
            in_cols,
        } => TileIrOp::ReduceMin {
            value: rewrite(value),
            axis,
            in_rows,
            in_cols,
        },
        TileIrOp::Scan {
            value,
            identity,
            axis,
            in_rows,
            in_cols,
            reverse,
            combiner,
        } => TileIrOp::Scan {
            value: rewrite(value),
            identity: rewrite(identity),
            axis,
            in_rows,
            in_cols,
            reverse,
            combiner,
        },
        TileIrOp::Cat {
            lhs,
            rhs,
            axis,
            lhs_rows,
            lhs_cols,
            rhs_rows,
            rhs_cols,
        } => TileIrOp::Cat {
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            axis,
            lhs_rows,
            lhs_cols,
            rhs_rows,
            rhs_cols,
        },
        TileIrOp::ReduceProd {
            value,
            axis,
            in_rows,
            in_cols,
        } => TileIrOp::ReduceProd {
            value: rewrite(value),
            axis,
            in_rows,
            in_cols,
        },
        TileIrOp::Broadcast { value, rows, cols } => TileIrOp::Broadcast {
            value: rewrite(value),
            rows,
            cols,
        },
        TileIrOp::Reshape { value, rows, cols } => TileIrOp::Reshape {
            value: rewrite(value),
            rows,
            cols,
        },
        TileIrOp::Map { expr, rows, cols } => TileIrOp::Map {
            expr: rewrite_map_expr(expr, replacements),
            rows,
            cols,
        },
        TileIrOp::Extract {
            tile,
            row_coord,
            col_coord,
        } => TileIrOp::Extract {
            tile: rewrite(tile),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
        },
        TileIrOp::ExtractSlice {
            source,
            row_idx,
            col_idx,
            in_rows,
            in_cols,
            out_rows,
            out_cols,
        } => TileIrOp::ExtractSlice {
            source: rewrite(source),
            row_idx: rewrite(row_idx),
            col_idx: rewrite(col_idx),
            in_rows,
            in_cols,
            out_rows,
            out_cols,
        },
        TileIrOp::InsertSlice {
            source,
            destination,
            row_idx,
            col_idx,
            in_rows,
            in_cols,
            out_rows,
            out_cols,
        } => TileIrOp::InsertSlice {
            source: rewrite(source),
            destination: rewrite(destination),
            row_idx: rewrite(row_idx),
            col_idx: rewrite(col_idx),
            in_rows,
            in_cols,
            out_rows,
            out_cols,
        },
        TileIrOp::IBinary { op, lhs, rhs } => TileIrOp::IBinary {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        TileIrOp::FBinary { op, lhs, rhs } => TileIrOp::FBinary {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        TileIrOp::ICmp { op, lhs, rhs } => TileIrOp::ICmp {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        TileIrOp::ConstI64(v) => TileIrOp::ConstI64(v),
        TileIrOp::ConstF64(v) => TileIrOp::ConstF64(v),
        TileIrOp::ShapeDim { shape, dim } => TileIrOp::ShapeDim {
            shape: rewrite(shape),
            dim,
        },
        TileIrOp::GridDim { dim } => TileIrOp::GridDim { dim },
        TileIrOp::NumTiles {
            shape,
            dim,
            tile_extent,
        } => TileIrOp::NumTiles {
            shape: rewrite(shape),
            dim,
            tile_extent,
        },
    }
}

fn rewrite_map_expr(
    expr: crate::TileMapExpr,
    replacements: &HashMap<ValueId, ValueId>,
) -> crate::TileMapExpr {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match expr {
        crate::TileMapExpr::Value(value) => crate::TileMapExpr::Value(rewrite(value)),
        // A sequence reads no value, so there is nothing to replace.
        crate::TileMapExpr::Iota => crate::TileMapExpr::Iota,
        crate::TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => crate::TileMapExpr::LoadPtrTko {
            buf: rewrite(buf),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            outer_coords: outer_coords.iter().map(|c| rewrite(*c)).collect(),
            rows,
            cols,
            stride_shape_idx,
        },
        crate::TileMapExpr::Splat { value } => crate::TileMapExpr::Splat { value },
        crate::TileMapExpr::Cmp { op, lhs, rhs } => crate::TileMapExpr::Cmp {
            op,
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Add { lhs, rhs } => crate::TileMapExpr::Add {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Sub { lhs, rhs } => crate::TileMapExpr::Sub {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Mul { lhs, rhs } => crate::TileMapExpr::Mul {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Div { lhs, rhs } => crate::TileMapExpr::Div {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Rem { lhs, rhs } => crate::TileMapExpr::Rem {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Pow { lhs, rhs } => crate::TileMapExpr::Pow {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Atan2 { lhs, rhs } => crate::TileMapExpr::Atan2 {
            lhs: Box::new(rewrite_map_expr(*lhs, replacements)),
            rhs: Box::new(rewrite_map_expr(*rhs, replacements)),
        },
        crate::TileMapExpr::Fma { a, b, c } => crate::TileMapExpr::Fma {
            a: Box::new(rewrite_map_expr(*a, replacements)),
            b: Box::new(rewrite_map_expr(*b, replacements)),
            c: Box::new(rewrite_map_expr(*c, replacements)),
        },
        crate::TileMapExpr::Transpose { value } => crate::TileMapExpr::Transpose {
            value: Box::new(rewrite_map_expr(*value, replacements)),
        },
        crate::TileMapExpr::Unary { op, operand } => crate::TileMapExpr::Unary {
            op,
            operand: Box::new(rewrite_map_expr(*operand, replacements)),
        },
        crate::TileMapExpr::Select {
            cond,
            on_true,
            on_false,
        } => crate::TileMapExpr::Select {
            cond: Box::new(rewrite_map_expr(*cond, replacements)),
            on_true: Box::new(rewrite_map_expr(*on_true, replacements)),
            on_false: Box::new(rewrite_map_expr(*on_false, replacements)),
        },
        crate::TileMapExpr::Broadcast {
            value,
            src_rows,
            src_cols,
        } => crate::TileMapExpr::Broadcast {
            value: Box::new(rewrite_map_expr(*value, replacements)),
            src_rows,
            src_cols,
        },
    }
}

fn rewrite_terminator(
    term: TileIrTerminator,
    replacements: &HashMap<ValueId, ValueId>,
) -> TileIrTerminator {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match term {
        TileIrTerminator::Jump { target, args } => TileIrTerminator::Jump {
            target,
            args: args.into_iter().map(rewrite).collect(),
        },
        TileIrTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => TileIrTerminator::Branch {
            cond: rewrite(cond),
            true_target,
            true_args: true_args.into_iter().map(rewrite).collect(),
            false_target,
            false_args: false_args.into_iter().map(rewrite).collect(),
        },
        TileIrTerminator::Return => TileIrTerminator::Return,
    }
}

fn resolve_replacement(value: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut current = value;
    while let Some(next) = replacements.get(&current).copied() {
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn collect_defined_values(func: &TileIrFunction) -> HashMap<ValueId, ()> {
    func.params
        .iter()
        .map(|param| (param.value, ()))
        .chain(
            func.blocks
                .iter()
                .flat_map(|block| {
                    block
                        .params
                        .iter()
                        .copied()
                        .chain(block.insts.iter().map(|inst| inst.result))
                })
                .map(|value| (value, ())),
        )
        .collect()
}
