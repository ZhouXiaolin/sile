use std::collections::HashMap;

use crate::{BinOp, TileIrFunction, TileIrInst, TileIrOp, TileIrTerminator, ValueId};

pub fn run(mut func: TileIrFunction) -> TileIrFunction {
    for block in &mut func.blocks {
        let mut expr_to_value = HashMap::<String, ValueId>::new();
        let mut replacements = HashMap::<ValueId, ValueId>::new();
        let mut new_insts = Vec::with_capacity(block.insts.len());

        for inst in &block.insts {
            let rewritten_op = rewrite_op(inst.op.clone(), &replacements);
            let rewritten_inst = TileIrInst {
                result: inst.result,
                op: rewritten_op,
            };

            if let Some(key) = expr_key(&rewritten_inst.op) {
                if let Some(existing) = expr_to_value.get(&key).copied() {
                    replacements.insert(rewritten_inst.result, existing);
                    continue;
                }
                expr_to_value.insert(key, rewritten_inst.result);
            }

            new_insts.push(rewritten_inst);
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
        TileIrOp::ICmp { op, lhs, rhs } => Some(format!("icmp:{op:?}:{}:{}", lhs.0, rhs.0)),
        TileIrOp::Splat { value, rows, cols } => {
            Some(format!("splat:{}:{rows}:{cols}", value.to_bits()))
        }
        TileIrOp::NegF {
            operand,
            rows,
            cols,
        } => Some(format!("negf:{}:{rows}:{cols}", operand.0)),
        TileIrOp::Exp {
            operand,
            rows,
            cols,
        } => Some(format!("exp:{}:{rows}:{cols}", operand.0)),
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
        TileIrOp::Map { .. } => None,
        TileIrOp::ShapeDim { shape, dim } => Some(format!("shape_dim:{}:{dim}", shape.0)),
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
        | TileIrOp::ReduceMax { .. } => None,
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
            rows,
            cols,
            stride_shape_idx,
        } => TileIrOp::LoadPtrTko {
            buf: rewrite(buf),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            rows,
            cols,
            stride_shape_idx,
        },
        TileIrOp::StorePtrTko {
            buf,
            value,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => TileIrOp::StorePtrTko {
            buf: rewrite(buf),
            value: rewrite(value),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            rows,
            cols,
            stride_shape_idx,
        },
        TileIrOp::SileAtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            stride_shape_idx,
        } => TileIrOp::SileAtomicAdd {
            buf: rewrite(buf),
            value: rewrite(value),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
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
        TileIrOp::NegF {
            operand,
            rows,
            cols,
        } => TileIrOp::NegF {
            operand: rewrite(operand),
            rows,
            cols,
        },
        TileIrOp::Exp {
            operand,
            rows,
            cols,
        } => TileIrOp::Exp {
            operand: rewrite(operand),
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
        TileIrOp::Broadcast { value, rows, cols } => TileIrOp::Broadcast {
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
        TileIrOp::IBinary { op, lhs, rhs } => TileIrOp::IBinary {
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
    }
}

fn rewrite_map_expr(
    expr: crate::TileMapExpr,
    replacements: &HashMap<ValueId, ValueId>,
) -> crate::TileMapExpr {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match expr {
        crate::TileMapExpr::Value(value) => crate::TileMapExpr::Value(rewrite(value)),
        crate::TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => crate::TileMapExpr::LoadPtrTko {
            buf: rewrite(buf),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            rows,
            cols,
            stride_shape_idx,
        },
        crate::TileMapExpr::Splat { value } => crate::TileMapExpr::Splat { value },
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
        crate::TileMapExpr::Neg { operand } => crate::TileMapExpr::Neg {
            operand: Box::new(rewrite_map_expr(*operand, replacements)),
        },
        crate::TileMapExpr::Exp { operand } => crate::TileMapExpr::Exp {
            operand: Box::new(rewrite_map_expr(*operand, replacements)),
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
