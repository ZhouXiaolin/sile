use std::collections::HashMap;

use crate::{BinOp, MirFunction, MirInst, MirOp, MirTerminator, ValueId};

pub fn run(mut func: MirFunction) -> MirFunction {
    for block in &mut func.blocks {
        let mut expr_to_value = HashMap::<String, ValueId>::new();
        let mut replacements = HashMap::<ValueId, ValueId>::new();
        let mut new_insts = Vec::with_capacity(block.insts.len());

        for inst in &block.insts {
            let rewritten_op = rewrite_op(inst.op.clone(), &replacements);
            let rewritten_inst = MirInst {
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

fn expr_key(op: &MirOp) -> Option<String> {
    match op {
        MirOp::ConstI64(v) => Some(format!("const_i64:{v}")),
        MirOp::ConstF64(v) => Some(format!("const_f64:{}", v.to_bits())),
        MirOp::ProgramId { dim } => Some(format!("program_id:{dim}")),
        MirOp::ShapeDim { buf, dim } => Some(format!("shape_dim:{}:{dim}", buf.0)),
        MirOp::IBinary { op, lhs, rhs } => {
            let (lhs, rhs) = canonical_pair(*op, *lhs, *rhs);
            Some(format!("ibinary:{op:?}:{}:{}", lhs.0, rhs.0))
        }
        MirOp::ICmp { op, lhs, rhs } => Some(format!("icmp:{op:?}:{}:{}", lhs.0, rhs.0)),
        MirOp::TileConstant { value, rows, cols } => {
            Some(format!("tile_const:{}:{rows}:{cols}", value.to_bits()))
        }
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => Some(format!("tile_unary:{op:?}:{}:{rows}:{cols}", operand.0)),
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let (lhs, rhs) = canonical_pair(*op, *lhs, *rhs);
            Some(format!(
                "tile_binary:{op:?}:{}:{}:{rows}:{cols}",
                lhs.0, rhs.0
            ))
        }
        MirOp::TileBroadcast { value, rows, cols } => {
            Some(format!("tile_broadcast:{}:{rows}:{cols}", value.0))
        }
        MirOp::TileLoad { .. }
        | MirOp::TileStore { .. }
        | MirOp::TileMma { .. }
        | MirOp::TileReduce { .. } => None,
    }
}

fn canonical_pair(op: BinOp, lhs: ValueId, rhs: ValueId) -> (ValueId, ValueId) {
    if matches!(op, BinOp::Add | BinOp::Mul) && lhs.0 > rhs.0 {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn rewrite_op(op: MirOp, replacements: &HashMap<ValueId, ValueId>) -> MirOp {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match op {
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => MirOp::TileLoad {
            buf: rewrite(buf),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            rows,
            cols,
            stride_shape_idx,
        },
        MirOp::TileStore {
            buf,
            value,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => MirOp::TileStore {
            buf: rewrite(buf),
            value: rewrite(value),
            row_coord: rewrite(row_coord),
            col_coord: rewrite(col_coord),
            rows,
            cols,
            stride_shape_idx,
        },
        MirOp::TileConstant { value, rows, cols } => MirOp::TileConstant { value, rows, cols },
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => MirOp::TileBinary {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
            rows,
            cols,
        },
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => MirOp::TileUnary {
            op,
            operand: rewrite(operand),
            rows,
            cols,
        },
        MirOp::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => MirOp::TileMma {
            a: rewrite(a),
            b: rewrite(b),
            acc: rewrite(acc),
            tile_m,
            tile_n,
            tile_k,
        },
        MirOp::TileReduce {
            op,
            value,
            axis,
            in_rows,
            in_cols,
        } => MirOp::TileReduce {
            op,
            value: rewrite(value),
            axis,
            in_rows,
            in_cols,
        },
        MirOp::TileBroadcast { value, rows, cols } => MirOp::TileBroadcast {
            value: rewrite(value),
            rows,
            cols,
        },
        MirOp::IBinary { op, lhs, rhs } => MirOp::IBinary {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        MirOp::ICmp { op, lhs, rhs } => MirOp::ICmp {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        MirOp::ConstI64(v) => MirOp::ConstI64(v),
        MirOp::ConstF64(v) => MirOp::ConstF64(v),
        MirOp::ProgramId { dim } => MirOp::ProgramId { dim },
        MirOp::ShapeDim { buf, dim } => MirOp::ShapeDim {
            buf: rewrite(buf),
            dim,
        },
    }
}

fn rewrite_terminator(
    term: MirTerminator,
    replacements: &HashMap<ValueId, ValueId>,
) -> MirTerminator {
    let rewrite = |value: ValueId| resolve_replacement(value, replacements);
    match term {
        MirTerminator::Jump { target, args } => MirTerminator::Jump {
            target,
            args: args.into_iter().map(rewrite).collect(),
        },
        MirTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => MirTerminator::Branch {
            cond: rewrite(cond),
            true_target,
            true_args: true_args.into_iter().map(rewrite).collect(),
            false_target,
            false_args: false_args.into_iter().map(rewrite).collect(),
        },
        MirTerminator::Return => MirTerminator::Return,
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

fn collect_defined_values(func: &MirFunction) -> HashMap<ValueId, ()> {
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
