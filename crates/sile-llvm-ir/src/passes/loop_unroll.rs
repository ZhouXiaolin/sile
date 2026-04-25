use std::collections::HashMap;

use crate::{
    BasicBlock, BlockId, BlockParam, BinOp, Constant, CmpPred, Function, Inst, InstOp, Operand,
    Terminator, ValueId,
};

const MAX_UNROLL_TRIP_COUNT: i64 = 8;

pub fn run(mut func: Function) -> Function {
    let blocks = func.blocks.clone();
    let mut unrolled_blocks = Vec::new();
    let mut replaced_blocks = std::collections::HashSet::new();

    for block in &blocks {
        if replaced_blocks.contains(&block.id) {
            continue;
        }
        if let Some((trip_count, _header_idx, body_idx, latch_idx)) =
            detect_unrollable_loop(block, &blocks)
        {
            if trip_count > 0 && trip_count <= MAX_UNROLL_TRIP_COUNT {
                let new_blocks = unroll_loop(
                    &func, &blocks, trip_count, block, body_idx, latch_idx,
                );
                replaced_blocks.insert(block.id);
                replaced_blocks.insert(blocks[body_idx].id);
                if let Some(latch) = blocks.get(latch_idx) {
                    replaced_blocks.insert(latch.id);
                }
                unrolled_blocks.extend(new_blocks);
                continue;
            }
        }
        unrolled_blocks.push(block.clone());
    }

    func.blocks = unrolled_blocks;
    func
}

fn detect_unrollable_loop(
    header: &BasicBlock,
    blocks: &[BasicBlock],
) -> Option<(i64, usize, usize, usize)> {
    let Terminator::CondBr {
        cond,
        true_target,
        ..
    } = &header.terminator
    else {
        return None;
    };
    let cond_val = match cond {
        Operand::Value(id) => *id,
        _ => return None,
    };
    let cmp_inst = blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .find(|inst| inst.result == Some(cond_val))?;
    let InstOp::Cmp { pred, lhs, rhs } = &cmp_inst.op else {
        return None;
    };
    if !matches!(pred, CmpPred::Slt | CmpPred::Sle) {
        return None;
    }
    let (loop_param, bound_val) = match (lhs, rhs) {
        (Operand::Value(param), Operand::Const(Constant::Int(bound))) => (*param, *bound),
        _ => return None,
    };
    let param_idx = header.params.iter().position(|p| p.id == loop_param)?;

    let pred_count = blocks
        .iter()
        .filter(|b| {
            matches!(&b.terminator, Terminator::Br { target, .. } if *target == header.id)
                || matches!(&b.terminator, Terminator::CondBr { true_target, false_target, .. }
                    if *true_target == header.id || *false_target == header.id)
        })
        .count();
    if pred_count != 2 {
        return None;
    }

    let backedge_idx = blocks.iter().position(|b| {
        if let Terminator::Br { target, args } = &b.terminator {
            if *target != header.id {
                return false;
            }
            if let Some(arg) = args.get(param_idx) {
                if let Operand::Value(arg_id) = arg {
                    return blocks.iter().flat_map(|bb| bb.insts.iter()).any(|inst| {
                        inst.result == Some(*arg_id)
                            && matches!(&inst.op,
                                InstOp::Bin { op: BinOp::Add, lhs: Operand::Value(p), rhs: Operand::Const(Constant::Int(1)) }
                                if *p == loop_param
                            )
                    });
                }
            }
        }
        false
    })?;

    let body_idx = blocks.iter().position(|b| b.id == *true_target)?;

    let trip_count = match pred {
        CmpPred::Slt => bound_val,
        CmpPred::Sle => bound_val + 1,
        _ => return None,
    };

    Some((trip_count, 0, body_idx, backedge_idx))
}

fn unroll_loop(
    func: &Function,
    blocks: &[BasicBlock],
    trip_count: i64,
    header: &BasicBlock,
    body_idx: usize,
    _latch_idx: usize,
) -> Vec<BasicBlock> {
    let body = &blocks[body_idx];

    let Terminator::Br {
        target: btarget,
        args: backedge_args,
    } = &body.terminator
    else {
        return blocks.to_vec();
    };
    if *btarget != header.id {
        return blocks.to_vec();
    }

    let Terminator::CondBr {
        cond: _,
        true_target: _,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return blocks.to_vec();
    };
    let exit_id = *false_target;

    if body.params.len() != true_args.len() {
        return blocks.to_vec();
    }

    let max_block = blocks.iter().map(|b| b.id.0).max().unwrap_or(0);
    let max_value = blocks
        .iter()
        .flat_map(|b| {
            b.insts
                .iter()
                .filter_map(|i| i.result)
                .chain(b.params.iter().map(|p| p.id))
        })
        .chain(func.params.iter().map(|p| p.id))
        .map(|v| v.0)
        .max()
        .unwrap_or(0);

    let mut next_bid = max_block + 1;
    let mut next_vid = max_value + 1;

    let mut alloc_bid = || {
        let id = next_bid;
        next_bid += 1;
        BlockId(id)
    };
    let mut alloc_vid = || {
        let id = next_vid;
        next_vid += 1;
        ValueId(id)
    };

    let n = trip_count as usize;
    let copy_ids: Vec<BlockId> = (0..n).map(|_| alloc_bid()).collect();

    let header_passthrough_params: Vec<BlockParam> = header
        .params
        .iter()
        .map(|p| BlockParam {
            id: alloc_vid(),
            name: format!("{}.u", p.name),
            ty: p.ty.clone(),
        })
        .collect();

    let header_param_map: HashMap<ValueId, Operand> = header
        .params
        .iter()
        .zip(header_passthrough_params.iter())
        .map(|(orig, new_p)| (orig.id, Operand::Value(new_p.id)))
        .collect();

    let body0_args: Vec<Operand> = true_args
        .iter()
        .map(|arg| remap_operand(arg.clone(), &header_param_map))
        .collect();

    let mut result = vec![BasicBlock {
        id: header.id,
        name: format!("{}.unroll_entry", header.name),
        params: header_passthrough_params,
        insts: vec![],
        terminator: Terminator::Br {
            target: copy_ids[0],
            args: body0_args,
        },
    }];

    let mut prev_output_values: Option<Vec<Operand>> = None;

    for k in 0..n {
        let is_last = k == n - 1;

        let new_body_params: Vec<BlockParam> = body
            .params
            .iter()
            .map(|p| BlockParam {
                id: alloc_vid(),
                name: format!("{}.c{}", p.name, k),
                ty: p.ty.clone(),
            })
            .collect();

        let mut value_map: HashMap<ValueId, Operand> = HashMap::new();

        for (old_p, new_p) in body.params.iter().zip(new_body_params.iter()) {
            value_map.insert(old_p.id, Operand::Value(new_p.id));
        }

        if k == 0 {
            for (hp_id, val) in &header_param_map {
                value_map.insert(*hp_id, val.clone());
            }
        } else {
            let prev_outs = prev_output_values.as_ref().unwrap();
            for (l, val) in prev_outs.iter().enumerate() {
                if l < header.params.len() {
                    value_map.insert(header.params[l].id, val.clone());
                }
            }
        }

        let mut new_insts: Vec<Inst> = Vec::new();
        for inst in &body.insts {
            let new_result = inst.result.map(|_| alloc_vid());
            let new_op = remap_inst_op(inst.op.clone(), &value_map);
            new_insts.push(Inst {
                result: new_result,
                result_name: inst
                    .result_name
                    .as_ref()
                    .map(|n| format!("{}.c{}", n, k)),
                ty: inst.ty.clone(),
                op: new_op,
                metadata: inst.metadata.clone(),
            });
        }

        for (old_inst, new_inst) in body.insts.iter().zip(new_insts.iter()) {
            if let (Some(old_r), Some(new_r)) = (old_inst.result, new_inst.result) {
                value_map.insert(old_r, Operand::Value(new_r));
            }
        }

        let output_values: Vec<Operand> = backedge_args
            .iter()
            .map(|arg| remap_operand(arg.clone(), &value_map))
            .collect();

        let terminator = if is_last {
            let remapped_exit_args: Vec<Operand> = false_args
                .iter()
                .map(|arg| remap_operand(arg.clone(), &value_map))
                .collect();
            Terminator::Br {
                target: exit_id,
                args: remapped_exit_args,
            }
        } else {
            Terminator::Br {
                target: copy_ids[k + 1],
                args: output_values.clone(),
            }
        };

        result.push(BasicBlock {
            id: copy_ids[k],
            name: format!("{}.unroll.{}", body.name, k),
            params: new_body_params,
            insts: new_insts,
            terminator,
        });

        prev_output_values = Some(output_values);
    }

    result
}

fn remap_operand(
    operand: Operand,
    value_map: &HashMap<ValueId, Operand>,
) -> Operand {
    match operand {
        Operand::Value(id) => value_map.get(&id).cloned().unwrap_or(Operand::Value(id)),
        Operand::Const(_) => operand,
    }
}

fn remap_inst_op(
    op: InstOp,
    value_map: &HashMap<ValueId, Operand>,
) -> InstOp {
    let remap = |opr: Operand| remap_operand(opr, value_map);
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty,
            addr_space,
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: remap(base),
            indices: indices.into_iter().map(remap).collect(),
        },
        InstOp::Load { ptr } => InstOp::Load { ptr: remap(ptr) },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: remap(ptr),
            value: remap(value),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: remap(ptr),
            value: remap(value),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: remap(dst),
            src: remap(src),
            size: remap(size),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op,
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred,
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        InstOp::Cast { op: cop, value, to } => InstOp::Cast {
            op: cop,
            value: remap(value),
            to,
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: remap(cond),
            on_true: remap(on_true),
            on_false: remap(on_false),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func,
            args: args.into_iter().map(remap).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic,
            args: args.into_iter().map(remap).collect(),
        },
    }
}
