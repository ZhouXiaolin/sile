use std::collections::HashSet;

use crate::{BinOp, BlockId, CmpPred, Constant, Function, InstOp, Operand, Terminator, ValueId};

/// Canonicalize loop-induction backedge updates so the step expression uses
/// the loop header induction parameter directly.
///
/// Example:
///   %next = add %row_alias, 1   ; row_alias is known alias of %row
/// becomes:
///   %next = add %row, 1
pub fn run(mut func: Function) -> Function {
    let mut rewrites = Vec::<(ValueId, ValueId)>::new();

    for header in &func.blocks {
        let Terminator::CondBr {
            cond: Operand::Value(cond_id),
            true_target,
            ..
        } = &header.terminator
        else {
            continue;
        };
        let Some(cmp_inst) = get_inst_by_result(&func, *cond_id) else {
            continue;
        };
        let InstOp::Cmp { pred, lhs, .. } = &cmp_inst.op else {
            continue;
        };
        if !matches!(
            pred,
            CmpPred::Slt | CmpPred::Sle | CmpPred::Sgt | CmpPred::Sge
        ) {
            continue;
        }
        let Operand::Value(loop_param) = lhs else {
            continue;
        };
        let Some(induction_index) = header.params.iter().position(|param| param.id == *loop_param)
        else {
            continue;
        };

        let true_args = match &header.terminator {
            Terminator::CondBr { true_args, .. } => true_args,
            _ => unreachable!(),
        };
        let aliases =
            collect_loop_aliases(&func, header.id, *true_target, true_args, *loop_param);
        let backedges = find_loop_backedge_blocks(&func, header.id, *true_target);

        for backedge in backedges {
            let Some(arg) = header_arg_from_predecessor(&func, backedge, header.id, induction_index)
            else {
                continue;
            };
            let Operand::Value(step_value_id) = arg else {
                continue;
            };
            let Some(step_inst) = get_inst_by_result(&func, step_value_id) else {
                continue;
            };
            let InstOp::Bin { op, lhs, rhs } = &step_inst.op else {
                continue;
            };
            let lhs_alias = operand_alias_value(lhs, &aliases);
            let rhs_alias = operand_alias_value(rhs, &aliases);
            let uses_alias = lhs_alias.is_some() || rhs_alias.is_some();
            let is_step = matches!(
                (op, lhs, rhs),
                (BinOp::Add, Operand::Value(_), Operand::Const(Constant::Int(_)))
                    | (BinOp::Add, Operand::Const(Constant::Int(_)), Operand::Value(_))
                    | (BinOp::Sub, Operand::Value(_), Operand::Const(Constant::Int(_)))
            );
            if !uses_alias || !is_step {
                continue;
            }
            if let Some(alias_id) = lhs_alias
                && alias_id != *loop_param
            {
                rewrites.push((alias_id, *loop_param));
            }
            if let Some(alias_id) = rhs_alias
                && alias_id != *loop_param
            {
                rewrites.push((alias_id, *loop_param));
            }
        }
    }

    if rewrites.is_empty() {
        return func;
    }

    for (from, to) in rewrites {
        rewrite_value_uses(&mut func, from, to);
    }

    func
}

fn collect_loop_aliases(
    func: &Function,
    header_id: BlockId,
    true_target: BlockId,
    true_args: &[Operand],
    induction_param: ValueId,
) -> HashSet<ValueId> {
    let mut aliases = HashSet::from([induction_param]);
    let mut reachable = HashSet::new();
    let mut stack = vec![true_target];
    while let Some(block_id) = stack.pop() {
        if block_id == header_id || !reachable.insert(block_id) {
            continue;
        }
        let Some(block) = get_block(func, block_id) else {
            continue;
        };
        match &block.terminator {
            Terminator::Br { target, .. } => stack.push(*target),
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => {
                stack.push(*true_target);
                stack.push(*false_target);
            }
            Terminator::Switch { default, cases, .. } => {
                stack.push(*default);
                for (_, target) in cases {
                    stack.push(*target);
                }
            }
            Terminator::Ret { .. } => {}
        }
    }

    if let Some(entry_block) = get_block(func, true_target) {
        for (param, arg) in entry_block.params.iter().zip(true_args.iter()) {
            if let Operand::Value(src_id) = arg
                && aliases.contains(src_id)
            {
                aliases.insert(param.id);
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in &func.blocks {
            if !reachable.contains(&block.id) {
                continue;
            }
            match &block.terminator {
                Terminator::Br { target, args } => {
                    changed |= propagate_aliases(func, *target, args, &mut aliases);
                }
                Terminator::CondBr {
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                    ..
                } => {
                    changed |= propagate_aliases(func, *true_target, true_args, &mut aliases);
                    changed |= propagate_aliases(func, *false_target, false_args, &mut aliases);
                }
                Terminator::Switch { default, cases, .. } => {
                    changed |= propagate_aliases(func, *default, &[], &mut aliases);
                    for (_, target) in cases {
                        changed |= propagate_aliases(func, *target, &[], &mut aliases);
                    }
                }
                Terminator::Ret { .. } => {}
            }
        }
    }

    aliases
}

fn propagate_aliases(
    func: &Function,
    target: BlockId,
    args: &[Operand],
    aliases: &mut HashSet<ValueId>,
) -> bool {
    let Some(target_block) = get_block(func, target) else {
        return false;
    };
    let mut changed = false;
    for (param, arg) in target_block.params.iter().zip(args.iter()) {
        if let Operand::Value(src_id) = arg
            && aliases.contains(src_id)
            && aliases.insert(param.id)
        {
            changed = true;
        }
    }
    changed
}

fn find_loop_backedge_blocks(func: &Function, header_id: BlockId, true_target: BlockId) -> Vec<BlockId> {
    func.blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            Terminator::Br { target, .. }
                if *target == header_id
                    && block_reaches(func, true_target, block.id, &mut HashSet::new()) =>
            {
                Some(block.id)
            }
            _ => None,
        })
        .collect()
}

fn header_arg_from_predecessor(
    func: &Function,
    pred: BlockId,
    header: BlockId,
    index: usize,
) -> Option<Operand> {
    let pred_block = get_block(func, pred)?;
    match &pred_block.terminator {
        Terminator::Br { target, args } if *target == header => args.get(index).cloned(),
        _ => None,
    }
}

fn operand_alias_value(operand: &Operand, aliases: &HashSet<ValueId>) -> Option<ValueId> {
    let Operand::Value(id) = operand else {
        return None;
    };
    aliases.contains(id).then_some(*id)
}

fn rewrite_value_uses(func: &mut Function, from: ValueId, to: ValueId) {
    if from == to {
        return;
    }
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            inst.op = rewrite_op(inst.op.clone(), from, to);
        }
        block.terminator = rewrite_terminator(block.terminator.clone(), from, to);
    }
}

fn rewrite_op(op: InstOp, from: ValueId, to: ValueId) -> InstOp {
    let rewrite = |operand: Operand| rewrite_operand(operand, from, to);
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty,
            addr_space,
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: rewrite(base),
            indices: indices.into_iter().map(rewrite).collect(),
        },
        InstOp::Load { ptr } => InstOp::Load { ptr: rewrite(ptr) },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: rewrite(ptr),
            value: rewrite(value),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: rewrite(ptr),
            value: rewrite(value),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: rewrite(dst),
            src: rewrite(src),
            size: rewrite(size),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        InstOp::Cast { op, value, to: ty } => InstOp::Cast {
            op,
            value: rewrite(value),
            to: ty,
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: rewrite(cond),
            on_true: rewrite(on_true),
            on_false: rewrite(on_false),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func,
            args: args.into_iter().map(rewrite).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic,
            args: args.into_iter().map(rewrite).collect(),
        },
    }
}

fn rewrite_terminator(terminator: Terminator, from: ValueId, to: ValueId) -> Terminator {
    let rewrite = |operand: Operand| rewrite_operand(operand, from, to);
    match terminator {
        Terminator::Br { target, args } => Terminator::Br {
            target,
            args: args.into_iter().map(rewrite).collect(),
        },
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => Terminator::CondBr {
            cond: rewrite(cond),
            true_target,
            true_args: true_args.into_iter().map(rewrite).collect(),
            false_target,
            false_args: false_args.into_iter().map(rewrite).collect(),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: rewrite(value),
            default,
            cases,
        },
        Terminator::Ret { value } => Terminator::Ret {
            value: value.map(rewrite),
        },
    }
}

fn rewrite_operand(operand: Operand, from: ValueId, to: ValueId) -> Operand {
    match operand {
        Operand::Value(id) if id == from => Operand::Value(to),
        _ => operand,
    }
}

fn get_block(func: &Function, id: BlockId) -> Option<&crate::BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn get_inst_by_result(func: &Function, id: ValueId) -> Option<&crate::Inst> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(id))
}

fn block_reaches(
    func: &Function,
    start: BlockId,
    goal: BlockId,
    visiting: &mut HashSet<BlockId>,
) -> bool {
    if start == goal {
        return true;
    }
    if !visiting.insert(start) {
        return false;
    }
    let reaches = match get_block(func, start).map(|block| &block.terminator) {
        Some(Terminator::Br { target, .. }) => block_reaches(func, *target, goal, visiting),
        Some(Terminator::CondBr {
            true_target,
            false_target,
            ..
        }) => {
            block_reaches(func, *true_target, goal, visiting)
                || block_reaches(func, *false_target, goal, visiting)
        }
        Some(Terminator::Switch { default, cases, .. }) => {
            block_reaches(func, *default, goal, visiting)
                || cases
                    .iter()
                    .any(|(_, target)| block_reaches(func, *target, goal, visiting))
        }
        _ => false,
    };
    visiting.remove(&start);
    reaches
}
