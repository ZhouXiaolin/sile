use std::collections::HashMap;

use crate::{BlockId, Function, InstOp, Operand, Terminator, ValueId};

/// Fold trivial block-parameter aliases:
/// if all incoming edge arguments for a block parameter are the same SSA value,
/// replace uses of that parameter and remove it from the block signature.
pub fn run(mut func: Function) -> Function {
    loop {
        let replacements = collect_trivial_param_aliases(&func);
        if replacements.is_empty() {
            break;
        }
        rewrite_function_operands(&mut func, &replacements);
        remove_aliased_block_params(&mut func, &replacements);
    }
    func
}

fn collect_trivial_param_aliases(func: &Function) -> HashMap<ValueId, ValueId> {
    let mut replacements = HashMap::new();
    for block in &func.blocks {
        for (index, param) in block.params.iter().enumerate() {
            let incoming = incoming_args_at(func, block.id, index);
            if incoming.is_empty() {
                continue;
            }
            let mut normalized = Vec::with_capacity(incoming.len());
            let mut all_eligible = true;
            for (pred, op) in incoming {
                match op {
                    Operand::Value(id) if id == param.id => {
                        normalized.push(id);
                    }
                    Operand::Value(id) if is_safe_alias_source(func, pred, block.id, id) => {
                        normalized.push(resolve_replacement(id, &replacements));
                    }
                    Operand::Value(_) | Operand::Const(_) => {
                        all_eligible = false;
                        break;
                    }
                }
            }
            if !all_eligible || normalized.is_empty() {
                continue;
            }
            let first = normalized[0];
            if normalized.iter().all(|candidate| *candidate == first) && first != param.id {
                replacements.insert(param.id, first);
                continue;
            }
            let non_self = normalized
                .iter()
                .copied()
                .filter(|value| *value != param.id)
                .collect::<Vec<_>>();
            if !non_self.is_empty() {
                let candidate = non_self[0];
                if non_self.iter().all(|value| *value == candidate) {
                    replacements.insert(param.id, candidate);
                }
            }
        }
    }
    replacements
}

fn incoming_args_at(func: &Function, target: BlockId, index: usize) -> Vec<(BlockId, Operand)> {
    let mut incoming = Vec::new();
    for block in &func.blocks {
        match &block.terminator {
            Terminator::Br { target: succ, args } if *succ == target => {
                if let Some(arg) = args.get(index) {
                    incoming.push((block.id, arg.clone()));
                }
            }
            Terminator::CondBr {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                if *true_target == target
                    && let Some(arg) = true_args.get(index)
                {
                    incoming.push((block.id, arg.clone()));
                }
                if *false_target == target
                    && let Some(arg) = false_args.get(index)
                {
                    incoming.push((block.id, arg.clone()));
                }
            }
            _ => {}
        }
    }
    incoming
}

fn is_safe_alias_source(func: &Function, pred: BlockId, target: BlockId, value: ValueId) -> bool {
    if func.params.iter().any(|param| param.id == value) {
        return true;
    }
    if func
        .blocks
        .iter()
        .any(|block| block.params.iter().any(|param| param.id == value))
    {
        return true;
    }
    let Some(pred_block) = func.blocks.iter().find(|block| block.id == pred) else {
        return false;
    };
    if pred == target {
        return false;
    }
    pred_block
        .insts
        .iter()
        .any(|inst| inst.result == Some(value))
}

fn rewrite_function_operands(func: &mut Function, replacements: &HashMap<ValueId, ValueId>) {
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            inst.op = rewrite_op(inst.op.clone(), replacements);
        }
        block.terminator = rewrite_terminator(block.terminator.clone(), replacements);
    }
}

fn rewrite_op(op: InstOp, replacements: &HashMap<ValueId, ValueId>) -> InstOp {
    let rewrite = |operand: Operand| rewrite_operand(operand, replacements);
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
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op,
            value: rewrite(value),
            to,
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

fn rewrite_terminator(
    terminator: Terminator,
    replacements: &HashMap<ValueId, ValueId>,
) -> Terminator {
    let rewrite = |operand: Operand| rewrite_operand(operand, replacements);
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

fn rewrite_operand(operand: Operand, replacements: &HashMap<ValueId, ValueId>) -> Operand {
    match operand {
        Operand::Value(value) => Operand::Value(resolve_replacement(value, replacements)),
        Operand::Const(_) => operand,
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

fn remove_aliased_block_params(func: &mut Function, replacements: &HashMap<ValueId, ValueId>) {
    let mut removals = HashMap::new();
    for block in &func.blocks {
        let mut indices = block
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| replacements.contains_key(&param.id).then_some(index))
            .collect::<Vec<_>>();
        if !indices.is_empty() {
            indices.sort_unstable();
            removals.insert(block.id, indices);
        }
    }

    if removals.is_empty() {
        return;
    }

    for block in &mut func.blocks {
        if let Some(indices) = removals.get(&block.id) {
            for index in indices.iter().rev() {
                block.params.remove(*index);
            }
        }
    }

    for block in &mut func.blocks {
        match &mut block.terminator {
            Terminator::Br { target, args } => {
                if let Some(indices) = removals.get(target) {
                    for index in indices.iter().rev() {
                        args.remove(*index);
                    }
                }
            }
            Terminator::CondBr {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                if let Some(indices) = removals.get(true_target) {
                    for index in indices.iter().rev() {
                        true_args.remove(*index);
                    }
                }
                if let Some(indices) = removals.get(false_target) {
                    for index in indices.iter().rev() {
                        false_args.remove(*index);
                    }
                }
            }
            Terminator::Switch { .. } | Terminator::Ret { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BasicBlock, BlockId, BlockParam, CmpPred, Constant, Function, Inst, InstOp, Operand,
        Terminator, Type, ValueId,
    };

    #[test]
    fn folds_trivial_loop_body_param_aliases() {
        let func = Function {
            name: "phi_alias".into(),
            params: vec![],
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    name: "entry".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Br {
                        target: BlockId(1),
                        args: vec![Operand::Const(Constant::Int(0))],
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    name: "header".into(),
                    params: vec![BlockParam {
                        id: ValueId(1),
                        name: "i".into(),
                        ty: Type::I64,
                    }],
                    insts: vec![Inst {
                        result: Some(ValueId(2)),
                        result_name: Some("cond".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: CmpPred::Slt,
                            lhs: Operand::Value(ValueId(1)),
                            rhs: Operand::Const(Constant::Int(4)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::CondBr {
                        cond: Operand::Value(ValueId(2)),
                        true_target: BlockId(2),
                        true_args: vec![Operand::Value(ValueId(1))],
                        false_target: BlockId(3),
                        false_args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(2),
                    name: "body".into(),
                    params: vec![BlockParam {
                        id: ValueId(3),
                        name: "row_alias".into(),
                        ty: Type::I64,
                    }],
                    insts: vec![Inst {
                        result: Some(ValueId(4)),
                        result_name: Some("next".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: crate::BinOp::Add,
                            lhs: Operand::Value(ValueId(3)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::Br {
                        target: BlockId(1),
                        args: vec![Operand::Value(ValueId(4))],
                    },
                },
                BasicBlock {
                    id: BlockId(3),
                    name: "exit".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Ret { value: None },
                },
            ],
            entry: BlockId(0),
            metadata: vec![],
        };

        let out = super::run(func);
        let body = out
            .blocks
            .iter()
            .find(|block| block.id == BlockId(2))
            .unwrap();
        assert!(body.params.is_empty());
        let header = out
            .blocks
            .iter()
            .find(|block| block.id == BlockId(1))
            .unwrap();
        let Terminator::CondBr { true_args, .. } = &header.terminator else {
            panic!("expected condbr");
        };
        assert!(true_args.is_empty());
    }
}
