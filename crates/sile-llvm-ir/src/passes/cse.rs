use std::collections::{HashMap, HashSet};

use crate::{BinOp, BlockId, Constant, Function, Inst, InstOp, Operand, Terminator, ValueId};

/// Dominator-scoped common-subexpression elimination for side-effect-free LLVM IR ops.
///
/// The pass keeps the expression scope backend-independent: it only deduplicates
/// explicit SSA expressions and never invents target-specific helpers.
pub fn run(mut func: Function) -> Function {
    let dom_tree = compute_dominator_tree(&func);
    if dom_tree.entry.is_none() {
        return func;
    }

    let mut replacements = HashMap::<ValueId, ValueId>::new();
    let empty = HashMap::new();
    process_block(
        dom_tree.entry.unwrap(),
        &dom_tree,
        &mut func,
        &mut replacements,
        &empty,
    );
    func
}

struct DominatorTree {
    entry: Option<BlockId>,
    children: HashMap<BlockId, Vec<BlockId>>,
    block_indices: HashMap<BlockId, usize>,
}

fn process_block(
    block_id: BlockId,
    dom_tree: &DominatorTree,
    func: &mut Function,
    replacements: &mut HashMap<ValueId, ValueId>,
    inherited_exprs: &HashMap<String, ValueId>,
) {
    let Some(&block_idx) = dom_tree.block_indices.get(&block_id) else {
        return;
    };

    let mut expr_to_value = inherited_exprs.clone();
    let block = &mut func.blocks[block_idx];
    let old_insts = std::mem::take(&mut block.insts);
    let mut new_insts = Vec::with_capacity(old_insts.len());
    for inst in old_insts {
        let rewritten = rewrite_inst(inst, replacements);
        if rewritten.metadata.is_empty()
            && let Some(key) = expr_key(&rewritten.op)
        {
            if let Some(existing) = expr_to_value.get(&key).copied() {
                if let Some(result) = rewritten.result {
                    replacements.insert(result, existing);
                    continue;
                }
            } else if let Some(result) = rewritten.result {
                expr_to_value.insert(key, result);
            }
        }
        new_insts.push(rewritten);
    }
    block.insts = new_insts;
    block.terminator = rewrite_terminator(block.terminator.clone(), replacements);

    if let Some(children) = dom_tree.children.get(&block_id).cloned() {
        for child in children {
            process_block(child, dom_tree, func, replacements, &expr_to_value);
        }
    }
}

fn compute_dominator_tree(func: &Function) -> DominatorTree {
    let entry = Some(func.entry);
    let block_indices = func
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect::<HashMap<_, _>>();
    let block_ids = func.blocks.iter().map(|block| block.id).collect::<Vec<_>>();
    let all_blocks = block_ids.iter().copied().collect::<HashSet<_>>();
    let predecessors = compute_predecessors(func);

    let mut doms = HashMap::<BlockId, HashSet<BlockId>>::new();
    for block_id in &block_ids {
        if *block_id == func.entry {
            doms.insert(*block_id, HashSet::from([*block_id]));
        } else {
            doms.insert(*block_id, all_blocks.clone());
        }
    }

    loop {
        let mut changed = false;
        for block_id in &block_ids {
            if *block_id == func.entry {
                continue;
            }
            let preds = predecessors.get(block_id).cloned().unwrap_or_default();
            if preds.is_empty() {
                let new_set = HashSet::from([*block_id]);
                if doms.get(block_id) != Some(&new_set) {
                    doms.insert(*block_id, new_set);
                    changed = true;
                }
                continue;
            }

            let mut pred_iter = preds.into_iter();
            let mut new_set = doms
                .get(&pred_iter.next().unwrap())
                .cloned()
                .unwrap_or_else(HashSet::new);
            for pred in pred_iter {
                let pred_doms = doms.get(&pred).cloned().unwrap_or_default();
                new_set = new_set
                    .intersection(&pred_doms)
                    .copied()
                    .collect::<HashSet<_>>();
            }
            new_set.insert(*block_id);
            if doms.get(block_id) != Some(&new_set) {
                doms.insert(*block_id, new_set);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut children = HashMap::<BlockId, Vec<BlockId>>::new();
    for block_id in &block_ids {
        if *block_id == func.entry {
            continue;
        }
        let Some(idom) = immediate_dominator(*block_id, &doms) else {
            continue;
        };
        children.entry(idom).or_default().push(*block_id);
    }

    DominatorTree {
        entry,
        children,
        block_indices,
    }
}

fn immediate_dominator(
    block_id: BlockId,
    doms: &HashMap<BlockId, HashSet<BlockId>>,
) -> Option<BlockId> {
    let strict_doms = doms
        .get(&block_id)?
        .iter()
        .copied()
        .filter(|dom| *dom != block_id)
        .collect::<Vec<_>>();
    strict_doms.iter().copied().find(|candidate| {
        strict_doms
            .iter()
            .copied()
            .filter(|other| *other != *candidate)
            .all(|other| {
                !doms
                    .get(&other)
                    .is_some_and(|other_doms| other_doms.contains(candidate))
            })
    })
}

fn compute_predecessors(func: &Function) -> HashMap<BlockId, Vec<BlockId>> {
    let mut preds = HashMap::<BlockId, Vec<BlockId>>::new();
    for block in &func.blocks {
        for succ in successors(&block.terminator) {
            preds.entry(succ).or_default().push(block.id);
        }
    }
    preds
}

fn successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        Terminator::Switch { default, cases, .. } => {
            let mut out = Vec::with_capacity(cases.len() + 1);
            out.push(*default);
            out.extend(cases.iter().map(|(_, target)| *target));
            out
        }
        Terminator::Ret { .. } => Vec::new(),
    }
}

fn expr_key(op: &InstOp) -> Option<String> {
    match op {
        InstOp::Gep { base, indices } => Some(format!(
            "gep:{}:[{}]",
            operand_key(base),
            indices
                .iter()
                .map(operand_key)
                .collect::<Vec<_>>()
                .join(",")
        )),
        InstOp::Bin { op, lhs, rhs } => {
            let (lhs, rhs) = canonical_bin_pair(*op, operand_key(lhs), operand_key(rhs));
            Some(format!("bin:{op:?}:{lhs}:{rhs}"))
        }
        InstOp::Cmp { pred, lhs, rhs } => Some(format!(
            "cmp:{pred:?}:{}:{}",
            operand_key(lhs),
            operand_key(rhs)
        )),
        InstOp::Cast { op, value, to } => {
            Some(format!("cast:{op:?}:{}:{to:?}", operand_key(value)))
        }
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => Some(format!(
            "select:{}:{}:{}",
            operand_key(cond),
            operand_key(on_true),
            operand_key(on_false)
        )),
        InstOp::Intrinsic { intrinsic, args } if is_pure_intrinsic(intrinsic) => Some(format!(
            "intrinsic:{intrinsic:?}:[{}]",
            args.iter().map(operand_key).collect::<Vec<_>>().join(",")
        )),
        InstOp::Alloca { .. }
        | InstOp::Load { .. }
        | InstOp::Store { .. }
        | InstOp::AtomicAdd { .. }
        | InstOp::Memcpy { .. }
        | InstOp::Call { .. }
        | InstOp::Intrinsic { .. } => None,
    }
}

fn is_pure_intrinsic(intrinsic: &crate::Intrinsic) -> bool {
    !matches!(intrinsic, crate::Intrinsic::Barrier { .. })
}

fn operand_key(operand: &Operand) -> String {
    match operand {
        Operand::Value(value) => format!("%{}", value.0),
        Operand::Const(constant) => const_key(constant),
    }
}

fn const_key(constant: &Constant) -> String {
    match constant {
        Constant::Int(v) => format!("i{v}"),
        Constant::Float(v) => format!("f{}", v.to_bits()),
        Constant::Bool(v) => format!("b{}", u8::from(*v)),
    }
}

fn canonical_bin_pair(op: BinOp, lhs: String, rhs: String) -> (String, String) {
    if matches!(op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or) && lhs > rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn rewrite_inst(inst: Inst, replacements: &HashMap<ValueId, ValueId>) -> Inst {
    Inst {
        op: rewrite_op(inst.op, replacements),
        ..inst
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

#[cfg(test)]
mod tests {
    use crate::{
        BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst, InstOp, Operand,
        Terminator, Type, ValueId,
    };

    #[test]
    fn eliminates_dominated_duplicate_scalar_exprs_across_blocks() {
        let func = Function {
            name: "global_cse".into(),
            params: vec![],
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    name: "entry".into(),
                    params: vec![],
                    insts: vec![Inst {
                        result: Some(ValueId(0)),
                        result_name: Some("seed".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Mul,
                            lhs: Operand::Const(Constant::Int(7)),
                            rhs: Operand::Const(Constant::Int(8)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::Br {
                        target: BlockId(1),
                        args: vec![Operand::Const(Constant::Bool(true))],
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    name: "header".into(),
                    params: vec![BlockParam {
                        id: ValueId(1),
                        name: "cond".into(),
                        ty: Type::I1,
                    }],
                    insts: vec![],
                    terminator: Terminator::CondBr {
                        cond: Operand::Value(ValueId(1)),
                        true_target: BlockId(2),
                        true_args: vec![],
                        false_target: BlockId(3),
                        false_args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(2),
                    name: "body".into(),
                    params: vec![],
                    insts: vec![Inst {
                        result: Some(ValueId(2)),
                        result_name: Some("dup".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Mul,
                            lhs: Operand::Const(Constant::Int(7)),
                            rhs: Operand::Const(Constant::Int(8)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::Ret {
                        value: Some(Operand::Value(ValueId(2))),
                    },
                },
                BasicBlock {
                    id: BlockId(3),
                    name: "exit".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Ret {
                        value: Some(Operand::Const(Constant::Int(0))),
                    },
                },
            ],
            entry: BlockId(0),
            metadata: vec![],
        };

        let optimized = super::run(func);
        let body = optimized
            .blocks
            .iter()
            .find(|block| block.id == BlockId(2))
            .unwrap();
        assert!(body.insts.is_empty());
        assert_eq!(
            body.terminator,
            Terminator::Ret {
                value: Some(Operand::Value(ValueId(0))),
            }
        );
    }

    #[test]
    fn does_not_reuse_expr_only_available_on_one_branch() {
        let func = Function {
            name: "branch_local".into(),
            params: vec![],
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    name: "entry".into(),
                    params: vec![],
                    insts: vec![Inst {
                        result: Some(ValueId(0)),
                        result_name: Some("cond".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: CmpPred::Eq,
                            lhs: Operand::Const(Constant::Int(1)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::CondBr {
                        cond: Operand::Value(ValueId(0)),
                        true_target: BlockId(1),
                        true_args: vec![],
                        false_target: BlockId(2),
                        false_args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    name: "lhs".into(),
                    params: vec![],
                    insts: vec![Inst {
                        result: Some(ValueId(1)),
                        result_name: Some("lhs_mul".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Mul,
                            lhs: Operand::Const(Constant::Int(3)),
                            rhs: Operand::Const(Constant::Int(9)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::Br {
                        target: BlockId(3),
                        args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(2),
                    name: "rhs".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Br {
                        target: BlockId(3),
                        args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(3),
                    name: "join".into(),
                    params: vec![],
                    insts: vec![Inst {
                        result: Some(ValueId(2)),
                        result_name: Some("join_mul".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Mul,
                            lhs: Operand::Const(Constant::Int(3)),
                            rhs: Operand::Const(Constant::Int(9)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::Ret {
                        value: Some(Operand::Value(ValueId(2))),
                    },
                },
            ],
            entry: BlockId(0),
            metadata: vec![],
        };

        let optimized = super::run(func);
        let join = optimized
            .blocks
            .iter()
            .find(|block| block.id == BlockId(3))
            .unwrap();
        assert_eq!(join.insts.len(), 1);
        assert_eq!(join.insts[0].result, Some(ValueId(2)));
    }
}
