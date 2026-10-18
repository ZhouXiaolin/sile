use std::collections::HashMap;

use crate::{
    BasicBlock, BinOp, BlockId, Constant, Function, Inst, InstOp, Operand, Terminator, ValueId,
};

/// Common-subexpression elimination over the CFG.
///
/// The pass keeps the expression scope backend-independent: it only deduplicates
/// explicit SSA expressions and never invents target-specific helpers.
///
/// Availability is computed by a forward must-analysis, not by a dominator
/// walk. A pure expression is available at a join only if every predecessor
/// agrees on it, and a load is available only until a write can have happened.
/// The sibling arm of a branch is exactly the case a tree walk gets wrong: the
/// entry's load is still in scope at the join even though the other arm wrote
/// through the same pointer.
pub fn run(mut func: Function) -> Function {
    let available = available_at_entry(&func);
    let mut replacements = HashMap::<ValueId, ValueId>::new();
    for block in &mut func.blocks {
        // A value available at the block entry is a definition from a
        // dominating block, so it is in scope for every instruction here. A
        // value defined in this block is added as the walk passes its
        // definition. The map is carried forward rather than rebuilt from the
        // instructions kept so far, because a rebuild would put back a load
        // that an earlier store in this same block killed.
        let mut visible = available.get(&block.id).cloned().unwrap_or_default();
        let old_insts = std::mem::take(&mut block.insts);
        let mut new_insts: Vec<Inst> = Vec::with_capacity(old_insts.len());
        for inst in old_insts {
            let rewritten = rewrite_inst(inst, &replacements);

            // A write kills every load, including one the block entry still
            // offered, so the kill happens before the lookup.
            if writes_memory(&rewritten.op) {
                visible.retain(|key, _| !key.starts_with("load:"));
            }

            if rewritten.metadata.is_empty()
                && let Some(key) = expr_key(&rewritten.op)
            {
                if let Some(existing) = visible.get(&key).copied() {
                    if let Some(result) = rewritten.result {
                        // A value offered by the entry state may itself have
                        // been replaced while an earlier block was rewritten;
                        // the use must name the value that still exists.
                        let existing = resolve(existing, &replacements);
                        replacements.insert(result, existing);
                        continue;
                    }
                } else if let Some(result) = rewritten.result {
                    visible.insert(key, result);
                }
            }
            new_insts.push(rewritten);
        }
        block.insts = new_insts;
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }
    // A later replacement can make an earlier rewrite point at a value the
    // pass removed, so every operand is resolved against the final map.
    for block in &mut func.blocks {
        block.insts = std::mem::take(&mut block.insts)
            .into_iter()
            .map(|inst| rewrite_inst(inst, &replacements))
            .collect();
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }
    func
}

/// Follow a replacement chain to the value that still has a definition.
fn resolve(value: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut value = value;
    let mut hops = 0;
    while let Some(next) = replacements.get(&value) {
        value = *next;
        hops += 1;
        if hops > 64 {
            break;
        }
    }
    value
}

/// Expressions available at the entry of every block, keyed by expression.
///
/// A pure expression defined in a block is available to every block that
/// definition dominates. A load is available only until a store can have
/// happened, and a sibling arm of a branch is exactly where a dominator walk
/// gets that wrong: the load before the diamond is still "in scope" at the
/// join, but the other arm overwrote the memory it names. So availability is a
/// forward must-analysis over the CFG, where a join keeps only what every
/// predecessor agrees on, and a store kills every load.
///
/// The entry of a block is computed separately from its exit. The exit includes
/// the block's own definitions; the entry must not, or an instruction would be
/// folded into itself and deleted.
fn available_at_entry(func: &Function) -> HashMap<BlockId, HashMap<String, ValueId>> {
    let block_ids: Vec<BlockId> = func.blocks.iter().map(|block| block.id).collect();
    let preds = compute_predecessors(func);
    let blocks: HashMap<BlockId, &BasicBlock> =
        func.blocks.iter().map(|block| (block.id, block)).collect();

    let mut entries = HashMap::<BlockId, HashMap<String, ValueId>>::new();
    let mut exits = HashMap::<BlockId, HashMap<String, ValueId>>::new();
    for block_id in &block_ids {
        entries.insert(*block_id, HashMap::new());
        exits.insert(*block_id, HashMap::new());
    }

    let transfer = |state: &HashMap<String, ValueId>, block: &BasicBlock| {
        let mut out = state.clone();
        for inst in &block.insts {
            if writes_memory(&inst.op) {
                out.retain(|key, _| !key.starts_with("load:"));
            }
            if inst.metadata.is_empty()
                && let Some(key) = expr_key(&inst.op)
                && let Some(result) = inst.result
            {
                out.entry(key).or_insert(result);
            }
        }
        out
    };

    loop {
        let mut changed = false;
        for block_id in &block_ids {
            let mut incoming: Option<HashMap<String, ValueId>> = None;
            for pred in preds.get(block_id).map(Vec::as_slice).unwrap_or(&[]) {
                let pred_out = &exits[pred];
                incoming = Some(match incoming {
                    None => pred_out.clone(),
                    Some(current) => current
                        .into_iter()
                        .filter(|(key, value)| pred_out.get(key) == Some(value))
                        .collect(),
                });
            }
            let entry_state = if *block_id == func.entry {
                HashMap::new()
            } else {
                incoming.unwrap_or_default()
            };
            let block = blocks.get(block_id).expect("block id came from the function");
            let exit_state = transfer(&entry_state, block);

            if entries.get(block_id) != Some(&entry_state) {
                entries.insert(*block_id, entry_state);
                changed = true;
            }
            if exits.get(block_id) != Some(&exit_state) {
                exits.insert(*block_id, exit_state);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    entries
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
        // A vector load reads memory exactly as a scalar load does, so it takes
        // the same key prefix and the same store kill. Naming it `intrinsic:`
        // would have let a store in between go unnoticed.
        InstOp::Intrinsic {
            intrinsic: crate::Intrinsic::VecLoad { .. },
            args,
        } => Some(format!(
            "load:vec:[{}]",
            args.iter().map(operand_key).collect::<Vec<_>>().join(",")
        )),
        InstOp::Intrinsic { intrinsic, args } if is_pure_intrinsic(intrinsic) => Some(format!(
            "intrinsic:{intrinsic:?}:[{}]",
            args.iter().map(operand_key).collect::<Vec<_>>().join(",")
        )),
        InstOp::Load { ptr } => Some(format!("load:{}", operand_key(ptr))),
        InstOp::Alloca { .. }
        | InstOp::Store { .. }
        | InstOp::AtomicAdd { .. }
        | InstOp::Memcpy { .. }
        | InstOp::Call { .. }
        | InstOp::Intrinsic { .. } => None,
    }
}

/// Does this instruction change what a later load may read?
///
/// A store, an atomic, a memcpy and a call can all write memory. `VecStore`
/// writes too. A barrier writes nothing itself, but it makes the writes of
/// every other invocation visible, so a load may not cross it either. All of
/// these are reasons a cached load must be dropped, and none of them is a
/// reason to keep a cached arithmetic expression.
fn writes_memory(op: &InstOp) -> bool {
    match op {
        InstOp::Store { .. } | InstOp::AtomicAdd { .. } | InstOp::Memcpy { .. } | InstOp::Call { .. } => {
            true
        }
        InstOp::Intrinsic { intrinsic, .. } => matches!(
            intrinsic,
            crate::Intrinsic::Barrier { .. } | crate::Intrinsic::VecStore { .. }
        ),
        _ => false,
    }
}

fn is_pure_intrinsic(intrinsic: &crate::Intrinsic) -> bool {
    !writes_memory(&InstOp::Intrinsic {
        intrinsic: intrinsic.clone(),
        args: Vec::new(),
    })
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
    if matches!(
        op,
        BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor
    ) && lhs > rhs
    {
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
