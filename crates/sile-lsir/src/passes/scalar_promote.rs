//! Promotion of private scalar storage to SSA values.
//!
//! The lowering keeps every loop-carried scalar in an `alloca` and writes it
//! back on each iteration: the induction variable, and every scalar the tile
//! loops carry. A backend that emits memory in place reads and writes private
//! memory once per iteration for a value that never needs it. This pass turns
//! each such slot into a block parameter — the block-argument form of a phi —
//! so the slot, its loads and its stores disappear from the IR.
//!
//! The pass inserts block parameters where the standard phi placement puts
//! them: at the iterated dominance frontier of the stores. It then renames along
//! the dominator tree, which is what makes the result SSA-correct on every
//! path, not only on the one that falls through.
//!
//! Array allocations are [`super::sroa`]'s business; a pointer that escapes, a
//! non-scalar allocation, or a load that no store dominates aborts that one
//! allocation and leaves the function as it was.

use std::collections::{HashMap, HashSet};

use crate::passes::cfg::{dominators, predecessors, reverse_postorder};
use crate::{
    AddressSpace, BlockId, BlockParam, Function, InstOp, Operand, Terminator, Type, ValueId,
};

pub fn run(mut func: Function) -> Function {
    loop {
        let candidates = scalar_allocas(&func);
        let mut promoted_any = false;
        for alloca in candidates {
            if promote(&mut func, alloca) {
                promoted_any = true;
                break;
            }
        }
        if !promoted_any {
            break;
        }
    }
    func
}

/// Whether a value of this type fits one register on every target.
fn is_scalar(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I1 | Type::I32 | Type::I64 | Type::F16 | Type::F32 | Type::F64
    )
}

/// Every private allocation of one scalar, in block order.
fn scalar_allocas(func: &Function) -> Vec<ValueId> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.op {
            InstOp::Alloca {
                alloc_ty,
                addr_space: AddressSpace::Private,
            } if is_scalar(alloc_ty) => inst.result,
            _ => None,
        })
        .collect()
}

/// The id no value in the function uses yet.
fn next_value_id(func: &Function) -> u32 {
    let mut highest = 0;
    for param in &func.params {
        highest = highest.max(param.id.0);
    }
    for block in &func.blocks {
        for param in &block.params {
            highest = highest.max(param.id.0);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                highest = highest.max(result.0);
            }
        }
    }
    highest + 1
}

/// One access this pass can follow.
enum Access {
    Load { result: ValueId },
    Store { value: Operand },
}

/// Try to promote one `alloca`. False means the allocation was not provable,
/// and the function is unchanged.
fn promote(func: &mut Function, alloca: ValueId) -> bool {
    let Some(alloc_ty) = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find_map(|inst| match &inst.op {
            InstOp::Alloca { alloc_ty, .. } if inst.result == Some(alloca) => {
                Some(alloc_ty.clone())
            }
            _ => None,
        })
    else {
        return false;
    };

    // The address must be used as one, and only as the slot of a load or a
    // store. A Gep, a cast or an argument to anything else is a use this pass
    // cannot follow, so the allocation stays in memory.
    let mut accesses: HashMap<BlockId, Vec<Access>> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                InstOp::Load { ptr } => {
                    if operand_is(ptr, alloca) {
                        let Some(result) = inst.result else {
                            return false;
                        };
                        accesses
                            .entry(block.id)
                            .or_default()
                            .push(Access::Load { result });
                    }
                }
                InstOp::Store { ptr, value } => {
                    if operand_is(ptr, alloca) {
                        if mentions(value, alloca) {
                            return false;
                        }
                        accesses.entry(block.id).or_default().push(Access::Store {
                            value: value.clone(),
                        });
                    } else if mentions(value, alloca) {
                        return false;
                    }
                }
                op => {
                    if mentions_op(op, alloca) {
                        return false;
                    }
                }
            }
        }
        if mentions_terminator(&block.terminator, alloca) {
            return false;
        }
    }

    if !accesses.values().any(|list| {
        list.iter()
            .any(|access| matches!(access, Access::Load { .. }))
    }) {
        // Nothing reads the slot, so there is nothing to promote: the stores
        // are dead and `Dce` owns them.
        return false;
    }

    let order = reverse_postorder(func);
    if order.len() != func.blocks.len() {
        // An unreachable block can hold a side edge into the loop; leaving it
        // in place while promoting the rest is not worth the risk.
        return false;
    }
    let dominators = dominators(func);
    let idoms = immediate_dominators(func, &dominators);
    let frontiers = dominance_frontiers(func, &idoms);

    let definitions: Vec<BlockId> = func
        .blocks
        .iter()
        .filter(|block| {
            accesses
                .get(&block.id)
                .is_some_and(|list| list.iter().any(|a| matches!(a, Access::Store { .. })))
        })
        .map(|block| block.id)
        .collect();

    let phi_blocks = placement(func, &definitions, &frontiers);
    let phi_blocks = prune_to_live_in(func, &accesses, phi_blocks);
    if !promotion_is_provable(func, &accesses, &phi_blocks) {
        return false;
    }

    // Insert one block parameter per placement block, appended so the edge
    // arguments keep the order they already have.
    let mut phi_values: HashMap<BlockId, ValueId> = HashMap::new();
    let next_id = next_value_id(func);
    let base = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(alloca))
        .and_then(|inst| inst.result_name.clone())
        .unwrap_or_else(|| format!("v{}", alloca.0));
    for (offset, block_id) in phi_blocks.iter().enumerate() {
        let id = ValueId(next_id + offset as u32);
        let Some(block) = func.blocks.iter_mut().find(|block| block.id == *block_id) else {
            return false;
        };
        block.params.push(BlockParam {
            id,
            name: format!("{base}_p{}", block_id.0),
            ty: alloc_ty.clone(),
        });
        phi_values.insert(*block_id, id);
    }

    let children = dominator_children(func, &idoms);
    let mut replacements: HashMap<ValueId, Operand> = HashMap::new();
    let mut stack: Vec<Operand> = Vec::new();
    rename_block(
        func,
        func.entry,
        &accesses,
        &phi_values,
        &children,
        &mut stack,
        &mut replacements,
    );

    // Every load and store on this slot is now redundant: each load's result
    // is a replacement, and the values each store wrote travel on the edges.
    for block in &mut func.blocks {
        block.insts.retain(|inst| {
            !(matches!(inst.op, InstOp::Alloca { .. }) && inst.result == Some(alloca))
                && !(matches!(&inst.op, InstOp::Load { ptr } if operand_is(ptr, alloca)))
                && !(matches!(&inst.op, InstOp::Store { ptr, .. } if operand_is(ptr, alloca)))
        });
        for inst in &mut block.insts {
            inst.op = rewrite_op(inst.op.clone(), &replacements);
        }
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }

    true
}

/// Whether the slot holds a value on entry to each block.
///
/// A block is assigned on entry only when every path into it stored: one
/// unassigned predecessor makes the value on entry undefined. That is a
/// must-analysis, so it starts from "assigned" everywhere but the entry and
/// narrows as the paths are walked. Starting from "unassigned" instead would
/// never rise above it: the loop body's own back edge keeps every block inside
/// the loop unassigned forever, and a loop the entry alone initializes would
/// look unprovable.
fn assigned_on_entry(
    func: &Function,
    accesses: &HashMap<BlockId, Vec<Access>>,
) -> HashMap<BlockId, bool> {
    let order = reverse_postorder(func);
    let predecessors = predecessors(func);
    let mut in_assigned: HashMap<BlockId, bool> = HashMap::new();
    let mut out_assigned: HashMap<BlockId, bool> = HashMap::new();
    for block in &order {
        let initial = *block != func.entry;
        in_assigned.insert(*block, initial);
        out_assigned.insert(*block, initial);
    }

    loop {
        let mut changed = false;
        for block_id in &order {
            let entry = if *block_id == func.entry {
                false
            } else {
                predecessors
                    .get(block_id)
                    .map(|preds| {
                        !preds.is_empty()
                            && preds
                                .iter()
                                .all(|pred| out_assigned.get(pred).copied().unwrap_or(false))
                    })
                    .unwrap_or(false)
            };
            if in_assigned.get(block_id) != Some(&entry) {
                in_assigned.insert(*block_id, entry);
                changed = true;
            }
            let stored = accesses
                .get(block_id)
                .is_some_and(|list| list.iter().any(|a| matches!(a, Access::Store { .. })));
            let exit = entry || stored;
            if out_assigned.get(block_id) != Some(&exit) {
                out_assigned.insert(*block_id, exit);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    in_assigned
}

/// Whether every load is reached by a store, and every edge into a placement
/// block has a value to carry.
///
/// Both are the same fact about the slot: a value exists on a path exactly
/// when a store ran on it. Checking before the rewrite keeps the function
/// unchanged when the promotion is not provable.
fn promotion_is_provable(
    func: &Function,
    accesses: &HashMap<BlockId, Vec<Access>>,
    phi_blocks: &[BlockId],
) -> bool {
    let in_assigned = assigned_on_entry(func, accesses);

    for block in &func.blocks {
        let Some(list) = accesses.get(&block.id) else {
            continue;
        };
        let mut assigned = in_assigned.get(&block.id).copied().unwrap_or(false);
        for access in list {
            match access {
                Access::Store { .. } => assigned = true,
                Access::Load { .. } => {
                    if !assigned {
                        return false;
                    }
                }
            }
        }
    }

    let phi_set: HashSet<BlockId> = phi_blocks.iter().copied().collect();
    if phi_set.is_empty() {
        return true;
    }
    // A block parameter cannot be passed through a `switch` edge, which
    // carries no arguments in this IR.
    for block in &func.blocks {
        if matches!(block.terminator, Terminator::Switch { .. })
            && phi_set.contains(&block.id)
        {
            return false;
        }
    }
    if phi_set.contains(&func.entry) {
        return false;
    }
    for block in &func.blocks {
        let mut exits: Vec<BlockId> = Vec::new();
        match &block.terminator {
            Terminator::Br { target, .. } => exits.push(*target),
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => {
                exits.push(*true_target);
                exits.push(*false_target);
            }
            Terminator::Switch { .. } | Terminator::Ret { .. } => {}
        }
        if !exits.iter().any(|target| phi_set.contains(target)) {
            continue;
        }
        let stored = accesses
            .get(&block.id)
            .is_some_and(|list| list.iter().any(|a| matches!(a, Access::Store { .. })));
        if !stored && !in_assigned.get(&block.id).copied().unwrap_or(false) {
            return false;
        }
    }
    true
}

/// The block that dominates `block` and no other dominator does.
fn immediate_dominators(
    func: &Function,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> HashMap<BlockId, BlockId> {
    let mut idoms = HashMap::new();
    for block in &func.blocks {
        let Some(doms) = dominators.get(&block.id) else {
            continue;
        };
        let mut best: Option<(usize, BlockId)> = None;
        for candidate in doms.iter().filter(|candidate| **candidate != block.id) {
            let depth = dominators
                .get(candidate)
                .map(HashSet::len)
                .unwrap_or(usize::MAX);
            if best.is_none_or(|(best_depth, _)| depth > best_depth) {
                best = Some((depth, *candidate));
            }
        }
        if let Some((_, idom)) = best {
            idoms.insert(block.id, idom);
        }
    }
    idoms
}

fn dominance_frontiers(
    func: &Function,
    idoms: &HashMap<BlockId, BlockId>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let predecessors = predecessors(func);
    let mut frontiers: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for block in &func.blocks {
        let preds = &predecessors[&block.id];
        if preds.len() < 2 {
            continue;
        }
        let Some(idom) = idoms.get(&block.id).copied() else {
            continue;
        };
        for pred in preds {
            let mut runner = *pred;
            let mut hops = 0;
            while runner != idom {
                frontiers.entry(runner).or_default().insert(block.id);
                let Some(next) = idoms.get(&runner).copied() else {
                    break;
                };
                if next == runner {
                    break;
                }
                runner = next;
                hops += 1;
                if hops > func.blocks.len() {
                    break;
                }
            }
        }
    }
    frontiers
}

/// The blocks that need a block parameter: the iterated dominance frontier of
/// the stores.
fn placement(
    func: &Function,
    definitions: &[BlockId],
    frontiers: &HashMap<BlockId, HashSet<BlockId>>,
) -> Vec<BlockId> {
    let mut placed: HashSet<BlockId> = HashSet::new();
    let mut worklist: Vec<BlockId> = definitions.to_vec();
    let mut scheduled: HashSet<BlockId> = definitions.iter().copied().collect();
    while let Some(block) = worklist.pop() {
        let Some(blocks) = frontiers.get(&block) else {
            continue;
        };
        for candidate in blocks {
            if !placed.insert(*candidate) {
                continue;
            }
            if scheduled.insert(*candidate) {
                worklist.push(*candidate);
            }
        }
    }
    // Keep the block order so the parameters appear where a reader expects.
    func.blocks
        .iter()
        .map(|block| block.id)
        .filter(|block| placed.contains(block))
        .collect()
}

/// Drop placement blocks the slot is not live on entry for.
///
/// The frontier of a store can reach far past the region the value lives in: a
/// slot that is allocated and used inside one loop body has its frontier reach
/// the enclosing loop header, where the slot holds nothing yet. A block
/// parameter there would have to carry a value that does not exist. SSA
/// construction prunes exactly these blocks, using liveness: a block gets a
/// parameter only when the slot is live on entry to it.
fn prune_to_live_in(func: &Function, accesses: &HashMap<BlockId, Vec<Access>>, mut blocks: Vec<BlockId>) -> Vec<BlockId> {
    let order = reverse_postorder(func);
    let successors_of = |id: BlockId| -> Vec<BlockId> {
        func.blocks
            .iter()
            .find(|block| block.id == id)
            .map(|block| match &block.terminator {
                Terminator::Br { target, .. } => vec![*target],
                Terminator::CondBr {
                    true_target,
                    false_target,
                    ..
                } => vec![*true_target, *false_target],
                Terminator::Switch { default, cases, .. } => {
                    let mut targets = vec![*default];
                    targets.extend(cases.iter().map(|(_, target)| *target));
                    targets
                }
                Terminator::Ret { .. } => Vec::new(),
            })
            .unwrap_or_default()
    };

    // Live-in: a block is live on entry when a use below it reads the slot
    // before any store writes it.
    let mut live_in: HashMap<BlockId, bool> = func
        .blocks
        .iter()
        .map(|block| (block.id, false))
        .collect();
    // Live-out, computed the same way from the successors.
    let mut live_out: HashMap<BlockId, bool> = func
        .blocks
        .iter()
        .map(|block| (block.id, false))
        .collect();

    // A block defines the slot in itself when its first access is a store.
    let block_defines = |id: BlockId| -> bool {
        accesses
            .get(&id)
            .and_then(|list| list.first())
            .is_some_and(|access| matches!(access, Access::Store { .. }))
    };

    loop {
        let mut changed = false;
        for block_id in order.iter().rev() {
            let out = successors_of(*block_id)
                .iter()
                .any(|successor| live_in.get(successor).copied().unwrap_or(false));
            if live_out.get(block_id) != Some(&out) {
                live_out.insert(*block_id, out);
                changed = true;
            }
            let has_load = accesses
                .get(block_id)
                .is_some_and(|list| list.iter().any(|a| matches!(a, Access::Load { .. })));
            let inside = has_load || out;
            let entry = if block_defines(*block_id) {
                false
            } else {
                inside
            };
            if live_in.get(block_id) != Some(&entry) {
                live_in.insert(*block_id, entry);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    blocks.retain(|block| live_in.get(block).copied().unwrap_or(false));
    blocks
}

fn dominator_children(
    func: &Function,
    idoms: &HashMap<BlockId, BlockId>,
) -> HashMap<BlockId, Vec<BlockId>> {
    let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for block in &func.blocks {
        if let Some(idom) = idoms.get(&block.id) {
            children.entry(*idom).or_default().push(block.id);
        }
    }
    children
}

fn rename_block(
    func: &mut Function,
    block: BlockId,
    accesses: &HashMap<BlockId, Vec<Access>>,
    phi_values: &HashMap<BlockId, ValueId>,
    children: &HashMap<BlockId, Vec<BlockId>>,
    stack: &mut Vec<Operand>,
    replacements: &mut HashMap<ValueId, Operand>,
) {
    let depth = stack.len();
    if let Some(phi) = phi_values.get(&block) {
        stack.push(Operand::Value(*phi));
    }
    if let Some(list) = accesses.get(&block) {
        for access in list {
            match access {
                Access::Store { value } => stack.push(resolve(value.clone(), replacements)),
                Access::Load { result } => {
                    if let Some(top) = stack.last().cloned() {
                        replacements.insert(*result, top);
                    }
                }
            }
        }
    }

    // Every edge into a placement block carries the value this block leaves
    // behind. Both arms of a branch are filled, so a target reached twice
    // receives the same argument on each edge.
    let Some(block_data) = func.blocks.iter().find(|candidate| candidate.id == block) else {
        stack.truncate(depth);
        return;
    };
    let terminator = block_data.terminator.clone();
    let mut edges: Vec<BlockId> = Vec::new();
    match &terminator {
        Terminator::Br { target, .. } => edges.push(*target),
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => {
            edges.push(*true_target);
            edges.push(*false_target);
        }
        Terminator::Switch { .. } | Terminator::Ret { .. } => {}
    }
    let mut filled: HashSet<BlockId> = HashSet::new();
    for target in edges {
        if !phi_values.contains_key(&target) || !filled.insert(target) {
            continue;
        }
        let Some(value) = stack.last().cloned() else {
            continue;
        };
        append_edge_arg(func, block, target, value);
    }

    let kids = children.get(&block).cloned().unwrap_or_default();
    for child in kids {
        rename_block(
            func,
            child,
            accesses,
            phi_values,
            children,
            stack,
            replacements,
        );
    }
    stack.truncate(depth);
}

/// Append one argument to every edge from `from` to `target`.
fn append_edge_arg(func: &mut Function, from: BlockId, target: BlockId, value: Operand) {
    let Some(block) = func.blocks.iter_mut().find(|block| block.id == from) else {
        return;
    };
    match &mut block.terminator {
        Terminator::Br {
            target: succ,
            args,
        } if *succ == target => args.push(value),
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == target {
                true_args.push(value.clone());
            }
            if *false_target == target {
                false_args.push(value);
            }
        }
        _ => {}
    }
}

fn operand_is(operand: &Operand, value: ValueId) -> bool {
    matches!(operand, Operand::Value(id) if *id == value)
}

fn mentions(operand: &Operand, value: ValueId) -> bool {
    operand_is(operand, value)
}

fn mentions_op(op: &InstOp, value: ValueId) -> bool {
    let mut operands = Vec::new();
    collect_op_operands(op, &mut operands);
    operands.iter().any(|operand| mentions(operand, value))
}

fn mentions_terminator(terminator: &Terminator, value: ValueId) -> bool {
    let mut operands = Vec::new();
    collect_terminator_operands(terminator, &mut operands);
    operands.iter().any(|operand| mentions(operand, value))
}

fn collect_op_operands(op: &InstOp, operands: &mut Vec<Operand>) {
    match op {
        InstOp::Alloca { .. } => {}
        InstOp::Gep { base, indices } => {
            operands.push(base.clone());
            operands.extend(indices.iter().cloned());
        }
        InstOp::Load { ptr } => operands.push(ptr.clone()),
        InstOp::Store { ptr, value } => {
            operands.push(ptr.clone());
            operands.push(value.clone());
        }
        InstOp::AtomicAdd { ptr, value } => {
            operands.push(ptr.clone());
            operands.push(value.clone());
        }
        InstOp::Memcpy { dst, src, size } => {
            operands.push(dst.clone());
            operands.push(src.clone());
            operands.push(size.clone());
        }
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => {
            operands.push(lhs.clone());
            operands.push(rhs.clone());
        }
        InstOp::Cast { value, .. } => operands.push(value.clone()),
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => {
            operands.push(cond.clone());
            operands.push(on_true.clone());
            operands.push(on_false.clone());
        }
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => {
            operands.extend(args.iter().cloned());
        }
    }
}

fn collect_terminator_operands(terminator: &Terminator, operands: &mut Vec<Operand>) {
    match terminator {
        Terminator::Br { args, .. } => operands.extend(args.iter().cloned()),
        Terminator::CondBr {
            cond,
            true_args,
            false_args,
            ..
        } => {
            operands.push(cond.clone());
            operands.extend(true_args.iter().cloned());
            operands.extend(false_args.iter().cloned());
        }
        Terminator::Switch { value, .. } => operands.push(value.clone()),
        Terminator::Ret { value } => {
            if let Some(value) = value {
                operands.push(value.clone());
            }
        }
    }
}

fn resolve(operand: Operand, replacements: &HashMap<ValueId, Operand>) -> Operand {
    let mut operand = operand;
    let mut hops = 0;
    while let Operand::Value(value) = &operand {
        let Some(next) = replacements.get(value) else {
            break;
        };
        operand = next.clone();
        hops += 1;
        if hops > 64 {
            break;
        }
    }
    operand
}

fn rewrite_op(op: InstOp, replacements: &HashMap<ValueId, Operand>) -> InstOp {
    let rewrite = |operand: Operand| resolve(operand, replacements);
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
    replacements: &HashMap<ValueId, Operand>,
) -> Terminator {
    let rewrite = |operand: Operand| resolve(operand, replacements);
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
