//! Scalar replacement for private aggregate storage.
//!
//! After a small tile loop is unrolled, a tile's `alloca` is only read and
//! written through constant indices. The array then holds nothing the SSA
//! values do not already hold: every store records a value and every load
//! reads the value of the store that reached it. This pass proves that by a
//! forward dataflow over the CFG and replaces the loads with those values, so
//! the array, its stores and its loads disappear.
//!
//! A load the dataflow cannot resolve, an index that is not constant, or any
//! other use of the pointer aborts the whole allocation. The pass never
//! inserts a phi, so it cannot turn a correct program into a wrong one.

use std::collections::{HashMap, HashSet};

use crate::passes::cfg::{predecessors, reverse_postorder};
use crate::{
    AddressSpace, BlockId, Function, InstOp, Operand, Type, ValueId,
};

pub fn run(mut func: Function) -> Function {
    loop {
        let candidates: Vec<ValueId> = func
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(|inst| match &inst.op {
                InstOp::Alloca {
                    addr_space: AddressSpace::Private,
                    ..
                } => inst.result,
                _ => None,
            })
            .collect();

        let mut replaced_any = false;
        for alloca in candidates {
            if promote(&mut func, alloca) {
                replaced_any = true;
                break;
            }
        }
        if !replaced_any {
            break;
        }
    }
    func
}

/// One memory access the promotion understands.
enum Access {
    Load {
        slot: usize,
        result: ValueId,
    },
    Store {
        slot: usize,
        value: Operand,
    },
}

/// The dimension extents of an allocation's element array, outermost first.
pub(crate) fn array_dims(ty: &Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

fn slot_stride(dims: &[usize], index: usize) -> usize {
    dims.iter().skip(index + 1).product()
}

/// The slot a constant-indexed access names, with the allocation's bounds.
fn slot_of(indices: &[Operand], dims: &[usize]) -> Option<usize> {
    if indices.len() != dims.len() {
        return None;
    }
    let mut slot = 0usize;
    for (position, (index, dim)) in indices.iter().zip(dims).enumerate() {
        let Operand::Const(crate::Constant::Int(value)) = index else {
            return None;
        };
        let value = usize::try_from(*value).ok()?;
        if value >= *dim {
            return None;
        }
        slot += value * slot_stride(dims, position);
    }
    Some(slot)
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
    let dims = array_dims(&alloc_ty);
    if dims.is_empty() {
        return false;
    }
    let total: usize = dims.iter().product();
    if total == 0 {
        return false;
    }

    // Every use of the allocation must be a constant-indexed load or store;
    // every use of the address must be one of those two instructions.
    let mut accesses: HashMap<BlockId, Vec<Access>> = HashMap::new();
    let mut allowed_values: HashSet<ValueId> = HashSet::from([alloca]);
    let mut gep_instructions: Vec<ValueId> = Vec::new();
    let mut load_results: HashSet<ValueId> = HashSet::new();

    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else { continue };
            if let InstOp::Gep { base, indices } = &inst.op
                && matches!(base, Operand::Value(value) if *value == alloca)
            {
                let Some(slot) = slot_of(indices, &dims) else {
                    return false;
                };
                allowed_values.insert(result);
                gep_instructions.push(result);
                let _ = slot;
            }
        }
    }

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                InstOp::Load { ptr } if points_into(ptr, &allowed_values, alloca) => {
                    let Some(result) = inst.result else {
                        return false;
                    };
                    let Some(slot) = slot_of_pointer(ptr, &dims, &func) else {
                        return false;
                    };
                    accesses.entry(block.id).or_default().push(Access::Load {
                        slot,
                        result,
                    });
                    load_results.insert(result);
                }
                InstOp::Store { ptr, value } if points_into(ptr, &allowed_values, alloca) => {
                    if operand_mentions(value, &allowed_values) {
                        return false;
                    }
                    let Some(slot) = slot_of_pointer(ptr, &dims, &func) else {
                        return false;
                    };
                    accesses.entry(block.id).or_default().push(Access::Store {
                        slot,
                        value: value.clone(),
                    });
                }
                _ => {}
            }
        }
    }

    // A pointer value this allocation creates must appear only as a load or a
    // store address; anything else is an escape this promotion cannot follow.
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                // A load of another pointer has nothing to do with this
                // allocation; a load through one of its addresses is fine.
                InstOp::Load { .. } => {}
                InstOp::Store { ptr, value } => {
                    if operand_mentions(value, &allowed_values) {
                        return false;
                    }
                    if points_into(ptr, &allowed_values, alloca)
                        && slot_of_pointer(ptr, &dims, func).is_none()
                    {
                        return false;
                    }
                }
                op => {
                    for value in &allowed_values {
                        if *value == alloca {
                            // The alloca itself is defined here.
                            continue;
                        }
                        if mentions_value(op, *value) {
                            return false;
                        }
                    }
                }
            }
        }
        for value in &allowed_values {
            if terminator_mentions(&block.terminator, *value) {
                return false;
            }
        }
    }

    // Forward dataflow: the value each slot holds at the start of each block.
    let order = reverse_postorder(func);
    let predecessors = predecessors(func);
    let mut in_values: HashMap<BlockId, HashMap<usize, Operand>> = HashMap::new();
    let mut out_values: HashMap<BlockId, HashMap<usize, Operand>> = HashMap::new();
    for block in &order {
        in_values.insert(*block, HashMap::new());
    }

    loop {
        let mut changed = false;
        for block_id in &order {
            let mut entry: Option<HashMap<usize, Operand>> = None;
            for predecessor in &predecessors[block_id] {
                let Some(predecessor_out) = out_values.get(predecessor) else {
                    // A block outside the reachable order cannot provide a
                    // value; the allocation is not promoted.
                    return false;
                };
                entry = Some(match entry {
                    None => predecessor_out.clone(),
                    Some(current) => current
                        .into_iter()
                        .filter(|(slot, value)| predecessor_out.get(slot) == Some(value))
                        .collect(),
                });
            }
            let entry = entry.unwrap_or_default();
            if in_values.get(block_id) != Some(&entry) {
                in_values.insert(*block_id, entry.clone());
                changed = true;
            }
            let mut current = entry;
            for access in accesses.get(block_id).map(Vec::as_slice).unwrap_or(&[]) {
                match access {
                    Access::Store { slot, value } => {
                        current.insert(*slot, value.clone());
                    }
                    Access::Load { .. } => {}
                }
            }
            if out_values.get(block_id) != Some(&current) {
                out_values.insert(*block_id, current);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Second walk: every load must resolve to a value the dataflow proved.
    let mut replacements: HashMap<ValueId, Operand> = HashMap::new();
    for block in &func.blocks {
        let Some(start_values) = in_values.get(&block.id) else {
            return false;
        };
        let mut current = start_values.clone();
        for access in accesses.get(&block.id).map(Vec::as_slice).unwrap_or(&[]) {
            match access {
                Access::Store { slot, value } => {
                    let value = resolve(value.clone(), &replacements);
                    current.insert(*slot, value);
                }
                Access::Load { slot, result } => {
                    let Some(operand) = current.get(slot).cloned() else {
                        return false;
                    };
                    let operand = resolve(operand, &replacements);
                    replacements.insert(*result, operand);
                }
            }
        }
    }

    // Apply: rewrite every remaining use, then drop the memory instructions.
    let mut store_positions: HashMap<BlockId, HashSet<usize>> = HashMap::new();
    for block in &func.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            if let InstOp::Store { ptr, .. } = &inst.op
                && points_into(ptr, &allowed_values, alloca)
            {
                store_positions
                    .entry(block.id)
                    .or_default()
                    .insert(index);
            }
        }
    }

    for block in &mut func.blocks {
        let stores = store_positions.remove(&block.id).unwrap_or_default();
        let mut insts = Vec::with_capacity(block.insts.len());
        for (index, inst) in block.insts.drain(..).enumerate() {
            if stores.contains(&index) {
                continue;
            }
            let mut inst = inst;
            inst.op = rewrite_op(inst.op, &replacements);
            if matches!(
                &inst.op,
                InstOp::Alloca { .. } | InstOp::Gep { .. } | InstOp::Load { .. }
            ) && inst.result.is_some()
                && (inst.result == Some(alloca)
                    || gep_instructions.contains(&inst.result.unwrap())
                    || load_results.contains(&inst.result.unwrap()))
            {
                continue;
            }
            insts.push(inst);
        }
        block.insts = insts;
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }

    true
}

fn slot_of_pointer(ptr: &Operand, dims: &[usize], func: &Function) -> Option<usize> {
    let Operand::Value(value) = ptr else {
        return None;
    };
    for block in &func.blocks {
        for inst in &block.insts {
            if inst.result == Some(*value)
                && let InstOp::Gep { indices, .. } = &inst.op
            {
                return slot_of(indices, dims);
            }
        }
    }
    None
}

fn points_into(ptr: &Operand, allowed: &HashSet<ValueId>, alloca: ValueId) -> bool {
    match ptr {
        Operand::Value(value) => *value == alloca || allowed.contains(value),
        Operand::Const(_) => false,
    }
}

fn operand_mentions(operand: &Operand, values: &HashSet<ValueId>) -> bool {
    matches!(operand, Operand::Value(value) if values.contains(value))
}

fn mentions_value(op: &InstOp, value: ValueId) -> bool {
    let mut operands = Vec::new();
    collect_op_operands(op, &mut operands);
    operands
        .iter()
        .any(|operand| matches!(operand, Operand::Value(used) if *used == value))
}

fn terminator_mentions(terminator: &crate::Terminator, value: ValueId) -> bool {
    let mut operands = Vec::new();
    collect_terminator_operands(terminator, &mut operands);
    operands
        .iter()
        .any(|operand| matches!(operand, Operand::Value(used) if *used == value))
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

fn collect_terminator_operands(terminator: &crate::Terminator, operands: &mut Vec<Operand>) {
    match terminator {
        crate::Terminator::Br { args, .. } => operands.extend(args.iter().cloned()),
        crate::Terminator::CondBr {
            cond,
            true_args,
            false_args,
            ..
        } => {
            operands.push(cond.clone());
            operands.extend(true_args.iter().cloned());
            operands.extend(false_args.iter().cloned());
        }
        crate::Terminator::Switch { value, .. } => operands.push(value.clone()),
        crate::Terminator::Ret { value } => {
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
    terminator: crate::Terminator,
    replacements: &HashMap<ValueId, Operand>,
) -> crate::Terminator {
    let rewrite = |operand: Operand| resolve(operand, replacements);
    match terminator {
        crate::Terminator::Br { target, args } => crate::Terminator::Br {
            target,
            args: args.into_iter().map(rewrite).collect(),
        },
        crate::Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => crate::Terminator::CondBr {
            cond: rewrite(cond),
            true_target,
            true_args: true_args.into_iter().map(rewrite).collect(),
            false_target,
            false_args: false_args.into_iter().map(rewrite).collect(),
        },
        crate::Terminator::Switch {
            value,
            default,
            cases,
        } => crate::Terminator::Switch {
            value: rewrite(value),
            default,
            cases,
        },
        crate::Terminator::Ret { value } => crate::Terminator::Ret {
            value: value.map(rewrite),
        },
    }
}
