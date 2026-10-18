//! Full unrolling of small loops with a constant trip count.
//!
//! Tile lowering writes a tile as two nested loops over constant extents (a
//! 2x2 or 4x8 tile, for example). Those loops hide the tile's values behind a
//! temporary array and repeat the loop control on every launch. Unrolling them
//! turns the tile into straight-line code, which exposes the temporary to
//! scalar replacement and folds the index arithmetic into constants.
//!
//! The matcher is deliberately narrow: one header, one latch, a header test of
//! `induction < constant`, an induction step of `induction + constant`, and no
//! edge out of the body except the header's false branch. A loop that does not
//! match is left alone, so a tricky loop costs a missed optimization, not
//! correctness.

use std::collections::{HashMap, HashSet};

use crate::passes::cfg::{natural_loops, preheader};
use crate::{
    BasicBlock, BlockId, BlockParam, Constant, CmpPred, Function, Inst, InstOp, Operand,
    Terminator, ValueId,
};

/// The most iterations a full unroll will copy.
const MAX_TRIP_COUNT: i64 = 8;
/// The most body instructions, times the trip count, a full unroll will create.
const MAX_UNROLLED_INSTRS: usize = 80;
/// The most instructions a function may hold after unrolling, so a nest of
/// constant loops cannot turn one kernel into thousands of instructions.
const MAX_FUNCTION_INSTRS: usize = 600;

pub fn run(mut func: Function) -> Function {
    loop {
        let loops = natural_loops(&func);
        let mut remaining = MAX_FUNCTION_INSTRS.saturating_sub(total_instructions(&func));
        let mut unrolled = false;
        for loop_blocks in &loops {
            let body_insts: usize = func
                .blocks
                .iter()
                .filter(|block| loop_blocks.blocks.contains(&block.id)
                    && block.id != loop_blocks.header)
                .map(|block| block.insts.len())
                .sum();
            let Some(info) = analyze(&func, loop_blocks) else {
                continue;
            };
            if info.trip <= 1 || info.trip > MAX_TRIP_COUNT {
                continue;
            }
            if body_insts * info.trip as usize > MAX_UNROLLED_INSTRS {
                continue;
            }
            // Unrolling pays for itself only where it exposes a private tile
            // array to scalar replacement. A loop that computes without
            // materializing a tile is left as a loop: copying it would grow
            // the shader without removing any work.
            if !writes_private_storage(&func, loop_blocks) {
                continue;
            }
            // The copies of an inner loop are unrolled too, so the cost of
            // this loop is the nest it contains, not one loop body.
            let nested_trips = nested_trip_product(&func, &loops, loop_blocks);
            let projected = (body_insts as i64)
                .saturating_mul(info.trip)
                .saturating_mul(nested_trips);
            let added = projected.saturating_sub(body_insts as i64);
            if added > remaining as i64 {
                continue;
            }
            if unroll_loop(&mut func, &info, &loop_blocks.blocks) {
                remaining = remaining.saturating_sub(added.max(0) as usize);
                unrolled = true;
                break;
            }
        }
        if !unrolled {
            break;
        }
    }
    func
}

/// True when the loop stores into a private alloca.
///
/// That is the tile materialization pattern: the unrolled loop feeds SROA, so
/// the array, its stores and its loads leave the program. Without it there is
/// nothing to remove, and the copy would only cost code size.
fn writes_private_storage(func: &Function, loop_blocks: &crate::passes::cfg::NaturalLoop) -> bool {
    let mut private: HashSet<ValueId> = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Alloca {
                addr_space: crate::AddressSpace::Private,
                ..
            } = &inst.op
                && let Some(result) = inst.result
            {
                private.insert(result);
            }
        }
    }
    let mut addresses: HashSet<ValueId> = HashSet::new();
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        for inst in &block.insts {
            if let InstOp::Gep { base, .. } = &inst.op
                && let Operand::Value(value) = base
                && private.contains(value)
                && let Some(result) = inst.result
            {
                addresses.insert(result);
            }
        }
    }
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        for inst in &block.insts {
            if let InstOp::Store { ptr, .. } = &inst.op
                && let Operand::Value(value) = ptr
                && (private.contains(value) || addresses.contains(value))
            {
                return true;
            }
        }
    }
    false
}

/// The instruction count of the whole function.
fn total_instructions(func: &Function) -> usize {
    func.blocks.iter().map(|block| block.insts.len()).sum()
}

/// How many times the loops nested directly in a body repeat it, together.
fn nested_trip_product(
    func: &Function,
    loops: &[crate::passes::cfg::NaturalLoop],
    outer: &crate::passes::cfg::NaturalLoop,
) -> i64 {
    let mut product = 1i64;
    for inner in loops {
        if inner.header == outer.header
            || inner.blocks.len() >= outer.blocks.len()
            || !inner.blocks.is_subset(&outer.blocks)
        {
            continue;
        }
        let trip = analyze(func, inner).map(|info| info.trip).unwrap_or(1);
        product = product.saturating_mul(trip);
    }
    product
}

/// What the matcher learned about a loop, enough to copy its body.
struct LoopInfo {
    header: BlockId,
    preheader: BlockId,
    /// The header test's true target: the first body block.
    body_entry: BlockId,
    /// Values the header passes to the body, indexed like `params`.
    body_args: Vec<Operand>,
    /// The header's false target, outside the loop.
    exit: BlockId,
    /// Values the header passes out of the loop.
    exit_args: Vec<Operand>,
    params: Vec<BlockParam>,
    init: i64,
    step: i64,
    trip: i64,
}

fn analyze(func: &Function, loop_blocks: &crate::passes::cfg::NaturalLoop) -> Option<LoopInfo> {
    let header_id = loop_blocks.header;
    let header = func.blocks.iter().find(|block| block.id == header_id)?;

    // The header is a lone comparison plus the branch. Anything else could
    // produce a value the body reads, and a clone would have to reproduce it.
    if header.insts.len() != 1 {
        return None;
    }
    let Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return None;
    };
    let Operand::Value(cond_value) = cond else {
        return None;
    };
    let InstOp::Cmp { pred, lhs, rhs } = &header.insts[0].op else {
        return None;
    };
    if !matches!(pred, CmpPred::Slt | CmpPred::Sle) {
        return None;
    }
    if header.insts[0].result != Some(*cond_value) {
        return None;
    }
    let induction = header.params.first()?.id;
    if !matches!(lhs, Operand::Value(value) if *value == induction) {
        return None;
    }
    let Operand::Const(Constant::Int(bound)) = rhs else {
        return None;
    };
    if !loop_blocks.blocks.contains(true_target) || loop_blocks.blocks.contains(false_target) {
        return None;
    }

    // One block outside the loop jumps into the header, and it does so with a
    // constant start value.
    let preheader_id = preheader(func, header_id, &loop_blocks.blocks)?;
    let preheader_block = func.blocks.iter().find(|block| block.id == preheader_id)?;
    let Terminator::Br {
        target: preheader_target,
        args: preheader_args,
    } = &preheader_block.terminator
    else {
        return None;
    };
    if *preheader_target != header_id || preheader_args.len() != header.params.len() {
        return None;
    }
    let Operand::Const(Constant::Int(init)) = preheader_args.first()? else {
        return None;
    };
    let init = *init;

    // One latch, and it increments the induction by a positive constant.
    let mut latch = None;
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) || block.id == header_id {
            continue;
        }
        if let Terminator::Br { target, args } = &block.terminator
            && *target == header_id
        {
            if latch.is_some() || args.len() != header.params.len() {
                return None;
            }
            latch = Some((block.id, args.clone()));
        }
    }
    let (latch_id, latch_args) = latch?;
    let Operand::Value(step_value) = &latch_args[0] else {
        return None;
    };
    let mut step = None;
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        for inst in &block.insts {
            if inst.result != Some(*step_value) {
                continue;
            }
            let InstOp::Bin {
                op: crate::BinOp::Add,
                lhs,
                rhs,
            } = &inst.op
            else {
                return None;
            };
            step = match (lhs, rhs) {
                (Operand::Value(value), Operand::Const(Constant::Int(amount)))
                    if *value == induction && *amount > 0 =>
                {
                    Some(*amount)
                }
                (Operand::Const(Constant::Int(amount)), Operand::Value(value))
                    if *value == induction && *amount > 0 =>
                {
                    Some(*amount)
                }
                _ => return None,
            };
        }
    }
    let step = step?;

    // No edge leaves the body except the header's false branch, and every edge
    // into the loop comes from the preheader.
    let predecessors = crate::passes::cfg::predecessors(func);
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) || block.id == header_id {
            continue;
        }
        if block.id == latch_id {
            // The latch may only jump to the header.
            if !matches!(&block.terminator, Terminator::Br { target, .. } if *target == header_id) {
                return None;
            }
            continue;
        }
        for target in crate::passes::cfg::successors(&block.terminator) {
            if !loop_blocks.blocks.contains(&target) {
                return None;
            }
            if target == header_id {
                // A second edge back to the header is a loop this pass does not
                // model.
                return None;
            }
        }
    }
    for block in &loop_blocks.blocks {
        for predecessor in &predecessors[block] {
            if loop_blocks.blocks.contains(predecessor) {
                continue;
            }
            if *block != header_id || *predecessor != preheader_id {
                return None;
            }
        }
    }

    // The header's arguments and the preheader's start value may name the
    // header's own parameters, constants, or values defined outside the loop.
    // Anything else would need a value that this pass cannot remap.
    let mut param_map: HashMap<ValueId, ()> = HashMap::new();
    for param in &header.params {
        param_map.insert(param.id, ());
    }
    let remappable = |operand: &Operand| match operand {
        Operand::Const(_) => true,
        Operand::Value(value) => {
            !defined_in_loop(func, loop_blocks, *value) || param_map.contains_key(value)
        }
    };
    if !true_args.iter().all(remappable)
        || !false_args.iter().all(remappable)
        || !preheader_args.iter().all(|operand| {
            matches!(operand, Operand::Const(_))
                || matches!(operand, Operand::Value(value) if !defined_in_loop(func, loop_blocks, *value))
        })
    {
        return None;
    }

    // A value the loop defines must not escape it directly: the only way out is
    // the exit argument list, which this pass rewrites.
    let mut loop_values: HashSet<ValueId> = header.params.iter().map(|param| param.id).collect();
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        for param in &block.params {
            loop_values.insert(param.id);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                loop_values.insert(result);
            }
        }
    }
    for block in &func.blocks {
        if loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        for operand in block_inst_operands(&block.insts, &block.terminator) {
            if let Operand::Value(value) = operand
                && loop_values.contains(&value)
            {
                return None;
            }
        }
    }

    let trip = trip_count(init, *bound, step, matches!(pred, CmpPred::Sle))?;
    Some(LoopInfo {
        header: header_id,
        preheader: preheader_id,
        body_entry: *true_target,
        body_args: true_args.clone(),
        exit: *false_target,
        exit_args: false_args.clone(),
        params: header.params.clone(),
        init,
        step,
        trip,
    })
}

fn trip_count(init: i64, bound: i64, step: i64, inclusive: bool) -> Option<i64> {
    let init = init as i128;
    let bound = bound as i128;
    let step = step as i128;
    if step <= 0 {
        return None;
    }
    let distance = if inclusive {
        bound - init + 1
    } else {
        bound - init
    };
    if distance <= 0 {
        return Some(0);
    }
    Some(((distance + step - 1) / step) as i64)
}

fn defined_in_loop(
    func: &Function,
    loop_blocks: &crate::passes::cfg::NaturalLoop,
    value: ValueId,
) -> bool {
    for block in &func.blocks {
        if !loop_blocks.blocks.contains(&block.id) {
            continue;
        }
        if block.params.iter().any(|param| param.id == value) {
            return true;
        }
        if block.insts.iter().any(|inst| inst.result == Some(value)) {
            return true;
        }
    }
    false
}

fn block_inst_operands(insts: &[Inst], terminator: &Terminator) -> Vec<Operand> {
    let mut operands = Vec::new();
    for inst in insts {
        collect_inst_operands(&inst.op, &mut operands);
    }
    collect_terminator_operands(terminator, &mut operands);
    operands
}

fn collect_inst_operands(op: &InstOp, operands: &mut Vec<Operand>) {
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

fn unroll_loop(
    func: &mut Function,
    info: &LoopInfo,
    loop_blocks: &HashSet<BlockId>,
) -> bool {
    if loop_blocks.contains(&func.entry) {
        return false;
    }

    // Body blocks in source order, without the header.
    let body: Vec<BasicBlock> = func
        .blocks
        .iter()
        .filter(|block| loop_blocks.contains(&block.id) && block.id != info.header)
        .cloned()
        .collect();
    if body.is_empty() {
        return false;
    }

    let mut next_value = next_value_id(func);
    let mut next_block = next_block_id(func);
    let mut iteration_entries: Vec<BlockId> = Vec::new();
    let mut iteration_entry_args: Vec<Vec<Operand>> = Vec::new();
    let mut iteration_exit_args: Vec<Vec<Operand>> = Vec::new();
    let mut backedges: Vec<(BlockId, Vec<Operand>)> = Vec::new();
    let mut new_blocks: Vec<BasicBlock> = Vec::new();
    let mut param_map: HashMap<ValueId, Operand> = info
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let value = if index == 0 {
                Operand::Const(Constant::Int(info.init))
            } else {
                preheader_args(func, info.preheader)
                    .get(index)
                    .cloned()
                    .unwrap_or(Operand::Const(Constant::Int(0)))
            };
            (param.id, value)
        })
        .collect();

    for iteration in 0..info.trip {
        let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();
        for original in &body {
            block_map.insert(original.id, BlockId(next_block));
            next_block += 1;
        }
        // Parameters first, so an instruction that reads one can be remapped.
        let mut cloned_params: HashMap<BlockId, Vec<BlockParam>> = HashMap::new();
        for original in &body {
            let params = original
                .params
                .iter()
                .map(|param| {
                    let id = ValueId(next_value);
                    next_value += 1;
                    value_map.insert(param.id, id);
                    BlockParam {
                        id,
                        // The copy needs a name no other copy repeats, because
                        // the backends spell C and WGSL variables by name.
                        name: format!("{}_u{iteration}", param.name),
                        ty: param.ty.clone(),
                    }
                })
                .collect();
            cloned_params.insert(original.id, params);
        }

        for original in &body {
            let mut insts = Vec::with_capacity(original.insts.len());
            for inst in &original.insts {
                let mut cloned = inst.clone();
                cloned.op = remap_op(&inst.op, &param_map, &value_map);
                if let Some(old_result) = inst.result {
                    let new_result = ValueId(next_value);
                    next_value += 1;
                    value_map.insert(old_result, new_result);
                    cloned.result = Some(new_result);
                    cloned.result_name = inst
                        .result_name
                        .as_ref()
                        .map(|name| format!("{name}_u{iteration}"));
                }
                insts.push(cloned);
            }
            let mut terminator =
                remap_terminator(&original.terminator, &param_map, &value_map, &block_map);
            // A branch back to the header is patched once the next iteration
            // exists; keep its arguments for that patch.
            if terminator_targets_header(&terminator, info.header) {
                backedges.push((block_map[&original.id], terminator_args(&terminator)));
            }
            if let Terminator::Br { target, args } = &terminator
                && *target == info.header
            {
                // Placeholder: replaced below. Keep the args in `backedges`.
                terminator = Terminator::Br {
                    target: info.header,
                    args: args.clone(),
                };
            }
            new_blocks.push(BasicBlock {
                id: block_map[&original.id],
                name: format!("{}_u{iteration}", original.name),
                params: cloned_params[&original.id].clone(),
                insts,
                terminator,
            });
        }

        let entry = block_map[&info.body_entry];
        let entry_args = map_operands(&info.body_args, &param_map);
        let exit_args = map_operands(&info.exit_args, &param_map);
        iteration_entries.push(entry);
        iteration_entry_args.push(entry_args);
        iteration_exit_args.push(exit_args);

        // The next iteration starts from the header arguments the body passed
        // back, with the induction advanced by one constant step.
        let backedge = backedges.last().cloned();
        if iteration + 1 < info.trip {
            let Some((_, args)) = backedge else {
                return false;
            };
            if args.len() != info.params.len() {
                return false;
            }
            param_map = info
                .params
                .iter()
                .enumerate()
                .map(|(index, param)| {
                    let value = if index == 0 {
                        Operand::Const(Constant::Int(info.init + (iteration + 1) * info.step))
                    } else {
                        args[index].clone()
                    };
                    (param.id, value)
                })
                .collect();
        }
    }

    // The header branch that left the loop becomes the exit edge of the last
    // iteration, and each earlier back edge enters the next copy.
    for (iteration, (block, _args)) in backedges.iter().enumerate() {
        let replacement = if iteration + 1 < iteration_entries.len() {
            Terminator::Br {
                target: iteration_entries[iteration + 1],
                args: iteration_entry_args[iteration + 1].clone(),
            }
        } else {
            Terminator::Br {
                target: info.exit,
                args: iteration_exit_args[iteration].clone(),
            }
        };
        if let Some(entry) = new_blocks.iter_mut().find(|entry| entry.id == *block) {
            entry.terminator = replacement;
        } else {
            return false;
        }
    }

    // The preheader now enters the first copy.
    if let Some(preheader_block) = func.blocks.iter_mut().find(|block| block.id == info.preheader) {
        preheader_block.terminator = Terminator::Br {
            target: iteration_entries[0],
            args: iteration_entry_args[0].clone(),
        };
    } else {
        return false;
    }

    let insert_at = func
        .blocks
        .iter()
        .position(|block| block.id == info.header)
        .unwrap_or(0);
    func.blocks.retain(|block| !loop_blocks.contains(&block.id));
    let insert_at = insert_at.min(func.blocks.len());
    for (offset, block) in new_blocks.into_iter().enumerate() {
        func.blocks.insert(insert_at + offset, block);
    }
    true
}

fn preheader_args(func: &Function, preheader: BlockId) -> Vec<Operand> {
    func.blocks
        .iter()
        .find(|block| block.id == preheader)
        .and_then(|block| match &block.terminator {
            Terminator::Br { args, .. } => Some(args.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn map_operands(operands: &[Operand], param_map: &HashMap<ValueId, Operand>) -> Vec<Operand> {
    operands
        .iter()
        .map(|operand| match operand {
            Operand::Value(value) => param_map
                .get(value)
                .cloned()
                .unwrap_or(Operand::Value(*value)),
            Operand::Const(_) => operand.clone(),
        })
        .collect()
}

fn remap_operand(
    operand: &Operand,
    param_map: &HashMap<ValueId, Operand>,
    value_map: &HashMap<ValueId, ValueId>,
) -> Operand {
    match operand {
        Operand::Value(value) => param_map
            .get(value)
            .cloned()
            .or_else(|| value_map.get(value).map(|new| Operand::Value(*new)))
            .unwrap_or(Operand::Value(*value)),
        Operand::Const(_) => operand.clone(),
    }
}

fn remap_op(
    op: &InstOp,
    param_map: &HashMap<ValueId, Operand>,
    value_map: &HashMap<ValueId, ValueId>,
) -> InstOp {
    let operand = |operand: &Operand| remap_operand(operand, param_map, value_map);
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty: alloc_ty.clone(),
            addr_space: addr_space.clone(),
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: operand(base),
            indices: indices.iter().map(operand).collect(),
        },
        InstOp::Load { ptr } => InstOp::Load { ptr: operand(ptr) },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: operand(ptr),
            value: operand(value),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: operand(ptr),
            value: operand(value),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: operand(dst),
            src: operand(src),
            size: operand(size),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op: *op,
            lhs: operand(lhs),
            rhs: operand(rhs),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred: *pred,
            lhs: operand(lhs),
            rhs: operand(rhs),
        },
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op: *op,
            value: operand(value),
            to: to.clone(),
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: operand(cond),
            on_true: operand(on_true),
            on_false: operand(on_false),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func: func.clone(),
            args: args.iter().map(operand).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic: intrinsic.clone(),
            args: args.iter().map(operand).collect(),
        },
    }
}


fn remap_terminator(
    terminator: &Terminator,
    param_map: &HashMap<ValueId, Operand>,
    value_map: &HashMap<ValueId, ValueId>,
    block_map: &HashMap<BlockId, BlockId>,
) -> Terminator {
    let operand = |operand: &Operand| remap_operand(operand, param_map, value_map);
    let block = |block: BlockId| block_map.get(&block).copied().unwrap_or(block);
    match terminator {
        Terminator::Br { target, args } => Terminator::Br {
            target: block(*target),
            args: args.iter().map(operand).collect(),
        },
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => Terminator::CondBr {
            cond: operand(cond),
            true_target: block(*true_target),
            true_args: true_args.iter().map(operand).collect(),
            false_target: block(*false_target),
            false_args: false_args.iter().map(operand).collect(),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: operand(value),
            default: block(*default),
            cases: cases.iter().map(|(case, target)| (*case, block(*target))).collect(),
        },
        Terminator::Ret { value } => Terminator::Ret {
            value: value.as_ref().map(operand),
        },
    }
}

fn terminator_targets_header(terminator: &Terminator, header: BlockId) -> bool {
    match terminator {
        Terminator::Br { target, .. } => *target == header,
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => *true_target == header || *false_target == header,
        _ => false,
    }
}

fn terminator_args(terminator: &Terminator) -> Vec<Operand> {
    match terminator {
        Terminator::Br { args, .. } => args.clone(),
        _ => Vec::new(),
    }
}

fn next_value_id(func: &Function) -> u32 {
    let mut max_value = 0u32;
    for param in &func.params {
        max_value = max_value.max(param.id.0);
    }
    for block in &func.blocks {
        for param in &block.params {
            max_value = max_value.max(param.id.0);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                max_value = max_value.max(result.0);
            }
        }
    }
    max_value + 1
}

fn next_block_id(func: &Function) -> u32 {
    func.blocks
        .iter()
        .map(|block| block.id.0)
        .max()
        .map_or(1, |max| max + 1)
}
