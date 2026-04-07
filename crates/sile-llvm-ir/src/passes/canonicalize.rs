use std::collections::{HashMap, HashSet};

use crate::{BasicBlock, BlockId, Function, Inst, InstOp, Intrinsic, Operand, Terminator, ValueId};

/// Canonical LLVM IR rewrites live here.
///
/// The active implementation starts with a narrow, profitable rewrite: sink
/// materialized pointwise tile-expression loops into their scalar consumers so
/// Tile IR lowering does not need to encode that choice permanently.
pub fn run(mut func: Function) -> Function {
    sink_pointwise_tile_exprs(&mut func);
    compact_rank2_tile_store_loops(&mut func);
    eliminate_dead_insts(&mut func);
    func
}

#[derive(Clone)]
struct PointwiseExprLoop {
    tile: ValueId,
    preheader: BlockId,
    continue_block: BlockId,
    blocks: HashSet<BlockId>,
    body_params: Vec<ValueId>,
    template_insts: Vec<Inst>,
    scalar: Operand,
}

fn sink_pointwise_tile_exprs(func: &mut Function) {
    loop {
        let loops = collect_pointwise_expr_loops(func);
        if loops.is_empty() {
            break;
        }

        let mut changed = false;
        for expr_loop in loops {
            if inline_pointwise_expr_loop(func, &expr_loop) {
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }
}

fn compact_rank2_tile_store_loops(func: &mut Function) {
    loop {
        let mut next_value = next_value_id(func);
        let block_ids: Vec<_> = func.blocks.iter().map(|block| block.id).collect();
        let mut changed = false;
        for block_id in block_ids {
            if compact_rank2_tile_store_loop(func, block_id, &mut next_value) {
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
}

fn compact_rank2_tile_store_loop(
    func: &mut Function,
    row_header_id: BlockId,
    next_value: &mut u32,
) -> bool {
    let Some(row_header_idx) = func
        .blocks
        .iter()
        .position(|block| block.id == row_header_id)
    else {
        return false;
    };
    if !func.blocks[row_header_idx]
        .name
        .starts_with("tile_store_row_header_")
    {
        return false;
    }

    let (
        row_setup_id,
        row_header_true_arg,
        row_setup_row,
        col_header_id,
        body_id,
        row_latch_id,
        body_row_offset_param,
        row_setup_insts,
    ) = {
        let row_header = &func.blocks[row_header_idx];
        let Terminator::CondBr {
            true_target,
            true_args,
            false_args,
            ..
        } = &row_header.terminator
        else {
            return false;
        };
        if true_args.len() != 1 || !false_args.is_empty() {
            return false;
        }
        let row_header_true_arg = true_args[0].clone();
        let Some(row_setup_idx) = func
            .blocks
            .iter()
            .position(|block| block.id == *true_target)
        else {
            return false;
        };
        let row_setup = &func.blocks[row_setup_idx];
        if !row_setup.name.starts_with("tile_store_row_setup_")
            || row_setup.params.len() != 1
            || row_setup.insts.len() != 2
            || count_block_predecessors(func, row_setup.id) != 1
        {
            return false;
        }
        let row_setup_row = row_setup.params[0].id;
        let add_inst = &row_setup.insts[0];
        let mul_inst = &row_setup.insts[1];
        let Some(dst_row) = add_inst.result else {
            return false;
        };
        let Some(row_offset) = mul_inst.result else {
            return false;
        };
        if !matches!(
            &add_inst.op,
            InstOp::Bin {
                op: crate::BinOp::Add,
                lhs,
                rhs,
            } if matches!(lhs, Operand::Value(id) if *id == row_setup_row)
                || matches!(rhs, Operand::Value(id) if *id == row_setup_row)
        ) {
            return false;
        }
        if !matches!(
            &mul_inst.op,
            InstOp::Bin {
                op: crate::BinOp::Mul,
                lhs,
                rhs,
            } if matches!(lhs, Operand::Value(id) if *id == dst_row)
                || matches!(rhs, Operand::Value(id) if *id == dst_row)
        ) {
            return false;
        }
        let Terminator::Br {
            target: col_header_id,
            args,
        } = &row_setup.terminator
        else {
            return false;
        };
        if args.len() != 3
            || !matches!(args[0], Operand::Value(id) if id == row_setup_row)
            || !matches!(args[1], Operand::Value(id) if id == row_offset)
            || !matches!(args[2], Operand::Const(crate::Constant::Int(0)))
        {
            return false;
        }
        let Some(col_header_idx) = func
            .blocks
            .iter()
            .position(|block| block.id == *col_header_id)
        else {
            return false;
        };
        let col_header = &func.blocks[col_header_idx];
        if !col_header.name.starts_with("tile_store_col_header_") || col_header.params.len() != 3 {
            return false;
        }
        let row_offset_param = col_header.params[1].id;
        let Terminator::CondBr {
            true_target: body_id,
            true_args,
            false_target: row_latch_id,
            false_args,
            ..
        } = &col_header.terminator
        else {
            return false;
        };
        if true_args.len() != 3
            || false_args.len() != 1
            || !matches!(true_args[1], Operand::Value(id) if id == row_offset_param)
        {
            return false;
        }
        let Some(body_idx) = func.blocks.iter().position(|block| block.id == *body_id) else {
            return false;
        };
        let body = &func.blocks[body_idx];
        if !body.name.starts_with("tile_store_body_") || body.params.len() != 3 {
            return false;
        }
        let body_row_offset_param = body.params[1].id;
        if !matches!(
            &body.terminator,
            Terminator::Br { target, args }
                if *target == *col_header_id
                    && args.len() == 3
                    && matches!(args[1], Operand::Value(id) if id == body_row_offset_param)
        ) {
            return false;
        }
        let Some(row_latch_idx) = func
            .blocks
            .iter()
            .position(|block| block.id == *row_latch_id)
        else {
            return false;
        };
        let row_latch = &func.blocks[row_latch_idx];
        if !row_latch.name.starts_with("tile_store_row_latch_") {
            return false;
        }
        (
            row_setup.id,
            row_header_true_arg,
            row_setup_row,
            *col_header_id,
            *body_id,
            *row_latch_id,
            body_row_offset_param,
            row_setup.insts.clone(),
        )
    };

    {
        let row_header = &mut func.blocks[row_header_idx];
        let Terminator::CondBr {
            cond,
            true_target: _,
            true_args: _,
            false_target,
            false_args,
        } = row_header.terminator.clone()
        else {
            return false;
        };
        row_header.terminator = Terminator::CondBr {
            cond,
            true_target: col_header_id,
            true_args: vec![row_header_true_arg, Operand::Const(crate::Constant::Int(0))],
            false_target,
            false_args,
        };
    }

    let Some(col_header_idx) = func
        .blocks
        .iter()
        .position(|block| block.id == col_header_id)
    else {
        return false;
    };
    {
        let col_header = &mut func.blocks[col_header_idx];
        col_header.params.remove(1);
        let Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } = col_header.terminator.clone()
        else {
            return false;
        };
        if true_target != body_id || false_target != row_latch_id || true_args.len() != 3 {
            return false;
        }
        col_header.terminator = Terminator::CondBr {
            cond,
            true_target,
            true_args: vec![true_args[0].clone(), true_args[2].clone()],
            false_target,
            false_args,
        };
    }

    let Some(body_idx) = func.blocks.iter().position(|block| block.id == body_id) else {
        return false;
    };
    {
        let body = &mut func.blocks[body_idx];
        let body_row = body.params[0].id;
        body.params.remove(1);

        let mut operand_map = HashMap::new();
        operand_map.insert(row_setup_row, Operand::Value(body_row));
        let mut result_map = HashMap::new();
        let mut cloned_setup = Vec::with_capacity(row_setup_insts.len());
        for inst in &row_setup_insts {
            let cloned = clone_inst(inst, &operand_map, &result_map, next_value);
            if let (Some(old_result), Some(new_result)) = (inst.result, cloned.result) {
                result_map.insert(old_result, new_result);
            }
            cloned_setup.push(cloned);
        }
        let Some(new_row_offset) = row_setup_insts
            .last()
            .and_then(|inst| inst.result)
            .and_then(|result| result_map.get(&result).copied())
        else {
            return false;
        };
        let replacements = HashMap::from([(body_row_offset_param, Operand::Value(new_row_offset))]);
        let mut new_insts = cloned_setup;
        new_insts.extend(
            body.insts
                .clone()
                .into_iter()
                .map(|inst| rewrite_inst_operands(inst, &replacements)),
        );
        body.insts = new_insts;

        let Terminator::Br { target, args } =
            rewrite_terminator_operands(body.terminator.clone(), &replacements)
        else {
            return false;
        };
        if target != col_header_id || args.len() != 3 {
            return false;
        }
        body.terminator = Terminator::Br {
            target,
            args: vec![args[0].clone(), args[2].clone()],
        };
    }

    func.blocks.retain(|block| block.id != row_setup_id);
    true
}

fn collect_pointwise_expr_loops(func: &Function) -> Vec<PointwiseExprLoop> {
    let mut expr_loops = Vec::new();
    let block_lookup: HashMap<_, _> = func.blocks.iter().map(|block| (block.id, block)).collect();

    for (idx, block) in func.blocks.iter().enumerate() {
        let Some(tile) = block.insts.iter().find_map(|inst| match inst.op {
            InstOp::Alloca {
                addr_space: crate::AddressSpace::Private,
                ..
            } => inst.result,
            _ => None,
        }) else {
            continue;
        };

        let Terminator::Br { target: entry, .. } = &block.terminator else {
            continue;
        };

        let Some(entry_block) = block_lookup.get(entry) else {
            continue;
        };
        if !entry_block.name.starts_with("tile_expr_loop_") {
            continue;
        }

        let mut blocks = HashSet::new();
        let mut body_block = None;
        let mut continue_block = None;
        for candidate in func.blocks.iter().skip(idx + 1) {
            if !candidate.name.starts_with("tile_expr_loop_") {
                break;
            }
            if candidate.name.contains("_body_") {
                body_block = Some(candidate);
            }
            if candidate.name.contains("_continue_") {
                continue_block = Some(candidate);
            }
            blocks.insert(candidate.id);
        }

        let (Some(body_block), Some(continue_block)) = (body_block, continue_block) else {
            continue;
        };

        let Some((template_insts, scalar)) = extract_pointwise_expr_body(body_block, tile) else {
            continue;
        };

        expr_loops.push(PointwiseExprLoop {
            tile,
            preheader: block.id,
            continue_block: continue_block.id,
            blocks,
            body_params: body_block.params.iter().map(|param| param.id).collect(),
            template_insts,
            scalar,
        });
    }

    expr_loops
}

fn extract_pointwise_expr_body(body: &BasicBlock, tile: ValueId) -> Option<(Vec<Inst>, Operand)> {
    let (store_idx, store_ptr) =
        body.insts
            .iter()
            .enumerate()
            .find_map(|(idx, inst)| match &inst.op {
                InstOp::Store {
                    ptr: Operand::Value(ptr_id),
                    ..
                } => matches!(
                    body.insts.iter().find(|candidate| candidate.result == Some(*ptr_id)),
                    Some(Inst {
                        op:
                            InstOp::Gep {
                                base: Operand::Value(base_id),
                                ..
                            },
                        ..
                    }) if *base_id == tile
                )
                .then_some((idx, *ptr_id)),
                _ => None,
            })?;

    let InstOp::Store { value, .. } = &body.insts[store_idx].op else {
        return None;
    };

    let template_insts = body.insts[..store_idx]
        .iter()
        .filter(|inst| inst.result != Some(store_ptr))
        .cloned()
        .collect();

    Some((template_insts, value.clone()))
}

fn inline_pointwise_expr_loop(func: &mut Function, expr_loop: &PointwiseExprLoop) -> bool {
    let tile_uses_before = count_value_uses_outside(func, expr_loop.tile, &expr_loop.blocks);
    if tile_uses_before == 0 {
        return false;
    }

    let mut next_value = next_value_id(func);
    let mut changed = false;
    for block in &mut func.blocks {
        if expr_loop.blocks.contains(&block.id) {
            continue;
        }
        if rewrite_block_uses(block, expr_loop, &mut next_value) {
            changed = true;
        }
    }

    if changed {
        eliminate_dead_insts(func);
    }

    let remaining_uses = count_value_uses_outside(func, expr_loop.tile, &expr_loop.blocks);
    if !changed || remaining_uses != 0 {
        return changed;
    }

    remove_expr_loop(func, expr_loop);
    true
}

fn rewrite_block_uses(
    block: &mut BasicBlock,
    expr_loop: &PointwiseExprLoop,
    next_value: &mut u32,
) -> bool {
    let mut defs = HashMap::new();
    for inst in &block.insts {
        if let Some(result) = inst.result {
            defs.insert(result, inst.clone());
        }
    }

    let mut replacements = HashMap::new();
    let mut new_insts = Vec::with_capacity(block.insts.len());
    let mut changed = false;

    for inst in &block.insts {
        let inst = rewrite_inst_operands(inst.clone(), &replacements);
        if let Some(load_result) = inst.result {
            if let Some((cloned_insts, scalar)) =
                inline_load_from_expr_tile(&inst, &defs, expr_loop, next_value, &replacements)
            {
                new_insts.extend(cloned_insts);
                replacements.insert(load_result, scalar);
                changed = true;
                continue;
            }
        }
        new_insts.push(inst);
    }

    if changed {
        block.terminator = rewrite_terminator_operands(block.terminator.clone(), &replacements);
        block.insts = new_insts;
    }

    changed
}

fn inline_load_from_expr_tile(
    load_inst: &Inst,
    defs: &HashMap<ValueId, Inst>,
    expr_loop: &PointwiseExprLoop,
    next_value: &mut u32,
    replacements: &HashMap<ValueId, Operand>,
) -> Option<(Vec<Inst>, Operand)> {
    let InstOp::Load {
        ptr: Operand::Value(ptr_id),
    } = &load_inst.op
    else {
        return None;
    };
    let gep_inst = defs.get(ptr_id)?;
    let InstOp::Gep {
        base: Operand::Value(base_id),
        indices,
    } = &gep_inst.op
    else {
        return None;
    };
    if *base_id != expr_loop.tile || indices.len() != expr_loop.body_params.len() {
        return None;
    }

    let mut operand_map = HashMap::new();
    for (param, index) in expr_loop.body_params.iter().zip(indices.iter()) {
        operand_map.insert(*param, rewrite_operand(index.clone(), replacements));
    }

    let mut result_map = HashMap::new();
    let mut cloned = Vec::with_capacity(expr_loop.template_insts.len());
    for template in &expr_loop.template_insts {
        let cloned_inst = clone_inst(template, &operand_map, &result_map, next_value);
        if let (Some(old_result), Some(new_result)) = (template.result, cloned_inst.result) {
            result_map.insert(old_result, new_result);
        }
        cloned.push(cloned_inst);
    }

    let scalar = remap_operand(expr_loop.scalar.clone(), &operand_map, &result_map);
    Some((cloned, scalar))
}

fn clone_inst(
    inst: &Inst,
    operand_map: &HashMap<ValueId, Operand>,
    result_map: &HashMap<ValueId, ValueId>,
    next_value: &mut u32,
) -> Inst {
    let result = inst.result.map(|_| fresh_value(next_value));
    Inst {
        result,
        result_name: result.map(|value| format!("v{}", value.0)),
        ty: inst.ty.clone(),
        op: remap_inst_op(inst.op.clone(), operand_map, result_map),
        metadata: inst.metadata.clone(),
    }
}

fn remove_expr_loop(func: &mut Function, expr_loop: &PointwiseExprLoop) {
    let Some(preheader_idx) = func
        .blocks
        .iter()
        .position(|block| block.id == expr_loop.preheader)
    else {
        return;
    };
    let Some(continue_idx) = func
        .blocks
        .iter()
        .position(|block| block.id == expr_loop.continue_block)
    else {
        return;
    };

    let continue_block = func.blocks[continue_idx].clone();
    let preheader = &mut func.blocks[preheader_idx];
    preheader
        .insts
        .retain(|inst| inst.result != Some(expr_loop.tile));
    preheader.insts.extend(continue_block.insts);
    preheader.terminator = continue_block.terminator;

    func.blocks
        .retain(|block| !expr_loop.blocks.contains(&block.id));
}

fn eliminate_dead_insts(func: &mut Function) {
    loop {
        let used = collect_used_values(func);
        let mut changed = false;

        for block in &mut func.blocks {
            let before = block.insts.len();
            block.insts.retain(|inst| match inst.result {
                Some(result) if !used.contains(&result) => !is_removable_inst(&inst.op),
                _ => true,
            });
            changed |= before != block.insts.len();
        }

        if !changed {
            break;
        }
    }
}

fn collect_used_values(func: &Function) -> HashSet<ValueId> {
    let mut used = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            collect_inst_uses(&inst.op, &mut used);
        }
        collect_terminator_uses(&block.terminator, &mut used);
    }
    used
}

fn count_value_uses_outside(
    func: &Function,
    value: ValueId,
    excluded_blocks: &HashSet<BlockId>,
) -> usize {
    let mut count = 0usize;
    for block in &func.blocks {
        if excluded_blocks.contains(&block.id) {
            continue;
        }
        for inst in &block.insts {
            count += inst_operands(&inst.op)
                .into_iter()
                .filter(|operand| matches!(operand, Operand::Value(id) if *id == value))
                .count();
        }
        count += terminator_operands(&block.terminator)
            .into_iter()
            .filter(|operand| matches!(operand, Operand::Value(id) if *id == value))
            .count();
    }
    count
}

fn count_block_predecessors(func: &Function, target: BlockId) -> usize {
    func.blocks
        .iter()
        .map(|block| match &block.terminator {
            Terminator::Br { target: succ, .. } => usize::from(*succ == target),
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => usize::from(*true_target == target) + usize::from(*false_target == target),
            Terminator::Switch { default, cases, .. } => {
                usize::from(*default == target)
                    + cases
                        .iter()
                        .filter(|(_, case_target)| *case_target == target)
                        .count()
            }
            Terminator::Ret { .. } => 0,
        })
        .sum()
}

fn collect_inst_uses(op: &InstOp, used: &mut HashSet<ValueId>) {
    for operand in inst_operands(op) {
        if let Operand::Value(value) = operand {
            used.insert(value);
        }
    }
}

fn collect_terminator_uses(terminator: &Terminator, used: &mut HashSet<ValueId>) {
    for operand in terminator_operands(terminator) {
        if let Operand::Value(value) = operand {
            used.insert(value);
        }
    }
}

fn inst_operands(op: &InstOp) -> Vec<Operand> {
    match op {
        InstOp::Alloca { .. } => Vec::new(),
        InstOp::Gep { base, indices } => {
            let mut operands = vec![base.clone()];
            operands.extend(indices.clone());
            operands
        }
        InstOp::Load { ptr } => vec![ptr.clone()],
        InstOp::Store { ptr, value } => vec![ptr.clone(), value.clone()],
        InstOp::AtomicAdd { ptr, value } => vec![ptr.clone(), value.clone()],
        InstOp::Memcpy { dst, src, size } => vec![dst.clone(), src.clone(), size.clone()],
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => {
            vec![lhs.clone(), rhs.clone()]
        }
        InstOp::Cast { value, .. } => vec![value.clone()],
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond.clone(), on_true.clone(), on_false.clone()],
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<Operand> {
    match terminator {
        Terminator::Br { args, .. } => args.clone(),
        Terminator::CondBr {
            cond,
            true_args,
            false_args,
            ..
        } => {
            let mut operands = vec![cond.clone()];
            operands.extend(true_args.clone());
            operands.extend(false_args.clone());
            operands
        }
        Terminator::Switch { value, .. } => vec![value.clone()],
        Terminator::Ret { value } => value.iter().cloned().collect(),
    }
}

fn is_removable_inst(op: &InstOp) -> bool {
    match op {
        InstOp::Store { .. }
        | InstOp::AtomicAdd { .. }
        | InstOp::Memcpy { .. }
        | InstOp::Call { .. } => false,
        InstOp::Intrinsic {
            intrinsic: Intrinsic::Barrier { .. },
            ..
        } => false,
        _ => true,
    }
}

fn rewrite_inst_operands(inst: Inst, replacements: &HashMap<ValueId, Operand>) -> Inst {
    Inst {
        op: remap_inst_op(inst.op, replacements, &HashMap::new()),
        ..inst
    }
}

fn rewrite_terminator_operands(
    terminator: Terminator,
    replacements: &HashMap<ValueId, Operand>,
) -> Terminator {
    match terminator {
        Terminator::Br { target, args } => Terminator::Br {
            target,
            args: args
                .into_iter()
                .map(|operand| rewrite_operand(operand, replacements))
                .collect(),
        },
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => Terminator::CondBr {
            cond: rewrite_operand(cond, replacements),
            true_target,
            true_args: true_args
                .into_iter()
                .map(|operand| rewrite_operand(operand, replacements))
                .collect(),
            false_target,
            false_args: false_args
                .into_iter()
                .map(|operand| rewrite_operand(operand, replacements))
                .collect(),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: rewrite_operand(value, replacements),
            default,
            cases,
        },
        Terminator::Ret { value } => Terminator::Ret {
            value: value.map(|operand| rewrite_operand(operand, replacements)),
        },
    }
}

fn remap_inst_op(
    op: InstOp,
    operand_map: &HashMap<ValueId, Operand>,
    result_map: &HashMap<ValueId, ValueId>,
) -> InstOp {
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty,
            addr_space,
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: remap_operand(base, operand_map, result_map),
            indices: indices
                .into_iter()
                .map(|operand| remap_operand(operand, operand_map, result_map))
                .collect(),
        },
        InstOp::Load { ptr } => InstOp::Load {
            ptr: remap_operand(ptr, operand_map, result_map),
        },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: remap_operand(ptr, operand_map, result_map),
            value: remap_operand(value, operand_map, result_map),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: remap_operand(ptr, operand_map, result_map),
            value: remap_operand(value, operand_map, result_map),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: remap_operand(dst, operand_map, result_map),
            src: remap_operand(src, operand_map, result_map),
            size: remap_operand(size, operand_map, result_map),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op,
            lhs: remap_operand(lhs, operand_map, result_map),
            rhs: remap_operand(rhs, operand_map, result_map),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred,
            lhs: remap_operand(lhs, operand_map, result_map),
            rhs: remap_operand(rhs, operand_map, result_map),
        },
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op,
            value: remap_operand(value, operand_map, result_map),
            to,
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: remap_operand(cond, operand_map, result_map),
            on_true: remap_operand(on_true, operand_map, result_map),
            on_false: remap_operand(on_false, operand_map, result_map),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func,
            args: args
                .into_iter()
                .map(|operand| remap_operand(operand, operand_map, result_map))
                .collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic,
            args: args
                .into_iter()
                .map(|operand| remap_operand(operand, operand_map, result_map))
                .collect(),
        },
    }
}

fn remap_operand(
    operand: Operand,
    operand_map: &HashMap<ValueId, Operand>,
    result_map: &HashMap<ValueId, ValueId>,
) -> Operand {
    match operand {
        Operand::Value(value) => operand_map
            .get(&value)
            .cloned()
            .or_else(|| result_map.get(&value).copied().map(Operand::Value))
            .unwrap_or(Operand::Value(value)),
        Operand::Const(_) => operand,
    }
}

fn rewrite_operand(operand: Operand, replacements: &HashMap<ValueId, Operand>) -> Operand {
    match operand {
        Operand::Value(value) => replacements
            .get(&value)
            .cloned()
            .unwrap_or(Operand::Value(value)),
        Operand::Const(_) => operand,
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

fn fresh_value(next_value: &mut u32) -> ValueId {
    let value = ValueId(*next_value);
    *next_value += 1;
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressSpace, BinOp, BlockParam, Constant, Metadata, Param, Type};

    #[test]
    fn sinks_materialized_pointwise_tile_expr_into_consumers() {
        let func = Function {
            name: "softmax_like".into(),
            params: vec![param(1, "src"), param(2, "bias"), param(3, "dst")],
            blocks: vec![
                block(
                    0,
                    "bb0",
                    vec![],
                    vec![alloca(10)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![int(0)],
                    },
                ),
                block(
                    1,
                    "tile_expr_loop_row_header_1",
                    vec![block_param(11, "row")],
                    vec![icmp_slt(12, value(11), int(2))],
                    Terminator::CondBr {
                        cond: value(12),
                        true_target: BlockId(2),
                        true_args: vec![value(11), int(0)],
                        false_target: BlockId(5),
                        false_args: vec![],
                    },
                ),
                block(
                    2,
                    "tile_expr_loop_col_header_2",
                    vec![block_param(13, "row"), block_param(14, "col")],
                    vec![icmp_slt(15, value(14), int(8))],
                    Terminator::CondBr {
                        cond: value(15),
                        true_target: BlockId(3),
                        true_args: vec![value(13), value(14)],
                        false_target: BlockId(4),
                        false_args: vec![value(13)],
                    },
                ),
                block(
                    3,
                    "tile_expr_loop_body_3",
                    vec![block_param(16, "row"), block_param(17, "col")],
                    vec![
                        gep(18, 1, vec![value(16), value(17)], AddressSpace::Global),
                        load(19, 18),
                        gep(20, 2, vec![value(16), int(0)], AddressSpace::Global),
                        load(21, 20),
                        bin(22, BinOp::Sub, value(19), value(21), Type::F32),
                        intrinsic_exp(23, value(22)),
                        gep(24, 10, vec![value(16), value(17)], AddressSpace::Private),
                        store(24, value(23)),
                        bin(25, BinOp::Add, value(17), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(2),
                        args: vec![value(16), value(25)],
                    },
                ),
                block(
                    4,
                    "tile_expr_loop_row_latch_4",
                    vec![block_param(26, "row")],
                    vec![bin(27, BinOp::Add, value(26), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![value(27)],
                    },
                ),
                block(
                    5,
                    "tile_expr_loop_continue_5",
                    vec![],
                    vec![],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![int(0)],
                    },
                ),
                block(
                    6,
                    "consumer_a",
                    vec![block_param(28, "row")],
                    vec![
                        gep(29, 10, vec![value(28), int(0)], AddressSpace::Private),
                        load(30, 29),
                        bin(31, BinOp::Add, value(30), int(1), Type::F32),
                    ],
                    Terminator::CondBr {
                        cond: bool_const(true),
                        true_target: BlockId(7),
                        true_args: vec![],
                        false_target: BlockId(7),
                        false_args: vec![],
                    },
                ),
                block(
                    7,
                    "consumer_b",
                    vec![],
                    vec![
                        gep(32, 10, vec![int(1), int(2)], AddressSpace::Private),
                        load(33, 32),
                        gep(34, 3, vec![int(0)], AddressSpace::Global),
                        store(34, value(33)),
                    ],
                    Terminator::Ret { value: None },
                ),
            ],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let canonical = run(func);
        let names: Vec<_> = canonical
            .blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect();
        assert!(!names.iter().any(|name| name.starts_with("tile_expr_loop_")));
        assert!(canonical.blocks.iter().all(|block| {
            block
                .insts
                .iter()
                .all(|inst| !matches!(inst.result, Some(ValueId(10))))
        }));
        assert!(
            canonical
                .blocks
                .iter()
                .any(|block| block.name == "consumer_a"
                    && block.insts.iter().all(|inst| !matches!(
                        inst.op,
                        InstOp::Gep {
                            base: Operand::Value(ValueId(10)),
                            ..
                        }
                    )))
        );
        assert!(
            canonical
                .blocks
                .iter()
                .any(|block| block.name == "consumer_b"
                    && block.insts.iter().any(|inst| matches!(
                        inst.op,
                        InstOp::Intrinsic {
                            intrinsic: Intrinsic::Exp,
                            ..
                        }
                    )))
        );
    }

    #[test]
    fn sinks_pointwise_expr_into_loop_body_consumers() {
        let func = Function {
            name: "loop_consumer".into(),
            params: vec![param(1, "src"), param(2, "bias"), param(3, "dst")],
            blocks: vec![
                block(
                    0,
                    "bb0",
                    vec![],
                    vec![alloca(10)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![int(0)],
                    },
                ),
                block(
                    1,
                    "tile_expr_loop_row_header_1",
                    vec![block_param(11, "row")],
                    vec![icmp_slt(12, value(11), int(2))],
                    Terminator::CondBr {
                        cond: value(12),
                        true_target: BlockId(2),
                        true_args: vec![value(11), int(0)],
                        false_target: BlockId(5),
                        false_args: vec![],
                    },
                ),
                block(
                    2,
                    "tile_expr_loop_col_header_2",
                    vec![block_param(13, "row"), block_param(14, "col")],
                    vec![icmp_slt(15, value(14), int(8))],
                    Terminator::CondBr {
                        cond: value(15),
                        true_target: BlockId(3),
                        true_args: vec![value(13), value(14)],
                        false_target: BlockId(4),
                        false_args: vec![value(13)],
                    },
                ),
                block(
                    3,
                    "tile_expr_loop_body_3",
                    vec![block_param(16, "row"), block_param(17, "col")],
                    vec![
                        gep(18, 1, vec![value(16), value(17)], AddressSpace::Global),
                        load(19, 18),
                        gep(20, 2, vec![value(16), int(0)], AddressSpace::Global),
                        load(21, 20),
                        bin(22, BinOp::Sub, value(19), value(21), Type::F32),
                        intrinsic_exp(23, value(22)),
                        gep(24, 10, vec![value(16), value(17)], AddressSpace::Private),
                        store(24, value(23)),
                        bin(25, BinOp::Add, value(17), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(2),
                        args: vec![value(16), value(25)],
                    },
                ),
                block(
                    4,
                    "tile_expr_loop_row_latch_4",
                    vec![block_param(26, "row")],
                    vec![bin(27, BinOp::Add, value(26), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![value(27)],
                    },
                ),
                block(
                    5,
                    "tile_expr_loop_continue_5",
                    vec![],
                    vec![],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![int(0)],
                    },
                ),
                block(
                    6,
                    "tile_store_row_header_6",
                    vec![block_param(28, "row")],
                    vec![icmp_slt(29, value(28), int(2))],
                    Terminator::CondBr {
                        cond: value(29),
                        true_target: BlockId(7),
                        true_args: vec![value(28), int(0)],
                        false_target: BlockId(10),
                        false_args: vec![],
                    },
                ),
                block(
                    7,
                    "tile_store_col_header_7",
                    vec![block_param(30, "row"), block_param(31, "col")],
                    vec![icmp_slt(32, value(31), int(8))],
                    Terminator::CondBr {
                        cond: value(32),
                        true_target: BlockId(8),
                        true_args: vec![value(30), value(31)],
                        false_target: BlockId(9),
                        false_args: vec![value(30)],
                    },
                ),
                block(
                    8,
                    "tile_store_body_8",
                    vec![block_param(33, "row"), block_param(34, "col")],
                    vec![
                        gep(35, 10, vec![value(33), value(34)], AddressSpace::Private),
                        load(36, 35),
                        gep(37, 3, vec![value(33), value(34)], AddressSpace::Global),
                        store(37, value(36)),
                        bin(38, BinOp::Add, value(34), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(7),
                        args: vec![value(33), value(38)],
                    },
                ),
                block(
                    9,
                    "tile_store_row_latch_9",
                    vec![block_param(39, "row")],
                    vec![bin(40, BinOp::Add, value(39), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![value(40)],
                    },
                ),
                block(
                    10,
                    "tile_store_continue_10",
                    vec![],
                    vec![],
                    Terminator::Ret { value: None },
                ),
            ],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let canonical = run(func);
        let names: Vec<_> = canonical
            .blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect();
        assert!(!names.iter().any(|name| name.starts_with("tile_expr_loop_")));
        let store_body = canonical
            .blocks
            .iter()
            .find(|block| block.name == "tile_store_body_8")
            .unwrap();
        assert!(store_body.insts.iter().any(|inst| {
            matches!(
                inst.op,
                InstOp::Intrinsic {
                    intrinsic: Intrinsic::Exp,
                    ..
                }
            )
        }));
        assert!(!store_body.insts.iter().any(|inst| {
            matches!(
                inst.op,
                InstOp::Gep {
                    base: Operand::Value(ValueId(10)),
                    ..
                }
            )
        }));
    }

    #[test]
    fn sinks_pointwise_expr_into_multiple_loop_consumers() {
        let func = Function {
            name: "two_loop_consumers".into(),
            params: vec![
                param(1, "src"),
                param(2, "bias"),
                param(3, "dst"),
                param(4, "tmp"),
            ],
            blocks: vec![
                block(
                    0,
                    "bb0",
                    vec![],
                    vec![alloca(10)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![int(0)],
                    },
                ),
                block(
                    1,
                    "tile_expr_loop_row_header_1",
                    vec![block_param(11, "row")],
                    vec![icmp_slt(12, value(11), int(2))],
                    Terminator::CondBr {
                        cond: value(12),
                        true_target: BlockId(2),
                        true_args: vec![value(11), int(0)],
                        false_target: BlockId(5),
                        false_args: vec![],
                    },
                ),
                block(
                    2,
                    "tile_expr_loop_col_header_2",
                    vec![block_param(13, "row"), block_param(14, "col")],
                    vec![icmp_slt(15, value(14), int(8))],
                    Terminator::CondBr {
                        cond: value(15),
                        true_target: BlockId(3),
                        true_args: vec![value(13), value(14)],
                        false_target: BlockId(4),
                        false_args: vec![value(13)],
                    },
                ),
                block(
                    3,
                    "tile_expr_loop_body_3",
                    vec![block_param(16, "row"), block_param(17, "col")],
                    vec![
                        gep(18, 1, vec![value(16), value(17)], AddressSpace::Global),
                        load(19, 18),
                        gep(20, 2, vec![value(16), int(0)], AddressSpace::Global),
                        load(21, 20),
                        bin(22, BinOp::Sub, value(19), value(21), Type::F32),
                        intrinsic_exp(23, value(22)),
                        gep(24, 10, vec![value(16), value(17)], AddressSpace::Private),
                        store(24, value(23)),
                        bin(25, BinOp::Add, value(17), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(2),
                        args: vec![value(16), value(25)],
                    },
                ),
                block(
                    4,
                    "tile_expr_loop_row_latch_4",
                    vec![block_param(26, "row")],
                    vec![bin(27, BinOp::Add, value(26), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![value(27)],
                    },
                ),
                block(
                    5,
                    "tile_expr_loop_continue_5",
                    vec![],
                    vec![],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![int(0)],
                    },
                ),
                block(
                    6,
                    "reduce_loop_header_6",
                    vec![block_param(28, "col")],
                    vec![icmp_slt(29, value(28), int(8))],
                    Terminator::CondBr {
                        cond: value(29),
                        true_target: BlockId(7),
                        true_args: vec![value(28)],
                        false_target: BlockId(8),
                        false_args: vec![],
                    },
                ),
                block(
                    7,
                    "reduce_loop_body_7",
                    vec![block_param(30, "col")],
                    vec![
                        gep(31, 10, vec![int(0), value(30)], AddressSpace::Private),
                        load(32, 31),
                        gep(33, 4, vec![value(30)], AddressSpace::Global),
                        store(33, value(32)),
                        bin(34, BinOp::Add, value(30), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![value(34)],
                    },
                ),
                block(
                    8,
                    "store_loop_header_8",
                    vec![block_param(35, "col")],
                    vec![icmp_slt(36, value(35), int(8))],
                    Terminator::CondBr {
                        cond: value(36),
                        true_target: BlockId(9),
                        true_args: vec![value(35)],
                        false_target: BlockId(10),
                        false_args: vec![],
                    },
                ),
                block(
                    9,
                    "store_loop_body_9",
                    vec![block_param(37, "col")],
                    vec![
                        gep(38, 10, vec![int(1), value(37)], AddressSpace::Private),
                        load(39, 38),
                        gep(40, 3, vec![int(1), value(37)], AddressSpace::Global),
                        store(40, value(39)),
                        bin(41, BinOp::Add, value(37), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(8),
                        args: vec![value(41)],
                    },
                ),
                block(10, "done", vec![], vec![], Terminator::Ret { value: None }),
            ],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let canonical = run(func);
        assert!(
            !canonical
                .blocks
                .iter()
                .map(|block| block.name.as_str())
                .any(|name| name.starts_with("tile_expr_loop_"))
        );
        for consumer in ["reduce_loop_body_7", "store_loop_body_9"] {
            let block = canonical
                .blocks
                .iter()
                .find(|block| block.name == consumer)
                .unwrap();
            assert!(block.insts.iter().any(|inst| {
                matches!(
                    inst.op,
                    InstOp::Intrinsic {
                        intrinsic: Intrinsic::Exp,
                        ..
                    }
                )
            }));
            assert!(!block.insts.iter().any(|inst| {
                matches!(
                    inst.op,
                    InstOp::Gep {
                        base: Operand::Value(ValueId(10)),
                        ..
                    }
                )
            }));
        }
    }

    #[test]
    fn sinks_chained_pointwise_expr_loops_without_stale_templates() {
        let func = Function {
            name: "chained_expr_loops".into(),
            params: vec![param(1, "src"), param(2, "bias"), param(3, "dst")],
            blocks: vec![
                block(
                    0,
                    "bb0",
                    vec![],
                    vec![alloca(10)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![int(0)],
                    },
                ),
                block(
                    1,
                    "tile_expr_loop_row_header_1",
                    vec![block_param(11, "row")],
                    vec![icmp_slt(12, value(11), int(2))],
                    Terminator::CondBr {
                        cond: value(12),
                        true_target: BlockId(2),
                        true_args: vec![value(11), int(0)],
                        false_target: BlockId(5),
                        false_args: vec![],
                    },
                ),
                block(
                    2,
                    "tile_expr_loop_col_header_2",
                    vec![block_param(13, "row"), block_param(14, "col")],
                    vec![icmp_slt(15, value(14), int(8))],
                    Terminator::CondBr {
                        cond: value(15),
                        true_target: BlockId(3),
                        true_args: vec![value(13), value(14)],
                        false_target: BlockId(4),
                        false_args: vec![value(13)],
                    },
                ),
                block(
                    3,
                    "tile_expr_loop_body_3",
                    vec![block_param(16, "row"), block_param(17, "col")],
                    vec![
                        gep(18, 1, vec![value(16), value(17)], AddressSpace::Global),
                        load(19, 18),
                        gep(20, 2, vec![value(16), int(0)], AddressSpace::Global),
                        load(21, 20),
                        bin(22, BinOp::Sub, value(19), value(21), Type::F32),
                        intrinsic_exp(23, value(22)),
                        gep(24, 10, vec![value(16), value(17)], AddressSpace::Private),
                        store(24, value(23)),
                        bin(25, BinOp::Add, value(17), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(2),
                        args: vec![value(16), value(25)],
                    },
                ),
                block(
                    4,
                    "tile_expr_loop_row_latch_4",
                    vec![block_param(26, "row")],
                    vec![bin(27, BinOp::Add, value(26), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![value(27)],
                    },
                ),
                block(
                    5,
                    "tile_expr_loop_continue_5",
                    vec![],
                    vec![alloca(40)],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![int(0)],
                    },
                ),
                block(
                    6,
                    "tile_expr_loop_row_header_6",
                    vec![block_param(41, "row")],
                    vec![icmp_slt(42, value(41), int(2))],
                    Terminator::CondBr {
                        cond: value(42),
                        true_target: BlockId(7),
                        true_args: vec![value(41), int(0)],
                        false_target: BlockId(10),
                        false_args: vec![],
                    },
                ),
                block(
                    7,
                    "tile_expr_loop_col_header_7",
                    vec![block_param(43, "row"), block_param(44, "col")],
                    vec![icmp_slt(45, value(44), int(8))],
                    Terminator::CondBr {
                        cond: value(45),
                        true_target: BlockId(8),
                        true_args: vec![value(43), value(44)],
                        false_target: BlockId(9),
                        false_args: vec![value(43)],
                    },
                ),
                block(
                    8,
                    "tile_expr_loop_body_8",
                    vec![block_param(46, "row"), block_param(47, "col")],
                    vec![
                        gep(48, 10, vec![value(46), value(47)], AddressSpace::Private),
                        load(49, 48),
                        bin(50, BinOp::Add, value(49), int(1), Type::F32),
                        gep(51, 40, vec![value(46), value(47)], AddressSpace::Private),
                        store(51, value(50)),
                        bin(52, BinOp::Add, value(47), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(7),
                        args: vec![value(46), value(52)],
                    },
                ),
                block(
                    9,
                    "tile_expr_loop_row_latch_9",
                    vec![block_param(53, "row")],
                    vec![bin(54, BinOp::Add, value(53), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(6),
                        args: vec![value(54)],
                    },
                ),
                block(
                    10,
                    "tile_expr_loop_continue_10",
                    vec![],
                    vec![],
                    Terminator::Br {
                        target: BlockId(11),
                        args: vec![int(0)],
                    },
                ),
                block(
                    11,
                    "store_loop_header_11",
                    vec![block_param(55, "col")],
                    vec![icmp_slt(56, value(55), int(8))],
                    Terminator::CondBr {
                        cond: value(56),
                        true_target: BlockId(12),
                        true_args: vec![value(55)],
                        false_target: BlockId(13),
                        false_args: vec![],
                    },
                ),
                block(
                    12,
                    "store_loop_body_12",
                    vec![block_param(57, "col")],
                    vec![
                        gep(58, 40, vec![int(1), value(57)], AddressSpace::Private),
                        load(59, 58),
                        gep(60, 3, vec![int(1), value(57)], AddressSpace::Global),
                        store(60, value(59)),
                        bin(61, BinOp::Add, value(57), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(11),
                        args: vec![value(61)],
                    },
                ),
                block(13, "done", vec![], vec![], Terminator::Ret { value: None }),
            ],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let canonical = run(func);
        assert!(
            !canonical
                .blocks
                .iter()
                .map(|block| block.name.as_str())
                .any(|name| name.starts_with("tile_expr_loop_"))
        );
        assert!(crate::passes::verify::verify_function(&canonical).is_ok());
        let store_body = canonical
            .blocks
            .iter()
            .find(|block| block.name == "store_loop_body_12")
            .unwrap();
        assert!(store_body.insts.iter().any(|inst| {
            matches!(
                inst.op,
                InstOp::Intrinsic {
                    intrinsic: Intrinsic::Exp,
                    ..
                }
            )
        }));
        assert!(!store_body.insts.iter().any(|inst| {
            matches!(
                inst.op,
                InstOp::Gep {
                    base: Operand::Value(ValueId(10) | ValueId(40)),
                    ..
                }
            )
        }));
    }

    #[test]
    fn compacts_rank2_tile_store_row_setup_block() {
        let func = Function {
            name: "rank2_store".into(),
            params: vec![param(1, "src"), param(2, "dst")],
            blocks: vec![
                block(
                    0,
                    "bb0",
                    vec![],
                    vec![],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![int(0)],
                    },
                ),
                block(
                    1,
                    "tile_store_row_header_1",
                    vec![block_param(10, "row")],
                    vec![icmp_slt(11, value(10), int(2))],
                    Terminator::CondBr {
                        cond: value(11),
                        true_target: BlockId(2),
                        true_args: vec![value(10)],
                        false_target: BlockId(6),
                        false_args: vec![],
                    },
                ),
                block(
                    2,
                    "tile_store_row_setup_2",
                    vec![block_param(12, "row")],
                    vec![
                        bin(13, BinOp::Add, int(8), value(12), Type::I64),
                        bin(14, BinOp::Mul, value(13), int(32), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(3),
                        args: vec![value(12), value(14), int(0)],
                    },
                ),
                block(
                    3,
                    "tile_store_col_header_3",
                    vec![
                        block_param(15, "row"),
                        block_param(16, "row_offset"),
                        block_param(17, "col"),
                    ],
                    vec![icmp_slt(18, value(17), int(8))],
                    Terminator::CondBr {
                        cond: value(18),
                        true_target: BlockId(4),
                        true_args: vec![value(15), value(16), value(17)],
                        false_target: BlockId(5),
                        false_args: vec![value(15)],
                    },
                ),
                block(
                    4,
                    "tile_store_body_4",
                    vec![
                        block_param(19, "row"),
                        block_param(20, "row_offset"),
                        block_param(21, "col"),
                    ],
                    vec![
                        gep(22, 1, vec![value(19), value(21)], AddressSpace::Private),
                        load(23, 22),
                        bin(24, BinOp::Add, int(16), value(21), Type::I64),
                        bin(25, BinOp::Add, value(20), value(24), Type::I64),
                        gep(26, 2, vec![value(25)], AddressSpace::Global),
                        store(26, value(23)),
                        bin(27, BinOp::Add, value(21), int(1), Type::I64),
                    ],
                    Terminator::Br {
                        target: BlockId(3),
                        args: vec![value(19), value(20), value(27)],
                    },
                ),
                block(
                    5,
                    "tile_store_row_latch_5",
                    vec![block_param(28, "row")],
                    vec![bin(29, BinOp::Add, value(28), int(1), Type::I64)],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![value(29)],
                    },
                ),
                block(
                    6,
                    "tile_store_continue_6",
                    vec![],
                    vec![],
                    Terminator::Ret { value: None },
                ),
            ],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let canonical = run(func);
        assert!(
            !canonical
                .blocks
                .iter()
                .any(|block| block.name.starts_with("tile_store_row_setup_"))
        );

        let col_header = canonical
            .blocks
            .iter()
            .find(|block| block.name == "tile_store_col_header_3")
            .unwrap();
        assert_eq!(col_header.params.len(), 2);

        let body = canonical
            .blocks
            .iter()
            .find(|block| block.name == "tile_store_body_4")
            .unwrap();
        assert_eq!(body.params.len(), 2);
        assert!(
            body.insts
                .iter()
                .any(|inst| { matches!(inst.op, InstOp::Bin { op: BinOp::Mul, .. }) })
        );
        assert!(crate::passes::verify::verify_function(&canonical).is_ok());
    }

    fn param(id: u32, name: &str) -> Param {
        Param {
            id: ValueId(id),
            name: name.into(),
            ty: Type::ptr(AddressSpace::Global, Type::F32),
            abi: None,
        }
    }

    fn block_param(id: u32, name: &str) -> BlockParam {
        BlockParam {
            id: ValueId(id),
            name: name.into(),
            ty: Type::I64,
        }
    }

    fn block(
        id: u32,
        name: &str,
        params: Vec<BlockParam>,
        insts: Vec<Inst>,
        terminator: Terminator,
    ) -> BasicBlock {
        BasicBlock {
            id: BlockId(id),
            name: name.into(),
            params,
            insts,
            terminator,
        }
    }

    fn value(id: u32) -> Operand {
        Operand::Value(ValueId(id))
    }

    fn int(value: i64) -> Operand {
        Operand::Const(Constant::Int(value))
    }

    fn bool_const(value: bool) -> Operand {
        Operand::Const(Constant::Bool(value))
    }

    fn alloca(id: u32) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty: Type::ptr(
                AddressSpace::Private,
                Type::array(2, Type::array(8, Type::F32)),
            ),
            op: InstOp::Alloca {
                alloc_ty: Type::array(2, Type::array(8, Type::F32)),
                addr_space: AddressSpace::Private,
            },
            metadata: vec![Metadata::Alignment(16)],
        }
    }

    fn gep(id: u32, base: u32, indices: Vec<Operand>, addr_space: AddressSpace) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty: Type::ptr(addr_space, Type::F32),
            op: InstOp::Gep {
                base: value(base),
                indices,
            },
            metadata: Vec::new(),
        }
    }

    fn load(id: u32, ptr: u32) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty: Type::F32,
            op: InstOp::Load { ptr: value(ptr) },
            metadata: Vec::new(),
        }
    }

    fn store(ptr: u32, stored: Operand) -> Inst {
        Inst {
            result: None,
            result_name: None,
            ty: Type::Void,
            op: InstOp::Store {
                ptr: value(ptr),
                value: stored,
            },
            metadata: Vec::new(),
        }
    }

    fn bin(id: u32, op: BinOp, lhs: Operand, rhs: Operand, ty: Type) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty,
            op: InstOp::Bin { op, lhs, rhs },
            metadata: Vec::new(),
        }
    }

    fn icmp_slt(id: u32, lhs: Operand, rhs: Operand) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty: Type::I1,
            op: InstOp::Cmp {
                pred: crate::CmpPred::Slt,
                lhs,
                rhs,
            },
            metadata: Vec::new(),
        }
    }

    fn intrinsic_exp(id: u32, arg: Operand) -> Inst {
        Inst {
            result: Some(ValueId(id)),
            result_name: Some(format!("v{id}")),
            ty: Type::F32,
            op: InstOp::Intrinsic {
                intrinsic: Intrinsic::Exp,
                args: vec![arg],
            },
            metadata: Vec::new(),
        }
    }
}
