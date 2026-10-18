use std::collections::{HashMap, HashSet};

use crate::{
    BasicBlock, BlockId, Constant, Function, Inst, InstOp, Intrinsic, Operand, Terminator, ValueId,
};

/// Canonical LLVM IR rewrites live here.
///
/// The active implementation starts with a narrow, profitable rewrite: sink
/// materialized pointwise tile-expression loops into their scalar consumers so
/// Tile IR lowering does not need to encode that choice permanently.
pub fn run(mut func: Function) -> Function {
    sink_pointwise_tile_exprs(&mut func);
    compact_rank2_tile_store_loops(&mut func);
    fold_algebraic_identities(&mut func);
    fold_constant_comparisons(&mut func);
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

fn fold_algebraic_identities(func: &mut Function) {
    use crate::BinOp::*;
    use crate::Constant::*;

    loop {
        let mut replacements: HashMap<ValueId, Operand> = HashMap::new();

        for block in &func.blocks {
            for inst in &block.insts {
                let Some(result) = inst.result else { continue };
                // Skip if this result was already replaced by an earlier rule
                if replacements.contains_key(&result) {
                    continue;
                }
                let InstOp::Bin { op, lhs, rhs } = &inst.op else {
                    continue;
                };
                // Rewrite operands through prior replacements
                let lhs = rewrite_operand(lhs.clone(), &replacements);
                let rhs = rewrite_operand(rhs.clone(), &replacements);

                let folded = match (op, &lhs, &rhs) {
                    // Integer: 0 * X → 0, X * 0 → 0
                    (Mul, Operand::Const(Int(0)), _) => Some(Operand::Const(Int(0))),
                    (Mul, _, Operand::Const(Int(0))) => Some(Operand::Const(Int(0))),
                    // Integer: X * 1 → X, 1 * X → X
                    (Mul, _, Operand::Const(Int(1))) => Some(lhs.clone()),
                    (Mul, Operand::Const(Int(1)), _) => Some(rhs.clone()),
                    // Integer: 0 + X → X, X + 0 → X
                    (Add, Operand::Const(Int(0)), _) => Some(rhs.clone()),
                    (Add, _, Operand::Const(Int(0))) => Some(lhs.clone()),
                    // Integer: X - 0 → X
                    (Sub, _, Operand::Const(Int(0))) => Some(lhs.clone()),
                    // Integer: 0 / X → 0
                    (Div, Operand::Const(Int(0)), _) => Some(Operand::Const(Int(0))),
                    // Float identities that hold for every IEEE-754 value.
                    //
                    // `0.0 * X` is not one: with X = Inf it is NaN and with
                    // X = NaN it is NaN, so folding it to 0.0 changes the
                    // answer. `0.0 + X` is not one either: with X = -0.0 the
                    // sum is +0.0, and the sign of a zero is observable through
                    // a later division. The compiler has no fast-math flag, so
                    // only the exact rewrites are here.
                    (Mul, _, Operand::Const(Float(f))) if *f == 1.0 => Some(lhs.clone()),
                    (Mul, Operand::Const(Float(f)), _) if *f == 1.0 => Some(rhs.clone()),
                    (Sub, _, Operand::Const(Float(f))) if *f == 0.0 => Some(lhs.clone()),
                    _ => None,
                };

                if let Some(simplified) = folded {
                    replacements.insert(result, simplified);
                }
            }
        }

        if replacements.is_empty() {
            break;
        }

        // Rewrite all uses of folded values and remove the folded Bin instructions
        for block in &mut func.blocks {
            block.insts = std::mem::take(&mut block.insts)
                .into_iter()
                .map(|inst| rewrite_inst_operands(inst, &replacements))
                .filter(|inst| inst.result.map_or(true, |r| !replacements.contains_key(&r)))
                .collect();
            block.terminator = rewrite_terminator_operands(block.terminator.clone(), &replacements);
        }
    }
}

/// Fold `Cmp` instructions with two constant operands into `Constant::Bool`.
fn fold_constant_comparisons(func: &mut Function) {
    let mut replacements: HashMap<ValueId, Operand> = HashMap::new();

    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else { continue };
            let InstOp::Cmp { pred, lhs, rhs } = &inst.op else {
                continue;
            };
            let lhs = rewrite_operand(lhs.clone(), &replacements);
            let rhs = rewrite_operand(rhs.clone(), &replacements);
            let folded = match (pred, &lhs, &rhs) {
                (
                    crate::CmpPred::Slt,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a < b),
                (
                    crate::CmpPred::Sle,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a <= b),
                (
                    crate::CmpPred::Sgt,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a > b),
                (
                    crate::CmpPred::Sge,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a >= b),
                (
                    crate::CmpPred::Eq,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a == b),
                (
                    crate::CmpPred::Ne,
                    Operand::Const(Constant::Int(a)),
                    Operand::Const(Constant::Int(b)),
                ) => Some(a != b),
                (
                    crate::CmpPred::Olt,
                    Operand::Const(Constant::Float(a)),
                    Operand::Const(Constant::Float(b)),
                ) => Some(a < b),
                (
                    crate::CmpPred::Ole,
                    Operand::Const(Constant::Float(a)),
                    Operand::Const(Constant::Float(b)),
                ) => Some(a <= b),
                (
                    crate::CmpPred::Ogt,
                    Operand::Const(Constant::Float(a)),
                    Operand::Const(Constant::Float(b)),
                ) => Some(a > b),
                (
                    crate::CmpPred::Oge,
                    Operand::Const(Constant::Float(a)),
                    Operand::Const(Constant::Float(b)),
                ) => Some(a >= b),
                _ => None,
            };
            if let Some(val) = folded {
                replacements.insert(result, Operand::Const(Constant::Bool(val)));
            }
        }
    }

    if replacements.is_empty() {
        return;
    }

    for block in &mut func.blocks {
        block.insts = std::mem::take(&mut block.insts)
            .into_iter()
            .map(|inst| rewrite_inst_operands(inst, &replacements))
            .filter(|inst| inst.result.map_or(true, |r| !replacements.contains_key(&r)))
            .collect();
        block.terminator = rewrite_terminator_operands(block.terminator.clone(), &replacements);
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
            intrinsic: Intrinsic::Barrier { .. } | Intrinsic::VecStore { .. },
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
