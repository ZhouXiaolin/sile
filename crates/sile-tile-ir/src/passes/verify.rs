use std::collections::{HashMap, HashSet};

use crate::{BlockId, TileIrFunction, TileIrOp, TileIrTerminator, ValueId};

pub fn verify_function(func: &TileIrFunction) -> Result<(), String> {
    let mut block_ids = HashSet::new();
    let mut block_param_counts = HashMap::new();
    let mut value_defs = HashSet::new();

    for param in &func.params {
        if !value_defs.insert(param.value) {
            return Err(format!("duplicate function param value {}", param.value.0));
        }
        let Some(param_ty) = func.types.get(&param.value) else {
            return Err(format!("missing type for function param {}", param.value.0));
        };
        if param_ty != &param.ty {
            return Err(format!(
                "param {} type mismatch: param says {:?}, map says {:?}",
                param.value.0, param.ty, param_ty
            ));
        }
    }

    for block in &func.blocks {
        if !block_ids.insert(block.id) {
            return Err(format!("duplicate block id {}", block.id.0));
        }
        block_param_counts.insert(block.id, block.params.len());

        for value in &block.params {
            if !value_defs.insert(*value) {
                return Err(format!(
                    "duplicate block param value {} in block {}",
                    value.0, block.id.0
                ));
            }
            if !func.types.contains_key(value) {
                return Err(format!("missing type for block param {}", value.0));
            }
        }

        for inst in &block.insts {
            if !value_defs.insert(inst.result) {
                return Err(format!("duplicate instruction result {}", inst.result.0));
            }
            if !func.types.contains_key(&inst.result) {
                return Err(format!(
                    "missing type for instruction result {}",
                    inst.result.0
                ));
            }
        }
    }

    if !block_ids.contains(&func.entry) {
        return Err(format!("entry block {} does not exist", func.entry.0));
    }

    let mut def_blocks = HashMap::new();
    for block in &func.blocks {
        for param in &block.params {
            def_blocks.insert(*param, block.id);
        }
        for inst in &block.insts {
            def_blocks.insert(inst.result, block.id);
        }
    }
    let dominators = compute_dominators(func);

    for block in &func.blocks {
        for inst in &block.insts {
            for used in TileIrFunction::inst_uses(&inst.op) {
                if !value_defs.contains(&used) {
                    return Err(format!(
                        "undefined value {} used in block {}",
                        used.0, block.id.0
                    ));
                }
                // A function param has no defining block and is in scope
                // everywhere. Any other value must be defined in a block that
                // dominates this use, or the value is undefined on some path.
                if let Some(def_block) = def_blocks.get(&used)
                    && *def_block != block.id
                    && !dominates(&dominators, *def_block, block.id)
                {
                    return Err(format!(
                        "value {} is defined in block {}, which does not dominate its use in block {}",
                        used.0, def_block.0, block.id.0
                    ));
                }
            }
        }

        verify_terminator(
            &block.terminator,
            block.id,
            &block_ids,
            &block_param_counts,
            &value_defs,
            &def_blocks,
            &dominators,
        )?;
    }

    Ok(())
}

fn compute_dominators(func: &TileIrFunction) -> HashMap<BlockId, HashSet<BlockId>> {
    let block_ids: Vec<BlockId> = func.blocks.iter().map(|block| block.id).collect();
    let all: HashSet<BlockId> = block_ids.iter().copied().collect();
    let mut predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for block in &func.blocks {
        for succ in successors(&block.terminator) {
            predecessors.entry(succ).or_default().push(block.id);
        }
    }

    let mut doms: HashMap<BlockId, HashSet<BlockId>> = block_ids
        .iter()
        .map(|block| {
            if *block == func.entry {
                (*block, HashSet::from([*block]))
            } else {
                (*block, all.clone())
            }
        })
        .collect();

    loop {
        let mut changed = false;
        for block_id in &block_ids {
            if *block_id == func.entry {
                continue;
            }
            let preds = predecessors.get(block_id).cloned().unwrap_or_default();
            let mut new_set: HashSet<BlockId> = match preds.first() {
                Some(first) => doms.get(first).cloned().unwrap_or_default(),
                None => HashSet::new(),
            };
            for pred in preds.iter().skip(1) {
                let pred_doms = doms.get(pred).cloned().unwrap_or_default();
                new_set.retain(|block| pred_doms.contains(block));
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
    doms
}

fn successors(terminator: &TileIrTerminator) -> Vec<BlockId> {
    match terminator {
        TileIrTerminator::Jump { target, .. } => vec![*target],
        TileIrTerminator::Branch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        TileIrTerminator::Return => Vec::new(),
    }
}

fn dominates(
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
    definition: BlockId,
    use_site: BlockId,
) -> bool {
    dominators
        .get(&use_site)
        .is_some_and(|doms| doms.contains(&definition))
}

fn verify_terminator(
    term: &TileIrTerminator,
    block_id: BlockId,
    block_ids: &HashSet<BlockId>,
    block_param_counts: &HashMap<BlockId, usize>,
    value_defs: &HashSet<ValueId>,
    def_blocks: &HashMap<ValueId, BlockId>,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> Result<(), String> {
    let check = |value: ValueId| -> Result<(), String> {
        verify_defined(value, block_id, value_defs)?;
        if let Some(def_block) = def_blocks.get(&value)
            && *def_block != block_id
            && !dominates(dominators, *def_block, block_id)
        {
            return Err(format!(
                "value {} is defined in block {}, which does not dominate its use in block {}",
                value.0, def_block.0, block_id.0
            ));
        }
        Ok(())
    };
    match term {
        TileIrTerminator::Jump { target, args } => {
            verify_target_arity(*target, args.len(), block_ids, block_param_counts)?;
            for value in args {
                check(*value)?;
            }
        }
        TileIrTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            check(*cond)?;
            verify_target_arity(*true_target, true_args.len(), block_ids, block_param_counts)?;
            verify_target_arity(
                *false_target,
                false_args.len(),
                block_ids,
                block_param_counts,
            )?;
            for value in true_args.iter().chain(false_args.iter()) {
                check(*value)?;
            }
        }
        TileIrTerminator::Return => {}
    }
    Ok(())
}

fn verify_target_arity(
    target: BlockId,
    arg_count: usize,
    block_ids: &HashSet<BlockId>,
    block_param_counts: &HashMap<BlockId, usize>,
) -> Result<(), String> {
    if !block_ids.contains(&target) {
        return Err(format!("branch target {} does not exist", target.0));
    }
    let expected = block_param_counts
        .get(&target)
        .copied()
        .ok_or_else(|| format!("missing block metadata for {}", target.0))?;
    if expected != arg_count {
        return Err(format!(
            "branch target {} expects {} args, got {}",
            target.0, expected, arg_count
        ));
    }
    Ok(())
}

fn verify_defined(
    value: ValueId,
    block_id: BlockId,
    value_defs: &HashSet<ValueId>,
) -> Result<(), String> {
    if value_defs.contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "undefined value {} referenced by terminator in block {}",
            value.0, block_id.0
        ))
    }
}

#[allow(dead_code)]
fn _is_side_effecting(op: &TileIrOp) -> bool {
    matches!(
        op,
        TileIrOp::StorePtrTko { .. } | TileIrOp::SileAtomicAdd { .. }
    )
}
