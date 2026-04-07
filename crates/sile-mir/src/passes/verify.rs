use std::collections::{HashMap, HashSet};

use crate::{BlockId, MirFunction, MirOp, MirTerminator, ValueId};

pub fn verify_function(func: &MirFunction) -> Result<(), String> {
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

    for block in &func.blocks {
        for inst in &block.insts {
            for used in MirFunction::inst_uses(&inst.op) {
                if !value_defs.contains(&used) {
                    return Err(format!(
                        "undefined value {} used in block {}",
                        used.0, block.id.0
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
        )?;
    }

    Ok(())
}

fn verify_terminator(
    term: &MirTerminator,
    block_id: BlockId,
    block_ids: &HashSet<BlockId>,
    block_param_counts: &HashMap<BlockId, usize>,
    value_defs: &HashSet<ValueId>,
) -> Result<(), String> {
    match term {
        MirTerminator::Jump { target, args } => {
            verify_target_arity(*target, args.len(), block_ids, block_param_counts)?;
            for value in args {
                verify_defined(*value, block_id, value_defs)?;
            }
        }
        MirTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            verify_defined(*cond, block_id, value_defs)?;
            verify_target_arity(*true_target, true_args.len(), block_ids, block_param_counts)?;
            verify_target_arity(
                *false_target,
                false_args.len(),
                block_ids,
                block_param_counts,
            )?;
            for value in true_args.iter().chain(false_args.iter()) {
                verify_defined(*value, block_id, value_defs)?;
            }
        }
        MirTerminator::Return => {}
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
fn _is_side_effecting(op: &MirOp) -> bool {
    matches!(op, MirOp::TileStore { .. } | MirOp::AtomicAdd { .. })
}
