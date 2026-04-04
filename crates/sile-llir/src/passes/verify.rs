use std::collections::{HashMap, HashSet};

use crate::{BlockId, Function, Terminator};

pub fn verify_function(func: &Function) -> Result<(), String> {
    let mut block_ids = HashSet::new();
    let mut block_params = HashMap::new();
    let mut value_defs = HashSet::new();

    for param in &func.params {
        if !value_defs.insert(param.id) {
            return Err(format!("duplicate function param value id {}", param.id.0));
        }
    }

    for block in &func.blocks {
        if !block_ids.insert(block.id) {
            return Err(format!("duplicate block id {}", block.id.0));
        }
        block_params.insert(block.id, block.params.len());

        for param in &block.params {
            if !value_defs.insert(param.id) {
                return Err(format!("duplicate block param value id {}", param.id.0));
            }
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                if !value_defs.insert(result) {
                    return Err(format!(
                        "duplicate instruction result value id {}",
                        result.0
                    ));
                }
            }
        }
    }

    if !block_ids.contains(&func.entry) {
        return Err(format!("entry block {} does not exist", func.entry.0));
    }

    for block in &func.blocks {
        verify_terminator(&block.terminator, &block_params)?;
    }

    Ok(())
}

fn verify_terminator(
    terminator: &Terminator,
    block_params: &HashMap<BlockId, usize>,
) -> Result<(), String> {
    match terminator {
        Terminator::Br { target, args } => verify_branch_target(*target, args.len(), block_params),
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            verify_branch_target(*true_target, true_args.len(), block_params)?;
            verify_branch_target(*false_target, false_args.len(), block_params)
        }
        Terminator::Switch { default, cases, .. } => {
            verify_branch_target(*default, 0, block_params)?;
            for (_, target) in cases {
                verify_branch_target(*target, 0, block_params)?;
            }
            Ok(())
        }
        Terminator::Ret { .. } => Ok(()),
    }
}

fn verify_branch_target(
    target: BlockId,
    arg_count: usize,
    block_params: &HashMap<BlockId, usize>,
) -> Result<(), String> {
    let expected = block_params
        .get(&target)
        .ok_or_else(|| format!("branch target {} does not exist", target.0))?;
    if *expected != arg_count {
        return Err(format!(
            "branch target {} expects {} args but got {}",
            target.0, expected, arg_count
        ));
    }
    Ok(())
}
