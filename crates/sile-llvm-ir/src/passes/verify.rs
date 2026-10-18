use std::collections::{HashMap, HashSet};

use crate::{BlockId, Function, InstOp, Operand, Terminator, ValueId};

pub fn verify_function(func: &Function) -> Result<(), String> {
    let mut block_ids = HashSet::new();
    let mut block_params = HashMap::new();
    let mut value_defs = HashSet::new();
    let mut value_def_sites = HashMap::new();

    for param in &func.params {
        if !value_defs.insert(param.id) {
            return Err(format!("duplicate function param value id {}", param.id.0));
        }
        value_def_sites.insert(param.id, DefSite::FunctionParam);
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
            value_def_sites.insert(param.id, DefSite::BlockParam { block: block.id });
        }
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result {
                if !value_defs.insert(result) {
                    return Err(format!(
                        "duplicate instruction result value id {}",
                        result.0
                    ));
                }
                value_def_sites.insert(
                    result,
                    DefSite::Inst {
                        block: block.id,
                        inst_idx,
                    },
                );
            }
        }
    }

    if !block_ids.contains(&func.entry) {
        return Err(format!("entry block {} does not exist", func.entry.0));
    }

    let dominators = compute_dominators(func);
    for block in &func.blocks {
        verify_block_uses(block, &value_defs, &value_def_sites, &dominators)?;
        verify_terminator(
            &block.terminator,
            block.id,
            &block_params,
            &value_defs,
            &value_def_sites,
            &dominators,
        )?;
    }

    Ok(())
}

/// Blocks that dominate every block in the map, keyed by block.
///
/// SSA requires a definition to dominate every use. Checking that inside one
/// block is not enough: a pass that hoists an instruction out of a branch, or
/// that lets one arm define a value the join reads, produces IR that is
/// well-formed by name but undefined on at least one path. That is exactly the
/// shape of a silent miscompile, so the verifier computes dominators and
/// rejects it.
fn compute_dominators(func: &Function) -> HashMap<BlockId, HashSet<BlockId>> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefSite {
    FunctionParam,
    BlockParam { block: BlockId },
    Inst { block: BlockId, inst_idx: usize },
}

fn verify_block_uses(
    block: &crate::BasicBlock,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> Result<(), String> {
    for (inst_idx, inst) in block.insts.iter().enumerate() {
        for operand in inst_operands(&inst.op) {
            verify_operand(
                operand,
                block.id,
                Some(inst_idx),
                value_defs,
                value_def_sites,
                dominators,
            )?;
        }
    }
    Ok(())
}

fn verify_terminator(
    terminator: &Terminator,
    block_id: BlockId,
    block_params: &HashMap<BlockId, usize>,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> Result<(), String> {
    match terminator {
        Terminator::Br { target, args } => {
            verify_branch_target(*target, args.len(), block_params)?;
            for operand in args {
                verify_operand(operand, block_id, None, value_defs, value_def_sites, dominators)?;
            }
            Ok(())
        }
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            verify_branch_target(*true_target, true_args.len(), block_params)?;
            verify_branch_target(*false_target, false_args.len(), block_params)?;
            verify_operand(cond, block_id, None, value_defs, value_def_sites, dominators)?;
            for operand in true_args.iter().chain(false_args.iter()) {
                verify_operand(operand, block_id, None, value_defs, value_def_sites, dominators)?;
            }
            Ok(())
        }
        Terminator::Switch {
            value,
            default,
            cases,
        } => {
            verify_branch_target(*default, 0, block_params)?;
            for (_, target) in cases {
                verify_branch_target(*target, 0, block_params)?;
            }
            verify_operand(value, block_id, None, value_defs, value_def_sites, dominators)?;
            Ok(())
        }
        Terminator::Ret { value } => {
            if let Some(operand) = value {
                verify_operand(operand, block_id, None, value_defs, value_def_sites, dominators)?;
            }
            Ok(())
        }
    }
}

fn verify_operand(
    operand: &Operand,
    use_block: BlockId,
    use_inst_idx: Option<usize>,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> Result<(), String> {
    let Operand::Value(value) = operand else {
        return Ok(());
    };

    if !value_defs.contains(value) {
        return Err(format!("use of undefined value id {}", value.0));
    }

    let Some(def_site) = value_def_sites.get(value).copied() else {
        return Err(format!("missing def-site for value id {}", value.0));
    };

    match def_site {
        // A function parameter is in scope everywhere.
        DefSite::FunctionParam => Ok(()),
        DefSite::BlockParam { block: def_block } => {
            if def_block == use_block || dominates(dominators, def_block, use_block) {
                Ok(())
            } else {
                Err(format!(
                    "value id {} is defined in block {}, which does not dominate its use in block {}",
                    value.0, def_block.0, use_block.0
                ))
            }
        }
        DefSite::Inst {
            block: def_block,
            inst_idx: def_inst_idx,
        } => {
            if def_block != use_block && !dominates(dominators, def_block, use_block) {
                return Err(format!(
                    "value id {} is defined in block {}, which does not dominate its use in block {}",
                    value.0, def_block.0, use_block.0
                ));
            }
            if def_block == use_block
                && let Some(use_inst_idx) = use_inst_idx
                && def_inst_idx >= use_inst_idx
            {
                return Err(format!(
                    "value id {} is used before its definition in block {}",
                    value.0, use_block.0
                ));
            }
            Ok(())
        }
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

fn inst_operands(op: &InstOp) -> Vec<&Operand> {
    match op {
        InstOp::Alloca { .. } => Vec::new(),
        InstOp::Gep { base, indices } => {
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(base);
            operands.extend(indices.iter());
            operands
        }
        InstOp::Load { ptr } => vec![ptr],
        InstOp::Store { ptr, value } => vec![ptr, value],
        InstOp::AtomicAdd { ptr, value } => vec![ptr, value],
        InstOp::Memcpy { dst, src, size } => vec![dst, src, size],
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
        InstOp::Cast { value, .. } => vec![value],
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond, on_true, on_false],
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => args.iter().collect(),
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
