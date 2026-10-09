use std::collections::HashMap;

use crate::{BlockId, Function, Operand, Terminator, ValueId};

/// Inline constant arguments through preheader blocks.
///
/// A preheader block with a single predecessor may have parameters that receive
/// constant values from that predecessor. This pass replaces references to those
/// parameters in the preheader's terminator with the actual constant values,
/// then removes the now-unused parameters from both the preheader and the
/// predecessor's branch arguments.
///
/// This eliminates unnecessary variable assignments in generated C code like:
///   loop_col47_pre = 0;
///   for (loop_col47 = loop_col47_pre; ...)
/// and produces instead:
///   for (loop_col47 = 0; ...)
pub fn run(mut func: Function) -> Function {
    loop {
        let predecessors = build_predecessor_map(&func);
        let mut changed = false;

        for block_idx in 0..func.blocks.len() {
            let block_id = func.blocks[block_idx].id;

            // Only process blocks with exactly one predecessor.
            let preds = predecessors.get(&block_id).cloned().unwrap_or_default();
            if preds.len() != 1 {
                continue;
            }
            let pred_id = preds[0];

            // Collect block properties by clone (avoids borrow issues).
            let block_insts_empty = func.blocks[block_idx].insts.is_empty();
            let block_params: Vec<_> = func.blocks[block_idx].params.clone();
            let forward_args: Option<Vec<Operand>> = match &func.blocks[block_idx].terminator {
                Terminator::Br { args, .. } => Some(args.clone()),
                _ => None,
            };

            let pred_idx = func.blocks.iter().position(|b| b.id == pred_id);
            let incoming_args =
                pred_idx.and_then(|pidx| get_incoming_args(&func.blocks[pidx], block_id));

            if !block_insts_empty || forward_args.is_none() || incoming_args.is_none() {
                continue;
            }
            let forward_args = forward_args.unwrap();
            let incoming_args = incoming_args.unwrap();

            // Parameter count must match incoming arg count.
            if block_params.len() != incoming_args.len() {
                continue;
            }

            // Find parameters that receive constant values.
            let mut params_to_inline: Vec<(usize, Operand)> = Vec::new();
            for (i, arg) in incoming_args.iter().enumerate() {
                if matches!(arg, Operand::Const(_)) {
                    params_to_inline.push((i, arg.clone()));
                }
            }

            if params_to_inline.is_empty() {
                continue;
            }

            let pred_idx = pred_idx.unwrap();

            // 1. Replace parameter references in forward_args with constants.
            let mut new_forward_args = forward_args;
            for (param_idx, const_arg) in &params_to_inline {
                let param_id = block_params[*param_idx].id;
                replace_operand_value(&mut new_forward_args, param_id, const_arg.clone());
            }

            // 2. Modify block: remove inlined parameters and update terminator.
            {
                let block = &mut func.blocks[block_idx];
                let mut indices_to_remove: Vec<usize> =
                    params_to_inline.iter().map(|(i, _)| *i).collect();
                indices_to_remove.sort_by(|a, b| b.cmp(a));
                for idx in indices_to_remove {
                    block.params.remove(idx);
                }
                if let Terminator::Br { ref mut args, .. } = block.terminator {
                    *args = new_forward_args;
                }
            }

            // 3. Modify predecessor: remove corresponding args.
            {
                let pred = &mut func.blocks[pred_idx];
                remove_pred_args_at_indices(
                    pred,
                    block_id,
                    &params_to_inline.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                );
            }

            changed = true;
        }

        if !changed {
            break;
        }
    }

    func
}

fn build_predecessor_map(func: &Function) -> HashMap<BlockId, Vec<BlockId>> {
    let mut map: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for block in &func.blocks {
        match &block.terminator {
            Terminator::Br { target, .. } => {
                map.entry(*target).or_default().push(block.id);
            }
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => {
                map.entry(*true_target).or_default().push(block.id);
                map.entry(*false_target).or_default().push(block.id);
            }
            Terminator::Switch { default, cases, .. } => {
                map.entry(*default).or_default().push(block.id);
                for (_, target) in cases {
                    map.entry(*target).or_default().push(block.id);
                }
            }
            Terminator::Ret { .. } => {}
        }
    }
    map
}

fn get_incoming_args(pred: &crate::BasicBlock, target: BlockId) -> Option<Vec<Operand>> {
    match &pred.terminator {
        Terminator::Br { target: t, args } if *t == target => Some(args.clone()),
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == target {
                Some(true_args.clone())
            } else if *false_target == target {
                Some(false_args.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn replace_operand_value(operands: &mut [Operand], old_id: ValueId, new_operand: Operand) {
    for op in operands.iter_mut() {
        if let Operand::Value(id) = op {
            if *id == old_id {
                *op = new_operand.clone();
            }
        }
    }
}

fn remove_pred_args_at_indices(pred: &mut crate::BasicBlock, target: BlockId, indices: &[usize]) {
    let mut sorted_indices: Vec<usize> = indices.to_vec();
    sorted_indices.sort_by(|a, b| b.cmp(a)); // descending

    match &mut pred.terminator {
        Terminator::Br { target: t, args } if *t == target => {
            for idx in sorted_indices {
                if idx < args.len() {
                    args.remove(idx);
                }
            }
        }
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == target {
                for idx in &sorted_indices {
                    if *idx < true_args.len() {
                        true_args.remove(*idx);
                    }
                }
            } else if *false_target == target {
                for idx in &sorted_indices {
                    if *idx < false_args.len() {
                        false_args.remove(*idx);
                    }
                }
            }
        }
        _ => {}
    }
}
