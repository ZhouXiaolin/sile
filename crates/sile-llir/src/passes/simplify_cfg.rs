use std::collections::{HashMap, HashSet, VecDeque};

use crate::{BlockId, Function, Terminator};

/// Simplify CFG by removing unreachable blocks and fixing dangling branch targets.
///
/// This is intentionally conservative: it does not merge blocks or rewrite SSA
/// shape aggressively, so it is safe to run in any pipeline stage.
pub fn run(mut func: Function) -> Function {
    let reachable = collect_reachable_blocks(&func);
    if reachable.len() == func.blocks.len() {
        return func;
    }

    func.blocks.retain(|block| reachable.contains(&block.id));

    // Entry is guaranteed reachable by construction, but keep this guard to
    // avoid malformed IR if the caller constructed a broken function.
    if !func.blocks.iter().any(|block| block.id == func.entry) {
        if let Some(first) = func.blocks.first() {
            func.entry = first.id;
        }
    }

    let arities: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .map(|block| (block.id, block.params.len()))
        .collect();
    for block in &mut func.blocks {
        sanitize_terminator(&mut block.terminator, &reachable, &arities);
    }

    func
}

fn collect_reachable_blocks(func: &Function) -> HashSet<BlockId> {
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(func.entry);

    while let Some(block_id) = queue.pop_front() {
        if !reachable.insert(block_id) {
            continue;
        }
        let Some(block) = func.blocks.iter().find(|block| block.id == block_id) else {
            continue;
        };
        match &block.terminator {
            Terminator::Br { target, .. } => queue.push_back(*target),
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => {
                queue.push_back(*true_target);
                queue.push_back(*false_target);
            }
            Terminator::Switch { default, cases, .. } => {
                queue.push_back(*default);
                for (_, target) in cases {
                    queue.push_back(*target);
                }
            }
            Terminator::Ret { .. } => {}
        }
    }

    reachable
}

fn sanitize_terminator(
    terminator: &mut Terminator,
    reachable: &HashSet<BlockId>,
    arities: &HashMap<BlockId, usize>,
) {
    match terminator {
        Terminator::Br { target, args } => {
            if !reachable.contains(target) {
                *terminator = Terminator::Ret { value: None };
                return;
            }
            if let Some(expected) = arities.get(target).copied() {
                args.truncate(expected);
            }
        }
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            let true_ok = reachable.contains(true_target);
            let false_ok = reachable.contains(false_target);
            if !true_ok && !false_ok {
                *terminator = Terminator::Ret { value: None };
                return;
            }
            if !true_ok {
                *true_target = *false_target;
                *true_args = false_args.clone();
            } else if !false_ok {
                *false_target = *true_target;
                *false_args = true_args.clone();
            }
            if let Some(expected) = arities.get(true_target).copied() {
                true_args.truncate(expected);
            }
            if let Some(expected) = arities.get(false_target).copied() {
                false_args.truncate(expected);
            }
        }
        Terminator::Switch { default, cases, .. } => {
            cases.retain(|(_, target)| reachable.contains(target));
            if !reachable.contains(default) {
                if let Some((_, replacement)) = cases.first() {
                    *default = *replacement;
                } else {
                    *terminator = Terminator::Ret { value: None };
                }
            }
        }
        Terminator::Ret { .. } => {}
    }
}
