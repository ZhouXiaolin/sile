//! Control-flow facts the IR passes share.
//!
//! The IR keeps blocks in a list and names them by id, so a pass that needs
//! predecessors, dominators or loops builds them here once instead of walking
//! the block list in every pass. The functions are small on purpose: a kernel
//! has tens of blocks, not thousands.

use std::collections::{HashMap, HashSet};

use crate::{BlockId, Function, Terminator};

/// The blocks a terminator can transfer control to.
pub(crate) fn successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        Terminator::Switch { default, cases, .. } => {
            let mut targets = vec![*default];
            targets.extend(cases.iter().map(|(_, target)| *target));
            targets
        }
        Terminator::Ret { .. } => Vec::new(),
    }
}

/// Every block with an edge to a block, in block order.
pub(crate) fn predecessors(func: &Function) -> HashMap<BlockId, Vec<BlockId>> {
    let mut predecessors: HashMap<BlockId, Vec<BlockId>> = func
        .blocks
        .iter()
        .map(|block| (block.id, Vec::new()))
        .collect();
    for block in &func.blocks {
        for target in successors(&block.terminator) {
            if let Some(list) = predecessors.get_mut(&target) {
                list.push(block.id);
            }
        }
    }
    predecessors
}

/// The blocks in reverse postorder from the entry.
///
/// A pass that walks the CFG in this order sees a block after every
/// predecessor that is not a loop back edge, so a forward dataflow converges
/// in one pass over straight-line code and a few passes around loops.
pub(crate) fn reverse_postorder(func: &Function) -> Vec<BlockId> {
    fn visit(
        func: &Function,
        block: BlockId,
        seen: &mut HashSet<BlockId>,
        order: &mut Vec<BlockId>,
    ) {
        if !seen.insert(block) {
            return;
        }
        if let Some(data) = func.blocks.iter().find(|candidate| candidate.id == block) {
            for successor in successors(&data.terminator) {
                visit(func, successor, seen, order);
            }
        }
        order.push(block);
    }

    let mut seen = HashSet::new();
    let mut order = Vec::new();
    visit(func, func.entry, &mut seen, &mut order);
    order.reverse();
    order
}

/// Each reachable block's dominator set, entry included.
///
/// An unreachable block is its own only dominator, so no back edge is
/// attributed to a block that cannot reach the header. The iteration follows
/// reverse postorder, which is what keeps this analysis linear on the long
/// straight-line chains an unroller leaves behind: walking the block list in
/// source order would need one pass per block of the chain.
pub(crate) fn dominators(func: &Function) -> HashMap<BlockId, HashSet<BlockId>> {
    let order = reverse_postorder(func);
    let reachable: HashSet<BlockId> = order.iter().copied().collect();
    let all: HashSet<BlockId> = reachable.clone();
    let mut dominators: HashMap<BlockId, HashSet<BlockId>> = func
        .blocks
        .iter()
        .map(|block| {
            if block.id == func.entry {
                (block.id, HashSet::from([block.id]))
            } else if !reachable.contains(&block.id) {
                (block.id, HashSet::from([block.id]))
            } else {
                (block.id, all.clone())
            }
        })
        .collect();
    let predecessors = predecessors(func);

    loop {
        let mut changed = false;
        for block_id in &order {
            if *block_id == func.entry {
                continue;
            }
            let preds = &predecessors[block_id];
            let mut new_set: Option<HashSet<BlockId>> = None;
            for pred in preds {
                let pred_set = &dominators[pred];
                new_set = Some(match new_set {
                    None => pred_set.clone(),
                    Some(current) => current.intersection(pred_set).copied().collect(),
                });
            }
            let mut new_set = new_set.unwrap_or_default();
            new_set.insert(*block_id);
            if new_set != dominators[block_id] {
                dominators.insert(*block_id, new_set);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dominators
}

/// One natural loop: a header and every block that can reach its latch.
#[derive(Clone, Debug)]
pub(crate) struct NaturalLoop {
    pub header: BlockId,
    pub blocks: HashSet<BlockId>,
}

/// Every natural loop of the function, outermost first.
///
/// A loop is found by a back edge: a predecessor the header dominates. The loop
/// body is the header plus every block that reaches the latch without crossing
/// the header. A loop nested inside another appears with a smaller body, so
/// sorting by size puts the outer loop first, which is the order a full
/// unroller wants.
pub(crate) fn natural_loops(func: &Function) -> Vec<NaturalLoop> {
    let dominators = dominators(func);
    let predecessors = predecessors(func);
    let mut loops = Vec::new();

    for block in &func.blocks {
        for pred in &predecessors[&block.id] {
            // A back edge is an edge whose target dominates its source.
            let dominates_pred = dominators
                .get(pred)
                .is_some_and(|pred_dominators| pred_dominators.contains(&block.id));
            if !dominates_pred || *pred == block.id {
                continue;
            }
            let mut blocks: HashSet<BlockId> = HashSet::from([block.id]);
            let mut worklist = vec![*pred];
            while let Some(current) = worklist.pop() {
                if !blocks.insert(current) {
                    continue;
                }
                for predecessor in &predecessors[&current] {
                    worklist.push(*predecessor);
                }
            }
            loops.push(NaturalLoop {
                header: block.id,
                blocks,
            });
        }
    }

    loops.sort_by(|left, right| right.blocks.len().cmp(&left.blocks.len()));
    loops
}

/// The one block outside the loop that jumps into the header, when there is one.
pub(crate) fn preheader(
    func: &Function,
    header: BlockId,
    loop_blocks: &HashSet<BlockId>,
) -> Option<BlockId> {
    let predecessors = predecessors(func);
    let outside: Vec<BlockId> = predecessors[&header]
        .iter()
        .copied()
        .filter(|pred| !loop_blocks.contains(pred))
        .collect();
    if outside.len() != 1 {
        return None;
    }
    let candidate = outside[0];
    let block = func.blocks.iter().find(|block| block.id == candidate)?;
    match &block.terminator {
        Terminator::Br { target, .. } if *target == header => Some(candidate),
        _ => None,
    }
}
