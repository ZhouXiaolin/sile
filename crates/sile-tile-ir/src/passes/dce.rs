use std::collections::HashSet;

use crate::{TileIrFunction, TileIrOp};

/// Dead code elimination: remove instructions whose results are never used.
pub fn run(mut func: TileIrFunction) -> TileIrFunction {
    let live = find_live_values(&func);

    for block in &mut func.blocks {
        block.insts.retain(|inst| {
            // Always keep side-effecting instructions
            if is_side_effect(&inst.op) {
                return true;
            }
            live.contains(&inst.result)
        });
    }

    // Clean up unused types
    let all_values: HashSet<_> = func
        .blocks
        .iter()
        .flat_map(|b| {
            b.params
                .iter()
                .copied()
                .chain(b.insts.iter().map(|i| i.result))
        })
        .chain(func.params.iter().map(|p| p.value))
        .collect();
    func.types.retain(|k, _| all_values.contains(k));

    func
}

/// Find all values that are transitively used by side-effecting instructions or terminators.
fn find_live_values(func: &TileIrFunction) -> HashSet<crate::ValueId> {
    let mut live = HashSet::new();
    let mut worklist = Vec::new();

    // Seed: all values used by terminators and side-effecting instructions
    for block in &func.blocks {
        for inst in &block.insts {
            if is_side_effect(&inst.op) {
                for u in TileIrFunction::inst_uses(&inst.op) {
                    worklist.push(u);
                }
            }
        }
        for u in TileIrFunction::terminator_uses(&block.terminator) {
            worklist.push(u);
        }
    }

    // Propagate: walk use chains backward
    while let Some(value) = worklist.pop() {
        if !live.insert(value) {
            continue;
        }
        // Find the instruction that defines this value
        for block in &func.blocks {
            for inst in &block.insts {
                if inst.result == value {
                    for u in TileIrFunction::inst_uses(&inst.op) {
                        worklist.push(u);
                    }
                }
            }
        }
    }

    live
}

fn is_side_effect(op: &TileIrOp) -> bool {
    matches!(
        op,
        TileIrOp::StorePtrTko { .. } | TileIrOp::SileAtomicAdd { .. }
    )
}
