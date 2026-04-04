use std::collections::{HashMap, HashSet};

use crate::{MirBlock, MirFunction, MirOp, MirType, ValueId};

#[derive(Clone, Debug, Default)]
pub struct LlirLoweringPlan {
    deferred_tiles: HashSet<ValueId>,
}

impl LlirLoweringPlan {
    pub fn should_defer(&self, value: ValueId) -> bool {
        self.deferred_tiles.contains(&value)
    }

    pub fn can_eval_tile_value(
        &self,
        value: ValueId,
        pending_tiles: &HashMap<ValueId, MirOp>,
        mir: &MirFunction,
    ) -> bool {
        let Some(op) = pending_tiles.get(&value) else {
            return matches!(mir.types.get(&value), Some(MirType::Tile { .. }));
        };
        match op {
            MirOp::TileConstant { .. } | MirOp::TileLoad { .. } => true,
            MirOp::TileBinary { lhs, rhs, .. } => {
                self.can_eval_tile_value(*lhs, pending_tiles, mir)
                    && self.can_eval_tile_value(*rhs, pending_tiles, mir)
            }
            MirOp::TileUnary { operand, .. } => {
                self.can_eval_tile_value(*operand, pending_tiles, mir)
            }
            MirOp::TileBroadcast { value, .. } => {
                self.can_eval_tile_value(*value, pending_tiles, mir)
            }
            _ => false,
        }
    }
}

pub fn build_llir_lowering_plan(func: &MirFunction) -> LlirLoweringPlan {
    let mut deferred_tiles = HashSet::new();

    for block in &func.blocks {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            if is_defer_candidate(&inst.op)
                && is_single_use_within_block(block, inst_idx, inst.result)
            {
                deferred_tiles.insert(inst.result);
            }
        }
    }

    LlirLoweringPlan { deferred_tiles }
}

fn is_defer_candidate(op: &MirOp) -> bool {
    matches!(
        op,
        MirOp::TileConstant { .. }
            | MirOp::TileLoad { .. }
            | MirOp::TileBinary { .. }
            | MirOp::TileUnary { .. }
            | MirOp::TileBroadcast { .. }
    )
}

fn is_single_use_within_block(block: &MirBlock, inst_idx: usize, value: ValueId) -> bool {
    if MirFunction::terminator_uses(&block.terminator).contains(&value) {
        return false;
    }

    let mut uses = 0usize;
    for inst in block.insts.iter().skip(inst_idx + 1) {
        uses += MirFunction::inst_uses(&inst.op)
            .into_iter()
            .filter(|used| *used == value)
            .count();
        if uses > 1 {
            return false;
        }
    }
    uses == 1
}
