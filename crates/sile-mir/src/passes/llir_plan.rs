use std::collections::{HashMap, HashSet};

use crate::{BlockId, MirBlock, MirFunction, MirOp, ValueId};

#[derive(Clone, Debug, Default)]
pub struct LlirLoweringPlan {
    deferred_tiles: HashSet<ValueId>,
    deferred_tile_ops: HashMap<ValueId, MirOp>,
    materialize_before_inst: HashMap<(BlockId, usize), Vec<ValueId>>,
}

impl LlirLoweringPlan {
    pub fn should_defer(&self, value: ValueId) -> bool {
        self.deferred_tiles.contains(&value)
    }

    pub fn deferred_tile_op(&self, value: ValueId) -> Option<&MirOp> {
        self.deferred_tile_ops.get(&value)
    }

    pub fn deferred_tiles_with_root(
        &self,
        result: ValueId,
        root_op: MirOp,
    ) -> HashMap<ValueId, MirOp> {
        let mut deferred_tiles = self.deferred_tile_ops.clone();
        deferred_tiles.insert(result, root_op);
        deferred_tiles
    }

    pub fn values_to_materialize(&self, block: BlockId, inst_idx: usize) -> &[ValueId] {
        self.materialize_before_inst
            .get(&(block, inst_idx))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

pub fn build_llir_lowering_plan(func: &MirFunction) -> LlirLoweringPlan {
    let mut deferred_tiles = HashSet::new();
    let mut deferred_tile_ops = HashMap::new();
    let mut materialize_before_inst = HashMap::new();

    for block in &func.blocks {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            if is_defer_candidate(&inst.op)
                && is_single_use_within_block(block, inst_idx, inst.result)
            {
                deferred_tiles.insert(inst.result);
                deferred_tile_ops.insert(inst.result, inst.op.clone());
            }
        }
    }

    for block in &func.blocks {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let values = values_requiring_materialization(&inst.op, &deferred_tiles);
            if !values.is_empty() {
                materialize_before_inst.insert((block.id, inst_idx), values);
            }
        }
    }

    LlirLoweringPlan {
        deferred_tiles,
        deferred_tile_ops,
        materialize_before_inst,
    }
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

fn values_requiring_materialization(op: &MirOp, deferred_tiles: &HashSet<ValueId>) -> Vec<ValueId> {
    let candidates = match op {
        MirOp::TileStore { value, .. } | MirOp::TileReduce { value, .. } => vec![*value],
        MirOp::TileMma { a, b, acc, .. } => vec![*a, *b, *acc],
        _ => Vec::new(),
    };

    candidates
        .into_iter()
        .filter(|value| deferred_tiles.contains(value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::{MirInst, MirTerminator, MirType};

    #[test]
    fn single_use_pointwise_tile_expr_is_deferred() {
        let value = ValueId(2);
        let func = MirFunction {
            name: "tile_expr_defer".into(),
            params: Vec::new(),
            blocks: vec![MirBlock {
                id: crate::BlockId(0),
                params: Vec::new(),
                insts: vec![
                    inst(
                        ValueId(1),
                        MirOp::TileLoad {
                            buf: ValueId(10),
                            row_coord: ValueId(11),
                            col_coord: ValueId(12),
                            rows: 2,
                            cols: 8,
                            stride_shape_idx: 1,
                        },
                    ),
                    inst(
                        value,
                        MirOp::TileUnary {
                            op: crate::UnaryOp::Exp,
                            operand: ValueId(1),
                            rows: 2,
                            cols: 8,
                        },
                    ),
                    inst(
                        ValueId(3),
                        MirOp::TileReduce {
                            op: crate::ReduceOp::Sum,
                            value,
                            axis: 1,
                            in_rows: 2,
                            in_cols: 8,
                        },
                    ),
                ],
                terminator: MirTerminator::Return,
            }],
            entry: crate::BlockId(0),
            types: HashMap::from([
                (ValueId(1), MirType::Tile { rows: 2, cols: 8 }),
                (value, MirType::Tile { rows: 2, cols: 8 }),
                (ValueId(3), MirType::Tile { rows: 2, cols: 1 }),
                (ValueId(10), MirType::Buffer { rank: 2 }),
                (ValueId(11), MirType::I64),
                (ValueId(12), MirType::I64),
            ]),
        };

        let plan = build_llir_lowering_plan(&func);
        assert!(plan.should_defer(value));
    }

    #[test]
    fn pointwise_tile_over_materialized_leaf_is_not_deferred() {
        let value = ValueId(3);
        let func = MirFunction {
            name: "tile_broadcast".into(),
            params: Vec::new(),
            blocks: vec![MirBlock {
                id: crate::BlockId(0),
                params: Vec::new(),
                insts: vec![
                    inst(
                        ValueId(1),
                        MirOp::TileLoad {
                            buf: ValueId(10),
                            row_coord: ValueId(11),
                            col_coord: ValueId(12),
                            rows: 2,
                            cols: 8,
                            stride_shape_idx: 1,
                        },
                    ),
                    inst(
                        ValueId(2),
                        MirOp::TileReduce {
                            op: crate::ReduceOp::Sum,
                            value: ValueId(1),
                            axis: 1,
                            in_rows: 2,
                            in_cols: 8,
                        },
                    ),
                    inst(
                        value,
                        MirOp::TileBroadcast {
                            value: ValueId(2),
                            rows: 2,
                            cols: 8,
                        },
                    ),
                ],
                terminator: MirTerminator::Return,
            }],
            entry: crate::BlockId(0),
            types: HashMap::from([
                (ValueId(1), MirType::Tile { rows: 2, cols: 8 }),
                (ValueId(2), MirType::Tile { rows: 2, cols: 1 }),
                (value, MirType::Tile { rows: 2, cols: 8 }),
                (ValueId(10), MirType::Buffer { rank: 2 }),
                (ValueId(11), MirType::I64),
                (ValueId(12), MirType::I64),
            ]),
        };

        let plan = build_llir_lowering_plan(&func);
        assert!(!plan.should_defer(value));
    }

    #[test]
    fn schedules_materialization_before_tile_store() {
        let deferred = ValueId(2);
        let func = MirFunction {
            name: "tile_store_materialize".into(),
            params: Vec::new(),
            blocks: vec![MirBlock {
                id: crate::BlockId(0),
                params: Vec::new(),
                insts: vec![
                    inst(
                        ValueId(1),
                        MirOp::TileLoad {
                            buf: ValueId(10),
                            row_coord: ValueId(11),
                            col_coord: ValueId(12),
                            rows: 2,
                            cols: 8,
                            stride_shape_idx: 1,
                        },
                    ),
                    inst(
                        deferred,
                        MirOp::TileUnary {
                            op: crate::UnaryOp::Exp,
                            operand: ValueId(1),
                            rows: 2,
                            cols: 8,
                        },
                    ),
                    inst(
                        ValueId(3),
                        MirOp::TileStore {
                            buf: ValueId(13),
                            value: deferred,
                            row_coord: ValueId(11),
                            col_coord: ValueId(12),
                            rows: 2,
                            cols: 8,
                            stride_shape_idx: 1,
                        },
                    ),
                ],
                terminator: MirTerminator::Return,
            }],
            entry: crate::BlockId(0),
            types: HashMap::from([
                (ValueId(1), MirType::Tile { rows: 2, cols: 8 }),
                (deferred, MirType::Tile { rows: 2, cols: 8 }),
                (ValueId(10), MirType::Buffer { rank: 2 }),
                (ValueId(11), MirType::I64),
                (ValueId(12), MirType::I64),
                (ValueId(13), MirType::Buffer { rank: 2 }),
            ]),
        };

        let plan = build_llir_lowering_plan(&func);
        assert_eq!(
            plan.values_to_materialize(crate::BlockId(0), 2),
            &[deferred]
        );
        assert!(matches!(
            plan.deferred_tile_op(deferred),
            Some(MirOp::TileUnary { .. })
        ));
    }

    fn inst(result: ValueId, op: MirOp) -> MirInst {
        MirInst { result, op }
    }
}
