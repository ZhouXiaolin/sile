use std::collections::{HashMap, HashSet};

use sile_llvm_ir as llvm_ir;

use crate::ir::*;

use super::block_scalar::lower_scalar_inst;
use super::block_terminator::lower_terminator;
use super::core::{BlockLowerer, LowerLlvmIrCtx, llvm_ir_block, llvm_ir_type, llvm_ir_value};
use super::tile_compute::{
    FusedAccInit, FusedTileLoad, lower_fused_tile_mma_inst, lower_tile_mma_inst,
    lower_tile_reduce_inst,
};
use super::tile_expr::{
    PointwiseTileDefs, build_pointwise_tile_defs, lower_pointwise_rank1_store_inst,
    lower_pointwise_rank1_store_map_inst, lower_tile_expr_inst, lower_tile_map_inst,
};
use super::tile_memory::{
    lower_tile_constant_inst, lower_tile_load_inst, lower_tile_store_inst, lower_tile_uninit_inst,
};

pub(crate) fn lower_block(
    block: &TileIrBlock,
    tile_ir: &TileIrFunction,
    ctx: &mut LowerLlvmIrCtx,
) -> Vec<llvm_ir::BasicBlock> {
    let params = block
        .params
        .iter()
        .map(|param| {
            let id = llvm_ir_value(*param);
            let name = format!("v{}", param.0);
            ctx.names.insert(id, name.clone());
            ctx.operands.insert(*param, llvm_ir::Operand::Value(id));
            llvm_ir::BlockParam {
                id,
                name,
                ty: llvm_ir_type(tile_ir.types.get(param).unwrap_or(&TileIrType::Void)),
            }
        })
        .collect::<Vec<_>>();

    let mut builder = BlockLowerer::new(
        tile_ir,
        ctx,
        llvm_ir_block(block.id),
        format!("bb{}", block.id.0),
        params,
    );
    let pointwise_plan = PointwiseLoweringPlan::build(block, tile_ir);

    for inst in &block.insts {
        if let Some(expr) = pointwise_plan.store_maps.get(&inst.result) {
            let TileIrOp::StorePtrTko {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                ..
            } = &inst.op
            else {
                unreachable!("store map plan must only be attached to store ops");
            };
            lower_pointwise_rank1_store_map_inst(
                *buf,
                expr.clone(),
                *row_coord,
                *col_coord,
                *rows,
                *cols,
                &mut builder,
            );
            continue;
        }
        if let Some(pending_tiles) = pointwise_plan.store_fusions.get(&inst.result) {
            let TileIrOp::StorePtrTko {
                buf,
                value,
                row_coord,
                col_coord,
                rows,
                cols,
                ..
            } = &inst.op
            else {
                unreachable!("store fusion plan must only be attached to store ops");
            };
            lower_pointwise_rank1_store_inst(
                *buf,
                *value,
                *row_coord,
                *col_coord,
                *rows,
                *cols,
                pending_tiles.clone(),
                &mut builder,
            );
            continue;
        }
        if pointwise_plan.skipped_values.contains(&inst.result) {
            continue;
        }
        match &inst.op {
            TileIrOp::Splat { value, rows, cols } => {
                if pointwise_plan.uninitialized_splats.contains(&inst.result) {
                    lower_tile_uninit_inst(inst.result, *rows, *cols, &mut builder);
                } else {
                    lower_tile_constant_inst(inst.result, *value, *rows, *cols, &mut builder);
                }
            }
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
            } => {
                lower_tile_load_inst(
                    inst.result,
                    *buf,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
                );
            }
            TileIrOp::AddF { rows, cols, .. }
            | TileIrOp::SubF { rows, cols, .. }
            | TileIrOp::MulF { rows, cols, .. }
            | TileIrOp::DivF { rows, cols, .. }
            | TileIrOp::NegF { rows, cols, .. }
            | TileIrOp::Exp { rows, cols, .. }
            | TileIrOp::Broadcast { rows, cols, .. } => {
                let pending_tiles = pointwise_plan
                    .expr_fusions
                    .get(&inst.result)
                    .cloned()
                    .unwrap_or_else(|| HashMap::from([(inst.result, inst.op.clone())]));
                lower_tile_expr_inst(inst.result, *rows, *cols, pending_tiles, &mut builder);
            }
            TileIrOp::Map { expr, rows, cols } => {
                lower_tile_map_inst(inst.result, expr.clone(), *rows, *cols, &mut builder);
            }
            TileIrOp::StorePtrTko {
                buf,
                value,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
            } => {
                lower_tile_store_inst(
                    *buf,
                    *value,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
                );
            }
            TileIrOp::MmaF {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                tile_k,
            } => match pointwise_plan.mma_fusions.get(&inst.result) {
                Some(plan) => lower_fused_tile_mma_inst(
                    inst.result,
                    plan.a_load,
                    plan.b_load,
                    plan.acc,
                    plan.acc_init,
                    plan.tile_m,
                    plan.tile_n,
                    plan.tile_k,
                    &mut builder,
                ),
                None => lower_tile_mma_inst(
                    inst.result,
                    *a,
                    *b,
                    *acc,
                    *tile_m,
                    *tile_n,
                    *tile_k,
                    &mut builder,
                ),
            },
            TileIrOp::ReduceSum {
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    false,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            TileIrOp::ReduceMax {
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    true,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            _ => {
                builder.with_current_insts(|ctx, tile_ir, out| {
                    lower_scalar_inst(inst, tile_ir, ctx, out)
                });
            }
        }
    }

    let terminator = lower_terminator(&block.terminator, builder.ctx());
    builder.finish(terminator)
}

struct PointwiseLoweringPlan {
    mma_fusions: HashMap<ValueId, MmaFusionPlan>,
    store_maps: HashMap<ValueId, TileMapExpr>,
    store_fusions: HashMap<ValueId, PointwiseTileDefs>,
    expr_fusions: HashMap<ValueId, PointwiseTileDefs>,
    uninitialized_splats: HashSet<ValueId>,
    skipped_values: HashSet<ValueId>,
}

impl PointwiseLoweringPlan {
    fn build(block: &TileIrBlock, tile_ir: &TileIrFunction) -> Self {
        let defs = block
            .insts
            .iter()
            .map(|inst| (inst.result, inst.op.clone()))
            .collect::<HashMap<_, _>>();
        let global_defs = tile_ir
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .map(|inst| (inst.result, inst.op.clone()))
            .collect::<HashMap<_, _>>();
        let param_owners = build_param_owners(tile_ir);
        let predecessors = build_predecessors(tile_ir);
        let use_counts = build_use_counts(tile_ir);
        let uninitialized_splats = collect_loop_carried_acc_seed_skips(
            tile_ir,
            &global_defs,
            &param_owners,
            &predecessors,
            &use_counts,
        );
        let mut skipped_values = HashSet::new();

        let mut mma_fusions = HashMap::new();
        let mut store_maps = HashMap::new();
        let mut store_fusions = HashMap::new();
        let mut expr_fusions = HashMap::new();

        for inst in &block.insts {
            let TileIrOp::MmaF {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                tile_k,
            } = &inst.op
            else {
                continue;
            };
            let Some(TileIrOp::LoadPtrTko {
                buf: a_buf,
                row_coord: a_row_coord,
                col_coord: a_col_coord,
                rows: a_rows,
                cols: a_cols,
                stride_shape_idx: a_stride_shape_idx,
            }) = defs.get(a)
            else {
                continue;
            };
            let Some(TileIrOp::LoadPtrTko {
                buf: b_buf,
                row_coord: b_row_coord,
                col_coord: b_col_coord,
                rows: b_rows,
                cols: b_cols,
                stride_shape_idx: b_stride_shape_idx,
            }) = defs.get(b)
            else {
                continue;
            };
            if use_counts.get(a).copied().unwrap_or(0) != 1
                || use_counts.get(b).copied().unwrap_or(0) != 1
                || use_counts.get(acc).copied().unwrap_or(0) != 1
            {
                continue;
            }
            mma_fusions.insert(
                inst.result,
                MmaFusionPlan {
                    a_load: FusedTileLoad {
                        buf: *a_buf,
                        row_coord: *a_row_coord,
                        col_coord: *a_col_coord,
                        rows: *a_rows,
                        cols: *a_cols,
                        stride_shape_idx: *a_stride_shape_idx,
                    },
                    b_load: FusedTileLoad {
                        buf: *b_buf,
                        row_coord: *b_row_coord,
                        col_coord: *b_col_coord,
                        rows: *b_rows,
                        cols: *b_cols,
                        stride_shape_idx: *b_stride_shape_idx,
                    },
                    acc: *acc,
                    acc_init: infer_loop_carried_acc_init(
                        *acc,
                        inst.result,
                        *tile_m,
                        *tile_n,
                        *a_col_coord,
                        *b_row_coord,
                        &global_defs,
                        &param_owners,
                        &predecessors,
                    )
                    .map(|analysis| FusedAccInit::LoopCarriedSplat {
                        value: analysis.value,
                        first_k_coord: analysis.first_k_coord,
                    })
                    .unwrap_or(FusedAccInit::ExistingTile),
                    tile_m: *tile_m,
                    tile_n: *tile_n,
                    tile_k: *tile_k,
                },
            );
            skipped_values.insert(*a);
            skipped_values.insert(*b);
        }

        for inst in &block.insts {
            let TileIrOp::StorePtrTko {
                buf, value, rows, ..
            } = &inst.op
            else {
                continue;
            };
            if *rows != 1 || buffer_rank_of(*buf, tile_ir) != 1 {
                continue;
            }
            if let Some(TileIrOp::Map { expr, .. }) = defs.get(value) {
                if use_counts.get(value).copied().unwrap_or(0) == 1 {
                    store_maps.insert(inst.result, expr.clone());
                    skipped_values.insert(*value);
                    continue;
                }
            }
            let Some(pending_tiles) =
                build_pointwise_tile_defs(*value, &defs, tile_ir, &use_counts, true)
            else {
                continue;
            };
            skipped_values.extend(pending_tiles.keys().copied());
            store_fusions.insert(inst.result, pending_tiles);
        }

        for inst in &block.insts {
            if skipped_values.contains(&inst.result) {
                continue;
            }
            let rows_cols = match &inst.op {
                TileIrOp::AddF { rows, cols, .. }
                | TileIrOp::SubF { rows, cols, .. }
                | TileIrOp::MulF { rows, cols, .. }
                | TileIrOp::DivF { rows, cols, .. }
                | TileIrOp::NegF { rows, cols, .. }
                | TileIrOp::Exp { rows, cols, .. }
                | TileIrOp::Broadcast { rows, cols, .. } => Some((*rows, *cols)),
                _ => None,
            };
            if rows_cols.is_none() {
                continue;
            }
            let Some(pending_tiles) =
                build_pointwise_tile_defs(inst.result, &defs, tile_ir, &use_counts, false)
            else {
                continue;
            };
            if pending_tiles.len() <= 1 {
                continue;
            }
            skipped_values.extend(
                pending_tiles
                    .keys()
                    .copied()
                    .filter(|value| *value != inst.result),
            );
            expr_fusions.insert(inst.result, pending_tiles);
        }

        Self {
            mma_fusions,
            store_maps,
            store_fusions,
            expr_fusions,
            uninitialized_splats,
            skipped_values,
        }
    }
}

#[derive(Clone, Copy)]
struct MmaFusionPlan {
    a_load: FusedTileLoad,
    b_load: FusedTileLoad,
    acc: ValueId,
    acc_init: FusedAccInit,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
}

#[derive(Clone)]
struct IncomingEdge {
    args: Vec<ValueId>,
}

#[derive(Clone, Copy, PartialEq)]
struct SplatSeed {
    value_id: ValueId,
    value: f64,
    rows: i64,
    cols: i64,
}

struct AccSeedTrace {
    seed: Option<SplatSeed>,
    has_backedge: bool,
}

struct LoopCarriedAccInitAnalysis {
    seed_value: ValueId,
    value: f64,
    first_k_coord: ValueId,
}

fn build_use_counts(tile_ir: &TileIrFunction) -> HashMap<ValueId, usize> {
    let mut counts = HashMap::new();
    for block in &tile_ir.blocks {
        for inst in &block.insts {
            for used in TileIrFunction::inst_uses(&inst.op) {
                *counts.entry(used).or_insert(0) += 1;
            }
        }
        for used in TileIrFunction::terminator_uses(&block.terminator) {
            *counts.entry(used).or_insert(0) += 1;
        }
    }
    counts
}

fn build_param_owners(tile_ir: &TileIrFunction) -> HashMap<ValueId, (BlockId, usize)> {
    let mut owners = HashMap::new();
    for block in &tile_ir.blocks {
        for (index, param) in block.params.iter().copied().enumerate() {
            owners.insert(param, (block.id, index));
        }
    }
    owners
}

fn build_predecessors(tile_ir: &TileIrFunction) -> HashMap<BlockId, Vec<IncomingEdge>> {
    let mut predecessors = HashMap::new();
    for block in &tile_ir.blocks {
        match &block.terminator {
            TileIrTerminator::Jump { target, args } => {
                predecessors
                    .entry(*target)
                    .or_insert_with(Vec::new)
                    .push(IncomingEdge { args: args.clone() });
            }
            TileIrTerminator::Branch {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                predecessors
                    .entry(*true_target)
                    .or_insert_with(Vec::new)
                    .push(IncomingEdge {
                        args: true_args.clone(),
                    });
                predecessors
                    .entry(*false_target)
                    .or_insert_with(Vec::new)
                    .push(IncomingEdge {
                        args: false_args.clone(),
                    });
            }
            TileIrTerminator::Return => {}
        }
    }
    predecessors
}

fn infer_loop_carried_acc_init(
    acc: ValueId,
    mma_result: ValueId,
    tile_m: i64,
    tile_n: i64,
    a_k_coord: ValueId,
    b_k_coord: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    param_owners: &HashMap<ValueId, (BlockId, usize)>,
    predecessors: &HashMap<BlockId, Vec<IncomingEdge>>,
) -> Option<LoopCarriedAccInitAnalysis> {
    if a_k_coord != b_k_coord {
        return None;
    }
    let trace = trace_loop_carried_splat_seed(
        acc,
        mma_result,
        defs,
        param_owners,
        predecessors,
        &mut HashSet::new(),
    )?;
    let seed = trace.seed?;
    if !trace.has_backedge || seed.rows != tile_m || seed.cols != tile_n {
        return None;
    }
    Some(LoopCarriedAccInitAnalysis {
        seed_value: seed.value_id,
        value: seed.value,
        first_k_coord: a_k_coord,
    })
}

fn collect_loop_carried_acc_seed_skips(
    tile_ir: &TileIrFunction,
    defs: &HashMap<ValueId, TileIrOp>,
    param_owners: &HashMap<ValueId, (BlockId, usize)>,
    predecessors: &HashMap<BlockId, Vec<IncomingEdge>>,
    use_counts: &HashMap<ValueId, usize>,
) -> HashSet<ValueId> {
    let mut skipped = HashSet::new();
    for block in &tile_ir.blocks {
        let block_defs = block
            .insts
            .iter()
            .map(|inst| (inst.result, inst.op.clone()))
            .collect::<HashMap<_, _>>();
        for inst in &block.insts {
            let TileIrOp::MmaF {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                ..
            } = &inst.op
            else {
                continue;
            };
            let Some(TileIrOp::LoadPtrTko { col_coord, .. }) = block_defs.get(a) else {
                continue;
            };
            let Some(TileIrOp::LoadPtrTko { row_coord, .. }) = block_defs.get(b) else {
                continue;
            };
            if use_counts.get(a).copied().unwrap_or(0) != 1
                || use_counts.get(b).copied().unwrap_or(0) != 1
                || use_counts.get(acc).copied().unwrap_or(0) != 1
            {
                continue;
            }
            if let Some(analysis) = infer_loop_carried_acc_init(
                *acc,
                inst.result,
                *tile_m,
                *tile_n,
                *col_coord,
                *row_coord,
                defs,
                param_owners,
                predecessors,
            ) {
                skipped.insert(analysis.seed_value);
            }
        }
    }
    skipped
}

fn trace_loop_carried_splat_seed(
    value: ValueId,
    mma_result: ValueId,
    defs: &HashMap<ValueId, TileIrOp>,
    param_owners: &HashMap<ValueId, (BlockId, usize)>,
    predecessors: &HashMap<BlockId, Vec<IncomingEdge>>,
    visiting: &mut HashSet<ValueId>,
) -> Option<AccSeedTrace> {
    if value == mma_result {
        return Some(AccSeedTrace {
            seed: None,
            has_backedge: true,
        });
    }

    if let Some(TileIrOp::Splat {
        value: splat_value,
        rows,
        cols,
    }) = defs.get(&value)
    {
        return Some(AccSeedTrace {
            seed: Some(SplatSeed {
                value_id: value,
                value: *splat_value,
                rows: *rows,
                cols: *cols,
            }),
            has_backedge: false,
        });
    }

    let (block_id, param_index) = param_owners.get(&value).copied()?;
    if !visiting.insert(value) {
        return None;
    }

    let predecessor_edges = predecessors.get(&block_id)?;
    let mut merged_seed = None;
    let mut has_backedge = false;
    for edge in predecessor_edges {
        let incoming = *edge.args.get(param_index)?;
        let trace = trace_loop_carried_splat_seed(
            incoming,
            mma_result,
            defs,
            param_owners,
            predecessors,
            visiting,
        )?;
        has_backedge |= trace.has_backedge;
        if let Some(seed) = trace.seed {
            match merged_seed {
                None => merged_seed = Some(seed),
                Some(existing)
                    if existing.value == seed.value
                        && existing.rows == seed.rows
                        && existing.cols == seed.cols => {}
                Some(_) => {
                    visiting.remove(&value);
                    return None;
                }
            }
        }
    }

    visiting.remove(&value);
    Some(AccSeedTrace {
        seed: merged_seed,
        has_backedge,
    })
}

fn buffer_rank_of(value: ValueId, tile_ir: &TileIrFunction) -> usize {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Buffer { rank }) => *rank,
        _ => 0,
    }
}
