use std::collections::{HashMap, HashSet};

use crate as llvm_ir;

use sile_tile_ir::function_ir::*;

use super::block_scalar::lower_scalar_inst;
use super::block_terminator::lower_terminator;
use super::core::{
    BlockLowerer, LowerLlvmIrCtx, llvm_ir_block, llvm_ir_type, llvm_ir_value, resolve_operand,
};
use super::tile_compute::{
    FusedAccInit, FusedTileLoad, LoadReduceFusion, MapReduceFusion, ReduceCombiner,
    lower_fused_load_reduce_inst, lower_fused_map_reduce_inst, lower_fused_tile_mma_inst,
    lower_tile_cat_inst, lower_tile_extract_slice_inst, lower_tile_insert_slice_inst,
    lower_tile_mma_inst, lower_tile_reduce_inst, lower_tile_scan_inst,
};
use super::tile_expr::{
    PointwiseTileDefs, build_pointwise_tile_defs, lower_pointwise_store_inst, lower_tile_expr_inst,
    lower_tile_map_inst,
};
use super::tile_memory::{
    lower_tile_constant_inst, lower_tile_load_inst, lower_tile_reshape_inst, lower_tile_store_inst,
    lower_tile_uninit_inst,
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
        if let Some(fusion) = pointwise_plan.load_reduce_fusions.get(&inst.result) {
            lower_fused_load_reduce_inst(inst.result, fusion, &mut builder);
            continue;
        }
        if let Some(fusion) = pointwise_plan.map_reduce_fusions.get(&inst.result) {
            lower_fused_map_reduce_inst(inst.result, fusion, &mut builder);
            continue;
        }
        if let Some(plan) = pointwise_plan.store_fusions.get(&inst.result) {
            let buf = match &inst.op {
                TileIrOp::StorePtrTko { buf, .. } => *buf,
                TileIrOp::StoreViewTko { view, .. } => *view,
                _ => unreachable!("store fusion plan must only be attached to store ops"),
            };
            lower_pointwise_store_inst(
                buf,
                plan.value,
                &plan.outer_coords,
                plan.row_coord,
                plan.col_coord,
                plan.rows,
                plan.cols,
                plan.stride_shape_idx,
                plan.pending_tiles.clone(),
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
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
                ..
            } => {
                lower_tile_load_inst(
                    inst.result,
                    *buf,
                    outer_coords,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
                );
            }
            // A view names the buffer it describes, so an access through it is
            // the same access: the view's operand was filled when the view was
            // made, and the addressing reads the buffer it resolves to.
            TileIrOp::MakePartitionView { source, .. } => {
                // The view resolves to the buffer it describes, so an access
                // through the view addresses that buffer.
                let operand = resolve_operand(*source, builder.ctx());
                builder.with_current_insts(|ctx, _, _| {
                    ctx.operands.insert(inst.result, operand);
                    ctx.view_sources.insert(inst.result, *source);
                });
            }
            TileIrOp::LoadViewTko {
                view,
                row_coord,
                col_coord,
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
                ..
            } => {
                lower_tile_load_inst(
                    inst.result,
                    *view,
                    outer_coords,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
                );
            }
            TileIrOp::StoreViewTko {
                view,
                value,
                row_coord,
                col_coord,
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
                ..
            } => {
                lower_tile_store_inst(
                    *view,
                    *value,
                    outer_coords,
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
            | TileIrOp::UnaryF { rows, cols, .. }
            | TileIrOp::Fma { rows, cols, .. }
            | TileIrOp::Select { rows, cols, .. }
            | TileIrOp::MaxF { rows, cols, .. }
            | TileIrOp::MinF { rows, cols, .. }
            | TileIrOp::Broadcast { rows, cols, .. }
                // A pointwise op over tiles fills private storage. Over
                // scalars it is one value, which the scalar path spells
                // without a loop: `min` of two counts must stay a count.
                if matches!(
                    builder.tile_ir().types.get(&inst.result),
                    Some(TileIrType::Tile { .. })
                ) =>
            {
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
            TileIrOp::Reshape { value, rows, cols } => {
                lower_tile_reshape_inst(inst.result, *value, *rows, *cols, &mut builder);
            }
            TileIrOp::StorePtrTko {
                buf,
                value,
                row_coord,
                col_coord,
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
                ..
            } => {
                lower_tile_store_inst(
                    *buf,
                    *value,
                    outer_coords,
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
                    ReduceCombiner::Sum,
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
                    ReduceCombiner::Max,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            TileIrOp::ReduceMin {
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    ReduceCombiner::Min,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            TileIrOp::ReduceProd {
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    ReduceCombiner::Prod,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            TileIrOp::Cat {
                lhs,
                rhs,
                axis,
                lhs_rows,
                lhs_cols,
                rhs_rows,
                rhs_cols,
            } => {
                lower_tile_cat_inst(
                    inst.result,
                    *lhs,
                    *rhs,
                    *axis,
                    *lhs_rows,
                    *lhs_cols,
                    *rhs_rows,
                    *rhs_cols,
                    &mut builder,
                );
            }
            TileIrOp::ExtractSlice {
                source,
                row_idx,
                col_idx,
                out_rows,
                out_cols,
                ..
            } => {
                lower_tile_extract_slice_inst(
                    inst.result,
                    *source,
                    *row_idx,
                    *col_idx,
                    *out_rows,
                    *out_cols,
                    &mut builder,
                );
            }
            TileIrOp::InsertSlice {
                source,
                destination,
                row_idx,
                col_idx,
                in_rows,
                in_cols,
                out_rows,
                out_cols,
                ..
            } => {
                lower_tile_insert_slice_inst(
                    inst.result,
                    *source,
                    *destination,
                    *row_idx,
                    *col_idx,
                    *in_rows,
                    *in_cols,
                    *out_rows,
                    *out_cols,
                    &mut builder,
                );
            }
            TileIrOp::Scan {
                value,
                identity,
                axis,
                in_rows,
                in_cols,
                reverse,
                combiner,
            } => {
                lower_tile_scan_inst(
                    inst.result,
                    *value,
                    *identity,
                    *combiner,
                    *axis,
                    *in_rows,
                    *in_cols,
                    *reverse,
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
    store_fusions: HashMap<ValueId, PointwiseStorePlan>,
    expr_fusions: HashMap<ValueId, PointwiseTileDefs>,
    load_reduce_fusions: HashMap<ValueId, LoadReduceFusion>,
    map_reduce_fusions: HashMap<ValueId, MapReduceFusion>,
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
        let mut store_fusions = HashMap::new();
        let mut expr_fusions = HashMap::new();
        let mut load_reduce_fusions = HashMap::new();
        let mut map_reduce_fusions = HashMap::new();

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
                outer_coords: a_outer,
                rows: a_rows,
                cols: a_cols,
                stride_shape_idx: a_stride_shape_idx,
                ..
            }) = defs.get(a)
            else {
                continue;
            };
            let Some(TileIrOp::LoadPtrTko {
                buf: b_buf,
                row_coord: b_row_coord,
                col_coord: b_col_coord,
                outer_coords: b_outer,
                rows: b_rows,
                cols: b_cols,
                stride_shape_idx: b_stride_shape_idx,
                ..
            }) = defs.get(b)
            else {
                continue;
            };
            // The fusion reads each operand at the coordinates the loop is
            // standing on, which only names the operand when the tile is the
            // whole of the axes above it.
            if !a_outer.is_empty() || !b_outer.is_empty() {
                continue;
            }
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
            // A store through a view writes the buffer the view describes; the
            // view's own value stands in for that buffer here.
            let (
                buf,
                value,
                outer_coords,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
            ) = match &inst.op {
                TileIrOp::StorePtrTko {
                    buf,
                    value,
                    outer_coords,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                    ..
                } => (
                    buf,
                    value,
                    outer_coords,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                ),
                TileIrOp::StoreViewTko {
                    view,
                    value,
                    outer_coords,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                    ..
                } => (
                    view,
                    value,
                    outer_coords,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                ),
                _ => continue,
            };
            // A rank-0 output receives one scalar per program through an
            // atomic add, so there is no tile walk to fuse into.
            if buffer_rank_of(*buf, tile_ir) == 0 {
                continue;
            }
            let Some(pending_tiles) =
                build_pointwise_tile_defs(*value, &defs, tile_ir, &use_counts, true)
            else {
                continue;
            };
            skipped_values.extend(pending_tiles.keys().copied());
            store_fusions.insert(
                inst.result,
                PointwiseStorePlan {
                    value: *value,
                    outer_coords: outer_coords.clone(),
                    row_coord: *row_coord,
                    col_coord: *col_coord,
                    rows: *rows,
                    cols: *cols,
                    stride_shape_idx: *stride_shape_idx,
                    pending_tiles,
                },
            );
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
                | TileIrOp::UnaryF { rows, cols, .. }
                | TileIrOp::Broadcast { rows, cols, .. } => Some((*rows, *cols)),
                // A chain root must fill storage; a scalar root is spelled in
                // place by the scalar path, so it never becomes a fusion.
                TileIrOp::Fma { rows, cols, .. }
                | TileIrOp::Select { rows, cols, .. }
                | TileIrOp::MaxF { rows, cols, .. }
                | TileIrOp::MinF { rows, cols, .. }
                    if matches!(
                        tile_ir.types.get(&inst.result),
                        Some(TileIrType::Tile { .. })
                    ) =>
                {
                    Some((*rows, *cols))
                }
                _ => None,
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

        for inst in &block.insts {
            if skipped_values.contains(&inst.result) {
                continue;
            }
            let (value, combiner, axis, in_rows, in_cols) = match &inst.op {
                TileIrOp::ReduceSum {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                } => (*value, ReduceCombiner::Sum, *axis, *in_rows, *in_cols),
                TileIrOp::ReduceMax {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                } => (*value, ReduceCombiner::Max, *axis, *in_rows, *in_cols),
                TileIrOp::ReduceMin {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                } => (*value, ReduceCombiner::Min, *axis, *in_rows, *in_cols),
                TileIrOp::ReduceProd {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                } => (*value, ReduceCombiner::Prod, *axis, *in_rows, *in_cols),
                _ => continue,
            };
            // Try Load-Reduce fusion first (safe even with multiple consumers
            // since the fused version reads directly from global memory).
            if let Some(TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                outer_coords,
                rows,
                cols,
                stride_shape_idx,
                ..
            }) = defs.get(&value)
            {
                load_reduce_fusions.insert(
                    inst.result,
                    LoadReduceFusion {
                        buf: *buf,
                        row_coord: *row_coord,
                        col_coord: *col_coord,
                        outer_coords: outer_coords.clone(),
                        src_rows: *rows,
                        src_cols: *cols,
                        stride_shape_idx: *stride_shape_idx,
                        combiner,
                        axis,
                        in_rows,
                        in_cols,
                    },
                );
                // Only skip the load value if it has a single consumer —
                // otherwise the load must still be emitted for other users.
                if use_counts.get(&value).copied().unwrap_or(0) == 1 {
                    skipped_values.insert(value);
                }
                continue;
            }
            // Try Map-Reduce fusion.  Only when the map has a single consumer
            // can we skip the standalone map emission and fuse the reduce.
            // When use_count > 1, the map is emitted standalone (alloca) and
            // the reduce falls back to lower_tile_reduce_inst which reads from
            // that alloca, avoiding duplicate computation.
            if let Some(TileIrOp::Map { expr, rows, cols }) = defs.get(&value) {
                let map_use_count = use_counts.get(&value).copied().unwrap_or(0);
                if map_use_count == 1 {
                    skipped_values.insert(value);
                    map_reduce_fusions.insert(
                        inst.result,
                        MapReduceFusion {
                            expr: expr.clone(),
                            src_rows: *rows,
                            src_cols: *cols,
                            combiner,
                            axis,
                            in_rows,
                            in_cols,
                        },
                    );
                }
            }
        }

        Self {
            mma_fusions,
            store_fusions,
            expr_fusions,
            load_reduce_fusions,
            map_reduce_fusions,
            uninitialized_splats,
            skipped_values,
        }
    }
}

/// One store lowered as the tail of a pointwise chain.
///
/// The chain's tiles are evaluated element by element straight into the output
/// buffer, so no private tile is allocated and no second loop reads it back.
/// `pending_tiles` holds every chain step including the root; a chain whose
/// root is a `Map` looks the same as one of plain arithmetic.
struct PointwiseStorePlan {
    value: ValueId,
    outer_coords: Vec<ValueId>,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    pending_tiles: PointwiseTileDefs,
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
        Some(TileIrType::Buffer { rank, .. }) => *rank,
        Some(TileIrType::TensorView(tv)) => tv.shape.len(),
        Some(TileIrType::PartitionView(pv)) => pv.tensor_view.shape.len(),
        Some(TileIrType::GatherScatterView(gv)) => gv.tensor_view.shape.len(),
        Some(TileIrType::StridedView(sv)) => sv.tensor_view.shape.len(),
        _ => 0,
    }
}
