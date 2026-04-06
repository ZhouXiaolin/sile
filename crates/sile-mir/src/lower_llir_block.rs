use sile_llir as llir;

use crate::ir::*;
use crate::lower_llir_block_scalar::lower_scalar_inst;
use crate::lower_llir_block_terminator::lower_terminator;
use crate::lower_llir_core::{BlockLowerer, LowerLlirCtx, llir_block, llir_type, llir_value};
use crate::lower_llir_tile_compute::{lower_tile_mma_inst, lower_tile_reduce_inst};
use crate::lower_llir_tile_deferred::{lower_planned_tile_op, materialize_deferred_tile};
use crate::lower_llir_tile_memory::lower_tile_store_inst;
use crate::passes::LlirLoweringPlan;

pub(crate) fn lower_block(
    block: &MirBlock,
    mir: &MirFunction,
    plan: &LlirLoweringPlan,
    ctx: &mut LowerLlirCtx,
) -> Vec<llir::BasicBlock> {
    let params = block
        .params
        .iter()
        .map(|param| {
            let id = llir_value(*param);
            let name = format!("v{}", param.0);
            ctx.names.insert(id, name.clone());
            ctx.operands.insert(*param, llir::Operand::Value(id));
            llir::BlockParam {
                id,
                name,
                ty: llir_type(mir.types.get(param).unwrap_or(&MirType::Void)),
            }
        })
        .collect::<Vec<_>>();

    let mut builder = BlockLowerer::new(
        mir,
        plan,
        ctx,
        llir_block(block.id),
        format!("bb{}", block.id.0),
        params,
    );

    for (inst_idx, inst) in block.insts.iter().enumerate() {
        let to_materialize = builder
            .plan()
            .values_to_materialize(block.id, inst_idx)
            .to_vec();
        for value in to_materialize {
            materialize_deferred_tile(value, &mut builder);
        }

        if lower_planned_tile_op(inst.result, &inst.op, &mut builder) {
            continue;
        }

        match &inst.op {
            MirOp::TileStore {
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
            MirOp::TileMma {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                tile_k,
            } => {
                lower_tile_mma_inst(
                    inst.result,
                    *a,
                    *b,
                    *acc,
                    *tile_m,
                    *tile_n,
                    *tile_k,
                    &mut builder,
                );
            }
            MirOp::TileReduce {
                op,
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    *op,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            _ => {
                builder.with_current_insts(|ctx, mir, out| lower_scalar_inst(inst, mir, ctx, out));
            }
        }
    }

    let terminator = lower_terminator(&block.terminator, builder.ctx());
    builder.finish(terminator)
}
