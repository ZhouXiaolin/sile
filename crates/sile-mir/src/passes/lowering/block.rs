use sile_llir as llir;

use crate::ir::*;

use super::block_scalar::lower_scalar_inst;
use super::block_terminator::lower_terminator;
use super::core::{BlockLowerer, LowerLlirCtx, llir_block, llir_type, llir_value};
use super::tile_compute::{lower_tile_mma_inst, lower_tile_reduce_inst};
use super::tile_expr::lower_tile_expr_inst;
use super::tile_memory::{lower_tile_constant_inst, lower_tile_load_inst, lower_tile_store_inst};

pub(crate) fn lower_block(
    block: &MirBlock,
    mir: &MirFunction,
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
        ctx,
        llir_block(block.id),
        format!("bb{}", block.id.0),
        params,
    );

    for inst in &block.insts {
        match &inst.op {
            MirOp::TileConstant { value, rows, cols } => {
                lower_tile_constant_inst(inst.result, *value, *rows, *cols, &mut builder);
            }
            MirOp::TileLoad {
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
            MirOp::TileBinary { rows, cols, .. }
            | MirOp::TileUnary { rows, cols, .. }
            | MirOp::TileBroadcast { rows, cols, .. } => {
                lower_tile_expr_inst(inst.result, inst.op.clone(), *rows, *cols, &mut builder);
            }
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
