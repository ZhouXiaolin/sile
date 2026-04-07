use sile_llvm_ir as llvm_ir;

use crate::ir::*;

use super::block_scalar::lower_scalar_inst;
use super::block_terminator::lower_terminator;
use super::core::{BlockLowerer, LowerLlvmIrCtx, llvm_ir_block, llvm_ir_type, llvm_ir_value};
use super::tile_compute::{lower_tile_mma_inst, lower_tile_reduce_inst};
use super::tile_expr::lower_tile_expr_inst;
use super::tile_memory::{lower_tile_constant_inst, lower_tile_load_inst, lower_tile_store_inst};

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

    for inst in &block.insts {
        match &inst.op {
            TileIrOp::Splat { value, rows, cols } => {
                lower_tile_constant_inst(inst.result, *value, *rows, *cols, &mut builder);
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
                lower_tile_expr_inst(inst.result, inst.op.clone(), *rows, *cols, &mut builder);
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
