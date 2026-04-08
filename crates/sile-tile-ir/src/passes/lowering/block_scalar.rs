use sile_llvm_ir as llvm_ir;

use super::core::{
    LowerLlvmIrCtx, const_i64, emit_bin, emit_gep, emit_shape_dim, llvm_ir_type, llvm_ir_value,
    load_tile_scalar_dynamic, lower_bin_op, lower_cmp_pred, resolve_operand,
};
use crate::ir::*;

pub(crate) fn lower_scalar_inst(
    inst: &TileIrInst,
    tile_ir: &TileIrFunction,
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
) {
    match &inst.op {
        TileIrOp::ConstI64(value) => {
            ctx.operands.insert(
                inst.result,
                llvm_ir::Operand::Const(llvm_ir::Constant::Int(*value)),
            );
            ctx.names
                .insert(llvm_ir_value(inst.result), format!("v{}", inst.result.0));
        }
        TileIrOp::ConstF64(value) => {
            ctx.operands.insert(
                inst.result,
                llvm_ir::Operand::Const(llvm_ir::Constant::Float(*value)),
            );
            ctx.names
                .insert(llvm_ir_value(inst.result), format!("v{}", inst.result.0));
        }
        TileIrOp::ShapeDim { shape, dim } => {
            let shape_desc = resolve_operand(*shape, ctx);
            if let llvm_ir::Operand::Value(shape_desc_id) = shape_desc.clone()
                && let Some(cached) = ctx.shape_dim_cache.get(&(shape_desc_id, *dim)).cloned()
            {
                ctx.operands.insert(inst.result, cached);
                return;
            }
            let ptr = emit_gep(
                ctx,
                out,
                shape_desc.clone(),
                vec![const_i64(*dim as i64)],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
            );
            let value = llvm_ir::Operand::Value(llvm_ir_value(inst.result));
            let llir_id = llvm_ir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            out.push(llvm_ir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llvm_ir::Type::I64,
                op: llvm_ir::InstOp::Load { ptr },
                metadata: Vec::new(),
            });
            ctx.operands.insert(inst.result, value.clone());
            if let llvm_ir::Operand::Value(shape_desc_id) = shape_desc {
                ctx.shape_dim_cache
                    .entry((shape_desc_id, *dim))
                    .or_insert(value);
            }
        }
        TileIrOp::IBinary { op, lhs, rhs } => {
            let llir_id = llvm_ir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llvm_ir::Operand::Value(llir_id));
            out.push(llvm_ir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llvm_ir_type(tile_ir.types.get(&inst.result).unwrap_or(&TileIrType::I64)),
                op: llvm_ir::InstOp::Bin {
                    op: lower_bin_op(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        TileIrOp::ICmp { op, lhs, rhs } => {
            let llir_id = llvm_ir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llvm_ir::Operand::Value(llir_id));
            out.push(llvm_ir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llvm_ir::Type::I1,
                op: llvm_ir::InstOp::Cmp {
                    pred: lower_cmp_pred(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        TileIrOp::SileAtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            stride_shape_idx,
        } => {
            let buf_operand = resolve_operand(*buf, ctx);
            let row_operand = resolve_operand(*row_coord, ctx);
            let col_operand = resolve_operand(*col_coord, ctx);
            let linear_index = match tile_ir.types.get(buf) {
                Some(TileIrType::Buffer { rank }) if *rank <= 1 => col_operand,
                _ => {
                    let stride = emit_shape_dim(ctx, out, buf_operand.clone(), *stride_shape_idx);
                    let row_offset = emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Mul,
                        row_operand,
                        stride,
                        llvm_ir::Type::I64,
                    );
                    emit_bin(
                        ctx,
                        out,
                        llvm_ir::BinOp::Add,
                        row_offset,
                        col_operand,
                        llvm_ir::Type::I64,
                    )
                }
            };
            let ptr = emit_gep(
                ctx,
                out,
                buf_operand,
                vec![linear_index],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
            );
            let value_operand = match tile_ir.types.get(value) {
                Some(TileIrType::Tile { rows: 1, cols: 1 }) => load_tile_scalar_dynamic(
                    ctx,
                    out,
                    resolve_operand(*value, ctx),
                    const_i64(0),
                    const_i64(0),
                ),
                _ => resolve_operand(*value, ctx),
            };
            out.push(llvm_ir::Inst {
                result: None,
                result_name: None,
                ty: llvm_ir::Type::Void,
                op: llvm_ir::InstOp::AtomicAdd {
                    ptr,
                    value: value_operand,
                },
                metadata: Vec::new(),
            });
        }
        TileIrOp::Extract {
            tile,
            row_coord,
            col_coord,
        } => {
            let value = load_tile_scalar_dynamic(
                ctx,
                out,
                resolve_operand(*tile, ctx),
                resolve_operand(*row_coord, ctx),
                resolve_operand(*col_coord, ctx),
            );
            ctx.operands.insert(inst.result, value);
        }
        TileIrOp::LoadPtrTko { .. }
        | TileIrOp::StorePtrTko { .. }
        | TileIrOp::Splat { .. }
        | TileIrOp::AddF { .. }
        | TileIrOp::SubF { .. }
        | TileIrOp::MulF { .. }
        | TileIrOp::DivF { .. }
        | TileIrOp::NegF { .. }
        | TileIrOp::Exp { .. }
        | TileIrOp::MmaF { .. }
        | TileIrOp::ReduceSum { .. }
        | TileIrOp::ReduceMax { .. }
        | TileIrOp::Broadcast { .. }
        | TileIrOp::Map { .. } => {
            unreachable!("tile ops are lowered by dedicated tile lowering paths")
        }
    }
}
