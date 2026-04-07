use sile_llir as llir;

use super::core::{
    LowerLlirCtx, const_i64, emit_bin, emit_gep, emit_shape_dim, llir_type, llir_value,
    load_tile_scalar_dynamic, lower_bin_op, lower_cmp_pred, resolve_operand,
};
use crate::ir::*;

pub(crate) fn lower_scalar_inst(
    inst: &MirInst,
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
) {
    match &inst.op {
        MirOp::ConstI64(value) => {
            ctx.operands.insert(
                inst.result,
                llir::Operand::Const(llir::Constant::Int(*value)),
            );
            ctx.names
                .insert(llir_value(inst.result), format!("v{}", inst.result.0));
        }
        MirOp::ConstF64(value) => {
            ctx.operands.insert(
                inst.result,
                llir::Operand::Const(llir::Constant::Float(*value)),
            );
            ctx.names
                .insert(llir_value(inst.result), format!("v{}", inst.result.0));
        }
        MirOp::ProgramId { dim } => {
            if let Some(operand) = ctx.program_ids.get(&(*dim as u8)).cloned() {
                ctx.operands.insert(inst.result, operand);
                return;
            }
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            let operand = llir::Operand::Value(llir_id);
            ctx.operands.insert(inst.result, operand.clone());
            ctx.program_ids.insert(*dim as u8, operand);
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir_type(mir.types.get(&inst.result).unwrap_or(&MirType::I64)),
                op: llir::InstOp::Intrinsic {
                    intrinsic: llir::Intrinsic::BlockId { dim: *dim as u8 },
                    args: vec![],
                },
                metadata: Vec::new(),
            });
        }
        MirOp::ShapeDim { buf, dim } => {
            if let Some(operand) = ctx.shape_dims.get(&(*buf, *dim)).cloned() {
                ctx.operands.insert(inst.result, operand);
                return;
            }
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            let operand = llir::Operand::Value(llir_id);
            ctx.operands.insert(inst.result, operand.clone());
            ctx.shape_dims.insert((*buf, *dim), operand);
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir::Type::I64,
                op: llir::InstOp::ShapeDim {
                    buf: resolve_operand(*buf, ctx),
                    dim: *dim,
                },
                metadata: Vec::new(),
            });
        }
        MirOp::IBinary { op, lhs, rhs } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir_type(mir.types.get(&inst.result).unwrap_or(&MirType::I64)),
                op: llir::InstOp::Bin {
                    op: lower_bin_op(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        MirOp::ICmp { op, lhs, rhs } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir::Type::I1,
                op: llir::InstOp::Cmp {
                    pred: lower_cmp_pred(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        MirOp::AtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            stride_shape_idx,
        } => {
            let buf_operand = resolve_operand(*buf, ctx);
            let row_operand = resolve_operand(*row_coord, ctx);
            let col_operand = resolve_operand(*col_coord, ctx);
            let linear_index = match mir.types.get(buf) {
                Some(MirType::Buffer { rank }) if *rank <= 1 => col_operand,
                _ => {
                    let stride = emit_shape_dim(ctx, out, buf_operand.clone(), *stride_shape_idx);
                    let row_offset = emit_bin(
                        ctx,
                        out,
                        llir::BinOp::Mul,
                        row_operand,
                        stride,
                        llir::Type::I64,
                    );
                    emit_bin(
                        ctx,
                        out,
                        llir::BinOp::Add,
                        row_offset,
                        col_operand,
                        llir::Type::I64,
                    )
                }
            };
            let ptr = emit_gep(
                ctx,
                out,
                buf_operand,
                vec![linear_index],
                llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
            );
            let value_operand = match mir.types.get(value) {
                Some(MirType::Tile { rows: 1, cols: 1 }) => load_tile_scalar_dynamic(
                    ctx,
                    out,
                    resolve_operand(*value, ctx),
                    const_i64(0),
                    const_i64(0),
                ),
                _ => resolve_operand(*value, ctx),
            };
            out.push(llir::Inst {
                result: None,
                result_name: None,
                ty: llir::Type::Void,
                op: llir::InstOp::AtomicAdd {
                    ptr,
                    value: value_operand,
                },
                metadata: Vec::new(),
            });
        }
        MirOp::TileExtract {
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
        MirOp::TileLoad { .. }
        | MirOp::TileStore { .. }
        | MirOp::TileConstant { .. }
        | MirOp::TileBinary { .. }
        | MirOp::TileUnary { .. }
        | MirOp::TileMma { .. }
        | MirOp::TileReduce { .. }
        | MirOp::TileBroadcast { .. } => {
            unreachable!("tile ops are lowered by dedicated tile lowering paths")
        }
    }
}
