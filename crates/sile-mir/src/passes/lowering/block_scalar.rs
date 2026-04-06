use sile_llir as llir;

use super::core::{
    LowerLlirCtx, llir_type, llir_value, lower_bin_op, lower_cmp_pred, resolve_operand,
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
