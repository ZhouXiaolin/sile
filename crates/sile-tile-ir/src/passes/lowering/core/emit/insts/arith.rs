use sile_llvm_ir as llvm_ir;

use crate::passes::lowering::core::block::LowerLlvmIrCtx;

pub(crate) fn lower_1d_tile_coord(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    row_coord: llvm_ir::Operand,
    col_coord: llvm_ir::Operand,
) -> llvm_ir::Operand {
    let non_zero = emit_cmp(
        ctx,
        out,
        llvm_ir::CmpPred::Ne,
        col_coord.clone(),
        const_i64(0),
        llvm_ir::Type::I1,
    );
    emit_select(ctx, out, non_zero, col_coord, row_coord, llvm_ir::Type::I64)
}

pub(crate) fn emit_bin(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    op: llvm_ir::BinOp,
    lhs: llvm_ir::Operand,
    rhs: llvm_ir::Operand,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Bin { op, lhs, rhs },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn emit_cmp(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    pred: llvm_ir::CmpPred,
    lhs: llvm_ir::Operand,
    rhs: llvm_ir::Operand,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Cmp { pred, lhs, rhs },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn emit_select(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    cond: llvm_ir::Operand,
    on_true: llvm_ir::Operand,
    on_false: llvm_ir::Operand,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Select {
            cond,
            on_true,
            on_false,
        },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn emit_intrinsic(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    intrinsic: llvm_ir::Intrinsic,
    args: Vec<llvm_ir::Operand>,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Intrinsic { intrinsic, args },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn const_i64(value: i64) -> llvm_ir::Operand {
    llvm_ir::Operand::Const(llvm_ir::Constant::Int(value))
}

pub(crate) fn const_f32(value: f64) -> llvm_ir::Operand {
    llvm_ir::Operand::Const(llvm_ir::Constant::Float(value))
}
