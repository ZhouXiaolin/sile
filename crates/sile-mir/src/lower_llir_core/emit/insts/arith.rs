use sile_llir as llir;

use crate::lower_llir_core::block::LowerLlirCtx;

pub(crate) fn lower_1d_tile_coord(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    row_coord: llir::Operand,
    col_coord: llir::Operand,
) -> llir::Operand {
    let non_zero = emit_cmp(
        ctx,
        out,
        llir::CmpPred::Ne,
        col_coord.clone(),
        const_i64(0),
        llir::Type::I1,
    );
    emit_select(ctx, out, non_zero, col_coord, row_coord, llir::Type::I64)
}

pub(crate) fn emit_bin(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    op: llir::BinOp,
    lhs: llir::Operand,
    rhs: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Bin { op, lhs, rhs },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_cmp(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    pred: llir::CmpPred,
    lhs: llir::Operand,
    rhs: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Cmp { pred, lhs, rhs },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_select(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    cond: llir::Operand,
    on_true: llir::Operand,
    on_false: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Select {
            cond,
            on_true,
            on_false,
        },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_intrinsic(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    intrinsic: llir::Intrinsic,
    args: Vec<llir::Operand>,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Intrinsic { intrinsic, args },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn const_i64(value: i64) -> llir::Operand {
    llir::Operand::Const(llir::Constant::Int(value))
}

pub(crate) fn const_f32(value: f64) -> llir::Operand {
    llir::Operand::Const(llir::Constant::Float(value))
}
