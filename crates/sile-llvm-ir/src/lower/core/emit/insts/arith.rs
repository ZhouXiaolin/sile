use crate as llvm_ir;

use crate::lower::core::block::LowerLlvmIrCtx;

pub(crate) fn lower_1d_tile_coord(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    row_coord: llvm_ir::Operand,
    col_coord: llvm_ir::Operand,
) -> llvm_ir::Operand {
    if let llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)) = row_coord {
        return col_coord;
    }
    if let llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)) = col_coord {
        return row_coord;
    }
    if row_coord == col_coord {
        return row_coord;
    }
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
    // Integer arithmetic whose value is already known here stays out of the
    // emitted source: every target would otherwise spell the same fold as
    // text, and a degenerate axis division (`index / 1`) reads worse than the
    // value it computes.
    if matches!(ty, llvm_ir::Type::I64 | llvm_ir::Type::I32) {
        if let llvm_ir::BinOp::Div | llvm_ir::BinOp::Rem = op
            && let llvm_ir::Operand::Const(llvm_ir::Constant::Int(1)) = rhs
        {
            return if matches!(op, llvm_ir::BinOp::Div) {
                lhs
            } else {
                const_i64(0)
            };
        }
        if let (
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(a)),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(b)),
        ) = (&lhs, &rhs)
            && let Some(value) = fold_int_const(op, *a, *b)
        {
            return const_i64(value);
        }
    }
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

/// The value of an integer constant operation, `None` when it has none: a
/// division by zero folds nowhere, and overflow leaves the operation for the
/// target to spell.
fn fold_int_const(op: llvm_ir::BinOp, a: i64, b: i64) -> Option<i64> {
    match op {
        llvm_ir::BinOp::Add => a.checked_add(b),
        llvm_ir::BinOp::Sub => a.checked_sub(b),
        llvm_ir::BinOp::Mul => a.checked_mul(b),
        llvm_ir::BinOp::Div if b != 0 => a.checked_div(b),
        llvm_ir::BinOp::Rem if b != 0 => a.checked_rem(b),
        llvm_ir::BinOp::And => Some(a & b),
        llvm_ir::BinOp::Or => Some(a | b),
        llvm_ir::BinOp::Xor => Some(a ^ b),
        _ => None,
    }
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

/// A conversion from one scalar type to another.
///
/// The instruction names both the conversion and the type it produces, so a
/// target that has no implicit conversion still knows what to write.
pub(crate) fn emit_cast(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    op: llvm_ir::CastOp,
    value: llvm_ir::Operand,
    to: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty: to.clone(),
        op: llvm_ir::InstOp::Cast { op, value, to },
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
