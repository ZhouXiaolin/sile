use sile_llir as llir;

use crate::passes::lowering::core::block::LowerLlirCtx;

pub(crate) fn load_tile_scalar_dynamic(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    tile: llir::Operand,
    row: llir::Operand,
    col: llir::Operand,
) -> llir::Operand {
    let ptr = emit_gep(
        ctx,
        out,
        tile,
        vec![row, col],
        llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
    );
    emit_load(ctx, out, ptr, llir::Type::F32)
}

pub(crate) fn emit_shape_dim(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    buf: llir::Operand,
    dim: usize,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty: llir::Type::I64,
        op: llir::InstOp::ShapeDim { buf, dim },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_gep(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    base: llir::Operand,
    indices: Vec<llir::Operand>,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Gep { base, indices },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_load(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    ptr: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Load { ptr },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

pub(crate) fn emit_store(out: &mut Vec<llir::Inst>, ptr: llir::Operand, value: llir::Operand) {
    out.push(llir::Inst {
        result: None,
        result_name: None,
        ty: llir::Type::Void,
        op: llir::InstOp::Store { ptr, value },
        metadata: Vec::new(),
    });
}
