use sile_llvm_ir as llvm_ir;

use crate::passes::lowering::core::block::LowerLlvmIrCtx;
use crate::passes::lowering::core::const_i64;

pub(crate) fn load_tile_scalar_dynamic(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    tile: llvm_ir::Operand,
    row: llvm_ir::Operand,
    col: llvm_ir::Operand,
) -> llvm_ir::Operand {
    let ptr = emit_gep(
        ctx,
        out,
        tile,
        vec![row, col],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
    );
    emit_load(ctx, out, ptr, llvm_ir::Type::F32)
}

pub(crate) fn emit_shape_dim(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    dim: usize,
) -> llvm_ir::Operand {
    let llvm_ir::Operand::Value(buf_id) = buf else {
        panic!("shape_dim lowering requires buffer operand to be a value")
    };
    let shape_offset = *ctx
        .shape_offsets
        .get(&buf_id)
        .unwrap_or_else(|| panic!("missing shape offset for buffer value {}", buf_id.0));
    let shapes_param = ctx
        .shapes_param
        .clone()
        .expect("shape_dim lowering requires explicit __sile_shapes parameter");
    let ptr = emit_gep(
        ctx,
        out,
        shapes_param,
        vec![const_i64((shape_offset + dim) as i64)],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
    );
    emit_load(ctx, out, ptr, llvm_ir::Type::I64)
}

pub(crate) fn emit_gep(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    base: llvm_ir::Operand,
    indices: Vec<llvm_ir::Operand>,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Gep { base, indices },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn emit_load(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    ptr: llvm_ir::Operand,
    ty: llvm_ir::Type,
) -> llvm_ir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llvm_ir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llvm_ir::InstOp::Load { ptr },
        metadata: Vec::new(),
    });
    llvm_ir::Operand::Value(id)
}

pub(crate) fn emit_store(
    out: &mut Vec<llvm_ir::Inst>,
    ptr: llvm_ir::Operand,
    value: llvm_ir::Operand,
) {
    out.push(llvm_ir::Inst {
        result: None,
        result_name: None,
        ty: llvm_ir::Type::Void,
        op: llvm_ir::InstOp::Store { ptr, value },
        metadata: Vec::new(),
    });
}
