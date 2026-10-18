use sile_llvm_ir as llvm_ir;

use super::core::{
    LowerLlvmIrCtx, comparison_pred, const_i64, emit_gep, emit_load, llvm_ir_type, llvm_ir_value,
    load_tile_scalar_dynamic, lower_bin_op, resolve_operand,
};
use crate::ir::*;
use crate::passes::lowering::tile_memory::emit_strided_element_index;

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
            // `shape` is a ShapeDesc value, not a buffer: read its `dim` entry.
            let loaded = load_shape_desc_dim(*shape, *dim, ctx, out);
            ctx.operands.insert(inst.result, loaded);
        }
        TileIrOp::GridDim { dim } => {
            let llir_id = llvm_ir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llvm_ir::Operand::Value(llir_id));
            out.push(llvm_ir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llvm_ir::Type::I64,
                op: llvm_ir::InstOp::Intrinsic {
                    intrinsic: llvm_ir::Intrinsic::GridDim {
                        dim: u8::try_from(*dim).unwrap_or(0),
                    },
                    args: Vec::new(),
                },
                metadata: Vec::new(),
            });
        }
        TileIrOp::NumTiles {
            shape,
            dim,
            tile_extent,
        } => {
            // num_tiles = shape_extent / tile_extent. The extent is the same
            // gep+load ShapeDim uses, so reuse that read and divide by the tile
            // extent the lowering already knows.
            let extent = load_shape_desc_dim(*shape, *dim, ctx, out);
            let llir_id = llvm_ir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llvm_ir::Operand::Value(llir_id));
            out.push(llvm_ir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llvm_ir::Type::I64,
                op: llvm_ir::InstOp::Bin {
                    op: llvm_ir::BinOp::Div,
                    lhs: extent,
                    rhs: const_i64(*tile_extent),
                },
                metadata: Vec::new(),
            });
        }
        TileIrOp::IBinary { op, lhs, rhs } | TileIrOp::FBinary { op, lhs, rhs } => {
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
                    // A comparison of two counts is a signed integer
                    // comparison, and a comparison of two elements is an
                    // ordered float one. Both are one instruction here, and
                    // the operands say which one this is.
                    pred: comparison_pred(*op, *lhs, *rhs, tile_ir),
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
            outer_coords,
            stride_shape_idx,
        } => {
            let buf_operand = resolve_operand(*buf, ctx);
            let row_operand = resolve_operand(*row_coord, ctx);
            let col_operand = resolve_operand(*col_coord, ctx);
            let rank = match tile_ir.types.get(buf) {
                Some(TileIrType::Buffer { rank }) => *rank,
                _ => 1,
            };
            let linear_index = if rank <= 1 {
                col_operand
            } else {
                let outer_operands: Vec<llvm_ir::Operand> = outer_coords
                    .iter()
                    .map(|coord| resolve_operand(*coord, ctx))
                    .collect();
                emit_strided_element_index(
                    ctx,
                    out,
                    buf_operand.clone(),
                    &outer_operands,
                    *stride_shape_idx,
                    row_operand,
                    col_operand,
                )
                .0
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
        | TileIrOp::UnaryF { .. }
        | TileIrOp::MmaF { .. }
        | TileIrOp::ReduceSum { .. }
        | TileIrOp::ReduceMax { .. }
        | TileIrOp::ReduceMin { .. }
        | TileIrOp::Broadcast { .. }
        | TileIrOp::Reshape { .. }
        | TileIrOp::Map { .. } => {
            unreachable!("tile ops are lowered by dedicated tile lowering paths")
        }
    }
}

/// Read entry `dim` of the shape descriptor behind `shape`.
///
/// `shape` is a ShapeDesc value, so it indexes the shapes array directly through
/// a gep plus a load, and repeats of the same `(descriptor, dim)` pair reuse the
/// first load.
fn load_shape_desc_dim(
    shape: crate::ir::ValueId,
    dim: usize,
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
) -> llvm_ir::Operand {
    let shape_desc = resolve_operand(shape, ctx);
    if let llvm_ir::Operand::Value(shape_desc_id) = shape_desc.clone()
        && let Some(cached) = ctx.shape_dim_cache.get(&(shape_desc_id, dim)).cloned()
    {
        return cached;
    }
    let ptr = emit_gep(
        ctx,
        out,
        shape_desc.clone(),
        vec![const_i64(dim as i64)],
        llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
    );
    let loaded = emit_load(ctx, out, ptr.clone(), llvm_ir::Type::I64);
    if let llvm_ir::Operand::Value(shape_desc_id) = shape_desc {
        ctx.shape_dim_cache
            .entry((shape_desc_id, dim))
            .or_insert(loaded.clone());
    }
    loaded
}
