use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, LowerLlvmIrCtx, alloc_tile_result, const_i64, emit_bin, emit_cmp, emit_gep,
    emit_select, emit_store, load_tile_scalar_dynamic, lower_nested_tile_loop, resolve_operand,
};

pub(crate) fn reduce_combine(
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    acc: llvm_ir::Operand,
    next: llvm_ir::Operand,
    is_max: bool,
) -> llvm_ir::Operand {
    if is_max {
        let is_gt = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Ogt,
            next.clone(),
            acc.clone(),
            llvm_ir::Type::I1,
        );
        emit_select(ctx, out, is_gt, next, acc, llvm_ir::Type::F32)
    } else {
        emit_bin(ctx, out, llvm_ir::BinOp::Add, acc, next, llvm_ir::Type::F32)
    }
}

pub(crate) fn lower_tile_reduce_inst(
    result: ValueId,
    value: ValueId,
    is_max: bool,
    axis: i64,
    in_rows: i64,
    in_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if axis == 1 {
        (in_rows, 1)
    } else {
        (1, in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(value, builder.ctx());
    let reduce_extent = if axis == 1 { in_cols } else { in_rows };
    let prefix = format!("tile_reduce_{}", result.0);

    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        out_rows,
        out_cols,
        move |ctx, _, out, row, col| {
            let mut acc = load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                if axis == 1 { row.clone() } else { const_i64(0) },
                if axis == 1 { const_i64(0) } else { col.clone() },
            );

            for idx in 1..reduce_extent {
                let next = load_tile_scalar_dynamic(
                    ctx,
                    out,
                    src_tile.clone(),
                    if axis == 1 {
                        row.clone()
                    } else {
                        const_i64(idx)
                    },
                    if axis == 1 {
                        const_i64(idx)
                    } else {
                        col.clone()
                    },
                );
                acc = reduce_combine(ctx, out, acc, next, is_max);
            }

            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, acc);
        },
    );
}
