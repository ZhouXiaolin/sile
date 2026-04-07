use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, emit_bin, emit_gep, emit_store, load_tile_scalar_dynamic,
    lower_nested_tile_loop, resolve_operand,
};

pub(crate) fn lower_tile_mma_inst(
    result: ValueId,
    a: ValueId,
    b: ValueId,
    acc: ValueId,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, tile_m, tile_n);
    let a_tile = resolve_operand(a, builder.ctx());
    let b_tile = resolve_operand(b, builder.ctx());
    let acc_tile = resolve_operand(acc, builder.ctx());
    let prefix = format!("tile_mma_{}", result.0);

    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        tile_m,
        tile_n,
        move |ctx, _, out, row, col| {
            let mut sum =
                load_tile_scalar_dynamic(ctx, out, acc_tile.clone(), row.clone(), col.clone());

            for kk in 0..tile_k {
                let lhs = load_tile_scalar_dynamic(
                    ctx,
                    out,
                    a_tile.clone(),
                    row.clone(),
                    llvm_ir::Operand::Const(llvm_ir::Constant::Int(kk)),
                );
                let rhs = load_tile_scalar_dynamic(
                    ctx,
                    out,
                    b_tile.clone(),
                    llvm_ir::Operand::Const(llvm_ir::Constant::Int(kk)),
                    col.clone(),
                );
                let product = emit_bin(ctx, out, llvm_ir::BinOp::Mul, lhs, rhs, llvm_ir::Type::F32);
                sum = emit_bin(
                    ctx,
                    out,
                    llvm_ir::BinOp::Add,
                    sum,
                    product,
                    llvm_ir::Type::F32,
                );
            }

            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, sum);
        },
    );
}
