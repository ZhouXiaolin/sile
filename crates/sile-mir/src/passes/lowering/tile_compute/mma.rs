use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, const_i64, emit_call_void, resolve_operand,
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

    builder.with_current_insts(|_, _, out| {
        emit_call_void(
            out,
            "tile_mma_accumulate_2d_f32",
            vec![
                dst_tile,
                a_tile,
                b_tile,
                acc_tile,
                const_i64(tile_m),
                const_i64(tile_n),
                const_i64(tile_k),
            ],
        );
    });
}
