use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, const_i64, emit_call_void, resolve_operand,
};
use crate::{ReduceOp, ValueId};

pub(crate) fn lower_tile_reduce_inst(
    result: ValueId,
    value: ValueId,
    op: ReduceOp,
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
    let helper = reduce_helper_name(op, axis);

    builder.with_current_insts(|_, _, out| {
        emit_call_void(
            out,
            helper,
            vec![dst_tile, src_tile, const_i64(in_rows), const_i64(in_cols)],
        );
    });
}

fn reduce_helper_name(op: ReduceOp, axis: i64) -> &'static str {
    match (op, axis) {
        (ReduceOp::Sum, 0) => "tile_reduce_sum_axis0_2d_f32",
        (ReduceOp::Sum, _) => "tile_reduce_sum_axis1_2d_f32",
        (ReduceOp::Max, 0) => "tile_reduce_max_axis0_2d_f32",
        (ReduceOp::Max, _) => "tile_reduce_max_axis1_2d_f32",
    }
}
