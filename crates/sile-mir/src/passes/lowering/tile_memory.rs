use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, const_f32, const_i64, emit_call_void, resolve_operand,
};

pub(crate) fn lower_tile_constant_inst(
    result: ValueId,
    value: f64,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    builder.with_current_insts(|_, _, out| {
        emit_call_void(
            out,
            "tile_fill_2d_f32",
            vec![dst_tile, const_f32(value), const_i64(rows), const_i64(cols)],
        );
    });
}

pub(crate) fn lower_tile_load_inst(
    result: ValueId,
    buf: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let buf_operand = resolve_operand(buf, builder.ctx());
    let row_operand = resolve_operand(row_coord, builder.ctx());
    let col_operand = resolve_operand(col_coord, builder.ctx());
    builder.with_current_insts(|_, _, out| {
        emit_call_void(
            out,
            "tile_load_2d_f32",
            vec![
                dst_tile,
                buf_operand,
                row_operand,
                col_operand,
                const_i64(rows),
                const_i64(cols),
                const_i64(stride_shape_idx as i64),
            ],
        );
    });
}

pub(crate) fn lower_tile_store_inst(
    buf: ValueId,
    value: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let buf_operand = resolve_operand(buf, builder.ctx());
    let value_operand = resolve_operand(value, builder.ctx());
    let row_operand = resolve_operand(row_coord, builder.ctx());
    let col_operand = resolve_operand(col_coord, builder.ctx());
    builder.with_current_insts(|_, _, out| {
        emit_call_void(
            out,
            "tile_store_2d_f32",
            vec![
                buf_operand,
                value_operand,
                row_operand,
                col_operand,
                const_i64(rows),
                const_i64(cols),
                const_i64(stride_shape_idx as i64),
            ],
        );
    });
}
