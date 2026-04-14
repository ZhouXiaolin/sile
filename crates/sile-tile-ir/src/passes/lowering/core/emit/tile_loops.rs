use sile_llvm_ir as llvm_ir;

use crate::TileIrFunction;
use crate::passes::lowering::core::block::{BlockLowerer, LowerLlvmIrCtx};

mod grid;
mod single_axis;

use grid::lower_full_2d_loop;
use single_axis::{lower_single_col_loop, lower_single_row_loop};

pub(crate) use single_axis::{
    REDUCE_UNROLL_THRESHOLD, lower_reduce_extent_loop, lower_reduce_extent_loop_step,
};

pub(crate) fn lower_nested_tile_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    body: impl FnMut(
        &mut LowerLlvmIrCtx,
        &TileIrFunction,
        &mut Vec<llvm_ir::Inst>,
        llvm_ir::Operand,
        llvm_ir::Operand,
    ),
) {
    if rows == 1 {
        lower_single_col_loop(builder, prefix, cols, body);
        return;
    }

    if cols == 1 {
        lower_single_row_loop(builder, prefix, rows, body);
        return;
    }

    lower_full_2d_loop(builder, prefix, rows, cols, body);
}
