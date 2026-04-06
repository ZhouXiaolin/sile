use sile_llir as llir;

use crate::MirFunction;
use crate::passes::lowering::core::block::{BlockLowerer, LowerLlirCtx};

mod grid;
mod single_axis;

use grid::lower_full_2d_loop;
use single_axis::{lower_single_col_loop, lower_single_row_loop};

pub(crate) fn lower_nested_tile_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    rows: i64,
    cols: i64,
    body: impl FnMut(
        &mut LowerLlirCtx,
        &MirFunction,
        &mut Vec<llir::Inst>,
        llir::Operand,
        llir::Operand,
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
