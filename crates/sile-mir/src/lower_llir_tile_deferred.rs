use crate::lower_llir_core::BlockLowerer;
use crate::lower_llir_tile_expr::lower_tile_expr_inst;
use crate::lower_llir_tile_memory::{lower_tile_constant_inst, lower_tile_load_inst};
use crate::{MirOp, ValueId};

pub(crate) fn materialize_deferred_tile(value: ValueId, builder: &mut BlockLowerer<'_>) {
    let Some(op) = builder.begin_materialize_tile(value) else {
        return;
    };

    match op {
        MirOp::TileConstant {
            value: constant,
            rows,
            cols,
        } => {
            lower_tile_constant_inst(value, constant, rows, cols, builder);
        }
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            lower_tile_load_inst(
                value,
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
                builder,
            );
        }
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let current_op = MirOp::TileBinary {
                op,
                lhs,
                rhs,
                rows,
                cols,
            };
            lower_tile_expr_inst(value, current_op, rows, cols, builder);
        }
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => {
            let current_op = MirOp::TileUnary {
                op,
                operand,
                rows,
                cols,
            };
            lower_tile_expr_inst(value, current_op, rows, cols, builder);
        }
        MirOp::TileBroadcast {
            value: src,
            rows,
            cols,
        } => {
            let current_op = MirOp::TileBroadcast {
                value: src,
                rows,
                cols,
            };
            lower_tile_expr_inst(value, current_op, rows, cols, builder);
        }
        _ => {}
    }
}

pub(crate) fn lower_planned_tile_op(
    result: ValueId,
    op: &MirOp,
    builder: &mut BlockLowerer<'_>,
) -> bool {
    match op {
        MirOp::TileConstant { value, rows, cols } => {
            if !builder.plan().should_defer(result) {
                lower_tile_constant_inst(result, *value, *rows, *cols, builder);
            }
            true
        }
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            if !builder.plan().should_defer(result) {
                lower_tile_load_inst(
                    result,
                    *buf,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    builder,
                );
            }
            true
        }
        MirOp::TileBinary { rows, cols, .. }
        | MirOp::TileUnary { rows, cols, .. }
        | MirOp::TileBroadcast { rows, cols, .. } => {
            if !builder.plan().should_defer(result) {
                lower_tile_expr_inst(result, op.clone(), *rows, *cols, builder);
            }
            true
        }
        _ => false,
    }
}
