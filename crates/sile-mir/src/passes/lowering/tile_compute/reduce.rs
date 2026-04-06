use sile_llir as llir;

use crate::passes::lowering::core::{
    BlockLowerer, LowerLlirCtx, alloc_tile_result, const_f32, const_i64, emit_bin, emit_cmp,
    emit_gep, emit_load, emit_select, emit_store, load_tile_scalar_dynamic, resolve_operand,
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
    let (out_rows, out_cols, reduce_extent) = if axis == 1 {
        (in_rows, 1, in_cols)
    } else {
        (1, in_cols, in_rows)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(value, builder.ctx());

    let (outer_header, outer_params) = builder.create_block(
        "tile_reduce_outer_header",
        vec![("reduce_outer", llir::Type::I64)],
    );
    let (outer_body, outer_body_params) = builder.create_block(
        "tile_reduce_outer_body",
        vec![("reduce_outer", llir::Type::I64)],
    );
    let (inner_header, inner_header_params) = builder.create_block(
        "tile_reduce_inner_header",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_idx", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (inner_body, inner_body_params) = builder.create_block(
        "tile_reduce_inner_body",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_idx", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (inner_exit, inner_exit_params) = builder.create_block(
        "tile_reduce_inner_exit",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (continue_block, _) = builder.create_block("tile_reduce_continue", vec![]);

    builder.set_current_terminator(llir::Terminator::Br {
        target: outer_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(outer_header);
    let outer_idx = llir::Operand::Value(outer_params[0].id);
    let outer_cond_limit = if axis == 1 { out_rows } else { out_cols };
    let outer_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            outer_idx.clone(),
            const_i64(outer_cond_limit),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: outer_body,
            true_args: vec![outer_idx.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(outer_term);

    builder.switch_to(outer_body);
    let outer_body_idx = llir::Operand::Value(outer_body_params[0].id);
    let outer_body_term = builder.with_current_insts(|ctx, _, out| {
        let acc_init = if matches!(op, ReduceOp::Sum) {
            const_f32(0.0)
        } else if axis == 1 {
            load_tile_scalar(
                ctx,
                out,
                src_tile.clone(),
                outer_body_idx.clone(),
                const_i64(0),
            )
        } else {
            load_tile_scalar(
                ctx,
                out,
                src_tile.clone(),
                const_i64(0),
                outer_body_idx.clone(),
            )
        };
        let start_idx = if matches!(op, ReduceOp::Max) { 1 } else { 0 };
        llir::Terminator::Br {
            target: inner_header,
            args: vec![outer_body_idx.clone(), const_i64(start_idx), acc_init],
        }
    });
    builder.set_current_terminator(outer_body_term);

    builder.switch_to(inner_header);
    let inner_outer = llir::Operand::Value(inner_header_params[0].id);
    let inner_idx = llir::Operand::Value(inner_header_params[1].id);
    let inner_acc = llir::Operand::Value(inner_header_params[2].id);
    let inner_header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            inner_idx.clone(),
            const_i64(reduce_extent),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: inner_body,
            true_args: vec![inner_outer.clone(), inner_idx.clone(), inner_acc.clone()],
            false_target: inner_exit,
            false_args: vec![inner_outer.clone(), inner_acc.clone()],
        }
    });
    builder.set_current_terminator(inner_header_term);

    builder.switch_to(inner_body);
    let body_outer = llir::Operand::Value(inner_body_params[0].id);
    let body_idx = llir::Operand::Value(inner_body_params[1].id);
    let body_acc = llir::Operand::Value(inner_body_params[2].id);
    let inner_body_term = builder.with_current_insts(|ctx, _, out| {
        let value = if axis == 1 {
            load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                body_outer.clone(),
                body_idx.clone(),
            )
        } else {
            load_tile_scalar_dynamic(
                ctx,
                out,
                src_tile.clone(),
                body_idx.clone(),
                body_outer.clone(),
            )
        };
        let next_acc = match op {
            ReduceOp::Sum => emit_bin(
                ctx,
                out,
                llir::BinOp::Add,
                body_acc.clone(),
                value,
                llir::Type::F32,
            ),
            ReduceOp::Max => emit_max(ctx, out, body_acc.clone(), value),
        };
        let next_idx = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_idx.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: inner_header,
            args: vec![body_outer.clone(), next_idx, next_acc],
        }
    });
    builder.set_current_terminator(inner_body_term);

    builder.switch_to(inner_exit);
    let exit_outer = llir::Operand::Value(inner_exit_params[0].id);
    let exit_acc = llir::Operand::Value(inner_exit_params[1].id);
    let inner_exit_term = builder.with_current_insts(|ctx, _, out| {
        let (dst_row, dst_col) = if axis == 1 {
            (exit_outer.clone(), const_i64(0))
        } else {
            (const_i64(0), exit_outer.clone())
        };
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![dst_row, dst_col],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        emit_store(out, dst_ptr, exit_acc.clone());
        let next_outer = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            exit_outer.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: outer_header,
            args: vec![next_outer],
        }
    });
    builder.set_current_terminator(inner_exit_term);
    builder.switch_to(continue_block);
}

fn load_tile_scalar(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    tile: llir::Operand,
    row: impl Into<llir::Operand>,
    col: impl Into<llir::Operand>,
) -> llir::Operand {
    let ptr = emit_gep(
        ctx,
        out,
        tile,
        vec![row.into(), col.into()],
        llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
    );
    emit_load(ctx, out, ptr, llir::Type::F32)
}

fn emit_max(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    lhs: llir::Operand,
    rhs: llir::Operand,
) -> llir::Operand {
    let cond = emit_cmp(
        ctx,
        out,
        llir::CmpPred::Ogt,
        rhs.clone(),
        lhs.clone(),
        llir::Type::I1,
    );
    emit_select(ctx, out, cond, rhs, lhs, llir::Type::F32)
}
