use sile_llir as llir;

use crate::lower_llir_core::{
    BlockLowerer, const_i64, emit_bin, emit_cmp, emit_gep, emit_load, emit_store,
};

pub(crate) fn lower_tile_load_rank2_loop(
    builder: &mut BlockLowerer<'_>,
    dst_tile: llir::Operand,
    buf_operand: llir::Operand,
    row_base: llir::Operand,
    col_base: llir::Operand,
    stride: llir::Operand,
    rows: i64,
    cols: i64,
) {
    let (row_header, row_params) =
        builder.create_block("tile_load_row_header", vec![("loop_row", llir::Type::I64)]);
    let (row_setup, row_setup_params) =
        builder.create_block("tile_load_row_setup", vec![("loop_row", llir::Type::I64)]);
    let (col_header, col_header_params) = builder.create_block(
        "tile_load_col_header",
        vec![
            ("loop_row", llir::Type::I64),
            ("row_offset", llir::Type::I64),
            ("loop_col", llir::Type::I64),
        ],
    );
    let (body_block, body_params) = builder.create_block(
        "tile_load_body",
        vec![
            ("loop_row", llir::Type::I64),
            ("row_offset", llir::Type::I64),
            ("loop_col", llir::Type::I64),
        ],
    );
    let (row_latch, row_latch_params) =
        builder.create_block("tile_load_row_latch", vec![("loop_row", llir::Type::I64)]);
    let (continue_block, _) = builder.create_block("tile_load_continue", vec![]);

    builder.set_current_terminator(llir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row = llir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            row.clone(),
            const_i64(rows),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: row_setup,
            true_args: vec![row.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(row_setup);
    let setup_row = llir::Operand::Value(row_setup_params[0].id);
    let row_setup_term = builder.with_current_insts(|ctx, _, out| {
        let src_row = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            row_base.clone(),
            setup_row.clone(),
            llir::Type::I64,
        );
        let row_offset = emit_bin(
            ctx,
            out,
            llir::BinOp::Mul,
            src_row,
            stride.clone(),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![setup_row.clone(), row_offset, const_i64(0)],
        }
    });
    builder.set_current_terminator(row_setup_term);

    builder.switch_to(col_header);
    let col_row = llir::Operand::Value(col_header_params[0].id);
    let row_offset = llir::Operand::Value(col_header_params[1].id);
    let col = llir::Operand::Value(col_header_params[2].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            col.clone(),
            const_i64(cols),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![col_row.clone(), row_offset.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(body_block);
    let body_row = llir::Operand::Value(body_params[0].id);
    let body_row_offset = llir::Operand::Value(body_params[1].id);
    let body_col = llir::Operand::Value(body_params[2].id);
    let body_term = builder.with_current_insts(|ctx, _, out| {
        let src_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            col_base.clone(),
            body_col.clone(),
            llir::Type::I64,
        );
        let linear_index = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_row_offset.clone(),
            src_col,
            llir::Type::I64,
        );
        let src_ptr = emit_gep(
            ctx,
            out,
            buf_operand.clone(),
            vec![linear_index],
            llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
        );
        let loaded = emit_load(ctx, out, src_ptr, llir::Type::F32);
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![body_row.clone(), body_col.clone()],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        emit_store(out, dst_ptr, loaded);
        let next_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_col.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![body_row.clone(), body_row_offset.clone(), next_col],
        }
    });
    builder.set_current_terminator(body_term);

    builder.switch_to(row_latch);
    let latch_row = llir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}

pub(crate) fn lower_tile_store_rank2_loop(
    builder: &mut BlockLowerer<'_>,
    buf_operand: llir::Operand,
    value_operand: llir::Operand,
    row_base: llir::Operand,
    col_base: llir::Operand,
    stride: llir::Operand,
    rows: i64,
    cols: i64,
) {
    let (row_header, row_params) =
        builder.create_block("tile_store_row_header", vec![("loop_row", llir::Type::I64)]);
    let (row_setup, row_setup_params) =
        builder.create_block("tile_store_row_setup", vec![("loop_row", llir::Type::I64)]);
    let (col_header, col_header_params) = builder.create_block(
        "tile_store_col_header",
        vec![
            ("loop_row", llir::Type::I64),
            ("row_offset", llir::Type::I64),
            ("loop_col", llir::Type::I64),
        ],
    );
    let (body_block, body_params) = builder.create_block(
        "tile_store_body",
        vec![
            ("loop_row", llir::Type::I64),
            ("row_offset", llir::Type::I64),
            ("loop_col", llir::Type::I64),
        ],
    );
    let (row_latch, row_latch_params) =
        builder.create_block("tile_store_row_latch", vec![("loop_row", llir::Type::I64)]);
    let (continue_block, _) = builder.create_block("tile_store_continue", vec![]);

    builder.set_current_terminator(llir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row = llir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            row.clone(),
            const_i64(rows),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: row_setup,
            true_args: vec![row.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(row_setup);
    let setup_row = llir::Operand::Value(row_setup_params[0].id);
    let row_setup_term = builder.with_current_insts(|ctx, _, out| {
        let dst_row = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            row_base.clone(),
            setup_row.clone(),
            llir::Type::I64,
        );
        let row_offset = emit_bin(
            ctx,
            out,
            llir::BinOp::Mul,
            dst_row,
            stride.clone(),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![setup_row.clone(), row_offset, const_i64(0)],
        }
    });
    builder.set_current_terminator(row_setup_term);

    builder.switch_to(col_header);
    let col_row = llir::Operand::Value(col_header_params[0].id);
    let row_offset = llir::Operand::Value(col_header_params[1].id);
    let col = llir::Operand::Value(col_header_params[2].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            col.clone(),
            const_i64(cols),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![col_row.clone(), row_offset.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(body_block);
    let body_row = llir::Operand::Value(body_params[0].id);
    let body_row_offset = llir::Operand::Value(body_params[1].id);
    let body_col = llir::Operand::Value(body_params[2].id);
    let body_term = builder.with_current_insts(|ctx, _, out| {
        let src_ptr = emit_gep(
            ctx,
            out,
            value_operand.clone(),
            vec![body_row.clone(), body_col.clone()],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        let scalar = emit_load(ctx, out, src_ptr, llir::Type::F32);
        let dst_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            col_base.clone(),
            body_col.clone(),
            llir::Type::I64,
        );
        let linear_index = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_row_offset.clone(),
            dst_col,
            llir::Type::I64,
        );
        let dst_ptr = emit_gep(
            ctx,
            out,
            buf_operand.clone(),
            vec![linear_index],
            llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
        );
        emit_store(out, dst_ptr, scalar);
        let next_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_col.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![body_row.clone(), body_row_offset.clone(), next_col],
        }
    });
    builder.set_current_terminator(body_term);

    builder.switch_to(row_latch);
    let latch_row = llir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}
