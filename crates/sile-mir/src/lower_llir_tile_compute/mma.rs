use sile_llir as llir;

use crate::ValueId;
use crate::lower_llir_core::{
    BlockLowerer, alloc_tile_result, const_i64, emit_bin, emit_cmp, emit_gep, emit_store,
    load_tile_scalar_dynamic, resolve_operand,
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

    lower_tile_mma_loop(
        builder,
        dst_tile,
        resolve_operand(a, builder.ctx()),
        resolve_operand(b, builder.ctx()),
        resolve_operand(acc, builder.ctx()),
        tile_m,
        tile_n,
        tile_k,
    );
}

fn lower_tile_mma_loop(
    builder: &mut BlockLowerer<'_>,
    dst_tile: llir::Operand,
    a_tile: llir::Operand,
    b_tile: llir::Operand,
    acc_tile: llir::Operand,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
) {
    let (row_header, row_params) =
        builder.create_block("mma_row_header", vec![("mma_row", llir::Type::I64)]);
    let (col_header, col_params) = builder.create_block(
        "mma_col_header",
        vec![("mma_row", llir::Type::I64), ("mma_col", llir::Type::I64)],
    );
    let (k_preheader, k_pre_params) = builder.create_block(
        "mma_k_preheader",
        vec![("mma_row", llir::Type::I64), ("mma_col", llir::Type::I64)],
    );
    let (k_header, k_header_params) = builder.create_block(
        "mma_k_header",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_k", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (k_body, k_body_params) = builder.create_block(
        "mma_k_body",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_k", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (k_exit, k_exit_params) = builder.create_block(
        "mma_k_exit",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (row_latch, row_latch_params) =
        builder.create_block("mma_row_latch", vec![("mma_row", llir::Type::I64)]);
    let (continue_block, _) = builder.create_block("mma_continue", vec![]);

    let row = llir::Operand::Value(row_params[0].id);
    let col = llir::Operand::Value(col_params[1].id);
    let col_row = llir::Operand::Value(col_params[0].id);
    let pre_row = llir::Operand::Value(k_pre_params[0].id);
    let pre_col = llir::Operand::Value(k_pre_params[1].id);
    let k_row = llir::Operand::Value(k_header_params[0].id);
    let k_col = llir::Operand::Value(k_header_params[1].id);
    let k_idx = llir::Operand::Value(k_header_params[2].id);
    let k_acc = llir::Operand::Value(k_header_params[3].id);
    let body_row = llir::Operand::Value(k_body_params[0].id);
    let body_col = llir::Operand::Value(k_body_params[1].id);
    let body_k = llir::Operand::Value(k_body_params[2].id);
    let body_acc = llir::Operand::Value(k_body_params[3].id);
    let exit_row = llir::Operand::Value(k_exit_params[0].id);
    let exit_col = llir::Operand::Value(k_exit_params[1].id);
    let exit_acc = llir::Operand::Value(k_exit_params[2].id);
    let latch_row = llir::Operand::Value(row_latch_params[0].id);

    builder.set_current_terminator(llir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let row_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            row.clone(),
            const_i64(tile_m),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: row_cond,
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(col_header);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let col_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            col.clone(),
            const_i64(tile_n),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: col_cond,
            true_target: k_preheader,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(k_preheader);
    let k_pre_term = builder.with_current_insts(|ctx, _, out| {
        let acc_init =
            load_tile_scalar_dynamic(ctx, out, acc_tile.clone(), pre_row.clone(), pre_col.clone());
        llir::Terminator::Br {
            target: k_header,
            args: vec![pre_row.clone(), pre_col.clone(), const_i64(0), acc_init],
        }
    });
    builder.set_current_terminator(k_pre_term);

    builder.switch_to(k_header);
    let k_header_term = builder.with_current_insts(|ctx, _, out| {
        let k_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            k_idx.clone(),
            const_i64(tile_k),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: k_cond,
            true_target: k_body,
            true_args: vec![k_row.clone(), k_col.clone(), k_idx.clone(), k_acc.clone()],
            false_target: k_exit,
            false_args: vec![k_row.clone(), k_col.clone(), k_acc.clone()],
        }
    });
    builder.set_current_terminator(k_header_term);

    builder.switch_to(k_body);
    let k_body_term = builder.with_current_insts(|ctx, _, out| {
        let a =
            load_tile_scalar_dynamic(ctx, out, a_tile.clone(), body_row.clone(), body_k.clone());
        let b =
            load_tile_scalar_dynamic(ctx, out, b_tile.clone(), body_k.clone(), body_col.clone());
        let product = emit_bin(ctx, out, llir::BinOp::Mul, a, b, llir::Type::F32);
        let next_acc = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_acc.clone(),
            product,
            llir::Type::F32,
        );
        let next_k = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_k.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: k_header,
            args: vec![body_row.clone(), body_col.clone(), next_k, next_acc],
        }
    });
    builder.set_current_terminator(k_body_term);

    builder.switch_to(k_exit);
    let k_exit_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![exit_row.clone(), exit_col.clone()],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        emit_store(out, dst_ptr, exit_acc.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            exit_col.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![exit_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(k_exit_term);

    builder.switch_to(row_latch);
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
