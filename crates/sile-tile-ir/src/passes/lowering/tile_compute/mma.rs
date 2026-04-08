use sile_llvm_ir as llvm_ir;

use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, const_f32, const_i64, emit_bin, emit_cmp, emit_gep, emit_load,
    emit_shape_dim, emit_store, load_tile_scalar_dynamic, resolve_operand,
};

#[derive(Clone, Copy)]
pub(crate) struct FusedTileLoad {
    pub(crate) buf: ValueId,
    pub(crate) row_coord: ValueId,
    pub(crate) col_coord: ValueId,
    pub(crate) rows: i64,
    pub(crate) cols: i64,
    pub(crate) stride_shape_idx: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum FusedAccInit {
    ExistingTile,
    LoopCarriedSplat { value: f64, first_k_coord: ValueId },
}

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
    let (row_header, row_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (col_header, col_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (init_block, init_params) = builder.create_block(
        &format!("{prefix}_init"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (k_header, k_params) = builder.create_block(
        &format!("{prefix}_k_header"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_k", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
        ],
    );
    let (k_body, k_body_params) = builder.create_block(
        &format!("{prefix}_k_body"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_k", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
        ],
    );
    let (store_block, store_params) = builder.create_block(
        &format!("{prefix}_store"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
        ],
    );
    let (row_latch, row_latch_params) = builder.create_block(
        &format!("{prefix}_row_latch"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row = llvm_ir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            row.clone(),
            const_i64(tile_m),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(col_header);
    let col_row = llvm_ir::Operand::Value(col_params[0].id);
    let col = llvm_ir::Operand::Value(col_params[1].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            col.clone(),
            const_i64(tile_n),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: init_block,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(init_block);
    let init_row = llvm_ir::Operand::Value(init_params[0].id);
    let init_col = llvm_ir::Operand::Value(init_params[1].id);
    let init_term = builder.with_current_insts(|ctx, _, out| {
        let sum0 = load_tile_scalar_dynamic(
            ctx,
            out,
            acc_tile.clone(),
            init_row.clone(),
            init_col.clone(),
        );
        llvm_ir::Terminator::Br {
            target: k_header,
            args: vec![init_row.clone(), init_col.clone(), const_i64(0), sum0],
        }
    });
    builder.set_current_terminator(init_term);

    builder.switch_to(k_header);
    let k_row = llvm_ir::Operand::Value(k_params[0].id);
    let k_col = llvm_ir::Operand::Value(k_params[1].id);
    let k = llvm_ir::Operand::Value(k_params[2].id);
    let sum = llvm_ir::Operand::Value(k_params[3].id);
    let k_header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            k.clone(),
            const_i64(tile_k),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: k_body,
            true_args: vec![k_row.clone(), k_col.clone(), k.clone(), sum.clone()],
            false_target: store_block,
            false_args: vec![k_row.clone(), k_col.clone(), sum.clone()],
        }
    });
    builder.set_current_terminator(k_header_term);

    builder.switch_to(k_body);
    let body_row = llvm_ir::Operand::Value(k_body_params[0].id);
    let body_col = llvm_ir::Operand::Value(k_body_params[1].id);
    let body_k = llvm_ir::Operand::Value(k_body_params[2].id);
    let body_sum = llvm_ir::Operand::Value(k_body_params[3].id);
    let k_body_term = builder.with_current_insts(|ctx, _, out| {
        let lhs =
            load_tile_scalar_dynamic(ctx, out, a_tile.clone(), body_row.clone(), body_k.clone());
        let rhs =
            load_tile_scalar_dynamic(ctx, out, b_tile.clone(), body_k.clone(), body_col.clone());
        let product = emit_bin(ctx, out, llvm_ir::BinOp::Mul, lhs, rhs, llvm_ir::Type::F32);
        let next_sum = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_sum.clone(),
            product,
            llvm_ir::Type::F32,
        );
        let next_k = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_k.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: k_header,
            args: vec![body_row.clone(), body_col.clone(), next_k, next_sum],
        }
    });
    builder.set_current_terminator(k_body_term);

    builder.switch_to(store_block);
    let store_row = llvm_ir::Operand::Value(store_params[0].id);
    let store_col = llvm_ir::Operand::Value(store_params[1].id);
    let store_sum = llvm_ir::Operand::Value(store_params[2].id);
    let store_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![store_row.clone(), store_col.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, store_sum.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            store_col.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: col_header,
            args: vec![store_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(store_term);

    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}

pub(crate) fn lower_fused_tile_mma_inst(
    result: ValueId,
    a_load: FusedTileLoad,
    b_load: FusedTileLoad,
    acc: ValueId,
    acc_init: FusedAccInit,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let acc_tile = resolve_operand(acc, builder.ctx());
    builder.with_current_insts(|ctx, _, _| {
        ctx.operands.insert(result, acc_tile.clone());
    });

    let a_buf = resolve_operand(a_load.buf, builder.ctx());
    let b_buf = resolve_operand(b_load.buf, builder.ctx());
    let (a_origin, a_stride, b_origin, b_stride) = builder.with_current_insts(|ctx, _, out| {
        let a_origin = compute_rank2_tile_origin(
            ctx,
            out,
            a_buf.clone(),
            a_load.row_coord,
            a_load.col_coord,
            a_load.rows,
            a_load.cols,
            a_load.stride_shape_idx,
        );
        let a_stride = emit_shape_dim(ctx, out, a_buf.clone(), a_load.stride_shape_idx);
        let b_origin = compute_rank2_tile_origin(
            ctx,
            out,
            b_buf.clone(),
            b_load.row_coord,
            b_load.col_coord,
            b_load.rows,
            b_load.cols,
            b_load.stride_shape_idx,
        );
        let b_stride = emit_shape_dim(ctx, out, b_buf.clone(), b_load.stride_shape_idx);
        (a_origin, a_stride, b_origin, b_stride)
    });

    let prefix = format!("tile_mma_{}", result.0);
    let (row_header, row_params) = builder.create_block(
        &format!("{prefix}_row_header"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (col_header, col_params) = builder.create_block(
        &format!("{prefix}_col_header"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (init_block, init_params) = builder.create_block(
        &format!("{prefix}_init"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let zero_init_block = match acc_init {
        FusedAccInit::LoopCarriedSplat { .. } => Some(builder.create_block(
            &format!("{prefix}_init_zero"),
            vec![
                ("loop_row", llvm_ir::Type::I64),
                ("loop_col", llvm_ir::Type::I64),
                ("loop_a_index", llvm_ir::Type::I64),
                ("loop_b_index", llvm_ir::Type::I64),
            ],
        )),
        FusedAccInit::ExistingTile => None,
    };
    let zero_init_block_id = zero_init_block.as_ref().map(|(block, _)| *block);
    let load_init_block = match acc_init {
        FusedAccInit::LoopCarriedSplat { .. } => Some(builder.create_block(
            &format!("{prefix}_init_load"),
            vec![
                ("loop_row", llvm_ir::Type::I64),
                ("loop_col", llvm_ir::Type::I64),
                ("loop_a_index", llvm_ir::Type::I64),
                ("loop_b_index", llvm_ir::Type::I64),
            ],
        )),
        FusedAccInit::ExistingTile => None,
    };
    let load_init_block_id = load_init_block.as_ref().map(|(block, _)| *block);
    let (k_header, k_params) = builder.create_block(
        &format!("{prefix}_k_header"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_k", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
            ("loop_a_index", llvm_ir::Type::I64),
            ("loop_b_index", llvm_ir::Type::I64),
        ],
    );
    let (k_body, k_body_params) = builder.create_block(
        &format!("{prefix}_k_body"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_k", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
            ("loop_a_index", llvm_ir::Type::I64),
            ("loop_b_index", llvm_ir::Type::I64),
        ],
    );
    let (store_block, store_params) = builder.create_block(
        &format!("{prefix}_store"),
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
        ],
    );
    let (row_latch, row_latch_params) = builder.create_block(
        &format!("{prefix}_row_latch"),
        vec![("loop_row", llvm_ir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llvm_ir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row = llvm_ir::Operand::Value(row_params[0].id);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            row.clone(),
            const_i64(tile_m),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(col_header);
    let col_row = llvm_ir::Operand::Value(col_params[0].id);
    let col = llvm_ir::Operand::Value(col_params[1].id);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            col.clone(),
            const_i64(tile_n),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: init_block,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(init_block);
    let init_row = llvm_ir::Operand::Value(init_params[0].id);
    let init_col = llvm_ir::Operand::Value(init_params[1].id);
    let init_term = builder.with_current_insts(|ctx, _, out| match acc_init {
        FusedAccInit::ExistingTile => {
            let a_row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                init_row.clone(),
                a_stride.clone(),
                llvm_ir::Type::I64,
            );
            let a_index0 = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                a_origin.clone(),
                a_row_offset,
                llvm_ir::Type::I64,
            );
            let b_index0 = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                b_origin.clone(),
                init_col.clone(),
                llvm_ir::Type::I64,
            );
            let sum0 = load_tile_scalar_dynamic(
                ctx,
                out,
                acc_tile.clone(),
                init_row.clone(),
                init_col.clone(),
            );
            llvm_ir::Terminator::Br {
                target: k_header,
                args: vec![
                    init_row.clone(),
                    init_col.clone(),
                    const_i64(0),
                    sum0,
                    a_index0,
                    b_index0,
                ],
            }
        }
        FusedAccInit::LoopCarriedSplat { first_k_coord, .. } => {
            let a_row_offset = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Mul,
                init_row.clone(),
                a_stride.clone(),
                llvm_ir::Type::I64,
            );
            let a_index0 = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                a_origin.clone(),
                a_row_offset,
                llvm_ir::Type::I64,
            );
            let b_index0 = emit_bin(
                ctx,
                out,
                llvm_ir::BinOp::Add,
                b_origin.clone(),
                init_col.clone(),
                llvm_ir::Type::I64,
            );
            let is_first_iter = emit_cmp(
                ctx,
                out,
                llvm_ir::CmpPred::Eq,
                resolve_operand(first_k_coord, ctx),
                const_i64(0),
                llvm_ir::Type::I1,
            );
            llvm_ir::Terminator::CondBr {
                cond: is_first_iter,
                true_target: zero_init_block_id
                    .expect("loop-carried splat init must create zero-init block"),
                true_args: vec![
                    init_row.clone(),
                    init_col.clone(),
                    a_index0.clone(),
                    b_index0.clone(),
                ],
                false_target: load_init_block_id
                    .expect("loop-carried splat init must create load-init block"),
                false_args: vec![init_row.clone(), init_col.clone(), a_index0, b_index0],
            }
        }
    });
    builder.set_current_terminator(init_term);

    if let Some((zero_init_block, zero_init_params)) = zero_init_block {
        builder.switch_to(zero_init_block);
        let zero_row = llvm_ir::Operand::Value(zero_init_params[0].id);
        let zero_col = llvm_ir::Operand::Value(zero_init_params[1].id);
        let zero_a_index = llvm_ir::Operand::Value(zero_init_params[2].id);
        let zero_b_index = llvm_ir::Operand::Value(zero_init_params[3].id);
        let zero_term = builder.with_current_insts(|_, _, _| {
            let sum0 = match acc_init {
                FusedAccInit::LoopCarriedSplat { value, .. } => const_f32(value),
                FusedAccInit::ExistingTile => unreachable!("zero-init block only exists for splat"),
            };
            llvm_ir::Terminator::Br {
                target: k_header,
                args: vec![
                    zero_row.clone(),
                    zero_col.clone(),
                    const_i64(0),
                    sum0,
                    zero_a_index.clone(),
                    zero_b_index.clone(),
                ],
            }
        });
        builder.set_current_terminator(zero_term);
    }

    if let Some((load_init_block, load_init_params)) = load_init_block {
        builder.switch_to(load_init_block);
        let load_row = llvm_ir::Operand::Value(load_init_params[0].id);
        let load_col = llvm_ir::Operand::Value(load_init_params[1].id);
        let load_a_index = llvm_ir::Operand::Value(load_init_params[2].id);
        let load_b_index = llvm_ir::Operand::Value(load_init_params[3].id);
        let load_term = builder.with_current_insts(|ctx, _, out| {
            let sum0 = load_tile_scalar_dynamic(
                ctx,
                out,
                acc_tile.clone(),
                load_row.clone(),
                load_col.clone(),
            );
            llvm_ir::Terminator::Br {
                target: k_header,
                args: vec![
                    load_row.clone(),
                    load_col.clone(),
                    const_i64(0),
                    sum0,
                    load_a_index.clone(),
                    load_b_index.clone(),
                ],
            }
        });
        builder.set_current_terminator(load_term);
    }

    builder.switch_to(k_header);
    let k_row = llvm_ir::Operand::Value(k_params[0].id);
    let k_col = llvm_ir::Operand::Value(k_params[1].id);
    let k = llvm_ir::Operand::Value(k_params[2].id);
    let sum = llvm_ir::Operand::Value(k_params[3].id);
    let a_index = llvm_ir::Operand::Value(k_params[4].id);
    let b_index = llvm_ir::Operand::Value(k_params[5].id);
    let k_header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llvm_ir::CmpPred::Slt,
            k.clone(),
            const_i64(tile_k),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: k_body,
            true_args: vec![
                k_row.clone(),
                k_col.clone(),
                k.clone(),
                sum.clone(),
                a_index.clone(),
                b_index.clone(),
            ],
            false_target: store_block,
            false_args: vec![k_row.clone(), k_col.clone(), sum.clone()],
        }
    });
    builder.set_current_terminator(k_header_term);

    builder.switch_to(k_body);
    let body_row = llvm_ir::Operand::Value(k_body_params[0].id);
    let body_col = llvm_ir::Operand::Value(k_body_params[1].id);
    let body_k = llvm_ir::Operand::Value(k_body_params[2].id);
    let body_sum = llvm_ir::Operand::Value(k_body_params[3].id);
    let body_a_index = llvm_ir::Operand::Value(k_body_params[4].id);
    let body_b_index = llvm_ir::Operand::Value(k_body_params[5].id);
    let k_body_term = builder.with_current_insts(|ctx, _, out| {
        let a_ptr = emit_gep(
            ctx,
            out,
            a_buf.clone(),
            vec![body_a_index.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
        );
        let lhs = emit_load(ctx, out, a_ptr, llvm_ir::Type::F32);
        let b_ptr = emit_gep(
            ctx,
            out,
            b_buf.clone(),
            vec![body_b_index.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
        );
        let rhs = emit_load(ctx, out, b_ptr, llvm_ir::Type::F32);
        let product = emit_bin(ctx, out, llvm_ir::BinOp::Mul, lhs, rhs, llvm_ir::Type::F32);
        let next_sum = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_sum.clone(),
            product,
            llvm_ir::Type::F32,
        );
        let next_k = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_k.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        let next_a_index = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_a_index.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        let next_b_index = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            body_b_index.clone(),
            b_stride.clone(),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: k_header,
            args: vec![
                body_row.clone(),
                body_col.clone(),
                next_k,
                next_sum,
                next_a_index,
                next_b_index,
            ],
        }
    });
    builder.set_current_terminator(k_body_term);

    builder.switch_to(store_block);
    let store_row = llvm_ir::Operand::Value(store_params[0].id);
    let store_col = llvm_ir::Operand::Value(store_params[1].id);
    let store_sum = llvm_ir::Operand::Value(store_params[2].id);
    let store_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            acc_tile.clone(),
            vec![store_row.clone(), store_col.clone()],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        );
        emit_store(out, dst_ptr, store_sum.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            store_col.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: col_header,
            args: vec![store_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(store_term);

    builder.switch_to(row_latch);
    let latch_row = llvm_ir::Operand::Value(row_latch_params[0].id);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);
    builder.switch_to(continue_block);
}

fn compute_rank2_tile_origin(
    ctx: &mut crate::passes::lowering::core::LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
    buf: llvm_ir::Operand,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
) -> llvm_ir::Operand {
    let row_tile_base = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        resolve_operand(row_coord, ctx),
        const_i64(rows),
        llvm_ir::Type::I64,
    );
    let col_tile_base = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        resolve_operand(col_coord, ctx),
        const_i64(cols),
        llvm_ir::Type::I64,
    );
    let stride = emit_shape_dim(ctx, out, buf.clone(), stride_shape_idx);
    let row_origin = emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Mul,
        row_tile_base,
        stride,
        llvm_ir::Type::I64,
    );
    emit_bin(
        ctx,
        out,
        llvm_ir::BinOp::Add,
        row_origin,
        col_tile_base,
        llvm_ir::Type::I64,
    )
}
