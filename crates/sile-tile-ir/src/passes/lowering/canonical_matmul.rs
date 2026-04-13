use sile_llvm_ir as llvm_ir;

use super::core::{
    BlockLowerer, LowerLlvmIrCtx, const_f32, const_i64, emit_bin, emit_cmp, emit_gep, emit_load,
    emit_shape_dim, emit_store, llvm_ir_block, resolve_operand,
};
use crate::{
    BinOp, CmpOp, TileIrBlock, TileIrFunction, TileIrInst, TileIrOp, TileIrParamKind,
    TileIrTerminator, ValueId,
};

#[derive(Clone, Copy, Debug)]
struct CanonicalMatmulPlan {
    a_buf: ValueId,
    b_buf: ValueId,
    c_buf: ValueId,
    row_launch: ValueId,
    col_launch: ValueId,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
}

pub(crate) fn lower_canonical_matmul(
    tile_ir: &TileIrFunction,
    ctx: &mut LowerLlvmIrCtx,
) -> Option<Vec<llvm_ir::BasicBlock>> {
    let plan = CanonicalMatmulPlan::detect(tile_ir)?;
    let entry = tile_ir.get_block(tile_ir.entry)?;
    let mut builder = BlockLowerer::new(
        tile_ir,
        ctx,
        llvm_ir_block(entry.id),
        format!("bb{}", entry.id.0),
        Vec::new(),
    );

    let (
        a_buf,
        b_buf,
        c_buf,
        a_stride,
        b_stride,
        c_stride,
        k_limit,
        a_tile_base,
        b_tile_base,
        c_tile_base,
    ) = builder.with_current_insts(|ctx, _, out| {
        let a_buf = resolve_operand(plan.a_buf, ctx);
        let b_buf = resolve_operand(plan.b_buf, ctx);
        let c_buf = resolve_operand(plan.c_buf, ctx);
        let row_launch = resolve_operand(plan.row_launch, ctx);
        let col_launch = resolve_operand(plan.col_launch, ctx);
        let a_stride = emit_shape_dim(ctx, out, a_buf.clone(), 1);
        let b_stride = emit_shape_dim(ctx, out, b_buf.clone(), 1);
        let c_stride = emit_shape_dim(ctx, out, c_buf.clone(), 1);
        let k_tiles = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Div,
            a_stride.clone(),
            const_i64(plan.tile_k),
            llvm_ir::Type::I64,
        );
        let k_limit = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            k_tiles,
            const_i64(plan.tile_k),
            llvm_ir::Type::I64,
        );
        let row_tile_origin = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_launch,
            const_i64(plan.tile_m),
            llvm_ir::Type::I64,
        );
        let col_tile_origin = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            col_launch,
            const_i64(plan.tile_n),
            llvm_ir::Type::I64,
        );
        let a_tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_tile_origin.clone(),
            a_stride.clone(),
            llvm_ir::Type::I64,
        );
        let c_row_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            row_tile_origin,
            c_stride.clone(),
            llvm_ir::Type::I64,
        );
        let c_tile_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            c_row_base,
            col_tile_origin.clone(),
            llvm_ir::Type::I64,
        );
        (
            a_buf,
            b_buf,
            c_buf,
            a_stride,
            b_stride,
            c_stride,
            k_limit,
            a_tile_base,
            col_tile_origin,
            c_tile_base,
        )
    });

    let (row_header, row_params) =
        builder.create_block("matmul_row_header", vec![("loop_row", llvm_ir::Type::I64)]);
    let (col_header, col_params) = builder.create_block(
        "matmul_col_header",
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (cell_init, cell_init_params) = builder.create_block(
        "matmul_cell_init",
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
        ],
    );
    let (k_header, k_params) = builder.create_block(
        "matmul_k_header",
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
        "matmul_k_body",
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
        "matmul_store",
        vec![
            ("loop_row", llvm_ir::Type::I64),
            ("loop_col", llvm_ir::Type::I64),
            ("loop_sum", llvm_ir::Type::F32),
        ],
    );
    let (row_latch, row_latch_params) =
        builder.create_block("matmul_row_latch", vec![("loop_row", llvm_ir::Type::I64)]);
    let (continue_block, _) = builder.create_block("matmul_continue", vec![]);

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
            const_i64(plan.tile_m),
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
            const_i64(plan.tile_n),
            llvm_ir::Type::I1,
        );
        llvm_ir::Terminator::CondBr {
            cond,
            true_target: cell_init,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(cell_init);
    let init_row = llvm_ir::Operand::Value(cell_init_params[0].id);
    let init_col = llvm_ir::Operand::Value(cell_init_params[1].id);
    let init_term = builder.with_current_insts(|ctx, _, out| {
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
            a_tile_base.clone(),
            a_row_offset,
            llvm_ir::Type::I64,
        );
        let b_index0 = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            b_tile_base.clone(),
            init_col.clone(),
            llvm_ir::Type::I64,
        );
        llvm_ir::Terminator::Br {
            target: k_header,
            args: vec![
                init_row.clone(),
                init_col.clone(),
                const_i64(0),
                const_f32(0.0),
                a_index0,
                b_index0,
            ],
        }
    });
    builder.set_current_terminator(init_term);

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
            k_limit.clone(),
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
        let c_row_offset = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Mul,
            store_row.clone(),
            c_stride.clone(),
            llvm_ir::Type::I64,
        );
        let c_row_base = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            c_tile_base.clone(),
            c_row_offset,
            llvm_ir::Type::I64,
        );
        let c_index = emit_bin(
            ctx,
            out,
            llvm_ir::BinOp::Add,
            c_row_base,
            store_col.clone(),
            llvm_ir::Type::I64,
        );
        let c_ptr = emit_gep(
            ctx,
            out,
            c_buf.clone(),
            vec![c_index],
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
        );
        emit_store(out, c_ptr, store_sum.clone());
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
    Some(builder.finish(llvm_ir::Terminator::Ret { value: None }))
}

impl CanonicalMatmulPlan {
    fn detect(tile_ir: &TileIrFunction) -> Option<Self> {
        let entry = tile_ir.get_block(tile_ir.entry)?;
        let TileIrTerminator::Jump {
            target: loop_header_id,
            args: entry_args,
        } = &entry.terminator
        else {
            return None;
        };
        if entry_args.len() != 2 {
            return None;
        }

        let k_seed = find_inst(entry, entry_args[0])?;
        let acc_seed = find_inst(entry, entry_args[1])?;
        if !matches!(&k_seed.op, TileIrOp::ConstI64(0)) {
            return None;
        }
        let (tile_m, tile_n) = match &acc_seed.op {
            TileIrOp::Splat { value, rows, cols } if *value == 0.0 => (*rows, *cols),
            TileIrOp::Splat { .. } => return None,
            _ => return None,
        };

        let loop_header = tile_ir.get_block(*loop_header_id)?;
        if loop_header.params.len() != 2 {
            return None;
        }
        let loop_k = loop_header.params[0];
        let loop_acc = loop_header.params[1];
        let TileIrTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } = &loop_header.terminator
        else {
            return None;
        };
        if true_args.as_slice() != [loop_k, loop_acc] || false_args.as_slice() != [loop_acc] {
            return None;
        }

        let cmp_inst = find_inst(loop_header, *cond)?;
        let (k_tiles_value, cmp_k) = match &cmp_inst.op {
            TileIrOp::ICmp {
                op: CmpOp::Lt,
                lhs,
                rhs,
            } => (*rhs, *lhs),
            _ => return None,
        };
        if cmp_k != loop_k {
            return None;
        }
        let k_tiles_inst = find_inst(loop_header, k_tiles_value)?;
        let (shape_dim_value, tile_k_value) = match &k_tiles_inst.op {
            TileIrOp::IBinary {
                op: BinOp::Div,
                lhs,
                rhs,
            } => (*lhs, *rhs),
            _ => return None,
        };
        let tile_k_inst = find_inst(loop_header, tile_k_value)?;
        let tile_k = match &tile_k_inst.op {
            TileIrOp::ConstI64(value) => *value,
            _ => return None,
        };
        let shape_dim_inst = find_inst(loop_header, shape_dim_value)?;
        let a_buf = match &shape_dim_inst.op {
            TileIrOp::ShapeDim { shape, dim: 1 } => shape_desc_source(tile_ir, *shape)?,
            _ => return None,
        };

        let body = tile_ir.get_block(*true_target)?;
        if body.params.len() != 2 {
            return None;
        }
        let body_k = body.params[0];
        let body_acc = body.params[1];
        if true_args.as_slice() != [loop_k, loop_acc] {
            return None;
        }
        let TileIrTerminator::Jump {
            target: back_target,
            args: back_args,
        } = &body.terminator
        else {
            return None;
        };
        if *back_target != loop_header.id || back_args.len() != 2 {
            return None;
        }

        let next_k_inst = find_inst(body, back_args[0])?;
        let (add_lhs, add_rhs) = match &next_k_inst.op {
            TileIrOp::IBinary {
                op: BinOp::Add,
                lhs,
                rhs,
            } => (*lhs, *rhs),
            _ => return None,
        };
        let one_value = if add_lhs == body_k {
            add_rhs
        } else if add_rhs == body_k {
            add_lhs
        } else {
            return None;
        };
        if !matches!(&find_inst(body, one_value)?.op, TileIrOp::ConstI64(1)) {
            return None;
        }

        let mma_inst = find_inst(body, back_args[1])?;
        let (a_tile, b_tile, mma_acc, mma_m, mma_n, mma_k) = match &mma_inst.op {
            TileIrOp::MmaF {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                tile_k,
            } => (*a, *b, *acc, *tile_m, *tile_n, *tile_k),
            _ => return None,
        };
        if mma_acc != body_acc || mma_m != tile_m || mma_n != tile_n || mma_k != tile_k {
            return None;
        }

        let a_load_inst = find_inst(body, a_tile)?;
        let (a_buf_loaded, row_launch, a_k, a_rows, a_cols) = match &a_load_inst.op {
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx: 1,
            } => (*buf, *row_coord, *col_coord, *rows, *cols),
            _ => return None,
        };
        let b_load_inst = find_inst(body, b_tile)?;
        let (b_buf, b_k, col_launch, b_rows, b_cols) = match &b_load_inst.op {
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx: 1,
            } => (*buf, *row_coord, *col_coord, *rows, *cols),
            _ => return None,
        };
        if a_buf_loaded != a_buf
            || a_k != body_k
            || b_k != body_k
            || a_rows != tile_m
            || a_cols != tile_k
            || b_rows != tile_k
            || b_cols != tile_n
        {
            return None;
        }
        if !is_launch_param(tile_ir, row_launch) || !is_launch_param(tile_ir, col_launch) {
            return None;
        }

        let exit = tile_ir.get_block(*false_target)?;
        if exit.params.len() != 1 || false_args.as_slice() != [loop_acc] {
            return None;
        }
        let exit_acc = exit.params[0];
        let [store_inst] = exit.insts.as_slice() else {
            return None;
        };
        let (c_buf, store_value, store_row, store_col, store_rows, store_cols) =
            match &store_inst.op {
                TileIrOp::StorePtrTko {
                    buf,
                    value,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx: 1,
                } => (*buf, *value, *row_coord, *col_coord, *rows, *cols),
                _ => return None,
            };
        if store_value != exit_acc
            || store_row != row_launch
            || store_col != col_launch
            || store_rows != tile_m
            || store_cols != tile_n
            || !matches!(&exit.terminator, TileIrTerminator::Return)
        {
            return None;
        }

        if !is_buffer_param(tile_ir, a_buf)
            || !is_buffer_param(tile_ir, b_buf)
            || !is_buffer_param(tile_ir, c_buf)
        {
            return None;
        }

        Some(Self {
            a_buf,
            b_buf,
            c_buf,
            row_launch,
            col_launch,
            tile_m,
            tile_n,
            tile_k,
        })
    }
}

fn find_inst(block: &TileIrBlock, value: ValueId) -> Option<&TileIrInst> {
    block.insts.iter().find(|inst| inst.result == value)
}

fn shape_desc_source(tile_ir: &TileIrFunction, value: ValueId) -> Option<ValueId> {
    let param = tile_ir.params.iter().find(|param| param.value == value)?;
    match &param.kind {
        TileIrParamKind::ShapeDesc { source } => Some(*source),
        TileIrParamKind::Buffer | TileIrParamKind::LaunchIndex { .. } => None,
    }
}

fn is_launch_param(tile_ir: &TileIrFunction, value: ValueId) -> bool {
    tile_ir.params.iter().any(|param| {
        param.value == value && matches!(&param.kind, TileIrParamKind::LaunchIndex { .. })
    })
}

fn is_buffer_param(tile_ir: &TileIrFunction, value: ValueId) -> bool {
    tile_ir
        .params
        .iter()
        .any(|param| param.value == value && matches!(&param.kind, TileIrParamKind::Buffer))
}
