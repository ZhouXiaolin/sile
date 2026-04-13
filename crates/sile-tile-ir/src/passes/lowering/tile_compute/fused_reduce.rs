use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::TileMapExpr;
use crate::ValueId;
use crate::passes::lowering::core::{
    BlockLowerer, alloc_tile_result, buffer_rank_of, const_i64, emit_gep,
    emit_load, emit_store, lower_nested_tile_loop, resolve_operand,
};
use crate::passes::lowering::tile_compute::reduce::reduce_combine;
use crate::passes::lowering::tile_expr::{collect_map_load_bases, eval_map_expr_scalar};
use crate::passes::lowering::tile_memory::{build_buffer_index_base, emit_buffer_element_ptr};

pub(crate) struct LoadReduceFusion {
    pub(crate) buf: ValueId,
    pub(crate) row_coord: ValueId,
    pub(crate) col_coord: ValueId,
    pub(crate) src_rows: i64,
    pub(crate) src_cols: i64,
    pub(crate) stride_shape_idx: usize,
    pub(crate) is_max: bool,
    pub(crate) axis: i64,
    pub(crate) in_rows: i64,
    pub(crate) in_cols: i64,
}

pub(crate) fn lower_fused_load_reduce_inst(
    result: ValueId,
    fusion: &LoadReduceFusion,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if fusion.axis == 1 {
        (fusion.in_rows, 1)
    } else {
        (1, fusion.in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);

    let buf_operand = resolve_operand(fusion.buf, builder.ctx());
    let row_operand = resolve_operand(fusion.row_coord, builder.ctx());
    let col_operand = resolve_operand(fusion.col_coord, builder.ctx());
    let rank = buffer_rank_of(fusion.buf, builder.tile_ir());
    let index_base = builder.with_current_insts(|ctx, _, out| {
        build_buffer_index_base(
            ctx,
            out,
            buf_operand.clone(),
            row_operand.clone(),
            col_operand.clone(),
            fusion.src_rows,
            fusion.src_cols,
            fusion.stride_shape_idx,
            rank,
        )
    });

    let reduce_extent = if fusion.axis == 1 {
        fusion.src_cols
    } else {
        fusion.src_rows
    };
    let prefix = format!("tile_fused_load_reduce_{}", result.0);

    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        out_rows,
        out_cols,
        move |ctx, _, out, row, col| {
            let first_r = if fusion.axis == 1 {
                row.clone()
            } else {
                const_i64(0)
            };
            let first_c = if fusion.axis == 1 {
                const_i64(0)
            } else {
                col.clone()
            };
            let first_ptr = emit_buffer_element_ptr(
                ctx,
                out,
                buf_operand.clone(),
                index_base.clone(),
                first_r,
                first_c,
            );
            let mut acc = emit_load(ctx, out, first_ptr, llvm_ir::Type::F32);

            for idx in 1..reduce_extent {
                let next_r = if fusion.axis == 1 {
                    row.clone()
                } else {
                    const_i64(idx)
                };
                let next_c = if fusion.axis == 1 {
                    const_i64(idx)
                } else {
                    col.clone()
                };
                let next_ptr = emit_buffer_element_ptr(
                    ctx,
                    out,
                    buf_operand.clone(),
                    index_base.clone(),
                    next_r,
                    next_c,
                );
                let next = emit_load(ctx, out, next_ptr, llvm_ir::Type::F32);
                acc = reduce_combine(ctx, out, acc, next, fusion.is_max);
            }

            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, acc);
        },
    );
}

pub(crate) struct MapReduceFusion {
    pub(crate) expr: TileMapExpr,
    pub(crate) src_rows: i64,
    pub(crate) src_cols: i64,
    pub(crate) is_max: bool,
    pub(crate) axis: i64,
    pub(crate) in_rows: i64,
    pub(crate) in_cols: i64,
}

pub(crate) fn lower_fused_map_reduce_inst(
    result: ValueId,
    fusion: &MapReduceFusion,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols) = if fusion.axis == 1 {
        (fusion.in_rows, 1)
    } else {
        (1, fusion.in_cols)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);

    let fused_load_bases = builder.with_current_insts(|ctx, tile_ir, out| {
        let mut bases = HashMap::new();
        let mut coord_bases = HashMap::new();
        collect_map_load_bases(&fusion.expr, ctx, tile_ir, out, &mut bases, &mut coord_bases);
        bases
    });

    let reduce_extent = if fusion.axis == 1 {
        fusion.src_cols
    } else {
        fusion.src_rows
    };
    let prefix = format!("tile_fused_map_reduce_{}", result.0);

    lower_nested_tile_loop(
        builder,
        prefix.as_str(),
        out_rows,
        out_cols,
        move |ctx, tile_ir, out, row, col| {
            let first_r = if fusion.axis == 1 {
                row.clone()
            } else {
                const_i64(0)
            };
            let first_c = if fusion.axis == 1 {
                const_i64(0)
            } else {
                col.clone()
            };
            let mut acc = eval_map_expr_scalar(
                &fusion.expr,
                first_r,
                first_c,
                ctx,
                tile_ir,
                &fused_load_bases,
                out,
            );

            for idx in 1..reduce_extent {
                let next_r = if fusion.axis == 1 {
                    row.clone()
                } else {
                    const_i64(idx)
                };
                let next_c = if fusion.axis == 1 {
                    const_i64(idx)
                } else {
                    col.clone()
                };
                let next = eval_map_expr_scalar(
                    &fusion.expr,
                    next_r,
                    next_c,
                    ctx,
                    tile_ir,
                    &fused_load_bases,
                    out,
                );
                acc = reduce_combine(ctx, out, acc, next, fusion.is_max);
            }

            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
            );
            emit_store(out, dst_ptr, acc);
        },
    );
}
