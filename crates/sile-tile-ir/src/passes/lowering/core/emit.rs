mod alloc;
mod insts;
mod tile_loops;

pub(crate) use alloc::alloc_tile_result;
pub(crate) use insts::{
    const_f32, const_i64, emit_bin, emit_cmp, emit_gep, emit_intrinsic, emit_load, emit_select,
    emit_shape_dim, emit_store, load_tile_scalar_dynamic, lower_1d_tile_coord,
};
pub(crate) use tile_loops::{lower_nested_tile_loop, lower_reduce_extent_loop, lower_reduce_extent_loop_step, REDUCE_UNROLL_THRESHOLD};
