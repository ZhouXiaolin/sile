pub mod ir;
pub mod lower;
pub mod lower_llir;
mod lower_llir_block;
mod lower_llir_block_scalar;
mod lower_llir_block_terminator;
mod lower_llir_core;
mod lower_llir_tile_compute;
mod lower_llir_tile_deferred;
mod lower_llir_tile_expr;
mod lower_llir_tile_loops;
mod lower_llir_tile_memory;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_mir;
pub use lower_llir::{lower_mir_to_llir, lower_mir_to_llir_raw, lower_mir_to_llir_raw_with_plan};
pub use passes::dce;
