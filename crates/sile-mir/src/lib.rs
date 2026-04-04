pub mod ir;
pub mod lower;
pub mod lower_llir;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_mir;
pub use lower_llir::{lower_mir_to_llir, lower_mir_to_llir_raw, lower_mir_to_llir_raw_with_plan};
pub use passes::dce;
