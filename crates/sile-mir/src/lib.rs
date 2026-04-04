pub mod ir;
pub mod lower;
pub mod lower_lir;
pub mod passes;
pub mod print;

pub use ir::*;
pub use lower::lower_to_mir;
pub use lower_lir::lower_mir_to_lir;
pub use passes::dce;
