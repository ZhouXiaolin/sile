pub mod lower_hir;
pub mod lower_lir;
pub mod mir;
pub mod passes;

pub use mir::ir::{SsaOpcode, SsaProgram, SsaValue};
pub use lower_hir::lower_typed_kernel_to_ssa;

use sile_hir::typeck::TypedKernel;
use sile_lir::Function;

pub fn compile(typed: &TypedKernel) -> Function {
    let ssa = lower_hir::lower_typed_kernel_to_ssa(typed);
    let ssa = passes::canonicalize::run(ssa);
    let ssa = passes::dce::run(ssa);
    lower_lir::lower_ssa_to_lir(&ssa, typed)
}
