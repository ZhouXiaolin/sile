pub mod lower_hir;
pub mod lower_lir;
pub mod mir;
pub mod passes;

pub use lower_hir::lower_typed_kernel_to_ssa;
pub use mir::ir::{SsaOpcode, SsaProgram, SsaValue};

use sile_hir::typeck::TypedKernel;
use sile_lir::ExecutableKernel;

/// New pipeline: HIR → MIR → DCE → LIR
pub fn compile(typed: &TypedKernel) -> ExecutableKernel {
    let mir = sile_mir::lower::lower_to_mir(typed);
    let mir = sile_mir::passes::dce::run(mir);
    sile_mir::lower_lir::lower_mir_to_lir(&mir, typed)
}

/// Old pipeline (kept for comparison / fallback)
pub fn compile_legacy(typed: &TypedKernel) -> ExecutableKernel {
    let ssa = lower_hir::lower_typed_kernel_to_ssa(typed);
    let ssa = passes::canonicalize::run(ssa);
    let ssa = passes::dce::run(ssa);
    lower_lir::lower_ssa_to_lir(&ssa, typed)
}
