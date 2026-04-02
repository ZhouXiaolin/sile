use crate::backend_ir::ir::{BackendKernel, BackendOp};
use crate::ssa::ir::SsaProgram;

pub fn lower_ssa_to_backend_ir(ssa: &SsaProgram) -> BackendKernel {
    let op = if ssa
        .instructions
        .iter()
        .any(|inst| inst.opcode_name() == "reduce_max")
    {
        BackendOp::Softmax2D
    } else {
        BackendOp::VecAdd1D
    };
    BackendKernel {
        op,
        tile_rank: if matches!(op, BackendOp::Softmax2D) {
            2
        } else {
            1
        },
        tile_shape_symbols: if matches!(op, BackendOp::Softmax2D) {
            vec!["BM".into(), "BN".into()]
        } else {
            vec!["S".into()]
        },
    }
}
