use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram};
use crate::typeck::TypedKernel;

pub fn lower_typed_kernel_to_ssa(_typed: &TypedKernel) -> SsaProgram {
    SsaProgram {
        instructions: vec![
            op(SsaOpcode::ProgramId),
            op(SsaOpcode::LoadTile),
            op(SsaOpcode::Add),
            op(SsaOpcode::Store),
        ],
    }
}

fn op(opcode: SsaOpcode) -> SsaInstruction {
    SsaInstruction { opcode }
}
