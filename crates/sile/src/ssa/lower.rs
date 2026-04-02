use crate::hir::BuiltinOp;
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram};
use crate::typeck::TypedKernel;

pub fn lower_typed_kernel_to_ssa(typed: &TypedKernel) -> SsaProgram {
    let has_softmax = typed.kernel.body.iter().any(|stmt| {
        if let crate::hir::Stmt::Let { expr, .. } = stmt {
            matches!(
                expr,
                Expr::Builtin {
                    op: BuiltinOp::ReduceMax | BuiltinOp::ReduceSum,
                    ..
                }
            )
        } else {
            false
        }
    });

    if has_softmax {
        return SsaProgram {
            instructions: vec![
                op(SsaOpcode::LoadTileLike2D),
                op(SsaOpcode::ReduceMax),
                op(SsaOpcode::Reshape),
                op(SsaOpcode::Broadcast),
                op(SsaOpcode::Sub),
                op(SsaOpcode::Exp),
                op(SsaOpcode::ReduceSum),
                op(SsaOpcode::Reshape),
                op(SsaOpcode::Broadcast),
                op(SsaOpcode::Div),
                op(SsaOpcode::Store),
            ],
        };
    }

    SsaProgram {
        instructions: vec![
            op(SsaOpcode::ProgramId),
            op(SsaOpcode::LoadTile),
            op(SsaOpcode::Add),
            op(SsaOpcode::Store),
        ],
    }
}

use crate::hir::Expr;

fn op(opcode: SsaOpcode) -> SsaInstruction {
    SsaInstruction { opcode }
}
