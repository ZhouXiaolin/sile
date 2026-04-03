use std::collections::HashMap;

use crate::hir::{BuiltinOp, Expr, Stmt};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

pub fn lower_typed_kernel_to_ssa(typed: &TypedKernel) -> SsaProgram {
    let mut locals: HashMap<String, SsaValue> = HashMap::new();
    let mut instructions = Vec::new();
    let mut next_local = 0usize;

    for stmt in &typed.kernel.body {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let value = lower_expr(expr, &mut instructions, &mut locals, &mut next_local);
                locals.insert(name.clone(), value);
            }
            Stmt::Store { target, value } => {
                let val = lower_expr(value, &mut instructions, &locals, &mut next_local);
                let def = SsaValue::Local(next_local);
                next_local += 1;
                instructions.push(SsaInstruction {
                    def,
                    opcode: SsaOpcode::Store,
                    uses: vec![val],
                    immediates: vec![],
                });
            }
        }
    }

    SsaProgram { instructions }
}

fn lower_expr(
    expr: &Expr,
    instructions: &mut Vec<SsaInstruction>,
    locals: &HashMap<String, SsaValue>,
    next_local: &mut usize,
) -> SsaValue {
    match expr {
        Expr::Var(name) => locals.get(name).cloned().unwrap_or(SsaValue::Const(0)),
        Expr::ScalarI32(v) => SsaValue::Const(*v as i64),
        Expr::Shape(_) => SsaValue::Const(0),
        Expr::Builtin { op, args } => {
            let uses: Vec<SsaValue> = args
                .iter()
                .map(|a| lower_expr(a, instructions, locals, next_local))
                .collect();
            let immediates: Vec<i64> = uses
                .iter()
                .filter_map(|v| {
                    if let SsaValue::Const(c) = v {
                        Some(*c)
                    } else {
                        None
                    }
                })
                .collect();

            let opcode = match op {
                BuiltinOp::ProgramId => SsaOpcode::ProgramId,
                BuiltinOp::LoadTile => SsaOpcode::LoadTile,
                BuiltinOp::LoadTileLike2D => SsaOpcode::LoadTileLike2D,
                BuiltinOp::Add => SsaOpcode::Add,
                BuiltinOp::Sub => SsaOpcode::Sub,
                BuiltinOp::Mul => SsaOpcode::Mul,
                BuiltinOp::Div => SsaOpcode::Div,
                BuiltinOp::Exp => SsaOpcode::Exp,
                BuiltinOp::ReduceMax => SsaOpcode::ReduceMax,
                BuiltinOp::ReduceSum => SsaOpcode::ReduceSum,
                BuiltinOp::Reshape => SsaOpcode::Reshape,
                BuiltinOp::Broadcast => SsaOpcode::Broadcast,
                BuiltinOp::Store => SsaOpcode::Store,
                BuiltinOp::ShapeOf => SsaOpcode::ShapeOf,
            };

            let def = SsaValue::Local(*next_local);
            *next_local += 1;
            instructions.push(SsaInstruction {
                def,
                opcode,
                uses,
                immediates,
            });
            def
        }
    }
}
