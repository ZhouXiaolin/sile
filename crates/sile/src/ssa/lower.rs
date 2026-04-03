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
            Stmt::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                let start_val = lower_expr(start, &mut instructions, &mut locals, &mut next_local);
                let end_val = lower_expr(end, &mut instructions, &mut locals, &mut next_local);
                let start_i64 = if let SsaValue::Const(c) = start_val {
                    c
                } else {
                    0
                };
                let end_i64 = if let SsaValue::Const(c) = end_val {
                    c
                } else {
                    0
                };
                for i in start_i64..end_i64 {
                    locals.insert(var.clone(), SsaValue::Const(i));
                    for body_stmt in body {
                        match body_stmt {
                            Stmt::Let { name, expr, .. } => {
                                let value = lower_expr(
                                    expr,
                                    &mut instructions,
                                    &mut locals,
                                    &mut next_local,
                                );
                                locals.insert(name.clone(), value);
                            }
                            Stmt::Store { target, value } => {
                                let val =
                                    lower_expr(value, &mut instructions, &locals, &mut next_local);
                                let def = SsaValue::Local(next_local);
                                next_local += 1;
                                instructions.push(SsaInstruction {
                                    def,
                                    opcode: SsaOpcode::Store,
                                    uses: vec![val],
                                    immediates: vec![],
                                });
                            }
                            Stmt::ForLoop { .. } => {
                                // Nested loops not supported yet
                            }
                        }
                    }
                }
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
        Expr::ScalarF32(v) => SsaValue::Const((*v).to_bits() as i64),
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
                BuiltinOp::Mma => SsaOpcode::Mma,
                BuiltinOp::Constant => SsaOpcode::Constant,
                BuiltinOp::ScalarDiv => SsaOpcode::ScalarDiv,
                BuiltinOp::ShapeDim => SsaOpcode::ShapeDim,
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
