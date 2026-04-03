use crate::backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp, ReduceKind};
use crate::ssa::ir::{SsaOpcode, SsaProgram, SsaValue};

pub fn lower_ssa_to_backend_ir(ssa: &SsaProgram) -> BackendKernel {
    let has_reduce = ssa
        .instructions
        .iter()
        .any(|inst| matches!(inst.opcode, SsaOpcode::ReduceMax | SsaOpcode::ReduceSum));
    let has_mma = ssa
        .instructions
        .iter()
        .any(|inst| matches!(inst.opcode, SsaOpcode::Mma));
    let has_constant = ssa
        .instructions
        .iter()
        .any(|inst| matches!(inst.opcode, SsaOpcode::Constant));

    let op = if has_mma || has_constant {
        BackendOp::MatMul2D
    } else if has_reduce {
        BackendOp::Softmax2D
    } else {
        BackendOp::VecAdd1D
    };

    let tile_rank = match op {
        BackendOp::MatMul2D => 3,
        BackendOp::Softmax2D => 2,
        BackendOp::VecAdd1D => 1,
    };
    let tile_shape_symbols = match op {
        BackendOp::MatMul2D => vec!["BM".into(), "BN".into(), "BK".into()],
        BackendOp::Softmax2D => vec!["BM".into(), "BN".into()],
        BackendOp::VecAdd1D => vec!["S".into()],
    };

    let mut value_names: Vec<String> = Vec::new();
    let mut instructions = Vec::new();

    for inst in &ssa.instructions {
        let dest = format!("v{}", value_names.len());
        value_names.push(dest.clone());

        let backend_inst = match inst.opcode {
            SsaOpcode::ProgramId => BackendInstruction::Compute {
                dest: dest.clone(),
                op: "pid".into(),
                args: vec![],
            },
            SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                let indices: Vec<String> = inst
                    .uses
                    .iter()
                    .skip(1)
                    .map(|v| value_name(v, &value_names))
                    .collect();
                BackendInstruction::Load {
                    dest: dest.clone(),
                    src,
                    indices,
                }
            }
            SsaOpcode::Add | SsaOpcode::Sub | SsaOpcode::Mul | SsaOpcode::Div | SsaOpcode::Exp => {
                let op_name = match inst.opcode {
                    SsaOpcode::Add => "add",
                    SsaOpcode::Sub => "sub",
                    SsaOpcode::Mul => "mul",
                    SsaOpcode::Div => "div",
                    SsaOpcode::Exp => "exp",
                    _ => unreachable!(),
                };
                let args: Vec<String> = inst
                    .uses
                    .iter()
                    .map(|v| value_name(v, &value_names))
                    .collect();
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: op_name.into(),
                    args,
                }
            }
            SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
                let kind = match inst.opcode {
                    SsaOpcode::ReduceMax => ReduceKind::Max,
                    SsaOpcode::ReduceSum => ReduceKind::Sum,
                    _ => unreachable!(),
                };
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                let axis = inst.immediates.first().copied().unwrap_or(1);
                BackendInstruction::Reduce {
                    dest: dest.clone(),
                    src,
                    axis,
                    kind,
                }
            }
            SsaOpcode::Reshape | SsaOpcode::Broadcast | SsaOpcode::ShapeOf => {
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: inst.opcode_name().into(),
                    args: vec![src],
                }
            }
            SsaOpcode::Store => {
                let src = if inst.uses.is_empty() {
                    "result".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                BackendInstruction::Store { src }
            }
            SsaOpcode::Mma => {
                let a = value_name(&inst.uses[0], &value_names);
                let b = value_name(&inst.uses[1], &value_names);
                let c = value_name(&inst.uses[2], &value_names);
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: "mma".into(),
                    args: vec![a, b, c],
                }
            }
            SsaOpcode::Constant => {
                let value = inst.immediates.first().copied().unwrap_or(0);
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: "constant".into(),
                    args: vec![format!("{}", value)],
                }
            }
            SsaOpcode::ScalarDiv => {
                let args: Vec<String> = inst
                    .uses
                    .iter()
                    .map(|v| value_name(v, &value_names))
                    .collect();
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: "scalar_div".into(),
                    args,
                }
            }
            SsaOpcode::ShapeDim => {
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                let dim = inst.immediates.get(1).copied().unwrap_or(0);
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: "shape_dim".into(),
                    args: vec![src, format!("{}", dim)],
                }
            }
        };
        instructions.push(backend_inst);
    }

    BackendKernel {
        op,
        tile_rank,
        tile_shape_symbols,
        instructions,
    }
}

fn value_name(value: &SsaValue, names: &[String]) -> String {
    match value {
        SsaValue::Param(i) => format!("param{}", i),
        SsaValue::Local(i) => names.get(*i).cloned().unwrap_or_else(|| format!("v{}", i)),
        SsaValue::Const(c) => format!("{}", c),
    }
}
