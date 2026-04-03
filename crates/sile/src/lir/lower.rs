use std::collections::HashMap;

use crate::lir::builder::LirBuilder;
use crate::lir::ir::{Function, Param, Type, Value};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

/// Lower SSA to LIR producing a flat body of per-element operations.
/// No explicit loop control flow is generated — the loop structure comes
/// from the C codegen's OpenMP wrapper.
pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> Function {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);

    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let param_names: Vec<String> = typed.kernel.params.iter().map(|p| p.name.clone()).collect();

    // Map SSA params to LIR params
    for (i, _) in typed.kernel.params.iter().enumerate() {
        value_map.insert(i, Value::Param(i));
    }

    // Single flat body block
    let body = builder.append_block("body");
    builder.switch_to_block(&body);

    for inst in &ssa.instructions {
        lower_ssa_instruction(inst, &mut builder, &mut value_map, &param_names);
    }

    builder.ret(None);
    builder.finish()
}

fn lower_ssa_instruction(
    inst: &SsaInstruction,
    builder: &mut LirBuilder,
    value_map: &mut HashMap<usize, Value>,
    param_names: &[String],
) {
    let def_idx = get_def_index(&inst.def);

    let lir_inst = match inst.opcode {
        // ProgramId is handled by the C codegen's OpenMP loop (variable `i`).
        // Don't emit any LIR instruction.
        SsaOpcode::ProgramId => return,

        SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
            let base_param = if inst.uses.is_empty() {
                0
            } else {
                get_param_index(&inst.uses[0], value_map, param_names)
            };
            // Direct element load: buffer[i]
            builder.load(Value::Param(base_param), Type::f32())
        }

        SsaOpcode::Add => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.add(lhs, rhs)
        }
        SsaOpcode::Sub => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.sub(lhs, rhs)
        }
        SsaOpcode::Mul => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.mul(lhs, rhs)
        }
        SsaOpcode::Div => {
            let lhs = resolve_value(&inst.uses[0], value_map);
            let rhs = resolve_value(&inst.uses[1], value_map);
            builder.push_instruction(crate::lir::ir::Instruction::Div(lhs, rhs))
        }
        SsaOpcode::Exp => {
            let val = resolve_value(&inst.uses[0], value_map);
            builder.exp(val)
        }
        SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
            // Reductions need special handling in a later pass.
            // For now, passthrough the source operand.
            let src = if inst.uses.is_empty() {
                Value::Param(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            };
            src
        }
        SsaOpcode::Store => {
            let val = if inst.uses.is_empty() {
                Value::Param(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            };
            let out_idx = find_output_param(param_names);
            builder.store(Value::Param(out_idx), val);
            return; // store doesn't produce a value
        }
        SsaOpcode::Mma => {
            let a = resolve_value(&inst.uses[0], value_map);
            let b = resolve_value(&inst.uses[1], value_map);
            builder.mul(a, b)
        }
        SsaOpcode::Constant => {
            let val = inst.immediates.first().copied().unwrap_or(0);
            builder.const_float(val as f64)
        }
        SsaOpcode::Reshape
        | SsaOpcode::Broadcast
        | SsaOpcode::ShapeOf
        | SsaOpcode::ScalarDiv
        | SsaOpcode::ShapeDim => {
            // Passthrough first operand
            if inst.uses.is_empty() {
                builder.const_int(0)
            } else {
                resolve_value(&inst.uses[0], value_map)
            }
        }
    };

    value_map.insert(def_idx, lir_inst);
}

fn resolve_value(v: &SsaValue, value_map: &HashMap<usize, Value>) -> Value {
    match v {
        SsaValue::Param(i) => Value::Param(*i),
        SsaValue::Local(i) => value_map
            .get(i)
            .cloned()
            .unwrap_or(Value::Const(crate::lir::ir::Constant::Int(0))),
        SsaValue::Const(c) => Value::Const(crate::lir::ir::Constant::Int(*c)),
    }
}

fn get_def_index(def: &SsaValue) -> usize {
    match def {
        SsaValue::Local(i) => *i,
        _ => 0,
    }
}

fn get_param_index(
    v: &SsaValue,
    value_map: &HashMap<usize, Value>,
    _param_names: &[String],
) -> usize {
    match v {
        SsaValue::Param(i) => *i,
        SsaValue::Local(i) => value_map
            .get(i)
            .and_then(|val| match val {
                Value::Param(p) => Some(*p),
                _ => None,
            })
            .unwrap_or(0),
        _ => 0,
    }
}

/// Find the index of the output parameter by checking param kinds.
fn find_output_param(param_names: &[String]) -> usize {
    // First try to find by convention name
    param_names
        .iter()
        .position(|n| n == "c" || n == "y" || n == "out")
        .unwrap_or(2)
}

fn lower_kernel_params(typed: &TypedKernel) -> Vec<Param> {
    typed
        .kernel
        .params
        .iter()
        .map(|p| Param {
            name: p.name.clone(),
            ty: Type::ptr(Type::f32()),
        })
        .collect()
}
