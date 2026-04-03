use std::collections::HashMap;

use crate::hir::ParamKind;
use crate::lir::builder::LirBuilder;
use crate::lir::ir::{CmpOp, Function, Param, Type, Value};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> Function {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);

    let entry = builder.append_block("entry");
    builder.switch_to_block(&entry);

    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let mut param_names: Vec<String> = Vec::new();
    for p in &typed.kernel.params {
        param_names.push(p.name.clone());
    }

    for (i, p) in typed.kernel.params.iter().enumerate() {
        value_map.insert(i, Value::Param(i));
        if p.kind == ParamKind::Output {
            let _ptr = builder.alloca(Type::ptr(Type::f32()));
            builder.store(Value::Param(i), Value::Param(i));
        }
    }

    let loop_info = analyze_ssa_loops(ssa);

    if loop_info.has_loops {
        generate_loop_nesting(&mut builder, ssa, &mut value_map, &param_names, &loop_info);
    } else {
        for inst in &ssa.instructions {
            lower_ssa_instruction(inst, &mut builder, &mut value_map, &param_names);
        }
    }

    builder.ret(None);
    builder.finish()
}

struct LoopInfo {
    has_loops: bool,
    #[allow(dead_code)]
    loop_bounds: Vec<(String, i64, i64)>,
}

fn analyze_ssa_loops(ssa: &SsaProgram) -> LoopInfo {
    let has_program_id = ssa
        .instructions
        .iter()
        .any(|i| i.opcode == SsaOpcode::ProgramId);
    LoopInfo {
        has_loops: has_program_id,
        loop_bounds: vec![],
    }
}

fn generate_loop_nesting(
    builder: &mut LirBuilder,
    ssa: &SsaProgram,
    value_map: &mut HashMap<usize, Value>,
    param_names: &[String],
    _loop_info: &LoopInfo,
) {
    let loop_var = builder.alloca(Type::i64());
    builder.store(loop_var, builder.const_int(0));

    let header = builder.append_block("loop_header");
    let body = builder.append_block("loop_body");
    let exit = builder.append_block("loop_exit");

    builder.br(&header);

    builder.switch_to_block(&header);
    let idx = builder.load(loop_var, Type::i64());
    let bound = builder.const_int(256);
    let cond = builder.icmp(CmpOp::Slt, idx, bound);
    builder.cond_br(cond, &body, &exit);

    builder.switch_to_block(&body);
    for inst in &ssa.instructions {
        if inst.opcode == SsaOpcode::ProgramId {
            let idx_val = builder.load(loop_var, Type::i64());
            let def_idx = get_def_index(&inst.def);
            value_map.insert(def_idx, idx_val);
        } else {
            lower_ssa_instruction(inst, builder, value_map, param_names);
        }
    }

    let next = builder.add(idx, builder.const_int(1));
    builder.store(loop_var, next);
    builder.br(&header);

    builder.switch_to_block(&exit);
}

fn lower_ssa_instruction(
    inst: &SsaInstruction,
    builder: &mut LirBuilder,
    value_map: &mut HashMap<usize, Value>,
    param_names: &[String],
) {
    let def_idx = get_def_index(&inst.def);

    let lir_inst = match inst.opcode {
        SsaOpcode::ProgramId => return,
        SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
            let base_param = if inst.uses.is_empty() {
                0
            } else {
                get_param_index(&inst.uses[0], value_map, param_names)
            };
            let ptr = Value::Param(base_param);
            let indices: Vec<Value> = inst.uses[1..]
                .iter()
                .map(|v| resolve_value(v, value_map))
                .collect();
            if indices.is_empty() {
                let alloca_tmp = builder.alloca(Type::i64());
                let load_tmp = builder.load(alloca_tmp, Type::i64());
                let gep = builder.gep(ptr, vec![load_tmp]);
                builder.load(gep, Type::f32())
            } else {
                let gep = builder.gep(ptr, indices);
                builder.load(gep, Type::f32())
            }
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
            let out_idx = param_names
                .iter()
                .position(|n| n == "c" || n == "y" || n == "out")
                .unwrap_or(2);
            let out_ptr = Value::Param(out_idx);
            let alloca_tmp = builder.alloca(Type::i64());
            let load_tmp = builder.load(alloca_tmp, Type::i64());
            let gep = builder.gep(out_ptr, vec![load_tmp]);
            builder.store(gep, val);
            return;
        }
        SsaOpcode::Mma => {
            let a = resolve_value(&inst.uses[0], value_map);
            let b = resolve_value(&inst.uses[1], value_map);
            let _c = resolve_value(
                &inst.uses.get(2).cloned().unwrap_or(SsaValue::Const(0)),
                value_map,
            );
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
    param_names: &[String],
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
