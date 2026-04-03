use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_lir::builder::LirBuilder;
use sile_lir::ir::{Constant, Function, Instruction, Param, Type, Value};

use crate::mir::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};

pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> Function {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);

    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let mut opcode_map: HashMap<usize, SsaOpcode> = HashMap::new();
    let mut tile_shapes: HashMap<usize, Vec<i64>> = HashMap::new();

    for (i, _) in typed.kernel.params.iter().enumerate() {
        value_map.insert(i, Value::Param(i));
    }

    let body = builder.append_block("body");
    builder.switch_to_block(&body);

    for inst in &ssa.instructions {
        lower_ssa_instruction(
            inst,
            &mut builder,
            &mut value_map,
            &mut opcode_map,
            &mut tile_shapes,
        );
    }

    builder.ret(None);
    builder.finish()
}

fn lower_ssa_instruction(
    inst: &SsaInstruction,
    builder: &mut LirBuilder,
    value_map: &mut HashMap<usize, Value>,
    opcode_map: &mut HashMap<usize, SsaOpcode>,
    tile_shapes: &mut HashMap<usize, Vec<i64>>,
) {
    let def_idx = get_def_index(&inst.def);
    let lowered = match inst.opcode {
        SsaOpcode::ProgramId => Some(builder.const_int(0)),
        SsaOpcode::ShapeDim => {
            let dim = inst.immediates.first().copied().unwrap_or(0);
            if inst.uses.first().and_then(|value| match value {
                SsaValue::Local(idx) => opcode_map.get(idx),
                _ => None,
            }) == Some(&SsaOpcode::ProgramId)
            {
                Some(builder.get_tile_coord(dim))
            } else {
                Some(Value::ShapeDim(dim as usize))
            }
        }
        SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
            let rank = inst.immediates.first().copied().unwrap_or(1);
            if rank == 2 && inst.uses.len() >= 3 {
                let rows = inst.immediates.get(1).copied().unwrap_or(1);
                let cols = inst.immediates.get(2).copied().unwrap_or(1);
                let stride_shape_idx = inst.immediates.get(3).copied().unwrap_or(1) as usize;
                let buf = resolve_value(&inst.uses[0], value_map);
                let row_tile = resolve_value(&inst.uses[1], value_map);
                let col_tile = resolve_value(&inst.uses[2], value_map);
                Some(builder.tile_load_2d(buf, rows, cols, row_tile, col_tile, stride_shape_idx))
            } else {
                let ptr = resolve_value(&inst.uses[0], value_map);
                Some(builder.load(ptr, Type::f32()))
            }
        }
        SsaOpcode::Add => Some(builder.add(
            resolve_value(&inst.uses[0], value_map),
            resolve_value(&inst.uses[1], value_map),
        )),
        SsaOpcode::Sub => Some(builder.sub(
            resolve_value(&inst.uses[0], value_map),
            resolve_value(&inst.uses[1], value_map),
        )),
        SsaOpcode::Mul => Some(builder.mul(
            resolve_value(&inst.uses[0], value_map),
            resolve_value(&inst.uses[1], value_map),
        )),
        SsaOpcode::Div => Some(builder.push_instruction(Instruction::Div(
            resolve_value(&inst.uses[0], value_map),
            resolve_value(&inst.uses[1], value_map),
        ))),
        SsaOpcode::Exp => Some(builder.exp(resolve_value(&inst.uses[0], value_map))),
        SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
            let source = inst
                .uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .unwrap_or_else(|| builder.const_int(0));
            let source_shape = inst
                .uses
                .first()
                .and_then(|value| get_value_shape(value, tile_shapes));
            if let Some(shape) = source_shape {
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);
                let axis = inst.immediates.first().copied().unwrap_or(0);
                let value = match inst.opcode {
                    SsaOpcode::ReduceMax => builder.tile_reduce_max(source, axis, rows, cols),
                    SsaOpcode::ReduceSum => builder.tile_reduce_sum(source, axis, rows, cols),
                    _ => unreachable!(),
                };
                tile_shapes.insert(def_idx, vec![rows, 1]);
                Some(value)
            } else {
                Some(source)
            }
        }
        SsaOpcode::Store => {
            let output = resolve_value(&inst.uses[0], value_map);
            let value = resolve_value(&inst.uses[1], value_map);
            let rank = inst.immediates.first().copied().unwrap_or(1);
            if rank == 2 && inst.uses.len() >= 4 {
                let rows = inst.immediates.get(1).copied().unwrap_or(1);
                let cols = inst.immediates.get(2).copied().unwrap_or(1);
                let stride_shape_idx = inst.immediates.get(3).copied().unwrap_or(1) as usize;
                let row_tile = resolve_value(&inst.uses[2], value_map);
                let col_tile = resolve_value(&inst.uses[3], value_map);
                builder.tile_store_2d(
                    output,
                    value,
                    rows,
                    cols,
                    row_tile,
                    col_tile,
                    stride_shape_idx,
                );
            } else {
                builder.store(output, value);
            }
            None
        }
        SsaOpcode::Mma => {
            if inst.immediates.len() >= 3 {
                Some(builder.tile_mma(
                    resolve_value(&inst.uses[0], value_map),
                    resolve_value(&inst.uses[1], value_map),
                    resolve_value(&inst.uses[2], value_map),
                    inst.immediates[0],
                    inst.immediates[1],
                    inst.immediates[2],
                ))
            } else {
                Some(builder.mul(
                    resolve_value(&inst.uses[0], value_map),
                    resolve_value(&inst.uses[1], value_map),
                ))
            }
        }
        SsaOpcode::Constant => {
            if inst.immediates.len() >= 3 {
                let init = f32::from_bits(inst.immediates[0] as u32) as f64;
                Some(builder.tile_alloc(inst.immediates[1], inst.immediates[2], init))
            } else {
                let value = inst.immediates.first().copied().unwrap_or(0);
                Some(builder.const_float(f32::from_bits(value as u32) as f64))
            }
        }
        SsaOpcode::Reshape => {
            let value = inst
                .uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .unwrap_or_else(|| builder.const_int(0));
            if !inst.immediates.is_empty() {
                tile_shapes.insert(def_idx, normalize_shape(&inst.immediates));
            }
            Some(value)
        }
        SsaOpcode::Broadcast => {
            let value = inst
                .uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .unwrap_or_else(|| builder.const_int(0));
            if inst.immediates.len() >= 2 {
                let rows = inst.immediates[0];
                let cols = inst.immediates[1];
                tile_shapes.insert(def_idx, vec![rows, cols]);
                Some(builder.tile_broadcast(value, rows, cols))
            } else {
                Some(value)
            }
        }
        SsaOpcode::ShapeOf | SsaOpcode::ScalarDiv => inst
            .uses
            .first()
            .map(|value| resolve_value(value, value_map))
            .or_else(|| Some(builder.const_int(0))),
    };

    if let Some(value) = lowered {
        value_map.insert(def_idx, value);
    }
    match inst.opcode {
        SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D | SsaOpcode::Constant => {
            if inst.immediates.len() >= 3 && inst.immediates[0] == 2 {
                tile_shapes.insert(def_idx, vec![inst.immediates[1], inst.immediates[2]]);
            } else if inst.immediates.len() >= 2 && matches!(inst.opcode, SsaOpcode::Constant) {
                tile_shapes.insert(def_idx, vec![inst.immediates[1]]);
            }
        }
        SsaOpcode::Mma => {
            if inst.immediates.len() >= 2 {
                tile_shapes.insert(def_idx, vec![inst.immediates[0], inst.immediates[1]]);
            }
        }
        SsaOpcode::Add | SsaOpcode::Sub | SsaOpcode::Mul | SsaOpcode::Div | SsaOpcode::Exp => {
            if let Some(shape) = inst
                .uses
                .iter()
                .find_map(|value| get_value_shape(value, tile_shapes))
            {
                tile_shapes.insert(def_idx, shape);
            }
        }
        _ => {}
    }
    opcode_map.insert(def_idx, inst.opcode.clone());
}

fn resolve_value(v: &SsaValue, value_map: &HashMap<usize, Value>) -> Value {
    match v {
        SsaValue::Param(i) => Value::Param(*i),
        SsaValue::Local(i) => value_map
            .get(i)
            .cloned()
            .unwrap_or(Value::Const(Constant::Int(0))),
        SsaValue::Const(c) => Value::Const(Constant::Int(*c)),
    }
}

fn get_def_index(def: &SsaValue) -> usize {
    match def {
        SsaValue::Local(i) => *i,
        SsaValue::Param(i) => *i,
        SsaValue::Const(_) => 0,
    }
}

fn lower_kernel_params(typed: &TypedKernel) -> Vec<Param> {
    typed
        .kernel
        .params
        .iter()
        .map(|param| Param {
            name: param.name.clone(),
            ty: Type::ptr(Type::f32()),
        })
        .collect()
}

fn get_value_shape(value: &SsaValue, tile_shapes: &HashMap<usize, Vec<i64>>) -> Option<Vec<i64>> {
    match value {
        SsaValue::Local(idx) => tile_shapes.get(idx).cloned(),
        _ => None,
    }
}

fn normalize_shape(dims: &[i64]) -> Vec<i64> {
    match dims {
        [dim] => vec![*dim, 1],
        values => values.to_vec(),
    }
}
