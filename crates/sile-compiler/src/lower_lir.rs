use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_hir::{ElemType, Param as HirParam, Type as HirType};
use sile_lir::builder::LirBuilder;
use sile_lir::{
    Constant, ExecutableKernel, Instruction, KernelAbi, KernelParamAbi, LaunchSemantics, Param,
    ParamPassing, ShapeLayout, Type, Value, ValueInfo, ValueInfoTable,
};

use crate::mir::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};

pub fn lower_ssa_to_lir(ssa: &SsaProgram, typed: &TypedKernel) -> ExecutableKernel {
    let params = lower_kernel_params(typed);
    let mut builder = LirBuilder::new(&typed.kernel.name, params, Type::Void);

    let param_info = typed
        .kernel
        .params
        .iter()
        .map(|param| ValueInfo::Buffer {
            elem: elem_of_param(param),
            rank: rank_of_param(param),
        })
        .collect::<Vec<_>>();
    let mut local_info: HashMap<usize, ValueInfo> = HashMap::new();
    let mut instruction_info = Vec::new();
    let mut value_map: HashMap<usize, Value> = HashMap::new();
    let mut opcode_map: HashMap<usize, SsaOpcode> = HashMap::new();
    let mut max_program_id_dim = 0usize;

    for (i, _) in typed.kernel.params.iter().enumerate() {
        value_map.insert(i, Value::Param(i));
    }

    let body = builder.append_block("body");
    builder.switch_to_block(&body);

    for inst in &ssa.instructions {
        lower_ssa_instruction(
            inst,
            &mut builder,
            &param_info,
            &mut local_info,
            &mut instruction_info,
            &mut value_map,
            &mut opcode_map,
            &mut max_program_id_dim,
        );
    }

    builder.ret(None);
    let func = builder.finish();

    ExecutableKernel {
        name: typed.kernel.name.clone(),
        abi: build_kernel_abi(typed, max_program_id_dim.saturating_add(1).max(1)),
        func,
        value_info: ValueInfoTable {
            params: param_info,
            instructions: instruction_info,
        },
    }
}

fn lower_ssa_instruction(
    inst: &SsaInstruction,
    builder: &mut LirBuilder,
    param_info: &[ValueInfo],
    local_info: &mut HashMap<usize, ValueInfo>,
    instruction_info: &mut Vec<ValueInfo>,
    value_map: &mut HashMap<usize, Value>,
    opcode_map: &mut HashMap<usize, SsaOpcode>,
    max_program_id_dim: &mut usize,
) {
    let def_idx = get_def_index(&inst.def);
    let mut record_local = |info: ValueInfo| {
        local_info.insert(def_idx, info);
    };

    let lowered = match inst.opcode {
        SsaOpcode::ProgramId => {
            record_local(ValueInfo::Index);
            Some(builder.const_int(0))
        }
        SsaOpcode::ShapeDim => {
            let dim = inst.immediates.first().copied().unwrap_or(0);
            if inst.uses.first().and_then(|value| match value {
                SsaValue::Local(idx) => opcode_map.get(idx),
                _ => None,
            }) == Some(&SsaOpcode::ProgramId)
            {
                let value = builder.get_tile_coord(dim);
                *max_program_id_dim = (*max_program_id_dim).max(dim as usize);
                record_local(ValueInfo::Index);
                instruction_info.push(ValueInfo::Index);
                Some(value)
            } else {
                record_local(ValueInfo::Shape);
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
                let value = builder.tile_load_2d(buf, rows, cols, row_tile, col_tile, stride_shape_idx);
                let info = ValueInfo::Tile {
                    elem: ElemType::F32,
                    rows,
                    cols,
                };
                record_local(info.clone());
                instruction_info.push(info);
                Some(value)
            } else {
                let ptr = resolve_value(&inst.uses[0], value_map);
                let value = builder.load(ptr, Type::f32());
                let info = ValueInfo::Scalar { elem: ElemType::F32 };
                record_local(info.clone());
                instruction_info.push(info);
                Some(value)
            }
        }
        SsaOpcode::Add => emit_binary_inst(
            builder.add(
                resolve_value(&inst.uses[0], value_map),
                resolve_value(&inst.uses[1], value_map),
            ),
            infer_binary_info(&inst.uses[0], &inst.uses[1], param_info, local_info),
            def_idx,
            local_info,
            instruction_info,
        ),
        SsaOpcode::Sub => emit_binary_inst(
            builder.sub(
                resolve_value(&inst.uses[0], value_map),
                resolve_value(&inst.uses[1], value_map),
            ),
            infer_binary_info(&inst.uses[0], &inst.uses[1], param_info, local_info),
            def_idx,
            local_info,
            instruction_info,
        ),
        SsaOpcode::Mul => emit_binary_inst(
            builder.mul(
                resolve_value(&inst.uses[0], value_map),
                resolve_value(&inst.uses[1], value_map),
            ),
            infer_binary_info(&inst.uses[0], &inst.uses[1], param_info, local_info),
            def_idx,
            local_info,
            instruction_info,
        ),
        SsaOpcode::Div => emit_binary_inst(
            builder.push_instruction(Instruction::Div(
                resolve_value(&inst.uses[0], value_map),
                resolve_value(&inst.uses[1], value_map),
            )),
            infer_binary_info(&inst.uses[0], &inst.uses[1], param_info, local_info),
            def_idx,
            local_info,
            instruction_info,
        ),
        SsaOpcode::Exp => {
            let value = builder.exp(resolve_value(&inst.uses[0], value_map));
            let info = info_for_ssa_value(&inst.uses[0], param_info, local_info)
                .unwrap_or(ValueInfo::Scalar { elem: ElemType::F32 });
            local_info.insert(def_idx, info.clone());
            instruction_info.push(info);
            Some(value)
        }
        SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
            let source = inst
                .uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .unwrap_or_else(|| builder.const_int(0));
            let source_shape = tile_dims_for_value(inst.uses.first(), param_info, local_info);
            if let Some((rows, cols)) = source_shape {
                let axis = inst.immediates.first().copied().unwrap_or(0);
                let value = match inst.opcode {
                    SsaOpcode::ReduceMax => builder.tile_reduce_max(source, axis, rows, cols),
                    SsaOpcode::ReduceSum => builder.tile_reduce_sum(source, axis, rows, cols),
                    _ => unreachable!(),
                };
                let info = if axis == 1 {
                    ValueInfo::Tile {
                        elem: ElemType::F32,
                        rows,
                        cols: 1,
                    }
                } else {
                    ValueInfo::Tile {
                        elem: ElemType::F32,
                        rows: 1,
                        cols,
                    }
                };
                local_info.insert(def_idx, info.clone());
                instruction_info.push(info);
                Some(value)
            } else {
                if let Some(info) = info_for_ssa_value(&inst.uses[0], param_info, local_info) {
                    local_info.insert(def_idx, info);
                }
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
            instruction_info.push(ValueInfo::Void);
            None
        }
        SsaOpcode::Mma => {
            if inst.immediates.len() >= 3 {
                let value = builder.tile_mma(
                    resolve_value(&inst.uses[0], value_map),
                    resolve_value(&inst.uses[1], value_map),
                    resolve_value(&inst.uses[2], value_map),
                    inst.immediates[0],
                    inst.immediates[1],
                    inst.immediates[2],
                );
                let info = ValueInfo::Tile {
                    elem: ElemType::F32,
                    rows: inst.immediates[0],
                    cols: inst.immediates[1],
                };
                local_info.insert(def_idx, info.clone());
                instruction_info.push(info);
                Some(value)
            } else {
                emit_binary_inst(
                    builder.mul(
                        resolve_value(&inst.uses[0], value_map),
                        resolve_value(&inst.uses[1], value_map),
                    ),
                    infer_binary_info(&inst.uses[0], &inst.uses[1], param_info, local_info),
                    def_idx,
                    local_info,
                    instruction_info,
                )
            }
        }
        SsaOpcode::Constant => {
            if inst.immediates.len() >= 3 {
                let init = f32::from_bits(inst.immediates[0] as u32) as f64;
                let rows = inst.immediates[1];
                let cols = inst.immediates[2];
                let value = builder.tile_alloc(rows, cols, init);
                let info = ValueInfo::Tile {
                    elem: ElemType::F32,
                    rows,
                    cols,
                };
                local_info.insert(def_idx, info.clone());
                instruction_info.push(info);
                Some(value)
            } else {
                local_info.insert(def_idx, ValueInfo::Scalar { elem: ElemType::F32 });
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
            let info = if !inst.immediates.is_empty() {
                let (rows, cols) = normalize_tile_dims(&inst.immediates);
                ValueInfo::Tile {
                    elem: ElemType::F32,
                    rows,
                    cols,
                }
            } else {
                info_for_ssa_value(&inst.uses[0], param_info, local_info)
                    .unwrap_or(ValueInfo::Scalar { elem: ElemType::F32 })
            };
            local_info.insert(def_idx, info);
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
                let value = builder.tile_broadcast(value, rows, cols);
                let info = ValueInfo::Tile {
                    elem: ElemType::F32,
                    rows,
                    cols,
                };
                local_info.insert(def_idx, info.clone());
                instruction_info.push(info);
                Some(value)
            } else {
                if let Some(info) = info_for_ssa_value(&inst.uses[0], param_info, local_info) {
                    local_info.insert(def_idx, info);
                }
                Some(value)
            }
        }
        SsaOpcode::ShapeOf => {
            local_info.insert(def_idx, ValueInfo::Shape);
            inst.uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .or_else(|| Some(builder.const_int(0)))
        }
        SsaOpcode::ScalarDiv => {
            local_info.insert(def_idx, ValueInfo::Scalar { elem: ElemType::F32 });
            inst.uses
                .first()
                .map(|value| resolve_value(value, value_map))
                .or_else(|| Some(builder.const_int(0)))
        }
    };

    if let Some(value) = lowered {
        value_map.insert(def_idx, value);
    }
    opcode_map.insert(def_idx, inst.opcode.clone());
}

fn emit_binary_inst(
    value: Value,
    info: ValueInfo,
    def_idx: usize,
    local_info: &mut HashMap<usize, ValueInfo>,
    instruction_info: &mut Vec<ValueInfo>,
) -> Option<Value> {
    local_info.insert(def_idx, info.clone());
    instruction_info.push(info);
    Some(value)
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

fn build_kernel_abi(typed: &TypedKernel, program_id_dims: usize) -> KernelAbi {
    let mut offsets = Vec::with_capacity(typed.kernel.params.len());
    let mut next = 0usize;
    let params = typed
        .kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let rank = rank_of_param(param);
            offsets.push(next);
            next += rank;
            KernelParamAbi {
                index,
                name: param.name.clone(),
                kind: param.kind,
                elem: elem_of_param(param),
                rank,
                passing: ParamPassing::Buffer,
            }
        })
        .collect();

    KernelAbi {
        params,
        shape_layout: ShapeLayout {
            total_dims: next,
            offsets,
        },
        launch: LaunchSemantics { program_id_dims },
    }
}

fn elem_of_param(param: &HirParam) -> ElemType {
    match &param.ty {
        HirType::Tensor { elem, .. } | HirType::Tile { elem, .. } | HirType::Scalar(elem) => *elem,
        HirType::Shape => ElemType::F32,
    }
}

fn rank_of_param(param: &HirParam) -> usize {
    match &param.ty {
        HirType::Tensor { shape, .. } | HirType::Tile { shape, .. } => shape.rank(),
        HirType::Shape | HirType::Scalar(_) => 0,
    }
}

fn infer_binary_info(
    lhs: &SsaValue,
    rhs: &SsaValue,
    param_info: &[ValueInfo],
    local_info: &HashMap<usize, ValueInfo>,
) -> ValueInfo {
    info_for_ssa_value(lhs, param_info, local_info)
        .or_else(|| info_for_ssa_value(rhs, param_info, local_info))
        .unwrap_or(ValueInfo::Scalar { elem: ElemType::F32 })
}

fn info_for_ssa_value(
    value: &SsaValue,
    param_info: &[ValueInfo],
    local_info: &HashMap<usize, ValueInfo>,
) -> Option<ValueInfo> {
    match value {
        SsaValue::Param(i) => param_info.get(*i).cloned(),
        SsaValue::Local(i) => local_info.get(i).cloned(),
        SsaValue::Const(_) => Some(ValueInfo::Scalar { elem: ElemType::F32 }),
    }
}

fn tile_dims_for_value(
    value: Option<&SsaValue>,
    param_info: &[ValueInfo],
    local_info: &HashMap<usize, ValueInfo>,
) -> Option<(i64, i64)> {
    match value.and_then(|value| info_for_ssa_value(value, param_info, local_info)) {
        Some(ValueInfo::Tile { rows, cols, .. }) => Some((rows, cols)),
        _ => None,
    }
}

fn normalize_tile_dims(dims: &[i64]) -> (i64, i64) {
    match dims {
        [dim] => (*dim, 1),
        [rows, cols, ..] => (*rows, *cols),
        [] => (1, 1),
    }
}
