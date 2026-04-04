use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_hir::{ElemType, Param as HirParam, Type as HirType};
use sile_lir::builder::LirBuilder;
use sile_lir::{
    Constant, ExecutableKernel, Instruction, KernelAbi, KernelParamAbi, LaunchSemantics,
    Param as LirParam, ParamPassing, ShapeLayout, Type as LirType, Value, ValueInfo,
    ValueInfoTable,
};

use crate::ir::*;

/// Lower a MIR function to an ExecutableKernel (LIR).
pub fn lower_mir_to_lir(mir: &MirFunction, typed: &TypedKernel) -> ExecutableKernel {
    let lir_params: Vec<LirParam> = mir
        .params
        .iter()
        .map(|p| LirParam {
            name: p.name.clone(),
            ty: LirType::ptr(LirType::f32()),
        })
        .collect();

    let mut builder = LirBuilder::new(&mir.name, lir_params, LirType::Void);

    let param_info: Vec<ValueInfo> = typed
        .kernel
        .params
        .iter()
        .map(|p| ValueInfo::Buffer {
            elem: elem_of_param(p),
            rank: rank_of_param(p),
        })
        .collect();

    let mut ctx = LowerLirCtx {
        value_map: HashMap::new(),
        local_info: HashMap::new(),
        instruction_info: Vec::new(),
        max_program_id_dim: 0,
    };

    // Map MIR params to LIR params
    for (i, param) in mir.params.iter().enumerate() {
        ctx.value_map.insert(param.value, Value::Param(i));
    }

    let mut block_labels = HashMap::new();
    for block in &mir.blocks {
        let label = format!("bb{}", block.id.0);
        builder.append_block(&label);
        block_labels.insert(block.id, label);
    }

    for block in &mir.blocks {
        let label = block_labels
            .get(&block.id)
            .cloned()
            .unwrap_or_else(|| format!("bb{}", block.id.0));
        builder.switch_to_block(&label);

        for param in &block.params {
            let info = mir
                .types
                .get(param)
                .map(value_info_from_mir_type)
                .unwrap_or(ValueInfo::Void);
            let phi_ty = mir
                .types
                .get(param)
                .map(phi_type_from_mir_type)
                .unwrap_or(LirType::Void);
            let lir_val = builder.push_instruction(Instruction::BlockParam);
            let dest = match &lir_val {
                Value::Inst(idx) => format!("v{}", idx),
                _ => unreachable!("block parameter must lower to an instruction value"),
            };
            builder.phi(&dest, phi_ty, Vec::new());
            ctx.record(*param, lir_val, info);
        }

        for inst in &block.insts {
            lower_mir_inst(inst, &mut builder, &param_info, &mut ctx);
        }

        lower_terminator(&block.terminator, &block_labels, &mut builder, &ctx);
    }

    for block in &mir.blocks {
        let pred_label = block_labels
            .get(&block.id)
            .cloned()
            .unwrap_or_else(|| format!("bb{}", block.id.0));
        match &block.terminator {
            MirTerminator::Jump { target, args } => {
                attach_phi_incoming(
                    *target,
                    args,
                    &pred_label,
                    &block_labels,
                    &mut builder,
                    &ctx,
                );
            }
            MirTerminator::Branch {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                attach_phi_incoming(
                    *true_target,
                    true_args,
                    &pred_label,
                    &block_labels,
                    &mut builder,
                    &ctx,
                );
                attach_phi_incoming(
                    *false_target,
                    false_args,
                    &pred_label,
                    &block_labels,
                    &mut builder,
                    &ctx,
                );
            }
            MirTerminator::Return => {}
        }
    }

    let func = builder.finish();

    ExecutableKernel {
        name: mir.name.clone(),
        abi: build_kernel_abi(typed, ctx.max_program_id_dim.saturating_add(1).max(1)),
        func,
        value_info: ValueInfoTable {
            params: param_info,
            instructions: ctx.instruction_info,
        },
    }
}

struct LowerLirCtx {
    value_map: HashMap<ValueId, Value>,
    local_info: HashMap<ValueId, ValueInfo>,
    instruction_info: Vec<ValueInfo>,
    max_program_id_dim: usize,
}

impl LowerLirCtx {
    fn resolve(&self, v: ValueId) -> Value {
        self.value_map
            .get(&v)
            .cloned()
            .unwrap_or(Value::Const(Constant::Int(0)))
    }

    fn record(&mut self, mir_val: ValueId, lir_val: Value, info: ValueInfo) {
        self.value_map.insert(mir_val, lir_val);
        self.local_info.insert(mir_val, info.clone());
        self.instruction_info.push(info);
    }

    fn record_no_info(&mut self, mir_val: ValueId, lir_val: Value) {
        self.value_map.insert(mir_val, lir_val);
    }
}

fn lower_mir_inst(
    inst: &MirInst,
    builder: &mut LirBuilder,
    _param_info: &[ValueInfo],
    ctx: &mut LowerLirCtx,
) {
    match &inst.op {
        MirOp::ConstI64(v) => {
            ctx.record_no_info(inst.result, Value::Const(Constant::Int(*v)));
        }
        MirOp::ConstF64(v) => {
            ctx.record_no_info(inst.result, Value::Const(Constant::Float(*v)));
        }
        MirOp::ProgramId { dim } => {
            let value = builder.get_tile_coord(*dim);
            ctx.max_program_id_dim = ctx.max_program_id_dim.max(*dim as usize);
            ctx.record(inst.result, value, ValueInfo::Index);
        }
        MirOp::ShapeDim { buf: _, dim } => {
            ctx.record_no_info(inst.result, Value::ShapeDim(*dim));
        }
        MirOp::TileConstant { value, rows, cols } => {
            let lir_val = builder.tile_alloc(*rows, *cols, *value);
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *rows,
                cols: *cols,
            };
            ctx.record(inst.result, lir_val, info);
        }
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            let lir_buf = ctx.resolve(*buf);
            let lir_row = ctx.resolve(*row_coord);
            let lir_col = ctx.resolve(*col_coord);
            let lir_val =
                builder.tile_load_2d(lir_buf, *rows, *cols, lir_row, lir_col, *stride_shape_idx);
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *rows,
                cols: *cols,
            };
            ctx.record(inst.result, lir_val, info);
        }
        MirOp::TileStore {
            buf,
            value,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            let lir_buf = ctx.resolve(*buf);
            let lir_val = ctx.resolve(*value);
            let lir_row = ctx.resolve(*row_coord);
            let lir_col = ctx.resolve(*col_coord);
            builder.tile_store_2d(
                lir_buf,
                lir_val,
                *rows,
                *cols,
                lir_row,
                lir_col,
                *stride_shape_idx,
            );
            ctx.record(inst.result, Value::Const(Constant::Int(0)), ValueInfo::Void);
        }
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let lir_lhs = ctx.resolve(*lhs);
            let lir_rhs = ctx.resolve(*rhs);
            let lir_val = match op {
                BinOp::Add => builder.add(lir_lhs, lir_rhs),
                BinOp::Sub => builder.sub(lir_lhs, lir_rhs),
                BinOp::Mul => builder.mul(lir_lhs, lir_rhs),
                BinOp::Div => builder.push_instruction(Instruction::Div(lir_lhs, lir_rhs)),
            };
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *rows,
                cols: *cols,
            };
            ctx.record(inst.result, lir_val, info);
        }
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => {
            let lir_op = ctx.resolve(*operand);
            let lir_val = match op {
                UnaryOp::Exp => builder.exp(lir_op),
                UnaryOp::Neg => builder.push_instruction(Instruction::FNeg(lir_op)),
            };
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *rows,
                cols: *cols,
            };
            ctx.record(inst.result, lir_val, info);
        }
        MirOp::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            let lir_a = ctx.resolve(*a);
            let lir_b = ctx.resolve(*b);
            let lir_acc = ctx.resolve(*acc);
            let lir_val = builder.tile_mma(lir_a, lir_b, lir_acc, *tile_m, *tile_n, *tile_k);
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *tile_m,
                cols: *tile_n,
            };
            ctx.record(inst.result, lir_val, info);
        }
        MirOp::TileReduce {
            op,
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            let lir_val = ctx.resolve(*value);
            let lir_result = match op {
                ReduceOp::Max => builder.tile_reduce_max(lir_val, *axis, *in_rows, *in_cols),
                ReduceOp::Sum => builder.tile_reduce_sum(lir_val, *axis, *in_rows, *in_cols),
            };
            let (out_rows, out_cols) = if *axis == 1 {
                (*in_rows, 1i64)
            } else {
                (1i64, *in_cols)
            };
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: out_rows,
                cols: out_cols,
            };
            ctx.record(inst.result, lir_result, info);
        }
        MirOp::TileBroadcast { value, rows, cols } => {
            let lir_val = ctx.resolve(*value);
            let lir_result = builder.tile_broadcast(lir_val, *rows, *cols);
            let info = ValueInfo::Tile {
                elem: ElemType::F32,
                rows: *rows,
                cols: *cols,
            };
            ctx.record(inst.result, lir_result, info);
        }
        MirOp::IBinary { op, lhs, rhs } => {
            let lir_lhs = ctx.resolve(*lhs);
            let lir_rhs = ctx.resolve(*rhs);
            let lir_val = match op {
                BinOp::Add => builder.add(lir_lhs, lir_rhs),
                BinOp::Sub => builder.sub(lir_lhs, lir_rhs),
                BinOp::Mul => builder.mul(lir_lhs, lir_rhs),
                BinOp::Div => builder.push_instruction(Instruction::Div(lir_lhs, lir_rhs)),
            };
            ctx.record(inst.result, lir_val, ValueInfo::Index);
        }
        MirOp::ICmp { op: _, lhs, rhs } => {
            let lir_lhs = ctx.resolve(*lhs);
            let lir_rhs = ctx.resolve(*rhs);
            let lir_val = builder.icmp(sile_lir::CmpOp::Slt, lir_lhs, lir_rhs);
            ctx.record(inst.result, lir_val, ValueInfo::Index);
        }
    }
}

fn lower_terminator(
    term: &MirTerminator,
    block_labels: &HashMap<BlockId, String>,
    builder: &mut LirBuilder,
    ctx: &LowerLirCtx,
) {
    match term {
        MirTerminator::Jump { target, .. } => {
            if let Some(label) = block_labels.get(target) {
                builder.br(label);
            }
        }
        MirTerminator::Branch {
            cond,
            true_target,
            false_target,
            ..
        } => {
            let lir_cond = ctx.resolve(*cond);
            let true_label = block_labels
                .get(true_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", true_target.0));
            let false_label = block_labels
                .get(false_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", false_target.0));
            builder.cond_br(lir_cond, &true_label, &false_label);
        }
        MirTerminator::Return => builder.ret(None),
    }
}

fn attach_phi_incoming(
    target: BlockId,
    args: &[ValueId],
    pred_label: &str,
    block_labels: &HashMap<BlockId, String>,
    builder: &mut LirBuilder,
    ctx: &LowerLirCtx,
) {
    let Some(target_label) = block_labels.get(&target) else {
        return;
    };
    let Some(block) = builder.func.get_block_mut(target_label) else {
        return;
    };

    for (idx, arg) in args.iter().enumerate() {
        let Some(phi) = block.phi_nodes.get_mut(idx) else {
            continue;
        };
        phi.incoming
            .push((ctx.resolve(*arg), pred_label.to_string()));
    }
}

// ── ABI construction helpers (reused from old lower_lir) ───────────

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

fn value_info_from_mir_type(ty: &MirType) -> ValueInfo {
    match ty {
        MirType::I64 => ValueInfo::Index,
        MirType::F32 => ValueInfo::Scalar {
            elem: ElemType::F32,
        },
        MirType::Tile { rows, cols } => ValueInfo::Tile {
            elem: ElemType::F32,
            rows: *rows,
            cols: *cols,
        },
        MirType::Buffer { rank } => ValueInfo::Buffer {
            elem: ElemType::F32,
            rank: *rank,
        },
        MirType::Void => ValueInfo::Void,
    }
}

fn phi_type_from_mir_type(ty: &MirType) -> LirType {
    match ty {
        MirType::I64 => LirType::i64(),
        MirType::F32 => LirType::f32(),
        _ => LirType::Void,
    }
}
