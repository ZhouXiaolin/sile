use std::collections::HashMap;

use sile_hir::Type as HirType;
use sile_hir::typeck::TypedKernel;
use sile_llir as llir;

use crate::ir::*;

pub fn lower_mir_to_llir(mir: &MirFunction, typed: &TypedKernel) -> llir::Function {
    let mut ctx = LowerLlirCtx {
        operands: HashMap::new(),
        names: HashMap::new(),
    };
    let param_abis = lower_param_abis(typed);

    let params = mir
        .params
        .iter()
        .enumerate()
        .map(|(idx, param)| {
            let id = llir_value(param.value);
            ctx.names.insert(id, param.name.clone());
            ctx.operands.insert(param.value, llir::Operand::Value(id));
            llir::Param {
                id,
                name: param.name.clone(),
                ty: llir_type(&param.ty),
                abi: param_abis.get(idx).cloned().flatten(),
            }
        })
        .collect();

    let blocks = mir
        .blocks
        .iter()
        .map(|block| lower_block(block, mir, &mut ctx))
        .collect();

    llir::Function {
        name: mir.name.clone(),
        params,
        blocks,
        entry: llir_block(mir.entry),
        metadata: Vec::new(),
    }
}

fn lower_param_abis(typed: &TypedKernel) -> Vec<Option<llir::ParamAbi>> {
    let mut next_shape_offset = 0usize;
    typed
        .kernel
        .params
        .iter()
        .map(|param| match &param.ty {
            HirType::Tensor { shape, .. } | HirType::Tile { shape, .. } => {
                let abi = llir::ParamAbi {
                    rank: shape.rank(),
                    shape_offset: next_shape_offset,
                };
                next_shape_offset += abi.rank;
                Some(abi)
            }
            HirType::Shape | HirType::Scalar(_) => None,
        })
        .collect()
}

struct LowerLlirCtx {
    operands: HashMap<ValueId, llir::Operand>,
    names: HashMap<llir::ValueId, String>,
}

fn lower_block(block: &MirBlock, mir: &MirFunction, ctx: &mut LowerLlirCtx) -> llir::BasicBlock {
    let params = block
        .params
        .iter()
        .map(|param| {
            let id = llir_value(*param);
            let name = format!("v{}", param.0);
            ctx.names.insert(id, name.clone());
            ctx.operands.insert(*param, llir::Operand::Value(id));
            llir::BlockParam {
                id,
                name,
                ty: llir_type(mir.types.get(param).unwrap_or(&MirType::Void)),
            }
        })
        .collect();

    let mut insts = Vec::new();
    for inst in &block.insts {
        lower_inst(inst, mir, ctx, &mut insts);
    }

    let terminator = lower_terminator(&block.terminator, ctx);

    llir::BasicBlock {
        id: llir_block(block.id),
        name: format!("bb{}", block.id.0),
        params,
        insts,
        terminator,
    }
}

fn lower_inst(
    inst: &MirInst,
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
) {
    match &inst.op {
        MirOp::ConstI64(value) => {
            ctx.operands.insert(
                inst.result,
                llir::Operand::Const(llir::Constant::Int(*value)),
            );
            ctx.names
                .insert(llir_value(inst.result), format!("v{}", inst.result.0));
        }
        MirOp::ConstF64(value) => {
            ctx.operands.insert(
                inst.result,
                llir::Operand::Const(llir::Constant::Float(*value)),
            );
            ctx.names
                .insert(llir_value(inst.result), format!("v{}", inst.result.0));
        }
        MirOp::ProgramId { dim } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir_type(mir.types.get(&inst.result).unwrap_or(&MirType::I64)),
                op: llir::InstOp::Intrinsic {
                    intrinsic: llir::Intrinsic::BlockId { dim: *dim as u8 },
                    args: vec![],
                },
                metadata: Vec::new(),
            });
        }
        MirOp::ShapeDim { buf, dim } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir::Type::I64,
                op: llir::InstOp::Call {
                    func: "shape_dim".into(),
                    args: vec![
                        resolve_operand(*buf, ctx),
                        llir::Operand::Const(llir::Constant::Int(*dim as i64)),
                    ],
                },
                metadata: Vec::new(),
            });
        }
        MirOp::IBinary { op, lhs, rhs } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir_type(mir.types.get(&inst.result).unwrap_or(&MirType::I64)),
                op: llir::InstOp::Bin {
                    op: lower_bin_op(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        MirOp::ICmp { op, lhs, rhs } => {
            let llir_id = llir_value(inst.result);
            ctx.names.insert(llir_id, format!("v{}", inst.result.0));
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(format!("v{}", inst.result.0)),
                ty: llir::Type::I1,
                op: llir::InstOp::Cmp {
                    pred: lower_cmp_pred(*op),
                    lhs: resolve_operand(*lhs, ctx),
                    rhs: resolve_operand(*rhs, ctx),
                },
                metadata: Vec::new(),
            });
        }
        MirOp::TileConstant { value, rows, cols } => {
            let tile_ty = tile_ptr_type(*rows, *cols);
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ty.clone(),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*rows, *cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(void_call(
                "tile_splat_f32",
                vec![
                    llir::Operand::Value(llir_id),
                    llir::Operand::Const(llir::Constant::Float(*value)),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                ],
            ));
        }
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            let tile_ty = tile_ptr_type(*rows, *cols);
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ty.clone(),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*rows, *cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(void_call(
                "tile_load_2d_f32",
                vec![
                    llir::Operand::Value(llir_id),
                    resolve_operand(*buf, ctx),
                    resolve_operand(*row_coord, ctx),
                    resolve_operand(*col_coord, ctx),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                    llir::Operand::Const(llir::Constant::Int(*stride_shape_idx as i64)),
                ],
            ));
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
            out.push(void_call(
                "tile_store_2d_f32",
                vec![
                    resolve_operand(*buf, ctx),
                    resolve_operand(*value, ctx),
                    resolve_operand(*row_coord, ctx),
                    resolve_operand(*col_coord, ctx),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                    llir::Operand::Const(llir::Constant::Int(*stride_shape_idx as i64)),
                ],
            ));
        }
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => {
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ptr_type(*rows, *cols),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*rows, *cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(void_call(
                tile_binary_helper(*op),
                vec![
                    llir::Operand::Value(llir_id),
                    resolve_operand(*lhs, ctx),
                    resolve_operand(*rhs, ctx),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                ],
            ));
        }
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => {
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ptr_type(*rows, *cols),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*rows, *cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(void_call(
                tile_unary_helper(*op),
                vec![
                    llir::Operand::Value(llir_id),
                    resolve_operand(*operand, ctx),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                ],
            ));
        }
        MirOp::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ptr_type(*tile_m, *tile_n),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*tile_m, *tile_n),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(llir::Inst {
                result: None,
                result_name: None,
                ty: llir::Type::Void,
                op: llir::InstOp::Intrinsic {
                    intrinsic: llir::Intrinsic::MatmulFragment,
                    args: vec![
                        llir::Operand::Value(llir_id),
                        resolve_operand(*a, ctx),
                        resolve_operand(*b, ctx),
                        resolve_operand(*acc, ctx),
                        llir::Operand::Const(llir::Constant::Int(*tile_m)),
                        llir::Operand::Const(llir::Constant::Int(*tile_n)),
                        llir::Operand::Const(llir::Constant::Int(*tile_k)),
                    ],
                },
                metadata: vec![llir::Metadata::Unroll(4)],
            });
        }
        MirOp::TileReduce {
            op,
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            let (out_rows, out_cols) = if *axis == 1 {
                (*in_rows, 1)
            } else {
                (1, *in_cols)
            };
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ptr_type(out_rows, out_cols),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(out_rows, out_cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(llir::Inst {
                result: None,
                result_name: None,
                ty: llir::Type::Void,
                op: llir::InstOp::Intrinsic {
                    intrinsic: match op {
                        ReduceOp::Max => llir::Intrinsic::ReduceMax,
                        ReduceOp::Sum => llir::Intrinsic::ReduceAdd,
                    },
                    args: vec![
                        llir::Operand::Value(llir_id),
                        resolve_operand(*value, ctx),
                        llir::Operand::Const(llir::Constant::Int(*axis)),
                        llir::Operand::Const(llir::Constant::Int(*in_rows)),
                        llir::Operand::Const(llir::Constant::Int(*in_cols)),
                    ],
                },
                metadata: vec![llir::Metadata::Reduction],
            });
        }
        MirOp::TileBroadcast { value, rows, cols } => {
            let llir_id = llir_value(inst.result);
            let name = format!("v{}", inst.result.0);
            ctx.names.insert(llir_id, name.clone());
            ctx.operands
                .insert(inst.result, llir::Operand::Value(llir_id));
            out.push(llir::Inst {
                result: Some(llir_id),
                result_name: Some(name.clone()),
                ty: tile_ptr_type(*rows, *cols),
                op: llir::InstOp::Alloca {
                    alloc_ty: tile_storage_type(*rows, *cols),
                    addr_space: llir::AddressSpace::Private,
                },
                metadata: vec![llir::Metadata::Alignment(16)],
            });
            out.push(void_call(
                "tile_broadcast_f32",
                vec![
                    llir::Operand::Value(llir_id),
                    resolve_operand(*value, ctx),
                    llir::Operand::Const(llir::Constant::Int(*rows)),
                    llir::Operand::Const(llir::Constant::Int(*cols)),
                ],
            ));
        }
    }
}

fn lower_terminator(term: &MirTerminator, ctx: &LowerLlirCtx) -> llir::Terminator {
    match term {
        MirTerminator::Jump { target, args } => llir::Terminator::Br {
            target: llir_block(*target),
            args: args.iter().map(|arg| resolve_operand(*arg, ctx)).collect(),
        },
        MirTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => llir::Terminator::CondBr {
            cond: resolve_operand(*cond, ctx),
            true_target: llir_block(*true_target),
            true_args: true_args
                .iter()
                .map(|arg| resolve_operand(*arg, ctx))
                .collect(),
            false_target: llir_block(*false_target),
            false_args: false_args
                .iter()
                .map(|arg| resolve_operand(*arg, ctx))
                .collect(),
        },
        MirTerminator::Return => llir::Terminator::Ret { value: None },
    }
}

fn resolve_operand(value: ValueId, ctx: &LowerLlirCtx) -> llir::Operand {
    ctx.operands
        .get(&value)
        .cloned()
        .unwrap_or_else(|| llir::Operand::Value(llir_value(value)))
}

fn llir_value(value: ValueId) -> llir::ValueId {
    llir::ValueId(value.0)
}

fn llir_block(block: BlockId) -> llir::BlockId {
    llir::BlockId(block.0)
}

fn llir_type(ty: &MirType) -> llir::Type {
    match ty {
        MirType::I64 => llir::Type::I64,
        MirType::F32 => llir::Type::F32,
        MirType::Tile { rows, cols } => tile_ptr_type(*rows, *cols),
        MirType::Buffer { .. } => llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
        MirType::Void => llir::Type::Void,
    }
}

fn tile_storage_type(rows: i64, cols: i64) -> llir::Type {
    llir::Type::array(
        rows as usize,
        llir::Type::array(cols as usize, llir::Type::F32),
    )
}

fn tile_ptr_type(rows: i64, cols: i64) -> llir::Type {
    llir::Type::ptr(llir::AddressSpace::Private, tile_storage_type(rows, cols))
}

fn lower_bin_op(op: BinOp) -> llir::BinOp {
    match op {
        BinOp::Add => llir::BinOp::Add,
        BinOp::Sub => llir::BinOp::Sub,
        BinOp::Mul => llir::BinOp::Mul,
        BinOp::Div => llir::BinOp::Div,
    }
}

fn lower_cmp_pred(op: CmpOp) -> llir::CmpPred {
    match op {
        CmpOp::Lt => llir::CmpPred::Slt,
        CmpOp::Le => llir::CmpPred::Sle,
        CmpOp::Gt => llir::CmpPred::Sgt,
        CmpOp::Ge => llir::CmpPred::Sge,
        CmpOp::Eq => llir::CmpPred::Eq,
        CmpOp::Ne => llir::CmpPred::Ne,
    }
}

fn tile_binary_helper(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "tile_add_f32",
        BinOp::Sub => "tile_sub_f32",
        BinOp::Mul => "tile_mul_f32",
        BinOp::Div => "tile_div_f32",
    }
}

fn tile_unary_helper(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Exp => "tile_exp_f32",
        UnaryOp::Neg => "tile_neg_f32",
    }
}

fn void_call(func: &str, args: Vec<llir::Operand>) -> llir::Inst {
    llir::Inst {
        result: None,
        result_name: None,
        ty: llir::Type::Void,
        op: llir::InstOp::Call {
            func: func.into(),
            args,
        },
        metadata: Vec::new(),
    }
}
