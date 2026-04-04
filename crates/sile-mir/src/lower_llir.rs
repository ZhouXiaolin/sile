use std::collections::HashMap;

use sile_hir::Type as HirType;
use sile_hir::typeck::TypedKernel;
use sile_llir as llir;

use crate::ir::*;

pub fn lower_mir_to_llir(mir: &MirFunction, typed: &TypedKernel) -> llir::Function {
    let mut ctx = LowerLlirCtx {
        operands: HashMap::new(),
        names: HashMap::new(),
        next_llir_value: next_llir_value(mir),
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
    next_llir_value: u32,
}

impl LowerLlirCtx {
    fn fresh_value(&mut self, prefix: &str) -> (llir::ValueId, String) {
        let id = llir::ValueId(self.next_llir_value);
        self.next_llir_value += 1;
        let name = format!("{prefix}{}", id.0);
        self.names.insert(id, name.clone());
        (id, name)
    }
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
                op: llir::InstOp::ShapeDim {
                    buf: resolve_operand(*buf, ctx),
                    dim: *dim,
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
            lower_tile_load(
                mir,
                ctx,
                out,
                llir::Operand::Value(llir_id),
                *buf,
                *row_coord,
                *col_coord,
                *rows,
                *cols,
                *stride_shape_idx,
            );
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
            lower_tile_store(
                mir,
                ctx,
                out,
                *buf,
                *value,
                *row_coord,
                *col_coord,
                *rows,
                *cols,
                *stride_shape_idx,
            );
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

fn next_llir_value(mir: &MirFunction) -> u32 {
    mir.types.keys().map(|id| id.0).max().unwrap_or(0) + 1
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

fn buffer_rank_of(value: ValueId, mir: &MirFunction) -> usize {
    match mir.types.get(&value) {
        Some(MirType::Buffer { rank }) => *rank,
        _ => 1,
    }
}

fn lower_tile_load(
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    buf: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
) {
    let buf_operand = resolve_operand(buf, ctx);
    let row_operand = resolve_operand(row_coord, ctx);
    let col_operand = resolve_operand(col_coord, ctx);
    let rank = buffer_rank_of(buf, mir);
    let tile_coord_1d = if rank <= 1 {
        Some(lower_1d_tile_coord(
            ctx,
            out,
            row_operand.clone(),
            col_operand.clone(),
        ))
    } else {
        None
    };
    let stride = if rank > 1 {
        Some(emit_shape_dim(
            ctx,
            out,
            buf_operand.clone(),
            stride_shape_idx,
        ))
    } else {
        None
    };

    for row in 0..rows {
        for col in 0..cols {
            let linear_index = if let Some(tile_coord) = tile_coord_1d.clone() {
                let tile_base = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    tile_coord,
                    const_i64(cols),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    tile_base,
                    const_i64(col),
                    llir::Type::I64,
                )
            } else {
                let src_row = emit_index_affine(ctx, out, row_operand.clone(), rows, row);
                let src_col = emit_index_affine(ctx, out, col_operand.clone(), cols, col);
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    src_row,
                    stride.clone().expect("stride for rank-2 load"),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_offset,
                    src_col,
                    llir::Type::I64,
                )
            };

            let src_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
            );
            let loaded = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, loaded);
        }
    }
}

fn lower_tile_store(
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    buf: ValueId,
    value: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
) {
    let buf_operand = resolve_operand(buf, ctx);
    let value_operand = resolve_operand(value, ctx);
    let row_operand = resolve_operand(row_coord, ctx);
    let col_operand = resolve_operand(col_coord, ctx);
    let rank = buffer_rank_of(buf, mir);
    let tile_coord_1d = if rank <= 1 {
        Some(lower_1d_tile_coord(
            ctx,
            out,
            row_operand.clone(),
            col_operand.clone(),
        ))
    } else {
        None
    };
    let stride = if rank > 1 {
        Some(emit_shape_dim(
            ctx,
            out,
            buf_operand.clone(),
            stride_shape_idx,
        ))
    } else {
        None
    };

    for row in 0..rows {
        for col in 0..cols {
            let src_ptr = emit_gep(
                ctx,
                out,
                value_operand.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let scalar = emit_load(ctx, out, src_ptr, llir::Type::F32);

            let linear_index = if let Some(tile_coord) = tile_coord_1d.clone() {
                let tile_base = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    tile_coord,
                    const_i64(cols),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    tile_base,
                    const_i64(col),
                    llir::Type::I64,
                )
            } else {
                let dst_row = emit_index_affine(ctx, out, row_operand.clone(), rows, row);
                let dst_col = emit_index_affine(ctx, out, col_operand.clone(), cols, col);
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    dst_row,
                    stride.clone().expect("stride for rank-2 store"),
                    llir::Type::I64,
                );
                emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_offset,
                    dst_col,
                    llir::Type::I64,
                )
            };

            let dst_ptr = emit_gep(
                ctx,
                out,
                buf_operand.clone(),
                vec![linear_index],
                llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        }
    }
}

fn lower_1d_tile_coord(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    row_coord: llir::Operand,
    col_coord: llir::Operand,
) -> llir::Operand {
    let non_zero = emit_cmp(
        ctx,
        out,
        llir::CmpPred::Ne,
        col_coord.clone(),
        const_i64(0),
        llir::Type::I1,
    );
    emit_select(ctx, out, non_zero, col_coord, row_coord, llir::Type::I64)
}

fn emit_index_affine(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    tile_coord: llir::Operand,
    tile_extent: i64,
    offset: i64,
) -> llir::Operand {
    let base = emit_bin(
        ctx,
        out,
        llir::BinOp::Mul,
        tile_coord,
        const_i64(tile_extent),
        llir::Type::I64,
    );
    if offset == 0 {
        base
    } else {
        emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            base,
            const_i64(offset),
            llir::Type::I64,
        )
    }
}

fn emit_shape_dim(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    buf: llir::Operand,
    dim: usize,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty: llir::Type::I64,
        op: llir::InstOp::ShapeDim { buf, dim },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn emit_gep(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    base: llir::Operand,
    indices: Vec<llir::Operand>,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Gep { base, indices },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn emit_load(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    ptr: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Load { ptr },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn emit_store(out: &mut Vec<llir::Inst>, ptr: llir::Operand, value: llir::Operand) {
    out.push(llir::Inst {
        result: None,
        result_name: None,
        ty: llir::Type::Void,
        op: llir::InstOp::Store { ptr, value },
        metadata: Vec::new(),
    });
}

fn emit_bin(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    op: llir::BinOp,
    lhs: llir::Operand,
    rhs: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Bin { op, lhs, rhs },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn emit_cmp(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    pred: llir::CmpPred,
    lhs: llir::Operand,
    rhs: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Cmp { pred, lhs, rhs },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn emit_select(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    cond: llir::Operand,
    on_true: llir::Operand,
    on_false: llir::Operand,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Select {
            cond,
            on_true,
            on_false,
        },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn const_i64(value: i64) -> llir::Operand {
    llir::Operand::Const(llir::Constant::Int(value))
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
