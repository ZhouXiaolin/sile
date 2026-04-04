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
        next_llir_block: next_llir_block(mir),
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
        .flat_map(|block| lower_block(block, mir, &mut ctx))
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
    next_llir_block: u32,
}

impl LowerLlirCtx {
    fn fresh_value(&mut self, prefix: &str) -> (llir::ValueId, String) {
        let id = llir::ValueId(self.next_llir_value);
        self.next_llir_value += 1;
        let name = format!("{prefix}{}", id.0);
        self.names.insert(id, name.clone());
        (id, name)
    }

    fn fresh_block_id(&mut self) -> llir::BlockId {
        let id = llir::BlockId(self.next_llir_block);
        self.next_llir_block += 1;
        id
    }
}

#[derive(Clone)]
struct PendingBlock {
    id: llir::BlockId,
    name: String,
    params: Vec<llir::BlockParam>,
    insts: Vec<llir::Inst>,
    terminator: Option<llir::Terminator>,
}

struct BlockLowerer<'a> {
    mir: &'a MirFunction,
    ctx: &'a mut LowerLlirCtx,
    blocks: Vec<PendingBlock>,
    current: usize,
}

impl<'a> BlockLowerer<'a> {
    fn new(
        mir: &'a MirFunction,
        ctx: &'a mut LowerLlirCtx,
        id: llir::BlockId,
        name: String,
        params: Vec<llir::BlockParam>,
    ) -> Self {
        Self {
            mir,
            ctx,
            blocks: vec![PendingBlock {
                id,
                name,
                params,
                insts: Vec::new(),
                terminator: None,
            }],
            current: 0,
        }
    }

    fn with_current_insts<R>(
        &mut self,
        f: impl FnOnce(&mut LowerLlirCtx, &MirFunction, &mut Vec<llir::Inst>) -> R,
    ) -> R {
        let current = self.current;
        f(self.ctx, self.mir, &mut self.blocks[current].insts)
    }

    fn set_current_terminator(&mut self, term: llir::Terminator) {
        self.blocks[self.current].terminator = Some(term);
    }

    fn create_block(
        &mut self,
        prefix: &str,
        params: Vec<(&str, llir::Type)>,
    ) -> (llir::BlockId, Vec<llir::BlockParam>) {
        let id = self.ctx.fresh_block_id();
        let block_params = params
            .into_iter()
            .map(|(param_prefix, ty)| {
                let (id, name) = self.ctx.fresh_value(param_prefix);
                llir::BlockParam { id, name, ty }
            })
            .collect::<Vec<_>>();
        self.blocks.push(PendingBlock {
            id,
            name: format!("{prefix}_{}", id.0),
            params: block_params.clone(),
            insts: Vec::new(),
            terminator: None,
        });
        (id, block_params)
    }

    fn switch_to(&mut self, id: llir::BlockId) {
        self.current = self
            .blocks
            .iter()
            .position(|block| block.id == id)
            .expect("LLIR block must exist");
    }

    fn finish(mut self, final_terminator: llir::Terminator) -> Vec<llir::BasicBlock> {
        if self.blocks[self.current].terminator.is_none() {
            self.blocks[self.current].terminator = Some(final_terminator);
        }
        self.blocks
            .into_iter()
            .map(|block| llir::BasicBlock {
                id: block.id,
                name: block.name,
                params: block.params,
                insts: block.insts,
                terminator: block.terminator.expect("LLIR block missing terminator"),
            })
            .collect()
    }
}

fn lower_block(
    block: &MirBlock,
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
) -> Vec<llir::BasicBlock> {
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
        .collect::<Vec<_>>();

    let mut builder = BlockLowerer::new(
        mir,
        ctx,
        llir_block(block.id),
        format!("bb{}", block.id.0),
        params,
    );

    for inst in &block.insts {
        match &inst.op {
            MirOp::TileConstant { value, rows, cols } => {
                lower_tile_constant_inst(inst.result, *value, *rows, *cols, &mut builder);
            }
            MirOp::TileLoad {
                buf,
                row_coord,
                col_coord,
                rows,
                cols,
                stride_shape_idx,
            } => {
                lower_tile_load_inst(
                    inst.result,
                    *buf,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
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
                lower_tile_store_inst(
                    *buf,
                    *value,
                    *row_coord,
                    *col_coord,
                    *rows,
                    *cols,
                    *stride_shape_idx,
                    &mut builder,
                );
            }
            MirOp::TileBinary {
                op,
                lhs,
                rhs,
                rows,
                cols,
            } => {
                lower_tile_binary_inst(
                    inst.result,
                    *lhs,
                    *rhs,
                    lower_bin_op(*op),
                    *rows,
                    *cols,
                    &mut builder,
                );
            }
            MirOp::TileUnary {
                op,
                operand,
                rows,
                cols,
            } => {
                lower_tile_unary_inst(inst.result, *operand, *op, *rows, *cols, &mut builder);
            }
            MirOp::TileMma {
                a,
                b,
                acc,
                tile_m,
                tile_n,
                tile_k,
            } => {
                lower_tile_mma_inst(
                    inst.result,
                    *a,
                    *b,
                    *acc,
                    *tile_m,
                    *tile_n,
                    *tile_k,
                    &mut builder,
                );
            }
            MirOp::TileReduce {
                op,
                value,
                axis,
                in_rows,
                in_cols,
            } => {
                lower_tile_reduce_inst(
                    inst.result,
                    *value,
                    *op,
                    *axis,
                    *in_rows,
                    *in_cols,
                    &mut builder,
                );
            }
            MirOp::TileBroadcast { value, rows, cols } => {
                lower_tile_broadcast_inst(inst.result, *value, *rows, *cols, &mut builder);
            }
            _ => {
                builder.with_current_insts(|ctx, mir, out| lower_inst(inst, mir, ctx, out));
            }
        }
    }

    let terminator = lower_terminator(&block.terminator, builder.ctx);
    builder.finish(terminator)
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
            lower_tile_constant(
                ctx,
                out,
                llir::Operand::Value(llir_id),
                *value,
                *rows,
                *cols,
            );
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
            lower_tile_binary(
                ctx,
                out,
                llir::Operand::Value(llir_id),
                resolve_operand(*lhs, ctx),
                resolve_operand(*rhs, ctx),
                lower_bin_op(*op),
                *rows,
                *cols,
            );
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
            lower_tile_unary(
                ctx,
                out,
                llir::Operand::Value(llir_id),
                resolve_operand(*operand, ctx),
                *op,
                *rows,
                *cols,
            );
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
            lower_tile_mma(
                ctx,
                out,
                llir::Operand::Value(llir_id),
                resolve_operand(*a, ctx),
                resolve_operand(*b, ctx),
                resolve_operand(*acc, ctx),
                *tile_m,
                *tile_n,
                *tile_k,
            );
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
            lower_tile_reduce(
                ctx,
                out,
                llir::Operand::Value(llir_id),
                resolve_operand(*value, ctx),
                *op,
                *axis,
                *in_rows,
                *in_cols,
            );
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
            lower_tile_broadcast(
                mir,
                ctx,
                out,
                llir::Operand::Value(llir_id),
                *value,
                *rows,
                *cols,
            );
        }
    }
}

fn alloc_tile_result(
    builder: &mut BlockLowerer<'_>,
    result: ValueId,
    rows: i64,
    cols: i64,
) -> llir::Operand {
    let llir_id = llir_value(result);
    let name = format!("v{}", result.0);
    builder.ctx.names.insert(llir_id, name.clone());
    builder
        .ctx
        .operands
        .insert(result, llir::Operand::Value(llir_id));
    builder.with_current_insts(|_, _, out| {
        out.push(llir::Inst {
            result: Some(llir_id),
            result_name: Some(name),
            ty: tile_ptr_type(rows, cols),
            op: llir::InstOp::Alloca {
                alloc_ty: tile_storage_type(rows, cols),
                addr_space: llir::AddressSpace::Private,
            },
            metadata: vec![llir::Metadata::Alignment(16)],
        });
    });
    llir::Operand::Value(llir_id)
}

fn lower_linear_index_loop(
    builder: &mut BlockLowerer<'_>,
    prefix: &str,
    trip_count: i64,
    mut body: impl FnMut(&mut LowerLlirCtx, &MirFunction, &mut Vec<llir::Inst>, llir::Operand),
) {
    let (header, header_params) = builder.create_block(prefix, vec![("loop_idx", llir::Type::I64)]);
    let (body_block, body_params) = builder.create_block(
        &format!("{prefix}_body"),
        vec![("loop_idx", llir::Type::I64)],
    );
    let (continue_block, _) = builder.create_block(&format!("{prefix}_continue"), vec![]);

    builder.set_current_terminator(llir::Terminator::Br {
        target: header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(header);
    let header_idx = llir::Operand::Value(header_params[0].id);
    let header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            header_idx.clone(),
            const_i64(trip_count),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: vec![header_idx.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(header_term);

    builder.switch_to(body_block);
    let body_idx = llir::Operand::Value(body_params[0].id);
    let body_term = builder.with_current_insts(|ctx, mir, out| {
        body(ctx, mir, out, body_idx.clone());
        let next_idx = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_idx.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: header,
            args: vec![next_idx],
        }
    });
    builder.set_current_terminator(body_term);
    builder.switch_to(continue_block);
}

fn linear_index_to_coords(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    idx: llir::Operand,
    cols: i64,
) -> (llir::Operand, llir::Operand) {
    let row = emit_bin(
        ctx,
        out,
        llir::BinOp::Div,
        idx.clone(),
        const_i64(cols),
        llir::Type::I64,
    );
    let row_base = emit_bin(
        ctx,
        out,
        llir::BinOp::Mul,
        row.clone(),
        const_i64(cols),
        llir::Type::I64,
    );
    let col = emit_bin(ctx, out, llir::BinOp::Sub, idx, row_base, llir::Type::I64);
    (row, col)
}

fn lower_tile_constant_inst(
    result: ValueId,
    value: f64,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    lower_linear_index_loop(
        builder,
        "tile_const_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (row, col) = linear_index_to_coords(ctx, out, idx, cols);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, const_f32(value));
        },
    );
}

fn lower_tile_load_inst(
    result: ValueId,
    buf: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let buf_operand = resolve_operand(buf, builder.ctx);
    let row_operand = resolve_operand(row_coord, builder.ctx);
    let col_operand = resolve_operand(col_coord, builder.ctx);
    let rank = buffer_rank_of(buf, builder.mir);

    let (tile_base, row_base, col_base, stride) = builder.with_current_insts(|ctx, _, out| {
        if rank <= 1 {
            let tile_coord =
                lower_1d_tile_coord(ctx, out, row_operand.clone(), col_operand.clone());
            let base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                tile_coord,
                const_i64(rows * cols),
                llir::Type::I64,
            );
            (Some(base), None, None, None)
        } else {
            let row_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                row_operand.clone(),
                const_i64(rows),
                llir::Type::I64,
            );
            let col_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                col_operand.clone(),
                const_i64(cols),
                llir::Type::I64,
            );
            let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
            (None, Some(row_base), Some(col_base), Some(stride))
        }
    });

    lower_linear_index_loop(
        builder,
        "tile_load_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (local_row, local_col) = linear_index_to_coords(ctx, out, idx.clone(), cols);
            let linear_index = if let Some(tile_base) = tile_base.clone() {
                emit_bin(ctx, out, llir::BinOp::Add, tile_base, idx, llir::Type::I64)
            } else {
                let src_row = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_base.clone().expect("row base"),
                    local_row.clone(),
                    llir::Type::I64,
                );
                let src_col = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    col_base.clone().expect("col base"),
                    local_col.clone(),
                    llir::Type::I64,
                );
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    src_row,
                    stride.clone().expect("stride"),
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
                vec![local_row, local_col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, loaded);
        },
    );
}

fn lower_tile_store_inst(
    buf: ValueId,
    value: ValueId,
    row_coord: ValueId,
    col_coord: ValueId,
    rows: i64,
    cols: i64,
    stride_shape_idx: usize,
    builder: &mut BlockLowerer<'_>,
) {
    let buf_operand = resolve_operand(buf, builder.ctx);
    let value_operand = resolve_operand(value, builder.ctx);
    let row_operand = resolve_operand(row_coord, builder.ctx);
    let col_operand = resolve_operand(col_coord, builder.ctx);
    let rank = buffer_rank_of(buf, builder.mir);

    let (tile_base, row_base, col_base, stride) = builder.with_current_insts(|ctx, _, out| {
        if rank <= 1 {
            let tile_coord =
                lower_1d_tile_coord(ctx, out, row_operand.clone(), col_operand.clone());
            let base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                tile_coord,
                const_i64(rows * cols),
                llir::Type::I64,
            );
            (Some(base), None, None, None)
        } else {
            let row_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                row_operand.clone(),
                const_i64(rows),
                llir::Type::I64,
            );
            let col_base = emit_bin(
                ctx,
                out,
                llir::BinOp::Mul,
                col_operand.clone(),
                const_i64(cols),
                llir::Type::I64,
            );
            let stride = emit_shape_dim(ctx, out, buf_operand.clone(), stride_shape_idx);
            (None, Some(row_base), Some(col_base), Some(stride))
        }
    });

    lower_linear_index_loop(
        builder,
        "tile_store_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (local_row, local_col) = linear_index_to_coords(ctx, out, idx.clone(), cols);
            let src_ptr = emit_gep(
                ctx,
                out,
                value_operand.clone(),
                vec![local_row.clone(), local_col.clone()],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let scalar = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let linear_index = if let Some(tile_base) = tile_base.clone() {
                emit_bin(ctx, out, llir::BinOp::Add, tile_base, idx, llir::Type::I64)
            } else {
                let dst_row = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    row_base.clone().expect("row base"),
                    local_row,
                    llir::Type::I64,
                );
                let dst_col = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Add,
                    col_base.clone().expect("col base"),
                    local_col,
                    llir::Type::I64,
                );
                let row_offset = emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Mul,
                    dst_row,
                    stride.clone().expect("stride"),
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
        },
    );
}

fn lower_tile_binary_inst(
    result: ValueId,
    lhs: ValueId,
    rhs: ValueId,
    op: llir::BinOp,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let lhs_tile = resolve_operand(lhs, builder.ctx);
    let rhs_tile = resolve_operand(rhs, builder.ctx);
    lower_linear_index_loop(
        builder,
        "tile_binary_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (row, col) = linear_index_to_coords(ctx, out, idx, cols);
            let lhs =
                load_tile_scalar_dynamic(ctx, out, lhs_tile.clone(), row.clone(), col.clone());
            let rhs =
                load_tile_scalar_dynamic(ctx, out, rhs_tile.clone(), row.clone(), col.clone());
            let result = emit_bin(ctx, out, op, lhs, rhs, llir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, result);
        },
    );
}

fn lower_tile_unary_inst(
    result: ValueId,
    operand: ValueId,
    op: UnaryOp,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let src_tile = resolve_operand(operand, builder.ctx);
    lower_linear_index_loop(
        builder,
        "tile_unary_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (row, col) = linear_index_to_coords(ctx, out, idx, cols);
            let src =
                load_tile_scalar_dynamic(ctx, out, src_tile.clone(), row.clone(), col.clone());
            let result = match op {
                UnaryOp::Neg => emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Sub,
                    const_f32(0.0),
                    src,
                    llir::Type::F32,
                ),
                UnaryOp::Exp => {
                    emit_intrinsic(ctx, out, llir::Intrinsic::Exp, vec![src], llir::Type::F32)
                }
            };
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, result);
        },
    );
}

fn lower_tile_broadcast_inst(
    result: ValueId,
    value: ValueId,
    rows: i64,
    cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, rows, cols);
    let src_tile = resolve_operand(value, builder.ctx);
    let (src_rows, src_cols) = tile_dims_of(value, builder.mir).unwrap_or((1, 1));
    lower_linear_index_loop(
        builder,
        "tile_broadcast_loop",
        rows * cols,
        move |ctx, _, out, idx| {
            let (row, col) = linear_index_to_coords(ctx, out, idx, cols);
            let src_row = if src_rows == 1 {
                const_i64(0)
            } else {
                row.clone()
            };
            let src_col = if src_cols == 1 {
                const_i64(0)
            } else {
                col.clone()
            };
            let scalar = load_tile_scalar_dynamic(ctx, out, src_tile.clone(), src_row, src_col);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![row, col],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        },
    );
}

fn lower_tile_reduce_inst(
    result: ValueId,
    value: ValueId,
    op: ReduceOp,
    axis: i64,
    in_rows: i64,
    in_cols: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let (out_rows, out_cols, reduce_extent) = if axis == 1 {
        (in_rows, 1, in_cols)
    } else {
        (1, in_cols, in_rows)
    };
    let dst_tile = alloc_tile_result(builder, result, out_rows, out_cols);
    let src_tile = resolve_operand(value, builder.ctx);

    let (outer_header, outer_params) = builder.create_block(
        "tile_reduce_outer_header",
        vec![("reduce_outer", llir::Type::I64)],
    );
    let (outer_body, outer_body_params) = builder.create_block(
        "tile_reduce_outer_body",
        vec![("reduce_outer", llir::Type::I64)],
    );
    let (inner_header, inner_header_params) = builder.create_block(
        "tile_reduce_inner_header",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_idx", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (inner_body, inner_body_params) = builder.create_block(
        "tile_reduce_inner_body",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_idx", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (inner_exit, inner_exit_params) = builder.create_block(
        "tile_reduce_inner_exit",
        vec![
            ("reduce_outer", llir::Type::I64),
            ("reduce_acc", llir::Type::F32),
        ],
    );
    let (continue_block, _) = builder.create_block("tile_reduce_continue", vec![]);

    builder.set_current_terminator(llir::Terminator::Br {
        target: outer_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(outer_header);
    let outer_idx = llir::Operand::Value(outer_params[0].id);
    let outer_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            outer_idx.clone(),
            const_i64(out_rows * out_cols),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: outer_body,
            true_args: vec![outer_idx.clone()],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(outer_term);

    builder.switch_to(outer_body);
    let outer_body_idx = llir::Operand::Value(outer_body_params[0].id);
    let outer_body_term = builder.with_current_insts(|ctx, _, out| {
        let (out_row, out_col) = if axis == 1 {
            (outer_body_idx.clone(), const_i64(0))
        } else {
            (const_i64(0), outer_body_idx.clone())
        };
        let init_acc = match op {
            ReduceOp::Sum => const_f32(0.0),
            ReduceOp::Max => {
                let src_row = if axis == 1 {
                    out_row.clone()
                } else {
                    const_i64(0)
                };
                let src_col = if axis == 1 {
                    const_i64(0)
                } else {
                    out_col.clone()
                };
                load_tile_scalar_dynamic(ctx, out, src_tile.clone(), src_row, src_col)
            }
        };
        let start_idx = if matches!(op, ReduceOp::Max) { 1 } else { 0 };
        let _ = (out_row, out_col);
        llir::Terminator::Br {
            target: inner_header,
            args: vec![outer_body_idx.clone(), const_i64(start_idx), init_acc],
        }
    });
    builder.set_current_terminator(outer_body_term);

    builder.switch_to(inner_header);
    let inner_outer = llir::Operand::Value(inner_header_params[0].id);
    let inner_idx = llir::Operand::Value(inner_header_params[1].id);
    let inner_acc = llir::Operand::Value(inner_header_params[2].id);
    let inner_header_term = builder.with_current_insts(|ctx, _, out| {
        let cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            inner_idx.clone(),
            const_i64(reduce_extent),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond,
            true_target: inner_body,
            true_args: vec![inner_outer.clone(), inner_idx.clone(), inner_acc.clone()],
            false_target: inner_exit,
            false_args: vec![inner_outer.clone(), inner_acc.clone()],
        }
    });
    builder.set_current_terminator(inner_header_term);

    builder.switch_to(inner_body);
    let body_outer = llir::Operand::Value(inner_body_params[0].id);
    let body_idx = llir::Operand::Value(inner_body_params[1].id);
    let body_acc = llir::Operand::Value(inner_body_params[2].id);
    let inner_body_term = builder.with_current_insts(|ctx, _, out| {
        let (src_row, src_col) = if axis == 1 {
            (body_outer.clone(), body_idx.clone())
        } else {
            (body_idx.clone(), body_outer.clone())
        };
        let value = load_tile_scalar_dynamic(ctx, out, src_tile.clone(), src_row, src_col);
        let next_acc = match op {
            ReduceOp::Sum => emit_bin(
                ctx,
                out,
                llir::BinOp::Add,
                body_acc.clone(),
                value,
                llir::Type::F32,
            ),
            ReduceOp::Max => emit_max(ctx, out, body_acc.clone(), value),
        };
        let next_idx = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_idx.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: inner_header,
            args: vec![body_outer.clone(), next_idx, next_acc],
        }
    });
    builder.set_current_terminator(inner_body_term);

    builder.switch_to(inner_exit);
    let exit_outer = llir::Operand::Value(inner_exit_params[0].id);
    let exit_acc = llir::Operand::Value(inner_exit_params[1].id);
    let inner_exit_term = builder.with_current_insts(|ctx, _, out| {
        let (dst_row, dst_col) = if axis == 1 {
            (exit_outer.clone(), const_i64(0))
        } else {
            (const_i64(0), exit_outer.clone())
        };
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![dst_row, dst_col],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        emit_store(out, dst_ptr, exit_acc.clone());
        let next_outer = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            exit_outer.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: outer_header,
            args: vec![next_outer],
        }
    });
    builder.set_current_terminator(inner_exit_term);
    builder.switch_to(continue_block);
}

fn lower_tile_mma_inst(
    result: ValueId,
    a: ValueId,
    b: ValueId,
    acc: ValueId,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
    builder: &mut BlockLowerer<'_>,
) {
    let dst_tile = alloc_tile_result(builder, result, tile_m, tile_n);

    lower_tile_mma_loop(
        builder,
        dst_tile,
        resolve_operand(a, builder.ctx),
        resolve_operand(b, builder.ctx),
        resolve_operand(acc, builder.ctx),
        tile_m,
        tile_n,
        tile_k,
    );
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

fn next_llir_block(mir: &MirFunction) -> u32 {
    mir.blocks.iter().map(|block| block.id.0).max().unwrap_or(0) + 1
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

fn tile_dims_of(value: ValueId, mir: &MirFunction) -> Option<(i64, i64)> {
    match mir.types.get(&value) {
        Some(MirType::Tile { rows, cols }) => Some((*rows, *cols)),
        _ => None,
    }
}

fn lower_tile_constant(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    value: f64,
    rows: i64,
    cols: i64,
) {
    for row in 0..rows {
        for col in 0..cols {
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, const_f32(value));
        }
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

fn lower_tile_binary(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    lhs_tile: llir::Operand,
    rhs_tile: llir::Operand,
    op: llir::BinOp,
    rows: i64,
    cols: i64,
) {
    for row in 0..rows {
        for col in 0..cols {
            let lhs_ptr = emit_gep(
                ctx,
                out,
                lhs_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let lhs = emit_load(ctx, out, lhs_ptr, llir::Type::F32);
            let rhs_ptr = emit_gep(
                ctx,
                out,
                rhs_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let rhs = emit_load(ctx, out, rhs_ptr, llir::Type::F32);
            let result = emit_bin(ctx, out, op, lhs, rhs, llir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, result);
        }
    }
}

fn lower_tile_unary(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    src_tile: llir::Operand,
    op: UnaryOp,
    rows: i64,
    cols: i64,
) {
    for row in 0..rows {
        for col in 0..cols {
            let src_ptr = emit_gep(
                ctx,
                out,
                src_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let src = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let result = match op {
                UnaryOp::Neg => emit_bin(
                    ctx,
                    out,
                    llir::BinOp::Sub,
                    const_f32(0.0),
                    src,
                    llir::Type::F32,
                ),
                UnaryOp::Exp => {
                    emit_intrinsic(ctx, out, llir::Intrinsic::Exp, vec![src], llir::Type::F32)
                }
            };
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, result);
        }
    }
}

fn lower_tile_broadcast(
    mir: &MirFunction,
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    src: ValueId,
    rows: i64,
    cols: i64,
) {
    let src_tile = resolve_operand(src, ctx);
    let (src_rows, src_cols) = tile_dims_of(src, mir).unwrap_or((1, 1));
    for row in 0..rows {
        for col in 0..cols {
            let src_row = if src_rows == 1 { 0 } else { row };
            let src_col = if src_cols == 1 { 0 } else { col };
            let src_ptr = emit_gep(
                ctx,
                out,
                src_tile.clone(),
                vec![const_i64(src_row), const_i64(src_col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            let scalar = emit_load(ctx, out, src_ptr, llir::Type::F32);
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, scalar);
        }
    }
}

fn lower_tile_reduce(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    src_tile: llir::Operand,
    op: ReduceOp,
    axis: i64,
    in_rows: i64,
    in_cols: i64,
) {
    match axis {
        1 => {
            for row in 0..in_rows {
                let reduced = lower_reduce_accumulate(ctx, out, src_tile.clone(), op, row, in_cols);
                let dst_ptr = emit_gep(
                    ctx,
                    out,
                    dst_tile.clone(),
                    vec![const_i64(row), const_i64(0)],
                    llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
                );
                emit_store(out, dst_ptr, reduced);
            }
        }
        0 => {
            for col in 0..in_cols {
                let reduced =
                    lower_reduce_accumulate_col(ctx, out, src_tile.clone(), op, col, in_rows);
                let dst_ptr = emit_gep(
                    ctx,
                    out,
                    dst_tile.clone(),
                    vec![const_i64(0), const_i64(col)],
                    llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
                );
                emit_store(out, dst_ptr, reduced);
            }
        }
        _ => {}
    }
}

fn lower_tile_mma(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    dst_tile: llir::Operand,
    a_tile: llir::Operand,
    b_tile: llir::Operand,
    acc_tile: llir::Operand,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
) {
    for row in 0..tile_m {
        for col in 0..tile_n {
            let mut acc = load_tile_scalar(ctx, out, acc_tile.clone(), row, col);
            for k in 0..tile_k {
                let a = load_tile_scalar(ctx, out, a_tile.clone(), row, k);
                let b = load_tile_scalar(ctx, out, b_tile.clone(), k, col);
                let product = emit_bin(ctx, out, llir::BinOp::Mul, a, b, llir::Type::F32);
                acc = emit_bin(ctx, out, llir::BinOp::Add, acc, product, llir::Type::F32);
            }
            let dst_ptr = emit_gep(
                ctx,
                out,
                dst_tile.clone(),
                vec![const_i64(row), const_i64(col)],
                llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
            );
            emit_store(out, dst_ptr, acc);
        }
    }
}

fn lower_tile_mma_loop(
    builder: &mut BlockLowerer<'_>,
    dst_tile: llir::Operand,
    a_tile: llir::Operand,
    b_tile: llir::Operand,
    acc_tile: llir::Operand,
    tile_m: i64,
    tile_n: i64,
    tile_k: i64,
) {
    let (row_header, row_params) =
        builder.create_block("mma_row_header", vec![("mma_row", llir::Type::I64)]);
    let (col_header, col_params) = builder.create_block(
        "mma_col_header",
        vec![("mma_row", llir::Type::I64), ("mma_col", llir::Type::I64)],
    );
    let (k_preheader, k_pre_params) = builder.create_block(
        "mma_k_preheader",
        vec![("mma_row", llir::Type::I64), ("mma_col", llir::Type::I64)],
    );
    let (k_header, k_header_params) = builder.create_block(
        "mma_k_header",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_k", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (k_body, k_body_params) = builder.create_block(
        "mma_k_body",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_k", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (k_exit, k_exit_params) = builder.create_block(
        "mma_k_exit",
        vec![
            ("mma_row", llir::Type::I64),
            ("mma_col", llir::Type::I64),
            ("mma_acc", llir::Type::F32),
        ],
    );
    let (row_latch, row_latch_params) =
        builder.create_block("mma_row_latch", vec![("mma_row", llir::Type::I64)]);
    let (continue_block, _) = builder.create_block("mma_continue", vec![]);

    let row = llir::Operand::Value(row_params[0].id);
    let col = llir::Operand::Value(col_params[1].id);
    let col_row = llir::Operand::Value(col_params[0].id);
    let pre_row = llir::Operand::Value(k_pre_params[0].id);
    let pre_col = llir::Operand::Value(k_pre_params[1].id);
    let k_row = llir::Operand::Value(k_header_params[0].id);
    let k_col = llir::Operand::Value(k_header_params[1].id);
    let k_idx = llir::Operand::Value(k_header_params[2].id);
    let k_acc = llir::Operand::Value(k_header_params[3].id);
    let body_row = llir::Operand::Value(k_body_params[0].id);
    let body_col = llir::Operand::Value(k_body_params[1].id);
    let body_k = llir::Operand::Value(k_body_params[2].id);
    let body_acc = llir::Operand::Value(k_body_params[3].id);
    let exit_row = llir::Operand::Value(k_exit_params[0].id);
    let exit_col = llir::Operand::Value(k_exit_params[1].id);
    let exit_acc = llir::Operand::Value(k_exit_params[2].id);
    let latch_row = llir::Operand::Value(row_latch_params[0].id);

    builder.set_current_terminator(llir::Terminator::Br {
        target: row_header,
        args: vec![const_i64(0)],
    });

    builder.switch_to(row_header);
    let row_term = builder.with_current_insts(|ctx, _, out| {
        let row_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            row.clone(),
            const_i64(tile_m),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: row_cond,
            true_target: col_header,
            true_args: vec![row.clone(), const_i64(0)],
            false_target: continue_block,
            false_args: vec![],
        }
    });
    builder.set_current_terminator(row_term);

    builder.switch_to(col_header);
    let col_term = builder.with_current_insts(|ctx, _, out| {
        let col_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            col.clone(),
            const_i64(tile_n),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: col_cond,
            true_target: k_preheader,
            true_args: vec![col_row.clone(), col.clone()],
            false_target: row_latch,
            false_args: vec![col_row.clone()],
        }
    });
    builder.set_current_terminator(col_term);

    builder.switch_to(k_preheader);
    let k_pre_term = builder.with_current_insts(|ctx, _, out| {
        let acc_init =
            load_tile_scalar_dynamic(ctx, out, acc_tile.clone(), pre_row.clone(), pre_col.clone());
        llir::Terminator::Br {
            target: k_header,
            args: vec![pre_row.clone(), pre_col.clone(), const_i64(0), acc_init],
        }
    });
    builder.set_current_terminator(k_pre_term);

    builder.switch_to(k_header);
    let k_header_term = builder.with_current_insts(|ctx, _, out| {
        let k_cond = emit_cmp(
            ctx,
            out,
            llir::CmpPred::Slt,
            k_idx.clone(),
            const_i64(tile_k),
            llir::Type::I1,
        );
        llir::Terminator::CondBr {
            cond: k_cond,
            true_target: k_body,
            true_args: vec![k_row.clone(), k_col.clone(), k_idx.clone(), k_acc.clone()],
            false_target: k_exit,
            false_args: vec![k_row.clone(), k_col.clone(), k_acc.clone()],
        }
    });
    builder.set_current_terminator(k_header_term);

    builder.switch_to(k_body);
    let k_body_term = builder.with_current_insts(|ctx, _, out| {
        let a =
            load_tile_scalar_dynamic(ctx, out, a_tile.clone(), body_row.clone(), body_k.clone());
        let b =
            load_tile_scalar_dynamic(ctx, out, b_tile.clone(), body_k.clone(), body_col.clone());
        let product = emit_bin(ctx, out, llir::BinOp::Mul, a, b, llir::Type::F32);
        let next_acc = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_acc.clone(),
            product,
            llir::Type::F32,
        );
        let next_k = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            body_k.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: k_header,
            args: vec![body_row.clone(), body_col.clone(), next_k, next_acc],
        }
    });
    builder.set_current_terminator(k_body_term);

    builder.switch_to(k_exit);
    let k_exit_term = builder.with_current_insts(|ctx, _, out| {
        let dst_ptr = emit_gep(
            ctx,
            out,
            dst_tile.clone(),
            vec![exit_row.clone(), exit_col.clone()],
            llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
        );
        emit_store(out, dst_ptr, exit_acc.clone());
        let next_col = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            exit_col.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: col_header,
            args: vec![exit_row.clone(), next_col],
        }
    });
    builder.set_current_terminator(k_exit_term);

    builder.switch_to(row_latch);
    let row_latch_term = builder.with_current_insts(|ctx, _, out| {
        let next_row = emit_bin(
            ctx,
            out,
            llir::BinOp::Add,
            latch_row.clone(),
            const_i64(1),
            llir::Type::I64,
        );
        llir::Terminator::Br {
            target: row_header,
            args: vec![next_row],
        }
    });
    builder.set_current_terminator(row_latch_term);

    builder.switch_to(continue_block);
}

fn lower_reduce_accumulate(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    src_tile: llir::Operand,
    op: ReduceOp,
    row: i64,
    cols: i64,
) -> llir::Operand {
    let mut acc = match op {
        ReduceOp::Sum => const_f32(0.0),
        ReduceOp::Max => load_tile_scalar(ctx, out, src_tile.clone(), row, 0),
    };
    let start_col = if matches!(op, ReduceOp::Max) { 1 } else { 0 };
    for col in start_col..cols {
        let value = load_tile_scalar(ctx, out, src_tile.clone(), row, col);
        acc = match op {
            ReduceOp::Sum => emit_bin(ctx, out, llir::BinOp::Add, acc, value, llir::Type::F32),
            ReduceOp::Max => emit_max(ctx, out, acc, value),
        };
    }
    acc
}

fn lower_reduce_accumulate_col(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    src_tile: llir::Operand,
    op: ReduceOp,
    col: i64,
    rows: i64,
) -> llir::Operand {
    let mut acc = match op {
        ReduceOp::Sum => const_f32(0.0),
        ReduceOp::Max => load_tile_scalar(ctx, out, src_tile.clone(), 0, col),
    };
    let start_row = if matches!(op, ReduceOp::Max) { 1 } else { 0 };
    for row in start_row..rows {
        let value = load_tile_scalar(ctx, out, src_tile.clone(), row, col);
        acc = match op {
            ReduceOp::Sum => emit_bin(ctx, out, llir::BinOp::Add, acc, value, llir::Type::F32),
            ReduceOp::Max => emit_max(ctx, out, acc, value),
        };
    }
    acc
}

fn load_tile_scalar(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    tile: llir::Operand,
    row: i64,
    col: i64,
) -> llir::Operand {
    let ptr = emit_gep(
        ctx,
        out,
        tile,
        vec![const_i64(row), const_i64(col)],
        llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
    );
    emit_load(ctx, out, ptr, llir::Type::F32)
}

fn load_tile_scalar_dynamic(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    tile: llir::Operand,
    row: llir::Operand,
    col: llir::Operand,
) -> llir::Operand {
    let ptr = emit_gep(
        ctx,
        out,
        tile,
        vec![row, col],
        llir::Type::ptr(llir::AddressSpace::Private, llir::Type::F32),
    );
    emit_load(ctx, out, ptr, llir::Type::F32)
}

fn emit_max(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    lhs: llir::Operand,
    rhs: llir::Operand,
) -> llir::Operand {
    let cond = emit_cmp(
        ctx,
        out,
        llir::CmpPred::Ogt,
        rhs.clone(),
        lhs.clone(),
        llir::Type::I1,
    );
    emit_select(ctx, out, cond, rhs, lhs, llir::Type::F32)
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

fn emit_intrinsic(
    ctx: &mut LowerLlirCtx,
    out: &mut Vec<llir::Inst>,
    intrinsic: llir::Intrinsic,
    args: Vec<llir::Operand>,
    ty: llir::Type,
) -> llir::Operand {
    let (id, name) = ctx.fresh_value("v");
    out.push(llir::Inst {
        result: Some(id),
        result_name: Some(name),
        ty,
        op: llir::InstOp::Intrinsic { intrinsic, args },
        metadata: Vec::new(),
    });
    llir::Operand::Value(id)
}

fn const_i64(value: i64) -> llir::Operand {
    llir::Operand::Const(llir::Constant::Int(value))
}

fn const_f32(value: f64) -> llir::Operand {
    llir::Operand::Const(llir::Constant::Float(value))
}
