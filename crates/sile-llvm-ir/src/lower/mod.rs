mod block;
mod block_scalar;
mod block_terminator;
mod canonical_matmul;
mod core;
mod tile_compute;
mod tile_expr;
mod op_lowering;
mod tile_memory;

use crate as llvm_ir;
use sile_tile_ir::{TileIrFunction, TileIrOp, TileIrParamKind, TileIrType, ValueId};

use std::collections::HashMap;

use self::block::lower_block;
use self::canonical_matmul::lower_canonical_matmul;
use self::core::{LowerLlvmIrCtx, llvm_ir_block, llvm_ir_type, llvm_ir_value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmIrLoweringPassKind {
    LowerParams,
    LowerBlocks,
    FinalizeFunction,
}

pub const RECOMMENDED_LLVM_IR_LOWERING_PIPELINE: &[LlvmIrLoweringPassKind] = &[
    LlvmIrLoweringPassKind::LowerParams,
    LlvmIrLoweringPassKind::LowerBlocks,
    LlvmIrLoweringPassKind::FinalizeFunction,
];

pub const ACTIVE_LLVM_IR_LOWERING_PIPELINE: &[LlvmIrLoweringPassKind] =
    RECOMMENDED_LLVM_IR_LOWERING_PIPELINE;

const SHAPES_PARAM_NAME: &str = llvm_ir::SHAPES_PARAM_NAME;
const SCALARS_PARAM_NAME: &str = llvm_ir::SCALARS_PARAM_NAME;
const SCALARS_I64_PARAM_NAME: &str = llvm_ir::SCALARS_I64_PARAM_NAME;

struct LlvmIrLoweringState<'a> {
    tile_ir: &'a TileIrFunction,
    ctx: Option<LowerLlvmIrCtx>,
    params: Option<Vec<llvm_ir::Param>>,
    blocks: Option<Vec<llvm_ir::BasicBlock>>,
    output: Option<llvm_ir::Function>,
}

impl<'a> LlvmIrLoweringState<'a> {
    fn new(tile_ir: &'a TileIrFunction) -> Self {
        Self {
            tile_ir,
            ctx: None,
            params: None,
            blocks: None,
            output: None,
        }
    }
}

pub fn run_llvm_ir_lowering_pipeline(
    tile_ir: &TileIrFunction,
    pipeline: &[LlvmIrLoweringPassKind],
) -> Result<llvm_ir::Function, String> {
    run_llvm_ir_lowering_pipeline_inner(LlvmIrLoweringState::new(tile_ir), pipeline)
}

pub fn run_default_llvm_ir_lowering_pipeline(
    tile_ir: &TileIrFunction,
) -> Result<llvm_ir::Function, String> {
    run_llvm_ir_lowering_pipeline(tile_ir, ACTIVE_LLVM_IR_LOWERING_PIPELINE)
}

pub fn lower_tile_ir_to_llvm_ir(tile_ir: &TileIrFunction) -> llvm_ir::Function {
    run_default_llvm_ir_lowering_pipeline(tile_ir)
        .expect("default Tile IR->LLVM IR lowering pipeline must produce LLVM IR function")
}

fn run_llvm_ir_lowering_pipeline_inner(
    mut state: LlvmIrLoweringState<'_>,
    pipeline: &[LlvmIrLoweringPassKind],
) -> Result<llvm_ir::Function, String> {
    for pass in pipeline {
        match pass {
            LlvmIrLoweringPassKind::LowerParams => {
                let mut ctx = LowerLlvmIrCtx::new(state.tile_ir);
                let mut params = state
                    .tile_ir
                    .params
                    .iter()
                    .filter_map(|param| match &param.kind {
                        TileIrParamKind::Buffer => {
                            let id = llvm_ir_value(param.value);
                            ctx.names.insert(id, param.name.clone());
                            ctx.operands
                                .insert(param.value, llvm_ir::Operand::Value(id));
                            // The shape ABI was recorded when HIR lowered to
                            // Tile IR, so the buffer parameters line up with
                            // their `__sile_shapes` offsets without the HIR.
                            let abi = param.abi.as_ref().map(|abi| llvm_ir::ParamAbi {
                                rank: abi.rank,
                                shape_offset: abi.shape_offset,
                                tile: None,
                            });
                            if let Some(abi) = &abi {
                                ctx.shape_offsets.insert(id, abi.shape_offset);
                            }
                            Some(llvm_ir::Param {
                                id,
                                name: param.name.clone(),
                                ty: llvm_ir_type(&param.ty),
                                abi,
                            })
                        }
                        TileIrParamKind::LaunchIndex { .. }
                        | TileIrParamKind::ShapeDesc { .. }
                        | TileIrParamKind::Scalar { .. } => None,
                    })
                    .collect::<Vec<_>>();
                let (shapes_id, _) = ctx.fresh_value("arg");
                ctx.names.insert(shapes_id, SHAPES_PARAM_NAME.into());
                let shapes_operand = llvm_ir::Operand::Value(shapes_id);
                ctx.shapes_param = Some(shapes_operand);
                params.push(llvm_ir::Param {
                    id: shapes_id,
                    name: SHAPES_PARAM_NAME.into(),
                    ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                    abi: None,
                });
                let (scalars_id, _) = ctx.fresh_value("arg");
                ctx.names.insert(scalars_id, SCALARS_PARAM_NAME.into());
                ctx.scalars_param = Some(llvm_ir::Operand::Value(scalars_id));
                params.push(llvm_ir::Param {
                    id: scalars_id,
                    name: SCALARS_PARAM_NAME.into(),
                    ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::F32),
                    abi: None,
                });
                // The integer scalars travel in a descriptor of their own, and
                // only a kernel that takes one declares it. A signature that
                // carries no integer scalar is therefore unchanged.
                let has_i64_scalar = state.tile_ir.params.iter().any(|param| {
                    matches!(param.kind, TileIrParamKind::Scalar { .. })
                        && matches!(param.ty, TileIrType::I64)
                });
                if has_i64_scalar {
                    let (scalars_i64_id, _) = ctx.fresh_value("arg");
                    ctx.names
                        .insert(scalars_i64_id, SCALARS_I64_PARAM_NAME.into());
                    ctx.scalars_i64_param = Some(llvm_ir::Operand::Value(scalars_i64_id));
                    params.push(llvm_ir::Param {
                        id: scalars_i64_id,
                        name: SCALARS_I64_PARAM_NAME.into(),
                        ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                        abi: None,
                    });
                }
                state.ctx = Some(ctx);
                state.params = Some(params);
            }
            LlvmIrLoweringPassKind::LowerBlocks => {
                let ctx = state.ctx.as_mut().ok_or_else(|| {
                    "LowerBlocks requires LowerParams earlier in Tile IR->LLVM IR pipeline"
                        .to_string()
                })?;
                let mut blocks = Vec::new();
                if let Some(prologue) = build_prologue_block(state.tile_ir, ctx) {
                    blocks.push(prologue);
                }
                if let Some(matmul_blocks) = lower_canonical_matmul(state.tile_ir, ctx) {
                    blocks.extend(matmul_blocks);
                } else {
                    blocks.extend(
                        state
                            .tile_ir
                            .blocks
                            .iter()
                            .flat_map(|block| lower_block(block, state.tile_ir, ctx)),
                    );
                }
                // A lowering that met an op it could not express has already
                // written part of the block. Reporting here is what keeps that
                // op from vanishing into a program that computes less than the
                // kernel asked for.
                if let Some(why) = ctx.unhandled_op.take() {
                    return Err(why);
                }
                state.blocks = Some(blocks);
            }
            LlvmIrLoweringPassKind::FinalizeFunction => {
                let mut params = state.params.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerParams earlier in Tile IR->LLVM IR pipeline"
                        .to_string()
                })?;
                let blocks = state.blocks.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerBlocks earlier in Tile IR->LLVM IR pipeline"
                        .to_string()
                })?;
                attach_tile_abi(&mut params, state.tile_ir);
                state.output = Some(llvm_ir::Function {
                    name: state.tile_ir.name.clone(),
                    params,
                    blocks,
                    entry: state
                        .ctx
                        .as_ref()
                        .and_then(|ctx| ctx.prologue_entry)
                        .unwrap_or_else(|| llvm_ir_block(state.tile_ir.entry)),
                    metadata: Vec::new(),
                });
            }
        }
    }

    state.output.ok_or_else(|| {
        "Tile IR->LLVM IR lowering pipeline did not produce LLVM IR function".to_string()
    })
}

fn build_prologue_block(
    tile_ir: &TileIrFunction,
    ctx: &mut LowerLlvmIrCtx,
) -> Option<llvm_ir::BasicBlock> {
    let synthetic = tile_ir
        .params
        .iter()
        .filter(|param| !matches!(param.kind, TileIrParamKind::Buffer))
        .collect::<Vec<_>>();
    if synthetic.is_empty() {
        return None;
    }

    let prologue_id = ctx.fresh_block_id();
    ctx.prologue_entry = Some(prologue_id);
    let mut insts = Vec::new();

    for param in synthetic {
        let result_id = llvm_ir_value(param.value);
        ctx.names.insert(result_id, param.name.clone());
        ctx.operands
            .insert(param.value, llvm_ir::Operand::Value(result_id));
        match &param.kind {
            TileIrParamKind::LaunchIndex { dim } => {
                insts.push(llvm_ir::Inst {
                    result: Some(result_id),
                    result_name: Some(param.name.clone()),
                    ty: llvm_ir::Type::I64,
                    op: llvm_ir::InstOp::Intrinsic {
                        intrinsic: llvm_ir::Intrinsic::BlockId { dim: *dim as u8 },
                        args: Vec::new(),
                    },
                    metadata: Vec::new(),
                });
            }
            TileIrParamKind::ShapeDesc { source } => {
                let source_operand = ctx
                    .operands
                    .get(source)
                    .cloned()
                    .expect("shape param source buffer must be lowered before prologue");
                let llvm_ir::Operand::Value(source_id) = source_operand else {
                    panic!("shape param source buffer must be an LLVM value");
                };
                let shape_offset = ctx
                    .shape_offsets
                    .get(&source_id)
                    .copied()
                    .expect("shape param source buffer must have ABI shape offset");
                let shapes_operand = ctx
                    .shapes_param
                    .clone()
                    .expect("shape param prologue requires explicit __sile_shapes parameter");
                insts.push(llvm_ir::Inst {
                    result: Some(result_id),
                    result_name: Some(param.name.clone()),
                    ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                    op: llvm_ir::InstOp::Gep {
                        base: shapes_operand,
                        indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(
                            shape_offset as i64,
                        ))],
                    },
                    metadata: Vec::new(),
                });
                ctx.buffer_shape_descs.insert(source_id, result_id);
                let rank = match param.ty {
                    sile_tile_ir::TileIrType::ShapeDesc { rank } => rank,
                    _ => 0,
                };
                for dim in 0..rank {
                    let gep_id = {
                        let (id, name) = ctx.fresh_value("v");
                        insts.push(llvm_ir::Inst {
                            result: Some(id),
                            result_name: Some(name),
                            ty: llvm_ir::Type::ptr(
                                llvm_ir::AddressSpace::Constant,
                                llvm_ir::Type::I64,
                            ),
                            op: llvm_ir::InstOp::Gep {
                                base: llvm_ir::Operand::Value(result_id),
                                indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(
                                    dim as i64,
                                ))],
                            },
                            metadata: Vec::new(),
                        });
                        id
                    };
                    let dim_value = {
                        let (id, name) = ctx.fresh_value("v");
                        insts.push(llvm_ir::Inst {
                            result: Some(id),
                            result_name: Some(name),
                            ty: llvm_ir::Type::I64,
                            op: llvm_ir::InstOp::Load {
                                ptr: llvm_ir::Operand::Value(gep_id),
                            },
                            metadata: Vec::new(),
                        });
                        llvm_ir::Operand::Value(id)
                    };
                    ctx.shape_dim_cache.insert((result_id, dim), dim_value);
                }
            }
            TileIrParamKind::Scalar { index } => {
                // The scalar descriptor is one array of every scalar the launch
                // passed, so the argument is a load at its own slot. Emitting
                // the load here rather than at each use keeps a scalar read out
                // of the kernel's loop bodies.
                //
                // A float scalar and an integer scalar read their own
                // descriptor, and the parameter's type says which.
                let (elem, scalars_operand) = match param.ty {
                    TileIrType::I64 => (
                        llvm_ir::Type::I64,
                        ctx.scalars_i64_param.clone().expect(
                            "integer scalar param prologue requires explicit \
                             __sile_scalars_i64 parameter",
                        ),
                    ),
                    _ => (
                        llvm_ir::Type::F32,
                        ctx.scalars_param.clone().expect(
                            "scalar param prologue requires explicit __sile_scalars parameter",
                        ),
                    ),
                };
                let slot = {
                    let (id, name) = ctx.fresh_value("v");
                    insts.push(llvm_ir::Inst {
                        result: Some(id),
                        result_name: Some(name),
                        ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, elem.clone()),
                        op: llvm_ir::InstOp::Gep {
                            base: scalars_operand,
                            indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(
                                *index as i64,
                            ))],
                        },
                        metadata: Vec::new(),
                    });
                    id
                };
                insts.push(llvm_ir::Inst {
                    result: Some(result_id),
                    result_name: Some(param.name.clone()),
                    ty: elem,
                    op: llvm_ir::InstOp::Load {
                        ptr: llvm_ir::Operand::Value(slot),
                    },
                    metadata: Vec::new(),
                });
            }
            TileIrParamKind::Buffer => {}
        }
    }

    Some(llvm_ir::BasicBlock {
        id: prologue_id,
        name: "prologue".into(),
        params: Vec::new(),
        insts,
        terminator: llvm_ir::Terminator::Br {
            target: llvm_ir_block(tile_ir.entry),
            args: Vec::new(),
        },
    })
}

/// Record each output buffer's tile dimensions in its LLVM IR parameter ABI.
///
/// The Tile IR `store_ptr_tko` operation already carries the tile shape, so the
/// lowering is the only stage that knows it directly. Publishing it on the
/// parameter ABI keeps backends from having to guess the tiling from the CFG,
/// which is both fragile and duplicated across targets.
fn attach_tile_abi(params: &mut [llvm_ir::Param], tile_ir: &TileIrFunction) {
    // buffer ValueId -> (rows, cols) from the first full-tile store through
    // that buffer.
    //
    // An atomic add is not a tile store. `parallel_reduce` accumulates one
    // scalar per program into a one-element output, and reading that as a
    // `[1, 1]` tile made every backend split the launch index by the output's
    // own extent: `sile_tiles_n = 1 / 1`, so every program ran the same row.
    let mut tile_by_buffer: HashMap<ValueId, (i64, i64)> = HashMap::new();
    for block in &tile_ir.blocks {
        for inst in &block.insts {
            if let TileIrOp::StorePtrTko {
                buf, rows, cols, ..
            } = &inst.op
                && *rows > 1
            {
                tile_by_buffer.entry(*buf).or_insert((*rows, *cols));
            }
        }
    }
    if tile_by_buffer.is_empty() {
        return;
    }

    for param in params.iter_mut() {
        let Some(abi) = param.abi.as_mut() else {
            continue;
        };
        // Match the LLVM IR param back to its Tile IR buffer by value id.
        let tile = tile_ir
            .params
            .iter()
            .filter(|p| matches!(p.kind, TileIrParamKind::Buffer))
            .find(|p| llvm_ir_value(p.value) == param.id)
            .and_then(|p| tile_by_buffer.get(&p.value));
        // A buffer with no full-tile store is not a tiled output: it is read
        // or accumulated into one element at a time. Leaving `tile` empty is
        // what tells the backends to treat the launch index as the program id
        // directly, without splitting it by an output extent that does not
        // describe the tiling.
        let Some(tile) = tile else {
            abi.tile = None;
            continue;
        };
        abi.tile = Some((
            usize::try_from(tile.0).unwrap_or(0),
            usize::try_from(tile.1).unwrap_or(0),
        ));
    }
}
