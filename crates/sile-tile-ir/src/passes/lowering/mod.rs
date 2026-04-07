mod block;
mod block_scalar;
mod block_terminator;
mod core;
mod tile_compute;
mod tile_expr;
mod tile_memory;

use sile_hir::Type as HirType;
use sile_hir::typeck::TypedKernel;
use sile_llvm_ir as llvm_ir;

use self::block::lower_block;
use self::core::{LowerLlvmIrCtx, llvm_ir_block, llvm_ir_type, llvm_ir_value};
use crate::{TileIrFunction, TileIrParamKind};

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

const SHAPES_PARAM_NAME: &str = "__sile_shapes";

struct LlvmIrLoweringState<'a> {
    tile_ir: &'a TileIrFunction,
    typed: &'a TypedKernel,
    ctx: Option<LowerLlvmIrCtx>,
    params: Option<Vec<llvm_ir::Param>>,
    blocks: Option<Vec<llvm_ir::BasicBlock>>,
    output: Option<llvm_ir::Function>,
}

impl<'a> LlvmIrLoweringState<'a> {
    fn new(tile_ir: &'a TileIrFunction, typed: &'a TypedKernel) -> Self {
        Self {
            tile_ir,
            typed,
            ctx: None,
            params: None,
            blocks: None,
            output: None,
        }
    }
}

pub fn run_llvm_ir_lowering_pipeline(
    tile_ir: &TileIrFunction,
    typed: &TypedKernel,
    pipeline: &[LlvmIrLoweringPassKind],
) -> Result<llvm_ir::Function, String> {
    run_llvm_ir_lowering_pipeline_inner(LlvmIrLoweringState::new(tile_ir, typed), pipeline)
}

pub fn run_default_llvm_ir_lowering_pipeline(
    tile_ir: &TileIrFunction,
    typed: &TypedKernel,
) -> Result<llvm_ir::Function, String> {
    run_llvm_ir_lowering_pipeline(tile_ir, typed, ACTIVE_LLVM_IR_LOWERING_PIPELINE)
}

pub fn lower_tile_ir_to_llvm_ir(
    tile_ir: &TileIrFunction,
    typed: &TypedKernel,
) -> llvm_ir::Function {
    run_default_llvm_ir_lowering_pipeline(tile_ir, typed)
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
                let param_abis = lower_param_abis(state.typed);
                let mut abi_idx = 0usize;
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
                            let abi = param_abis.get(abi_idx).cloned().flatten();
                            abi_idx += 1;
                            if let Some(abi) = abi.clone() {
                                ctx.shape_offsets.insert(id, abi.shape_offset);
                            }
                            Some(llvm_ir::Param {
                                id,
                                name: param.name.clone(),
                                ty: llvm_ir_type(&param.ty),
                                abi,
                            })
                        }
                        TileIrParamKind::SileProgramId { .. }
                        | TileIrParamKind::SileShapeDim { .. } => None,
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
                blocks.extend(
                    state
                        .tile_ir
                        .blocks
                        .iter()
                        .flat_map(|block| lower_block(block, state.tile_ir, ctx)),
                );
                state.blocks = Some(blocks);
            }
            LlvmIrLoweringPassKind::FinalizeFunction => {
                let params = state.params.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerParams earlier in Tile IR->LLVM IR pipeline"
                        .to_string()
                })?;
                let blocks = state.blocks.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerBlocks earlier in Tile IR->LLVM IR pipeline"
                        .to_string()
                })?;
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
            TileIrParamKind::SileProgramId { dim } => {
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
            TileIrParamKind::SileShapeDim { source, dim } => {
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
                let (ptr_id, ptr_name) = ctx.fresh_value("v");
                insts.push(llvm_ir::Inst {
                    result: Some(ptr_id),
                    result_name: Some(ptr_name),
                    ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                    op: llvm_ir::InstOp::Gep {
                        base: shapes_operand,
                        indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(
                            (shape_offset + dim) as i64,
                        ))],
                    },
                    metadata: Vec::new(),
                });
                insts.push(llvm_ir::Inst {
                    result: Some(result_id),
                    result_name: Some(param.name.clone()),
                    ty: llvm_ir::Type::I64,
                    op: llvm_ir::InstOp::Load {
                        ptr: llvm_ir::Operand::Value(ptr_id),
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

fn lower_param_abis(typed: &TypedKernel) -> Vec<Option<llvm_ir::ParamAbi>> {
    let mut next_shape_offset = 0usize;
    typed
        .kernel
        .params
        .iter()
        .map(|param| match &param.ty {
            HirType::Tensor { shape, .. } | HirType::Tile { shape, .. } => {
                let abi = llvm_ir::ParamAbi {
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
