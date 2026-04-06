mod block;
mod block_scalar;
mod block_terminator;
mod core;
mod tile_compute;
mod tile_expr;
mod tile_loops;
mod tile_memory;

use sile_hir::Type as HirType;
use sile_hir::typeck::TypedKernel;
use sile_llir as llir;

use self::block::lower_block;
use self::core::{LowerLlirCtx, llir_block, llir_type, llir_value};
use crate::MirFunction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlirLoweringPassKind {
    LowerParams,
    LowerBlocks,
    FinalizeFunction,
}

pub const RECOMMENDED_LLIR_LOWERING_PIPELINE: &[LlirLoweringPassKind] = &[
    LlirLoweringPassKind::LowerParams,
    LlirLoweringPassKind::LowerBlocks,
    LlirLoweringPassKind::FinalizeFunction,
];

pub const ACTIVE_LLIR_LOWERING_PIPELINE: &[LlirLoweringPassKind] =
    RECOMMENDED_LLIR_LOWERING_PIPELINE;

struct LlirLoweringState<'a> {
    mir: &'a MirFunction,
    typed: &'a TypedKernel,
    ctx: Option<LowerLlirCtx>,
    params: Option<Vec<llir::Param>>,
    blocks: Option<Vec<llir::BasicBlock>>,
    output: Option<llir::Function>,
}

impl<'a> LlirLoweringState<'a> {
    fn new(mir: &'a MirFunction, typed: &'a TypedKernel) -> Self {
        Self {
            mir,
            typed,
            ctx: None,
            params: None,
            blocks: None,
            output: None,
        }
    }
}

pub fn run_llir_lowering_pipeline(
    mir: &MirFunction,
    typed: &TypedKernel,
    pipeline: &[LlirLoweringPassKind],
) -> Result<llir::Function, String> {
    run_llir_lowering_pipeline_inner(LlirLoweringState::new(mir, typed), pipeline)
}

pub fn run_default_llir_lowering_pipeline(
    mir: &MirFunction,
    typed: &TypedKernel,
) -> Result<llir::Function, String> {
    run_llir_lowering_pipeline(mir, typed, ACTIVE_LLIR_LOWERING_PIPELINE)
}

pub fn lower_mir_to_llir(mir: &MirFunction, typed: &TypedKernel) -> llir::Function {
    run_default_llir_lowering_pipeline(mir, typed)
        .expect("default MIR->LLIR lowering pipeline must produce LLIR function")
}

fn run_llir_lowering_pipeline_inner(
    mut state: LlirLoweringState<'_>,
    pipeline: &[LlirLoweringPassKind],
) -> Result<llir::Function, String> {
    for pass in pipeline {
        match pass {
            LlirLoweringPassKind::LowerParams => {
                let mut ctx = LowerLlirCtx::new(state.mir);
                let param_abis = lower_param_abis(state.typed);
                let params = state
                    .mir
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
                state.ctx = Some(ctx);
                state.params = Some(params);
            }
            LlirLoweringPassKind::LowerBlocks => {
                let ctx = state.ctx.as_mut().ok_or_else(|| {
                    "LowerBlocks requires LowerParams earlier in MIR->LLIR pipeline".to_string()
                })?;
                let blocks = state
                    .mir
                    .blocks
                    .iter()
                    .flat_map(|block| lower_block(block, state.mir, ctx))
                    .collect();
                state.blocks = Some(blocks);
            }
            LlirLoweringPassKind::FinalizeFunction => {
                let params = state.params.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerParams earlier in MIR->LLIR pipeline"
                        .to_string()
                })?;
                let blocks = state.blocks.take().ok_or_else(|| {
                    "FinalizeFunction requires LowerBlocks earlier in MIR->LLIR pipeline"
                        .to_string()
                })?;
                state.output = Some(llir::Function {
                    name: state.mir.name.clone(),
                    params,
                    blocks,
                    entry: llir_block(state.mir.entry),
                    metadata: Vec::new(),
                });
            }
        }
    }

    state
        .output
        .ok_or_else(|| "MIR->LLIR lowering pipeline did not produce LLIR function".to_string())
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
