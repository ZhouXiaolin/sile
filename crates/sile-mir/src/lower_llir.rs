use sile_hir::Type as HirType;
use sile_hir::typeck::TypedKernel;
use sile_llir as llir;

use crate::MirFunction;
use crate::lower_llir_block::lower_block;
use crate::lower_llir_core::{LowerLlirCtx, llir_block, llir_type, llir_value};
use crate::passes::{LlirLoweringPlan, build_llir_lowering_plan};

/// Lowers MIR into raw LLIR without running any LLIR optimization pipeline.
///
/// This function is intentionally the semantic boundary between MIR and LLIR.
/// Target-independent cleanups and profitability-driven rewrites should live in
/// explicit MIR/LLIR passes instead of being encoded here long term.
pub fn lower_mir_to_llir_raw(mir: &MirFunction, typed: &TypedKernel) -> llir::Function {
    let plan = build_llir_lowering_plan(mir);
    lower_mir_to_llir_raw_with_plan(mir, typed, &plan)
}

pub fn lower_mir_to_llir_raw_with_plan(
    mir: &MirFunction,
    typed: &TypedKernel,
    plan: &LlirLoweringPlan,
) -> llir::Function {
    let mut ctx = LowerLlirCtx::new(mir);
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
        .flat_map(|block| lower_block(block, mir, plan, &mut ctx))
        .collect();

    llir::Function {
        name: mir.name.clone(),
        params,
        blocks,
        entry: llir_block(mir.entry),
        metadata: Vec::new(),
    }
}

pub fn lower_mir_to_llir(mir: &MirFunction, typed: &TypedKernel) -> llir::Function {
    lower_mir_to_llir_raw(mir, typed)
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
