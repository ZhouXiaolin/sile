use std::collections::{HashMap, HashSet};

use sile_llir as llir;

use crate::ir::*;
use crate::passes::LlirLoweringPlan;

use super::map::llir_value;

pub(crate) struct LowerLlirCtx {
    pub(crate) operands: HashMap<ValueId, llir::Operand>,
    pub(crate) names: HashMap<llir::ValueId, String>,
    pub(crate) program_ids: HashMap<u8, llir::Operand>,
    pub(crate) shape_dims: HashMap<(ValueId, usize), llir::Operand>,
    next_llir_value: u32,
    next_llir_block: u32,
}

impl LowerLlirCtx {
    pub(crate) fn new(mir: &MirFunction) -> Self {
        Self {
            operands: HashMap::new(),
            names: HashMap::new(),
            program_ids: HashMap::new(),
            shape_dims: HashMap::new(),
            next_llir_value: next_llir_value(mir),
            next_llir_block: next_llir_block(mir),
        }
    }

    pub(crate) fn fresh_value(&mut self, prefix: &str) -> (llir::ValueId, String) {
        let id = llir::ValueId(self.next_llir_value);
        self.next_llir_value += 1;
        let name = format!("{prefix}{}", id.0);
        self.names.insert(id, name.clone());
        (id, name)
    }

    pub(crate) fn fresh_block_id(&mut self) -> llir::BlockId {
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

pub(crate) struct BlockLowerer<'a> {
    mir: &'a MirFunction,
    plan: &'a LlirLoweringPlan,
    ctx: &'a mut LowerLlirCtx,
    blocks: Vec<PendingBlock>,
    current: usize,
    materialized_tiles: HashSet<ValueId>,
}

impl<'a> BlockLowerer<'a> {
    pub(crate) fn new(
        mir: &'a MirFunction,
        plan: &'a LlirLoweringPlan,
        ctx: &'a mut LowerLlirCtx,
        id: llir::BlockId,
        name: String,
        params: Vec<llir::BlockParam>,
    ) -> Self {
        Self {
            mir,
            plan,
            ctx,
            blocks: vec![PendingBlock {
                id,
                name,
                params,
                insts: Vec::new(),
                terminator: None,
            }],
            current: 0,
            materialized_tiles: HashSet::new(),
        }
    }

    pub(crate) fn with_current_insts<R>(
        &mut self,
        f: impl FnOnce(&mut LowerLlirCtx, &MirFunction, &mut Vec<llir::Inst>) -> R,
    ) -> R {
        let current = self.current;
        f(self.ctx, self.mir, &mut self.blocks[current].insts)
    }

    pub(crate) fn set_current_terminator(&mut self, term: llir::Terminator) {
        self.blocks[self.current].terminator = Some(term);
    }

    pub(crate) fn create_block(
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

    pub(crate) fn switch_to(&mut self, id: llir::BlockId) {
        self.current = self
            .blocks
            .iter()
            .position(|block| block.id == id)
            .expect("LLIR block must exist");
    }

    pub(crate) fn finish(mut self, final_terminator: llir::Terminator) -> Vec<llir::BasicBlock> {
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

    pub(crate) fn ctx(&self) -> &LowerLlirCtx {
        self.ctx
    }

    pub(crate) fn mir(&self) -> &MirFunction {
        self.mir
    }

    pub(crate) fn plan(&self) -> &LlirLoweringPlan {
        self.plan
    }

    pub(crate) fn begin_materialize_tile(&mut self, value: ValueId) -> Option<MirOp> {
        if !self.materialized_tiles.insert(value) {
            return None;
        }

        self.plan.deferred_tile_op(value).cloned()
    }
}

pub(crate) fn resolve_operand(value: ValueId, ctx: &LowerLlirCtx) -> llir::Operand {
    ctx.operands
        .get(&value)
        .cloned()
        .unwrap_or_else(|| llir::Operand::Value(llir_value(value)))
}

fn next_llir_value(mir: &MirFunction) -> u32 {
    mir.types.keys().map(|id| id.0).max().unwrap_or(0) + 1
}

fn next_llir_block(mir: &MirFunction) -> u32 {
    mir.blocks.iter().map(|block| block.id.0).max().unwrap_or(0) + 1
}
