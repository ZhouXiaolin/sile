use std::collections::HashMap;

use sile_llvm_ir as llvm_ir;

use crate::ir::*;

use super::map::llvm_ir_value;

pub(crate) struct LowerLlvmIrCtx {
    pub(crate) operands: HashMap<ValueId, llvm_ir::Operand>,
    pub(crate) names: HashMap<llvm_ir::ValueId, String>,
    pub(crate) shape_offsets: HashMap<llvm_ir::ValueId, usize>,
    pub(crate) shapes_param: Option<llvm_ir::Operand>,
    pub(crate) prologue_entry: Option<llvm_ir::BlockId>,
    next_llvm_ir_value: u32,
    next_llvm_ir_block: u32,
}

impl LowerLlvmIrCtx {
    pub(crate) fn new(tile_ir: &TileIrFunction) -> Self {
        Self {
            operands: HashMap::new(),
            names: HashMap::new(),
            shape_offsets: HashMap::new(),
            shapes_param: None,
            prologue_entry: None,
            next_llvm_ir_value: next_llvm_ir_value(tile_ir),
            next_llvm_ir_block: next_llvm_ir_block(tile_ir),
        }
    }

    pub(crate) fn fresh_value(&mut self, prefix: &str) -> (llvm_ir::ValueId, String) {
        let id = llvm_ir::ValueId(self.next_llvm_ir_value);
        self.next_llvm_ir_value += 1;
        let name = format!("{prefix}{}", id.0);
        self.names.insert(id, name.clone());
        (id, name)
    }

    pub(crate) fn fresh_block_id(&mut self) -> llvm_ir::BlockId {
        let id = llvm_ir::BlockId(self.next_llvm_ir_block);
        self.next_llvm_ir_block += 1;
        id
    }
}

#[derive(Clone)]
struct PendingBlock {
    id: llvm_ir::BlockId,
    name: String,
    params: Vec<llvm_ir::BlockParam>,
    insts: Vec<llvm_ir::Inst>,
    terminator: Option<llvm_ir::Terminator>,
}

pub(crate) struct BlockLowerer<'a> {
    tile_ir: &'a TileIrFunction,
    ctx: &'a mut LowerLlvmIrCtx,
    blocks: Vec<PendingBlock>,
    current: usize,
}

impl<'a> BlockLowerer<'a> {
    pub(crate) fn new(
        tile_ir: &'a TileIrFunction,
        ctx: &'a mut LowerLlvmIrCtx,
        id: llvm_ir::BlockId,
        name: String,
        params: Vec<llvm_ir::BlockParam>,
    ) -> Self {
        Self {
            tile_ir,
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

    pub(crate) fn with_current_insts<R>(
        &mut self,
        f: impl FnOnce(&mut LowerLlvmIrCtx, &TileIrFunction, &mut Vec<llvm_ir::Inst>) -> R,
    ) -> R {
        let current = self.current;
        f(self.ctx, self.tile_ir, &mut self.blocks[current].insts)
    }

    pub(crate) fn set_current_terminator(&mut self, term: llvm_ir::Terminator) {
        self.blocks[self.current].terminator = Some(term);
    }

    pub(crate) fn create_block(
        &mut self,
        prefix: &str,
        params: Vec<(&str, llvm_ir::Type)>,
    ) -> (llvm_ir::BlockId, Vec<llvm_ir::BlockParam>) {
        let id = self.ctx.fresh_block_id();
        let block_params = params
            .into_iter()
            .map(|(param_prefix, ty)| {
                let (id, name) = self.ctx.fresh_value(param_prefix);
                llvm_ir::BlockParam { id, name, ty }
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

    pub(crate) fn switch_to(&mut self, id: llvm_ir::BlockId) {
        self.current = self
            .blocks
            .iter()
            .position(|block| block.id == id)
            .expect("LLVM IR block must exist");
    }

    pub(crate) fn finish(
        mut self,
        final_terminator: llvm_ir::Terminator,
    ) -> Vec<llvm_ir::BasicBlock> {
        if self.blocks[self.current].terminator.is_none() {
            self.blocks[self.current].terminator = Some(final_terminator);
        }
        self.blocks
            .into_iter()
            .map(|block| llvm_ir::BasicBlock {
                id: block.id,
                name: block.name,
                params: block.params,
                insts: block.insts,
                terminator: block.terminator.expect("LLVM IR block missing terminator"),
            })
            .collect()
    }

    pub(crate) fn ctx(&self) -> &LowerLlvmIrCtx {
        self.ctx
    }

    pub(crate) fn tile_ir(&self) -> &TileIrFunction {
        self.tile_ir
    }
}

pub(crate) fn resolve_operand(value: ValueId, ctx: &LowerLlvmIrCtx) -> llvm_ir::Operand {
    ctx.operands
        .get(&value)
        .cloned()
        .unwrap_or_else(|| llvm_ir::Operand::Value(llvm_ir_value(value)))
}

fn next_llvm_ir_value(tile_ir: &TileIrFunction) -> u32 {
    tile_ir.types.keys().map(|id| id.0).max().unwrap_or(0) + 1
}

fn next_llvm_ir_block(tile_ir: &TileIrFunction) -> u32 {
    tile_ir
        .blocks
        .iter()
        .map(|block| block.id.0)
        .max()
        .unwrap_or(0)
        + 1
}
