//! Top-level module — the root IR container that owns all operations, values,
//! blocks, and regions via arena storage.
//!
//! Ported from `cutile-ir`'s `ir/module.rs`. The module-level `globals` table
//! and the bytecode value-index verification are dropped — sile emits neither.

use std::collections::HashSet;

use super::block::Block;
use super::location::Location;
use super::op::Operation;
use super::region::Region;
use super::types::Type;
use super::value::{BlockId, OpId, RegionId, Value, ValueData, ValueProducer};

/// The root container for a Tile IR program.
///
/// A `Module` owns all IR objects via flat `Vec` arenas. References between
/// objects use index-based handles ([`Value`], [`OpId`], [`BlockId`],
/// [`RegionId`]) — no Rust lifetimes required.
///
/// Typical usage:
/// 1. Create a `Module`.
/// 2. Use the builder API ([`super::builder`]) to populate it.
/// 3. Print it with [`Module::to_mlir_text`].
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    /// Optional producer identity.
    pub producer: Option<String>,

    // ------ Arenas ------
    pub(crate) values: Vec<ValueData>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) blocks: Vec<Block>,
    pub(crate) regions: Vec<Region>,

    /// The top-level entry functions in this module.
    pub functions: Vec<OpId>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            producer: None,
            values: Vec::new(),
            operations: Vec::new(),
            blocks: Vec::new(),
            regions: Vec::new(),
            functions: Vec::new(),
        }
    }

    // ------ Value arena ------

    pub fn alloc_value(&mut self, ty: Type, producer: ValueProducer) -> Value {
        let id = Value(self.values.len() as u32);
        self.values.push(ValueData { ty, producer });
        id
    }

    /// Number of values allocated so far. Usable as a dominance watermark:
    /// a value with `index() < num_values()` taken at some program point was
    /// allocated before that point.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    pub fn value_data(&self, v: Value) -> &ValueData {
        &self.values[v.0 as usize]
    }

    pub fn value_type(&self, v: Value) -> &Type {
        &self.values[v.0 as usize].ty
    }

    // ------ Operation arena ------

    pub fn alloc_op(&mut self, op: Operation) -> OpId {
        let id = OpId(self.operations.len() as u32);
        self.operations.push(op);
        id
    }

    /// Number of operations in the arena.
    pub fn num_ops(&self) -> usize {
        self.operations.len()
    }

    /// Every operation, in allocation order.
    pub fn ops(&self) -> impl Iterator<Item = (OpId, &Operation)> {
        self.operations
            .iter()
            .enumerate()
            .map(|(i, op)| (OpId(i as u32), op))
    }

    /// Every value, in allocation order.
    pub fn values(&self) -> impl Iterator<Item = (Value, &ValueData)> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, data)| (Value(i as u32), data))
    }

    pub fn op(&self, id: OpId) -> &Operation {
        &self.operations[id.0 as usize]
    }

    pub fn op_mut(&mut self, id: OpId) -> &mut Operation {
        &mut self.operations[id.0 as usize]
    }

    // ------ Block arena ------

    pub fn alloc_block(&mut self, block: Block) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(block);
        id
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    // ------ Region arena ------

    pub fn alloc_region(&mut self, region: Region) -> RegionId {
        let id = RegionId(self.regions.len() as u32);
        self.regions.push(region);
        id
    }

    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.0 as usize]
    }

    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        &mut self.regions[id.0 as usize]
    }

    // ------ Location ------

    pub fn op_location(&self, id: OpId) -> &Location {
        &self.operations[id.0 as usize].location
    }

    // ------ Verification ------

    /// Verify that every operand in the module references a value that
    /// dominates its use point.
    ///
    /// In structured control flow (no CFG edges), dominance is simple:
    /// - Block arguments dominate all operations in the block.
    /// - An operation's results dominate later operations in the same block.
    /// - All values visible in a parent scope dominate operations in nested
    ///   regions (for, if bodies).
    ///
    /// Call this after building IR to catch value reference ordering bugs
    /// with clear error messages.
    pub fn verify_dominance(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        for &func_id in &self.functions {
            let func_op = self.op(func_id);
            let func_name = func_op
                .attributes
                .iter()
                .find(|(k, _)| k == "sym_name")
                .map(|(_, v)| format!("{v:?}"))
                .unwrap_or_else(|| format!("op#{}", func_id.0));

            // Function-level ops have no operands to check; start with their regions.
            let visible = HashSet::new();
            for &region_id in &func_op.regions {
                self.verify_region(region_id, &visible, &func_name, &mut errors);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    fn verify_region(
        &self,
        region_id: RegionId,
        parent_visible: &HashSet<Value>,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let region = self.region(region_id);
        for &block_id in &region.blocks {
            self.verify_block(block_id, parent_visible, func_name, errors);
        }
    }

    fn verify_block(
        &self,
        block_id: BlockId,
        parent_visible: &HashSet<Value>,
        func_name: &str,
        errors: &mut Vec<String>,
    ) {
        let block = self.block(block_id);

        // Start with parent scope values + this block's arguments.
        let mut visible = parent_visible.clone();
        for &(arg_val, _) in &block.args {
            visible.insert(arg_val);
        }

        for &op_id in &block.ops {
            let op = self.op(op_id);

            // Check that every operand is visible at this point.
            for (i, &operand) in op.operands.iter().enumerate() {
                if !visible.contains(&operand) {
                    errors.push(format!(
                        "in {func_name}: operand {i} of {:?} (op#{}) references \
                         value %{} which does not dominate this use",
                        op.opcode, op_id.0, operand.0,
                    ));
                }
            }

            // Add this op's results to the visible set.
            for ri in 0..op.result_types.len() {
                // Results are allocated immediately after the op, so search
                // for the matching ValueProducer.
                if let Some(val) = self.find_op_result(op_id, ri as u32) {
                    visible.insert(val);
                }
            }

            // Recurse into regions attached to this op.
            for &region_id in &op.regions {
                self.verify_region(region_id, &visible, func_name, errors);
            }
        }
    }

    /// Find the `Value` handle for result `result_index` of operation `op_id`.
    fn find_op_result(&self, op_id: OpId, result_index: u32) -> Option<Value> {
        // Linear scan — the value arena is small per-function and this runs
        // once for verification, not on the hot path.
        for (i, vd) in self.values.iter().enumerate() {
            if vd.producer
                == (ValueProducer::OpResult {
                    op: op_id,
                    result_index,
                })
            {
                return Some(Value(i as u32));
            }
        }
        None
    }
}
