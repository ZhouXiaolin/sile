//! SSA value handles.
//!
//! Ported from `cutile-ir`'s `ir/value.rs`. Values are lightweight
//! index-based handles into the module's value table — no lifetime
//! parameters, so they can be stored and passed freely.
//!
//! The handle's index **is** the number the text printer emits: values are
//! allocated in program order, so `%N` is the N-th value ever allocated in
//! the module. Discarded operations leave holes — see [`crate::ir::builder`].

use super::types::Type;

/// Index-based handle to an SSA value in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub(crate) u32);

impl Value {
    pub fn index(self) -> u32 {
        self.0
    }

    /// The handle for arena slot `index`.
    ///
    /// A lowering walks the value arena to answer "which value did this op
    /// produce", so it needs the handle for a slot it found by index.
    pub fn new(index: u32) -> Self {
        Self(index)
    }
}

/// Metadata stored for each value in the module's value arena.
#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: Type,
    /// Which operation produced this value (result), or which block
    /// owns it (block argument).
    pub producer: ValueProducer,
}

/// Where a value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueProducer {
    /// Result `result_index` of operation `OpId`.
    OpResult { op: OpId, result_index: u32 },
    /// Argument `arg_index` of block `BlockId`.
    BlockArg { block: BlockId, arg_index: u32 },
}

/// Index-based handle to an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub(crate) u32);

impl OpId {
    /// The handle for arena slot `index`.
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    pub fn index(self) -> u32 {
        self.0
    }
}

/// Index-based handle to a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub(crate) u32);

/// Index-based handle to a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub(crate) u32);
