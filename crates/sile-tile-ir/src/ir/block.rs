//! Basic blocks — ordered sequences of operations.
//!
//! Ported from `cutile-ir`'s `ir/block.rs`.

use super::types::Type;
use super::value::{OpId, Value};

/// A basic block: a sequence of operations with typed entry arguments.
///
/// Block arguments serve as phi-like merge points (e.g., for-loop
/// induction variables, if-else yield values).
#[derive(Debug, Clone)]
pub struct Block {
    /// Typed arguments to this block.
    pub args: Vec<(Value, Type)>,
    /// Operations in this block, in order.
    pub ops: Vec<OpId>,
}
