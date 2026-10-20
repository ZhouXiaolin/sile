//! Tile IR operations.
//!
//! Ported from `cutile-ir`'s `ir/op.rs`. Each operation corresponds to an
//! opcode in [`crate::opcode`].

use super::attr::Attribute;
use super::location::Location;
use super::types::Type;
use super::value::{RegionId, Value};
use crate::opcode::Opcode;

/// A single IR operation.
///
/// Operations are the fundamental unit of computation. Each has an opcode,
/// zero or more operands (SSA values), zero or more results (typed SSA
/// values), named attributes, and optional regions (for control flow ops
/// like `for`, `if`).
#[derive(Debug, Clone)]
pub struct Operation {
    pub opcode: Opcode,
    pub operands: Vec<Value>,
    pub result_types: Vec<Type>,
    pub attributes: Vec<(String, Attribute)>,
    pub regions: Vec<RegionId>,
    pub location: Location,
}
