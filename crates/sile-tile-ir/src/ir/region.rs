//! Regions — containers of basic blocks attached to operations.
//!
//! Ported from `cutile-ir`'s `ir/region.rs`.

use super::value::BlockId;

/// A region is an ordered list of basic blocks, attached to an operation.
///
/// Control-flow operations (`for`, `if`) contain one or more regions.
/// Functions contain a single region with one entry block.
#[derive(Debug, Clone)]
pub struct Region {
    pub blocks: Vec<BlockId>,
}
