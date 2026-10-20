//! The one Tile IR: an arena-structured port of `cutile-ir`'s `ir` module.
//!
//! This is the IR both the HIR lowering produces and the text printer
//! consumes — there is no parallel text-shaped IR. Values are numbered by
//! allocation order, so the printer's `%N` is deterministic given the
//! builder's emission order; see [`builder`] for the allocate-is-number
//! contract and how discarded operations leave holes.
//!
//! Structure mirrors `cutile-rs/cutile-ir/src/ir/*` field for field, minus
//! what sile never uses (module globals, bytecode verification, debug-info
//! locations).

pub mod attr;
pub mod block;
pub mod builder;
pub mod fmt;
pub mod location;
pub mod module;
pub mod op;
pub mod region;
pub mod types;
pub mod value;

pub use attr::*;
pub use block::*;
pub use builder::*;
pub use location::*;
pub use module::*;
pub use op::*;
pub use region::*;
pub use types::*;
pub use value::*;
