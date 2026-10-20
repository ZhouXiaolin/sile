//! The cutile-facing half of the one Tile IR.
//!
//! [`emit`] lowers a typed kernel into a [`crate::ir::Module`] — the
//! arena-structured port of cutile-ir — and `crate::ir::fmt` prints that
//! module byte for byte the way cutile-rs does. There is no separate text IR:
//! the module the printer reads is the module the lowering built.
//!
//! The access checks are not here. cutile-rs's judgments live in
//! `sile-hir`'s `passes::{access, bounds, predicate, value_ids}`; the lowering
//! consumes their decisions.
//!
//! `crates/sile-tile-ir/TILEIR_ALIGNMENT.md` records the format rules and the
//! fixtures in `crates/sile/tests/fixtures/tileir/` are the target text.

pub mod emit;
