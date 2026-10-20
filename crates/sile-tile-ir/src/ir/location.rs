//! Source location tracking for operations.
//!
//! Trimmed port of `cutile-ir`'s `ir/location.rs`: the debug-info variants
//! (`DebugInfo`, `CallSite`) are dropped until sile emits debug info. The
//! text printer never prints locations.

/// Source location attached to operations for debug info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// No known source location.
    Unknown,
    /// File, line, column.
    FileLineCol {
        filename: String,
        line: u32,
        column: u32,
    },
}
