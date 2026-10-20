//! The Tile IR text, as cutile-rs prints it.
//!
//! There is one Tile IR: [`sile_tile_ir::ir`], the arena-structured port of
//! cutile-ir. [`sile_tile_ir::cutile::emit`] lowers a typed kernel into a
//! `Module` of it, and `Module::to_mlir_text` prints that module byte for byte
//! the way cutile-rs does. This module is the front door: it takes a typed
//! kernel and returns the text a fixture compares against.
//!
//! `crates/sile-tile-ir/TILEIR_ALIGNMENT.md` records the format. The fixtures in
//! `crates/sile/tests/fixtures/tileir/` are the target.

use sile_hir::typeck::TypedKernel;

/// The Tile IR text cutile-rs prints for this kernel.
///
/// An access check cutile decides at compile time can fail, and that failure is
/// the kernel's: a violated access is a compile error, not a device assert.
pub fn emit(typed: &TypedKernel) -> Result<String, String> {
    let (module, _checks) = sile_tile_ir::cutile::emit::emit(typed)?;
    Ok(module.to_mlir_text())
}

/// The access checks the compiler moved out of the kernel.
///
/// They are not part of the printed text: the text is the same whether a check
/// is an in-kernel `assert` or a host-side predicate, and these are the ones the
/// launcher must evaluate before it runs the kernel.
pub fn launch_checks(
    typed: &TypedKernel,
) -> Result<Vec<sile_hir::passes::predicate::LaunchCheck>, String> {
    let (_module, checks) = sile_tile_ir::cutile::emit::emit(typed)?;
    Ok(checks)
}
