//! The Tile IR text, as cutile-rs prints it.
//!
//! `sile-tile-ir` has two IRs. [`sile_tile_ir::ir`] is what the C, WGPU and PTX
//! lowerings read. [`sile_tile_ir::cutile`] exists only to print cutile's text,
//! and this module is its front door: it takes a typed kernel and returns the
//! text a fixture compares against.
//!
//! `crates/sile-tile-ir/TILEIR_ALIGNMENT.md` records the format. The fixtures in
//! `crates/sile/tests/fixtures/tileir/` are the target.

use sile_hir::typeck::TypedKernel;

/// The Tile IR text cutile-rs prints for this kernel.
pub fn emit(typed: &TypedKernel) -> String {
    let module = sile_tile_ir::cutile::emit::emit(&typed.kernel, &typed.locals);
    sile_tile_ir::cutile::print::format(&module)
}
