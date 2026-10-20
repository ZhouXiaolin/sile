//! Lowering from the arena Tile IR module to LLVM IR.
//!
//! One entry point: [`lower_module_to_llvm_ir`]. The module it reads is the
//! whole Tile IR, so this is the only path from a kernel to a backend's input.

pub mod module;

pub use module::lower_module_to_llvm_ir;
