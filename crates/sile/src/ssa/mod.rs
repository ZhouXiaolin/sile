pub mod ir;
mod lower;

pub use lower::lower_typed_kernel_to_ssa;
