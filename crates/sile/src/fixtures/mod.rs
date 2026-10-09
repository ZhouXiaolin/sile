//! Canonical kernel fixtures shared by the codegen tests and the ablation
//! report, so both measure the same compilation inputs.

pub mod kernels;

pub use kernels::{
    build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel,
};
