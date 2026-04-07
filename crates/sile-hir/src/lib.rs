pub mod kernel;
pub mod passes;
pub mod typeck;
pub mod types;

pub use kernel::{BuiltinOp, Expr, Kernel, Param, ParamKind, Stmt};
pub use passes::verify::{verify_kernel as verify_hir_kernel, verify_typed_kernel};
pub use passes::{
    ACTIVE_PIPELINE as ACTIVE_HIR_PIPELINE, HirPassKind,
    RECOMMENDED_PIPELINE as RECOMMENDED_HIR_PIPELINE, run_default_pipeline as run_hir_passes,
    run_pipeline as run_hir_pipeline,
};
pub use types::{ElemType, ShapeExpr, Type, ValueCategory};
