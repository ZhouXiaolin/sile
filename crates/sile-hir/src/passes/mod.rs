pub mod verify;

use crate::Kernel;
use crate::typeck::{self, TypeError, TypedKernel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HirPassKind {
    VerifyInput,
    TypeCheck,
    VerifyOutput,
}

pub const RECOMMENDED_PIPELINE: &[HirPassKind] = &[
    HirPassKind::VerifyInput,
    HirPassKind::TypeCheck,
    HirPassKind::VerifyOutput,
];

pub const ACTIVE_PIPELINE: &[HirPassKind] = RECOMMENDED_PIPELINE;

pub fn run_pipeline(kernel: &Kernel, pipeline: &[HirPassKind]) -> Result<TypedKernel, TypeError> {
    let mut typed = None;

    for pass in pipeline {
        match pass {
            HirPassKind::VerifyInput => verify::verify_kernel(kernel)?,
            HirPassKind::TypeCheck => {
                typed = Some(typeck::check_kernel(kernel)?);
            }
            HirPassKind::VerifyOutput => {
                let typed_ref = typed.as_ref().ok_or_else(|| {
                    TypeError::invalid_pipeline(
                        "HIR VerifyOutput requires TypeCheck earlier in pipeline",
                    )
                })?;
                verify::verify_typed_kernel(typed_ref)?;
            }
        }
    }

    typed.ok_or_else(|| {
        TypeError::invalid_pipeline("HIR pipeline did not produce TypedKernel via TypeCheck pass")
    })
}

pub fn run_default_pipeline(kernel: &Kernel) -> Result<TypedKernel, TypeError> {
    run_pipeline(kernel, ACTIVE_PIPELINE)
}
