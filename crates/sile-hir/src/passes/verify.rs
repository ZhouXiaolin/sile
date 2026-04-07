use std::collections::HashSet;

use crate::Kernel;
use crate::typeck::{TypeError, TypedKernel};

pub fn verify_kernel(kernel: &Kernel) -> Result<(), TypeError> {
    if kernel.name.trim().is_empty() {
        return Err(TypeError::invalid_kernel("kernel name must not be empty"));
    }

    let mut seen_params = HashSet::new();
    for param in &kernel.params {
        if param.name.trim().is_empty() {
            return Err(TypeError::invalid_kernel(
                "kernel param name must not be empty",
            ));
        }
        if !seen_params.insert(param.name.as_str()) {
            return Err(TypeError::invalid_kernel(format!(
                "duplicate kernel param name `{}`",
                param.name
            )));
        }
    }

    let mut seen_consts = HashSet::new();
    for (name, _) in &kernel.const_params {
        if name.trim().is_empty() {
            return Err(TypeError::invalid_kernel(
                "const param name must not be empty",
            ));
        }
        if !seen_consts.insert(name.as_str()) {
            return Err(TypeError::invalid_kernel(format!(
                "duplicate const param name `{name}`"
            )));
        }
    }

    Ok(())
}

pub fn verify_typed_kernel(typed: &TypedKernel) -> Result<(), TypeError> {
    verify_kernel(&typed.kernel)?;
    for local in typed.locals.keys() {
        if local.trim().is_empty() {
            return Err(TypeError::invalid_kernel(
                "typed local name must not be empty",
            ));
        }
    }
    Ok(())
}
