//! The WGPU backend: WGSL spelling of the same LLVM IR.

pub mod codegen;

use sile_core::Result;

/// Parse and validate WGSL source.
///
/// Naga is the front end of the compiler stack this backend feeds, so its
/// judgment of whether a shader is legal is the one that counts. Every other
/// target's text is checked by a compiler in the test or the driver; WGSL has
/// one here, so its bugs surface at compile time.
pub fn validate(source: &str) -> Result<()> {
    let module = naga::front::wgsl::Frontend::new()
        .parse(source)
        .map_err(|e| sile_core::Error::Compile(format!("generated WGSL does not parse: {e}")))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| sile_core::Error::Compile(format!("generated WGSL is not valid: {e}")))?;
    Ok(())
}
