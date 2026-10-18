//! The WGPU backend: WGSL source, plus the runtime that dispatches it.

pub mod codegen;
pub mod runtime;

use sile_core::Result;

/// Parse and validate WGSL source.
///
/// Naga is the front end of the compiler stack this backend feeds, so its
/// judgment of whether a shader is legal is the one that counts. Every other
/// target's text is checked by a compiler during the test or by the driver, and
/// WGSL has one here, so its bugs surface at compile time.
pub fn validate(source: &str) -> Result<()> {
    if std::env::var_os("SILE_PRINT_WGSL").is_some() {
        eprintln!("{source}");
    }
    let module = naga::front::wgsl::Frontend::new()
        .parse(source)
        .map_err(|e| sile_core::Error::Compile(format!("generated WGSL does not parse: {e}")))?;
    // The lowering makes every index an i64, which WGSL allows only when the
    // target enables the capability, so validation permits it rather than
    // rejecting a shader the device would accept.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| sile_core::Error::Compile(format!("generated WGSL is not valid: {e}")))?;
    Ok(())
}
