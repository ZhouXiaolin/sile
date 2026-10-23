//! The WGPU backend: WGSL source, plus the runtime that dispatches it.

pub mod codegen;
pub mod runtime;

use sile_core::Result;

/// The bytes one WGSL source needs in workgroup storage.
///
/// Only a module-scope `var<workgroup>` costs workgroup storage. A
/// function-scope `var` array is one invocation's own state — private memory,
/// not shared — so it counts against nothing here: naga and the driver keep it
/// out of `max_compute_workgroup_storage_size`, and counting it refused legal
/// kernels on adapters with a small shared budget.
pub fn local_array_bytes(source: &str) -> u32 {
    let mut total: u32 = 0;
    for line in source.lines() {
        // `var<workgroup> name: array<f32, N>;` — the f32 arrays are the whole
        // cost; an `i64` or an aggregate would need a wider rule, and none
        // exists here.
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("var<workgroup> ") else {
            continue;
        };
        let Some((_, spec)) = rest.split_once("array<f32, ") else {
            continue;
        };
        let Some((count, _)) = spec.split_once(">") else {
            continue;
        };
        let Ok(count) = count.trim().parse::<u32>() else {
            continue;
        };
        total = total.saturating_add(4 * count);
    }
    total
}

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
        .map_err(|e| {
            let line = e
                .labels()
                .next()
                .map(|(span, _)| span.location(source).line_number)
                .unwrap_or(0);
            sile_core::Error::Compile(format!("generated WGSL does not parse at line {line}: {e}"))
        })?;
    // Kernels spell 32-bit integers only, which needs no capability, so
    // validation keeps the default set: an i64 that sneaks back into the
    // spelling is a local compile error naming SHADER_INT64, not a device
    // that refuses the kernel at pipeline-creation time.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .map_err(|e| {
        // An entry point error's Display is one line with no cause; the chain
        // under it names the actual operation. This chain is the difference
        // between "is invalid" and a diagnosis.
        let mut detail = format!("{e}");
        let mut source = std::error::Error::source(&e);
        while let Some(cause) = source {
            detail.push_str(&format!(": {cause}"));
            source = cause.source();
        }
        sile_core::Error::Compile(format!("generated WGSL is not valid: {detail}"))
    })?;
    Ok(())
}
