pub mod canonicalize;
pub mod cse;
pub mod loop_simplify;
pub mod phi_alias;
pub mod verify;

use crate::Function;

/// Every source file a pass directory must account for.
///
/// A retired pass must be deleted, not left in the tree. A file with no `mod`
/// declaration does not compile, so no test and no warning can see it: three
/// retired LLVM IR passes sat here for 1097 lines while `cargo build` reported
/// nothing. This test is the alarm that catches the next one.
#[cfg(test)]
mod no_orphan_files {
    #[test]
    fn every_rs_file_in_passes_is_declared() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/passes");
        let declared = std::fs::read_to_string(dir.join("mod.rs"))
            .expect("read src/passes/mod.rs");
        let mut orphans = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("read src/passes directory")
            .map(|entry| entry.expect("dir entry").path())
            .collect();
        entries.sort();
        for path in entries {
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if stem == "mod" {
                continue;
            }
            let wants = format!("pub mod {stem};");
            if !declared.contains(&wants) {
                orphans.push(format!("{stem}.rs is not declared as `{wants}`"));
            }
        }
        assert!(
            orphans.is_empty(),
            "these files are not compiled, so nothing else can see them:\n{}",
            orphans.join("\n")
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmIrPassKind {
    VerifyInput,
    Canonicalize,
    LoopSimplify,
    PhiAlias,
    Cse,
    VerifyOutput,
}

/// Planned LLVM IR pipeline.
///
/// A pass belongs here only while removing it changes the IR of at least one real
/// kernel. `crates/sile/tests/pass_ablation_e2e.rs` measures exactly that, so a
/// pass that turns into ballast fails the suite and gets removed. Note the test
/// removes passes rather than running them alone: `Canonicalize` looks redundant
/// on its own in one order and indispensable in another, and only removal
/// reports the truth.
///
/// LLVM IR passes work on explicit SSA/CFG/memory semantics and stay
/// backend-independent. Target-specific legalizations run after this stage in
/// backend-owned code.
pub const RECOMMENDED_PIPELINE: &[LlvmIrPassKind] = &[
    LlvmIrPassKind::VerifyInput,
    LlvmIrPassKind::Canonicalize,
    LlvmIrPassKind::LoopSimplify,
    LlvmIrPassKind::PhiAlias,
    LlvmIrPassKind::Cse,
    LlvmIrPassKind::VerifyOutput,
];

/// Passes that are active today.
pub const ACTIVE_PIPELINE: &[LlvmIrPassKind] = RECOMMENDED_PIPELINE;

/// Every active transform, as `(name, runner)`.
///
/// The two verify kinds are checks, not transforms, so they are absent. This is
/// the list the fixture-level ablation reads, which keeps a retired pass from
/// keeping a fixture.
pub fn active_passes() -> impl Iterator<Item = (&'static str, fn(Function) -> Function)> {
    vec![
        ("Canonicalize", canonicalize::run as fn(Function) -> Function),
        ("LoopSimplify", loop_simplify::run as fn(Function) -> Function),
        ("PhiAlias", phi_alias::run as fn(Function) -> Function),
        ("Cse", cse::run as fn(Function) -> Function),
    ]
    .into_iter()
}

pub fn run_pipeline(mut func: Function, pipeline: &[LlvmIrPassKind]) -> Result<Function, String> {
    for pass in pipeline {
        func = match pass {
            LlvmIrPassKind::VerifyInput | LlvmIrPassKind::VerifyOutput => {
                verify::verify_function(&func)?;
                func
            }
            LlvmIrPassKind::Canonicalize => canonicalize::run(func),
            LlvmIrPassKind::LoopSimplify => loop_simplify::run(func),
            LlvmIrPassKind::PhiAlias => phi_alias::run(func),
            LlvmIrPassKind::Cse => cse::run(func),
        };
    }
    Ok(func)
}

pub fn run_default_pipeline(func: Function) -> Result<Function, String> {
    run_pipeline(func, ACTIVE_PIPELINE)
}
