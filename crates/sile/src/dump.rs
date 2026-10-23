//! Stage-by-stage IR dumping for analysis, off by default.
//!
//! A launch runs one pipeline: HIR, then Tile IR, then LSIR, then a text
//! backend. The pipeline is the compiler; a reader who wants to know what each
//! pass did has to see the IR after every one of them, not only at the end.
//!
//! Setting `SILE_DUMP_DIR` makes each distinct kernel write its whole history
//! under that directory, one file per stage and one per pass, plus a `metrics.tsv`
//! row per stage and a `timing.tsv` row per stage. Nothing here changes what the
//! compiler produces: every snapshot is a copy, and the launch compiles its own
//! kernels as usual.
//!
//! The files are numbered so that a directory listing reads in pipeline order.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sile_core::Result;
use sile_hir::typeck::TypedKernel;
use sile_lsir::{Function as LsirFunction, LsirPassKind};
use crate::compiler::{
    ACTIVE_HIR_TYPECK_PIPELINE, ACTIVE_LSIR_PIPELINE, CodegenTarget, compile_backend,
    run_hir_analysis_pipeline,
};

/// The environment variable that turns dumping on and names its directory.
pub const DUMP_DIR_ENV: &str = "SILE_DUMP_DIR";

/// The environment variable that limits a dump to the Tile IR text.
///
/// Aligning the Tile IR needs one stage. Every later stage costs a compile of
/// the whole pipeline and the backend sources, and none of them is compared, so
/// this switch stops the dump at the text the fixtures hold.
pub const TILEIR_ONLY_ENV: &str = "SILE_DUMP_TILEIR_ONLY";

/// Whether the dump stops after the Tile IR.
pub fn tileir_only() -> bool {
    std::env::var_os(TILEIR_ONLY_ENV).is_some()
}

/// The directory to dump into, when the environment asks for one.
pub fn dump_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(DUMP_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// One measured stage: its name, its artifact text, and how long it took.
struct Stage {
    label: String,
    text: String,
    micros: u128,
}

/// Dump every stage of `kernel`'s compilation into `root/<kernel-name>/`.
///
/// A failure to write is reported to the caller; a failure to compile is
/// reported the same way, because a dump of a kernel that does not compile is
/// exactly the artifact worth reading.
pub fn dump_kernel(kernel: &sile_hir::Kernel, root: &Path) -> Result<()> {
    let dir = unique_kernel_dir(root, &kernel.name);
    fs::create_dir_all(&dir)?;

    let mut stages: Vec<Stage> = Vec::new();

    // ── HIR ────────────────────────────────────────────────────────
    let started = Instant::now();
    stages.push(Stage {
        label: "hir-00-parsed".into(),
        text: sile_hir::format_kernel(kernel),
        micros: started.elapsed().as_micros(),
    });

    let started = Instant::now();
    let typed = run_hir_analysis_pipeline(kernel, ACTIVE_HIR_TYPECK_PIPELINE)
        .map_err(|err| sile_core::Error::Shape(err.to_string()))?;
    let typed_micros = started.elapsed().as_micros();
    stages.push(Stage {
        label: "hir-01-typed".into(),
        text: format_typed_kernel(&typed),
        micros: typed_micros,
    });

    // ── Tile IR, one snapshot per pass ─────────────────────────────
    //
    // The first snapshot is the text cutile-rs prints, not what the backends'
    // IR happens to say: that is the text the alignment fixtures compare, so it
    // is the one worth recording.
    let started = Instant::now();
    let module: sile_tile_ir::ir::Module = sile_tile_ir::cutile::emit::emit(&typed)
        .map_err(|err| sile_core::Error::Shape(err))?
        .0;
    stages.push(Stage {
        label: "tileir-00-lowered".into(),
        text: crate::tileir_text::emit(&typed)
            .map_err(sile_core::Error::Shape)?,
        micros: started.elapsed().as_micros(),
    });

    // The text is the whole point of this mode, so the dump ends here: the
    // later stages are a full compile of every backend, which nothing compares.
    if tileir_only() {
        for stage in &stages {
            fs::write(dir.join(format!("{}.txt", stage.label)), &stage.text)?;
        }
        return Ok(());
    }

    // A pass that rejects the IR still leaves a result worth reading, so the
    // failure becomes a stage of its own and every stage so far is written.
    let mut failure: Option<String> = None;

    // ── LSIR, one snapshot per pass ────────────────────────────
    let started = Instant::now();
    let mut lsir: LsirFunction =
        sile_lsir::lower_module_to_lsir(&module).map_err(sile_core::Error::Compile)?;
    let lowered_micros = started.elapsed().as_micros();
    stages.push(Stage {
        label: "lsir-00-lowered".into(),
        text: sile_lsir::format_lsir(&lsir),
        micros: lowered_micros,
    });

    for (index, kind) in ACTIVE_LSIR_PIPELINE.iter().enumerate() {
        if failure.is_some() {
            break;
        }
        let started = Instant::now();
        match sile_lsir::run_lsir_pipeline(lsir.clone(), &[*kind]) {
            Ok(next) => lsir = next,
            Err(err) => {
                let label = format!("lsir-{:02}-{}-FAILED", index + 1, lsir_pass_name(*kind));
                stages.push(Stage {
                    label,
                    text: err.clone(),
                    micros: started.elapsed().as_micros(),
                });
                failure = Some(err);
                break;
            }
        }
        let micros = started.elapsed().as_micros();
        let label = format!("lsir-{:02}-{}", index + 1, lsir_pass_name(*kind));
        stages.push(Stage {
            label,
            text: sile_lsir::format_lsir(&lsir),
            micros,
        });
    }

    // ── Mapping, one snapshot per stage ────────────────────────
    //
    // The mapping is per-target, and WGSL is the one parallel target, so the
    // dump follows the policy WGSL would run. The mapping here is for reading:
    // the backend sources below compile from the unmapped function, because
    // `compile_backend` maps per target itself and the serial targets must not
    // see a function that already carries a thread index.
    let mapping_policy =
        sile_backend::mapping::policy_for(CodegenTarget::Wgsl, &lsir);
    if mapping_policy.is_parallel() {
        let mut mapped = lsir.clone();
        for (index, stage) in sile_lsir::mapping::MAPPING_STAGES.iter().enumerate() {
            if failure.is_some() {
                break;
            }
            let started = Instant::now();
            match sile_lsir::mapping::run_stage(mapped.clone(), &mapping_policy, index) {
                Ok(next) => {
                    mapped = next;
                }
                Err(err) => {
                    stages.push(Stage {
                        label: format!("lsir-{}-mapping-{stage}-FAILED", index + 15),
                        text: err.clone(),
                        micros: started.elapsed().as_micros(),
                    });
                    // The dump records the refusal and stops the mapping walk;
                    // it is not a compile failure. The backend sources below
                    // map (or fall back to serial) on their own.
                    break;
                }
            }
            stages.push(Stage {
                label: format!("lsir-{}-mapping-{stage}", index + 15),
                text: sile_lsir::format_lsir(&mapped),
                micros: started.elapsed().as_micros(),
            });
        }
    }

    // ── Backend sources ───────────────────────────────────────────
    // Every target is generated from the same final LSIR, so the three (or
    // four) files next to each other are the whole multi-target surface of one
    // kernel.
    for target in [
        CodegenTarget::C,
        CodegenTarget::Metal,
        CodegenTarget::Wgsl,
        CodegenTarget::Ptx,
    ] {
        let started = Instant::now();
        match compile_backend(&lsir, target) {
            Ok(artifact) => {
                let (name, source) = match artifact {
                    sile_backend::BackendArtifact::CSource(source) => ("c", source),
                    sile_backend::BackendArtifact::MetalSource { source, .. } => ("metal", source),
                    sile_backend::BackendArtifact::WgslSource(source) => ("wgsl", source),
                    sile_backend::BackendArtifact::PtxSource(source) => ("ptx", source),
                };
                stages.push(Stage {
                    label: format!("source-{name}"),
                    text: source,
                    micros: started.elapsed().as_micros(),
                });
            }
            // A target that cannot spell this kernel is a result worth reading,
            // not a reason to abandon the dump.
            Err(err) => stages.push(Stage {
                label: format!("source-{}-FAILED", target_name(target)),
                text: format!("{err}"),
                micros: started.elapsed().as_micros(),
            }),
        }
    }

    for stage in &stages {
        fs::write(dir.join(format!("{}.txt", stage.label)), &stage.text)?;
    }

    // One row per stage: size and shape, which is what a pass is judged by.
    let mut metrics = String::from("stage\tbytes\tlines\tblocks\tinsts\talloca\tloops\n");
    for stage in &stages {
        let _ = writeln!(
            &mut metrics,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            stage.label,
            stage.text.len(),
            stage.text.lines().count(),
            stage.text.lines().filter(|line| is_block_label(line)).count(),
            stage.text.lines().filter(|line| is_instruction(line)).count(),
            stage.text.matches("alloca").count(),
            stage.text.matches("phi ").count(),
        );
    }
    fs::write(dir.join("metrics.tsv"), metrics)?;

    let mut timing = String::from("stage\tmicros\n");
    for stage in &stages {
        let _ = writeln!(&mut timing, "{}\t{}", stage.label, stage.micros);
    }
    fs::write(dir.join("timing.tsv"), timing)?;

    if let Some(err) = failure {
        return Err(sile_core::Error::Shape(err));
    }

    Ok(())
}

fn target_name(target: CodegenTarget) -> &'static str {
    match target {
        CodegenTarget::C => "c",
        CodegenTarget::Metal => "metal",
        CodegenTarget::Wgsl => "wgsl",
        CodegenTarget::Ptx => "ptx",
    }
}

fn lsir_pass_name(kind: LsirPassKind) -> &'static str {
    match kind {
        LsirPassKind::VerifyInput => "verify-input",
        LsirPassKind::Canonicalize => "canonicalize",
        LsirPassKind::LoopSimplify => "loop-simplify",
        LsirPassKind::PhiAlias => "phi-alias",
        LsirPassKind::Cse => "cse",
        LsirPassKind::InstCombine => "instcombine",
        LsirPassKind::LoopUnroll => "loop-unroll",
        LsirPassKind::Licm => "licm",
        LsirPassKind::Sroa => "sroa",
        LsirPassKind::ScalarPromote => "scalar-promote",
        LsirPassKind::Dce => "dce",
        LsirPassKind::VerifyOutput => "verify-output",
    }
}

/// A block label line: `name:` at the start of a line, with no indentation.
fn is_block_label(line: &str) -> bool {
    !line.starts_with(' ')
        && line.ends_with(':')
        && !line.contains(' ')
        && !line.starts_with(';')
        && !line.starts_with('}')
}

/// An instruction line in either printed form: one result or one effect.
fn is_instruction(line: &str) -> bool {
    let trimmed = line.trim_start();
    (trimmed.starts_with('%') && trimmed.contains(" = "))
        || trimmed.starts_with("store ")
        || trimmed.starts_with("br ")
        || trimmed.starts_with("ret ")
        || trimmed.starts_with("cf.")
}

/// A typed kernel's own text: the kernel plus the type of every binding.
///
/// The type table is what type checking added, and a pass that reasons about a
/// value needs it: the kernel text alone names bindings without widths.
fn format_typed_kernel(typed: &TypedKernel) -> String {
    let mut out = sile_hir::format_kernel(&typed.kernel);
    if typed.locals.is_empty() {
        return out;
    }
    out.push_str("  locals {\n");
    for (name, ty) in &typed.locals {
        let _ = writeln!(&mut out, "    {name}: {}", sile_hir::print::format_type(ty));
    }
    out.push_str("  }\n");
    out
}

/// A directory that does not collide with one an earlier kernel made.
///
/// Two instantiations of the same generic kernel share a name and differ in
/// their constants, so the second one gets a numeric suffix instead of
/// overwriting the first.
fn unique_kernel_dir(root: &Path, name: &str) -> PathBuf {
    let first = root.join(name);
    if !first.exists() {
        return first;
    }
    for suffix in 1..1000 {
        let candidate = root.join(format!("{name}-{suffix}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    root.join(format!("{name}-overflow"))
}
