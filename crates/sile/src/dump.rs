//! Stage-by-stage IR dumping for analysis, off by default.
//!
//! A launch runs one pipeline: HIR, then Tile IR, then LLVM IR, then a text
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
use sile_llvm_ir::{Function as LlvmIrFunction, LlvmIrPassKind};
use sile_tile_ir::{TileIrFunction, TileIrPassKind};

use crate::compiler::{
    ACTIVE_HIR_TYPECK_PIPELINE, ACTIVE_LLVM_IR_PIPELINE, ACTIVE_TILE_IR_PIPELINE, CodegenTarget,
    compile_backend, run_hir_analysis_pipeline,
};

/// The environment variable that turns dumping on and names its directory.
pub const DUMP_DIR_ENV: &str = "SILE_DUMP_DIR";

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
    let started = Instant::now();
    let mut tile_ir: TileIrFunction = sile_tile_ir::lower_to_tile_ir(&typed);
    stages.push(Stage {
        label: "tileir-00-lowered".into(),
        text: sile_tile_ir::format_tile_ir(&tile_ir),
        micros: started.elapsed().as_micros(),
    });

    for (index, kind) in ACTIVE_TILE_IR_PIPELINE.iter().enumerate() {
        let started = Instant::now();
        tile_ir = sile_tile_ir::run_tile_ir_pipeline(tile_ir, &[*kind])
            .map_err(sile_core::Error::Shape)?;
        let micros = started.elapsed().as_micros();
        let label = format!("tileir-{:02}-{}", index + 1, tile_ir_pass_name(*kind));
        stages.push(Stage {
            label,
            text: sile_tile_ir::format_tile_ir(&tile_ir),
            micros,
        });
    }

    // ── LLVM IR, one snapshot per pass ────────────────────────────
    let started = Instant::now();
    let mut llvm_ir: LlvmIrFunction = sile_llvm_ir::lower_tile_ir_to_llvm_ir(&tile_ir);
    let lowered_micros = started.elapsed().as_micros();
    stages.push(Stage {
        label: "llvmir-00-lowered".into(),
        text: sile_llvm_ir::format_llvm_ir(&llvm_ir),
        micros: lowered_micros,
    });

    for (index, kind) in ACTIVE_LLVM_IR_PIPELINE.iter().enumerate() {
        let started = Instant::now();
        llvm_ir = sile_llvm_ir::run_llvm_ir_pipeline(llvm_ir, &[*kind])
            .map_err(sile_core::Error::Shape)?;
        let micros = started.elapsed().as_micros();
        let label = format!("llvmir-{:02}-{}", index + 1, llvm_ir_pass_name(*kind));
        stages.push(Stage {
            label,
            text: sile_llvm_ir::format_llvm_ir(&llvm_ir),
            micros,
        });
    }

    // ── Backend sources ───────────────────────────────────────────
    // Every target is generated from the same final LLVM IR, so the three (or
    // four) files next to each other are the whole multi-target surface of one
    // kernel.
    for target in [
        CodegenTarget::C,
        CodegenTarget::Metal,
        CodegenTarget::Wgsl,
        CodegenTarget::Ptx,
    ] {
        let started = Instant::now();
        match compile_backend(&llvm_ir, target) {
            Ok(artifact) => {
                let (name, source) = match artifact {
                    sile_backend::BackendArtifact::CSource(source) => ("c", source),
                    sile_backend::BackendArtifact::MetalSource(source) => ("metal", source),
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

fn tile_ir_pass_name(kind: TileIrPassKind) -> &'static str {
    match kind {
        TileIrPassKind::VerifyInput => "verify-input",
        TileIrPassKind::PointwiseFusion => "pointwise-fusion",
        TileIrPassKind::TileExprCse => "tile-expr-cse",
        TileIrPassKind::Dce => "dce",
        TileIrPassKind::VerifyOutput => "verify-output",
    }
}

fn llvm_ir_pass_name(kind: LlvmIrPassKind) -> &'static str {
    match kind {
        LlvmIrPassKind::VerifyInput => "verify-input",
        LlvmIrPassKind::Canonicalize => "canonicalize",
        LlvmIrPassKind::LoopSimplify => "loop-simplify",
        LlvmIrPassKind::PhiAlias => "phi-alias",
        LlvmIrPassKind::Cse => "cse",
        LlvmIrPassKind::InstCombine => "instcombine",
        LlvmIrPassKind::LoopUnroll => "loop-unroll",
        LlvmIrPassKind::Licm => "licm",
        LlvmIrPassKind::Sroa => "sroa",
        LlvmIrPassKind::Dce => "dce",
        LlvmIrPassKind::VerifyOutput => "verify-output",
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
