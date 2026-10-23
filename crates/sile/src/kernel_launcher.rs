use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use sile_backend::{BackendArtifact, cpu::CpuBackend, ptx::PtxBackend, wgsl::runtime::WgpuBackend};
#[cfg(target_os = "macos")]
use sile_backend::metal::MetalBackend;
use sile_core::{Device, KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;
use sile_lsir::format_lsir;

use crate::compiler::{CodegenTarget, compile_backend, compile_kernel_to_module};

const PRINT_TILE_IR_ENV: &str = "SILE_PRINT_TILEIR";
const PRINT_LSIR_ENV: &str = "SILE_PRINT_LSIR";
const PRINT_SOURCE_ENV: &str = "SILE_PRINT_SOURCE";

/// One compiled kernel, shared by every launch of the same source.
///
/// Compilation is a pure function of the HIR: the device only chooses which
/// backend consumes the LSIR, never what the IR is. So the result is cached
/// once per distinct kernel and every target reuses it. Without this, a launch
/// inside a loop re-runs the whole front end and both lowering stages on every
/// iteration.
type CompiledParts = (
    sile_hir::typeck::TypedKernel,
    sile_tile_ir::ir::Module,
    sile_lsir::Function,
    Vec<sile_hir::passes::predicate::LaunchCheck>,
);

fn compile_cache() -> &'static Mutex<Vec<(Kernel, Arc<CompiledParts>)>> {
    static CACHE: OnceLock<Mutex<Vec<(Kernel, Arc<CompiledParts>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn compiled_parts(kernel: &Kernel) -> Result<Arc<CompiledParts>> {
    // The kernel itself is the cache key: a process holds a handful of
    // kernels, so a `PartialEq` walk over that many entries beats formatting
    // a multimegabyte debug form and hashing it on every launch.
    {
        let cache = compile_cache().lock().unwrap();
        if let Some((_, hit)) = cache.iter().find(|(known, _)| known == kernel) {
            return Ok(Arc::clone(hit));
        }
    }

    // A dump request is served before the cache is consulted for a value, but
    // after it: the second launch of a kernel has nothing new to record, and
    // the first has already recorded everything. `SILE_DUMP_DIR` therefore
    // costs one compile's worth of file writes per distinct kernel.
    if let Some(root) = crate::dump::dump_dir_from_env() {
        let mut reported = dumped_kernels().lock().unwrap();
        // The full debug form: two instantiations of one kernel (the causal
        // and unmasked `fmha`) share a name but dump different IR.
        if reported.insert(format!("{kernel:?}")) {
            drop(reported);
            crate::dump::dump_kernel(kernel, &root)?;
        }
    }

    let parts = Arc::new(compile_kernel_to_module(kernel)?);
    compile_cache()
        .lock()
        .unwrap()
        .push((kernel.clone(), Arc::clone(&parts)));
    Ok(parts)
}

/// The strides a buffer of `shape` has in row-major order.
fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    let mut acc = 1i64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = acc;
        acc *= shape[axis];
    }
    strides
}

/// The kernels this process has already dumped, so a repeated launch of the
/// same kernel does not rewrite the same files.
fn dumped_kernels() -> &'static Mutex<std::collections::HashSet<String>> {
    static DUMPED: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    DUMPED.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

pub struct KernelLauncher<'a> {
    kernel: Kernel,
    args: Vec<KernelArg<'a>>,
    /// Float scalar arguments, in kernel parameter order.
    ///
    /// A scalar travels beside the buffers rather than inside them, so the
    /// launch keeps the lists apart all the way to the backend. An integer
    /// scalar needs a list of its own: the two do not share a width.
    scalars: Vec<f32>,
    scalars_i64: Vec<i64>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(
        mut kernel: Kernel,
        args: Vec<KernelArg<'a>>,
        scalars: Vec<f32>,
        scalars_i64: Vec<i64>,
    ) -> Self {
        // The buffer's strides are a property of the launch, not of the kernel
        // source, so they are recorded here: a lowering that prints the buffer
        // address the way cutile does needs them, and the HIR is where the
        // parameter order and the buffer list meet.
        let mut buffers = args.iter();
        for param in &mut kernel.params {
            if matches!(param.ty, sile_hir::Type::Scalar(_)) {
                continue;
            }
            if let Some(arg) = buffers.next() {
                param.strides = Some(row_major_strides(arg.shape));
            }
        }
        Self {
            kernel,
            args,
            scalars,
            scalars_i64,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The grid this launch runs on: the one `.grid()` named, or the one the
    /// kernel's output shapes derive.
    ///
    /// A kernel whose output names every extent — `z: Tensor<f32, [BM, BN]>` —
    /// tiles its buffer, and the launch is one program per tile: the grid is
    /// the buffer extent over the declared extent, per axis. This is the check
    /// CuTile runs on a host partition, derived from the same two shapes; a
    /// kernel with a dynamic output still names its grid by hand.
    fn resolve_grid(&self, kernel: &Kernel) -> Result<[u32; 3]> {
        let constants: HashMap<&str, i64> = kernel
            .const_params
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        // Scalars do not travel in `args`, so the buffer parameters are the
        // non-scalar ones, in order — the same rule the backends bind by.
        let buffer_params = kernel
            .params
            .iter()
            .filter(|param| !matches!(param.ty, sile_hir::Type::Scalar(_)));
        let mut derived: Option<[u32; 3]> = None;
        for (param, arg) in buffer_params.zip(&self.args) {
            if param.kind != sile_hir::ParamKind::Output {
                continue;
            }
            let sile_hir::Type::Tensor { shape, .. } = &param.ty else {
                continue;
            };
            let Some(dims) = constant_dims(shape, &constants) else {
                continue;
            };
            if dims.len() != arg.shape.len() {
                return Err(sile_core::Error::Shape(format!(
                    "kernel output `{}` declares rank {}, but the buffer passed has rank {}",
                    param.name,
                    dims.len(),
                    arg.shape.len()
                )));
            }
            let mut candidate = [1u32; 3];
            for (axis, (extent, declared)) in arg.shape.iter().zip(&dims).enumerate() {
                if extent % declared != 0 {
                    return Err(sile_core::Error::Shape(format!(
                        "kernel output `{}`: buffer extent {extent} is not divisible by the \
                         declared extent {declared} on axis {axis}",
                        param.name
                    )));
                }
                candidate[axis] = (extent / declared) as u32;
            }
            if let Some(previous) = derived {
                if previous != candidate {
                    return Err(sile_core::Error::Shape(format!(
                        "two outputs derive different grids: {previous:?} and {candidate:?}"
                    )));
                }
            }
            derived = Some(candidate);
        }

        match (self.grid, derived) {
            (Some(explicit), Some(derived)) if explicit != derived => Err(
                sile_core::Error::Shape(format!(
                    "grid {explicit:?} disagrees with the one the output shapes derive, {derived:?}"
                )),
            ),
            (Some(explicit), _) => Ok(explicit),
            (None, Some(derived)) => Ok(derived),
            (None, None) => Err(sile_core::Error::Shape(
                "grid not set and no output names every extent, so the grid cannot be derived; \
                 pass `.grid(...)`"
                    .into(),
            )),
        }
    }

    pub fn apply(self, stream: &Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self.resolve_grid(&self.kernel)?,
        };

        let parts = compiled_parts(&self.kernel)?;
        let (_typed, tile_ir_module, lsir_func, checks) = parts.as_ref();

        // The checks came out of the same Tile IR emission the cache holds;
        // validating them per launch only reads the args.
        validate_launch_checks(checks, &self.args)?;

        if should_print(PRINT_TILE_IR_ENV) {
            eprintln!("{}", tile_ir_module.to_mlir_text());
        }
        if should_print(PRINT_LSIR_ENV) {
            eprintln!("{}", format_lsir(lsir_func));
        }

        // A dump that stops at the Tile IR has nothing to run: the launch would
        // execute the kernel on the device, which is minutes of work and, in
        // the CPU backend, a crash. The IR is already written.
        if crate::dump::tileir_only() {
            return Ok(());
        }

        match stream.device() {
            Device::Cpu(_) => {
                maybe_print_backend_source(&self.kernel, lsir_func, CodegenTarget::C)?;
                CpuBackend::new().execute_llir(
                    lsir_func,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
            }
            #[cfg(target_os = "macos")]
            Device::Metal(_) => {
                let typed = _typed;
                maybe_print_backend_source(&self.kernel, lsir_func, CodegenTarget::Metal)?;
                let backend = MetalBackend::new()?;
                // `args` holds the buffers only, so the output flags have to
                // line up with it: a scalar parameter is not in `args` and
                // would shift every output that follows it.
                let outputs: Vec<bool> = typed
                    .kernel
                    .params
                    .iter()
                    .filter(|param| !matches!(param.ty, sile_hir::Type::Scalar(_)))
                    .map(|param| param.kind == sile_hir::ParamKind::Output)
                    .collect();
                let artifact = compiled_artifact(&self.kernel, lsir_func, CodegenTarget::Metal)?;
                let BackendArtifact::MetalSource { source, threads } = artifact.as_ref() else {
                    unreachable!("the Metal target spells MSL source");
                };
                backend.execute_source(
                    lsir_func,
                    source,
                    &outputs,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    *threads,
                )
            }
            #[cfg(not(target_os = "macos"))]
            Device::Metal(_) => Err(sile_core::Error::UnsupportedBackend(
                "the Metal backend is only available on macOS; use C or WGPU",
            )),
            Device::Wgpu(_) => {
                maybe_print_backend_source(&self.kernel, lsir_func, CodegenTarget::Wgsl)?;
                let backend = WgpuBackend::shared()?;
                let source = compiled_artifact(&self.kernel, lsir_func, CodegenTarget::Wgsl)?;
                let BackendArtifact::WgslSource(source) = source.as_ref() else {
                    unreachable!("the WGSL target spells WGSL source");
                };
                backend.execute_source(
                    source,
                    lsir_func,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
            }
            Device::Cuda(_) => {
                maybe_print_backend_source(&self.kernel, lsir_func, CodegenTarget::Ptx)?;
                let backend = PtxBackend::new()?;
                backend.execute_llir(
                    lsir_func,
                    &self.args,
                    &self.scalars,
                    &self.scalars_i64,
                    &launch,
                    stream,
                )
            }
        }
    }
}

fn should_print(env: &str) -> bool {
    std::env::var_os(env).is_some()
}

/// The dimensions of a shape, when every one of them is a known constant.
///
/// A shape dimension is a literal, a const parameter, or `-1` for the dynamic
/// extent a kernel reads off the buffer. The first two resolve to a number
/// here; the last one and any unresolved name leave the shape unable to name
/// a grid, which is what `None` reports.
fn constant_dims(shape: &sile_hir::ShapeExpr, constants: &HashMap<&str, i64>) -> Option<Vec<i64>> {
    let dims = match shape {
        sile_hir::ShapeExpr::Tuple(dims) => dims,
        sile_hir::ShapeExpr::Constant(value) => return Some(vec![*value]),
        sile_hir::ShapeExpr::Symbol(name) => return constants.get(name.as_str()).copied().map(|v| vec![v]),
        sile_hir::ShapeExpr::Dynamic => return None,
    };
    let mut resolved = Vec::with_capacity(dims.len());
    for dim in dims {
        match dim {
            sile_hir::ShapeExpr::Constant(value) => resolved.push(*value),
            sile_hir::ShapeExpr::Symbol(name) => {
                resolved.push(*constants.get(name.as_str())?)
            }
            sile_hir::ShapeExpr::Dynamic | sile_hir::ShapeExpr::Tuple(_) => return None,
        }
    }
    Some(resolved)
}

fn maybe_print_backend_source(
    kernel: &Kernel,
    lsir_func: &sile_lsir::Function,
    target: CodegenTarget,
) -> Result<()> {
    if !should_print(PRINT_SOURCE_ENV) {
        return Ok(());
    }

    let artifact = compiled_artifact(kernel, lsir_func, target)?;
    match artifact.as_ref() {
        BackendArtifact::CSource(source)
        | BackendArtifact::MetalSource { source, .. }
        | BackendArtifact::WgslSource(source)
        | BackendArtifact::PtxSource(source) => {
            eprintln!("{}", source);
        }
    }
    Ok(())
}

/// The backend artifact of one kernel on one target, compiled once.
///
/// The source is a pure function of the LSIR the kernel lowers to, and that
/// lowers from the same kernel every launch, so the mapping, the emitter and
/// the WGSL validation run once per distinct kernel and target. Without this,
/// a launch inside a loop re-runs all three on every iteration — tens of
/// milliseconds against a kernel that runs in one.
type SourceCache = Vec<(Kernel, CodegenTarget, std::sync::Arc<BackendArtifact>)>;

fn source_cache() -> &'static Mutex<SourceCache> {
    static CACHE: OnceLock<Mutex<SourceCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn compiled_artifact(
    kernel: &Kernel,
    lsir_func: &sile_lsir::Function,
    target: CodegenTarget,
) -> Result<std::sync::Arc<BackendArtifact>> {
    {
        let cache = source_cache().lock().unwrap();
        if let Some((_, _, hit)) = cache
            .iter()
            .find(|(known, known_target, _)| known == kernel && known_target == &target)
        {
            return Ok(std::sync::Arc::clone(hit));
        }
    }
    let artifact = std::sync::Arc::new(compile_backend(lsir_func, target)?);
    source_cache()
        .lock()
        .unwrap()
        .push((kernel.clone(), target, std::sync::Arc::clone(&artifact)));
    Ok(artifact)
}

/// Evaluate the access checks the compiler moved out of the kernel.
///
/// Each predicate names its operands by parameter position and axis, so the
/// resolver reads the buffer the launch was handed. It fails closed: an atom
/// this launcher cannot resolve is an error, not a silent pass — a check that
/// nothing evaluates is worse than no check, because the kernel relies on it.
fn validate_launch_checks(
    checks: &[sile_hir::passes::predicate::LaunchCheck],
    args: &[KernelArg<'_>],
) -> Result<()> {
    use sile_hir::passes::predicate::Atom;
    for check in checks {
        let resolved = check.predicate.eval(&|atom: &Atom| match atom {
            Atom::Dim { param, axis } | Atom::ViewExtent { param, axis } => args
                .get(*param)
                .and_then(|arg| arg.shape.get(*axis))
                .map(|extent| *extent as i64),
            Atom::TileCount { param, axis, tile } => {
                let extent = args
                    .get(*param)
                    .and_then(|arg| arg.shape.get(*axis))
                    .copied()? as i64;
                if *tile < 1 {
                    return None;
                }
                Some((extent + *tile as i64 - 1) / *tile as i64)
            }
            Atom::NumTileBlocks(_) | Atom::TileBlockId(_) | Atom::Iv(_) => None,
        });
        match resolved {
            Some(true) => {}
            Some(false) => {
                return Err(sile_core::Error::Shape(format!(
                    "{} (checked at launch)",
                    check.cause
                )))
            }
            None => {
                return Err(sile_core::Error::Shape(format!(
                    "{}: the check names a value this launch cannot resolve",
                    check.cause
                )))
            }
        }
    }
    Ok(())
}
