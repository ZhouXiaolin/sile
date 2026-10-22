use sile_core::Result;
use sile_lsir::Function as LsirFunction;

mod emit;
mod verify;

pub mod mapping;

pub mod cpu;
pub mod metal;
pub mod ptx;
pub mod wgsl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTarget {
    C,
    Metal,
    Wgsl,
    Ptx,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendArtifact {
    CSource(String),
    MetalSource(String),
    WgslSource(String),
    PtxSource(String),
}

pub fn compile(lsir: &LsirFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    // The structured emitter walks the CFG recursively, and a mapped kernel's
    // tree steps and skeleton chains nest it close to a thousand frames deep.
    // The compile runs on a roomy stack instead of asking every CFG to stay
    // shallow enough for the default one.
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn_scoped(scope, || compile_inner(lsir, target))
            .expect("the compile thread spawns")
            .join()
            .expect("the compile thread's panic propagates")
    })
}

fn compile_inner(lsir: &LsirFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    // The contract is checked on both sides of the mapping: the input must be
    // the shape the lowering produces — tile intrinsics whole, no threads, no
    // barriers — and the output must be the shape the mapping promised — no
    // tile intrinsic left unmaterialized, shared storage and barriers paired.
    // A check on one side alone would let a pipeline change slip through
    // silently in either direction.
    let policy = mapping::policy_for(target, lsir);
    verify::for_target(lsir, target, "Backend compile input")?;
    // A kernel the parallel mapping rejects — a slot shape whose owner map
    // would break across threads — falls back to the serial one, which is
    // always legal: the tile runs on one invocation, exactly as the
    // pre-mapping pipeline did.
    let (mapped, policy) = if policy.is_parallel() {
        let fallback = mapping::policy_with_vector(policy.vector);
        match sile_lsir::mapping::run(lsir.clone(), &policy) {
            Ok(mapped) => match verify::parallel_contract(&mapped, "Backend mapped kernel") {
                Ok(()) => (Some(mapped), policy),
                Err(_) => (None, fallback),
            },
            Err(_) => (None, fallback),
        }
    } else {
        (None, policy)
    };
    let lsir = mapped.as_ref().unwrap_or(lsir);
    let artifact = emit::run(lsir, target, &policy)?;
    // WGSL is the one target with a real parser and validator in the tree, so
    // the IR to source step is checked here rather than trusting the emitter.
    // Without this, a kernel that only the GPU driver would reject ships as a
    // string and the first sign of trouble is a runtime failure.
    if let BackendArtifact::WgslSource(source) = &artifact {
        wgsl::validate(source)?;
    }
    Ok(artifact)
}
