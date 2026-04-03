use crate::ir::Function;
use sile_core::{KernelArg, LaunchConfig, Result, Stream};
use sile_hir::Kernel;

pub trait Backend: Send + Sync {
    fn execute(
        &self,
        func: &Function,
        kernel: &Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()>;
}
