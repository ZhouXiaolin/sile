use sile_core::{KernelArg, LaunchConfig, Result, Stream};

use crate::ExecutableKernel;

pub trait Backend: Send + Sync {
    fn execute(
        &self,
        kernel: &ExecutableKernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()>;
}
