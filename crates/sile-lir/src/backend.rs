use crate::ir::Function;
use sile_core::{KernelArg, LaunchConfig, Result, Stream};

pub trait Backend: Send + Sync {
    fn execute(
        &self,
        func: &Function,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()>;
}
