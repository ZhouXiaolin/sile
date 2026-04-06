use sile_core::{Error, Result};
use sile_llir::Function as LlirFunction;

pub fn run(llir: &LlirFunction, stage: &str) -> Result<()> {
    sile_llir::passes::verify::verify_function(llir)
        .map_err(|err| Error::Compile(format!("{stage}: {err}")))
}
