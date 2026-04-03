#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid SILE_DEVICE value: {0}")]
    InvalidDevice(String),
    #[error("backend not implemented: {0}")]
    UnsupportedBackend(&'static str),
    #[error("compile failed: {0}")]
    Compile(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

pub type Result<T> = std::result::Result<T, Error>;
