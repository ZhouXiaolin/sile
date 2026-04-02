#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend not implemented: {0}")]
    UnsupportedBackend(&'static str),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

pub type Result<T> = std::result::Result<T, Error>;
