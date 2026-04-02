use crate::{Result, Stream};

#[derive(Clone, Debug)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn cpu() -> Self {
        Self::Cpu
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Ok(Stream::new(self.clone()))
    }
}
