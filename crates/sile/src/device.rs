use crate::{Error, Result, Stream};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuDeviceOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuDeviceOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu(CpuDeviceOptions),
    Metal(MetalDeviceOptions),
    Cuda(CudaDeviceOptions),
    Wgpu(WgpuDeviceOptions),
}

impl Device {
    pub fn cpu() -> Self {
        Self::Cpu(CpuDeviceOptions)
    }
    pub fn metal() -> Self {
        Self::Metal(MetalDeviceOptions)
    }
    pub fn cuda() -> Self {
        Self::Cuda(CudaDeviceOptions)
    }
    pub fn wgpu() -> Self {
        Self::Wgpu(WgpuDeviceOptions)
    }

    pub fn default() -> Result<Self> {
        match std::env::var("SILE_DEVICE").ok().as_deref() {
            None | Some("C") | Some("CPU") => Ok(Self::cpu()),
            Some("METAL") => Ok(Self::metal()),
            Some("CUDA") => Ok(Self::cuda()),
            Some("WGPU") => Ok(Self::wgpu()),
            Some(other) => Err(Error::InvalidDevice(other.to_string())),
        }
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Ok(Stream::new(self.clone()))
    }
}
