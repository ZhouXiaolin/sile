pub mod cpu_c;

use crate::{kernel::LaunchConfig, kernel::KernelArg, Result, Stream};

pub trait Backend: Send + Sync {
    fn launch_kernel(
        &self,
        kernel: &crate::hir::Kernel,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
        stream: &Stream,
    ) -> Result<()>;
}

pub fn for_device(device: &crate::Device) -> Result<Box<dyn Backend>> {
    match device {
        crate::Device::Cpu(_) => Ok(Box::new(cpu_c::CpuBackend::new())),
        crate::Device::Metal(_) => Err(crate::Error::UnsupportedBackend("metal")),
        crate::Device::Cuda(_) => Err(crate::Error::UnsupportedBackend("cuda")),
        crate::Device::Wgpu(_) => Err(crate::Error::UnsupportedBackend("wgpu")),
    }
}
