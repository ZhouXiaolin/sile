use crate::{Device, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct KernelArg<'a> {
    pub ptr: *const f32,
    pub mut_ptr: *mut f32,
    pub shape: &'a [i64],
    pub device: &'a Device,
}

pub struct KernelLauncher<'a> {
    kernel: &'static crate::hir::Kernel,
    args: Vec<KernelArg<'a>>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(kernel: &'static crate::hir::Kernel, args: Vec<KernelArg<'a>>) -> Self {
        Self {
            kernel,
            args,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn kernel(&self) -> &'static crate::hir::Kernel {
        self.kernel
    }

    pub fn apply(self, stream: &crate::Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self.grid.ok_or_else(|| crate::Error::Shape("grid not set".into()))?,
        };
        crate::backend::for_device(stream.device())?
            .launch_kernel(self.kernel, &self.args, &launch, stream)
    }
}
