use crate::{Device, KernelSpec, Result};

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
    spec: &'static crate::spec::KernelSpecRef,
    args: Vec<KernelArg<'a>>,
    grid: Option<[u32; 3]>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(spec: &'static crate::spec::KernelSpecRef, args: Vec<KernelArg<'a>>) -> Self {
        Self {
            spec,
            args,
            grid: None,
        }
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some([grid.0, grid.1, grid.2]);
        self
    }

    pub fn spec_ref(&self) -> &'static crate::spec::KernelSpecRef {
        self.spec
    }

    pub fn apply(self, stream: &crate::Stream) -> Result<()> {
        let launch = LaunchConfig {
            grid: self.grid.ok_or_else(|| crate::Error::Shape("grid not set".into()))?,
        };
        let spec = KernelSpec::from(self.spec);
        spec.validate_launch(&self.args, &launch)?;
        let _ = stream;
        Err(crate::Error::UnsupportedBackend("backend not wired yet"))
    }
}
