use crate::device::Device;

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
