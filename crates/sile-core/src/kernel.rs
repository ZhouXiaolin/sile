use crate::device::Device;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
}

/// One buffer argument, as the host holds it.
///
/// The pointers are raw bytes because the element width is the buffer's own:
/// an f32 buffer and a half buffer travel through the same launch path, and
/// every backend sizes its uploads and read-backs in bytes.
#[derive(Clone, Copy, Debug)]
pub struct KernelArg<'a> {
    /// The buffer's bytes, as the host holds them.
    pub ptr: *const u8,
    /// The same storage, open for writing: a kernel's output is read back
    /// through it.
    pub mut_ptr: *mut u8,
    /// Bytes of one element.
    pub elem_size: usize,
    pub shape: &'a [i64],
    pub device: &'a Device,
}

impl KernelArg<'_> {
    /// The element count the shape names.
    pub fn len(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    /// The bytes of the whole buffer.
    pub fn len_in_bytes(&self) -> usize {
        self.len() * self.elem_size
    }
}
