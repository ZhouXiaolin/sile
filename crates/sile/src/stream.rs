use crate::Device;

#[derive(Clone, Debug)]
pub struct Stream {
    device: Device,
}

impl Stream {
    pub(crate) fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
