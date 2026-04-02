use crate::{Device, Result};

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
}

impl Tensor<f32> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        let shape = shape.into();
        let len = shape.iter().product::<i64>() as usize;
        Ok(Self {
            data: vec![0.0; len],
            shape,
            device: device.clone(),
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}
