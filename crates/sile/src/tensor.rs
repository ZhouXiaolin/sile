use crate::{Device, Error, Result};

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
}

impl Tensor<f32> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 0.0, device)
    }

    pub fn ones(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 1.0, device)
    }

    pub fn from_vec(
        data: Vec<f32>,
        shape: impl Into<Vec<i64>>,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let len = shape.iter().product::<i64>() as usize;
        if data.len() != len {
            return Err(Error::Shape(format!(
                "expected {len} elements, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            shape,
            device: device.clone(),
        })
    }

    fn filled(shape: Vec<i64>, value: f32, device: &Device) -> Result<Self> {
        let len = shape.iter().product::<i64>() as usize;
        Ok(Self {
            data: vec![value; len],
            shape,
            device: device.clone(),
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }
    pub fn to_vec(&self, _stream: &crate::Stream) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}
