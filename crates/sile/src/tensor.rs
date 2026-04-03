use crate::{Device, Error, Result};

#[derive(Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
}

impl<T: Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }
}

impl<T: Clone> Tensor<T> {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}

impl Tensor<f32> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 0.0, device)
    }

    pub fn ones(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 1.0, device)
    }

    pub fn random(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let len = shape.iter().product::<i64>() as usize;
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..len).map(|_| rng.random_range(0.0..1.0)).collect();
        Ok(Self {
            data,
            shape,
            device: device.clone(),
        })
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
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

    pub fn as_kernel_arg(&self) -> crate::kernel::KernelArg<'_> {
        crate::kernel::KernelArg {
            ptr: self.as_ptr(),
            mut_ptr: self.as_ptr() as *mut f32,
            shape: &self.shape,
            device: &self.device,
        }
    }

    pub fn as_kernel_arg_mut(&mut self) -> crate::kernel::KernelArg<'_> {
        crate::kernel::KernelArg {
            ptr: self.as_ptr(),
            mut_ptr: self.as_mut_ptr(),
            shape: &self.shape,
            device: &self.device,
        }
    }

    // Kernel-context stub methods (used inside #[kernel] bodies)
    pub fn load_tile<const N: usize, const M: usize>(
        &self,
        _tile_shape: [i64; N],
        _indices: [i64; M],
    ) -> crate::Tile<f32> {
        crate::Tile::new(_tile_shape.iter().copied().collect())
    }

    pub fn dim(&self, idx: usize) -> i64 {
        self.shape[idx]
    }

    pub fn load_tile_like_2d(&self, _target: &Tensor<f32>) -> crate::Tile<f32> {
        crate::Tile::new(_target.shape().to_vec())
    }

    pub fn store(&mut self, _value: crate::Tile<f32>) {
        // no-op in host context; handled by codegen
    }
}
