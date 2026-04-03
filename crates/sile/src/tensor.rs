use crate::{Device, Error, Result};

pub enum DList<const V: i32, R> {
    Cons(R),
}

pub struct DListNil;

pub trait Rank {}

impl Rank for DListNil {}

impl<const V: i32, R: Rank> Rank for DList<V, R> {}

#[derive(Clone, Debug)]
pub struct Partition<T> {
    pub parts: Vec<T>,
    pub tile_shape: Vec<i64>,
    pub grid_shape: Vec<i64>,
}

impl<T: Clone> Partition<T> {
    pub fn new(parts: Vec<T>, tile_shape: Vec<i64>, grid_shape: Vec<i64>) -> Self {
        Self {
            parts,
            tile_shape,
            grid_shape,
        }
    }
}

#[derive(Debug)]
pub struct Tensor<T, R: Rank = DListNil> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
    _rank: std::marker::PhantomData<R>,
}

impl<T: Clone, R: Rank> Clone for Tensor<T, R> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _rank: std::marker::PhantomData,
        }
    }
}

impl<T: Clone, R: Rank> Tensor<T, R> {
    pub fn partition(&self, tile_shape: impl Into<Vec<i64>>) -> Partition<Self> {
        let tile_shape = tile_shape.into();
        let grid_shape: Vec<i64> = self
            .shape
            .iter()
            .zip(tile_shape.iter())
            .map(|(&s, &t)| if t > 0 { s / t } else { 1 })
            .collect();
        let count: usize = grid_shape.iter().map(|&x| x as usize).product();
        Partition {
            parts: (0..count).map(|_| self.clone()).collect(),
            tile_shape,
            grid_shape,
        }
    }
}

impl Partition<Tensor<f32, DListNil>> {
    pub fn unpartition(self) -> Tensor<f32, DListNil> {
        let mut parts = self.parts.into_iter();
        let mut result = parts.next().expect("partition must have at least one part");
        for part in parts {
            result.data.extend_from_slice(&part.data);
            result.shape[0] += part.shape[0];
        }
        result
    }

    pub fn as_kernel_arg(&self) -> crate::kernel::KernelArg<'_> {
        self.parts[0].as_kernel_arg()
    }

    pub fn as_kernel_arg_mut(&mut self) -> crate::kernel::KernelArg<'_> {
        self.parts[0].as_kernel_arg_mut()
    }
}

pub fn unpartition(partition: Partition<Tensor<f32, DListNil>>) -> Tensor<f32, DListNil> {
    partition.unpartition()
}

impl Tensor<f32, DListNil> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 0.0, device)
    }

    pub fn ones(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 1.0, device)
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
            _rank: std::marker::PhantomData,
        })
    }

    fn filled(shape: Vec<i64>, value: f32, device: &Device) -> Result<Self> {
        let len = shape.iter().product::<i64>() as usize;
        Ok(Self {
            data: vec![value; len],
            shape,
            device: device.clone(),
            _rank: std::marker::PhantomData,
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
}
