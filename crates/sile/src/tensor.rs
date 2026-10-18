use crate::tile::Tile;
use sile_core::{Device, Error, KernelArg, Result};

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
        use rand::RngExt;
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
    pub fn to_vec(&self, _stream: &sile_core::Stream) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn as_kernel_arg(&self) -> KernelArg<'_> {
        KernelArg {
            ptr: self.as_ptr(),
            mut_ptr: self.as_ptr() as *mut f32,
            shape: &self.shape,
            device: &self.device,
        }
    }

    pub fn as_kernel_arg_mut(&mut self) -> KernelArg<'_> {
        KernelArg {
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
    ) -> Tile<f32> {
        Tile::new(_tile_shape)
    }

    pub fn dim(&self, idx: usize) -> i64 {
        self.shape[idx]
    }

    /// `x.partition(shape)`: a tiled view of this buffer.
    ///
    /// The view holds no data. The macro reads the tile extent from this call,
    /// so a later `part.load([i, j])` lowers to the same load `load_tile`
    /// produces without repeating the shape.
    pub fn partition<const N: usize>(&self, _shape: [i64; N]) -> Partition<f32> {
        Partition {
            shape: _shape.iter().copied().collect(),
            _elem: std::marker::PhantomData,
        }
    }

    /// `x.partition_mut(shape)`: the same view, opened for writing.
    pub fn partition_mut<const N: usize>(&mut self, _shape: [i64; N]) -> Partition<f32> {
        Partition {
            shape: _shape.iter().copied().collect(),
            _elem: std::marker::PhantomData,
        }
    }

    /// `x.load_like(y)`: a tile shaped like `y` at this program's coordinates.
    pub fn load_like(&self, y: &Tensor<f32>) -> Tile<f32> {
        Tile::new(y.shape())
    }

    /// `x.load()`: a tile shaped like this buffer.
    pub fn load(&self) -> Tile<f32> {
        Tile::new(self.shape.clone())
    }

    pub fn load_tile_like_2d(&self, _target: &Tensor<f32>) -> Tile<f32> {
        Tile::new(_target.shape().to_vec())
    }

    pub fn store(&mut self, _value: Tile<f32>) {
        // no-op in host context; handled by codegen
    }

    pub fn atomic_add(&mut self, _index: usize, _value: f32) {
        // no-op in host context; handled by codegen
    }
}

/// A tiled view of a buffer, as `x.partition(shape)` returns it.
///
/// It carries the tile extent and nothing else. The macro lowers it away: the
/// value a `load` receives is the buffer, and the extent comes from the type.
#[derive(Clone, Debug)]
pub struct Partition<T> {
    pub shape: Vec<i64>,
    pub _elem: std::marker::PhantomData<T>,
}

impl<T> Partition<T> {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// `part.load([i, j])`: the tile at the given partition coordinates.
    pub fn load<const N: usize>(&self, _index: [i64; N]) -> Tile<T> {
        Tile::new(self.shape.clone())
    }

    /// `part.store(tile, [i, j])`: write a tile at the given coordinates.
    pub fn store<const N: usize>(&mut self, _tile: Tile<T>, _index: [i64; N]) {
        // no-op in host context; handled by codegen
    }
}
