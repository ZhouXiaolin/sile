use crate::tile::Tile;
use sile_core::{Device, Error, KernelArg, Result};

/// An element a host tensor holds.
///
/// The buffer ABI sizes in bytes through [`HostElem::ELEM_SIZE`], and every
/// constructor speaks f32 — the values an example generates — so a half tensor
/// is a quantized copy of the same numbers, not a different program.
pub trait HostElem: Clone + std::fmt::Debug {
    /// Bytes of one element in the buffer ABI.
    const ELEM_SIZE: usize;

    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl HostElem for f32 {
    const ELEM_SIZE: usize = 4;

    fn from_f32(value: f32) -> Self {
        value
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl HostElem for half::f16 {
    const ELEM_SIZE: usize = 2;

    fn from_f32(value: f32) -> Self {
        half::f16::from_f32(value)
    }
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
}

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

impl<T: HostElem> Tensor<T> {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// `x.partition(shape)`: a tiled view of this buffer.
    ///
    /// The view holds no data. The macro reads the tile extent from this call,
    /// so a later `part.load([i, j])` lowers to the same load `load_tile`
    /// produces without repeating the shape.
    pub fn partition<const N: usize>(&self, shape: [i64; N]) -> Partition<'_, T> {
        Partition::new(self, shape)
    }

    /// `x.partition_mut(shape)`: the same view, opened for writing.
    pub fn partition_mut<const N: usize>(&mut self, shape: [i64; N]) -> PartitionMut<'_, T> {
        PartitionMut::new(self, shape)
    }

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
        let data: Vec<T> = (0..len).map(|_| T::from_f32(rng.random_range(0.0..1.0))).collect();
        Ok(Self {
            data,
            shape,
            device: device.clone(),
        })
    }

    pub fn from_vec(data: Vec<T>, shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
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
            data: vec![T::from_f32(value); len],
            shape,
            device: device.clone(),
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    pub fn to_vec(&self, _stream: &sile_core::Stream) -> Result<Vec<T>> {
        Ok(self.data.clone())
    }

    /// The values as f32, whatever the storage holds. A half tensor's values
    /// are exact here — widening is lossless — which is what makes one numpy
    /// reference carry both storages.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().map(|value| HostElem::to_f32(value.clone())).collect()
    }

    /// The values back, consuming the buffer: the move the async chain makes
    /// when the last stage copies to the host.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn as_kernel_arg(&self) -> KernelArg<'_> {
        KernelArg {
            ptr: self.as_ptr() as *const u8,
            mut_ptr: self.as_ptr() as *mut u8,
            elem_size: T::ELEM_SIZE,
            shape: &self.shape,
            device: &self.device,
        }
    }

    pub fn as_kernel_arg_mut(&mut self) -> KernelArg<'_> {
        KernelArg {
            ptr: self.as_ptr() as *const u8,
            mut_ptr: self.as_mut_ptr() as *mut u8,
            elem_size: T::ELEM_SIZE,
            shape: &self.shape,
            device: &self.device,
        }
    }

    /// `x.load()`: a tile shaped like this buffer.
    pub fn load(&self) -> Tile<T> {
        Tile::new(self.shape.clone())
    }
}

impl Tensor<f32> {
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

    /// `x.load_like(y)`: a tile shaped like `y` at this program's coordinates.
    pub fn load_like(&self, y: &Tensor<f32>) -> Tile<f32> {
        Tile::new(y.shape())
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
/// It names the buffer and the tile extent, and nothing else: the value a
/// `load` receives is the buffer, and the extent comes from the type. The tile
/// must divide the buffer along every axis, so a partition that cannot tile
/// the launch it names fails here rather than at the device.
#[derive(Clone, Debug)]
pub struct Partition<'a, T> {
    base: &'a Tensor<T>,
    tile: Vec<i64>,
}

/// The write half of a partition, as `x.partition_mut(shape)` returns it.
#[derive(Debug)]
pub struct PartitionMut<'a, T> {
    base: &'a mut Tensor<T>,
    tile: Vec<i64>,
}

/// Whether `tile` cuts `base` into whole tiles along every axis.
fn check_tiling(base_shape: &[i64], tile: &[i64]) {
    if tile.is_empty() {
        panic!("partition with an empty tile shape");
    }
    if tile.len() > base_shape.len() {
        panic!(
            "partition of rank {} over a buffer of rank {}",
            tile.len(),
            base_shape.len()
        );
    }
    for (axis, (&extent, &tile_extent)) in base_shape.iter().zip(tile).enumerate() {
        if tile_extent <= 0 || extent % tile_extent != 0 {
            panic!(
                "tile extent {tile_extent} does not divide buffer extent {extent} on axis {axis}"
            );
        }
    }
}

impl<'a, T> Partition<'a, T> {
    fn new<const N: usize>(base: &'a Tensor<T>, tile: [i64; N]) -> Self {
        check_tiling(&base.shape, &tile);
        Self {
            base,
            tile: tile.to_vec(),
        }
    }

    /// The tile extent the view was built with.
    pub fn tile(&self) -> &[i64] {
        &self.tile
    }

    /// The tile count along one axis: the buffer extent over the tile extent.
    pub fn tile_count(&self, axis: i64) -> i64 {
        self.base.shape[axis as usize] / self.tile[axis as usize]
    }

    /// `part.load([i, j])`: the tile at the given partition coordinates.
    ///
    /// A tile is always `f32`: a half buffer's load widens it at the buffer
    /// edge, the same conversion the generated kernel performs.
    pub fn load<const N: usize>(&self, _index: [i64; N]) -> Tile<f32> {
        Tile::new(self.tile.clone())
    }

    /// The buffer back, as cutile's `part.unpartition()` reads.
    pub fn unpartition(self) -> &'a Tensor<T> {
        self.base
    }
}

impl<'a, T> PartitionMut<'a, T> {
    fn new<const N: usize>(base: &'a mut Tensor<T>, tile: [i64; N]) -> Self {
        check_tiling(&base.shape, &tile);
        Self {
            base,
            tile: tile.to_vec(),
        }
    }

    /// The tile extent the view was built with.
    pub fn tile(&self) -> &[i64] {
        &self.tile
    }

    /// The tile count along one axis: the buffer extent over the tile extent.
    pub fn tile_count(&self, axis: i64) -> i64 {
        self.base.shape[axis as usize] / self.tile[axis as usize]
    }

    /// `part.store(tile, [i, j])`: write a tile at the given coordinates.
    pub fn store<const N: usize>(&mut self, _tile: Tile<T>, _index: [i64; N]) {
        // no-op in host context; handled by codegen
    }

    /// The buffer back, as cutile's `part.unpartition()` reads.
    pub fn unpartition(self) -> &'a mut Tensor<T> {
        self.base
    }
}

impl<T> std::ops::Deref for Partition<'_, T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Tensor<T> {
        self.base
    }
}

impl<T> std::ops::Deref for PartitionMut<'_, T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Tensor<T> {
        self.base
    }
}

impl<T> std::ops::DerefMut for PartitionMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Tensor<T> {
        self.base
    }
}
