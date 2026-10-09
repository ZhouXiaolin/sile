use std::ops::{Add, Div, Index, Mul, Sub};

/// A tile program's index, or a grid extent, as (x, y, z).
///
/// Tuple indexing reads one axis. Scalar arithmetic applies to every axis,
/// which is what CuTile Rust gets from `program_id(axis) -> i32`: an index can
/// be computed, not only read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TileId(pub i64, pub i64, pub i64);

impl std::ops::Add<i64> for TileId {
    type Output = Self;
    fn add(self, rhs: i64) -> Self {
        Self(self.0 + rhs, self.1 + rhs, self.2 + rhs)
    }
}

impl std::ops::Sub<i64> for TileId {
    type Output = Self;
    fn sub(self, rhs: i64) -> Self {
        Self(self.0 - rhs, self.1 - rhs, self.2 - rhs)
    }
}

impl std::ops::Mul<i64> for TileId {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

/// This tile program's index along grid axis `axis`.
///
/// Matches CuTile Rust `program_id(axis)`. Read the axis with `.0` / `.1` /
/// `.2`, exactly like the `TileId` spelling.
pub fn program_id(_axis: i64) -> TileId {
    TileId(0, 0, 0)
}

/// The launch grid extent along grid axis `axis`.
///
/// Matches CuTile Rust `num_programs(axis)`.
pub fn num_programs(_axis: i64) -> i64 {
    0
}

/// Tile count along one axis of a buffer.
///
/// Matches CuTile Rust `num_tiles(partition, axis)`.
pub fn num_tiles<T>(_tensor: &crate::tensor::Tensor<T>, _axis: i64) -> i64 {
    0
}

/// Deprecated spelling of [`program_id`].
pub fn id() -> TileId {
    TileId(0, 0, 0)
}

#[derive(Clone, Debug)]
pub struct Tile<T> {
    pub shape: Vec<i64>,
    pub _elem: std::marker::PhantomData<T>,
}

impl<T> Tile<T> {
    pub fn new(shape: Vec<i64>) -> Self {
        Self {
            shape,
            _elem: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn reduce_max(&self, _axis: i64) -> Tile<T> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reduce_sum(&self, _axis: i64) -> Tile<T> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reshape<const N: usize>(&self, new_shape: [i64; N]) -> Tile<T> {
        Tile::new(new_shape.iter().copied().collect())
    }

    pub fn broadcast(&self, _target_shape: &[i64]) -> Tile<T> {
        Tile::new(self.shape.clone())
    }

    pub fn exp(&self) -> Tile<T> {
        Tile::new(self.shape.clone())
    }
}

impl Add for Tile<f32> {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Sub for Tile<f32> {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Mul for Tile<f32> {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Div for Tile<f32> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Index<usize> for Tile<f32> {
    type Output = f32;

    fn index(&self, _index: usize) -> &Self::Output {
        static ZERO: f32 = 0.0;
        &ZERO
    }
}
