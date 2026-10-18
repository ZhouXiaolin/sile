use std::ops::{Add, BitAnd, Div, Index, Mul, Sub};

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
/// Matches CuTile Rust `program_id(axis) -> i32`: one scalar, not a triple.
/// The historical [`id`] returns all three axes at once and is kept for the
/// kernels written before the CuTile spelling was adopted.
pub fn program_id(_axis: i64) -> i64 {
    0
}

/// The launch grid extent along grid axis `axis`.
///
/// Matches CuTile Rust `num_programs(axis)`.
pub fn num_programs(_axis: i64) -> i64 {
    0
}

/// A buffer, or a view of one.
///
/// CuTile's `num_tiles` accepts a tensor and a partition alike, because a
/// partition names the buffer it views. The host stub needs one name that covers
/// both, and this is that name.
pub trait TileView {
    /// The tile count along one axis. A whole buffer is one tile of itself; a
    /// partition divides the buffer by the tile it was built with.
    fn tile_count(&self, axis: i64) -> i64;
}

impl<T> TileView for crate::tensor::Tensor<T> {
    fn tile_count(&self, _axis: i64) -> i64 {
        1
    }
}

impl<T> TileView for crate::tensor::Partition<'_, T> {
    fn tile_count(&self, axis: i64) -> i64 {
        crate::tensor::Partition::tile_count(self, axis)
    }
}

impl<T> TileView for crate::tensor::PartitionMut<'_, T> {
    fn tile_count(&self, axis: i64) -> i64 {
        crate::tensor::PartitionMut::tile_count(self, axis)
    }
}

/// Tile count along one axis of a buffer or a view of one.
///
/// Matches CuTile Rust `num_tiles(partition, axis)`. The tile extent comes from
/// the partition the buffer is used with, so the count is a property of the
/// kernel and not of the launch.
pub fn num_tiles<V: TileView>(view: &V, axis: i64) -> i64 {
    view.tile_count(axis)
}

/// The current tile program's block id, as `(x, y, z)`.
///
/// CuTile reads the launch index as one tuple and projects the axis it needs.
/// The kernel parser turns `pid.0` into `program_id(0)`, so the projection
/// costs nothing and the HIR keeps one representation of a launch index.
pub fn get_tile_block_id() -> TileId {
    TileId(0, 0, 0)
}

/// The launch grid, as `(x, y, z)`.
///
/// Matches CuTile Rust `get_num_tile_blocks()`. Like the block id, the kernel
/// parser reads one axis through the field access on the call.
pub fn get_num_tile_blocks() -> (i64, i64, i64) {
    (0, 0, 0)
}

/// A compile-time shape.
///
/// CuTile writes `shape![BM, BK]` because the rank is part of the type there.
/// Here the argument list carries the rank, so the macro is syntax: it builds
/// the array an equivalent literal would, and the kernel parser reads that as a
/// shape.
#[macro_export]
macro_rules! shape {
    ($($dim:expr),* $(,)?) => {
        [$($dim as i64),*]
    };
}

/// CuTile's second spelling of a compile-time shape. Both carry the same
/// dimension list, so both expand to the same array.
#[macro_export]
macro_rules! const_shape {
    ($($dim:expr),* $(,)?) => {
        [$($dim as i64),*]
    };
}

/// `load_tile_mut(y)`: the tile `y` covers at this program's coordinates.
///
/// CuTile names the mutability in the function, not in the tile, so this is the
/// same load `load_like` performs.
pub fn load_tile_mut(tensor: &crate::tensor::Tensor<f32>) -> Tile<f32> {
    Tile::new(tensor.shape())
}

/// CuTile lets a scalar splat across a tile: `a.broadcast(y.shape())`.
///
/// The kernel parser turns the call into the `Broadcast` builtin, so the host
/// side only has to make the method exist. The trait is separate from [`Tile`]
/// because the receiver is a plain `f32`.
pub trait ScalarBroadcast {
    fn broadcast(&self, shape: impl AsRef<[i64]>) -> Tile<f32>;
}

impl ScalarBroadcast for f32 {
    fn broadcast(&self, shape: impl AsRef<[i64]>) -> Tile<f32> {
        Tile::new(shape)
    }
}

/// A value a kernel can turn into an element.
///
/// CuTile's `constant` is generic over the element type, so `constant(true,
/// shape)` and `constant(1, shape)` are both legal there. This IR has one
/// element type, so the host stub needs one way to read any of them as that
/// element. A boolean is a mask value: `1.0` where it holds, `0.0` where it does
/// not, which is the encoding `gt_tile` produces.
pub trait IntoElement {
    fn into_element(self) -> f32;
}

impl IntoElement for f32 {
    fn into_element(self) -> f32 {
        self
    }
}

impl IntoElement for i32 {
    fn into_element(self) -> f32 {
        self as f32
    }
}

impl IntoElement for i64 {
    fn into_element(self) -> f32 {
        self as f32
    }
}

impl IntoElement for bool {
    fn into_element(self) -> f32 {
        if self { 1.0 } else { 0.0 }
    }
}

/// The larger or the smaller of two values.
///
/// CuTile names the scalar spelling (`max`) and the tile spelling (`max_tile`)
/// apart, but one operation serves both: the operands decide which it is. The
/// host body keeps one name for each direction, so it needs one trait that
/// accepts a scalar and a tile alike.
pub trait Extremum: Sized {
    fn max_of(self, other: Self) -> Self;
    fn min_of(self, other: Self) -> Self;
}

impl Extremum for f32 {
    fn max_of(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    fn min_of(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
}

impl Extremum for i32 {
    fn max_of(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    fn min_of(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
}

impl Extremum for i64 {
    fn max_of(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    fn min_of(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
}

/// A tile is a value: CuTile copies it freely (`rms + tx * tx` uses `tx`
/// twice), so the shape lives inline and the type stays `Copy`.
const MAX_RANK: usize = 4;

#[derive(Debug, PartialEq, Eq)]
pub struct Tile<T> {
    dims: [i64; MAX_RANK],
    rank: u8,
    _elem: std::marker::PhantomData<T>,
}

// Hand-written so the bounds stay off `T`: the payload is `PhantomData`, so a
// tile copies whatever its element type is.
impl<T> Clone for Tile<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Tile<T> {}

impl<T> Tile<T> {
    pub fn new(shape: impl AsRef<[i64]>) -> Self {
        let shape = shape.as_ref();
        let mut dims = [0i64; MAX_RANK];
        dims[..shape.len()].copy_from_slice(shape);
        Self {
            dims,
            rank: shape.len() as u8,
            _elem: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.dims[..self.rank as usize]
    }

    /// The shape a reduction along `axis` leaves.
    ///
    /// The host stub carries shapes so that a kernel can read `.shape()`, and a
    /// reduction removes the axis it reduces. An axis the tile does not have
    /// removes nothing, which keeps a rank-zero tile from reading an empty
    /// shape.
    pub fn reduced_shape(&self, axis: i64) -> Tile<T> {
        let count = self.rank as i64;
        let index = if axis < 0 { axis + count } else { axis };
        let index = usize::try_from(index).unwrap_or(usize::MAX);
        let mut dims = [0i64; MAX_RANK];
        let mut kept = 0usize;
        for (i, dim) in self.shape().iter().enumerate() {
            if i != index {
                dims[kept] = *dim;
                kept += 1;
            }
        }
        Tile::new(&dims[..kept])
    }

    pub fn reduce_max(&self, axis: i64) -> Tile<T> {
        self.reduced_shape(axis)
    }

    pub fn reduce_sum(&self, axis: i64) -> Tile<T> {
        self.reduced_shape(axis)
    }

    pub fn reshape<const N: usize>(&self, new_shape: [i64; N]) -> Tile<T> {
        Tile::new(new_shape)
    }

    /// Splat this tile across a larger one.
    ///
    /// The target is either a buffer's shape (`y.shape()`, a slice) or a
    /// compile-time shape (`shape![BM, BK]`, an array), so the parameter takes
    /// anything that reads as a slice.
    pub fn broadcast(&self, _target_shape: impl AsRef<[i64]>) -> Tile<T> {
        *self
    }

    pub fn exp(&self) -> Tile<T> {
        *self
    }

    /// `t.transpose()`: the same elements with the two tile axes exchanged.
    pub fn transpose(&self) -> Tile<T> {
        let mut dims = self.dims;
        let count = self.rank as usize;
        if count >= 2 {
            dims.swap(count - 1, count - 2);
        }
        Tile::new(dims[..count].to_vec())
    }
}

impl Add for Tile<f32> {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        self
    }
}

impl Sub for Tile<f32> {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        self
    }
}

impl Mul for Tile<f32> {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        self
    }
}

impl Div for Tile<f32> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        self
    }
}

/// `a & b` on two masks. A mask is `0.0` or `1.0`, so a conjunction is the
/// product, and the host body carries the left operand's shape.
impl BitAnd for Tile<f32> {
    type Output = Self;
    fn bitand(self, _rhs: Self) -> Self {
        self
    }
}

impl Index<usize> for Tile<f32> {
    type Output = f32;

    fn index(&self, _index: usize) -> &Self::Output {
        static ZERO: f32 = 0.0;
        &ZERO
    }
}

impl Extremum for Tile<f32> {
    fn max_of(self, _other: Self) -> Self {
        self
    }
    fn min_of(self, _other: Self) -> Self {
        self
    }
}
