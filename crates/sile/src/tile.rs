use std::ops::{Add, Div, Index, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct TileId(pub i64, pub i64, pub i64);

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
