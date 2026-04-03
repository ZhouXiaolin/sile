use crate::tensor::{DListNil, Rank};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct TileId(pub i64);

pub fn id() -> TileId {
    TileId(0)
}

#[derive(Clone, Debug)]
pub struct Tile<T, R: Rank = DListNil> {
    pub shape: Vec<i64>,
    pub _elem: std::marker::PhantomData<T>,
    pub _rank: std::marker::PhantomData<R>,
}

impl<T, R: Rank> Tile<T, R> {
    pub fn new(shape: Vec<i64>) -> Self {
        Self {
            shape,
            _elem: std::marker::PhantomData,
            _rank: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn reduce_max(&self, _axis: i32) -> Tile<T, DListNil> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reduce_sum(&self, _axis: i32) -> Tile<T, DListNil> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reshape<const N: usize>(&self, new_shape: [i32; N]) -> Tile<T, DListNil> {
        Tile::new(new_shape.iter().map(|&v| v as i64).collect())
    }

    pub fn broadcast(&self, _target_shape: &[i64]) -> Tile<T, DListNil> {
        Tile::new(self.shape.clone())
    }

    pub fn exp(&self) -> Tile<T, R> {
        Tile::new(self.shape.clone())
    }
}

impl Add for Tile<f32, DListNil> {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Sub for Tile<f32, DListNil> {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Mul for Tile<f32, DListNil> {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Div for Tile<f32, DListNil> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}
