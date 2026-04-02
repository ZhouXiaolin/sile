#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendOp {
    VecAdd1D,
    Softmax2D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendKernel {
    pub op: BackendOp,
    pub tile_rank: usize,
    pub tile_shape_symbols: Vec<String>,
}
