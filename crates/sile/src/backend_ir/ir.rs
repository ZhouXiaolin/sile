#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendOp {
    VecAdd1D,
    Softmax2D,
    MatMul2D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendKernel {
    pub op: BackendOp,
    pub tile_rank: usize,
    pub tile_shape_symbols: Vec<String>,
    pub instructions: Vec<BackendInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendInstruction {
    Load {
        dest: String,
        src: String,
        indices: Vec<String>,
    },
    Compute {
        dest: String,
        op: String,
        args: Vec<String>,
    },
    Reduce {
        dest: String,
        src: String,
        axis: i64,
        kind: ReduceKind,
    },
    Store {
        src: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    Max,
    Sum,
}
