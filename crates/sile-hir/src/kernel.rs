use crate::types::{ShapeExpr, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinOp {
    /// `program_id(axis)`: this tile program's index along grid axis `axis`.
    ProgramId,
    /// `num_programs(axis)`: the launch grid extent along grid axis `axis`.
    NumPrograms,
    /// `num_tiles(buffer, axis)`: tile count along one buffer axis.
    NumTiles,
    LoadTile,
    LoadTileLike2D,
    Store,
    Max,
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    ReduceMax,
    ReduceSum,
    Reshape,
    Broadcast,
    ShapeOf,
    Mma,
    Constant,
    ScalarDiv,
    ShapeDim,
    Index,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub const_params: Vec<(String, i64)>,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        expr: Expr,
    },
    Assign {
        name: String,
        expr: Expr,
    },
    Store {
        target: String,
        value: Expr,
    },
    AtomicAdd {
        target: String,
        index: Expr,
        value: Expr,
    },
    ForLoop {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var(String),
    Shape(ShapeExpr),
    ScalarI64(i64),
    ScalarF32(f32),
    /// A runtime tile index, one scalar per axis: `[pid_m, k]` in CuTile Rust.
    ///
    /// Distinct from `Shape` because a shape is compile-time metadata while an
    /// index is a computed value, and collapsing the two drops any arithmetic
    /// the index carries.
    TileIndex(Vec<Expr>),
    Builtin { op: BuiltinOp, args: Vec<Expr> },
}

impl Param {
    pub fn new(name: impl Into<String>, kind: ParamKind, ty: Type) -> Self {
        Self {
            name: name.into(),
            kind,
            ty,
        }
    }
}

impl Kernel {
    pub fn new(
        name: impl Into<String>,
        const_params: Vec<(String, i64)>,
        params: Vec<Param>,
        body: Vec<Stmt>,
    ) -> Self {
        Self {
            name: name.into(),
            const_params,
            params,
            body,
        }
    }
}

impl Expr {
    pub fn builtin(op: BuiltinOp) -> Self {
        Self::Builtin { op, args: vec![] }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Builtin { .. } => "builtin",
            Self::Var(_) => "var",
            Self::Shape(_) => "shape",
            Self::ScalarI64(_) => "scalar",
            Self::ScalarF32(_) => "scalar_f32",
            Self::TileIndex(_) => "tile_index",
        }
    }
}
