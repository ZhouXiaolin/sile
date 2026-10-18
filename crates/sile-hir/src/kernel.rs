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
    /// `x.partition(shape)`: a view of a buffer tiled by `shape`.
    ///
    /// It carries no data. It names the buffer and the tile extent, so a later
    /// `part.load([i, j])` knows what to load without a second shape argument.
    Partition,
    /// `out.partition_mut(shape)`: the same view, opened for writing.
    ///
    /// CuTile names mutability in the method, so a partition of an input is a
    /// read view and cannot be a store target. Keeping the two apart is what
    /// makes `out_part.store(t, [i, j])` a checked write and not a stray one.
    PartitionMut,
    /// `part.load([i, j])`: the tile at the given partition coordinates.
    Load,
    Store,
    Max,
    /// `min_tile(a, b)` and `min(a, b)`: the smaller of two values, element by
    /// element. Kept apart from `Max` because a minimum is not a maximum with
    /// the operands swapped.
    Min,
    /// `ceil_div(a, b)`: `a` over `b`, rounded up. CuTile names it as one
    /// operation, so it stays one here, and the lowering writes the integer
    /// steps it takes.
    CeilDiv,
    Add,
    Sub,
    Mul,
    Div,
    /// `%`: the remainder of a division. CuTile writes `pid.0 % h` for the
    /// head a program owns, so the operation is a scalar one here as well as an
    /// elementwise one.
    Rem,
    /// A one-operand float operation: `exp`, `log`, `sqrt`, and their
    /// relatives. The operation is data rather than a variant of its own,
    /// because every layer below dispatches on it in the same way: one
    /// transcendental then costs one name here and one instruction per target,
    /// and no traversal of the IR grows.
    Unary(UnaryOp),
    /// `gt_tile(a, b)` and its siblings: a tile of `0.0`/`1.0` from a
    /// comparison. `select` tests it, so the predicate stays a value.
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    /// `select(cond, a, b)`: `a` where `cond` is non-zero, else `b`.
    Select,
    /// `tile_to_scalar(t)`: the single element a one-element tile holds.
    TileToScalar,
    /// `scalar_to_tile(s)`: a one-element tile holding `s`.
    ScalarToTile,
    /// `convert_scalar(x)`: a scalar constant as a float value.
    ConvertScalar,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Reshape,
    Broadcast,
    /// `t.transpose()`: the same elements, with the two tile axes exchanged.
    Transpose,
    ShapeOf,
    Mma,
    Constant,
    ScalarDiv,
    ShapeDim,
    Index,
}

/// A one-operand float operation.
///
/// Each of these is one instruction on every target, so the IR names it
/// directly instead of writing it as a combination of the others: `exp2` is
/// not `exp(x * ln 2)` where the target has an `exp2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `exp(x)`: `e` raised to `x`.
    Exp,
    /// `exp2(x)`: two raised to `x`.
    Exp2,
    /// `log(x)`: the natural logarithm.
    Log,
    /// `log2(x)`: the base-two logarithm.
    Log2,
    /// `sqrt(x)`: the square root.
    Sqrt,
    /// `rsqrt(x)`: one over the square root.
    Rsqrt,
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
        /// The coordinates the store writes at.
        ///
        /// `y.store(t)` writes the tile this program owns, so the coordinates
        /// are the launch index and this is `None`. `part.store(t, [i, j])`
        /// names them, and then the launch index plays no part.
        index: Option<Expr>,
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
    /// `continue`: end this pass through the loop body.
    ///
    /// The statements after it in the same body do not run, and the loop's next
    /// iteration begins. The checker rejects a `continue` outside a loop, so
    /// this only ever appears inside a `ForLoop` body.
    Continue,
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
