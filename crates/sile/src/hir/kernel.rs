use super::{ShapeExpr, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinOp {
    ProgramId,
    LoadTile,
    LoadTileLike2D,
    Store,
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
    pub const_params: Vec<String>,
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
    Store {
        target: String,
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
    ScalarI32(i32),
    ScalarF32(f32),
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
        const_params: Vec<String>,
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
            Self::ScalarI32(_) => "scalar",
            Self::ScalarF32(_) => "scalar_f32",
        }
    }
}
