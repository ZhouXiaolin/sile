use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeExpr {
    Dynamic,
    Constant(i32),
    Symbol(String),
    Tuple(Vec<ShapeExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueCategory {
    Tensor,
    Tile,
    Shape,
    Scalar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Tensor { elem: ElemType, shape: ShapeExpr },
    Tile { elem: ElemType, shape: ShapeExpr },
    Shape,
    Scalar(ElemType),
}

impl ShapeExpr {
    pub fn dynamic() -> Self {
        Self::Dynamic
    }
    pub fn constant(value: i32) -> Self {
        Self::Constant(value)
    }
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }
    pub fn tuple(dims: impl IntoIterator<Item = ShapeExpr>) -> Self {
        Self::Tuple(dims.into_iter().collect())
    }
    pub fn rank(&self) -> usize {
        match self {
            Self::Tuple(dims) => dims.len(),
            _ => 1,
        }
    }
    pub fn contains_dynamic(&self) -> bool {
        match self {
            Self::Dynamic => true,
            Self::Constant(_) | Self::Symbol(_) => false,
            Self::Tuple(dims) => dims.iter().any(Self::contains_dynamic),
        }
    }
}

impl Type {
    pub fn tensor(elem: ElemType, shape: ShapeExpr) -> Self {
        Self::Tensor { elem, shape }
    }
    pub fn tile(elem: ElemType, shape: ShapeExpr) -> Self {
        Self::Tile { elem, shape }
    }
    pub fn category(&self) -> ValueCategory {
        match self {
            Self::Tensor { .. } => ValueCategory::Tensor,
            Self::Tile { .. } => ValueCategory::Tile,
            Self::Shape => ValueCategory::Shape,
            Self::Scalar(_) => ValueCategory::Scalar,
        }
    }
}

impl fmt::Display for ShapeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dynamic => write!(f, "-1"),
            Self::Constant(value) => write!(f, "{value}"),
            Self::Symbol(name) => write!(f, "{name}"),
            Self::Tuple(dims) => {
                write!(f, "[")?;
                for (idx, dim) in dims.iter().enumerate() {
                    if idx > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
                write!(f, "]")
            }
        }
    }
}
