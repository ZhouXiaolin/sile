use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeExpr {
    Dynamic,
    Constant(i64),
    Symbol(String),
    Tuple(Vec<ShapeExpr>),
}

/// The element type of a buffer, a tile, or a scalar argument.
///
/// A buffer and a tile hold floats, because that is what every target's memory
/// path carries. A scalar argument may also be an integer: a kernel divides a
/// launch index by one, and a division that rounds is a different program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    F32,
    I64,
}

impl ElemType {
    /// Whether a value of this type is a float.
    pub fn is_float(self) -> bool {
        matches!(self, ElemType::F32)
    }
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
    /// A tiled view of a buffer: `x.partition(shape)`.
    ///
    /// It holds no data. It names the buffer and the tile extent, so a later
    /// `part.load([i, j])` knows what to load without repeating the shape. Its
    /// category is `Shape` because it is metadata, not a value a store accepts.
    ///
    /// `source` is the buffer the view was taken from. A store through the view
    /// writes that buffer, so the checker follows the name back to the
    /// parameter and asks whether the parameter is an output.
    Partition {
        elem: ElemType,
        shape: ShapeExpr,
        source: Option<String>,
    },
    /// A shape expression, carrying the shape it names.
    ///
    /// The payload is what lets `let ts = shape![BM, BK];` keep its dimensions:
    /// the local's type is the shape, so a later `x.partition(ts)` reads the
    /// extent from the type instead of losing it.
    Shape(ShapeExpr),
    Scalar(ElemType),
}

impl ShapeExpr {
    pub fn dynamic() -> Self {
        Self::Dynamic
    }
    pub fn constant(value: i64) -> Self {
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
    /// Replace a symbolic shape with a fallback.
    ///
    /// A symbol here means "the argument did not name a shape", not a real
    /// dimension, so an operation that can read the shape elsewhere prefers
    /// that reading.
    pub fn if_symbolic(self, fallback: impl FnOnce() -> Self) -> Self {
        match self {
            Self::Symbol(_) => fallback(),
            other => other,
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
    pub fn partition(elem: ElemType, shape: ShapeExpr) -> Self {
        Self::Partition {
            elem,
            shape,
            source: None,
        }
    }
    /// A partition that names the buffer it views, so a store can follow it.
    pub fn partition_of(elem: ElemType, shape: ShapeExpr, source: impl Into<String>) -> Self {
        Self::Partition {
            elem,
            shape,
            source: Some(source.into()),
        }
    }
    pub fn category(&self) -> ValueCategory {
        match self {
            Self::Tensor { .. } => ValueCategory::Tensor,
            Self::Tile { .. } => ValueCategory::Tile,
            // A partition is a view, so it belongs with the metadata values.
            // Giving it the tile category would let `out.store(part)` pass.
            Self::Partition { .. } => ValueCategory::Shape,
            Self::Shape(_) => ValueCategory::Shape,
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
