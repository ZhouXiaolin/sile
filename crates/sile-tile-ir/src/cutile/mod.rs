//! A cutile-shaped IR whose only job is to print what cutile-rs prints.
//!
//! This is deliberately not [`crate::ir`]. That IR carries what the C, WGPU and
//! PTX lowerings read — shape descriptors, buffer ranks, `stride_shape_idx` —
//! and none of those have a place in cutile's text. Keeping the two apart means
//! the alignment work cannot break a backend, and the backends cannot pull the
//! text off its measured format.
//!
//! `crates/sile-tile-ir/TILEIR_ALIGNMENT.md` records the format rules and the
//! fixtures in `crates/sile/tests/fixtures/tileir/` are the target text.

pub mod bounds;
pub mod emit;
pub mod print;

use std::fmt;

/// An SSA value, numbered by allocation order.
///
/// cutile numbers values as it allocates them, so the number a value prints
/// with *is* its emission position. Nothing may be reordered to tidy the
/// output: that would renumber everything after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// One scalar (element) type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I1,
    I32,
    I64,
    F16,
    F32,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::I1 => "i1",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::F32 => "f32",
        };
        write!(f, "{name}")
    }
}

/// What a tile's elements are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileElement {
    Scalar(ScalarType),
    Pointer(ScalarType),
}

impl fmt::Display for TileElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(s) => write!(f, "{s}"),
            Self::Pointer(p) => write!(f, "ptr<{p}>"),
        }
    }
}

/// A tensor in global memory: element type, shape, and per-axis strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub element: ScalarType,
    pub shape: Vec<emit::Extent>,
    pub strides: Vec<i64>,
}

impl fmt::Display for TensorView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self
            .shape
            .iter()
            .map(|e| match e {
                emit::Extent::Literal(v) => v.to_string(),
                emit::Extent::Value(_) => "?".into(),
            })
            .collect::<Vec<_>>()
            .join("x");
        let strides = self
            .strides
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "tensor_view<{shape}x{}, strides=[{strides}]>",
            self.element
        )
    }
}

/// A tile grid laid over a tensor view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionView {
    pub tile_shape: Vec<i64>,
    /// `Some(Zero)` for a read view; a write view carries no padding.
    pub padding_zero: bool,
    pub tensor: TensorView,
}

impl fmt::Display for PartitionView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tile = self
            .tile_shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let padding = if self.padding_zero {
            ", padding_value = zero"
        } else {
            ""
        };
        write!(f, "partition_view<tile=({tile}){padding}, {}>", self.tensor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    /// A statically shaped tile. `shape` is empty for a scalar tile.
    Tile {
        shape: Vec<i64>,
        element: TileElement,
    },
    Token,
    TensorView(TensorView),
    PartitionView(PartitionView),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tile { shape, element } => {
                let dims = shape
                    .iter()
                    .map(|d| format!("{d}x"))
                    .collect::<Vec<_>>()
                    .join("");
                write!(f, "tile<{dims}{element}>")
            }
            Self::Token => write!(f, "token"),
            Self::TensorView(v) => write!(f, "{v}"),
            Self::PartitionView(v) => write!(f, "{v}"),
        }
    }
}

/// How an operation spells itself.
///
/// cutile's printer is a match over opcodes with a hand-written arm each. The
/// variants here name the shapes those arms fall into, so an op states which
/// shape it has instead of carrying a closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `%t = opcode %a, %b : <operand types> -> <result type>`
    Plain,
    /// `%t, %u, %w = get_tile_block_id : tile<i32>`
    GetTileBlockId,
    /// `%t, %u = load_view_tko weak %v[i, j] token = %k : ...`
    LoadViewTko,
    /// `%t = store_view_tko weak %val, %v[i] token = %k : ...`
    StoreViewTko,
    /// `%t = constant <i32: 2> : tile<i32>`
    Constant,
    /// `%t = assume bounded<0, ?>, %a : tile<i32>`, with the bounds inline.
    Assume,
    /// `%t = make_token : token`
    MakeToken,
    /// `%t = make_tensor_view %p, shape = [...], strides = [...] : ... -> ...`
    MakeTensorView,
    /// `%t = make_partition_view %v : partition_view<...>`
    MakePartitionView,
    /// `%t = assert %c, "message" : tile<i1>`
    Assert,
}

/// A named attribute, already in the form it prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attr {
    pub name: &'static str,
    pub value: String,
}

/// One operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op {
    pub results: Vec<ValueId>,
    pub format: Format,
    /// The name that follows the result list.
    pub opcode: &'static str,
    pub operands: Vec<ValueId>,
    pub result_types: Vec<Type>,
    pub attrs: Vec<Attr>,
    /// The text the format needs beyond the value names: bounds for `assume`,
    /// literal and element for `constant`, shape and strides for the views.
    pub text: String,
    /// The `: <operand types> ->` part before the results, when the op has one.
    pub operand_types: Option<Vec<Type>>,
    /// Extra indentation for an operation inside a region.
    pub indent: usize,
}

/// A body item: an operation, a terminator, or a region-carrying op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Op(Op),
    /// A statement whose printed form is not one line, such as `reduce`, which
    /// carries a region: the body that combines two elements.
    Raw(String),
    Return,
}

/// An entry function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub di_name: String,
    pub args: Vec<(ValueId, Type)>,
    pub body: Vec<Stmt>,
}

/// The root of the printed program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
}
