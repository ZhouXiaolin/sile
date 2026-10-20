use std::collections::HashMap;
use std::fmt;

pub use sile_hir::UnaryOp;

pub use crate::opcode::Opcode;

// ── Value & Block identifiers ──────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ── Types ──────────────────────────────────────────────────────────

/// One scalar (element) type.
///
/// CuTile names sixteen of these; every one is here so an op signature can be
/// read off this enum instead of a string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I1,
    I4,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    TF32,
    F64,
    F8E4M3FN,
    F8E5M2,
    F8E8M0FNU,
    F4E2M1FN,
    F8E5M3FNU,
}

impl ScalarType {
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I4 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn is_float(self) -> bool {
        !self.is_integer()
    }
}

/// What a tile's elements are: a scalar, or a pointer to one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileElementType {
    Scalar(ScalarType),
    Pointer(Box<PointerType>),
}

/// A pointer to a scalar in device memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub pointee: ScalarType,
}

/// The pointer classification CuTile attaches to a pointer or view.
///
/// Only `None` is public today; the variant exists so an absent attribute and
/// an explicit `none` stay distinguishable, as they are in CuTile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerAttribute {
    None,
}

/// A statically shaped tile of elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileType {
    pub shape: Vec<i64>,
    pub element_type: TileElementType,
}

/// A tensor in global memory: element type, shape, and per-axis strides.
///
/// A stride of [`DYNAMIC_STRIDE`] means the value is not known statically.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorViewType {
    pub element_type: ScalarType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

/// Where a partition view sits: a tile grid laid over a tensor view.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// A partition view whose tiles gather through a sparse index along one axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherScatterViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub sparse_dim: i32,
    pub padding_value: Option<PaddingValue>,
}

/// A partition view that walks the tensor with per-axis traversal strides.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StridedViewType {
    pub tile_shape: Vec<i32>,
    pub traversal_strides: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// What a view reads where its tile reaches past the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PaddingValue {
    Zero,
    NegZero,
    Nan,
    PosInf,
    NegInf,
}

/// A function signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncType {
    pub inputs: Vec<TileIrType>,
    pub results: Vec<TileIrType>,
}

/// Marker for a dimension or stride whose size is not known statically.
/// It matches CuTile's `?`; a negative extent never means a real one.
pub const DYNAMIC_STRIDE: i64 = i64::MIN;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileIrType {
    /// i64 index / scalar integer
    I64,
    /// f32 scalar
    F32,
    /// Any one of the sixteen scalar types, named.
    Scalar(ScalarType),
    /// A pointer to a scalar.
    Pointer(PointerType),
    /// Opaque shape descriptor view
    ShapeDesc { rank: usize },
    /// Tile of f32 with known shape
    ///
    /// This is the f32 two-axis shorthand for [`TileIrType::TileOf`]: the
    /// kernels sile lowers today are all f32 tiles over two axes, and spelling
    /// them this way keeps every shape match in the pipeline reading two
    /// numbers instead of a vector.
    Tile { rows: i64, cols: i64 },
    /// A tile of any element type and any rank.
    TileOf(TileType),
    /// A tensor in global memory.
    TensorView(TensorViewType),
    /// A tile grid over a tensor.
    PartitionView(PartitionViewType),
    /// A tile grid that gathers through a sparse index.
    GatherScatterView(GatherScatterViewType),
    /// A tile grid that walks with per-axis traversal strides.
    StridedView(StridedViewType),
    /// A function signature.
    Func(FuncType),
    /// The completion of memory operations, passed as a value.
    ///
    /// A token is how this IR states ordering: a load produces one, a store
    /// consumes one, and `JoinTokens` merges two. Nothing here is ordered by
    /// where an instruction sits in a block.
    Token,
    /// A type carrying a pointer classification.
    WithPointerAttribute(Box<TileIrType>, PointerAttribute),
    /// Opaque buffer pointer (kernel parameter). The element type is the
    /// buffer's storage: a half buffer converts at its loads and stores, and
    /// every tile between them stays f32.
    Buffer { rank: usize, elem: BufferElem },
    /// Unit (for instructions with no meaningful result)
    Void,
}

/// The element type a buffer's storage holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferElem {
    F32,
    F16,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ScalarType::I1 => "i1",
            ScalarType::I4 => "i4",
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F16 => "f16",
            ScalarType::BF16 => "bf16",
            ScalarType::F32 => "f32",
            ScalarType::TF32 => "tf32",
            ScalarType::F64 => "f64",
            ScalarType::F8E4M3FN => "f8e4m3fn",
            ScalarType::F8E5M2 => "f8e5m2",
            ScalarType::F8E8M0FNU => "f8e8m0fnu",
            ScalarType::F4E2M1FN => "f4e2m1fn",
            ScalarType::F8E5M3FNU => "f8e5m3fnu",
        };
        write!(f, "{name}")
    }
}

impl fmt::Display for TileElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileElementType::Scalar(s) => write!(f, "{s}"),
            TileElementType::Pointer(p) => write!(f, "ptr<{}>", p.pointee),
        }
    }
}

/// Print one extent, or `?` where it is dynamic.
fn fmt_extent(d: i64) -> String {
    if d == DYNAMIC_STRIDE || d < 0 {
        "?".into()
    } else {
        d.to_string()
    }
}

fn fmt_extents(shape: &[i64]) -> String {
    shape.iter().map(|d| fmt_extent(*d)).collect::<Vec<_>>().join("x")
}

impl fmt::Display for TensorViewType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let strides = self
            .strides
            .iter()
            .map(|d| fmt_extent(*d))
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "tensor_view<{}x{}, strides=[{strides}]>",
            fmt_extents(&self.shape),
            self.element_type
        )
    }
}

impl PartitionViewType {
    /// The `, padding_value = ...` suffix, or nothing when the view does not pad.
    fn padding_suffix(&self) -> String {
        match self.padding_value {
            Some(PaddingValue::Zero) => ", padding_value = zero".into(),
            Some(PaddingValue::NegZero) => ", padding_value = neg_zero".into(),
            Some(PaddingValue::Nan) => ", padding_value = nan".into(),
            Some(PaddingValue::PosInf) => ", padding_value = pos_inf".into(),
            Some(PaddingValue::NegInf) => ", padding_value = neg_inf".into(),
            None => String::new(),
        }
    }

    /// The `, dim_map = [...]` suffix, or nothing for the identity mapping.
    fn dim_map_suffix(&self) -> String {
        let identity = self.dim_map.iter().enumerate().all(|(i, d)| *d == i as i32);
        if identity {
            String::new()
        } else {
            let vals = self
                .dim_map
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(", dim_map=[{vals}]")
        }
    }

    fn tile_shape_str(&self) -> String {
        self.tile_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }
}

impl fmt::Display for PartitionViewType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "partition_view<tile=({}){}, {}{}>",
            self.tile_shape_str(),
            self.padding_suffix(),
            self.tensor_view,
            self.dim_map_suffix()
        )
    }
}

impl fmt::Display for GatherScatterViewType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tile_shape = self
            .tile_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let padding = match self.padding_value {
            Some(PaddingValue::Zero) => ", padding_value = zero",
            Some(PaddingValue::NegZero) => ", padding_value = neg_zero",
            Some(PaddingValue::Nan) => ", padding_value = nan",
            Some(PaddingValue::PosInf) => ", padding_value = pos_inf",
            Some(PaddingValue::NegInf) => ", padding_value = neg_inf",
            None => "",
        };
        write!(
            f,
            "gather_scatter_view<tile=({tile_shape}), sparse_dim={}{padding}, {}>",
            self.sparse_dim, self.tensor_view
        )
    }
}

impl fmt::Display for StridedViewType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tile_shape = self
            .tile_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let traversal = self
            .traversal_strides
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "strided_view<tile=({tile_shape}), traversal_strides=[{traversal}], {}>",
            self.tensor_view
        )
    }
}

impl fmt::Display for TileIrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileIrType::I64 => write!(f, "i64"),
            TileIrType::F32 => write!(f, "f32"),
            TileIrType::Scalar(s) => write!(f, "{s}"),
            TileIrType::Pointer(p) => write!(f, "ptr<{}>", p.pointee),
            TileIrType::ShapeDesc { rank } => write!(f, "shape_desc<rank={rank}>"),
            TileIrType::Tile { rows, cols } => write!(f, "tile<{rows}x{cols}xf32>"),
            TileIrType::TileOf(t) => {
                if t.shape.is_empty() {
                    write!(f, "tile<{}>", t.element_type)
                } else {
                    write!(f, "tile<{}x{}>", fmt_extents(&t.shape), t.element_type)
                }
            }
            TileIrType::TensorView(tv) => write!(f, "{tv}"),
            TileIrType::PartitionView(pv) => write!(f, "{pv}"),
            TileIrType::GatherScatterView(gv) => write!(f, "{gv}"),
            TileIrType::StridedView(sv) => write!(f, "{sv}"),
            TileIrType::Func(ft) => {
                let inputs = ft
                    .inputs
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let results = ft
                    .results
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "({inputs}) -> ({results})")
            }
            TileIrType::Token => write!(f, "token"),
            TileIrType::WithPointerAttribute(base, _) => {
                // The attribute is spelled the way CuTile spells it: inside the
                // closing angle bracket it qualifies.
                let text = base.to_string();
                match text.rfind('>') {
                    Some(end) => write!(
                        f,
                        "{}, #cuda_tile.ptr_attr<none>>",
                        &text[..end]
                    ),
                    None => write!(f, "{text}, #cuda_tile.ptr_attr<none>"),
                }
            }
            TileIrType::Buffer { rank, elem } => {
                let elem = match elem {
                    BufferElem::F32 => "f32",
                    BufferElem::F16 => "f16",
                };
                write!(f, "buffer<{elem}, rank={rank}>")
            }
            TileIrType::Void => write!(f, "void"),
        }
    }
}

impl TileIrType {
    /// The `(rows, cols)` of an f32 two-axis tile, if this is one.
    ///
    /// Every tile the current lowering produces is f32 over two axes, so the
    /// shape-only readers stay two numbers wide.
    pub fn tile_rows_cols(&self) -> Option<(i64, i64)> {
        match self {
            TileIrType::Tile { rows, cols } => Some((*rows, *cols)),
            TileIrType::TileOf(t) if t.shape.len() == 2 => Some((t.shape[0], t.shape[1])),
            _ => None,
        }
    }
}

// ── Memory semantics ───────────────────────────────────────────────

/// How wide a memory operation's effect is visible.
///
/// A token says an operation finished. A scope says who can see it. The two
/// are separate: a store can be complete and still not be visible to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    /// Visible to the threads of one tile block.
    TileBlock,
    /// Visible to the whole device.
    Device,
    /// Visible to the host.
    System,
}

/// How strongly a memory operation orders against others.
///
/// This is the atomics vocabulary, and it applies to the same three questions:
/// what may be reordered across it, and in which direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryOrdering {
    Weak,
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}

/// The combining function of an atomic read-modify-write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicRmwMode {
    And,
    Or,
    Xor,
    Add,
    AddF,
    Max,
    Min,
    UMax,
    UMin,
    Xchg,
}

/// The predicate of an integer or float comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonPredicate {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// How a rounding operation breaks a tie or a halfway case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    /// Round to nearest, ties to even.
    NearestEven,
    /// Round toward zero.
    Zero,
    /// Round toward positive infinity.
    PositiveInfinity,
    /// Round toward negative infinity.
    NegativeInfinity,
}

/// Whether an integer operation may wrap, and how it reports that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OverflowMode {
    None,
    /// Wrap on overflow.
    Wrap,
    /// Saturate at the type's bounds.
    Saturate,
    /// Produce a flag alongside the result.
    Flag,
}

/// One attribute a general op carries, as CuTile stores attributes: a name and
/// a value.
#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// A list of integers, such as `dim_map = [0, 1]`.
    IntList(Vec<i64>),
}

impl Attribute {
    /// The integer this attribute holds, if it holds one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Whether this attribute is the boolean `true`.
    pub fn is_true(&self) -> bool {
        matches!(self, Attribute::Bool(true))
    }
}

// ── Instructions ───────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct TileIrInst {
    pub result: ValueId,
    pub op: TileIrOp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TileMapExpr {
    Value(ValueId),
    LoadPtrTko {
        buf: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        /// One coordinate for each axis above the tile, outermost first.
        ///
        /// A tile covers the last two axes of the buffer, so every axis above
        /// them is a plane of planes: this coordinate picks which one. A
        /// rank-1 or rank-2 buffer has no such axis, so the list is empty and
        /// the addressing is what it was.
        outer_coords: Vec<ValueId>,
        rows: i64,
        cols: i64,
        stride_shape_idx: usize,
    },
    Splat {
        value: f64,
    },
    Cmp {
        op: CmpOp,
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    Add {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    Sub {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    Mul {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    Div {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    /// A one-operand float operation. The operation is data, so a traversal of
    /// this tree never has to know which transcendentals exist.
    Unary {
        op: UnaryOp,
        operand: Box<TileMapExpr>,
    },
    /// `pow(a, b)`: `a` raised to `b`, element by element. The two-argument
    /// float functions ride this tree the way the one-argument ones ride
    /// [`TileMapExpr::Unary`], because both are pointwise.
    Pow {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    /// `atan2(y, x)`: the angle of the point `(x, y)`.
    Atan2 {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    /// `fma(a, b, c)`: `a * b + c` with one rounding. The fused form rides
    /// this tree like the other pointwise functions.
    Fma {
        a: Box<TileMapExpr>,
        b: Box<TileMapExpr>,
        c: Box<TileMapExpr>,
    },
    /// The remainder of a division.
    Rem {
        lhs: Box<TileMapExpr>,
        rhs: Box<TileMapExpr>,
    },
    Select {
        cond: Box<TileMapExpr>,
        on_true: Box<TileMapExpr>,
        on_false: Box<TileMapExpr>,
    },
    Broadcast {
        value: Box<TileMapExpr>,
        src_rows: i64,
        src_cols: i64,
    },
    /// `iota(shape![N])`: the rank-one sequence `0, 1, …, N-1`.
    ///
    /// The element at each coordinate is the coordinate itself, so this map
    /// reads the walk it is standing on instead of an operand. CuTile names it
    /// for a rank-one shape only, so one element per column is the whole
    /// sequence and the expression carries no axis.
    Iota,
    /// `t.transpose()`: the same elements, read with the two coordinates
    /// exchanged.
    ///
    /// A map walks the tile domain and reads each operand at the coordinates it
    /// is standing on, so a transpose is that walk with the pair exchanged. The
    /// destination domain is the operand's, with its two extents swapped.
    Transpose {
        value: Box<TileMapExpr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TileIrOp {
    // ── Tile operations (shapes resolved to concrete i64) ──
    /// `cuda_tile.load_ptr_tko`
    LoadPtrTko {
        buf: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        /// One coordinate for each axis above the tile, outermost first.
        ///
        /// A tile covers the last two axes of the buffer, so every axis above
        /// them is a plane of planes: this coordinate picks which one. A
        /// rank-1 or rank-2 buffer has no such axis, so the list is empty and
        /// the addressing is what it was.
        outer_coords: Vec<ValueId>,
        rows: i64,
        cols: i64,
        /// Which shape dimension to use as stride
        stride_shape_idx: usize,
        /// The token this load waits for, held as the semantics struct so an
        /// op states its ordering, scope and token in one place.
        ///
        /// A load that waits for nothing says so; it does not mean unordered.
        semantics: crate::token::TkoSemantics,
    },
    /// `cuda_tile.store_ptr_tko`
    StorePtrTko {
        buf: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        /// One coordinate for each axis above the tile, outermost first.
        ///
        /// A tile covers the last two axes of the buffer, so every axis above
        /// them is a plane of planes: this coordinate picks which one. A
        /// rank-1 or rank-2 buffer has no such axis, so the list is empty and
        /// the addressing is what it was.
        outer_coords: Vec<ValueId>,
        rows: i64,
        cols: i64,
        stride_shape_idx: usize,
        /// The token this store waits for. This is the only place two ordering
        /// chains meet: the store names the join, and nothing else orders it.
        semantics: crate::token::TkoSemantics,
    },
    /// `cuda_tile.make_partition_view`: a tile grid laid over a buffer.
    ///
    /// This is CuTile's memory model. A kernel does not address a buffer; it
    /// names a view of one, and every access goes through the view. The view
    /// carries the tile it will load, so an access states no extent of its own.
    MakePartitionView {
        /// The buffer the view describes.
        source: ValueId,
        /// The tile a load or a store through this view reads or writes.
        tile_rows: i64,
        tile_cols: i64,
        /// The token the view is made under: the ordering of the memory the
        /// view describes.
        semantics: crate::token::TkoSemantics,
    },
    /// `cuda_tile.load_view_tko`: read one tile through a view.
    ///
    /// The tile is the view's, so the op names coordinates and no extent.
    LoadViewTko {
        view: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        /// One coordinate for each axis above the tile, outermost first.
        outer_coords: Vec<ValueId>,
        rows: i64,
        cols: i64,
        /// Which shape dimension to use as stride.
        stride_shape_idx: usize,
        semantics: crate::token::TkoSemantics,
    },
    /// `cuda_tile.store_view_tko`: write one tile through a view.
    ///
    /// The store names the token it waits for. That name is the only ordering
    /// edge it has: nothing about where the instruction sits orders it.
    StoreViewTko {
        view: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        outer_coords: Vec<ValueId>,
        rows: i64,
        cols: i64,
        stride_shape_idx: usize,
        semantics: crate::token::TkoSemantics,
    },
    /// `sile.atomic_add`
    SileAtomicAdd {
        buf: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        /// One coordinate for each axis above the tile, outermost first.
        ///
        /// A tile covers the last two axes of the buffer, so every axis above
        /// them is a plane of planes: this coordinate picks which one. A
        /// rank-1 or rank-2 buffer has no such axis, so the list is empty and
        /// the addressing is what it was.
        outer_coords: Vec<ValueId>,
        stride_shape_idx: usize,
        /// The token this atomic waits for.
        semantics: crate::token::TkoSemantics,
    },
    /// Splat a scalar constant to a full tile
    Splat { value: f64, rows: i64, cols: i64 },
    /// `cuda_tile.addf`
    AddF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.subf`
    SubF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.mulf`
    MulF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.divf`
    DivF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.fma`: `lhs * rhs + acc` with one rounding.
    ///
    /// CuTile spells the fused form as one op (`$lhs, $rhs, $acc`), and so do
    /// the targets that have the instruction: an FMA pipelines as one
    /// operation where a multiply and an add pipeline as two.
    Fma {
        lhs: ValueId,
        rhs: ValueId,
        acc: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.select`: `$cond, $val_if_true, $val_if_false`.
    ///
    /// The condition is a value like any other: a mask of ones and zeroes, or
    /// a count. An element is picked where it tests true.
    Select {
        cond: ValueId,
        on_true: ValueId,
        on_false: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.maxf`: the larger of two tiles, element by element.
    MaxF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.minf`: the smaller of two tiles, element by element.
    MinF {
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// A one-operand float operation over a tile.
    ///
    /// CuTile spells each of these as its own op (`cuda_tile.exp`), and so does
    /// every target. They share one variant here because the shape and the
    /// operand are what a traversal needs; which function it is, is data.
    UnaryF {
        op: UnaryOp,
        operand: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.mmaf`
    MmaF {
        a: ValueId,
        b: ValueId,
        acc: ValueId,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
    },
    /// `cuda_tile.reduce` with sum combiner
    ReduceSum {
        value: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
    },
    /// `cuda_tile.reduce` with max combiner
    ReduceMax {
        value: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
    },
    /// `cuda_tile.reduce` with min combiner
    ReduceMin {
        value: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
    },
    /// `cuda_tile.reduce` with prod combiner
    ReduceProd {
        value: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
    },
    /// `cuda_tile.scan_sum` and a `scan` with an add or a multiply combiner:
    /// the running fold along one axis. Unlike a reduce, every element of the
    /// output keeps the accumulated value up to and including itself, so the
    /// output is the operand's own extent.
    Scan {
        value: ValueId,
        /// The value the fold starts from: `out[0]` is `combine(identity, in[0])`.
        identity: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
        reverse: bool,
        combiner: ReduceCombiner,
    },
    /// `cuda_tile.cat`: the two tiles joined along one axis.
    Cat {
        lhs: ValueId,
        rhs: ValueId,
        axis: i64,
        lhs_rows: i64,
        lhs_cols: i64,
        rhs_rows: i64,
        rhs_cols: i64,
    },
    /// `cuda_tile.extract`: the subtile of the source at a block index. The
    /// block index is a value per axis; the subtile extent is static.
    ExtractSlice {
        source: ValueId,
        row_idx: ValueId,
        col_idx: ValueId,
        in_rows: i64,
        in_cols: i64,
        out_rows: i64,
        out_cols: i64,
    },
    /// `cuda_tile.insert`: the destination with the subtile written at a
    /// block index.
    InsertSlice {
        source: ValueId,
        destination: ValueId,
        row_idx: ValueId,
        col_idx: ValueId,
        in_rows: i64,
        in_cols: i64,
        out_rows: i64,
        out_cols: i64,
    },
    /// `cuda_tile.broadcast`
    Broadcast {
        value: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.reshape`
    ///
    /// The same elements in a new extent. A reshape is a view in CuTile, and
    /// row-major flattening makes the view a linear identity, but this IR
    /// materializes tiles in private memory whose every access is spelled by
    /// the extent the value was defined with. Re-typing the operand in place
    /// would leave the definition and the uses reading one storage with two
    /// strides, so the reshape is a value of its own: a walk over the new
    /// extent that reads the operand through the linear position.
    Reshape {
        value: ValueId,
        rows: i64,
        cols: i64,
    },
    /// Canonical fused pointwise map over a tile domain.
    Map {
        expr: TileMapExpr,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.extract`
    Extract {
        tile: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
    },

    // ── Scalar / index operations ──
    /// Integer binary operation
    IBinary {
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// Float binary operation on scalars.
    ///
    /// The two are separate ops, not one op with a type attribute: an integer
    /// division and a float division are different instructions all the way
    /// down, and the reader of the IR should not have to look up a type table
    /// to tell which one is meant.
    FBinary {
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// Integer comparison
    ICmp {
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// Integer constant
    ConstI64(i64),
    /// Float constant
    ConstF64(f64),
    /// Query one dimension from a shape descriptor view
    ShapeDim { shape: ValueId, dim: usize },
    /// Launch grid extent along `dim`. `num_programs(dim)`.
    GridDim { dim: i64 },
    /// Tile count along `dim` of the shape behind `shape`.
    /// `num_tiles(buffer, dim, tile_extent)`.
    NumTiles {
        shape: ValueId,
        dim: usize,
        tile_extent: i64,
    },

    // ── General form ──
    /// Any CUDA Tile op, carried by its opcode.
    ///
    /// CuTile has one representation for all 105 operations: an opcode, a list
    /// of operands, a list of result types, and a list of attributes. That is
    /// the right shape for an IR whose job is to state *what* runs, and it
    /// keeps a target from needing a variant per function it will lower to the
    /// same instruction.
    ///
    /// The variants above stay because they carry what a traversal reads most:
    /// tile extents, view axes, combiner choice. This one carries everything
    /// else, so no op is unrepresentable. An op lowers to its own variant once
    /// a pass needs to read its fields structurally.
    Opaque {
        opcode: Opcode,
        /// Extra operands beyond `result`, in CuTile's operand order.
        operands: Vec<ValueId>,
        /// The types this op defines, when it defines more than `result`
        /// (`load_ptr_tko` yields a tile and a token; `get_tile_block_id`
        /// yields three scalars).
        results: Vec<TileIrType>,
        attributes: Vec<(String, Attribute)>,
    },
}

/// Which combiner a fold applies to the elements it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceCombiner {
    Sum,
    Max,
    Min,
    Prod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// The remainder of a division.
    Rem,
    /// The bitwise and of two counts. A conjunction of masks is a product, so
    /// it uses `Mul` instead: the operands say which of the two an operator
    /// means, and they are the ones that know whether a value is a mask.
    And,
    /// The bitwise or of two counts. A disjunction of masks is a maximum, so it
    /// uses `Select` instead.
    Or,
    /// The exclusive or of two counts. There is no mask reading: one of two
    /// masks being set is their difference, not a value an element type holds.
    Xor,
    /// The bits of a count moved left.
    Shl,
    /// The bits of a count moved right.
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "add"),
            BinOp::Sub => write!(f, "sub"),
            BinOp::Mul => write!(f, "mul"),
            BinOp::Div => write!(f, "div"),
            BinOp::Rem => write!(f, "rem"),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
            BinOp::Xor => write!(f, "xor"),
            BinOp::Shl => write!(f, "shl"),
            BinOp::Shr => write!(f, "shr"),
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Lt => write!(f, "lt"),
            CmpOp::Le => write!(f, "le"),
            CmpOp::Gt => write!(f, "gt"),
            CmpOp::Ge => write!(f, "ge"),
            CmpOp::Eq => write!(f, "eq"),
            CmpOp::Ne => write!(f, "ne"),
        }
    }
}

// ── Terminators ────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum TileIrTerminator {
    /// Unconditional jump: goto target(args...)
    Jump { target: BlockId, args: Vec<ValueId> },
    /// Conditional branch
    Branch {
        cond: ValueId,
        true_target: BlockId,
        true_args: Vec<ValueId>,
        false_target: BlockId,
        false_args: Vec<ValueId>,
    },
    /// Return from kernel
    Return,
}

// ── Blocks ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TileIrBlock {
    pub id: BlockId,
    /// Block parameters (replace phi nodes)
    pub params: Vec<ValueId>,
    pub insts: Vec<TileIrInst>,
    pub terminator: TileIrTerminator,
}

// ── Function / Program ─────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TileIrParam {
    pub value: ValueId,
    pub name: String,
    pub ty: TileIrType,
    pub kind: TileIrParamKind,
    /// Shape ABI of a buffer parameter, recorded when HIR lowers to Tile IR.
    ///
    /// HIR knows the parameter order and the shape extent of each buffer, and
    /// the LLVM IR lowering needs both to place `__sile_shapes` offsets. The
    /// field carries that knowledge across the layer boundary, so the LLVM IR
    /// layer reads Tile IR alone instead of depending on HIR.
    pub abi: Option<TileIrParamAbi>,
}

/// The shape ABI a buffer parameter carries into LLVM IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileIrParamAbi {
    /// Number of shape entries the buffer declares.
    pub rank: usize,
    /// Index of its first entry in the `__sile_shapes` descriptor.
    pub shape_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileIrParamKind {
    Buffer,
    LaunchIndex {
        dim: i64,
    },
    ShapeDesc {
        source: ValueId,
    },
    /// A scalar kernel argument, read from slot `index` of the scalar
    /// descriptor.
    ///
    /// The descriptor is a side channel exactly as the shape descriptor is: the
    /// kernel receives one pointer and loads the value. Keeping it a load of a
    /// descriptor rather than a function parameter is what lets all three
    /// targets share one LLVM IR signature, because WGSL has no way to pass a
    /// value into an entry point.
    Scalar {
        index: usize,
    },
}

#[derive(Clone, Debug)]
pub struct TileIrFunction {
    pub name: String,
    pub params: Vec<TileIrParam>,
    pub blocks: Vec<TileIrBlock>,
    pub entry: BlockId,
    /// Type of every value in the function
    pub types: HashMap<ValueId, TileIrType>,
}

impl TileIrFunction {
    pub fn get_block(&self, id: BlockId) -> Option<&TileIrBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut TileIrBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Collect all ValueIds used (read) by a given instruction
    pub fn inst_uses(op: &TileIrOp) -> Vec<ValueId> {
        match op {
            // The general form states its operands; there is nothing to derive.
            TileIrOp::Opaque { operands, .. } => operands.clone(),
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                outer_coords,
                ..
            } => {
                let mut uses = vec![*buf, *row_coord, *col_coord];
                uses.extend(outer_coords.iter().copied());
                uses
            }
            TileIrOp::StorePtrTko {
                buf,
                value,
                row_coord,
                col_coord,
                outer_coords,
                ..
            } => {
                let mut uses = vec![*buf, *value, *row_coord, *col_coord];
                uses.extend(outer_coords.iter().copied());
                uses
            }
            TileIrOp::MakePartitionView { source, semantics, .. } => {
                let mut uses = vec![*source];
                uses.extend(semantics.token);
                uses
            }
            TileIrOp::LoadViewTko {
                view,
                row_coord,
                col_coord,
                outer_coords,
                semantics,
                ..
            } => {
                let mut uses = vec![*view, *row_coord, *col_coord];
                uses.extend(outer_coords.iter().copied());
                uses.extend(semantics.token);
                uses
            }
            TileIrOp::StoreViewTko {
                view,
                value,
                row_coord,
                col_coord,
                outer_coords,
                semantics,
                ..
            } => {
                let mut uses = vec![*view, *value, *row_coord, *col_coord];
                uses.extend(outer_coords.iter().copied());
                uses.extend(semantics.token);
                uses
            }
            TileIrOp::SileAtomicAdd {
                buf,
                value,
                row_coord,
                col_coord,
                outer_coords,
                ..
            } => {
                let mut uses = vec![*buf, *value, *row_coord, *col_coord];
                uses.extend(outer_coords.iter().copied());
                uses
            }
            TileIrOp::Splat { .. } => vec![],
            TileIrOp::AddF { lhs, rhs, .. }
            | TileIrOp::SubF { lhs, rhs, .. }
            | TileIrOp::MulF { lhs, rhs, .. }
            | TileIrOp::DivF { lhs, rhs, .. }
            | TileIrOp::MaxF { lhs, rhs, .. }
            | TileIrOp::MinF { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::Fma { lhs, rhs, acc, .. } => vec![*lhs, *rhs, *acc],
            TileIrOp::Select {
                cond,
                on_true,
                on_false,
                ..
            } => vec![*cond, *on_true, *on_false],
            TileIrOp::UnaryF { operand, .. } => vec![*operand],
            TileIrOp::MmaF { a, b, acc, .. } => vec![*a, *b, *acc],
            TileIrOp::ReduceSum { value, .. }
            | TileIrOp::ReduceMax { value, .. }
            | TileIrOp::ReduceMin { value, .. }
            | TileIrOp::ReduceProd { value, .. } => vec![*value],
            TileIrOp::Scan {
                value, identity, ..
            } => vec![*value, *identity],
            TileIrOp::Cat { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::ExtractSlice {
                source,
                row_idx,
                col_idx,
                ..
            } => vec![*source, *row_idx, *col_idx],
            TileIrOp::InsertSlice {
                source,
                destination,
                row_idx,
                col_idx,
                ..
            } => vec![*source, *destination, *row_idx, *col_idx],
            TileIrOp::Broadcast { value, .. } => vec![*value],
            TileIrOp::Reshape { value, .. } => vec![*value],
            TileIrOp::Map { expr, .. } => tile_map_expr_uses(expr),
            TileIrOp::Extract {
                tile,
                row_coord,
                col_coord,
            } => vec![*tile, *row_coord, *col_coord],
            TileIrOp::IBinary { lhs, rhs, .. } | TileIrOp::FBinary { lhs, rhs, .. } => {
                vec![*lhs, *rhs]
            }
            TileIrOp::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::ConstI64(_) | TileIrOp::ConstF64(_) => vec![],
            TileIrOp::ShapeDim { shape, .. } => vec![*shape],
            TileIrOp::GridDim { .. } => vec![],
            TileIrOp::NumTiles { shape, .. } => vec![*shape],
        }
    }

    /// Collect all ValueIds used in a terminator
    pub fn terminator_uses(term: &TileIrTerminator) -> Vec<ValueId> {
        match term {
            TileIrTerminator::Jump { args, .. } => args.clone(),
            TileIrTerminator::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                let mut uses = vec![*cond];
                uses.extend(true_args);
                uses.extend(false_args);
                uses
            }
            TileIrTerminator::Return => vec![],
        }
    }
}

fn tile_map_expr_uses(expr: &TileMapExpr) -> Vec<ValueId> {
    let mut uses = Vec::new();
    collect_tile_map_expr_uses(expr, &mut uses);
    uses
}

fn collect_tile_map_expr_uses(expr: &TileMapExpr, uses: &mut Vec<ValueId>) {
    match expr {
        TileMapExpr::Value(value) => uses.push(*value),
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            ..
        } => {
            uses.push(*buf);
            uses.push(*row_coord);
            uses.push(*col_coord);
        }
        // A sequence and a splat read no operand: the value is the coordinate
        // the walk stands on, and the constant the map carries.
        TileMapExpr::Splat { .. } | TileMapExpr::Iota => {}
        TileMapExpr::Cmp { lhs, rhs, .. }
        | TileMapExpr::Add { lhs, rhs }
        | TileMapExpr::Sub { lhs, rhs }
        | TileMapExpr::Mul { lhs, rhs }
        | TileMapExpr::Div { lhs, rhs }
        | TileMapExpr::Rem { lhs, rhs }
        | TileMapExpr::Pow { lhs, rhs }
        | TileMapExpr::Atan2 { lhs, rhs } => {
            collect_tile_map_expr_uses(lhs, uses);
            collect_tile_map_expr_uses(rhs, uses);
        }
        TileMapExpr::Fma { a, b, c } => {
            collect_tile_map_expr_uses(a, uses);
            collect_tile_map_expr_uses(b, uses);
            collect_tile_map_expr_uses(c, uses);
        }
        TileMapExpr::Unary { operand, .. } | TileMapExpr::Transpose { value: operand } => {
            collect_tile_map_expr_uses(operand, uses);
        }
        TileMapExpr::Select {
            cond,
            on_true,
            on_false,
        } => {
            collect_tile_map_expr_uses(cond, uses);
            collect_tile_map_expr_uses(on_true, uses);
            collect_tile_map_expr_uses(on_false, uses);
        }
        TileMapExpr::Broadcast { value, .. } => collect_tile_map_expr_uses(value, uses),
    }
}
