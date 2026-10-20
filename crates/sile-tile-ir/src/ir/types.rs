//! Tile IR type system.
//!
//! Ported from `cutile-ir`'s `ir/types.rs`. The bytecode-specific helpers
//! (`minimum_version`, `type_tag`) are dropped — sile never serializes
//! bytecode.

/// Marker for a dimension or stride whose size is not statically known.
/// Matches MLIR's `ShapedType::kDynamic` (`i64::MIN`); printed as `?` in text.
pub const DYNAMIC: i64 = i64::MIN;

/// Scalar (element) types — integers and floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Byte width of this scalar type (I1 rounds up to 1 byte).
    pub fn byte_width(self) -> usize {
        match self {
            Self::I1
            | Self::I4
            | Self::I8
            | Self::F8E4M3FN
            | Self::F8E5M2
            | Self::F8E8M0FNU
            | Self::F4E2M1FN
            | Self::F8E5M3FNU => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 | Self::TF32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

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

/// An element type that can appear inside a Tile (scalar or pointer).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileElementType {
    Scalar(ScalarType),
    Pointer(Box<PointerType>),
}

/// Pointer to a scalar value in global device memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub pointee: ScalarType,
}

/// Public Tile IR 13.4 pointer classifications. Absence is represented by an
/// unqualified type; explicit `None` remains distinct from absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PointerAttribute {
    None = 0,
}

/// A statically-shaped tile of elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileType {
    pub shape: Vec<i64>,
    pub element_type: TileElementType,
}

/// A reference to a tensor in global memory with shape and strides.
/// Use [`DYNAMIC`] for dimensions/strides that are not statically known.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorViewType {
    pub element_type: ScalarType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

/// A view into a tensor where tiles are laid out in a grid pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// A view that uses a 1D tensor of sparse indices along one tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GatherScatterViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub sparse_dim: i32,
    pub padding_value: Option<PaddingValue>,
}

/// A view that traverses a tensor with configurable per-dimension strides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedViewType {
    pub tile_shape: Vec<i32>,
    pub traversal_strides: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// Padding value for out-of-bounds accesses in a partition view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PaddingValue {
    Zero = 0,
    NegZero = 1,
    Nan = 2,
    PosInf = 3,
    NegInf = 4,
}

/// Function signature type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
}

/// The unified type enum — every value in the IR has one of these types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarType),
    Pointer(PointerType),
    Tile(TileType),
    TensorView(TensorViewType),
    PartitionView(PartitionViewType),
    GatherScatterView(GatherScatterViewType),
    StridedView(StridedViewType),
    Func(FuncType),
    Token,
    /// Attach a classification to a pointer, pointer tile, or a view's base
    /// tensor. A wrapper preserves the existing public carrier structs.
    WithPointerAttribute(Box<Type>, PointerAttribute),
}

impl Type {
    pub fn with_pointer_attribute(self, attribute: PointerAttribute) -> Self {
        Self::WithPointerAttribute(
            Box::new(self.without_pointer_attribute().clone()),
            attribute,
        )
    }

    pub fn without_pointer_attribute(&self) -> &Type {
        match self {
            Self::WithPointerAttribute(base, _) => base.without_pointer_attribute(),
            _ => self,
        }
    }

    pub fn pointer_attribute(&self) -> Option<PointerAttribute> {
        match self {
            Self::WithPointerAttribute(_, attr) => Some(*attr),
            _ => None,
        }
    }
}

impl From<ScalarType> for Type {
    fn from(s: ScalarType) -> Self {
        Self::Scalar(s)
    }
}

impl From<PointerType> for Type {
    fn from(p: PointerType) -> Self {
        Self::Pointer(p)
    }
}

impl From<TileType> for Type {
    fn from(t: TileType) -> Self {
        Self::Tile(t)
    }
}

impl From<TensorViewType> for Type {
    fn from(tv: TensorViewType) -> Self {
        Self::TensorView(tv)
    }
}
