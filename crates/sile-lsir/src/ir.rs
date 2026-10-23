#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// Name of the shape descriptor parameter.
///
/// Every lowered kernel takes one `ptr<constant i64>` carrying the extents of
/// its buffer arguments. The name is part of the ABI: the lowering writes it
/// and all three backends look it up, so it lives here rather than in each of
/// them.
pub const SHAPES_PARAM_NAME: &str = "__sile_shapes";

/// Name of the scalar descriptor parameter.
///
/// The same channel for scalar arguments: one `ptr<constant f32>` holding each
/// scalar the launch passed, in parameter order. A scalar is a load from here
/// rather than a function parameter, which is what keeps one LSIR signature
/// across targets: WGSL cannot pass a value into an entry point.
pub const SCALARS_PARAM_NAME: &str = "__sile_scalars";

/// Name of the integer scalar descriptor parameter.
///
/// An integer scalar cannot travel in the float channel without a lie about
/// its width, so it gets its own: one `ptr<constant i64>`. A kernel that takes
/// no integer scalar has no such parameter, so every signature that existed
/// before this channel did stays the same.
pub const SCALARS_I64_PARAM_NAME: &str = "__sile_scalars_i64";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Generic,
    Global,
    Constant,
    Shared,
    Private,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Void,
    I1,
    I32,
    I64,
    F16,
    F32,
    F64,
    Ptr {
        addr_space: AddressSpace,
        pointee: Box<Type>,
    },
    Vector {
        len: usize,
        elem: Box<Type>,
    },
    Array {
        len: usize,
        elem: Box<Type>,
    },
}

impl Type {
    pub fn ptr(addr_space: AddressSpace, pointee: Type) -> Self {
        Self::Ptr {
            addr_space,
            pointee: Box::new(pointee),
        }
    }

    pub fn vector(len: usize, elem: Type) -> Self {
        Self::Vector {
            len,
            elem: Box::new(elem),
        }
    }

    pub fn array(len: usize, elem: Type) -> Self {
        Self::Array {
            len,
            elem: Box::new(elem),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Value(ValueId),
    Const(Constant),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Metadata {
    Parallel,
    Reduction,
    /// The addressed slot is the accessing thread's own by construction: the
    /// builder checked the extent divides across the threads before spelling
    /// `tid + k * threads`. The mapping's placement check honors the word
    /// instead of re-deriving the arithmetic.
    OwnerOrdered,
    /// The addressed slot reads a `TileReduce`'s relay: the reduce stage hands
    /// every thread the same copied values, so the slot is a broadcast, not a
    /// partition the strip may split. The lowering says so where it emits the
    /// access, instead of a later pass re-deriving it from the pointer chain.
    ReduceRelay,
    VectorizeWidth(u32),
    Unroll(u32),
    Alignment(u32),
    NoAlias,
    ReadOnly,
    WriteOnly,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamAbi {
    pub rank: usize,
    pub shape_offset: usize,
    /// Tile dimensions this parameter is stored through, when known.
    ///
    /// Populated by the Tile IR lowering from the `store_ptr_tko` shape, so
    /// backends read the tile plan instead of inferring it from the CFG.
    pub tile: Option<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub id: ValueId,
    pub name: String,
    pub ty: Type,
    pub abi: Option<ParamAbi>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockParam {
    pub id: ValueId,
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// The remainder of a division: `frem` on floats, `srem` on integers.
    Rem,
    And,
    Or,
    Xor,
    /// The bits of a count moved left: `shl` on integers.
    Shl,
    /// The bits of a count moved right: `ashr` on the signed counts this IR
    /// carries.
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpPred {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Olt,
    Ole,
    Ogt,
    Oge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastOp {
    Bitcast,
    Sext,
    Zext,
    Trunc,
    Sitofp,
    Fptosi,
    /// Widen a float to a wider one: half to float, at a half buffer's loads.
    Fext,
    /// Narrow a float to a smaller one: float to half, at a half buffer's
    /// stores.
    Ftrunc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncScope {
    Workgroup,
    Device,
}

/// How a tile-level fold combines two elements.
///
/// A target spells these differently — a C `#pragma omp simd reduction`, a
/// WGSL loop — so the intrinsic names the operation, not its spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileReduceOp {
    Sum,
    Product,
    Max,
    Min,
}

impl TileReduceOp {
    pub const fn name(self) -> &'static str {
        match self {
            TileReduceOp::Sum => "sum",
            TileReduceOp::Product => "product",
            TileReduceOp::Max => "max",
            TileReduceOp::Min => "min",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    ThreadId {
        dim: u8,
    },
    BlockId {
        dim: u8,
    },
    /// Launch grid extent along axis `dim`. The CuTile `num_programs(dim)`.
    GridDim {
        dim: u8,
    },
    Barrier {
        scope: SyncScope,
    },
    Exp,
    /// `llvm.exp2.f32`: two raised to `x`.
    Exp2,
    /// `llvm.log.f32`: the natural logarithm.
    Log,
    /// `llvm.log2.f32`: the base-two logarithm.
    Log2,
    /// Square root of one float. The LLVM spelling is `llvm.sqrt.f32`.
    Sqrt,
    /// Absolute value of one float.
    Abs,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    /// The first integer not below `x`, as a float.
    Ceil,
    /// The first integer not above `x`, as a float.
    Floor,
    /// `x` raised to `y`, element by element. Two arguments.
    Pow,
    /// The angle of the point `(x, y)`. Two arguments, `y` first.
    Atan2,
    /// `a * b + c` under one rounding. Three arguments, in the order the
    /// CuTile `fma` names them: `lhs`, `rhs`, `acc`.
    Fma,
    /// Load `<len x elem_ty>` from contiguous memory starting at `ptr + offset`.
    /// Args: [ptr, offset]. Result type must be Vector.
    VecLoad {
        len: usize,
    },
    /// Store `<len x elem_ty>` to contiguous memory starting at `ptr + offset`.
    /// Args: [ptr, offset, value]. value must be Vector.
    VecStore {
        len: usize,
    },
    /// Horizontal reduction: sum all elements of a vector into a scalar.
    /// Args: [vector]. Result type must be scalar.
    VecReduceAdd,
    /// Fold one axis of a tile. Args: [tile]. `reduce` is the combining
    /// opcode, `extent` the folded axis length, `inner` the number of
    /// elements after it, `outer` the number of output elements. The target
    /// picks the shape of the loop nest, or `#pragma omp simd reduction`.
    TileReduce {
        reduce: TileReduceOp,
        extent: u64,
        inner: u64,
        outer: u64,
    },
    /// `dst = lhs @ rhs + acc` over whole tiles. Args: [lhs, rhs, acc] with
    /// `m`, `n`, `k` the multiply's extents and `rhs_cols` the right operand's
    /// row stride. Kept whole so the target can choose a micro-kernel.
    TileMma {
        m: u64,
        n: u64,
        k: u64,
        rhs_cols: u64,
    },
    /// The hardware fold across one SIMD-group: every active lane receives
    /// the combined value of the lanes' arguments. Args: [value]. Result F32.
    /// Metal spells `simd_sum` and kin; no other target in the tree carries
    /// the primitive, so the contracts hold it to Metal only.
    SimdReduce {
        reduce: TileReduceOp,
    },
    /// The calling lane's index inside its SIMD-group. Metal's
    /// `thread_index_in_simdgroup`: the map from thread index to lane is the
    /// hardware's to choose, so the spelled builtin is the only sanctioned
    /// source.
    SimdLaneId,
    /// The calling SIMD-group's index inside the threadgroup. Metal's
    /// `simdgroup_index_in_threadgroup`.
    SimdGroupId,
    /// The SIMD-group tile multiply, `lhs @ rhs + acc` left whole for the
    /// Metal dialect to spell as `simdgroup_matrix` fragments. Args:
    /// [lhs, rhs, acc, out] — the panels and both `m * n` staging buffers in
    /// workgroup storage, `acc` absent when the multiply has no accumulator.
    /// `row_groups` x `col_groups` names the SIMD-groups the tile's 8x8
    /// fragments split across; a group with no block idles. Result-less: the
    /// fragment stores land the tile in `out`.
    SimdGroupMma {
        m: u64,
        n: u64,
        k: u64,
        rhs_cols: u64,
        row_groups: u32,
        col_groups: u32,
    },
}

impl Intrinsic {
    /// The name a one-argument float function has on every target, when this
    /// intrinsic is one.
    ///
    /// The C library adds an `f` to the name for the `float` form, Metal puts
    /// it behind `metal::`, and WGSL uses the name as it stands. Naming the
    /// function once keeps the three spellings in step.
    pub fn float_function(&self) -> Option<&'static str> {
        match self {
            Intrinsic::Exp => Some("exp"),
            Intrinsic::Exp2 => Some("exp2"),
            Intrinsic::Log => Some("log"),
            Intrinsic::Log2 => Some("log2"),
            Intrinsic::Sqrt => Some("sqrt"),
            Intrinsic::Abs => Some("fabs"),
            Intrinsic::Sin => Some("sin"),
            Intrinsic::Cos => Some("cos"),
            Intrinsic::Tan => Some("tan"),
            Intrinsic::Sinh => Some("sinh"),
            Intrinsic::Cosh => Some("cosh"),
            Intrinsic::Tanh => Some("tanh"),
            Intrinsic::Ceil => Some("ceil"),
            Intrinsic::Floor => Some("floor"),
            _ => None,
        }
    }

    /// The name a two-argument float function has on every target, when this
    /// intrinsic is one.
    ///
    /// The same three-way spelling rule as [`Intrinsic::float_function`]: the
    /// C library adds an `f`, Metal puts the name behind `metal::`, and WGSL
    /// uses it as it stands.
    pub fn binary_float_function(&self) -> Option<&'static str> {
        match self {
            Intrinsic::Pow => Some("pow"),
            Intrinsic::Atan2 => Some("atan2"),
            _ => None,
        }
    }

    /// The name a three-argument float function has on every target, when this
    /// intrinsic is one.
    ///
    /// The same three-way spelling rule as [`Intrinsic::float_function`].
    pub fn ternary_float_function(&self) -> Option<&'static str> {
        match self {
            Intrinsic::Fma => Some("fma"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstOp {
    Alloca {
        alloc_ty: Type,
        addr_space: AddressSpace,
    },
    Gep {
        base: Operand,
        indices: Vec<Operand>,
    },
    Load {
        ptr: Operand,
    },
    Store {
        ptr: Operand,
        value: Operand,
    },
    AtomicAdd {
        ptr: Operand,
        value: Operand,
    },
    Memcpy {
        dst: Operand,
        src: Operand,
        size: Operand,
    },
    Bin {
        op: BinOp,
        lhs: Operand,
        rhs: Operand,
    },
    Cmp {
        pred: CmpPred,
        lhs: Operand,
        rhs: Operand,
    },
    Cast {
        op: CastOp,
        value: Operand,
        to: Type,
    },
    Select {
        cond: Operand,
        on_true: Operand,
        on_false: Operand,
    },
    Call {
        func: String,
        args: Vec<Operand>,
    },
    Intrinsic {
        intrinsic: Intrinsic,
        args: Vec<Operand>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inst {
    pub result: Option<ValueId>,
    pub result_name: Option<String>,
    pub ty: Type,
    pub op: InstOp,
    pub metadata: Vec<Metadata>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Terminator {
    Br {
        target: BlockId,
        args: Vec<Operand>,
    },
    CondBr {
        cond: Operand,
        true_target: BlockId,
        true_args: Vec<Operand>,
        false_target: BlockId,
        false_args: Vec<Operand>,
    },
    Switch {
        value: Operand,
        default: BlockId,
        cases: Vec<(i64, BlockId)>,
    },
    Ret {
        value: Option<Operand>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub name: String,
    pub params: Vec<BlockParam>,
    pub insts: Vec<Inst>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub metadata: Vec<Metadata>,
}
