#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

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
    And,
    Or,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncScope {
    Workgroup,
    Device,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    ThreadId { dim: u8 },
    BlockId { dim: u8 },
    Barrier { scope: SyncScope },
    MatmulFragment,
    ReduceAdd,
    ReduceMax,
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
