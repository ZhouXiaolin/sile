use std::collections::HashMap;
use std::fmt;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MirType {
    /// i64 index / scalar integer
    I64,
    /// f32 scalar
    F32,
    /// Tile of f32 with known shape
    Tile { rows: i64, cols: i64 },
    /// Opaque buffer pointer (kernel parameter)
    Buffer { rank: usize },
    /// Unit (for instructions with no meaningful result)
    Void,
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::I64 => write!(f, "i64"),
            MirType::F32 => write!(f, "f32"),
            MirType::Tile { rows, cols } => write!(f, "tile<f32, {rows}x{cols}>"),
            MirType::Buffer { rank } => write!(f, "buffer<f32, rank={rank}>"),
            MirType::Void => write!(f, "void"),
        }
    }
}

// ── Instructions ───────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct MirInst {
    pub result: ValueId,
    pub op: MirOp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MirOp {
    // ── Tile operations (shapes resolved to concrete i64) ──
    /// Load a tile from a buffer at given coordinates
    TileLoad {
        buf: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        rows: i64,
        cols: i64,
        /// Which shape dimension to use as stride
        stride_shape_idx: usize,
    },
    /// Store a tile into a buffer at given coordinates
    TileStore {
        buf: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        rows: i64,
        cols: i64,
        stride_shape_idx: usize,
    },
    /// Allocate a tile filled with a constant value
    TileConstant { value: f64, rows: i64, cols: i64 },
    /// Element-wise binary op on tiles
    TileBinary {
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        rows: i64,
        cols: i64,
    },
    /// Element-wise unary op on tiles
    TileUnary {
        op: UnaryOp,
        operand: ValueId,
        rows: i64,
        cols: i64,
    },
    /// Matrix multiply-accumulate: acc += a @ b
    TileMma {
        a: ValueId,
        b: ValueId,
        acc: ValueId,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
    },
    /// Reduce along an axis
    TileReduce {
        op: ReduceOp,
        value: ValueId,
        axis: i64,
        in_rows: i64,
        in_cols: i64,
    },
    /// Broadcast from reduced shape to full tile
    TileBroadcast {
        value: ValueId,
        rows: i64,
        cols: i64,
    },

    // ── Scalar / index operations ──
    /// Integer binary operation
    IBinary {
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
    /// Get tile coordinate (program_id dimension)
    ProgramId { dim: i64 },
    /// Get a dimension from the runtime shape array
    ShapeDim { buf: ValueId, dim: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Exp,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Max,
    Sum,
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
pub enum MirTerminator {
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
pub struct MirBlock {
    pub id: BlockId,
    /// Block parameters (replace phi nodes)
    pub params: Vec<ValueId>,
    pub insts: Vec<MirInst>,
    pub terminator: MirTerminator,
}

// ── Function / Program ─────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MirParam {
    pub value: ValueId,
    pub name: String,
    pub ty: MirType,
}

#[derive(Clone, Debug)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<MirParam>,
    pub blocks: Vec<MirBlock>,
    pub entry: BlockId,
    /// Type of every value in the function
    pub types: HashMap<ValueId, MirType>,
}

impl MirFunction {
    pub fn get_block(&self, id: BlockId) -> Option<&MirBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut MirBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Collect all ValueIds used (read) by a given instruction
    pub fn inst_uses(op: &MirOp) -> Vec<ValueId> {
        match op {
            MirOp::TileLoad {
                buf,
                row_coord,
                col_coord,
                ..
            } => {
                vec![*buf, *row_coord, *col_coord]
            }
            MirOp::TileStore {
                buf,
                value,
                row_coord,
                col_coord,
                ..
            } => {
                vec![*buf, *value, *row_coord, *col_coord]
            }
            MirOp::TileConstant { .. } => vec![],
            MirOp::TileBinary { lhs, rhs, .. } => vec![*lhs, *rhs],
            MirOp::TileUnary { operand, .. } => vec![*operand],
            MirOp::TileMma { a, b, acc, .. } => vec![*a, *b, *acc],
            MirOp::TileReduce { value, .. } => vec![*value],
            MirOp::TileBroadcast { value, .. } => vec![*value],
            MirOp::IBinary { lhs, rhs, .. } => vec![*lhs, *rhs],
            MirOp::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            MirOp::ConstI64(_) | MirOp::ConstF64(_) => vec![],
            MirOp::ProgramId { .. } => vec![],
            MirOp::ShapeDim { buf, .. } => vec![*buf],
        }
    }

    /// Collect all ValueIds used in a terminator
    pub fn terminator_uses(term: &MirTerminator) -> Vec<ValueId> {
        match term {
            MirTerminator::Jump { args, .. } => args.clone(),
            MirTerminator::Branch {
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
            MirTerminator::Return => vec![],
        }
    }
}
