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
pub enum TileIrType {
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

impl fmt::Display for TileIrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileIrType::I64 => write!(f, "i64"),
            TileIrType::F32 => write!(f, "f32"),
            TileIrType::Tile { rows, cols } => write!(f, "tile<f32, {rows}x{cols}>"),
            TileIrType::Buffer { rank } => write!(f, "buffer<f32, rank={rank}>"),
            TileIrType::Void => write!(f, "void"),
        }
    }
}

// ── Instructions ───────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct TileIrInst {
    pub result: ValueId,
    pub op: TileIrOp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TileIrOp {
    // ── Tile operations (shapes resolved to concrete i64) ──
    /// `cuda_tile.load_ptr_tko`
    LoadPtrTko {
        buf: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        rows: i64,
        cols: i64,
        /// Which shape dimension to use as stride
        stride_shape_idx: usize,
    },
    /// `cuda_tile.store_ptr_tko`
    StorePtrTko {
        buf: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        rows: i64,
        cols: i64,
        stride_shape_idx: usize,
    },
    /// `sile.atomic_add`
    SileAtomicAdd {
        buf: ValueId,
        value: ValueId,
        row_coord: ValueId,
        col_coord: ValueId,
        stride_shape_idx: usize,
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
    /// `cuda_tile.negf`
    NegF {
        operand: ValueId,
        rows: i64,
        cols: i64,
    },
    /// `cuda_tile.exp`
    Exp {
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
    /// `cuda_tile.broadcast`
    Broadcast {
        value: ValueId,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileIrParamKind {
    Buffer,
    SileProgramId { dim: i64 },
    SileShapeDim { source: ValueId, dim: usize },
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
            TileIrOp::LoadPtrTko {
                buf,
                row_coord,
                col_coord,
                ..
            } => {
                vec![*buf, *row_coord, *col_coord]
            }
            TileIrOp::StorePtrTko {
                buf,
                value,
                row_coord,
                col_coord,
                ..
            } => {
                vec![*buf, *value, *row_coord, *col_coord]
            }
            TileIrOp::SileAtomicAdd {
                buf,
                value,
                row_coord,
                col_coord,
                ..
            } => {
                vec![*buf, *value, *row_coord, *col_coord]
            }
            TileIrOp::Splat { .. } => vec![],
            TileIrOp::AddF { lhs, rhs, .. }
            | TileIrOp::SubF { lhs, rhs, .. }
            | TileIrOp::MulF { lhs, rhs, .. }
            | TileIrOp::DivF { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::NegF { operand, .. } | TileIrOp::Exp { operand, .. } => vec![*operand],
            TileIrOp::MmaF { a, b, acc, .. } => vec![*a, *b, *acc],
            TileIrOp::ReduceSum { value, .. } | TileIrOp::ReduceMax { value, .. } => vec![*value],
            TileIrOp::Broadcast { value, .. } => vec![*value],
            TileIrOp::Extract {
                tile,
                row_coord,
                col_coord,
            } => vec![*tile, *row_coord, *col_coord],
            TileIrOp::IBinary { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            TileIrOp::ConstI64(_) | TileIrOp::ConstF64(_) => vec![],
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
