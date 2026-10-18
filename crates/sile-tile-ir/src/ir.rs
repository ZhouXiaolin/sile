use std::collections::HashMap;
use std::fmt;

pub use sile_hir::UnaryOp;

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
    /// Opaque shape descriptor view
    ShapeDesc { rank: usize },
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
            TileIrType::ShapeDesc { rank } => write!(f, "shape_desc<rank={rank}>"),
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
            | TileIrOp::DivF { lhs, rhs, .. } => vec![*lhs, *rhs],
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
