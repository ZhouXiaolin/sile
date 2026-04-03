#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaProgram {
    pub instructions: Vec<SsaInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaInstruction {
    pub def: SsaValue,
    pub opcode: SsaOpcode,
    pub uses: Vec<SsaValue>,
    pub immediates: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsaValue {
    Param(usize),
    Local(usize),
    Const(i64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaOpcode {
    ProgramId,
    LoadTile,
    LoadTileLike2D,
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    ReduceMax,
    ReduceSum,
    Reshape,
    Broadcast,
    Store,
    ShapeOf,
    Mma,
    Constant,
    ScalarDiv,
    ShapeDim,
}

impl SsaInstruction {
    pub fn opcode_name(&self) -> &'static str {
        match self.opcode {
            SsaOpcode::ProgramId => "program_id",
            SsaOpcode::LoadTile => "load_tile",
            SsaOpcode::LoadTileLike2D => "load_tile_like_2d",
            SsaOpcode::Add => "add",
            SsaOpcode::Sub => "sub",
            SsaOpcode::Mul => "mul",
            SsaOpcode::Div => "div",
            SsaOpcode::Exp => "exp",
            SsaOpcode::ReduceMax => "reduce_max",
            SsaOpcode::ReduceSum => "reduce_sum",
            SsaOpcode::Reshape => "reshape",
            SsaOpcode::Broadcast => "broadcast",
            SsaOpcode::Store => "store",
            SsaOpcode::ShapeOf => "shape_of",
            SsaOpcode::Mma => "mma",
            SsaOpcode::Constant => "constant",
            SsaOpcode::ScalarDiv => "scalar_div",
            SsaOpcode::ShapeDim => "shape_dim",
        }
    }
}
