#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaProgram {
    pub instructions: Vec<SsaInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaInstruction {
    pub opcode: SsaOpcode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaOpcode {
    ProgramId,
    LoadTile,
    LoadTileLike2D,
    Add,
    Sub,
    Div,
    Exp,
    ReduceMax,
    ReduceSum,
    Reshape,
    Broadcast,
    Store,
}

impl SsaInstruction {
    pub fn opcode_name(&self) -> &'static str {
        match self.opcode {
            SsaOpcode::ProgramId => "program_id",
            SsaOpcode::LoadTile => "load_tile",
            SsaOpcode::LoadTileLike2D => "load_tile_like_2d",
            SsaOpcode::Add => "add",
            SsaOpcode::Sub => "sub",
            SsaOpcode::Div => "div",
            SsaOpcode::Exp => "exp",
            SsaOpcode::ReduceMax => "reduce_max",
            SsaOpcode::ReduceSum => "reduce_sum",
            SsaOpcode::Reshape => "reshape",
            SsaOpcode::Broadcast => "broadcast",
            SsaOpcode::Store => "store",
        }
    }
}
