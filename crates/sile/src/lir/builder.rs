use crate::lir::ir::*;

pub struct LirBuilder {
    pub func: Function,
    current_block: Option<String>,
    next_inst: usize,
}

impl LirBuilder {
    pub fn new(name: &str, params: Vec<Param>, return_type: Type) -> Self {
        let func = Function::new(name, params, return_type);
        Self {
            func,
            current_block: None,
            next_inst: 0,
        }
    }

    pub fn append_block(&mut self, label: &str) -> String {
        let label = label.to_string();
        self.func.blocks.push(BasicBlock {
            label: label.clone(),
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Ret(None),
        });
        if self.func.entry_block.is_empty() {
            self.func.entry_block = label.clone();
        }
        label
    }

    pub fn switch_to_block(&mut self, label: &str) {
        self.current_block = Some(label.to_string());
    }

    pub fn current_block_label(&self) -> &str {
        self.current_block.as_deref().unwrap_or("")
    }

    pub fn push_instruction(&mut self, inst: Instruction) -> Value {
        let val = Value::Inst(self.next_inst);
        self.next_inst += 1;
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.instructions.push(inst);
            }
        }
        val
    }

    pub fn alloca(&mut self, ty: Type) -> Value {
        self.push_instruction(Instruction::Alloca { ty })
    }

    pub fn load(&mut self, ptr: Value, ty: Type) -> Value {
        self.push_instruction(Instruction::Load {
            ptr,
            ty,
            align: None,
        })
    }

    pub fn store(&mut self, ptr: Value, value: Value) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.instructions.push(Instruction::Store {
                    ptr,
                    value,
                    align: None,
                });
            }
        }
    }

    pub fn gep(&mut self, ptr: Value, indices: Vec<Value>) -> Value {
        self.push_instruction(Instruction::Gep { ptr, indices })
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Add(lhs, rhs))
    }

    pub fn sub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Sub(lhs, rhs))
    }

    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Mul(lhs, rhs))
    }

    pub fn fcmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Fcmp(op, lhs, rhs))
    }

    pub fn icmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::Icmp(op, lhs, rhs))
    }

    pub fn fmax(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::FMax(lhs, rhs))
    }

    pub fn fmin(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push_instruction(Instruction::FMin(lhs, rhs))
    }

    pub fn exp(&mut self, val: Value) -> Value {
        self.push_instruction(Instruction::Exp(val))
    }

    pub fn br(&mut self, target: &str) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::Br {
                    target: target.to_string(),
                };
            }
        }
    }

    pub fn cond_br(&mut self, cond: Value, true_target: &str, false_target: &str) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::CondBr {
                    cond,
                    true_target: true_target.to_string(),
                    false_target: false_target.to_string(),
                };
            }
        }
    }

    pub fn ret(&mut self, value: Option<Value>) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.terminator = Terminator::Ret(value);
            }
        }
    }

    pub fn const_int(&self, v: i64) -> Value {
        Value::Const(Constant::Int(v))
    }

    pub fn const_float(&self, v: f64) -> Value {
        Value::Const(Constant::Float(v))
    }

    pub fn get_tile_coord(&mut self, dim: i64) -> Value {
        self.push_instruction(Instruction::GetTileCoord { dim })
    }

    pub fn tile_alloc(&mut self, rows: i64, cols: i64, init: f64) -> Value {
        self.push_instruction(Instruction::TileAlloc { rows, cols, init })
    }

    pub fn tile_load_2d(
        &mut self,
        buf: Value,
        rows: i64,
        cols: i64,
        row_tile: Value,
        col_tile: Value,
        stride_shape_idx: usize,
    ) -> Value {
        self.push_instruction(Instruction::TileLoad2D {
            buf,
            rows,
            cols,
            row_tile,
            col_tile,
            stride_shape_idx,
        })
    }

    pub fn tile_mma(
        &mut self,
        a: Value,
        b: Value,
        acc: Value,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
    ) -> Value {
        self.push_instruction(Instruction::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        })
    }

    pub fn tile_reduce_max(&mut self, value: Value, axis: i64, rows: i64, cols: i64) -> Value {
        self.push_instruction(Instruction::TileReduceMax {
            value,
            axis,
            rows,
            cols,
        })
    }

    pub fn tile_reduce_sum(&mut self, value: Value, axis: i64, rows: i64, cols: i64) -> Value {
        self.push_instruction(Instruction::TileReduceSum {
            value,
            axis,
            rows,
            cols,
        })
    }

    pub fn tile_broadcast(&mut self, value: Value, rows: i64, cols: i64) -> Value {
        self.push_instruction(Instruction::TileBroadcast { value, rows, cols })
    }

    pub fn tile_store_2d(
        &mut self,
        buf: Value,
        value: Value,
        rows: i64,
        cols: i64,
        row_tile: Value,
        col_tile: Value,
        stride_shape_idx: usize,
    ) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.instructions.push(Instruction::TileStore2D {
                    buf,
                    value,
                    rows,
                    cols,
                    row_tile,
                    col_tile,
                    stride_shape_idx,
                });
            }
        }
    }

    pub fn phi(&mut self, dest: &str, ty: Type, incoming: Vec<(Value, String)>) {
        if let Some(label) = &self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.phi_nodes.push(PhiNode {
                    dest: dest.to_string(),
                    ty,
                    incoming,
                });
            }
        }
    }

    pub fn finish(self) -> Function {
        self.func
    }
}
