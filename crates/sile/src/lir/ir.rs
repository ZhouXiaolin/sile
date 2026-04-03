use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Param(usize),
    Const(Constant),
    Inst(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerType {
    I8,
    I16,
    I32,
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatType {
    F16,
    F32,
    F64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Void,
    Int(IntegerType),
    Float(FloatType),
    Pointer(Box<Type>),
    Vector(Box<Type>, usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalVariable {
    pub name: String,
    pub ty: Type,
    pub initializer: Option<Constant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
    Olt,
    Ole,
    Ogt,
    Oge,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Alloca {
        ty: Type,
    },
    Load {
        ptr: Value,
        ty: Type,
        align: Option<u32>,
    },
    Store {
        ptr: Value,
        value: Value,
        align: Option<u32>,
    },
    Gep {
        ptr: Value,
        indices: Vec<Value>,
    },
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    FNeg(Value),
    FMax(Value, Value),
    FMin(Value, Value),
    Exp(Value),
    Icmp(CmpOp, Value, Value),
    Fcmp(CmpOp, Value, Value),
    Trunc(Value, Type),
    ZExt(Value, Type),
    SIToFP(Value, Type),
    FPToSI(Value, Type),
    BitCast(Value, Type),
    Call {
        func: String,
        args: Vec<Value>,
        ret_ty: Type,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Terminator {
    Br {
        target: String,
    },
    CondBr {
        cond: Value,
        true_target: String,
        false_target: String,
    },
    Switch {
        value: Value,
        default: String,
        cases: Vec<(i64, String)>,
    },
    Ret(Option<Value>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhiNode {
    pub dest: String,
    pub ty: Type,
    pub incoming: Vec<(Value, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock {
    pub label: String,
    pub phi_nodes: Vec<PhiNode>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: String,
}

impl Function {
    pub fn new(name: &str, params: Vec<Param>, return_type: Type) -> Self {
        Self {
            name: name.to_string(),
            params,
            return_type,
            blocks: Vec::new(),
            entry_block: String::new(),
        }
    }

    pub fn get_block(&self, label: &str) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.label == label)
    }

    pub fn get_block_mut(&mut self, label: &str) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.label == label)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub functions: Vec<Function>,
    pub globals: Vec<GlobalVariable>,
}

impl Type {
    pub fn i64() -> Self {
        Type::Int(IntegerType::I64)
    }

    pub fn f32() -> Self {
        Type::Float(FloatType::F32)
    }

    pub fn ptr(ty: Type) -> Self {
        Type::Pointer(Box::new(ty))
    }
}
