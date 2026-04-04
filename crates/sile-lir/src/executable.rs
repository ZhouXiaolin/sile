use sile_hir::{ParamKind, types::ElemType};

use crate::ir::Function;

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableKernel {
    pub name: String,
    pub abi: KernelAbi,
    pub func: Function,
    pub value_info: ValueInfoTable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    pub params: Vec<KernelParamAbi>,
    pub shape_layout: ShapeLayout,
    pub launch: LaunchSemantics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelParamAbi {
    pub index: usize,
    pub name: String,
    pub kind: ParamKind,
    pub elem: ElemType,
    pub rank: usize,
    pub passing: ParamPassing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamPassing {
    Buffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeLayout {
    pub total_dims: usize,
    pub offsets: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchSemantics {
    pub program_id_dims: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfoTable {
    pub params: Vec<ValueInfo>,
    pub instructions: Vec<ValueInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueInfo {
    Buffer {
        elem: ElemType,
        rank: usize,
    },
    Scalar {
        elem: ElemType,
    },
    Index,
    Shape,
    Tile {
        elem: ElemType,
        rows: i64,
        cols: i64,
    },
    Void,
}
