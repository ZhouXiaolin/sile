use sile_hir::Kernel;
use sile_hir::types::ElemType;

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableKernel {
    pub name: String,
    pub abi: KernelAbi,
    pub func: Kernel,
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
    pub kind: ElemType,
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
    pub program_id_dims: [u32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfoTable {
    pub params: Vec<KernelParamAbi>,
    pub instructions: Vec<ValueInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueInfo {
    Buffer {
        elem: ElemType,
        size: usize,
        param_index: usize,
    },
    Scalar {
        elem: ElemType,
    },
    Index {
        expr: usize,
    },
    Shape {
        total_dims: usize,
    },
    Tile {
        elem: ElemType,
        rank: usize,
        layout: ShapeLayout,
        param_index: usize,
    },
    Void,
}
