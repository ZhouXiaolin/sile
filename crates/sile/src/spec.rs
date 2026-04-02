use crate::kernel::{KernelArg, LaunchConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamRef {
    pub index: usize,
    pub kind: ParamKind,
    pub ty: ScalarType,
    pub shape: &'static [i64],
}

impl ParamRef {
    pub const fn input_f32(index: usize, shape: &'static [i64]) -> Self {
        Self {
            index,
            kind: ParamKind::Input,
            ty: ScalarType::F32,
            shape,
        }
    }

    pub const fn output_f32(index: usize, shape: &'static [i64]) -> Self {
        Self {
            index,
            kind: ParamKind::Output,
            ty: ScalarType::F32,
            shape,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeRef {
    Param {
        index: usize,
        kind: ParamKind,
        ty: ScalarType,
        shape: &'static [i64],
    },
    ConstF32 {
        value: f32,
        shape: &'static [i64],
    },
    LoadTile {
        param: usize,
        tile: TileExpr,
        shape: &'static [i64],
    },
    Binary {
        op: BinaryOp,
        lhs: u32,
        rhs: u32,
        shape: &'static [i64],
    },
}

impl NodeRef {
    pub const fn load_tile(param: usize, tile: TileExpr, shape: &'static [i64]) -> Self {
        Self::LoadTile {
            param,
            tile,
            shape,
        }
    }

    pub const fn binary(op: BinaryOp, lhs: u32, rhs: u32, shape: &'static [i64]) -> Self {
        Self::Binary {
            op,
            lhs,
            rhs,
            shape,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreRef {
    pub param: usize,
    pub tile: TileExpr,
    pub value: u32,
}

impl StoreRef {
    pub const fn new(param: usize, tile: TileExpr, value: u32) -> Self {
        Self {
            param,
            tile,
            value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelSpecRef {
    pub name: &'static str,
    pub params: &'static [ParamRef],
    pub nodes: &'static [NodeRef],
    pub stores: &'static [StoreRef],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<i64>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<i64>>) -> Self {
        Self {
            dims: dims.into(),
        }
    }

    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    pub fn element_count(&self) -> i64 {
        self.dims.iter().product()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileExpr {
    pub axis: usize,
}

impl TileExpr {
    pub fn grid_x() -> Self {
        Self { axis: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Param {
        index: usize,
        kind: ParamKind,
        ty: ScalarType,
        shape: Shape,
    },
    ConstF32 {
        value: f32,
        shape: Shape,
    },
    LoadTile {
        param: usize,
        tile: TileExpr,
        shape: Shape,
    },
    Binary {
        op: BinaryOp,
        lhs: u32,
        rhs: u32,
        shape: Shape,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Store {
    pub param: usize,
    pub tile: TileExpr,
    pub value: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub index: usize,
    pub kind: ParamKind,
    pub ty: ScalarType,
    pub shape: Shape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelSpec {
    pub name: String,
    pub params: Vec<Param>,
    pub nodes: Vec<Node>,
    pub stores: Vec<Store>,
}

impl From<&'static KernelSpecRef> for KernelSpec {
    fn from(spec: &'static KernelSpecRef) -> Self {
        Self {
            name: spec.name.to_string(),
            params: spec
                .params
                .iter()
                .map(|p| Param {
                    index: p.index,
                    kind: p.kind,
                    ty: p.ty,
                    shape: Shape::new(p.shape.to_vec()),
                })
                .collect(),
            nodes: spec
                .nodes
                .iter()
                .map(|n| match n {
                    NodeRef::Param {
                        index,
                        kind,
                        ty,
                        shape,
                    } => Node::Param {
                        index: *index,
                        kind: *kind,
                        ty: *ty,
                        shape: Shape::new(shape.to_vec()),
                    },
                    NodeRef::ConstF32 { value, shape } => Node::ConstF32 {
                        value: *value,
                        shape: Shape::new(shape.to_vec()),
                    },
                    NodeRef::LoadTile {
                        param,
                        tile,
                        shape,
                    } => Node::LoadTile {
                        param: *param,
                        tile: *tile,
                        shape: Shape::new(shape.to_vec()),
                    },
                    NodeRef::Binary {
                        op,
                        lhs,
                        rhs,
                        shape,
                    } => Node::Binary {
                        op: *op,
                        lhs: *lhs,
                        rhs: *rhs,
                        shape: Shape::new(shape.to_vec()),
                    },
                })
                .collect(),
            stores: spec
                .stores
                .iter()
                .map(|s| Store {
                    param: s.param,
                    tile: s.tile,
                    value: s.value,
                })
                .collect(),
        }
    }
}

impl KernelSpec {
    pub fn validate_launch(
        &self,
        args: &[KernelArg<'_>],
        launch: &LaunchConfig,
    ) -> crate::Result<()> {
        if launch.grid[1] != 1 || launch.grid[2] != 1 {
            return Err(crate::Error::Shape("MVP only supports 1D grid".into()));
        }

        let tile_size = self.tile_size()? as u32;
        let expected_len = launch.grid[0] * tile_size;

        for arg in args {
            if arg.shape.len() != 1 {
                return Err(crate::Error::Shape("MVP only supports 1D tensors".into()));
            }
            if arg.shape[0] as u32 != expected_len {
                return Err(crate::Error::Shape(format!(
                    "grid implies {expected_len} elements but tensor has {}",
                    arg.shape[0]
                )));
            }
        }

        Ok(())
    }

    pub fn tile_size(&self) -> crate::Result<i64> {
        let mut sizes = self.nodes.iter().filter_map(|node| match node {
            Node::LoadTile { shape, .. } => Some(shape.element_count()),
            _ => None,
        });
        let first = sizes
            .next()
            .ok_or_else(|| crate::Error::Shape("kernel has no tile loads".into()))?;
        if sizes.any(|size| size != first) {
            return Err(crate::Error::Shape("all tile shapes must match".into()));
        }
        Ok(first)
    }
}
