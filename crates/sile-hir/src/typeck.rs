use std::collections::BTreeMap;
use std::fmt;

use crate::types::{ElemType, ShapeExpr, Type};
use crate::{BuiltinOp, Expr, Kernel, Stmt};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    message: String,
}

impl TypeError {
    pub fn unsupported_expr(kind: &str) -> Self {
        Self {
            message: format!("unsupported expression kind: {kind}"),
        }
    }

    pub fn unsupported_builtin(name: impl Into<String>) -> Self {
        Self {
            message: format!("unsupported builtin: {}", name.into()),
        }
    }

    pub fn invalid_kernel(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn invalid_pipeline(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TypeError {}

#[derive(Clone, Debug)]
pub struct TypedKernel {
    pub kernel: Kernel,
    pub locals: BTreeMap<String, Type>,
}

pub fn check_kernel(kernel: &Kernel) -> Result<TypedKernel, TypeError> {
    let mut locals = BTreeMap::new();
    for stmt in &kernel.body {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let ty = infer_expr(expr, &locals)?;
                locals.insert(name.clone(), ty);
            }
            Stmt::Assign { name, expr } => {
                let ty = infer_expr(expr, &locals)?;
                locals.insert(name.clone(), ty);
            }
            Stmt::Store { .. } | Stmt::ForLoop { .. } => {}
            Stmt::AtomicAdd { index, value, .. } => {
                let _ = infer_expr(index, &locals)?;
                let _ = infer_expr(value, &locals)?;
            }
        }
    }
    Ok(TypedKernel {
        kernel: kernel.clone(),
        locals,
    })
}

fn infer_expr(expr: &Expr, _locals: &BTreeMap<String, Type>) -> Result<Type, TypeError> {
    match expr {
        Expr::Builtin { op, args } => infer_builtin(*op, args),
        Expr::Var(_) => Ok(Type::Shape),
        Expr::Shape(_) => Ok(Type::Shape),
        Expr::ScalarI64(_) => Ok(Type::Scalar(ElemType::F32)),
        Expr::ScalarF32(_) => Ok(Type::Scalar(ElemType::F32)),
    }
}

fn infer_builtin(op: BuiltinOp, _args: &[Expr]) -> Result<Type, TypeError> {
    match op {
        BuiltinOp::ProgramId => Ok(Type::Shape),
        BuiltinOp::LoadTile => Ok(Type::tile(ElemType::F32, ShapeExpr::symbol("S"))),
        BuiltinOp::LoadTileLike2D => Ok(Type::tile(
            ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
        )),
        BuiltinOp::ReduceMax | BuiltinOp::ReduceSum => Ok(Type::tile(
            ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM")]),
        )),
        BuiltinOp::Max | BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Div | BuiltinOp::Exp => {
            Ok(Type::tile(ElemType::F32, ShapeExpr::symbol("S")))
        }
        BuiltinOp::Reshape | BuiltinOp::Broadcast => {
            Ok(Type::tile(ElemType::F32, ShapeExpr::symbol("S")))
        }
        BuiltinOp::Mma => Ok(Type::tile(
            ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
        )),
        BuiltinOp::Constant => Ok(Type::tile(
            ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
        )),
        BuiltinOp::Index => Ok(Type::Scalar(ElemType::F32)),
        BuiltinOp::ScalarDiv | BuiltinOp::ShapeDim | BuiltinOp::ShapeOf => Ok(Type::Shape),
        other => Err(TypeError::unsupported_builtin(format!("{other:?}"))),
    }
}
