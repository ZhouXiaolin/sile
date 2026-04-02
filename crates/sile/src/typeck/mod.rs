pub mod error;

use std::collections::BTreeMap;

use crate::hir::{BuiltinOp, Expr, Kernel, ShapeExpr, Type};

#[derive(Clone, Debug)]
pub struct TypedKernel {
    pub kernel: Kernel,
    pub locals: BTreeMap<String, Type>,
}

pub fn check_kernel(kernel: &Kernel) -> Result<TypedKernel, error::TypeError> {
    let mut locals = BTreeMap::new();
    for stmt in &kernel.body {
        if let crate::hir::Stmt::Let { name, expr, .. } = stmt {
            let ty = infer_expr(expr, &locals)?;
            locals.insert(name.clone(), ty);
        }
    }
    Ok(TypedKernel {
        kernel: kernel.clone(),
        locals,
    })
}

fn infer_expr(expr: &Expr, _locals: &BTreeMap<String, Type>) -> Result<Type, error::TypeError> {
    match expr {
        Expr::Builtin { op, .. } => infer_builtin(*op, &[]),
        Expr::Var(_) => Ok(Type::Shape), // placeholder
        Expr::Shape(_) => Ok(Type::Shape),
        Expr::ScalarI32(_) => Ok(Type::Scalar(crate::hir::ElemType::F32)),
    }
}

fn infer_builtin(op: BuiltinOp, _args: &[Type]) -> Result<Type, error::TypeError> {
    match op {
        BuiltinOp::ProgramId => Ok(Type::Shape),
        BuiltinOp::LoadTile => Ok(Type::tile(
            crate::hir::ElemType::F32,
            ShapeExpr::symbol("S"),
        )),
        BuiltinOp::LoadTileLike2D => Ok(Type::tile(
            crate::hir::ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
        )),
        BuiltinOp::ReduceMax | BuiltinOp::ReduceSum => Ok(Type::tile(
            crate::hir::ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM")]),
        )),
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Div | BuiltinOp::Exp => Ok(Type::tile(
            crate::hir::ElemType::F32,
            ShapeExpr::symbol("S"),
        )),
        BuiltinOp::Reshape | BuiltinOp::Broadcast => Ok(Type::tile(
            crate::hir::ElemType::F32,
            ShapeExpr::symbol("S"),
        )),
        other => Err(error::TypeError::unsupported_builtin(format!("{other:?}"))),
    }
}
