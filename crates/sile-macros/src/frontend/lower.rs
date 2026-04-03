use quote::quote;

use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelStmt};

pub fn lower_kernel_to_hir(decl: &KernelDecl) -> proc_macro2::TokenStream {
    let name = decl.name.to_string();
    let params = decl.params.iter().map(|param| {
        let name = param.name.to_string();
        let kind = if param.is_mut {
            quote! { ::sile::hir::ParamKind::Output }
        } else {
            quote! { ::sile::hir::ParamKind::Input }
        };
        let shape = if let Some(dims) = &param.shape {
            let dim_exprs: Vec<_> = dims
                .iter()
                .map(|d| {
                    if *d == -1 {
                        quote! { ::sile::hir::ShapeExpr::dynamic() }
                    } else {
                        quote! { ::sile::hir::ShapeExpr::constant(#d as i64) }
                    }
                })
                .collect();
            quote! {
                ::sile::hir::ShapeExpr::tuple(vec![#(#dim_exprs),*])
            }
        } else {
            quote! { ::sile::hir::ShapeExpr::dynamic() }
        };
        quote! {
            ::sile::hir::Param::new(
                #name,
                #kind,
                ::sile::hir::Type::tensor(
                    ::sile::hir::ElemType::F32,
                    #shape,
                ),
            )
        }
    });
    let body = decl.body.iter().map(|stmt| lower_stmt(stmt));
    quote! {
        ::sile::hir::Kernel::new(
            #name,
            vec![],
            vec![#(#params),*],
            vec![#(#body),*],
        )
    }
}

fn lower_stmt(stmt: &KernelStmt) -> proc_macro2::TokenStream {
    match stmt {
        KernelStmt::Let { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr);
            quote! {
                ::sile::hir::Stmt::Let {
                    name: #name.to_string(),
                    ty: None,
                    expr: #expr,
                }
            }
        }
        KernelStmt::Store { target, value } => {
            let target = target.to_string();
            let value = lower_expr(value);
            quote! {
                ::sile::hir::Stmt::Store {
                    target: #target.to_string(),
                    value: #value,
                }
            }
        }
        KernelStmt::ForLoop {
            var,
            start,
            end,
            body,
        } => {
            let var_name = var.to_string();
            let start_val = lower_expr(start);
            let end_val = lower_expr(end);
            let body_stmts: Vec<_> = body.iter().map(|s| lower_stmt(s)).collect();
            quote! {
                ::sile::hir::Stmt::ForLoop {
                    var: #var_name.to_string(),
                    start: #start_val,
                    end: #end_val,
                    body: vec![#(#body_stmts),*],
                }
            }
        }
    }
}

fn lower_expr(expr: &KernelExpr) -> proc_macro2::TokenStream {
    match expr {
        KernelExpr::Var(ident) => {
            let name = ident.to_string();
            quote! { ::sile::hir::Expr::Var(#name.to_string()) }
        }
        KernelExpr::Lit(lit) => {
            let val: i64 = lit.base10_parse().unwrap_or(0);
            quote! { ::sile::hir::Expr::ScalarI64(#val) }
        }
        KernelExpr::FloatLit(lit) => {
            let val: f32 = lit.base10_parse().unwrap_or(0.0);
            quote! { ::sile::hir::Expr::ScalarF32(#val) }
        }
        KernelExpr::Call { func, args } => {
            let func_name = func.to_string();
            let args_exprs: Vec<_> = args.iter().map(lower_expr).collect();
            match func_name.as_str() {
                "load_tile_like_2d" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::LoadTileLike2D,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "constant" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::Constant,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "mma" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::Mma,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "reduce_max" | "reduce_sum" | "reshape" | "broadcast" | "exp" | "shape_of"
                | "id" => {
                    let op = match func_name.as_str() {
                        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
                        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
                        "reshape" => quote! { ::sile::hir::BuiltinOp::Reshape },
                        "broadcast" => quote! { ::sile::hir::BuiltinOp::Broadcast },
                        "exp" => quote! { ::sile::hir::BuiltinOp::Exp },
                        "shape_of" => quote! { ::sile::hir::BuiltinOp::ShapeOf },
                        "id" => quote! { ::sile::hir::BuiltinOp::ProgramId },
                        _ => unreachable!(),
                    };
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: #op,
                            args: vec![#(#args_exprs),*],
                        }
                    }
                }
                _ => {
                    quote! { ::sile::hir::Expr::Var(#func_name.to_string()) }
                }
            }
        }
        KernelExpr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let method_name = method.to_string();
            let receiver_expr = lower_expr(receiver);
            let args_exprs: Vec<_> = args.iter().map(lower_expr).collect();
            match method_name.as_str() {
                "load_tile" => {
                    let all_args = [receiver_expr].into_iter().chain(args_exprs);
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::LoadTile,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "reshape" | "broadcast" | "reduce_max" | "reduce_sum" | "exp" | "shape"
                | "load_tile_like_2d" => {
                    let op = match method_name.as_str() {
                        "reshape" => quote! { ::sile::hir::BuiltinOp::Reshape },
                        "broadcast" => quote! { ::sile::hir::BuiltinOp::Broadcast },
                        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
                        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
                        "exp" => quote! { ::sile::hir::BuiltinOp::Exp },
                        "shape" => quote! { ::sile::hir::BuiltinOp::ShapeOf },
                        "load_tile_like_2d" => quote! { ::sile::hir::BuiltinOp::LoadTileLike2D },
                        _ => unreachable!(),
                    };
                    let all_args = [receiver_expr].into_iter().chain(args_exprs);
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: #op,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "store" => {
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::Store,
                            args: vec![#receiver_expr],
                        }
                    }
                }
                _ => {
                    quote! { ::sile::hir::Expr::Var(#method_name.to_string()) }
                }
            }
        }
        KernelExpr::BinaryOp { left, op, right } => {
            let left_expr = lower_expr(left);
            let right_expr = lower_expr(right);
            let hir_op = match op {
                BinOpKind::Add => quote! { ::sile::hir::BuiltinOp::Add },
                BinOpKind::Sub => quote! { ::sile::hir::BuiltinOp::Sub },
                BinOpKind::Mul => quote! { ::sile::hir::BuiltinOp::Mul },
                BinOpKind::Div => quote! { ::sile::hir::BuiltinOp::Div },
            };
            quote! {
                ::sile::hir::Expr::Builtin {
                    op: #hir_op,
                    args: vec![#left_expr, #right_expr],
                }
            }
        }
        KernelExpr::Array(arr) => {
            let elems: Vec<_> = arr
                .elems
                .iter()
                .map(|elem| {
                    if let syn::Expr::Path(path) = elem {
                        if let Some(ident) = path.path.get_ident() {
                            let name = ident.to_string();
                            return quote! { ::sile::hir::ShapeExpr::symbol(#name) };
                        }
                    }
                    if let syn::Expr::Lit(lit) = elem {
                        if let syn::Lit::Int(lit_int) = &lit.lit {
                            let val: i64 = lit_int.base10_parse().unwrap_or(-1);
                            return quote! { ::sile::hir::ShapeExpr::constant(#val) };
                        }
                    }
                    quote! { ::sile::hir::ShapeExpr::dynamic() }
                })
                .collect();
            quote! {
                ::sile::hir::Expr::Shape(
                    ::sile::hir::ShapeExpr::tuple(vec![#(#elems),*])
                )
            }
        }
        KernelExpr::FieldAccess { target, field } => {
            // Handle a.shape().N -> ShapeDim(ShapeOf(a), N)
            if let KernelExpr::MethodCall {
                receiver, method, ..
            } = target.as_ref()
            {
                if method.to_string() == "shape" {
                    let dim_idx: i64 = field.parse().unwrap_or(0);
                    let receiver_lowered = lower_expr(receiver);
                    return quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ShapeDim,
                            args: vec![#receiver_lowered, ::sile::hir::Expr::ScalarI64(#dim_idx)],
                        }
                    };
                }
            }
            // Handle tile::id().N -> ShapeDim(ProgramId, N)
            if let KernelExpr::Call { func, .. } = target.as_ref() {
                if func.to_string() == "id" {
                    let dim_idx: i64 = field.parse().unwrap_or(0);
                    return quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ShapeDim,
                            args: vec![
                                ::sile::hir::Expr::Builtin {
                                    op: ::sile::hir::BuiltinOp::ProgramId,
                                    args: vec![],
                                },
                                ::sile::hir::Expr::ScalarI64(#dim_idx),
                            ],
                        }
                    };
                }
            }
            // Fallback: just pass through for .0, otherwise Var
            if field == "0" {
                lower_expr(target)
            } else {
                let field_name = field.to_string();
                quote! { ::sile::hir::Expr::Var(#field_name.to_string()) }
            }
        }
        KernelExpr::Index { target, index } => {
            // Handle a.shape()[N] -> ShapeDim(ShapeOf(a), N)
            if let KernelExpr::MethodCall {
                receiver, method, ..
            } = target.as_ref()
            {
                if method.to_string() == "shape" {
                    let dim_idx = match index.as_ref() {
                        KernelExpr::Lit(lit) => lit.base10_parse::<i64>().unwrap_or(0),
                        KernelExpr::Var(ident) => {
                            let name = ident.to_string();
                            return quote! {
                                ::sile::hir::Expr::Builtin {
                                    op: ::sile::hir::BuiltinOp::ShapeDim,
                                    args: vec![
                                        ::sile::hir::Expr::Builtin {
                                            op: ::sile::hir::BuiltinOp::ShapeOf,
                                            args: vec![#(lower_expr(receiver))],
                                        },
                                        ::sile::hir::Expr::Var(#name.to_string()),
                                    ],
                                }
                            };
                        }
                        _ => 0,
                    };
                    let receiver_lowered = lower_expr(receiver);
                    return quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ShapeDim,
                            args: vec![#receiver_lowered, ::sile::hir::Expr::ScalarI64(#dim_idx)],
                        }
                    };
                }
            }
            // Fallback: just lower both sides
            let target_lowered = lower_expr(target);
            let index_lowered = lower_expr(index);
            quote! { ::sile::hir::Expr::Var("idx".to_string()) }
        }
    }
}
