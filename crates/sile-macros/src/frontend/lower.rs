use quote::quote;

use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelShapeDim, KernelStmt};

pub fn lower_kernel_to_hir(decl: &KernelDecl) -> proc_macro2::TokenStream {
    let name = decl.name.to_string();
    let _const_params: Vec<String> = decl.const_params.iter().map(|p| p.to_string()).collect();
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
                .map(|d| match d {
                    KernelShapeDim::Dynamic => quote! { ::sile::hir::ShapeExpr::dynamic() },
                    KernelShapeDim::Constant(value) => {
                        quote! { ::sile::hir::ShapeExpr::constant(#value as i64) }
                    }
                    KernelShapeDim::Symbol(ident) => {
                        let name = ident.to_string();
                        quote! { ::sile::hir::ShapeExpr::symbol(#name) }
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
    let body = decl
        .body
        .iter()
        .map(|stmt| lower_stmt(stmt, &decl.const_params));
    let const_params_exprs: Vec<_> = decl
        .const_params
        .iter()
        .map(|p| {
            let name = p.to_string();
            let ident = p;
            quote! { (#name.to_string(), #ident as i64) }
        })
        .collect();
    quote! {
        ::sile::hir::Kernel::new(
            #name,
            vec![#(#const_params_exprs),*],
            vec![#(#params),*],
            vec![#(#body),*],
        )
    }
}

fn lower_stmt(stmt: &KernelStmt, const_params: &[syn::Ident]) -> proc_macro2::TokenStream {
    match stmt {
        KernelStmt::Let { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr, const_params);
            quote! {
                ::sile::hir::Stmt::Let {
                    name: #name.to_string(),
                    ty: None,
                    expr: #expr,
                }
            }
        }
        KernelStmt::Assign { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr, const_params);
            quote! {
                ::sile::hir::Stmt::Assign {
                    name: #name.to_string(),
                    expr: #expr,
                }
            }
        }
        KernelStmt::Store { target, value } => {
            let target = target.to_string();
            let value = lower_expr(value, const_params);
            quote! {
                ::sile::hir::Stmt::Store {
                    target: #target.to_string(),
                    value: #value,
                }
            }
        }
        KernelStmt::AtomicAdd {
            target,
            index,
            value,
        } => {
            let target = target.to_string();
            let index = lower_expr(index, const_params);
            let value = lower_expr(value, const_params);
            quote! {
                ::sile::hir::Stmt::AtomicAdd {
                    target: #target.to_string(),
                    index: #index,
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
            let start_val = lower_expr(start, const_params);
            let end_val = lower_expr(end, const_params);
            let body_stmts: Vec<_> = body.iter().map(|s| lower_stmt(s, const_params)).collect();
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

fn lower_expr(expr: &KernelExpr, const_params: &[syn::Ident]) -> proc_macro2::TokenStream {
    match expr {
        KernelExpr::Var(ident) => {
            let name = ident.to_string();
            // Leave as Var - will be resolved in SSA lowering using const_values map
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
            let args_exprs: Vec<_> = args.iter().map(|a| lower_expr(a, const_params)).collect();
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
                "max_tile" | "reduce_max" | "reduce_sum" | "exp" => {
                    let op = match func_name.as_str() {
                        "max_tile" => quote! { ::sile::hir::BuiltinOp::Max },
                        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
                        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
                        "exp" => quote! { ::sile::hir::BuiltinOp::Exp },
                        _ => unreachable!(),
                    };
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: #op,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                // tile::program_id(axis) -> ProgramId(axis). Without this
                // branch the call degrades to a plain variable, and any
                // arithmetic wrapped around it, such as `program_id(0) + 1`,
                // disappears before it reaches the HIR.
                "program_id" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ProgramId,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                // tile::num_programs(axis) -> NumPrograms(axis)
                "num_programs" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::NumPrograms,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                // tile::num_tiles(buf, axis) -> NumTiles(buf, axis)
                "num_tiles" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::NumTiles,
                            args: vec![#(#all_args),*],
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
            let receiver_expr = lower_expr(receiver, const_params);
            let args_exprs: Vec<_> = args.iter().map(|a| lower_expr(a, const_params)).collect();
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
                "clone" => {
                    // .clone() on a Tile is just a passthrough - return the receiver
                    receiver_expr
                }
                _ => {
                    quote! { ::sile::hir::Expr::Var(#method_name.to_string()) }
                }
            }
        }
        KernelExpr::BinaryOp { left, op, right } => {
            let left_expr = lower_expr(left, const_params);
            let right_expr = lower_expr(right, const_params);
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
            // An array literal is overloaded. In shape position it is a shape:
            // `load_tile([TILE], [pid])`. In index position it is a list of
            // runtime scalars, as CuTile Rust writes `part.load([pid_m, k])`.
            // Only a plain identifier or an integer literal is a shape
            // dimension. Anything else must survive as an expression, or the
            // arithmetic behind it is dropped and the tile reads the wrong data.
            let mut shape_elems = Vec::new();
            let mut index_elems = Vec::new();
            let mut pure_shape = true;
            for elem in &arr.elems {
                if let Some(dim) = shape_literal_dim(elem) {
                    shape_elems.push(quote! { ::sile::hir::ShapeExpr::constant(#dim) });
                } else if let Some(name) = shape_literal_symbol(elem) {
                    shape_elems.push(quote! { ::sile::hir::ShapeExpr::symbol(#name) });
                } else {
                    pure_shape = false;
                    let parsed = super::parse::parse_expr_pub(elem);
                    index_elems.push(match parsed {
                        Ok(kernel_expr) => lower_expr(&kernel_expr, const_params),
                        Err(_) => quote! { ::sile::hir::Expr::ScalarI64(0) },
                    });
                }
            }
            if pure_shape {
                quote! {
                    ::sile::hir::Expr::Shape(
                        ::sile::hir::ShapeExpr::tuple(vec![#(#shape_elems),*])
                    )
                }
            } else {
                quote! { ::sile::hir::Expr::TileIndex(vec![#(#index_elems),*]) }
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
                    let receiver_lowered = lower_expr(receiver, const_params);
                    return quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ShapeDim,
                            args: vec![#receiver_lowered, ::sile::hir::Expr::ScalarI64(#dim_idx)],
                        }
                    };
                }
            }
            // Handle tile::id().N and tile::program_id().N -> ShapeDim(ProgramId, N).
            // `id()` is the historical spelling; `program_id` matches CuTile Rust.
            if let KernelExpr::Call { func, .. } = target.as_ref() {
                let func_name = func.to_string();
                if func_name == "id" || func_name == "program_id" {
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
                lower_expr(target, const_params)
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
                                            args: vec![#(lower_expr(receiver, const_params))],
                                        },
                                        ::sile::hir::Expr::Var(#name.to_string()),
                                    ],
                                }
                            };
                        }
                        _ => 0,
                    };
                    let receiver_lowered = lower_expr(receiver, const_params);
                    return quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::ShapeDim,
                            args: vec![#receiver_lowered, ::sile::hir::Expr::ScalarI64(#dim_idx)],
                        }
                    };
                }
            }
            // Fallback: just lower both sides
            let _target_lowered = lower_expr(target, const_params);
            let _index_lowered = lower_expr(index, const_params);
            quote! {
                ::sile::hir::Expr::Builtin {
                    op: ::sile::hir::BuiltinOp::Index,
                    args: vec![#_target_lowered, #_index_lowered],
                }
            }
        }
    }
}

/// A plain integer in an array literal, or `-1` for a dynamic dimension.
fn shape_literal_dim(elem: &syn::Expr) -> Option<i64> {
    match elem {
        syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(v), .. }) => {
            Some(v.base10_parse().unwrap_or(-1))
        }
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr,
            ..
        }) => shape_literal_dim(expr).map(|v| -v),
        _ => None,
    }
}

/// A bare identifier, which is a shape symbol only when the array holds nothing
/// but shape dimensions.
fn shape_literal_symbol(elem: &syn::Expr) -> Option<String> {
    match elem {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => {
            path.path.get_ident().map(|ident| ident.to_string())
        }
        _ => None,
    }
}
