use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelParam, KernelStmt};

pub fn parse_kernel(input: &syn::ItemFn) -> syn::Result<KernelDecl> {
    let params = input
        .sig
        .inputs
        .iter()
        .map(parse_param)
        .collect::<syn::Result<Vec<_>>>()?;

    let mut body = Vec::new();
    for stmt in &input.block.stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                let name = match &local.pat {
                    syn::Pat::Ident(pat) => pat.ident.clone(),
                    syn::Pat::Type(pat_type) => {
                        if let syn::Pat::Ident(inner) = pat_type.pat.as_ref() {
                            inner.ident.clone()
                        } else {
                            return Err(syn::Error::new_spanned(
                                &local.pat,
                                "expected ident pattern",
                            ));
                        }
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "expected ident pattern",
                        ))
                    }
                };
                let expr = local
                    .init
                    .as_ref()
                    .map(|init| parse_expr(init.expr.as_ref()))
                    .transpose()?
                    .ok_or_else(|| {
                        syn::Error::new_spanned(local, "let binding requires initializer")
                    })?;
                body.push(KernelStmt::Let { name, expr });
            }
            syn::Stmt::Expr(expr, _) => {
                if let syn::Expr::MethodCall(call) = expr {
                    if call.method == "store" {
                        let target = match call.receiver.as_ref() {
                            syn::Expr::Path(path) => {
                                path.path.segments.last().unwrap().ident.clone()
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &call.receiver,
                                    "store target must be an ident",
                                ))
                            }
                        };
                        let value = parse_expr(call.args.first().ok_or_else(|| {
                            syn::Error::new_spanned(call, "store requires one argument")
                        })?)?;
                        body.push(KernelStmt::Store { target, value });
                        continue;
                    }
                }
                return Err(syn::Error::new_spanned(
                    expr,
                    "unsupported kernel statement",
                ));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported kernel statement",
                ))
            }
        }
    }

    Ok(KernelDecl {
        name: input.sig.ident.clone(),
        params,
        body,
    })
}

fn parse_param(arg: &syn::FnArg) -> syn::Result<KernelParam> {
    let syn::FnArg::Typed(arg) = arg else {
        return Err(syn::Error::new_spanned(
            arg,
            "receiver parameters are unsupported",
        ));
    };
    let syn::Pat::Ident(pat) = arg.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &arg.pat,
            "expected ident parameter",
        ));
    };
    let syn::Type::Reference(reference) = arg.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &arg.ty,
            "kernel parameter must be a reference",
        ));
    };
    Ok(KernelParam {
        name: pat.ident.clone(),
        is_mut: reference.mutability.is_some(),
    })
}

fn parse_expr(expr: &syn::Expr) -> syn::Result<KernelExpr> {
    match expr {
        syn::Expr::MethodCall(call) => {
            let receiver = parse_expr(&call.receiver)?;
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::MethodCall {
                receiver: Box::new(receiver),
                method: call.method.clone(),
                args,
            })
        }
        syn::Expr::Binary(binary) => {
            let left = parse_expr(&binary.left)?;
            let right = parse_expr(&binary.right)?;
            let op = match binary.op {
                syn::BinOp::Add(_) => BinOpKind::Add,
                syn::BinOp::Sub(_) => BinOpKind::Sub,
                syn::BinOp::Mul(_) => BinOpKind::Mul,
                syn::BinOp::Div(_) => BinOpKind::Div,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &binary.op,
                        "unsupported binary operator",
                    ))
                }
            };
            Ok(KernelExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        syn::Expr::Path(path) => {
            let ident = path.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(path, "expected simple variable reference")
            })?;
            Ok(KernelExpr::Var(ident))
        }
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(lit_int) = &lit.lit {
                Ok(KernelExpr::Lit(lit_int.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    &lit.lit,
                    "only integer literals supported in kernel expressions",
                ))
            }
        }
        syn::Expr::Array(arr) => Ok(KernelExpr::Array(arr.clone())),
        syn::Expr::Field(field) => {
            let target = parse_expr(&field.base)?;
            let field = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            Ok(KernelExpr::FieldAccess {
                target: Box::new(target),
                field,
            })
        }
        syn::Expr::Call(call) => {
            let func = match call.func.as_ref() {
                syn::Expr::Path(path) => {
                    // Handle both simple idents and qualified paths like `sile::tile::id`
                    path.path
                        .segments
                        .last()
                        .map(|seg| seg.ident.clone())
                        .ok_or_else(|| {
                            syn::Error::new_spanned(&call.func, "expected function name")
                        })?
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        &call.func,
                        "expected simple function call",
                    ))
                }
            };
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::Call { func, args })
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported expression kind in kernel",
        )),
    }
}
