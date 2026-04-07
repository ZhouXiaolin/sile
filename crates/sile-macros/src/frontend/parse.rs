use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelParam, KernelShapeDim, KernelStmt};

pub fn parse_kernel(input: &syn::ItemFn) -> syn::Result<KernelDecl> {
    // Extract const params from generic parameters
    let const_params: Vec<syn::Ident> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Const(c) = p {
                Some(c.ident.clone())
            } else {
                None
            }
        })
        .collect();

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
                        ));
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
                if let syn::Expr::ForLoop(for_loop) = expr {
                    parse_for_loop(for_loop, &mut body)?;
                    continue;
                }
                if let syn::Expr::Assign(assign) = expr {
                    let name = match assign.left.as_ref() {
                        syn::Expr::Path(path) => path.path.segments.last().unwrap().ident.clone(),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &assign.left,
                                "assignment target must be an ident",
                            ));
                        }
                    };
                    let expr = parse_expr(&assign.right)?;
                    body.push(KernelStmt::Assign { name, expr });
                    continue;
                }
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
                                ));
                            }
                        };
                        let value = parse_expr(call.args.first().ok_or_else(|| {
                            syn::Error::new_spanned(call, "store requires one argument")
                        })?)?;
                        body.push(KernelStmt::Store { target, value });
                        continue;
                    }
                    if call.method == "atomic_add" {
                        let target = match call.receiver.as_ref() {
                            syn::Expr::Path(path) => {
                                path.path.segments.last().unwrap().ident.clone()
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &call.receiver,
                                    "atomic_add target must be an ident",
                                ));
                            }
                        };
                        let mut args = call.args.iter();
                        let index = parse_expr(args.next().ok_or_else(|| {
                            syn::Error::new_spanned(call, "atomic_add requires index argument")
                        })?)?;
                        let value = parse_expr(args.next().ok_or_else(|| {
                            syn::Error::new_spanned(call, "atomic_add requires value argument")
                        })?)?;
                        if args.next().is_some() {
                            return Err(syn::Error::new_spanned(
                                call,
                                "atomic_add expects exactly two arguments",
                            ));
                        }
                        body.push(KernelStmt::AtomicAdd {
                            target,
                            index,
                            value,
                        });
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
                ));
            }
        }
    }

    Ok(KernelDecl {
        name: input.sig.ident.clone(),
        const_params,
        params,
        body,
    })
}

fn parse_for_loop(for_loop: &syn::ExprForLoop, body: &mut Vec<KernelStmt>) -> syn::Result<()> {
    let var = match for_loop.pat.as_ref() {
        syn::Pat::Ident(pat) => pat.ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.pat,
                "for loop variable must be an ident",
            ));
        }
    };
    let (start, end) = match for_loop.expr.as_ref() {
        syn::Expr::Range(range) => {
            let start_expr = range.start.as_ref().ok_or_else(|| {
                syn::Error::new_spanned(range, "for loop range must have a start")
            })?;
            let end_expr = range
                .end
                .as_ref()
                .ok_or_else(|| syn::Error::new_spanned(range, "for loop range must have an end"))?;
            (parse_expr(start_expr)?, parse_expr(end_expr)?)
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.expr,
                "for loop expression must be a range",
            ));
        }
    };
    let mut loop_body = Vec::new();
    for inner_stmt in &for_loop.body.stmts {
        match inner_stmt {
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
                        ));
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
                loop_body.push(KernelStmt::Let { name, expr });
            }
            syn::Stmt::Expr(inner_expr, _) => {
                if let syn::Expr::Assign(assign) = inner_expr {
                    let name = match assign.left.as_ref() {
                        syn::Expr::Path(path) => path.path.segments.last().unwrap().ident.clone(),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &assign.left,
                                "assignment target must be an ident",
                            ));
                        }
                    };
                    let expr = parse_expr(&assign.right)?;
                    loop_body.push(KernelStmt::Assign { name, expr });
                    continue;
                }
                if let syn::Expr::MethodCall(call) = inner_expr {
                    if call.method == "store" {
                        let target = match call.receiver.as_ref() {
                            syn::Expr::Path(path) => {
                                path.path.segments.last().unwrap().ident.clone()
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &call.receiver,
                                    "store target must be an ident",
                                ));
                            }
                        };
                        let value = parse_expr(call.args.first().ok_or_else(|| {
                            syn::Error::new_spanned(call, "store requires one argument")
                        })?)?;
                        loop_body.push(KernelStmt::Store { target, value });
                        continue;
                    }
                    if call.method == "atomic_add" {
                        let target = match call.receiver.as_ref() {
                            syn::Expr::Path(path) => {
                                path.path.segments.last().unwrap().ident.clone()
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &call.receiver,
                                    "atomic_add target must be an ident",
                                ));
                            }
                        };
                        let mut args = call.args.iter();
                        let index = parse_expr(args.next().ok_or_else(|| {
                            syn::Error::new_spanned(call, "atomic_add requires index argument")
                        })?)?;
                        let value = parse_expr(args.next().ok_or_else(|| {
                            syn::Error::new_spanned(call, "atomic_add requires value argument")
                        })?)?;
                        if args.next().is_some() {
                            return Err(syn::Error::new_spanned(
                                call,
                                "atomic_add expects exactly two arguments",
                            ));
                        }
                        loop_body.push(KernelStmt::AtomicAdd {
                            target,
                            index,
                            value,
                        });
                        continue;
                    }
                }
                return Err(syn::Error::new_spanned(
                    inner_expr,
                    "unsupported kernel statement in for loop",
                ));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported statement in for loop",
                ));
            }
        }
    }
    body.push(KernelStmt::ForLoop {
        var,
        start,
        end,
        body: loop_body,
    });
    Ok(())
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

    let shape = extract_shape_from_type(&reference.elem);

    Ok(KernelParam {
        name: pat.ident.clone(),
        is_mut: reference.mutability.is_some(),
        shape,
    })
}

fn extract_shape_from_type(ty: &syn::Type) -> Option<Vec<KernelShapeDim>> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let last_seg = type_path.path.segments.last()?;
    if last_seg.ident != "Tensor" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    if args.args.len() < 2 {
        return None;
    }
    let syn::GenericArgument::Const(const_expr) = &args.args[1] else {
        return None;
    };
    extract_shape_expr(const_expr)
}

fn extract_shape_expr(expr: &syn::Expr) -> Option<Vec<KernelShapeDim>> {
    match expr {
        syn::Expr::Block(block) => {
            let tail = block.block.stmts.last()?;
            if let syn::Stmt::Expr(expr, _) = tail {
                extract_shape_expr(expr)
            } else {
                None
            }
        }
        syn::Expr::Array(arr) => {
            let mut shape = Vec::new();
            for elem in &arr.elems {
                let dim = match elem {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(lit_int),
                        ..
                    }) => {
                        let value = lit_int.base10_parse::<i64>().unwrap_or(-1);
                        if value == -1 {
                            KernelShapeDim::Dynamic
                        } else {
                            KernelShapeDim::Constant(value)
                        }
                    }
                    syn::Expr::Unary(syn::ExprUnary {
                        op: syn::UnOp::Neg(_),
                        expr,
                        ..
                    }) => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Int(lit_int),
                            ..
                        }) = expr.as_ref()
                        {
                            let value = lit_int.base10_parse::<i64>().unwrap_or(1);
                            KernelShapeDim::Constant(-value)
                        } else {
                            KernelShapeDim::Dynamic
                        }
                    }
                    syn::Expr::Path(path) => path
                        .path
                        .get_ident()
                        .cloned()
                        .map(KernelShapeDim::Symbol)
                        .unwrap_or(KernelShapeDim::Dynamic),
                    _ => KernelShapeDim::Dynamic,
                };
                shape.push(dim);
            }
            if shape.is_empty() { None } else { Some(shape) }
        }
        _ => None,
    }
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
                    ));
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
            } else if let syn::Lit::Float(lit_float) = &lit.lit {
                Ok(KernelExpr::FloatLit(lit_float.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    &lit.lit,
                    "only integer and float literals supported in kernel expressions",
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
                syn::Expr::Path(path) => path
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.clone())
                    .ok_or_else(|| syn::Error::new_spanned(&call.func, "expected function name"))?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &call.func,
                        "expected simple function call",
                    ));
                }
            };
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::Call { func, args })
        }
        syn::Expr::Reference(reference) => parse_expr(&reference.expr),
        syn::Expr::Paren(paren) => parse_expr(&paren.expr),
        syn::Expr::Index(index) => {
            let target = parse_expr(&index.expr)?;
            let idx = parse_expr(&index.index)?;
            Ok(KernelExpr::Index {
                target: Box::new(target),
                index: Box::new(idx),
            })
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported expression kind in kernel",
        )),
    }
}
