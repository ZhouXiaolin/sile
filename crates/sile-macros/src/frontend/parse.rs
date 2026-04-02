use super::ast::{KernelDecl, KernelParam, KernelStmt};

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
                let syn::Pat::Ident(pat) = &local.pat else {
                    return Err(syn::Error::new_spanned(local, "expected ident pattern"));
                };
                let expr = local
                    .init
                    .as_ref()
                    .map(|init| init.expr.as_ref().clone())
                    .ok_or_else(|| {
                        syn::Error::new_spanned(local, "let binding requires initializer")
                    })?;
                body.push(KernelStmt::Let {
                    name: pat.ident.clone(),
                    expr,
                });
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
                        body.push(KernelStmt::Store {
                            target,
                            value: call.args.first().cloned().ok_or_else(|| {
                                syn::Error::new_spanned(call, "store requires one argument")
                            })?,
                        });
                        continue;
                    }
                }
                return Err(syn::Error::new_spanned(expr, "unsupported kernel statement"));
            }
            other => {
                return Err(syn::Error::new_spanned(other, "unsupported kernel statement"))
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
        return Err(syn::Error::new_spanned(&arg.pat, "expected ident parameter"));
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
