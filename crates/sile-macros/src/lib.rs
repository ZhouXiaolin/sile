mod frontend;

use proc_macro::TokenStream;
use syn::FnArg;

fn rewrite_type(ty: &syn::Type) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            let path = &type_path.path;
            if let Some(last_seg) = path.segments.last() {
                if last_seg.ident == "Tensor" || last_seg.ident == "Tile" {
                    if let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments {
                        let type_args: Vec<_> = args.args.iter().collect();
                        // Keep only the first type arg (elem type), drop the rank/shape param
                        if type_args.len() >= 2 {
                            if let syn::GenericArgument::Type(elem_type) = type_args[0] {
                                let mut new_path = path.clone();
                                if let Some(last_seg) = new_path.segments.last_mut() {
                                    let mut new_args: syn::punctuated::Punctuated<
                                        syn::GenericArgument,
                                        syn::Token![,],
                                    > = syn::punctuated::Punctuated::new();
                                    new_args.push(syn::GenericArgument::Type(elem_type.clone()));
                                    last_seg.arguments = syn::PathArguments::AngleBracketed(
                                        syn::AngleBracketedGenericArguments {
                                            colon2_token: None,
                                            lt_token: syn::Token![<](proc_macro2::Span::call_site()),
                                            args: new_args,
                                            gt_token: syn::Token![>](proc_macro2::Span::call_site()),
                                        },
                                    );
                                }
                                return syn::Type::Path(syn::TypePath {
                                    qself: type_path.qself.clone(),
                                    path: new_path,
                                });
                            }
                        }
                    }
                }
            }
            ty.clone()
        }
        syn::Type::Reference(type_ref) => {
            let new_elem = rewrite_type(&type_ref.elem);
            syn::Type::Reference(syn::TypeReference {
                elem: Box::new(new_elem),
                ..type_ref.clone()
            })
        }
        _ => ty.clone(),
    }
}

/// Recursively rewrite expressions in kernel body.
/// Transforms `expr.shape().N` into `expr.shape()[N]` so that
/// DSL tuple-style dimension access works with the runtime `&[i64]` return type.
fn rewrite_expr(expr: &syn::Expr) -> syn::Expr {
    match expr {
        syn::Expr::Field(field) => {
            // Check if base is a method call to `shape()`
            if let syn::Expr::MethodCall(call) = field.base.as_ref() {
                if call.method == "shape" && call.args.is_empty() {
                    if let syn::Member::Unnamed(idx) = &field.member {
                        // Rewrite to index expression: expr.shape()[N]
                        let index = syn::Expr::Lit(syn::ExprLit {
                            attrs: vec![],
                            lit: syn::Lit::Int(syn::LitInt::new(
                                &idx.index.to_string(),
                                proc_macro2::Span::call_site(),
                            )),
                        });
                        return syn::Expr::Index(syn::ExprIndex {
                            attrs: vec![],
                            expr: field.base.clone(),
                            bracket_token: syn::token::Bracket::default(),
                            index: Box::new(index),
                        });
                    }
                }
            }
            // Default: recurse into base
            syn::Expr::Field(syn::ExprField {
                base: Box::new(rewrite_expr(&field.base)),
                ..field.clone()
            })
        }
        syn::Expr::MethodCall(call) => syn::Expr::MethodCall(syn::ExprMethodCall {
            receiver: Box::new(rewrite_expr(&call.receiver)),
            args: call.args.iter().map(rewrite_expr).collect(),
            ..call.clone()
        }),
        syn::Expr::Binary(binary) => syn::Expr::Binary(syn::ExprBinary {
            left: Box::new(rewrite_expr(&binary.left)),
            right: Box::new(rewrite_expr(&binary.right)),
            ..binary.clone()
        }),
        syn::Expr::Call(call) => syn::Expr::Call(syn::ExprCall {
            func: Box::new(rewrite_expr(&call.func)),
            args: call.args.iter().map(rewrite_expr).collect(),
            ..call.clone()
        }),
        syn::Expr::Reference(reference) => syn::Expr::Reference(syn::ExprReference {
            expr: Box::new(rewrite_expr(&reference.expr)),
            ..reference.clone()
        }),
        syn::Expr::Array(arr) => syn::Expr::Array(syn::ExprArray {
            elems: arr.elems.iter().map(rewrite_expr).collect(),
            ..arr.clone()
        }),
        syn::Expr::Path(_) | syn::Expr::Lit(_) | syn::Expr::Range(_) => expr.clone(),
        syn::Expr::ForLoop(for_loop) => {
            // Rewrite statements inside for loop body
            let new_stmts: Vec<_> = for_loop
                .body
                .stmts
                .iter()
                .map(|s| match s {
                    syn::Stmt::Local(local) => {
                        if let syn::Pat::Type(pat_type) = &local.pat {
                            let inner_pat = pat_type.pat.as_ref().clone();
                            let init = local.init.as_ref().map(|init| syn::LocalInit {
                                eq_token: init.eq_token,
                                expr: Box::new(rewrite_expr(&init.expr)),
                                diverge: init.diverge.clone(),
                            });
                            return syn::Stmt::Local(syn::Local {
                                pat: inner_pat,
                                init,
                                ..local.clone()
                            });
                        }
                        s.clone()
                    }
                    syn::Stmt::Expr(e, semi) => syn::Stmt::Expr(rewrite_expr(e), *semi),
                    other => other.clone(),
                })
                .collect();
            syn::Expr::ForLoop(syn::ExprForLoop {
                body: syn::Block {
                    brace_token: for_loop.body.brace_token,
                    stmts: new_stmts,
                },
                ..for_loop.clone()
            })
        }
        other => other.clone(),
    }
}

fn rewrite_stmt(stmt: &syn::Stmt) -> syn::Stmt {
    match stmt {
        syn::Stmt::Local(local) => {
            // Strip type annotations from body let bindings.
            // Tile<T, { [shape] }> is DSL syntax for the macro parser only;
            // at runtime Tile is just a stub that holds shape info and doesn't
            // need rank type parameters.
            let inner_pat = if let syn::Pat::Type(pat_type) = &local.pat {
                pat_type.pat.as_ref().clone()
            } else {
                local.pat.clone()
            };
            let init = local.init.as_ref().map(|init| syn::LocalInit {
                eq_token: init.eq_token,
                expr: Box::new(rewrite_expr(&init.expr)),
                diverge: init.diverge.clone(),
            });
            return syn::Stmt::Local(syn::Local {
                pat: inner_pat,
                init,
                ..local.clone()
            });
        }
        syn::Stmt::Expr(expr, semi) => {
            let rewritten = rewrite_expr(expr);
            syn::Stmt::Expr(rewritten, *semi)
        }
        other => other.clone(),
    }
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    let decl = match frontend::parse::parse_kernel(&input) {
        Ok(value) => value,
        Err(err) => return err.to_compile_error().into(),
    };
    let kernel_hir = frontend::lower::lower_kernel_to_hir(&decl);
    let name = &input.sig.ident;
    let vis = &input.vis;
    let attrs = &input.attrs;

    let param_names: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pt) = arg {
                if let syn::Pat::Ident(ident) = pt.pat.as_ref() {
                    return Some(ident.ident.clone());
                }
            }
            None
        })
        .collect();

    let arg_exprs: Vec<_> = input
        .sig
        .inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let param_name = &param_names[i];
            if let FnArg::Typed(pt) = arg {
                if let syn::Type::Reference(r) = pt.ty.as_ref() {
                    if let syn::Type::Path(inner_path) = r.elem.as_ref() {
                        if let Some(last_seg) = inner_path.path.segments.last() {
                            if last_seg.ident == "Partition" {
                                if r.mutability.is_some() {
                                    return quote::quote! { #param_name.as_kernel_arg_mut() };
                                }
                                return quote::quote! { #param_name.as_kernel_arg() };
                            }
                        }
                    }
                    if r.mutability.is_some() {
                        return quote::quote! { #param_name.as_kernel_arg_mut() };
                    }
                }
            }
            quote::quote! { #param_name.as_kernel_arg() }
        })
        .collect();

    let params: Vec<FnArg> = input
        .sig
        .inputs
        .into_iter()
        .map(|mut arg| {
            if let FnArg::Typed(ref mut pt) = arg {
                pt.ty = Box::new(rewrite_type(&pt.ty));
                if let syn::Type::Reference(ref mut tr) = *pt.ty {
                    tr.lifetime = Some(syn::parse_quote!('kernel));
                }
            }
            arg
        })
        .collect();

    let const_generics: Vec<_> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Const(c) = p {
                let ident = &c.ident;
                let ty = &c.ty;
                Some(quote::quote! { const #ident: #ty })
            } else {
                None
            }
        })
        .collect();

    let body_stmts: Vec<_> = input
        .block
        .stmts
        .iter()
        .map(|stmt| rewrite_stmt(stmt))
        .collect();

    let expanded = quote::quote! {
        #( #attrs )*
        #vis fn #name<'kernel, #(#const_generics),*>(#(#params),*) -> ::sile::KernelLauncher<'kernel> {
            static KERNEL: std::sync::OnceLock<::sile::hir::Kernel> = std::sync::OnceLock::new();
            let kernel = KERNEL.get_or_init(|| #kernel_hir);
            #(#body_stmts)*
            ::sile::KernelLauncher::new(
                kernel,
                vec![#(#arg_exprs),*],
            )
        }
    };

    expanded.into()
}
