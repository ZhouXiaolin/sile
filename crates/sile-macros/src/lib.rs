mod frontend;

use proc_macro::TokenStream;
use syn::FnArg;

/// Whether a scalar parameter type names an integer.
///
/// `i32` is the spelling CuTile uses and `i64` is the one this front end
/// carries, so both are integers here. Every other scalar type is a float,
/// which the parser has already accepted or rejected.
fn is_integer_type(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(path) if path.path.is_ident("i64") || path.path.is_ident("i32"))
}

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
            args: call
                .args
                .iter()
                .filter(|arg| !is_mode_argument(arg))
                .map(rewrite_expr)
                .collect(),
            ..call.clone()
        }),
        syn::Expr::Binary(binary) => syn::Expr::Binary(syn::ExprBinary {
            left: Box::new(rewrite_expr(&binary.left)),
            right: Box::new(rewrite_expr(&binary.right)),
            ..binary.clone()
        }),
        syn::Expr::Call(call) => syn::Expr::Call(syn::ExprCall {
            func: Box::new(rewrite_expr(&call.func)),
            args: call
                .args
                .iter()
                .filter(|arg| !is_mode_argument(arg))
                .map(rewrite_expr)
                .collect(),
            ..call.clone()
        }),
        syn::Expr::Assign(assign) => syn::Expr::Assign(syn::ExprAssign {
            left: Box::new(rewrite_expr(&assign.left)),
            eq_token: assign.eq_token,
            right: Box::new(rewrite_expr(&assign.right)),
            attrs: assign.attrs.clone(),
        }),
        syn::Expr::Reference(reference) => syn::Expr::Reference(syn::ExprReference {
            expr: Box::new(rewrite_expr(&reference.expr)),
            ..reference.clone()
        }),
        // `unsafe { ... }` marks a promise about the operation, and a promise is
        // a property of the program rather than of the instruction stream. The
        // host body keeps the statements and drops the marker, so a kernel that
        // marks a call unsafe does not warn about a block that checks nothing.
        syn::Expr::Unsafe(unsafe_block) => syn::Expr::Block(syn::ExprBlock {
            attrs: unsafe_block.attrs.clone(),
            label: None,
            block: syn::Block {
                brace_token: unsafe_block.block.brace_token,
                stmts: unsafe_block.block.stmts.iter().map(rewrite_stmt).collect(),
            },
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

/// Whether a call argument names a policy rather than a value.
///
/// `rounding::Zero`, `ftz::Disabled` and `None::<scope::TileBlock>` all reach
/// the parser as a qualified path, and none of them is an operand. The front end
/// drops them when it builds the HIR; the host body the macro leaves behind has
/// to drop them too, because the stub they would be passed to takes one
/// argument. The front end owns the rule, so the two cannot disagree about a
/// path such as `f32::NEG_INFINITY`, which is a value.
fn is_mode_argument(arg: &syn::Expr) -> bool {
    matches!(arg, syn::Expr::Path(path) if frontend::parse::is_mode_path(&path.path))
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
    let kernel_hir = match frontend::lower::lower_kernel_to_hir(&decl) {
        Ok(value) => value,
        Err(err) => return err.to_compile_error().into(),
    };
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

    // A scalar parameter is passed by value, so it does not become a
    // `KernelArg`. The lists stay separate all the way to the backend, which
    // binds the buffers, the float scalars and the integer scalars through
    // three channels. A float and an integer cannot share one array without a
    // lie about a width, and a kernel that takes no integer scalar passes no
    // integer array.
    let mut arg_exprs = Vec::new();
    let mut scalar_exprs = Vec::new();
    let mut scalar_i64_exprs = Vec::new();
    for (i, arg) in input.sig.inputs.iter().enumerate() {
        let param_name = &param_names[i];
        let FnArg::Typed(pt) = arg else {
            continue;
        };
        let syn::Type::Reference(r) = pt.ty.as_ref() else {
            if is_integer_type(&pt.ty) {
                scalar_i64_exprs.push(quote::quote! { #param_name });
            } else {
                scalar_exprs.push(quote::quote! { #param_name });
            }
            continue;
        };
        if let syn::Type::Path(inner_path) = r.elem.as_ref()
            && let Some(last_seg) = inner_path.path.segments.last()
            && last_seg.ident == "Partition"
        {
            if r.mutability.is_some() {
                arg_exprs.push(quote::quote! { #param_name.as_kernel_arg_mut() });
            } else {
                arg_exprs.push(quote::quote! { #param_name.as_kernel_arg() });
            }
            continue;
        }
        if r.mutability.is_some() {
            arg_exprs.push(quote::quote! { #param_name.as_kernel_arg_mut() });
        } else {
            arg_exprs.push(quote::quote! { #param_name.as_kernel_arg() });
        }
    }

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

    // Build const params for function signature.
    //
    // An array parameter `const S: [i32; 2]` becomes one integer per dimension,
    // because stable Rust has no const generic of array type. The kernel body
    // and the HIR both name the expanded parameters, so the shape stays a
    // compile-time value all the way down.
    let const_generics: Vec<_> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Const(c) = p {
                let ident = &c.ident;
                if let Some(rank) = frontend::parse::const_array_rank(&c.ty) {
                    let dims: Vec<_> = (0..rank)
                        .map(|dim| {
                            let name = syn::Ident::new(
                                &format!("{}_{}", ident, dim),
                                ident.span(),
                            );
                            quote::quote! { const #name: i64 }
                        })
                        .collect();
                    return Some(quote::quote! { #(#dims),* });
                }
                let ty = &c.ty;
                return Some(quote::quote! { const #ident: #ty });
            }
            None
        })
        .collect();

    let body_stmts: Vec<_> = input
        .block
        .stmts
        .iter()
        .map(|stmt| rewrite_stmt(stmt))
        .collect();

    // The HIR is a constant of the monomorphization, not of the function.
    //
    // A `static` inside a generic function is shared by every instantiation:
    // `OnceLock` then hands the first `TILE` to every later one, and a kernel
    // compiled at `TILE = 4` silently runs the code built for `TILE = 2`. The
    // HIR is a compile-time value here, so it is built inline. The expensive
    // part of a launch is the Tile IR -> LLVM IR -> backend pipeline, and that
    // is cached by the launcher, keyed on the HIR it actually received.
    let expanded = quote::quote! {
        #( #attrs )*
        #vis fn #name<'kernel, #(#const_generics),*>(#(#params),*) -> ::sile::KernelLauncher<'kernel> {
            // The scalars are read before the body runs, and under names the
            // body cannot shadow. A kernel may rebind a parameter to a value of
            // another type -- flash attention writes `let qk_scale: f32 =
            // qk_scale / log2` -- and reading the parameter afterwards would
            // pass the rebound value to the launcher. The buffers are read
            // after the body, because a `KernelArg` borrows its buffer and the
            // body borrows it too.
            let __sile_kernel: ::sile::hir::Kernel = #kernel_hir;
            let __sile_scalars = vec![#(#scalar_exprs),*];
            let __sile_scalars_i64 = vec![#(#scalar_i64_exprs),*];
            #(#body_stmts)*
            ::sile::KernelLauncher::new(
                __sile_kernel,
                vec![#(#arg_exprs),*],
                __sile_scalars,
                __sile_scalars_i64,
            )
        }
    };

    expanded.into()
}
