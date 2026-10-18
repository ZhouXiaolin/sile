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
            block: rewrite_block(&unsafe_block.block),
        }),
        // A branch is a block, and a block holds bindings: an annotation inside
        // one is DSL syntax, the same as an annotation at the top of the body.
        syn::Expr::If(if_expr) => syn::Expr::If(syn::ExprIf {
            cond: Box::new(rewrite_expr(&if_expr.cond)),
            then_branch: rewrite_block(&if_expr.then_branch),
            else_branch: if_expr
                .else_branch
                .as_ref()
                .map(|(token, expr)| (*token, Box::new(rewrite_expr(expr)))),
            ..if_expr.clone()
        }),
        syn::Expr::Block(block) => syn::Expr::Block(syn::ExprBlock {
            block: rewrite_block(&block.block),
            ..block.clone()
        }),
        syn::Expr::Array(arr) => syn::Expr::Array(syn::ExprArray {
            elems: arr.elems.iter().map(rewrite_expr).collect(),
            ..arr.clone()
        }),
        syn::Expr::Path(_) | syn::Expr::Lit(_) | syn::Expr::Range(_) => expr.clone(),
        syn::Expr::ForLoop(for_loop) => {
            // Rewrite statements inside for loop body
            syn::Expr::ForLoop(syn::ExprForLoop {
                expr: Box::new(rewrite_expr(&for_loop.expr)),
                body: rewrite_block(&for_loop.body),
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

/// A block whose statements the host body can compile.
///
/// Every statement goes through [`rewrite_stmt`], which is what drops the DSL
/// type annotations and rewrites a shape access. A block nested in a branch or
/// a loop needs the same treatment as the body of the kernel.
fn rewrite_block(block: &syn::Block) -> syn::Block {
    syn::Block {
        brace_token: block.brace_token,
        stmts: block.stmts.iter().map(rewrite_stmt).collect(),
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
            // A `while` or a `loop` in the host body runs its statements once.
            // The loop's result is device work the host body never reads, and
            // an unbounded loop would never return, so the iteration is what
            // the stub keeps and the condition and the back edge are not.
            if let syn::Expr::While(while_expr) = &*expr {
                let body = rewrite_block(&while_expr.body);
                return syn::Stmt::Expr(
                    syn::Expr::Block(syn::ExprBlock {
                        attrs: Vec::new(),
                        label: None,
                        block: body,
                    }),
                    *semi,
                );
            }
            if let syn::Expr::Loop(loop_expr) = &*expr {
                let body = rewrite_block(&loop_expr.body);
                return syn::Stmt::Expr(
                    syn::Expr::Block(syn::ExprBlock {
                        attrs: Vec::new(),
                        label: None,
                        block: body,
                    }),
                    *semi,
                );
            }
            // A `break`, a `continue`, or a `return` ends a pass the host
            // body does not run. The return cannot stay either: this function
            // hands back a launcher, so a bare `return` would skip it.
            if matches!(
                &*expr,
                syn::Expr::Break(_)
                    | syn::Expr::Continue(_)
                    | syn::Expr::Return(syn::ExprReturn { expr: None, .. })
            ) {
                return syn::Stmt::Expr(
                    syn::Expr::Tuple(syn::ExprTuple {
                        attrs: Vec::new(),
                        elems: syn::punctuated::Punctuated::new(),
                        paren_token: Default::default(),
                    }),
                    *semi,
                );
            }
            let rewritten = rewrite_expr(expr);
            syn::Stmt::Expr(rewritten, *semi)
        }
        other => other.clone(),
    }
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    expand_kernel_fn(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// The entry keys a `#[sile::entry(...)]` accepts. They all name a compiler
/// behavior of the CUDA front end — diagnostics, proofs, tuning — and none of
/// them changes the program this front end builds, so each one is accepted by
/// name and dropped. A key that is not on the list is a typo, and a typo must
/// fail at compile time rather than pass silently.
const ENTRY_KEYS: &[&str] = &[
    "print_ir",
    "dump_mlir_dir",
    "unchecked_accesses",
    "deny_in_kernel_checks",
    "preconditions",
    "optimization_hints",
];

/// Accepts every key on [`ENTRY_KEYS`] and rejects every other one. The value
/// a key carries is not read: the behaviors they name live in the CUDA
/// compiler's pass pipeline, and this front end names its own through the
/// `SILE_PRINT_*` environment variables.
fn validate_entry_keys(attr: &syn::Attribute) -> syn::Result<()> {
    let syn::Meta::List(list) = &attr.meta else {
        return Ok(()); // `#[entry]` with no keys has nothing to check.
    };
    let keys = list.parse_args_with(
        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
    )?;
    for meta in keys {
        let key = meta
            .path()
            .get_ident()
            .map(syn::Ident::to_string)
            .unwrap_or_default();
        if !ENTRY_KEYS.contains(&key.as_str()) {
            return Err(syn::Error::new_spanned(
                &meta,
                format!(
                    "`{key}` is not an entry attribute; the keys are: {}",
                    ENTRY_KEYS.join(", ")
                ),
            ));
        }
    }
    Ok(())
}

fn expand_kernel_fn(input: &syn::ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let decl = frontend::parse::parse_kernel(input)?;
    let kernel_hir = frontend::lower::lower_kernel_to_hir(&decl)?;
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
        .iter()
        .cloned()
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
                            let name = syn::Ident::new(&format!("{}_{}", ident, dim), ident.span());
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
    // The host body is a stub: loops run once, a return hands back nothing.
    // The warnings that stub would raise are noise the device program does
    // not have, so the expansion silences exactly those two.
    let expanded = quote::quote! {
        #( #attrs )*
        #[allow(unused_assignments, unused_variables)]
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

    Ok(expanded)
}

#[proc_macro_attribute]
pub fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemMod);
    expand_module(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Expands a `#[sile::module]`: the entry functions become launchers, exactly
/// what `#[sile::kernel]` builds, and every device function stays a plain
/// function whose calls the entries inlined at their call sites.
fn expand_module(input: &syn::ItemMod) -> syn::Result<proc_macro2::TokenStream> {
    use std::collections::HashMap;

    let content = input
        .content
        .as_ref()
        .map(|(_, items)| items)
        .expect("a module macro on a mod with a body");
    let mut entries: Vec<syn::ItemFn> = Vec::new();
    let mut devices: HashMap<String, frontend::inline::DeviceFn> = HashMap::new();
    let mut kept: Vec<syn::Item> = Vec::new();
    // The attribute the surface spells `#[sile::entry]` reaches this macro
    // with the crate name as its first segment, so the last segment is what
    // names it.
    let is_entry_attr = |attr: &syn::Attribute| {
        attr.path()
            .segments
            .last()
            .map(|segment| segment.ident == "entry")
            .unwrap_or(false)
    };
    for item in content {
        match item {
            syn::Item::Fn(f) => {
                let entry_attr = f.attrs.iter().find(|attr| is_entry_attr(attr)).cloned();
                if let Some(attr) = entry_attr {
                    validate_entry_keys(&attr)?;
                    let mut f = f.clone();
                    f.attrs.retain(|attr| !is_entry_attr(attr));
                    entries.push(f);
                } else {
                    let params: Vec<String> = f
                        .sig
                        .inputs
                        .iter()
                        .filter_map(|arg| match arg {
                            syn::FnArg::Typed(pt) => match pt.pat.as_ref() {
                                syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
                                _ => None,
                            },
                            _ => None,
                        })
                        .collect();
                    devices.insert(
                        f.sig.ident.to_string(),
                        frontend::inline::DeviceFn {
                            params,
                            returns: !matches!(&f.sig.output, syn::ReturnType::Default),
                            body: f.block.stmts.clone(),
                        },
                    );
                    // The device function stays for the host: its signature
                    // keeps a plain Tile and its body keeps the stub calls,
                    // the same treatment an entry's body gets.
                    let mut f = f.clone();
                    // The calls this function would serve were inlined into
                    // the entries, so the host copy has no caller: it stays
                    // for the host build to type-check, and that is dead code
                    // by design.
                    f.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
                    f.sig.inputs = f
                        .sig
                        .inputs
                        .into_iter()
                        .map(|mut arg| {
                            if let syn::FnArg::Typed(ref mut pt) = arg {
                                pt.ty = Box::new(rewrite_type(&pt.ty));
                            }
                            arg
                        })
                        .collect();
                    f.block.stmts = f
                        .block
                        .stmts
                        .iter()
                        .map(rewrite_stmt)
                        .collect();
                    kept.push(syn::Item::Fn(f));
                }
            }
            other => kept.push(other.clone()),
        }
    }

    let mut names: Vec<String> = devices.keys().cloned().collect();
    names.sort();
    let mut expanded_entries = Vec::new();
    for entry in &entries {
        let mut stack = Vec::new();
        let mut fresh = 0usize;
        let stmts = frontend::inline::inline_device_calls(
            &entry.block.stmts,
            &devices,
            &mut stack,
            &mut fresh,
        )?;
        let flat = syn::ItemFn {
            attrs: Vec::new(),
            vis: entry.vis.clone(),
            sig: entry.sig.clone(),
            block: Box::new(syn::Block {
                brace_token: entry.block.brace_token,
                stmts,
            }),
        };
        expanded_entries.push(expand_kernel_fn(&flat)?);
    }

    let vis = &input.vis;
    let mod_token = input.mod_token;
    let ident = &input.ident;
    Ok(quote::quote! {
        #vis #mod_token #ident {
            #(#kept)*
            #(#expanded_entries)*
        }
    })
}
