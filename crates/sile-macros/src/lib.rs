mod frontend;

use proc_macro::TokenStream;
use syn::FnArg;

fn rewrite_shape_expr(expr: &syn::Expr) -> proc_macro2::TokenStream {
    if let syn::Expr::Array(arr) = expr {
        let dims: Vec<_> = arr.elems.iter().collect();
        if dims.is_empty() {
            return quote::quote! { ::sile::DListNil };
        }
        let mut result = quote::quote! { ::sile::DListNil };
        for dim in dims.iter().rev() {
            result = quote::quote! { ::sile::DList<{ #dim }, #result> };
        }
        result
    } else {
        quote::quote! { ::sile::DListNil }
    }
}

fn rewrite_type(ty: &syn::Type) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            let path = &type_path.path;
            if let Some(last_seg) = path.segments.last() {
                if last_seg.ident == "Tensor" || last_seg.ident == "Tile" {
                    if let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments {
                        let type_args: Vec<_> = args.args.iter().collect();
                        if type_args.len() >= 2 {
                            if let syn::GenericArgument::Type(elem_type) = type_args[0] {
                                let rank_type_tokens = match type_args[1] {
                                    syn::GenericArgument::Const(const_expr) => {
                                        rewrite_shape_expr(const_expr)
                                    }
                                    syn::GenericArgument::Type(ty) => {
                                        quote::quote! { #ty }
                                    }
                                    _ => quote::quote! { ::sile::DListNil },
                                };
                                let rank_type: syn::Type = syn::parse2(rank_type_tokens).unwrap();
                                let mut new_path = path.clone();
                                if let Some(last_seg) = new_path.segments.last_mut() {
                                    let mut args: syn::punctuated::Punctuated<
                                        syn::GenericArgument,
                                        syn::Token![,],
                                    > = syn::punctuated::Punctuated::new();
                                    args.push(syn::GenericArgument::Type(elem_type.clone()));
                                    args.push(syn::GenericArgument::Type(rank_type));
                                    last_seg.arguments = syn::PathArguments::AngleBracketed(
                                        syn::AngleBracketedGenericArguments {
                                            colon2_token: None,
                                            lt_token: syn::Token![<](proc_macro2::Span::call_site()),
                                            args,
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

fn rewrite_stmt(stmt: &syn::Stmt) -> syn::Stmt {
    match stmt {
        syn::Stmt::Local(local) => {
            if let syn::Pat::Type(pat_type) = &local.pat {
                let new_ty = rewrite_type(&pat_type.ty);
                let inner_pat = pat_type.pat.as_ref().clone();
                let new_pat = syn::Pat::Type(syn::PatType {
                    ty: Box::new(new_ty),
                    pat: Box::new(inner_pat),
                    ..pat_type.clone()
                });
                return syn::Stmt::Local(syn::Local {
                    pat: new_pat,
                    ..local.clone()
                });
            }
            stmt.clone()
        }
        _ => stmt.clone(),
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
                    // Check if inner type is Partition
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

    // Extract const generics from the original signature
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
