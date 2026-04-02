mod frontend;

use proc_macro::TokenStream;
use syn::FnArg;

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

    // Collect parameter names for building kernel arg list
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

    // Build kernel arg expressions
    let arg_exprs: Vec<_> = input
        .sig
        .inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let param_name = &param_names[i];
            if let FnArg::Typed(pt) = arg {
                if let syn::Type::Reference(r) = pt.ty.as_ref() {
                    if r.mutability.is_some() {
                        return quote::quote! { #param_name.as_kernel_arg_mut() };
                    }
                }
            }
            quote::quote! { #param_name.as_kernel_arg() }
        })
        .collect();

    // Rewrite parameters: add 'kernel lifetime to each reference type
    let params: Vec<FnArg> = input
        .sig
        .inputs
        .into_iter()
        .map(|mut arg| {
            if let FnArg::Typed(ref mut pt) = arg {
                if let syn::Type::Reference(ref mut tr) = *pt.ty {
                    tr.lifetime = Some(syn::parse_quote!('kernel));
                }
            }
            arg
        })
        .collect();

    let expanded = quote::quote! {
        #( #attrs )*
        #vis fn #name<'kernel>(#(#params),*) -> ::sile::KernelLauncher<'kernel> {
            static KERNEL: std::sync::OnceLock<::sile::hir::Kernel> = std::sync::OnceLock::new();
            let kernel = KERNEL.get_or_init(|| #kernel_hir);
            ::sile::KernelLauncher::new(
                kernel,
                vec![#(#arg_exprs),*],
            )
        }
    };

    expanded.into()
}
