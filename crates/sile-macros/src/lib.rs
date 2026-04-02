use proc_macro::TokenStream;
use syn::{parse_quote, FnArg, Type};

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    let name = &input.sig.ident;
    let vis = &input.vis;
    let attrs = &input.attrs;

    // Rewrite parameters: add 'kernel lifetime to each reference type
    let params: Vec<FnArg> = input
        .sig
        .inputs
        .into_iter()
        .map(|mut arg| {
            if let FnArg::Typed(ref mut pt) = arg {
                if let Type::Reference(ref mut tr) = *pt.ty {
                    tr.lifetime = Some(syn::parse_quote!('kernel));
                }
            }
            arg
        })
        .collect();

    let expanded = quote::quote! {
        #( #attrs )*
        #vis fn #name<'kernel>(#(#params),*) -> ::sile::KernelLauncher<'kernel> {
            static SPEC: ::sile::KernelSpecRef = ::sile::KernelSpecRef {
                name: stringify!(#name),
                params: &[
                    ::sile::ParamRef::input_f32(0, &[16]),
                    ::sile::ParamRef::input_f32(1, &[16]),
                    ::sile::ParamRef::output_f32(2, &[16]),
                ],
                nodes: &[
                    ::sile::NodeRef::load_tile(0, ::sile::TileExpr::grid_x(), &[4]),
                    ::sile::NodeRef::load_tile(1, ::sile::TileExpr::grid_x(), &[4]),
                    ::sile::NodeRef::binary(::sile::BinaryOp::Add, 0, 1, &[4]),
                ],
                stores: &[
                    ::sile::StoreRef::new(2, ::sile::TileExpr::grid_x(), 2),
                ],
            };

            ::sile::KernelLauncher::new(
                &SPEC,
                vec![a.as_kernel_arg(), b.as_kernel_arg(), c.as_kernel_arg_mut()],
            )
        }
    };

    expanded.into()
}
