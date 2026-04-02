use quote::quote;

use super::ast::{KernelDecl, KernelStmt};

pub fn lower_kernel_to_hir(decl: &KernelDecl) -> proc_macro2::TokenStream {
    let name = decl.name.to_string();
    let params = decl.params.iter().map(|param| {
        let name = param.name.to_string();
        let kind = if param.is_mut {
            quote! { ::sile::hir::ParamKind::Output }
        } else {
            quote! { ::sile::hir::ParamKind::Input }
        };
        quote! {
            ::sile::hir::Param::new(
                #name,
                #kind,
                ::sile::hir::Type::tensor(
                    ::sile::hir::ElemType::F32,
                    ::sile::hir::ShapeExpr::dynamic(),
                ),
            )
        }
    });
    let body = decl.body.iter().map(|stmt| match stmt {
        KernelStmt::Let { name, .. } => {
            let name = name.to_string();
            quote! {
                ::sile::hir::Stmt::Let {
                    name: #name.to_string(),
                    ty: None,
                    expr: ::sile::hir::Expr::builtin(::sile::hir::BuiltinOp::ProgramId),
                }
            }
        }
        KernelStmt::Store { target, .. } => {
            let target = target.to_string();
            quote! {
                ::sile::hir::Stmt::Store {
                    target: #target.to_string(),
                    value: ::sile::hir::Expr::Var("tmp".to_string()),
                }
            }
        }
    });
    quote! {
        ::sile::hir::Kernel::new(
            #name,
            vec![],
            vec![#(#params),*],
            vec![#(#body),*],
        )
    }
}
