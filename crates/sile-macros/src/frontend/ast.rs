#[derive(Clone, Debug)]
pub struct KernelDecl {
    pub name: syn::Ident,
    pub params: Vec<KernelParam>,
    pub body: Vec<KernelStmt>,
}

#[derive(Clone, Debug)]
pub struct KernelParam {
    pub name: syn::Ident,
    pub is_mut: bool,
}

#[derive(Clone, Debug)]
pub enum KernelStmt {
    Let {
        name: syn::Ident,
        expr: syn::Expr,
    },
    Store {
        target: syn::Ident,
        value: syn::Expr,
    },
}
