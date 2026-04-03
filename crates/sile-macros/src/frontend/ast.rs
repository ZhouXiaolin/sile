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
        expr: KernelExpr,
    },
    Store {
        target: syn::Ident,
        value: KernelExpr,
    },
}

#[derive(Clone, Debug)]
pub enum KernelExpr {
    Var(syn::Ident),
    Lit(syn::LitInt),
    MethodCall {
        receiver: Box<KernelExpr>,
        method: syn::Ident,
        args: Vec<KernelExpr>,
    },
    Call {
        func: syn::Ident,
        args: Vec<KernelExpr>,
    },
    BinaryOp {
        left: Box<KernelExpr>,
        op: BinOpKind,
        right: Box<KernelExpr>,
    },
    Array(syn::ExprArray),
    FieldAccess {
        target: Box<KernelExpr>,
        field: syn::Ident,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}
