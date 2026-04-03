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
    pub shape: Option<Vec<i64>>,
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
    ForLoop {
        var: syn::Ident,
        start: KernelExpr,
        end: KernelExpr,
        body: Vec<KernelStmt>,
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
        field: String,
    },
    FloatLit(syn::LitFloat),
    Index {
        target: Box<KernelExpr>,
        index: Box<KernelExpr>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}
