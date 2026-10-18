#[derive(Clone, Debug)]
pub struct KernelDecl {
    pub name: syn::Ident,
    pub const_params: Vec<ConstParam>,
    pub params: Vec<KernelParam>,
    pub body: Vec<KernelStmt>,
}

/// A const generic parameter.
///
/// CuTile writes a shape as `const S: [i32; 2]` and then names it in a type:
/// `Tensor<f32, S>`. Rust has no const generic of array type on stable, so the
/// macro expands one such parameter into one integer parameter per dimension,
/// `S_0` and `S_1`. The kernel source keeps the CuTile spelling; the call site
/// passes the dimensions, which is the same information the declaration names.
#[derive(Clone, Debug)]
pub struct ConstParam {
    pub ident: syn::Ident,
    /// `Some(rank)` for an array parameter, `None` for a plain integer one.
    pub array_rank: Option<usize>,
}

impl ConstParam {
    /// The names the emitted function declares, one per dimension.
    pub fn expanded_names(&self) -> Vec<syn::Ident> {
        match self.array_rank {
            None => vec![self.ident.clone()],
            Some(rank) => (0..rank)
                .map(|dim| syn::Ident::new(&format!("{}_{}", self.ident, dim), self.ident.span()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelParam {
    pub name: syn::Ident,
    pub is_mut: bool,
    pub shape: Option<Vec<KernelShapeDim>>,
    /// The element type of a scalar parameter, written `a: f32`.
    ///
    /// CuTile passes a scalar by value next to the buffers, so it carries no
    /// shape and no pointer. The kernel reads it as a value. A buffer parameter
    /// has `None` here: its element type is the type of the buffer.
    pub scalar_ty: Option<ScalarType>,
}

/// The type of a scalar parameter, as the kernel writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    I64,
}

#[derive(Clone, Debug)]
pub enum KernelShapeDim {
    Dynamic,
    Constant(i64),
    Symbol(syn::Ident),
}

#[derive(Clone, Debug)]
pub enum KernelStmt {
    Let {
        name: syn::Ident,
        expr: KernelExpr,
    },
    Assign {
        name: syn::Ident,
        expr: KernelExpr,
    },
    Store {
        target: syn::Ident,
        value: KernelExpr,
        /// `part.store(t, [i, j])` names coordinates; `y.store(t)` does not.
        index: Option<KernelExpr>,
    },
    AtomicAdd {
        target: syn::Ident,
        index: KernelExpr,
        value: KernelExpr,
    },
    ForLoop {
        var: syn::Ident,
        start: KernelExpr,
        end: KernelExpr,
        /// `(a..b).step_by(k)` names the step; a plain range steps by one.
        step: Option<KernelExpr>,
        body: Vec<KernelStmt>,
    },
    /// `while cond { .. }`.
    While {
        cond: KernelExpr,
        body: Vec<KernelStmt>,
    },
    /// `loop { .. }`: a while whose condition is always true, which is to say
    /// a loop that only a `break` leaves.
    Loop {
        body: Vec<KernelStmt>,
    },
    /// `break`: leave the enclosing `while` or `loop`.
    Break,
    /// `return;`: leave the kernel. A kernel returns nothing, so the form
    /// carries no value.
    Return,
    /// `if cond { .. } else { .. }`.
    ///
    /// `else if` reaches here as an `if` statement in the else body, so one
    /// variant carries every chain.
    If {
        cond: KernelExpr,
        then_body: Vec<KernelStmt>,
        else_body: Vec<KernelStmt>,
    },
    Continue,
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
    /// `-expr`: the negation CuTile writes on a tile or a scalar.
    Neg(Box<KernelExpr>),
    Array(syn::ExprArray),
    FieldAccess {
        target: Box<KernelExpr>,
        field: String,
    },
    FloatLit(syn::LitFloat),
    /// A float constant written as a path: `f32::NEG_INFINITY`.
    ///
    /// Rust spells an infinity as an associated constant, not as a literal, so
    /// the parser reads the path and carries the value.
    FloatConst(f32),
    Index {
        target: Box<KernelExpr>,
        index: Box<KernelExpr>,
    },
    /// `shape![..]` or `const_shape![..]`: a compile-time shape.
    ///
    /// CuTile writes a shape as a macro call because the rank is part of the
    /// type. Here the rank is carried by the argument list, so the macro is
    /// syntax and lowers to the same shape expression an array literal does.
    ShapeMacro(syn::ExprMacro),
    /// A qualified path in argument position: `rounding::NegativeInf`,
    /// `ftz::Disabled`, `None::<scope::TileBlock>`.
    ///
    /// None of these is an operand. A rounding mode and a flush-to-zero mode
    /// name a policy, and this IR carries one policy per operation, which is the
    /// target's default. A scope and a `None` name a parameter this IR does not
    /// have. So the path is accepted and dropped, which is what lets a CuTile
    /// kernel that spells its modes port unchanged.
    ModePath(syn::Path),
    /// `true` or `false`, as `constant(true, shape)` writes it.
    ///
    /// This IR has one element type, so a boolean is a mask value: `1.0` where
    /// it holds and `0.0` where it does not. That is the encoding `gt_tile`
    /// produces and `select` tests, so a boolean constant needs no second type.
    BoolLit(syn::LitBool),
    /// `|a, b| <one expression>`: the combiner a `scan` folds with.
    ///
    /// Only this shape parses. The two names are the running value and the
    /// next element, and the body is the one expression that joins them; a
    /// statement body would be a loop the combiner cannot name.
    Closure {
        params: Vec<syn::Ident>,
        body: Box<KernelExpr>,
    },
}

impl KernelExpr {
    /// Whether this expression names a policy rather than a value.
    ///
    /// A mode, a scope and a `None` all reach the parser as a qualified path,
    /// and none of them is an operand. Call lowering drops them, and the host
    /// body the macro leaves behind drops them too, so one rule serves both.
    pub fn is_mode(&self) -> bool {
        matches!(self, Self::ModePath(_))
    }

    /// The span a diagnostic about this expression should point at.
    ///
    /// A parsed expression keeps the token it came from, so an error about an
    /// unsupported construct can point at that construct instead of at the
    /// whole kernel.
    pub fn span(&self) -> proc_macro2::Span {
        use syn::spanned::Spanned;
        match self {
            Self::Var(ident) => ident.span(),
            Self::Lit(lit) => lit.span(),
            Self::FloatLit(lit) => lit.span(),
            Self::FloatConst(_) => proc_macro2::Span::call_site(),
            Self::Array(arr) => arr.span(),
            Self::MethodCall { method, .. } => method.span(),
            Self::Call { func, .. } => func.span(),
            Self::BinaryOp { left, .. } => left.span(),
            Self::Neg(inner) => inner.span(),
            Self::FieldAccess { target, .. } => target.span(),
            Self::Index { target, .. } => target.span(),
            Self::ShapeMacro(mac) => mac.span(),
            Self::ModePath(path) => path.span(),
            Self::BoolLit(lit) => lit.span(),
            Self::Closure { params, .. } => params
                .first()
                .map(syn::Ident::span)
                .unwrap_or_else(proc_macro2::Span::call_site),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `==`, `!=`, `<`, `<=`, `>`, `>=`: an elementwise comparison, whose value
    /// is `1.0` where it holds and `0.0` where it does not.
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `&` and `&&`. The two differ only in short circuiting, and an operand
    /// here has no effect to short circuit, so both are one operation.
    And,
    /// `|` and `||`. The same.
    Or,
}
