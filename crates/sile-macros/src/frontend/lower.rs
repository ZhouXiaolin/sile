use quote::quote;
use syn::spanned::Spanned;

use super::ast::{BinOpKind, ConstParam, KernelDecl, KernelExpr, KernelShapeDim, KernelStmt, ScalarType};

/// Lower the parsed kernel to the HIR constructor.
///
/// Every error here is a construct the front end cannot translate. Returning
/// `Ok` for one would emit a `Var` that the HIR has no binding for, and the
/// tile lowering would then read it as the constant zero: a kernel that
/// compiles, runs, and computes the wrong answer. A rejected kernel is the
/// cheaper failure, so an unknown call, method, field or literal is an error
/// with the span of the offending token.
pub fn lower_kernel_to_hir(decl: &KernelDecl) -> syn::Result<proc_macro2::TokenStream> {
    let name = decl.name.to_string();
    let params = decl
        .params
        .iter()
        .map(|param| lower_param(param, &decl.const_params))
        .collect::<syn::Result<Vec<_>>>()?;
    let body = decl
        .body
        .iter()
        .map(|stmt| lower_stmt(stmt, &decl.const_params))
        .collect::<syn::Result<Vec<_>>>()?;
    let const_params_exprs: Vec<_> = decl
        .const_params
        .iter()
        .flat_map(|param| {
            param.expanded_names().into_iter().map(|ident| {
                let name = ident.to_string();
                quote! { (#name.to_string(), #ident as i64) }
            })
        })
        .collect();
    Ok(quote! {
        ::sile::hir::Kernel::new(
            #name,
            vec![#(#const_params_exprs),*],
            vec![#(#params),*],
            vec![#(#body),*],
        )
    })
}

fn lower_param(
    param: &super::ast::KernelParam,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    let name = param.name.to_string();
    if let Some(scalar_ty) = param.scalar_ty {
        let elem = match scalar_ty {
            ScalarType::F32 => quote! { ::sile::hir::ElemType::F32 },
            ScalarType::I64 => quote! { ::sile::hir::ElemType::I64 },
        };
        return Ok(quote! {
            ::sile::hir::Param::new(
                #name,
                ::sile::hir::ParamKind::Input,
                ::sile::hir::Type::Scalar(#elem),
            )
        });
    }
    let kind = if param.is_mut {
        quote! { ::sile::hir::ParamKind::Output }
    } else {
        quote! { ::sile::hir::ParamKind::Input }
    };
    let shape = match &param.shape {
        Some(dims) => {
            let mut dim_exprs: Vec<proc_macro2::TokenStream> = Vec::new();
            for dim in dims {
                match dim {
                    KernelShapeDim::Dynamic => {
                        dim_exprs.push(quote! { ::sile::hir::ShapeExpr::dynamic() });
                    }
                    KernelShapeDim::Constant(value) => {
                        dim_exprs.push(quote! { ::sile::hir::ShapeExpr::constant(#value as i64) });
                    }
                    KernelShapeDim::Symbol(ident) => {
                        // A symbol in a shape must be a const parameter. An
                        // unknown name would lower to `-1` (dynamic) and the
                        // tile would read the wrong extent, so it is an error
                        // with the span of the name.
                        let Some(param) = const_params.iter().find(|p| p.ident == *ident) else {
                            return Err(syn::Error::new(
                                ident.span(),
                                format!(
                                    "`{ident}` is not a const parameter; a shape dimension is an integer or a const generic"
                                ),
                            ));
                        };
                        // `const S: [i32; 2]` names a whole shape, so it
                        // contributes one dimension per rank. `const N: i64`
                        // contributes one.
                        for name in param.expanded_names() {
                            let name = name.to_string();
                            dim_exprs.push(quote! { ::sile::hir::ShapeExpr::symbol(#name) });
                        }
                    }
                }
            }
            quote! {
                ::sile::hir::ShapeExpr::tuple(vec![#(#dim_exprs),*])
            }
        }
        None => quote! { ::sile::hir::ShapeExpr::dynamic() },
    };
    Ok(quote! {
        ::sile::hir::Param::new(
            #name,
            #kind,
            ::sile::hir::Type::tensor(
                ::sile::hir::ElemType::F32,
                #shape,
            ),
        )
    })
}

fn lower_stmt(
    stmt: &KernelStmt,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    Ok(match stmt {
        KernelStmt::Let { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr, const_params)?;
            quote! {
                ::sile::hir::Stmt::Let {
                    name: #name.to_string(),
                    ty: None,
                    expr: #expr,
                }
            }
        }
        KernelStmt::Assign { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr, const_params)?;
            quote! {
                ::sile::hir::Stmt::Assign {
                    name: #name.to_string(),
                    expr: #expr,
                }
            }
        }
        KernelStmt::Store {
            target,
            value,
            index,
        } => {
            let target = target.to_string();
            let value = lower_expr(value, const_params)?;
            let index = index
                .as_ref()
                .map(|index| lower_expr(index, const_params))
                .transpose()?
                .map(|index| quote! { Some(#index) })
                .unwrap_or_else(|| quote! { None });
            quote! {
                ::sile::hir::Stmt::Store {
                    target: #target.to_string(),
                    value: #value,
                    index: #index,
                }
            }
        }
        KernelStmt::AtomicAdd {
            target,
            index,
            value,
        } => {
            let target = target.to_string();
            let index = lower_expr(index, const_params)?;
            let value = lower_expr(value, const_params)?;
            quote! {
                ::sile::hir::Stmt::AtomicAdd {
                    target: #target.to_string(),
                    index: #index,
                    value: #value,
                }
            }
        }
        KernelStmt::ForLoop {
            var,
            start,
            end,
            body,
        } => {
            let var_name = var.to_string();
            let start_val = lower_expr(start, const_params)?;
            let end_val = lower_expr(end, const_params)?;
            let body_stmts = lower_stmts(body, const_params)?;
            quote! {
                ::sile::hir::Stmt::ForLoop {
                    var: #var_name.to_string(),
                    start: #start_val,
                    end: #end_val,
                    body: vec![#(#body_stmts),*],
                }
            }
        }
        KernelStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            let cond = lower_expr(cond, const_params)?;
            let then_body = lower_stmts(then_body, const_params)?;
            let else_body = lower_stmts(else_body, const_params)?;
            quote! {
                ::sile::hir::Stmt::If {
                    cond: #cond,
                    then_body: vec![#(#then_body),*],
                    else_body: vec![#(#else_body),*],
                }
            }
        }
        KernelStmt::Continue => quote! { ::sile::hir::Stmt::Continue },
    })
}

fn lower_stmts(
    stmts: &[KernelStmt],
    const_params: &[ConstParam],
) -> syn::Result<Vec<proc_macro2::TokenStream>> {
    stmts
        .iter()
        .map(|stmt| lower_stmt(stmt, const_params))
        .collect()
}

/// Free functions the front end knows how to translate.
///
/// The list is exhaustive on purpose: a name that is not here is a typo or an
/// unsupported operation, and both must fail at compile time rather than become
/// an unbound variable.
const BUILTIN_FUNCTIONS: &[&str] = &[
    "load_like",
    "load_tile_mut",
    "constant",
    "mma",
    "max_tile",
    "min_tile",
    "max",
    "min",
    "ceil_div",
    "iota",
    "reduce_max",
    "reduce_sum",
    "reduce_min",
    "transpose",
    "exp",
    "exp2",
    "log",
    "log2",
    "sqrt",
    "rsqrt",
    "select",
    "gt_tile",
    "ge_tile",
    "lt_tile",
    "le_tile",
    "eq_tile",
    "ne_tile",
    "true_div",
    "tile_to_scalar",
    "scalar_to_tile",
    "convert_scalar",
    "broadcast_scalar",
    "program_id",
    "get_tile_block_id",
    "num_programs",
    "num_tiles",
];

const BUILTIN_METHODS: &[&str] = &[
    "load_tile",
    "load",
    "load_like",
    "load_tile_mut",
    "partition",
    "partition_mut",
    "reshape",
    "broadcast",
    "reduce_max",
    "reduce_sum",
    "reduce_min",
    "exp",
    "sqrt",
    "rsqrt",
    "exp2",
    "shape",
    "store",
    "atomic_add",
    "clone",
];

fn lower_expr(
    expr: &KernelExpr,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    Ok(match expr {
        KernelExpr::Var(ident) => {
            let name = ident.to_string();
            quote! { ::sile::hir::Expr::Var(#name.to_string()) }
        }
        KernelExpr::Lit(lit) => {
            let val: i64 = lit.base10_parse().map_err(|_| {
                syn::Error::new(
                    lit.span(),
                    "integer literal in a kernel body must fit in i64",
                )
            })?;
            quote! { ::sile::hir::Expr::ScalarI64(#val) }
        }
        KernelExpr::FloatLit(lit) => {
            let val: f32 = lit.base10_parse().map_err(|_| {
                syn::Error::new(lit.span(), "float literal does not fit in f32")
            })?;
            quote! { ::sile::hir::Expr::ScalarF32(#val) }
        }
        // A float constant is emitted by name rather than as a token: the
        // values a kernel writes (`f32::NEG_INFINITY` and its neighbours) have
        // no literal spelling in Rust, and the names round-trip exactly.
        KernelExpr::FloatConst(value) => {
            let token = if value.is_nan() {
                quote! { ::core::primitive::f32::NAN }
            } else if *value == f32::NEG_INFINITY {
                quote! { ::core::primitive::f32::NEG_INFINITY }
            } else if *value == f32::INFINITY {
                quote! { ::core::primitive::f32::INFINITY }
            } else if *value == f32::EPSILON {
                quote! { ::core::primitive::f32::EPSILON }
            } else if *value == f32::MIN {
                quote! { ::core::primitive::f32::MIN }
            } else if *value == f32::MAX {
                quote! { ::core::primitive::f32::MAX }
            } else {
                return Err(syn::Error::new(
                    expr.span(),
                    "unsupported float constant in a kernel body",
                ));
            };
            quote! { ::sile::hir::Expr::ScalarF32(#token) }
        }
        // A boolean is a mask value, which is the one element type this IR has.
        // `constant(true, shape)` is therefore a tile of `1.0`, the same value
        // `gt_tile` produces where its comparison holds.
        KernelExpr::BoolLit(lit) => {
            let val: f32 = if lit.value { 1.0 } else { 0.0 };
            quote! { ::sile::hir::Expr::ScalarF32(#val) }
        }
        KernelExpr::Call { func, args } => lower_call(func, args, const_params)?,
        KernelExpr::MethodCall {
            receiver,
            method,
            args,
        } => lower_method_call(receiver, method, args, const_params)?,
        KernelExpr::BinaryOp { left, op, right } => {
            let left_expr = lower_expr(left, const_params)?;
            let right_expr = lower_expr(right, const_params)?;
            let hir_op = match op {
                BinOpKind::Add => quote! { ::sile::hir::BuiltinOp::Add },
                BinOpKind::Sub => quote! { ::sile::hir::BuiltinOp::Sub },
                BinOpKind::Mul => quote! { ::sile::hir::BuiltinOp::Mul },
                BinOpKind::Div => quote! { ::sile::hir::BuiltinOp::Div },
                BinOpKind::Rem => quote! { ::sile::hir::BuiltinOp::Rem },
                BinOpKind::Eq => quote! { ::sile::hir::BuiltinOp::Eq },
                BinOpKind::Ne => quote! { ::sile::hir::BuiltinOp::Ne },
                BinOpKind::Lt => quote! { ::sile::hir::BuiltinOp::Lt },
                BinOpKind::Le => quote! { ::sile::hir::BuiltinOp::Le },
                BinOpKind::Gt => quote! { ::sile::hir::BuiltinOp::Gt },
                BinOpKind::Ge => quote! { ::sile::hir::BuiltinOp::Ge },
                BinOpKind::And => quote! { ::sile::hir::BuiltinOp::And },
                BinOpKind::Or => quote! { ::sile::hir::BuiltinOp::Or },
            };
            quote! {
                ::sile::hir::Expr::Builtin {
                    op: #hir_op,
                    args: vec![#left_expr, #right_expr],
                }
            }
        }
        KernelExpr::Neg(inner) => {
            let inner_expr = lower_expr(inner, const_params)?;
            quote! {
                ::sile::hir::Expr::Builtin {
                    op: ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Neg),
                    args: vec![#inner_expr],
                }
            }
        }
        KernelExpr::Array(arr) => lower_array(arr, const_params)?,
        KernelExpr::ShapeMacro(mac) => lower_shape_macro(mac, const_params)?,
        KernelExpr::FieldAccess { target, field } => {
            lower_field_access(target, field, const_params)?
        }
        KernelExpr::Index { target, index } => lower_index(target, index, const_params)?,
        // A mode names a policy, so it belongs in argument position and nowhere
        // else. Reaching here means it was written where a value belongs.
        KernelExpr::ModePath(path) => {
            return Err(syn::Error::new_spanned(
                path,
                "a mode or scope is not a value; it may only be passed as a call argument",
            ));
        }
    })
}

/// The operands of a call, in order.
///
/// A mode, a scope or a `None` reaches the parser as a qualified path and is
/// not an operand: this IR carries one policy per operation, and no scope
/// parameter at all. Dropping them here is what keeps `sqrt(x, rounding::Zero,
/// ftz::Disabled)` the same operation as `sqrt(x)`.
fn call_operands<'a>(args: &'a [KernelExpr]) -> Vec<&'a KernelExpr> {
    args.iter().filter(|arg| !arg.is_mode()).collect()
}

fn lower_call(
    func: &syn::Ident,
    args: &[KernelExpr],
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    let func_name = func.to_string();
    let args = call_operands(args);
    let args_exprs = args
        .iter()
        .map(|a| lower_expr(a, const_params))
        .collect::<syn::Result<Vec<_>>>()?;

    let op = match func_name.as_str() {
        "load_like" | "load_tile_mut" => quote! { ::sile::hir::BuiltinOp::LoadLike },
        "constant" => quote! { ::sile::hir::BuiltinOp::Constant },
        "mma" => quote! { ::sile::hir::BuiltinOp::Mma },
        // A tile spelling and a scalar spelling of the same operation: the
        // operands decide which one it is, so the name does not.
        "max_tile" | "max" => quote! { ::sile::hir::BuiltinOp::Max },
        "min_tile" | "min" => quote! { ::sile::hir::BuiltinOp::Min },
        "ceil_div" => quote! { ::sile::hir::BuiltinOp::CeilDiv },
        "iota" => quote! { ::sile::hir::BuiltinOp::Iota },
        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
        // A minimum is a maximum with the operands swapped and the predicate
        // inverted, so it needs no op of its own.
        "reduce_min" => quote! { ::sile::hir::BuiltinOp::ReduceMin },
        "transpose" => quote! { ::sile::hir::BuiltinOp::Transpose },
        // Every target spells each of these as one instruction, so the
        // operation travels as data and the lowering picks the spelling.
        "exp" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Exp) },
        "exp2" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Exp2) },
        "log" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Log) },
        "log2" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Log2) },
        "sqrt" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Sqrt) },
        "rsqrt" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Rsqrt) },
        // `true_div(a, b)` is a division that does not truncate. This IR has one
        // division per type and the float one is already that, so the name adds
        // nothing to the operation.
        "true_div" => quote! { ::sile::hir::BuiltinOp::Div },
        "select" => quote! { ::sile::hir::BuiltinOp::Select },
        "gt_tile" => quote! { ::sile::hir::BuiltinOp::Gt },
        "ge_tile" => quote! { ::sile::hir::BuiltinOp::Ge },
        "lt_tile" => quote! { ::sile::hir::BuiltinOp::Lt },
        "le_tile" => quote! { ::sile::hir::BuiltinOp::Le },
        "eq_tile" => quote! { ::sile::hir::BuiltinOp::Eq },
        "ne_tile" => quote! { ::sile::hir::BuiltinOp::Ne },
        "tile_to_scalar" => quote! { ::sile::hir::BuiltinOp::TileToScalar },
        "scalar_to_tile" => quote! { ::sile::hir::BuiltinOp::ScalarToTile },
        "convert_scalar" => quote! { ::sile::hir::BuiltinOp::ConvertScalar },
        // `broadcast_scalar(s, shape)` is the free-function spelling of
        // `s.broadcast(shape)`, so it lowers to the same broadcast.
        "broadcast_scalar" => quote! { ::sile::hir::BuiltinOp::Broadcast },
        "program_id" | "id" | "get_tile_block_id" => {
            quote! { ::sile::hir::BuiltinOp::ProgramId }
        }
        "num_programs" => quote! { ::sile::hir::BuiltinOp::NumPrograms },
        "num_tiles" => quote! { ::sile::hir::BuiltinOp::NumTiles },
        other => {
            return Err(syn::Error::new(
                func.span(),
                format!(
                    "`{other}` is not a kernel builtin; supported calls are {}",
                    BUILTIN_FUNCTIONS.join(", ")
                ),
            ));
        }
    };

    Ok(quote! {
        ::sile::hir::Expr::Builtin {
            op: #op,
            args: vec![#(#args_exprs),*],
        }
    })
}

fn lower_method_call(
    receiver: &KernelExpr,
    method: &syn::Ident,
    args: &[KernelExpr],
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    let method_name = method.to_string();
    let receiver_expr = lower_expr(receiver, const_params)?;
    let args = call_operands(args);
    let args_exprs = args
        .iter()
        .map(|a| lower_expr(a, const_params))
        .collect::<syn::Result<Vec<_>>>()?;

    let op = match method_name.as_str() {
        "load_tile" => quote! { ::sile::hir::BuiltinOp::LoadTile },
        // `part.load([i, j])` is the same load as `x.load_tile(shape, [i, j])`:
        // the partition already carries the buffer and the tile extent, so the
        // coordinates are the only new argument.
        "load" => quote! { ::sile::hir::BuiltinOp::Load },
        "partition" => quote! { ::sile::hir::BuiltinOp::Partition },
        // The same view, opened for writing. The buffer's own parameter
        // decides whether the store through the view is legal.
        "partition_mut" => quote! { ::sile::hir::BuiltinOp::PartitionMut },
        // `x.load_like(y)` takes its tile shape from `y`.
        "load_like" | "load_tile_mut" => quote! { ::sile::hir::BuiltinOp::LoadLike },
        "reshape" => quote! { ::sile::hir::BuiltinOp::Reshape },
        "broadcast" => quote! { ::sile::hir::BuiltinOp::Broadcast },
        "transpose" => quote! { ::sile::hir::BuiltinOp::Transpose },
        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
        "reduce_min" => quote! { ::sile::hir::BuiltinOp::ReduceMin },
        "exp" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Exp) },
        "exp2" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Exp2) },
        "log" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Log) },
        "log2" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Log2) },
        "sqrt" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Sqrt) },
        "rsqrt" => quote! { ::sile::hir::BuiltinOp::Unary(::sile::hir::UnaryOp::Rsqrt) },
        "shape" => quote! { ::sile::hir::BuiltinOp::ShapeOf },
        "store" => {
            return Ok(quote! {
                ::sile::hir::Expr::Builtin {
                    op: ::sile::hir::BuiltinOp::Store,
                    args: vec![#receiver_expr],
                }
            });
        }
        "atomic_add" => {
            return Err(syn::Error::new(
                method.span(),
                "`atomic_add` is a statement, not an expression: write it as `out.atomic_add(index, value);`",
            ));
        }
        // A tile is a value, so cloning one is the identity. The clone exists
        // because Rust's ownership rules require it when a tile is read twice.
        "clone" => return Ok(receiver_expr),
        other => {
            return Err(syn::Error::new(
                method.span(),
                format!(
                    "`{other}` is not a tile method; supported methods are {}",
                    BUILTIN_METHODS.join(", ")
                ),
            ));
        }
    };

    let all_args = [receiver_expr].into_iter().chain(args_exprs);
    Ok(quote! {
        ::sile::hir::Expr::Builtin {
            op: #op,
            args: vec![#(#all_args),*],
        }
    })
}

fn lower_field_access(
    target: &KernelExpr,
    field: &str,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    // `tile::program_id(axis)` is a scalar in CuTile Rust, so a field access on
    // it is the tuple spelling and the axis wins. `get_tile_block_id()` has no
    // axis argument, so there the field selects the axis.
    if let KernelExpr::Call { func, args } = target {
        let func_name = func.to_string();
        if func_name == "program_id" || func_name == "get_tile_block_id" {
            let axis = if func_name == "program_id" {
                match args.first() {
                    Some(KernelExpr::Lit(lit)) => lit.base10_parse::<i64>().map_err(|_| {
                        syn::Error::new(lit.span(), "program_id axis must be an integer literal")
                    })?,
                    Some(other) => {
                        return Err(syn::Error::new(
                            other.span(),
                            "program_id axis must be an integer literal",
                        ));
                    }
                    None => 0,
                }
            } else {
                // `get_tile_block_id()` returns the whole launch index, so the
                // field selects the axis.
                field.parse::<i64>().map_err(|_| {
                    syn::Error::new(
                        func.span(),
                        "the block id has no such axis; read one of .0, .1 or .2",
                    )
                })?
            };
            if !(0..3).contains(&axis) {
                return Err(syn::Error::new(
                    target.span(),
                    "program_id axis must be 0, 1 or 2",
                ));
            }
            return Ok(quote! {
                ::sile::hir::Expr::Builtin {
                    op: ::sile::hir::BuiltinOp::ProgramId,
                    args: vec![::sile::hir::Expr::ScalarI64(#axis)],
                }
            });
        }
    }

    // `a.shape().N` and `a.shape()[N]` both read one extent of a tensor.
    if let KernelExpr::MethodCall {
        receiver, method, ..
    } = target
        && method == "shape"
    {
        let dim_idx: i64 = field.parse().map_err(|_| {
            syn::Error::new(
                target.span(),
                "a shape dimension is addressed by an integer, as in `.shape().1`",
            )
        })?;
        let receiver_lowered = lower_expr(receiver, const_params)?;
        return Ok(quote! {
            ::sile::hir::Expr::Builtin {
                op: ::sile::hir::BuiltinOp::ShapeDim,
                args: vec![#receiver_lowered, ::sile::hir::Expr::ScalarI64(#dim_idx)],
            }
        });
    }

    // Any other `.N` is a value field. Only `.0` has a meaning in the DSL: the
    // scalar behind a tile index expression, which the HIR already carries.
    // Reading any other field is a mistake, because the HIR has no field to
    // read and would silently substitute zero.
    if field == "0" {
        return lower_expr(target, const_params);
    }

    Err(syn::Error::new(
        target.span(),
        format!("`{field}` is not a field a kernel value has; only `.0` is defined"),
    ))
}

fn lower_index(
    target: &KernelExpr,
    index: &KernelExpr,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    // `a.shape()[N]` is a shape read, the indexed spelling of `.shape().N`.
    if let KernelExpr::MethodCall {
        receiver, method, ..
    } = target
        && method == "shape"
    {
        let dim_expr = match index {
            KernelExpr::Lit(lit) => {
                let dim_idx = lit.base10_parse::<i64>().map_err(|_| {
                    syn::Error::new(lit.span(), "shape index must be an integer literal")
                })?;
                quote! { ::sile::hir::Expr::ScalarI64(#dim_idx) }
            }
            KernelExpr::Var(ident) => {
                let name = ident.to_string();
                quote! { ::sile::hir::Expr::Var(#name.to_string()) }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "a shape index must be an integer literal or a const parameter",
                ));
            }
        };
        let receiver_lowered = lower_expr(receiver, const_params)?;
        return Ok(quote! {
            ::sile::hir::Expr::Builtin {
                op: ::sile::hir::BuiltinOp::ShapeDim,
                args: vec![#receiver_lowered, #dim_expr],
            }
        });
    }

    // Otherwise it is a tile element read, which lowers to an extract.
    let target_lowered = lower_expr(target, const_params)?;
    let index_lowered = lower_expr(index, const_params)?;
    Ok(quote! {
        ::sile::hir::Expr::Builtin {
            op: ::sile::hir::BuiltinOp::Index,
            args: vec![#target_lowered, #index_lowered],
        }
    })
}

/// Lower `shape![..]` and `const_shape![..]` to a shape expression.
///
/// Both macros carry a comma-separated list of dimensions, which is the same
/// list an array literal carries. So the body is re-read as an array literal
/// and goes through the one shape rule instead of a second copy of it.
fn lower_shape_macro(
    mac: &syn::ExprMacro,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    let tokens = &mac.mac.tokens;
    let elems = syn::parse::Parser::parse(
        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
        tokens.clone().into_iter().collect::<proc_macro2::TokenStream>().into(),
    )
    .map_err(|err| syn::Error::new(mac.span(), format!("malformed shape: {err}")))?;
    let mut array: syn::ExprArray = syn::parse_quote! { [] };
    array.elems = elems;
    lower_array(&array, const_params)
}

/// An array literal is overloaded. In shape position it is a shape:
/// `load_tile([TILE], [pid])`. In index position it is a list of runtime
/// scalars, as CuTile Rust writes `part.load([pid_m, k])`.
///
/// Only a plain identifier or an integer literal is a shape dimension. Anything
/// else must survive as an expression, or the arithmetic behind it is dropped
/// and the tile reads the wrong data.
fn lower_array(
    arr: &syn::ExprArray,
    const_params: &[ConstParam],
) -> syn::Result<proc_macro2::TokenStream> {
    // A mixed array is an index, so every element must survive as an
    // expression. Lowering only the elements that were not simple dropped the
    // rest: `[pid_m, 0 + k]` became a one-element index, the missing axis
    // defaulted to zero, and the tile read the wrong rows without an error.
    let pure_shape = arr
        .elems
        .iter()
        .all(|elem| shape_literal_dim(elem).is_some() || shape_literal_symbol(elem).is_some());

    if !pure_shape {
        let index_elems = arr
            .elems
            .iter()
            .map(|elem| {
                let parsed = super::parse::parse_expr_pub(elem)?;
                lower_expr(&parsed, const_params)
            })
            .collect::<syn::Result<Vec<_>>>()?;
        return Ok(quote! { ::sile::hir::Expr::TileIndex(vec![#(#index_elems),*]) });
    }

    let shape_elems = arr
        .elems
        .iter()
        .map(|elem| match (shape_literal_dim(elem), shape_literal_symbol(elem)) {
            (Some(dim), _) => Ok(quote! { ::sile::hir::ShapeExpr::constant(#dim) }),
            // A bare name is kept as a symbol here. Whether it is a shape
            // dimension (a const parameter) or a runtime tile index (a local)
            // depends on the position the array appears in, which the callee
            // knows; the type checker resolves that.
            (None, Some(name)) => Ok(quote! { ::sile::hir::ShapeExpr::symbol(#name) }),
            (None, None) => unreachable!("the array was checked to hold only shape dimensions"),
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        ::sile::hir::Expr::Shape(
            ::sile::hir::ShapeExpr::tuple(vec![#(#shape_elems),*])
        )
    })
}

/// A plain integer in an array literal, or `-1` for a dynamic dimension.
fn shape_literal_dim(elem: &syn::Expr) -> Option<i64> {
    match elem {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(v),
            ..
        }) => Some(v.base10_parse().unwrap_or(-1)),
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr,
            ..
        }) => shape_literal_dim(expr).map(|v| -v),
        _ => None,
    }
}

/// A bare identifier, which is a shape symbol only when the array holds nothing
/// but shape dimensions.
fn shape_literal_symbol(elem: &syn::Expr) -> Option<String> {
    match elem {
        syn::Expr::Path(path) if path.path.segments.len() == 1 => {
            path.path.get_ident().map(|ident| ident.to_string())
        }
        _ => None,
    }
}
