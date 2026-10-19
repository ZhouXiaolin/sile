use syn::spanned::Spanned;

use super::ast::{
    BinOpKind, ConstParam, KernelDecl, KernelExpr, KernelParam, KernelShapeDim, KernelStmt,
    ScalarType,
};

pub fn parse_kernel(input: &syn::ItemFn) -> syn::Result<KernelDecl> {
    // Extract const params from generic parameters.
    //
    // A plain `const N: i64` is one value. An array parameter
    // `const S: [i32; 2]` names a whole shape, which stable Rust cannot carry
    // as a const generic, so it expands to one integer per dimension and the
    // kernel's own types keep referring to `S`.
    let const_params: Vec<ConstParam> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Const(c) = p {
                Some(ConstParam {
                    ident: c.ident.clone(),
                    array_rank: const_array_rank(&c.ty),
                })
            } else {
                None
            }
        })
        .collect();

    let params = input
        .sig
        .inputs
        .iter()
        .map(parse_param)
        .collect::<syn::Result<Vec<_>>>()?;

    let mut body = Vec::new();
    for stmt in &input.block.stmts {
        parse_stmt(stmt, &mut body)?;
    }

    resolve_block_id_fields(&mut body);

    Ok(KernelDecl {
        name: input.sig.ident.clone(),
        const_params,
        params,
        body,
    })
}

/// Parse one statement of a kernel body, or of a loop body.
///
/// The two positions accept the same statements, so they share this function.
/// Two copies let `store` mean different things at top level and inside a loop.
fn parse_stmt(stmt: &syn::Stmt, body: &mut Vec<KernelStmt>) -> syn::Result<()> {
    match stmt {
        // `cuda_tile_print!("fmt {}", x);` with its semicolon is a macro
        // statement, not an expression statement. The format string is a
        // literal and every later argument is an expression to print.
        syn::Stmt::Macro(mac)
            if mac
                .mac
                .path
                .segments
                .last()
                .is_some_and(|seg| seg.ident == "cuda_tile_print") =>
        {
            let parsed = mac.mac.parse_body_with(
                syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
            )?;
            let mut items = parsed.into_iter();
            let format = match items.next() {
                Some(syn::Expr::Lit(lit)) => match lit.lit {
                    syn::Lit::Str(text) => text.value(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "the print format must be a string literal",
                        ));
                    }
                },
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "the print format must be a string literal",
                    ));
                }
            };
            let args = items
                .map(|arg| parse_expr(&arg))
                .collect::<syn::Result<Vec<_>>>()?;
            body.push(KernelStmt::Print { format, args });
            Ok(())
        }
        syn::Stmt::Local(local) => {
            let name = match &local.pat {
                syn::Pat::Ident(pat) => pat.ident.clone(),
                syn::Pat::Type(pat_type) => {
                    if let syn::Pat::Ident(inner) = pat_type.pat.as_ref() {
                        inner.ident.clone()
                    } else {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "expected ident pattern",
                        ));
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        &local.pat,
                        "expected ident pattern",
                    ));
                }
            };
            let expr = local
                .init
                .as_ref()
                .map(|init| parse_expr(init.expr.as_ref()))
                .transpose()?
                .ok_or_else(|| {
                    syn::Error::new_spanned(local, "let binding requires initializer")
                })?;
            body.push(KernelStmt::Let { name, expr });
            Ok(())
        }
        syn::Stmt::Expr(expr, _) => {
            if let syn::Expr::ForLoop(for_loop) = expr {
                parse_for_loop(for_loop, body)?;
                return Ok(());
            }
            if let syn::Expr::While(while_expr) = expr {
                let cond = parse_expr(&while_expr.cond)?;
                let loop_body = parse_block(&while_expr.body)?;
                body.push(KernelStmt::While {
                    cond,
                    body: loop_body,
                });
                return Ok(());
            }
            if let syn::Expr::Loop(loop_expr) = expr {
                let loop_body = parse_block(&loop_expr.body)?;
                body.push(KernelStmt::Loop { body: loop_body });
                return Ok(());
            }
            if let syn::Expr::Break(break_expr) = expr {
                if let Some(value) = break_expr.expr.as_ref() {
                    return Err(syn::Error::new_spanned(
                        value,
                        "`break` takes no value in a kernel",
                    ));
                }
                body.push(KernelStmt::Break);
                return Ok(());
            }
            if let syn::Expr::Return(ret) = expr {
                // A kernel returns nothing, so a bare `return` ends the body
                // and a value has no place to go.
                if let Some(value) = ret.expr.as_ref() {
                    return Err(syn::Error::new_spanned(
                        value,
                        "a kernel returns nothing; write `return;`",
                    ));
                }
                body.push(KernelStmt::Return);
                return Ok(());
            }
            if let syn::Expr::If(if_expr) = expr {
                body.push(parse_if(if_expr)?);
                return Ok(());
            }
            if let syn::Expr::Assign(assign) = expr {
                let name = match assign.left.as_ref() {
                    syn::Expr::Path(path) => path.path.segments.last().unwrap().ident.clone(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &assign.left,
                            "assignment target must be an ident",
                        ));
                    }
                };
                let expr = parse_expr(&assign.right)?;
                body.push(KernelStmt::Assign { name, expr });
                return Ok(());
            }
            if let syn::Expr::MethodCall(call) = expr {
                if call.method == "store" {
                    body.push(parse_store(call)?);
                    return Ok(());
                }
                if call.method == "atomic_add" {
                    body.push(parse_atomic_add(call)?);
                    return Ok(());
                }
            }
            // `store_tile(y, t)` is the free-function spelling of `y.store(t)`:
            // the buffer is the first argument instead of the receiver.
            if let syn::Expr::Call(call) = expr
                && let syn::Expr::Path(func_path) = call.func.as_ref()
                && func_path
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "store_tile")
            {
                let mut args = call.args.iter();
                let target = match args.next() {
                    Some(syn::Expr::Path(path)) => path.path.segments.last().unwrap().ident.clone(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &call.func,
                            "store_tile target must be an ident",
                        ));
                    }
                };
                let value = parse_expr(args.next().ok_or_else(|| {
                    syn::Error::new_spanned(call, "store_tile requires a value")
                })?)?;
                if args.next().is_some() {
                    return Err(syn::Error::new_spanned(
                        call,
                        "store_tile expects a buffer and a value",
                    ));
                }
                body.push(KernelStmt::Store {
                    target,
                    value,
                    index: None,
                });
                return Ok(());
            }
            // `unsafe { ... }` marks an operation whose contract the caller
            // promises to keep. That promise is a property of the program, not
            // of the instruction stream, so the block contributes its
            // statements and nothing else.
            if let syn::Expr::Unsafe(unsafe_block) = expr {
                for stmt in &unsafe_block.block.stmts {
                    parse_stmt(stmt, body)?;
                }
                return Ok(());
            }
            if matches!(expr, syn::Expr::Continue(_)) {
                body.push(KernelStmt::Continue);
                return Ok(());
            }
            Err(syn::Error::new_spanned(
                expr,
                "unsupported kernel statement",
            ))
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported kernel statement",
        )),
    }
}

/// `y.store(t)` writes the tile the program owns; `part.store(t, [i, j])`
/// writes at coordinates. The second argument, when present, names them.
fn parse_store(call: &syn::ExprMethodCall) -> syn::Result<KernelStmt> {
    let target = match call.receiver.as_ref() {
        syn::Expr::Path(path) => path.path.segments.last().unwrap().ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &call.receiver,
                "store target must be an ident",
            ));
        }
    };
    let mut args = call.args.iter();
    let value = parse_expr(
        args.next()
            .ok_or_else(|| syn::Error::new_spanned(call, "store requires a value"))?,
    )?;
    let index = args.next().map(parse_expr).transpose()?;
    if args.next().is_some() {
        return Err(syn::Error::new_spanned(
            call,
            "store expects a value and, optionally, one index",
        ));
    }
    Ok(KernelStmt::Store {
        target,
        value,
        index,
    })
}

fn parse_atomic_add(call: &syn::ExprMethodCall) -> syn::Result<KernelStmt> {
    let target = match call.receiver.as_ref() {
        syn::Expr::Path(path) => path.path.segments.last().unwrap().ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &call.receiver,
                "atomic_add target must be an ident",
            ));
        }
    };
    let mut args = call.args.iter();
    let index = parse_expr(
        args.next()
            .ok_or_else(|| syn::Error::new_spanned(call, "atomic_add requires index argument"))?,
    )?;
    let value = parse_expr(
        args.next()
            .ok_or_else(|| syn::Error::new_spanned(call, "atomic_add requires value argument"))?,
    )?;
    if args.next().is_some() {
        return Err(syn::Error::new_spanned(
            call,
            "atomic_add expects exactly two arguments",
        ));
    }
    Ok(KernelStmt::AtomicAdd {
        target,
        index,
        value,
    })
}

/// Replace `pid.0` with the block id it names, when `pid` holds one.
///
/// CuTile reads the launch index as a tuple, so a kernel binds it once and
/// projects the axis it needs. The projection is not a value field: it selects
/// a grid axis, which is what `program_id(axis)` already means. Rewriting the
/// field into that call keeps one representation of a launch index in the HIR.
fn resolve_block_id_fields(body: &mut [KernelStmt]) {
    let mut block_ids: Vec<syn::Ident> = Vec::new();
    collect_block_id_bindings(body, &mut block_ids);
    if block_ids.is_empty() {
        return;
    }
    rewrite_block_id_fields(body, &block_ids);
}

fn collect_block_id_bindings(body: &[KernelStmt], out: &mut Vec<syn::Ident>) {
    for stmt in body {
        match stmt {
            KernelStmt::Let { name, expr } => {
                if is_block_id_call(expr) {
                    out.push(name.clone());
                }
            }
            KernelStmt::ForLoop { body, .. } => collect_block_id_bindings(body, out),
            KernelStmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_block_id_bindings(then_body, out);
                collect_block_id_bindings(else_body, out);
            }
            _ => {}
        }
    }
}

fn is_block_id_call(expr: &KernelExpr) -> bool {
    matches!(
        expr,
        KernelExpr::Call { func, .. }
            if func == "get_tile_block_id" || func == "program_id"
    )
}

fn rewrite_block_id_fields(body: &mut [KernelStmt], block_ids: &[syn::Ident]) {
    for stmt in body.iter_mut() {
        match stmt {
            KernelStmt::Let { expr, .. } | KernelStmt::Assign { expr, .. } => {
                rewrite_block_id_expr(expr, block_ids);
            }
            KernelStmt::Store { value, index, .. } => {
                rewrite_block_id_expr(value, block_ids);
                if let Some(index) = index {
                    rewrite_block_id_expr(index, block_ids);
                }
            }
            KernelStmt::AtomicAdd { index, value, .. } => {
                rewrite_block_id_expr(index, block_ids);
                rewrite_block_id_expr(value, block_ids);
            }
            KernelStmt::Print { args, .. } => {
                for arg in args {
                    rewrite_block_id_expr(arg, block_ids);
                }
            }
            KernelStmt::ForLoop {
                start,
                end,
                step,
                body,
                ..
            } => {
                rewrite_block_id_expr(start, block_ids);
                rewrite_block_id_expr(end, block_ids);
                if let Some(step) = step {
                    rewrite_block_id_expr(step, block_ids);
                }
                rewrite_block_id_fields(body, block_ids);
            }
            KernelStmt::While { cond, body } => {
                rewrite_block_id_expr(cond, block_ids);
                rewrite_block_id_fields(body, block_ids);
            }
            KernelStmt::Loop { body } => rewrite_block_id_fields(body, block_ids),
            KernelStmt::Break | KernelStmt::Return => {}
            KernelStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                rewrite_block_id_expr(cond, block_ids);
                rewrite_block_id_fields(then_body, block_ids);
                rewrite_block_id_fields(else_body, block_ids);
            }
            KernelStmt::Continue => {}
        }
    }
}

fn rewrite_block_id_expr(expr: &mut KernelExpr, block_ids: &[syn::Ident]) {
    // A field of a block id is an axis. The binding named the whole tuple, so
    // the read becomes `get_tile_block_id().k`: the tuple spelling, which reads
    // the three ids once. CuTile's `program_id(axis)` is a different call that
    // reads them again every time, and writing it here would charge a kernel
    // for ids it never spends.
    if let KernelExpr::FieldAccess { target, field } = expr
        && let KernelExpr::Var(name) = target.as_ref()
        && block_ids.contains(name)
        && let Ok(axis) = field.parse::<u16>()
    {
        let func = syn::Ident::new("get_tile_block_id", name.span());
        *expr = KernelExpr::FieldAccess {
            target: Box::new(KernelExpr::Call { func, args: Vec::new() }),
            field: syn::LitInt::new(&axis.to_string(), name.span()).to_string(),
        };
        return;
    }
    match expr {
        KernelExpr::MethodCall { receiver, args, .. } => {
            rewrite_block_id_expr(receiver, block_ids);
            for arg in args {
                rewrite_block_id_expr(arg, block_ids);
            }
        }
        KernelExpr::Call { args, .. } => {
            for arg in args {
                rewrite_block_id_expr(arg, block_ids);
            }
        }
        KernelExpr::BinaryOp { left, right, .. } => {
            rewrite_block_id_expr(left, block_ids);
            rewrite_block_id_expr(right, block_ids);
        }
        KernelExpr::FieldAccess { target, .. } => rewrite_block_id_expr(target, block_ids),
        KernelExpr::Index { target, index } => {
            rewrite_block_id_expr(target, block_ids);
            rewrite_block_id_expr(index, block_ids);
        }
        // An index array holds raw tokens, not parsed expressions, so its
        // elements need the same rewrite on the `syn` side.
        KernelExpr::Array(arr) => {
            for elem in &mut arr.elems {
                rewrite_block_id_syn(elem, block_ids);
            }
        }
        _ => {}
    }
}

fn rewrite_block_id_syn(expr: &mut syn::Expr, block_ids: &[syn::Ident]) {
    if let syn::Expr::Field(field) = expr
        && let syn::Expr::Path(path) = field.base.as_ref()
        && let Some(name) = path.path.get_ident()
        && block_ids.contains(name)
        && let syn::Member::Unnamed(index) = &field.member
    {
        // The binding named the whole tuple, so the read stays a tuple read:
        // `get_tile_block_id().k` fetches the three ids once, while CuTile's
        // `program_id(axis)` fetches them again on every call.
        let index = syn::Member::Unnamed(index.clone());
        *expr = syn::parse_quote! { get_tile_block_id().#index };
        return;
    }
    match expr {
        syn::Expr::Binary(binary) => {
            rewrite_block_id_syn(&mut binary.left, block_ids);
            rewrite_block_id_syn(&mut binary.right, block_ids);
        }
        syn::Expr::Paren(paren) => rewrite_block_id_syn(&mut paren.expr, block_ids),
        syn::Expr::Unary(unary) => rewrite_block_id_syn(&mut unary.expr, block_ids),
        syn::Expr::Call(call) => {
            for arg in &mut call.args {
                rewrite_block_id_syn(arg, block_ids);
            }
        }
        syn::Expr::MethodCall(call) => {
            rewrite_block_id_syn(&mut call.receiver, block_ids);
            for arg in &mut call.args {
                rewrite_block_id_syn(arg, block_ids);
            }
        }
        _ => {}
    }
}

/// `if cond { .. } else { .. }`.
///
/// The arms hold statements, so they go through the same parser a loop body
/// does. An `else if` is an `if` statement inside the else arm, which is what
/// makes one variant carry a whole chain.
fn parse_if(if_expr: &syn::ExprIf) -> syn::Result<KernelStmt> {
    let cond = parse_expr(&if_expr.cond)?;
    let then_body = parse_block(&if_expr.then_branch)?;
    let else_body = match &if_expr.else_branch {
        None => Vec::new(),
        Some((_, branch)) => match branch.as_ref() {
            syn::Expr::Block(block) => parse_block(&block.block)?,
            syn::Expr::If(nested) => vec![parse_if(nested)?],
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "an else branch must be a block or another `if`",
                ));
            }
        },
    };
    Ok(KernelStmt::If {
        cond,
        then_body,
        else_body,
    })
}

fn parse_block(block: &syn::Block) -> syn::Result<Vec<KernelStmt>> {
    let mut body = Vec::new();
    for stmt in &block.stmts {
        parse_stmt(stmt, &mut body)?;
    }
    Ok(body)
}

fn parse_for_loop(for_loop: &syn::ExprForLoop, body: &mut Vec<KernelStmt>) -> syn::Result<()> {
    let var = match for_loop.pat.as_ref() {
        syn::Pat::Ident(pat) => pat.ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.pat,
                "for loop variable must be an ident",
            ));
        }
    };
    let (start, end, step) =
        match for_loop.expr.as_ref() {
            syn::Expr::Range(range) => {
                let start_expr = range.start.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(range, "for loop range must have a start")
                })?;
                let end_expr = range.end.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(range, "for loop range must have an end")
                })?;
                let start = parse_expr(start_expr)?;
                let end = parse_expr(end_expr)?;
                // `a..=b` includes `b`, so the exclusive bound the HIR compares
                // against is one past it. Rewriting here keeps one loop form.
                match range.limits {
                    syn::RangeLimits::Closed(_) => (
                        start,
                        KernelExpr::BinaryOp {
                            left: Box::new(end),
                            op: BinOpKind::Add,
                            right: Box::new(KernelExpr::Lit(syn::LitInt::new(
                                "1",
                                range.limits.span(),
                            ))),
                        },
                        None,
                    ),
                    syn::RangeLimits::HalfOpen(_) => (start, end, None),
                }
            }
            // `(a..b).step_by(k)` names a range with a step. The method is Rust's
            // iterator adapter, and the step is the only thing it adds here.
            syn::Expr::MethodCall(call) if call.method == "step_by" => {
                // The range may wear parentheses, which syn keeps as its own node.
                let receiver = match call.receiver.as_ref() {
                    syn::Expr::Paren(paren) => paren.expr.as_ref(),
                    other => other,
                };
                let syn::Expr::Range(range) = receiver else {
                    return Err(syn::Error::new_spanned(
                        &call.receiver,
                        "step_by must follow a range",
                    ));
                };
                let start_expr = range.start.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(range, "for loop range must have a start")
                })?;
                let end_expr = range.end.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(range, "for loop range must have an end")
                })?;
                let start = parse_expr(start_expr)?;
                let end = parse_expr(end_expr)?;
                let step = parse_expr(
                    call.args
                        .first()
                        .ok_or_else(|| syn::Error::new_spanned(call, "step_by needs a step"))?,
                )?;
                match range.limits {
                    syn::RangeLimits::Closed(_) => (
                        start,
                        KernelExpr::BinaryOp {
                            left: Box::new(end),
                            op: BinOpKind::Add,
                            right: Box::new(KernelExpr::Lit(syn::LitInt::new(
                                "1",
                                range.limits.span(),
                            ))),
                        },
                        Some(step),
                    ),
                    syn::RangeLimits::HalfOpen(_) => (start, end, Some(step)),
                }
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    &for_loop.expr,
                    "for loop expression must be a range",
                ));
            }
        };
    let loop_body = parse_block(&for_loop.body)?;
    body.push(KernelStmt::ForLoop {
        var,
        start,
        end,
        step,
        body: loop_body,
    });
    Ok(())
}

fn parse_param(arg: &syn::FnArg) -> syn::Result<KernelParam> {
    let syn::FnArg::Typed(arg) = arg else {
        return Err(syn::Error::new_spanned(
            arg,
            "receiver parameters are unsupported",
        ));
    };
    let syn::Pat::Ident(pat) = arg.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &arg.pat,
            "expected ident parameter",
        ));
    };
    // A bare scalar type names a value the kernel reads directly, the way
    // CuTile writes `a: f32`. It is not a buffer, so it has no shape and no
    // element to address.
    if let Some(scalar_ty) = scalar_type(&arg.ty) {
        return Ok(KernelParam {
            name: pat.ident.clone(),
            is_mut: false,
            shape: None,
            scalar_ty: Some(scalar_ty),
            elem: super::ast::BufferElemType::F32,
        });
    }

    let syn::Type::Reference(reference) = arg.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &arg.ty,
            "a kernel parameter is a reference, or a scalar such as `f32`",
        ));
    };

    // A parameter whose type is not a `Tensor`/`Tile` has no shape to lower.
    // Accepting it as dynamic lets a kernel declare `x: &Foo` and read it as a
    // one-dimensional buffer, which is a different program from the one
    // written.
    if !is_shape_bearing_type(&reference.elem) {
        return Err(syn::Error::new_spanned(
            &reference.elem,
            "a kernel parameter must be a Tensor or a Tile",
        ));
    }
    // The buffers carry f32 or f16. A kernel that names another element type
    // would otherwise compile and read its bytes as floats, which is a
    // different program from the one written. A half buffer converts at its
    // loads and stores; the compute between them stays f32.
    let elem = match buffer_element_spelling(&reference.elem).as_deref() {
        Some("f16") => super::ast::BufferElemType::F16,
        Some("f32") | None => super::ast::BufferElemType::F32,
        Some(other) => {
            return Err(syn::Error::new_spanned(
                &reference.elem,
                format!("a buffer parameter holds f32 or f16; `{other}` is not carried"),
            ));
        }
    };
    let shape = extract_shape_from_type(&reference.elem);

    Ok(KernelParam {
        name: pat.ident.clone(),
        is_mut: reference.mutability.is_some(),
        shape,
        scalar_ty: None,
        elem,
    })
}

/// The scalar parameter type the ABI carries, if the type is one.
///
/// A buffer holds f32 and a scalar argument may be f32 or an integer. CuTile
/// writes the integer as `i32`; this front end carries it as `i64`, because a
/// launch index is i64 here and a division mixes the two. The `i32` spelling
/// is accepted and widened, so a kernel written for CuTile reads as it does
/// there.
fn scalar_type(ty: &syn::Type) -> Option<ScalarType> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.path.is_ident("f32") {
        return Some(ScalarType::F32);
    }
    if path.path.is_ident("i64") || path.path.is_ident("i32") {
        return Some(ScalarType::I64);
    }
    None
}

/// The value of a float constant written as a path.
///
/// A kernel writes a limit value as the associated constant Rust spells it
/// with: the two infinities a sentinel needs, plus not-a-number, the smallest
/// and largest finite values, and the spacing of ones. Every other path is not
/// a value.
fn float_constant(path: &syn::Path) -> Option<f32> {
    let names: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    match names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["f32", "NEG_INFINITY"] => Some(f32::NEG_INFINITY),
        ["f32", "INFINITY"] => Some(f32::INFINITY),
        ["f32", "NAN"] => Some(f32::NAN),
        ["f32", "EPSILON"] => Some(f32::EPSILON),
        ["f32", "MIN"] => Some(f32::MIN),
        ["f32", "MAX"] => Some(f32::MAX),
        ["f32", "ZERO"] => Some(0.0),
        _ => None,
    }
}

/// Whether a qualified path names a mode or a scope.
///
/// A mode carries no value, so a call drops it: `sqrt(x, rounding::Zero,
/// ftz::Disabled)` is the same operation as `sqrt(x)`. The namespaces are the
/// ones CuTile writes, and a path with type arguments is a `None` scope, which
/// names a type rather than a value.
pub(crate) fn is_mode_path(path: &syn::Path) -> bool {
    let has_type_arguments = path
        .segments
        .iter()
        .any(|segment| !matches!(segment.arguments, syn::PathArguments::None));
    if has_type_arguments {
        return true;
    }
    matches!(
        path.segments.first().map(|segment| segment.ident.to_string()),
        Some(name)
            if matches!(
                name.as_str(),
                "ftz"
                    | "rounding"
                    | "ordering"
                    | "scope"
                    | "nan"
                    | "fast_acc"
                    | "saturating"
                    | "overflow"
                    | "predicate"
                    | "cmp_ordering"
                    | "reverse"
                    | "signedness"
            )
    )
}

/// The rank of a const parameter written as an array type, `[i32; 2]`.
///
/// Any other type is a plain integer parameter and returns `None`.
pub(crate) fn const_array_rank(ty: &syn::Type) -> Option<usize> {
    let syn::Type::Array(array) = ty else {
        return None;
    };
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(len),
        ..
    }) = &array.len
    else {
        return None;
    };
    len.base10_parse::<usize>().ok()
}

/// Is this a type the kernel ABI knows, even when it names no shape?
///
/// `Tensor<f32>` has no shape argument but is still a tensor, so it lowers to a
/// dynamic shape rather than being rejected.
fn is_shape_bearing_type(ty: &syn::Type) -> bool {
    let syn::Type::Path(type_path) = ty else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Tensor" || segment.ident == "Tile")
}

/// The element type a `Tensor`/`Tile` parameter names, as the source spells it.
///
/// A type with no element argument (`Tensor<f32>` has one, but a bare path does
/// not) or a generic parameter returns the name it carries either way, so the
/// caller can name it in an error.
fn buffer_element_spelling(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let last_seg = type_path.path.segments.last()?;
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(syn::Type::Path(elem)) = args.args.first()? else {
        return None;
    };
    // The element is named by its last path segment, so a crate-qualified
    // `half::f16` spells the same element as a bare `f16`.
    elem.path.segments.last().map(|seg| seg.ident.to_string())
}

fn extract_shape_from_type(ty: &syn::Type) -> Option<Vec<KernelShapeDim>> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let last_seg = type_path.path.segments.last()?;
    // A parameter may name its view as `Tensor` or as `Tile`; both carry the
    // shape the same way, so both extract it here.
    if last_seg.ident != "Tensor" && last_seg.ident != "Tile" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    if args.args.len() < 2 {
        return None;
    }
    // `Tensor<f32, S>` names the shape by naming a const parameter, which the
    // caller expands into one dimension per rank.
    if let syn::GenericArgument::Type(syn::Type::Path(path)) = &args.args[1]
        && let Some(ident) = path.path.get_ident()
    {
        return Some(vec![KernelShapeDim::Symbol(ident.clone())]);
    }
    let syn::GenericArgument::Const(const_expr) = &args.args[1] else {
        return None;
    };
    extract_shape_expr(const_expr)
}

fn extract_shape_expr(expr: &syn::Expr) -> Option<Vec<KernelShapeDim>> {
    match expr {
        syn::Expr::Block(block) => {
            let tail = block.block.stmts.last()?;
            if let syn::Stmt::Expr(expr, _) = tail {
                extract_shape_expr(expr)
            } else {
                None
            }
        }
        syn::Expr::Array(arr) => {
            let mut shape = Vec::new();
            for elem in &arr.elems {
                let dim = match elem {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(lit_int),
                        ..
                    }) => {
                        let value = lit_int.base10_parse::<i64>().unwrap_or(-1);
                        if value == -1 {
                            KernelShapeDim::Dynamic
                        } else {
                            KernelShapeDim::Constant(value)
                        }
                    }
                    syn::Expr::Unary(syn::ExprUnary {
                        op: syn::UnOp::Neg(_),
                        expr,
                        ..
                    }) => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Int(lit_int),
                            ..
                        }) = expr.as_ref()
                        {
                            let value = lit_int.base10_parse::<i64>().unwrap_or(1);
                            KernelShapeDim::Constant(-value)
                        } else {
                            KernelShapeDim::Dynamic
                        }
                    }
                    syn::Expr::Path(path) => path
                        .path
                        .get_ident()
                        .cloned()
                        .map(KernelShapeDim::Symbol)
                        .unwrap_or(KernelShapeDim::Dynamic),
                    _ => KernelShapeDim::Dynamic,
                };
                shape.push(dim);
            }
            if shape.is_empty() { None } else { Some(shape) }
        }
        _ => None,
    }
}

pub(crate) fn parse_expr_pub(expr: &syn::Expr) -> syn::Result<KernelExpr> {
    parse_expr(expr)
}

fn parse_expr(expr: &syn::Expr) -> syn::Result<KernelExpr> {
    match expr {
        syn::Expr::MethodCall(call) => {
            let receiver = parse_expr(&call.receiver)?;
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::MethodCall {
                receiver: Box::new(receiver),
                method: call.method.clone(),
                args,
            })
        }
        syn::Expr::Binary(binary) => {
            let left = parse_expr(&binary.left)?;
            let right = parse_expr(&binary.right)?;
            let op = match binary.op {
                syn::BinOp::Add(_) => BinOpKind::Add,
                syn::BinOp::Sub(_) => BinOpKind::Sub,
                syn::BinOp::Mul(_) => BinOpKind::Mul,
                syn::BinOp::Div(_) => BinOpKind::Div,
                syn::BinOp::Rem(_) => BinOpKind::Rem,
                syn::BinOp::Eq(_) => BinOpKind::Eq,
                syn::BinOp::Ne(_) => BinOpKind::Ne,
                syn::BinOp::Lt(_) => BinOpKind::Lt,
                syn::BinOp::Le(_) => BinOpKind::Le,
                syn::BinOp::Gt(_) => BinOpKind::Gt,
                syn::BinOp::Ge(_) => BinOpKind::Ge,
                // `&` and `&&` are one operation, and so are `|` and `||`: an
                // operand here is a value, so there is nothing to short circuit.
                syn::BinOp::BitAnd(_) | syn::BinOp::And(_) => BinOpKind::And,
                syn::BinOp::BitOr(_) | syn::BinOp::Or(_) => BinOpKind::Or,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &binary.op,
                        "unsupported binary operator",
                    ));
                }
            };
            Ok(KernelExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        syn::Expr::Unary(unary) => {
            let inner = parse_expr(&unary.expr)?;
            match unary.op {
                // CuTile writes `-x` for a negated tile or a negative scalar,
                // so a minus lowers to the negation operation the IR has.
                syn::UnOp::Neg(_) => Ok(KernelExpr::Neg(Box::new(inner))),
                _ => Err(syn::Error::new_spanned(
                    &unary.op,
                    "unsupported unary operator",
                )),
            }
        }
        syn::Expr::Path(path) => match path.path.get_ident().cloned() {
            Some(ident) => Ok(KernelExpr::Var(ident)),
            // A qualified path does not name a variable. Three things reach
            // here: a float constant (`f32::NEG_INFINITY`), a mode or a scope
            // (`ftz::Disabled`), and a `None` scope (`None::<TileBlock>`).
            // Call lowering drops a mode, so a path it does not recognize must
            // be an error: dropping one silently would drop an operand the
            // kernel wrote and read a default instead.
            None => {
                if let Some(value) = float_constant(&path.path) {
                    return Ok(KernelExpr::FloatConst(value));
                }
                if is_mode_path(&path.path) {
                    return Ok(KernelExpr::ModePath(path.path.clone()));
                }
                Err(syn::Error::new_spanned(
                    &path.path,
                    "a qualified path in a kernel names a mode, a scope or a float constant",
                ))
            }
        },
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(lit_int) => Ok(KernelExpr::Lit(lit_int.clone())),
            syn::Lit::Float(lit_float) => Ok(KernelExpr::FloatLit(lit_float.clone())),
            syn::Lit::Bool(lit_bool) => Ok(KernelExpr::BoolLit(lit_bool.clone())),
            other => Err(syn::Error::new_spanned(
                other,
                "only integer, float and boolean literals are supported in kernel expressions",
            )),
        },
        syn::Expr::Array(arr) => Ok(KernelExpr::Array(arr.clone())),
        // `|a, b| <one expression>`: the combiner a `scan` folds with. Two
        // plain names and one expression is the whole shape; anything else —
        // a typed parameter, a statement body, a different arity — is a
        // combiner this front end cannot carry.
        syn::Expr::Closure(closure) => {
            let mut params = Vec::new();
            for input in &closure.inputs {
                match input {
                    syn::Pat::Ident(pat) => params.push(pat.ident.clone()),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "a scan combiner's parameters must be plain names",
                        ));
                    }
                }
            }
            let body = parse_expr(&closure.body)?;
            if params.len() != 2 {
                return Err(syn::Error::new_spanned(
                    closure,
                    "a scan combiner takes two parameters",
                ));
            }
            Ok(KernelExpr::Closure {
                params,
                body: Box::new(body),
            })
        }
        // CuTile writes a compile-time shape as a macro call, because the rank
        // is part of the type there. The rank is already in the argument list
        // here, so the macro is syntax: it lowers to the same shape expression
        // an array literal does.
        syn::Expr::Macro(mac) => {
            let name = mac
                .mac
                .path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .unwrap_or_default();
            match name.as_str() {
                "shape" | "const_shape" => Ok(KernelExpr::ShapeMacro(mac.clone())),
                other => Err(syn::Error::new_spanned(
                    &mac.mac.path,
                    format!(
                        "`{other}!` is not a kernel macro; supported macros are shape!, const_shape!"
                    ),
                )),
            }
        }
        syn::Expr::Field(field) => {
            let target = parse_expr(&field.base)?;
            let field = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            Ok(KernelExpr::FieldAccess {
                target: Box::new(target),
                field,
            })
        }
        syn::Expr::Call(call) => {
            let func = match call.func.as_ref() {
                syn::Expr::Path(path) => path
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.clone())
                    .ok_or_else(|| syn::Error::new_spanned(&call.func, "expected function name"))?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &call.func,
                        "expected simple function call",
                    ));
                }
            };
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::Call { func, args })
        }
        syn::Expr::Reference(reference) => parse_expr(&reference.expr),
        syn::Expr::Paren(paren) => parse_expr(&paren.expr),
        // An `unsafe` block in value position — `let a = unsafe {
        // assume_div_by::<_, 8>(t) };` — marks a promise about the operation
        // it wraps, and a promise is a property of the program rather than of
        // the instruction stream. The block contributes its one expression
        // and nothing else; statement position strips the same marker.
        syn::Expr::Unsafe(unsafe_block) => {
            let message = "an `unsafe` block in a kernel holds one expression";
            let [only] = unsafe_block.block.stmts.as_slice() else {
                return Err(syn::Error::new_spanned(&unsafe_block.block, message));
            };
            let syn::Stmt::Expr(expr, None) = only else {
                return Err(syn::Error::new_spanned(only, message));
            };
            parse_expr(expr)
        }
        syn::Expr::Index(index) => {
            let target = parse_expr(&index.expr)?;
            let idx = parse_expr(&index.index)?;
            Ok(KernelExpr::Index {
                target: Box::new(target),
                index: Box::new(idx),
            })
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported expression kind in kernel",
        )),
    }
}
