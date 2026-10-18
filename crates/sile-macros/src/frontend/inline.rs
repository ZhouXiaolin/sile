//! The AST-level inlining that turns a device function's calls into its body.
//!
//! A `#[sile::module]` keeps device functions as plain functions and expands
//! every call to one at its call site, so the kernel pipeline sees one flat
//! body per entry — the same program a single-function kernel would name.
//! The expansion binds each parameter to a fresh name before the body runs,
//! which keeps a re-assignment inside the body from writing back into the
//! caller's variable, and it turns the body's `return` into a binding the
//! call site reads.

use std::collections::HashMap;

use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;

/// A device function the inliner knows how to expand.
pub(crate) struct DeviceFn {
    pub params: Vec<String>,
    /// Whether the signature names a return type: a body without one has no
    /// value to hand back to the call site.
    pub returns: bool,
    pub body: Vec<syn::Stmt>,
}

/// Whether an expression is a direct call to a device function.
fn is_device_call(expr: &syn::Expr, fns: &HashMap<String, DeviceFn>) -> bool {
    let syn::Expr::Call(call) = expr else {
        return false;
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    path.path
        .get_ident()
        .map(|ident| fns.contains_key(&ident.to_string()))
        .unwrap_or(false)
}

/// Inline every device-function call in a statement list, outer calls first
/// and the calls inside an expanded body as the expansion runs. `stack` holds
/// the functions currently expanding, so a cycle is a compile error rather
/// than a hang.
pub(crate) fn inline_device_calls(
    stmts: &[syn::Stmt],
    fns: &HashMap<String, DeviceFn>,
    stack: &mut Vec<String>,
    fresh: &mut usize,
) -> syn::Result<Vec<syn::Stmt>> {
    let mut out = Vec::new();
    for stmt in stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                let mut local = local.clone();
                if let Some(init) = local.init.as_mut() {
                    let (prefix, value) =
                        expand_expr(&init.expr, fns, stack, fresh)?;
                    out.extend(prefix);
                    init.expr = Box::new(value);
                }
                out.push(syn::Stmt::Local(local));
            }
            syn::Stmt::Expr(expr, semi) => {
                // A whole-statement call runs its body and drops the value;
                // any other expression expands the calls nested in it and
                // keeps its own shape.
                let (prefix, value) = expand_expr(expr, fns, stack, fresh)?;
                out.extend(prefix);
                if is_device_call(expr, fns) {
                    out.push(syn::Stmt::Expr(
                        syn::Expr::Tuple(syn::ExprTuple {
                            attrs: Vec::new(),
                            elems: syn::punctuated::Punctuated::new(),
                            paren_token: Default::default(),
                        }),
                        *semi,
                    ));
                } else {
                    out.push(syn::Stmt::Expr(value, *semi));
                }
            }
            other => out.push(other.clone()),
        }
    }
    Ok(out)
}

/// Expand the device-function calls inside one expression.
///
/// The returned statement list holds everything the value depends on, in
/// order, and the expression names the value they produced.
fn expand_expr(
    expr: &syn::Expr,
    fns: &HashMap<String, DeviceFn>,
    stack: &mut Vec<String>,
    fresh: &mut usize,
) -> syn::Result<(Vec<syn::Stmt>, syn::Expr)> {
    if is_device_call(expr, fns) {
        let syn::Expr::Call(call) = expr else {
            unreachable!("is_device_call matched a call");
        };
        let syn::Expr::Path(path) = call.func.as_ref() else {
            unreachable!("is_device_call matched a path");
        };
        let name = path
            .path
            .get_ident()
            .expect("is_device_call matched an ident")
            .to_string();
        if stack.iter().any(|frame| frame == &name) {
            let mut chain = stack.clone();
            chain.push(name.clone());
            return Err(syn::Error::new(
                call.span(),
                format!(
                    "device functions cannot call themselves: {}",
                    chain.join(" -> ")
                ),
            ));
        }
        // The arguments expand first, so `f(g(x))` binds g's result before
        // f's body runs.
        let mut prefix = Vec::new();
        let mut args = Vec::new();
        for arg in &call.args {
            let (arg_prefix, arg_value) = expand_expr(arg, fns, stack, fresh)?;
            prefix.extend(arg_prefix);
            args.push(arg_value);
        }
        stack.push(name.clone());
        let result = expand_call(&name, &fns[&name], &args, fns, stack, fresh);
        stack.pop();
        let (mut body, value) = result?;
        prefix.append(&mut body);
        return Ok((prefix, value));
    }
    // A call nested deeper — an argument, an operand, a return value —
    // expands the same way, and the statements it produces run before the
    // expression that reads it.
    let mut walker = NestedCalls {
        fns,
        stack,
        fresh,
        prefix: Vec::new(),
        error: None,
    };
    let mut value = expr.clone();
    walker.visit_expr_mut(&mut value);
    if let Some(err) = walker.error {
        return Err(err);
    }
    Ok((walker.prefix, value))
}

/// One expansion: the body with its parameters bound and its return turned
/// into the binding the call site reads.
fn expand_call(
    name: &str,
    device_fn: &DeviceFn,
    args: &[syn::Expr],
    fns: &HashMap<String, DeviceFn>,
    stack: &mut Vec<String>,
    fresh: &mut usize,
) -> syn::Result<(Vec<syn::Stmt>, syn::Expr)> {
    if args.len() != device_fn.params.len() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "`{name}` takes {} arguments, got {}",
                device_fn.params.len(),
                args.len()
            ),
        ));
    }
    *fresh += 1;
    let prefix = format!("{}_{}", name, fresh);
    let renames: HashMap<String, String> = device_fn
        .params
        .iter()
        .map(|param| (param.clone(), format!("{prefix}_{}", param)))
        .collect();
    let ret_name = syn::Ident::new(&format!("{prefix}_ret"), proc_macro2::Span::call_site());

    let body = inline_device_calls(&device_fn.body, fns, stack, fresh)?;
    let mut renamed = Renamer {
        renames: &renames,
    };
    let mut body_stmts = body;
    for stmt in body_stmts.iter_mut() {
        renamed.visit_stmt_mut(stmt);
    }

    let mut out = Vec::new();
    for (param, arg) in device_fn.params.iter().zip(args) {
        let fresh_name = syn::Ident::new(&renames[param], proc_macro2::Span::call_site());
        out.push(syn::parse_quote! {
            let mut #fresh_name = #arg;
        });
    }
    // A body ends one of two ways: a `return` names the value, or a trailing
    // expression does. Both become a binding of the same fresh name; a body
    // without a return type has neither and drops its trailing expression.
    for stmt in body_stmts {
        match stmt {
            syn::Stmt::Expr(syn::Expr::Return(ret), _) => {
                if let Some(value) = ret.expr {
                    out.push(local_stmt(ret_name.clone(), *value));
                }
            }
            other => out.push(other),
        }
    }
    if device_fn.returns && let Some(syn::Stmt::Expr(expr, None)) = out.pop() {
        out.push(local_stmt(ret_name.clone(), expr));
    }
    let value = if device_fn.returns {
        syn::Expr::Path(syn::ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: ret_name.into(),
        })
    } else {
        syn::Expr::Tuple(syn::ExprTuple {
            attrs: Vec::new(),
            elems: syn::punctuated::Punctuated::new(),
            paren_token: Default::default(),
        })
    };
    Ok((out, value))
}

fn local_stmt(name: syn::Ident, value: syn::Expr) -> syn::Stmt {
    syn::parse_quote! {
        let #name = #value;
    }
}

/// Renames every use of the parameter names, both where a statement reads one
/// and where the body rebinds one.
struct Renamer<'a> {
    renames: &'a HashMap<String, String>,
}

impl VisitMut for Renamer<'_> {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if let syn::Expr::Path(path) = expr
            && path.qself.is_none()
            && path.path.segments.len() == 1
        {
            let ident = &mut path.path.segments.first_mut().expect("one segment").ident;
            if let Some(new) = self.renames.get(&ident.to_string()) {
                *ident = syn::Ident::new(new, ident.span());
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }

    fn visit_pat_mut(&mut self, pat: &mut syn::Pat) {
        if let syn::Pat::Ident(ident) = pat
            && let Some(new) = self.renames.get(&ident.ident.to_string())
        {
            ident.ident = syn::Ident::new(new, ident.ident.span());
            return;
        }
        syn::visit_mut::visit_pat_mut(self, pat);
    }
}

/// Walks one expression and hoists every nested device call into statements
/// that run before it.
struct NestedCalls<'a> {
    fns: &'a HashMap<String, DeviceFn>,
    stack: &'a mut Vec<String>,
    fresh: &'a mut usize,
    prefix: Vec<syn::Stmt>,
    error: Option<syn::Error>,
}

impl VisitMut for NestedCalls<'_> {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if self.error.is_some() {
            return;
        }
        if is_device_call(expr, self.fns) {
            match expand_expr(expr, self.fns, self.stack, self.fresh) {
                Ok((mut stmts, value)) => {
                    self.prefix.append(&mut stmts);
                    *expr = value;
                }
                Err(err) => {
                    self.error = Some(err);
                }
            }
            return;
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }

    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        if self.error.is_some() {
            return;
        }
        match inline_device_calls(&block.stmts, self.fns, self.stack, self.fresh) {
            Ok(stmts) => block.stmts = stmts,
            Err(err) => {
                self.error = Some(err);
            }
        }
    }
}
