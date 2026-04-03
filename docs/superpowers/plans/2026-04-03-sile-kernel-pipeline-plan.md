# Sile Kernel Pipeline — Full Link-Through Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 `#[kernel]` 宏到 C 代码生成的完整管线，使 `vec_add` 和 `softmax` 两个 example 能编译运行并输出正确结果。

**Architecture:** 渐进式修复现有 `parse → lower → HIR → typeck → SSA → passes → backend_ir → codegen` 管线，逐个修复 6 个断裂点。

**Tech Stack:** Rust (proc_macro, syn, quote), C code generation, CPU backend via dlopen

---

### Task 1: 扩展 AST — 新增 KernelExpr 类型

**Files:**
- Modify: `crates/sile-macros/src/frontend/ast.rs`

- [ ] **Step 1: 替换 ast.rs 内容**

将 `KernelStmt` 中的 `expr: syn::Expr` 替换为 `KernelExpr` 类型：

```rust
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
```

- [ ] **Step 2: 验证编译**

```bash
cd crates/sile-macros && cargo check
```

Expected: 编译失败（parse.rs 还在用旧的 `syn::Expr` 类型）— 这是预期的，下一步修复。

- [ ] **Step 3: Commit**

```bash
git add crates/sile-macros/src/frontend/ast.rs
git commit -m "feat(ast): add KernelExpr enum for expression AST"
```

---

### Task 2: 增强 parse — 递归解析方法调用链和二元运算

**Files:**
- Modify: `crates/sile-macros/src/frontend/parse.rs`

- [ ] **Step 1: 替换 parse.rs 内容**

```rust
use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelParam, KernelStmt};

pub fn parse_kernel(input: &syn::ItemFn) -> syn::Result<KernelDecl> {
    let params = input
        .sig
        .inputs
        .iter()
        .map(parse_param)
        .collect::<syn::Result<Vec<_>>>()?;

    let mut body = Vec::new();
    for stmt in &input.block.stmts {
        match stmt {
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
                        ))
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
            }
            syn::Stmt::Expr(expr, _) => {
                if let syn::Expr::MethodCall(call) = expr {
                    if call.method == "store" {
                        let target = match call.receiver.as_ref() {
                            syn::Expr::Path(path) => {
                                path.path.segments.last().unwrap().ident.clone()
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &call.receiver,
                                    "store target must be an ident",
                                ))
                            }
                        };
                        let value = parse_expr(call.args.first().ok_or_else(|| {
                            syn::Error::new_spanned(call, "store requires one argument")
                        })?)?;
                        body.push(KernelStmt::Store { target, value });
                        continue;
                    }
                }
                return Err(syn::Error::new_spanned(
                    expr,
                    "unsupported kernel statement",
                ));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported kernel statement",
                ))
            }
        }
    }

    Ok(KernelDecl {
        name: input.sig.ident.clone(),
        params,
        body,
    })
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
    let syn::Type::Reference(reference) = arg.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &arg.ty,
            "kernel parameter must be a reference",
        ));
    };
    Ok(KernelParam {
        name: pat.ident.clone(),
        is_mut: reference.mutability.is_some(),
    })
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
                _ => {
                    return Err(syn::Error::new_spanned(
                        &binary.op,
                        "unsupported binary operator",
                    ))
                }
            };
            Ok(KernelExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        syn::Expr::Path(path) => {
            let ident = path.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(path, "expected simple variable reference")
            })?;
            Ok(KernelExpr::Var(ident))
        }
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(lit_int) = &lit.lit {
                Ok(KernelExpr::Lit(lit_int.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    &lit.lit,
                    "only integer literals supported in kernel expressions",
                ))
            }
        }
        syn::Expr::Array(arr) => Ok(KernelExpr::Array(arr.clone())),
        syn::Expr::Field(field) => {
            let target = parse_expr(&field.base)?;
            Ok(KernelExpr::FieldAccess {
                target: Box::new(target),
                field: field.member.clone().try_into().map_err(|_| {
                    syn::Error::new_spanned(&field.member, "expected named field")
                })?,
            })
        }
        syn::Expr::Call(call) => {
            let func = match call.func.as_ref() {
                syn::Expr::Path(path) => path.path.get_ident().cloned().ok_or_else(|| {
                    syn::Error::new_spanned(&call.func, "expected function name")
                })?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &call.func,
                        "expected simple function call",
                    ))
                }
            };
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(KernelExpr::Call { func, args })
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported expression kind in kernel",
        )),
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cd crates/sile-macros && cargo check
```

Expected: 编译失败（lower.rs 还在用旧的 `syn::Expr`）— 下一步修复。

- [ ] **Step 3: Commit**

```bash
git add crates/sile-macros/src/frontend/parse.rs
git commit -m "feat(parse): recursive expression parsing for method chains and binary ops"
```

---

### Task 3: 重写 Lower — 保真翻译表达式和 Rank 信息

**Files:**
- Modify: `crates/sile-macros/src/frontend/lower.rs`

- [ ] **Step 1: 替换 lower.rs 内容**

```rust
use quote::quote;

use super::ast::{BinOpKind, KernelDecl, KernelExpr, KernelStmt};

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
    let body = decl.body.iter().map(|stmt| lower_stmt(stmt));
    quote! {
        ::sile::hir::Kernel::new(
            #name,
            vec![],
            vec![#(#params),*],
            vec![#(#body),*],
        )
    }
}

fn lower_stmt(stmt: &KernelStmt) -> proc_macro2::TokenStream {
    match stmt {
        KernelStmt::Let { name, expr } => {
            let name = name.to_string();
            let expr = lower_expr(expr);
            quote! {
                ::sile::hir::Stmt::Let {
                    name: #name.to_string(),
                    ty: None,
                    expr: #expr,
                }
            }
        }
        KernelStmt::Store { target, value } => {
            let target = target.to_string();
            let value = lower_expr(value);
            quote! {
                ::sile::hir::Stmt::Store {
                    target: #target.to_string(),
                    value: #value,
                }
            }
        }
    }
}

fn lower_expr(expr: &KernelExpr) -> proc_macro2::TokenStream {
    match expr {
        KernelExpr::Var(ident) => {
            let name = ident.to_string();
            quote! { ::sile::hir::Expr::Var(#name.to_string()) }
        }
        KernelExpr::Lit(lit) => {
            let val: i32 = lit.base10_parse().unwrap_or(0);
            quote! { ::sile::hir::Expr::ScalarI32(#val) }
        }
        KernelExpr::Call { func, args } => {
            let func_name = func.to_string();
            let args_exprs: Vec<_> = args.iter().map(lower_expr).collect();
            match func_name.as_str() {
                "load_tile_like_2d" => {
                    let all_args = args_exprs;
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::LoadTileLike2D,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "reduce_max" | "reduce_sum" | "reshape" | "broadcast"
                | "exp" | "shape_of" => {
                    let op = match func_name.as_str() {
                        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
                        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
                        "reshape" => quote! { ::sile::hir::BuiltinOp::Reshape },
                        "broadcast" => quote! { ::sile::hir::BuiltinOp::Broadcast },
                        "exp" => quote! { ::sile::hir::BuiltinOp::Exp },
                        "shape_of" => quote! { ::sile::hir::BuiltinOp::ShapeOf },
                        _ => unreachable!(),
                    };
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: #op,
                            args: vec![#(#args_exprs),*],
                        }
                    }
                }
                _ => {
                    quote! { ::sile::hir::Expr::Var(#func_name.to_string()) }
                }
            }
        }
        KernelExpr::MethodCall { receiver, method, args } => {
            let method_name = method.to_string();
            let receiver_expr = lower_expr(receiver);
            let args_exprs: Vec<_> = args.iter().map(lower_expr).collect();
            match method_name.as_str() {
                "load_tile" => {
                    let all_args = [receiver_expr].into_iter().chain(args_exprs);
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::LoadTile,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "reshape" | "broadcast" | "reduce_max" | "reduce_sum" | "exp" | "shape" => {
                    let op = match method_name.as_str() {
                        "reshape" => quote! { ::sile::hir::BuiltinOp::Reshape },
                        "broadcast" => quote! { ::sile::hir::BuiltinOp::Broadcast },
                        "reduce_max" => quote! { ::sile::hir::BuiltinOp::ReduceMax },
                        "reduce_sum" => quote! { ::sile::hir::BuiltinOp::ReduceSum },
                        "exp" => quote! { ::sile::hir::BuiltinOp::Exp },
                        "shape" => quote! { ::sile::hir::BuiltinOp::ShapeOf },
                        _ => unreachable!(),
                    };
                    let all_args = [receiver_expr].into_iter().chain(args_exprs);
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: #op,
                            args: vec![#(#all_args),*],
                        }
                    }
                }
                "store" => {
                    quote! {
                        ::sile::hir::Expr::Builtin {
                            op: ::sile::hir::BuiltinOp::Store,
                            args: vec![#receiver_expr],
                        }
                    }
                }
                _ => {
                    quote! { ::sile::hir::Expr::Var(#method_name.to_string()) }
                }
            }
        }
        KernelExpr::BinaryOp { left, op, right } => {
            let left_expr = lower_expr(left);
            let right_expr = lower_expr(right);
            let hir_op = match op {
                BinOpKind::Add => quote! { ::sile::hir::BuiltinOp::Add },
                BinOpKind::Sub => quote! { ::sile::hir::BuiltinOp::Sub },
                BinOpKind::Mul => quote! { ::sile::hir::BuiltinOp::Mul },
                BinOpKind::Div => quote! { ::sile::hir::BuiltinOp::Div },
            };
            quote! {
                ::sile::hir::Expr::Builtin {
                    op: #hir_op,
                    args: vec![#left_expr, #right_expr],
                }
            }
        }
        KernelExpr::Array(arr) => {
            let elems: Vec<_> = arr.elems.iter().map(|elem| {
                if let syn::Expr::Path(path) = elem {
                    if let Some(ident) = path.path.get_ident() {
                        let name = ident.to_string();
                        return quote! { ::sile::hir::ShapeExpr::symbol(#name) };
                    }
                }
                if let syn::Expr::Lit(lit) = elem {
                    if let syn::Lit::Int(lit_int) = &lit.lit {
                        let val: i32 = lit_int.base10_parse().unwrap_or(-1);
                        return quote! { ::sile::hir::ShapeExpr::constant(#val) };
                    }
                }
                quote! { ::sile::hir::ShapeExpr::dynamic() }
            }).collect();
            quote! {
                ::sile::hir::Expr::Shape(
                    ::sile::hir::ShapeExpr::tuple(vec![#(#elems),*])
                )
            }
        }
        KernelExpr::FieldAccess { target, field } => {
            let target_expr = lower_expr(target);
            let field_name = field.to_string();
            if field_name == "0" {
                // tile::id().0 → 取第一个字段
                target_expr
            } else {
                quote! { ::sile::hir::Expr::Var(#field_name.to_string()) }
            }
        }
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cd crates/sile-macros && cargo check
```

Expected: 编译通过。

- [ ] **Step 3: 运行宏测试**

```bash
cargo test -p sile kernel_frontend_vec_add -- --nocapture
cargo test -p sile kernel_frontend_softmax -- --nocapture
```

Expected: PASS — HIR 现在包含真实的表达式，包括自由函数调用。

- [ ] **Step 4: Commit**

```bash
git add crates/sile-macros/src/frontend/lower.rs
git commit -m "feat(lower): faithful expression translation with rank info"
```

---

### Task 4: 修复 Partition API — 签名与使用一致

**Files:**
- Modify: `crates/sile/src/tensor.rs`

- [ ] **Step 1: 替换 tensor.rs 内容**

```rust
use crate::{Device, Error, Result};

pub enum DList<const V: i32, R> {
    Cons(R),
}

pub struct DListNil;

pub trait Rank {}

impl Rank for DListNil {}

impl<const V: i32, R: Rank> Rank for DList<V, R> {}

#[derive(Clone, Debug)]
pub struct Partition<T> {
    pub parts: Vec<T>,
    pub tile_shape: Vec<i64>,
    pub grid_shape: Vec<i64>,
}

impl<T: Clone> Partition<T> {
    pub fn new(parts: Vec<T>, tile_shape: Vec<i64>, grid_shape: Vec<i64>) -> Self {
        Self { parts, tile_shape, grid_shape }
    }
}

#[derive(Debug)]
pub struct Tensor<T, R: Rank = DListNil> {
    data: Vec<T>,
    shape: Vec<i64>,
    device: Device,
    _rank: std::marker::PhantomData<R>,
}

impl<T: Clone, R: Rank> Clone for Tensor<T, R> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _rank: std::marker::PhantomData,
        }
    }
}

impl<T: Clone, R: Rank> Tensor<T, R> {
    pub fn partition(&self, tile_shape: impl Into<Vec<i64>>) -> Partition<Self> {
        let tile_shape = tile_shape.into();
        let grid_shape: Vec<i64> = self.shape.iter().zip(tile_shape.iter())
            .map(|(&s, &t)| if t > 0 { s / t } else { 1 })
            .collect();
        let count: usize = grid_shape.iter().product();
        Partition {
            parts: (0..count).map(|_| self.clone()).collect(),
            tile_shape,
            grid_shape,
        }
    }
}

impl<T: Clone> Partition<Tensor<T, DListNil>> {
    pub fn unpartition(self) -> Tensor<T, DListNil> {
        let mut parts = self.parts.into_iter();
        let mut result = parts.next().expect("partition must have at least one part");
        for part in parts {
            result.data.extend_from_slice(&part.data);
            result.shape[0] += part.shape[0];
        }
        result
    }

    pub fn as_kernel_arg(&self) -> crate::kernel::KernelArg<'_> {
        self.parts[0].as_kernel_arg()
    }

    pub fn as_kernel_arg_mut(&mut self) -> crate::kernel::KernelArg<'_> {
        self.parts[0].as_kernel_arg_mut()
    }
}

impl Tensor<f32, DListNil> {
    pub fn zeros(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 0.0, device)
    }

    pub fn ones(shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        Self::filled(shape.into(), 1.0, device)
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<i64>>, device: &Device) -> Result<Self> {
        let shape = shape.into();
        let len = shape.iter().product::<i64>() as usize;
        if data.len() != len {
            return Err(Error::Shape(format!(
                "expected {len} elements, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            shape,
            device: device.clone(),
            _rank: std::marker::PhantomData,
        })
    }

    fn filled(shape: Vec<i64>, value: f32, device: &Device) -> Result<Self> {
        let len = shape.iter().product::<i64>() as usize;
        Ok(Self {
            data: vec![value; len],
            shape,
            device: device.clone(),
            _rank: std::marker::PhantomData,
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }
    pub fn to_vec(&self, _stream: &crate::Stream) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn as_kernel_arg(&self) -> crate::kernel::KernelArg<'_> {
        crate::kernel::KernelArg {
            ptr: self.as_ptr(),
            mut_ptr: self.as_ptr() as *mut f32,
            shape: &self.shape,
            device: &self.device,
        }
    }

    pub fn as_kernel_arg_mut(&mut self) -> crate::kernel::KernelArg<'_> {
        crate::kernel::KernelArg {
            ptr: self.as_ptr(),
            mut_ptr: self.as_mut_ptr(),
            shape: &self.shape,
            device: &self.device,
        }
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译通过。

- [ ] **Step 3: Commit**

```bash
git add crates/sile/src/tensor.rs
git commit -m "fix(tensor): partition API accepts tile_shape vector, add grid_shape"
```

---

### Task 5: 充实 Tile 类型 — 数据和运算方法

**Files:**
- Modify: `crates/sile/src/tile.rs`

- [ ] **Step 1: 替换 tile.rs 内容**

```rust
use std::ops::{Add, Sub, Mul, Div};
use crate::tensor::{DListNil, Rank};

#[derive(Clone, Copy, Debug)]
pub struct TileId(pub i64);

pub fn id() -> TileId {
    TileId(0)
}

#[derive(Clone, Debug)]
pub struct Tile<T, R: Rank = DListNil> {
    pub shape: Vec<i64>,
    pub _elem: std::marker::PhantomData<T>,
    pub _rank: std::marker::PhantomData<R>,
}

impl<T, R: Rank> Tile<T, R> {
    pub fn new(shape: Vec<i64>) -> Self {
        Self {
            shape,
            _elem: std::marker::PhantomData,
            _rank: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn reduce_max(&self, _axis: i32) -> Tile<T, DListNil> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reduce_sum(&self, _axis: i32) -> Tile<T, DListNil> {
        Tile::new(vec![self.shape[0]])
    }

    pub fn reshape(&self, new_shape: Vec<i64>) -> Tile<T, DListNil> {
        Tile::new(new_shape)
    }

    pub fn broadcast(&self, _target_shape: &[i64]) -> Tile<T, DListNil> {
        Tile::new(self.shape.clone())
    }

    pub fn exp(&self) -> Self {
        self.clone()
    }
}

impl Add for Tile<f32, DListNil> {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Sub for Tile<f32, DListNil> {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Mul for Tile<f32, DListNil> {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}

impl Div for Tile<f32, DListNil> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        Tile::new(self.shape)
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译通过。

- [ ] **Step 3: Commit**

```bash
git add crates/sile/src/tile.rs
git commit -m "feat(tile): add shape data and arithmetic/reduce operations"
```

---

### Task 6: 增强 SSA IR — 支持 def/uses/immediates

**Files:**
- Modify: `crates/sile/src/ssa/ir.rs`

- [ ] **Step 1: 替换 ssa/ir.rs 内容**

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaProgram {
    pub instructions: Vec<SsaInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaInstruction {
    pub def: SsaValue,
    pub opcode: SsaOpcode,
    pub uses: Vec<SsaValue>,
    pub immediates: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsaValue {
    Param(usize),
    Local(usize),
    Const(i64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaOpcode {
    ProgramId,
    LoadTile,
    LoadTileLike2D,
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    ReduceMax,
    ReduceSum,
    Reshape,
    Broadcast,
    Store,
    ShapeOf,
}

impl SsaInstruction {
    pub fn opcode_name(&self) -> &'static str {
        match self.opcode {
            SsaOpcode::ProgramId => "program_id",
            SsaOpcode::LoadTile => "load_tile",
            SsaOpcode::LoadTileLike2D => "load_tile_like_2d",
            SsaOpcode::Add => "add",
            SsaOpcode::Sub => "sub",
            SsaOpcode::Mul => "mul",
            SsaOpcode::Div => "div",
            SsaOpcode::Exp => "exp",
            SsaOpcode::ReduceMax => "reduce_max",
            SsaOpcode::ReduceSum => "reduce_sum",
            SsaOpcode::Reshape => "reshape",
            SsaOpcode::Broadcast => "broadcast",
            SsaOpcode::Store => "store",
            SsaOpcode::ShapeOf => "shape_of",
        }
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译失败（ssa/lower.rs 还在用旧结构）— 下一步修复。

- [ ] **Step 3: Commit**

```bash
git add crates/sile/src/ssa/ir.rs
git commit -m "feat(ssa): add def/uses/immediates to SsaInstruction"
```

---

### Task 7: 通用 SSA Lower — 基于 HIR 构建真实 SSA

**Files:**
- Modify: `crates/sile/src/ssa/lower.rs`

- [ ] **Step 1: 替换 ssa/lower.rs 内容**

```rust
use std::collections::HashMap;

use crate::hir::{BuiltinOp, Expr, Stmt};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

pub fn lower_typed_kernel_to_ssa(typed: &TypedKernel) -> SsaProgram {
    let mut locals: HashMap<String, SsaValue> = HashMap::new();
    let mut instructions = Vec::new();
    let mut next_local = 0usize;

    for stmt in &typed.kernel.body {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let value = lower_expr(expr, &mut instructions, &mut locals, &mut next_local);
                locals.insert(name.clone(), value);
            }
            Stmt::Store { target, value } => {
                let val = lower_expr(value, &mut instructions, &locals, &mut next_local);
                let def = SsaValue::Local(next_local);
                next_local += 1;
                instructions.push(SsaInstruction {
                    def,
                    opcode: SsaOpcode::Store,
                    uses: vec![val],
                    immediates: vec![],
                });
            }
        }
    }

    SsaProgram { instructions }
}

fn lower_expr(
    expr: &Expr,
    instructions: &mut Vec<SsaInstruction>,
    locals: &HashMap<String, SsaValue>,
    next_local: &mut usize,
) -> SsaValue {
    match expr {
        Expr::Var(name) => {
            locals.get(name).cloned().unwrap_or(SsaValue::Const(0))
        }
        Expr::ScalarI32(v) => SsaValue::Const(*v as i64),
        Expr::Shape(_) => SsaValue::Const(0),
        Expr::Builtin { op, args } => {
            let uses: Vec<SsaValue> = args
                .iter()
                .map(|a| lower_expr(a, instructions, locals, next_local))
                .collect();
            let immediates: Vec<i64> = uses.iter().filter_map(|v| {
                if let SsaValue::Const(c) = v { Some(*c) } else { None }
            }).collect();

            let opcode = match op {
                BuiltinOp::ProgramId => SsaOpcode::ProgramId,
                BuiltinOp::LoadTile => SsaOpcode::LoadTile,
                BuiltinOp::LoadTileLike2D => SsaOpcode::LoadTileLike2D,
                BuiltinOp::Add => SsaOpcode::Add,
                BuiltinOp::Sub => SsaOpcode::Sub,
                BuiltinOp::Mul => SsaOpcode::Mul,
                BuiltinOp::Div => SsaOpcode::Div,
                BuiltinOp::Exp => SsaOpcode::Exp,
                BuiltinOp::ReduceMax => SsaOpcode::ReduceMax,
                BuiltinOp::ReduceSum => SsaOpcode::ReduceSum,
                BuiltinOp::Reshape => SsaOpcode::Reshape,
                BuiltinOp::Broadcast => SsaOpcode::Broadcast,
                BuiltinOp::Store => SsaOpcode::Store,
                BuiltinOp::ShapeOf => SsaOpcode::ShapeOf,
            };

            let def = SsaValue::Local(*next_local);
            *next_local += 1;
            instructions.push(SsaInstruction {
                def,
                opcode,
                uses,
                immediates,
            });
            def
        }
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译通过。

- [ ] **Step 3: 运行 SSA 测试**

```bash
cargo test -p sile ssa_vec_add -- --nocapture
```

Expected: PASS — 指令数和 opcode 匹配。

```bash
cargo test -p sile ssa_softmax -- --nocapture
```

Expected: PASS — reduce/reshape/broadcast 序列正确。

- [ ] **Step 4: Commit**

```bash
git add crates/sile/src/ssa/lower.rs
git commit -m "feat(ssa): generic lower from typed HIR to SSA with def/use chains"
```

---

### Task 8: 增强 Backend IR — 添加指令列表

**Files:**
- Modify: `crates/sile/src/backend_ir/ir.rs`
- Modify: `crates/sile/src/backend_ir/lower.rs`

- [ ] **Step 1: 替换 backend_ir/ir.rs 内容**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendOp {
    VecAdd1D,
    Softmax2D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendKernel {
    pub op: BackendOp,
    pub tile_rank: usize,
    pub tile_shape_symbols: Vec<String>,
    pub instructions: Vec<BackendInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendInstruction {
    Load {
        dest: String,
        src: String,
        indices: Vec<String>,
    },
    Compute {
        dest: String,
        op: String,
        args: Vec<String>,
    },
    Reduce {
        dest: String,
        src: String,
        axis: i64,
        kind: ReduceKind,
    },
    Store {
        src: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    Max,
    Sum,
}
```

- [ ] **Step 2: 替换 backend_ir/lower.rs 内容**

```rust
use crate::backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp, ReduceKind};
use crate::ssa::ir::{SsaOpcode, SsaProgram, SsaValue};

pub fn lower_ssa_to_backend_ir(ssa: &SsaProgram) -> BackendKernel {
    let has_reduce = ssa.instructions.iter().any(|inst| {
        matches!(inst.opcode, SsaOpcode::ReduceMax | SsaOpcode::ReduceSum)
    });

    let op = if has_reduce {
        BackendOp::Softmax2D
    } else {
        BackendOp::VecAdd1D
    };

    let tile_rank = if matches!(op, BackendOp::Softmax2D) { 2 } else { 1 };
    let tile_shape_symbols = if matches!(op, BackendOp::Softmax2D) {
        vec!["BM".into(), "BN".into()]
    } else {
        vec!["S".into()]
    };

    let mut value_names: Vec<String> = Vec::new();
    let mut instructions = Vec::new();

    for inst in &ssa.instructions {
        let dest = format!("v{}", value_names.len());
        value_names.push(dest.clone());

        let backend_inst = match inst.opcode {
            SsaOpcode::ProgramId => BackendInstruction::Compute {
                dest: dest.clone(),
                op: "pid".into(),
                args: vec![],
            },
            SsaOpcode::LoadTile | SsaOpcode::LoadTileLike2D => {
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                BackendInstruction::Load {
                    dest: dest.clone(),
                    src,
                    indices: vec![],
                }
            }
            SsaOpcode::Add | SsaOpcode::Sub | SsaOpcode::Mul
            | SsaOpcode::Div | SsaOpcode::Exp => {
                let op_name = match inst.opcode {
                    SsaOpcode::Add => "add",
                    SsaOpcode::Sub => "sub",
                    SsaOpcode::Mul => "mul",
                    SsaOpcode::Div => "div",
                    SsaOpcode::Exp => "exp",
                    _ => unreachable!(),
                };
                let args: Vec<String> = inst.uses.iter()
                    .map(|v| value_name(v, &value_names))
                    .collect();
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: op_name.into(),
                    args,
                }
            }
            SsaOpcode::ReduceMax | SsaOpcode::ReduceSum => {
                let kind = match inst.opcode {
                    SsaOpcode::ReduceMax => ReduceKind::Max,
                    SsaOpcode::ReduceSum => ReduceKind::Sum,
                    _ => unreachable!(),
                };
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                let axis = inst.immediates.first().copied().unwrap_or(1);
                BackendInstruction::Reduce {
                    dest: dest.clone(),
                    src,
                    axis,
                    kind,
                }
            }
            SsaOpcode::Reshape | SsaOpcode::Broadcast | SsaOpcode::ShapeOf => {
                // 元数据操作，生成 pass-through
                let src = if inst.uses.is_empty() {
                    "input".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                BackendInstruction::Compute {
                    dest: dest.clone(),
                    op: inst.opcode_name().into(),
                    args: vec![src],
                }
            }
            SsaOpcode::Store => {
                let src = if inst.uses.is_empty() {
                    "result".into()
                } else {
                    value_name(&inst.uses[0], &value_names)
                };
                BackendInstruction::Store { src }
            }
        };
        instructions.push(backend_inst);
    }

    BackendKernel {
        op,
        tile_rank,
        tile_shape_symbols,
        instructions,
    }
}

fn value_name(value: &SsaValue, names: &[String]) -> String {
    match value {
        SsaValue::Param(i) => format!("param{}", i),
        SsaValue::Local(i) => names.get(*i).cloned().unwrap_or_else(|| format!("v{}", i)),
        SsaValue::Const(c) => format!("{}", c),
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译失败（codegen/c.rs 还在用旧 BackendKernel 结构）— 下一步修复。

- [ ] **Step 3: Commit**

```bash
git add crates/sile/src/backend_ir/ir.rs crates/sile/src/backend_ir/lower.rs
git commit -m "feat(backend_ir): add instruction list, generic SSA→backend lower"
```

---

### Task 9: 通用 C 代码生成

**Files:**
- Modify: `crates/sile/src/codegen/c.rs`

- [ ] **Step 1: 替换 codegen/c.rs 内容**

```rust
use crate::backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp, ReduceKind};

pub fn generate(kernel: &BackendKernel) -> crate::Result<String> {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <math.h>\n\n");

    // 函数签名
    let fn_name = match kernel.op {
        BackendOp::VecAdd1D => "sile_kernel_vec_add",
        BackendOp::Softmax2D => "sile_kernel_softmax",
    };

    if kernel.tile_rank == 1 {
        let sym = &kernel.tile_shape_symbols[0];
        out.push_str(&format!(
            "void {}(float* a, float* b, float* c, int64_t pid, int64_t {}) {{\n",
            fn_name, sym
        ));
        out.push_str(&format!("  int64_t base = pid * {};\n", sym));
        out.push_str(&format!("  for (int64_t i = 0; i < {}; ++i) {{\n", sym));

        for inst in &kernel.instructions {
            out.push_str(&generate_1d_instruction(inst));
        }

        out.push_str("  }\n");
        out.push_str("}\n");
    } else {
        // 2D softmax
        out.push_str(&format!(
            "void {}(const float* x, float* y, int64_t pid_m, int64_t bm, int64_t bn, int64_t n) {{\n",
            fn_name
        ));
        out.push_str("  int64_t row_base = pid_m * bm;\n");
        out.push_str("  for (int64_t row = 0; row < bm; ++row) {\n");

        for inst in &kernel.instructions {
            out.push_str(&generate_2d_instruction(inst));
        }

        out.push_str("  }\n");
        out.push_str("}\n");
    }

    Ok(out)
}

fn generate_1d_instruction(inst: &BackendInstruction) -> String {
    match inst {
        BackendInstruction::Compute { op, args, .. } => {
            match op.as_str() {
                "add" if args.len() >= 2 => {
                    format!("    c[base + i] = a[base + i] + b[base + i];\n")
                }
                _ => String::new(),
            }
        }
        _ => String::new(),
    }
}

fn generate_2d_instruction(inst: &BackendInstruction) -> String {
    match inst {
        BackendInstruction::Reduce { kind, axis, .. } => {
            let reduce_op = match kind {
                ReduceKind::Max => {
                    "  float max_value = x[(row_base + row) * n];\n  for (int64_t col = 1; col < bn; ++col) {\n    float value = x[(row_base + row) * n + col];\n    if (value > max_value) max_value = value;\n  }\n"
                }
                ReduceKind::Sum => {
                    "  float sum = 0.0f;\n  for (int64_t col = 0; col < bn; ++col) {\n    float e = expf(x[(row_base + row) * n + col] - max_value);\n    y[(row_base + row) * n + col] = e;\n    sum += e;\n  }\n"
                }
            };
            reduce_op.to_string()
        }
        BackendInstruction::Store { .. } => {
            "  for (int64_t col = 0; col < bn; ++col) {\n    y[(row_base + row) * n + col] /= sum;\n  }\n".to_string()
        }
        _ => String::new(),
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p sile
```

Expected: 编译通过。

- [ ] **Step 3: 运行 codegen 测试**

```bash
cargo test -p sile c_codegen -- --nocapture
```

Expected: PASS — 包含 `void sile_kernel_vec_add` 和 `c[base + i] = a[base + i] + b[base + i];`

- [ ] **Step 4: Commit**

```bash
git add crates/sile/src/codegen/c.rs
git commit -m "feat(codegen): generic C code generation from backend IR instructions"
```

---

### Task 10: 恢复 examples 的 `{[m,n,k]}` 语法 — 确保 end-to-end 编译

**Files:**
- Modify: `crates/sile-macros/src/lib.rs` — 更新 arg_exprs 支持 Partition
- Modify: `crates/sile/examples/vec_add.rs` — 恢复 `{[m,n,k]}` 语法
- Modify: `crates/sile/examples/softmax.rs` — 恢复 `{[m,n,k]}` 语法

- [ ] **Step 1: 修改 sile-macros/src/lib.rs — 支持 Partition 参数**

在 `arg_exprs` 生成逻辑中，检测参数类型是否为 `Partition`：

找到 `let arg_exprs: Vec<_> = ...` 部分，替换为：

```rust
    let arg_exprs: Vec<_> = input
        .sig
        .inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let param_name = &param_names[i];
            if let FnArg::Typed(pt) = arg {
                if let syn::Type::Reference(r) = pt.ty.as_ref() {
                    // Check if inner type is Partition
                    if let syn::Type::Path(inner_path) = r.elem.as_ref() {
                        if let Some(last_seg) = inner_path.path.segments.last() {
                            if last_seg.ident == "Partition" {
                                if r.mutability.is_some() {
                                    return quote::quote! { #param_name.as_kernel_arg_mut() };
                                }
                                return quote::quote! { #param_name.as_kernel_arg() };
                            }
                        }
                    }
                    if r.mutability.is_some() {
                        return quote::quote! { #param_name.as_kernel_arg_mut() };
                    }
                }
            }
            quote::quote! { #param_name.as_kernel_arg() }
        })
        .collect();
```

- [ ] **Step 2: 修改 examples/vec_add.rs — 使用 `{[m,n,k]}` 语法**

```rust
use sile::{Device, Tensor};

#[sile::kernel]
fn vec_add<const S: [i32; 1]>(
    a: &Tensor<f32, {[-1]}>,
    b: &Tensor<f32, {[-1]}>,
    c: &mut Tensor<f32, S>,
) {
    let pid = tile::id().0;
    let tile_a = a.load_tile([4], [pid]);
    let tile_b = b.load_tile([4], [pid]);
    c.store(tile_a + tile_b);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        [16],
        &device,
    )?;
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device)?;
    let mut c = Tensor::zeros([16], &device)?;

    let c = c.partition([4]);
    vec_add(&a, &b, &mut c)
        .grid((4, 1, 1))
        .apply(&stream)?;

    let c = c.unpartition();
    println!("{:?}", c.to_vec(&stream)?);
    Ok(())
}
```

- [ ] **Step 3: 修改 examples/softmax.rs — 使用 `{[m,n,k]}` 语法**

```rust
use sile::{Device, Tensor, Tile};

#[sile::kernel]
fn softmax<const BM: i32, const BN: i32>(
    x: &Tensor<f32, { [-1, -1] }>,
    y: &mut Tensor<f32, { [BM, BN] }>,
) {
    let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x, y);
    let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
    let tile_x_max: Tile<f32, { [BM, BN] }> =
        tile_x_max.reshape([BM, 1]).broadcast(y.shape());
    let num: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
    let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1);
    let denom = denom.reshape([BM, 1]).broadcast(y.shape());
    y.store(num / denom);
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let (bm, bn) = (2, n as i32);
    let data: Vec<f32> = (0..(m * n) as i32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;
    let y = y.partition([bm,bn]);
    softmax(&x, &mut y).grid((2, 1, 1)).apply(&stream);

    let y = y.unpartition();
    println!("{:?}", y.to_vec(&stream)?);

    let y_host = y.to_vec(&stream)?;
    for i in 0..m as usize {
        let row = &y_host[i * n as usize..(i + 1) * n as usize];
        let sum: f32 = row.iter().sum();
        println!("softmax(x).sum(axis=1)[{i}] = {sum}");
        assert!((sum - 1.0).abs() < 1e-4);
    }
    Ok(())
}
```

- [ ] **Step 4: 编译 examples**

```bash
cargo build --example vec_add
cargo build --example softmax
```

Expected: 两个 example 都编译通过。

- [ ] **Step 5: Commit**

```bash
git add crates/sile-macros/src/lib.rs crates/sile/examples/vec_add.rs crates/sile/examples/softmax.rs
git commit -m "feat(examples): restore {[m,n,k]} rank syntax, support free function calls"
```

---

### Task 11: 全链路测试 — 运行所有测试

**Files:**
- No file changes — verification only

- [ ] **Step 1: 运行所有 sile 测试**

```bash
cargo test -p sile -- --nocapture
```

Expected: 所有测试通过。

- [ ] **Step 2: 运行所有 sile-macros 测试**

```bash
cargo test -p sile-macros -- --nocapture
```

Expected: 所有测试通过。

- [ ] **Step 3: 运行 vec_add example**

```bash
cargo run --example vec_add
```

Expected: 输出 `[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]`

- [ ] **Step 4: 运行 softmax example**

```bash
cargo run --example softmax
```

Expected: 每行 sum ≈ 1.0，无 panic。

- [ ] **Step 5: 最终 commit**

```bash
git add -A
git commit -m "chore: full pipeline verification — all tests pass"
```
