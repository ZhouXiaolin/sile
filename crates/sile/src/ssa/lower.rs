use std::collections::HashMap;

use crate::hir::{BuiltinOp, Expr, ShapeExpr, Stmt, Type};
use crate::ssa::ir::{SsaInstruction, SsaOpcode, SsaProgram, SsaValue};
use crate::typeck::TypedKernel;

struct LowerCtx<'a> {
    typed: &'a TypedKernel,
    const_values: HashMap<String, i64>,
    locals: HashMap<String, SsaValue>,
    defs: HashMap<usize, SsaInstruction>,
    tile_shapes: HashMap<usize, Vec<i64>>,
    instructions: Vec<SsaInstruction>,
    next_local: usize,
}

pub fn lower_typed_kernel_to_ssa(typed: &TypedKernel) -> SsaProgram {
    let mut ctx = LowerCtx {
        typed,
        const_values: typed.kernel.const_params.iter().cloned().collect(),
        locals: HashMap::new(),
        defs: HashMap::new(),
        tile_shapes: HashMap::new(),
        instructions: Vec::new(),
        next_local: 0,
    };

    for (i, param) in typed.kernel.params.iter().enumerate() {
        ctx.locals.insert(param.name.clone(), SsaValue::Param(i));
    }

    for stmt in &typed.kernel.body {
        lower_stmt(stmt, &mut ctx);
    }

    SsaProgram {
        instructions: ctx.instructions,
    }
}

fn lower_stmt(stmt: &Stmt, ctx: &mut LowerCtx<'_>) {
    match stmt {
        Stmt::Let { name, expr, .. } | Stmt::Assign { name, expr } => {
            let value = lower_expr(expr, ctx);
            ctx.locals.insert(name.clone(), value);
        }
        Stmt::Store { target, value } => lower_store(target, value, ctx),
        Stmt::ForLoop {
            var,
            start,
            end,
            body,
        } => {
            let start_i64 = eval_i64(start, ctx);
            let end_i64 = eval_i64(end, ctx);
            for i in start_i64..end_i64 {
                ctx.locals.insert(var.clone(), SsaValue::Const(i));
                for inner in body {
                    lower_stmt(inner, ctx);
                }
            }
        }
    }
}

fn lower_store(target: &str, value: &Expr, ctx: &mut LowerCtx<'_>) {
    let output = ctx
        .locals
        .get(target)
        .copied()
        .unwrap_or(SsaValue::Const(0));
    let stored = lower_expr(value, ctx);
    let mut uses = vec![output, stored];
    let mut immediates = Vec::new();

    if let Some(shape) = tile_shape_of_value(stored, ctx) {
        match shape.as_slice() {
            [cols] => {
                uses.push(SsaValue::Const(0));
                uses.push(ensure_program_dim(ctx, 0));
                immediates.extend([2, 1, *cols, 0]);
            }
            [rows, cols] => {
                uses.push(ensure_program_dim(ctx, 0));
                uses.push(ensure_program_dim(ctx, 1));
                immediates.extend([2, *rows, *cols, 1]);
            }
            _ => {}
        }
    } else if let Some([rows, cols]) = param_shape_2d(target, ctx) {
        uses.push(ensure_program_dim(ctx, 0));
        uses.push(ensure_program_dim(ctx, 1));
        immediates.extend([2, rows, cols, 1]);
    }

    emit_instruction(ctx, SsaOpcode::Store, uses, immediates);
}

fn lower_expr(expr: &Expr, ctx: &mut LowerCtx<'_>) -> SsaValue {
    match expr {
        Expr::Var(name) => ctx
            .locals
            .get(name)
            .copied()
            .or_else(|| ctx.const_values.get(name).copied().map(SsaValue::Const))
            .unwrap_or(SsaValue::Const(0)),
        Expr::ScalarI64(v) => SsaValue::Const(*v),
        Expr::ScalarF32(v) => SsaValue::Const(v.to_bits() as i64),
        Expr::Shape(_) => SsaValue::Const(0),
        Expr::Builtin { op, args } => match op {
            BuiltinOp::ProgramId => emit_instruction(ctx, SsaOpcode::ProgramId, vec![], vec![]),
            BuiltinOp::ShapeDim => {
                let base = lower_expr(&args[0], ctx);
                let dim = eval_i64(&args[1], ctx);
                emit_instruction(ctx, SsaOpcode::ShapeDim, vec![base], vec![dim])
            }
            BuiltinOp::Constant => {
                let value_bits = match args.first() {
                    Some(Expr::ScalarF32(v)) => v.to_bits() as i64,
                    Some(Expr::ScalarI64(v)) => (*v as f32).to_bits() as i64,
                    Some(other) => eval_i64(other, ctx) as u32 as i64,
                    None => 0,
                };
                let shape = args
                    .get(1)
                    .map(|arg| extract_const_shape(arg, ctx))
                    .unwrap_or_default();
                let value = emit_instruction(ctx, SsaOpcode::Constant, vec![], {
                    let mut immediates = vec![value_bits];
                    immediates.extend(shape);
                    immediates
                });
                record_tile_shape(&value, args.get(1).map(|arg| extract_const_shape(arg, ctx)).unwrap_or_default(), ctx);
                value
            }
            BuiltinOp::LoadTile => {
                let base = lower_expr(&args[0], ctx);
                let tile_shape = args
                    .get(1)
                    .map(|arg| extract_const_shape(arg, ctx))
                    .unwrap_or_default();
                let coords = args
                    .get(2)
                    .map(|arg| extract_runtime_shape(arg, ctx))
                    .unwrap_or_default();

                let (uses, immediates) = match (tile_shape.as_slice(), coords.as_slice()) {
                    ([cols], [col_tile]) => (
                        vec![base, SsaValue::Const(0), *col_tile],
                        vec![2, 1, *cols, 0],
                    ),
                    ([rows, cols], [row_tile, col_tile]) => (
                        vec![base, *row_tile, *col_tile],
                        vec![2, *rows, *cols, 1],
                    ),
                    _ => {
                        let mut uses = vec![base];
                        uses.extend(coords);
                        let mut immediates = vec![tile_shape.len() as i64];
                        immediates.extend(tile_shape.clone());
                        (uses, immediates)
                    }
                };
                let value = emit_instruction(ctx, SsaOpcode::LoadTile, uses, immediates);
                record_tile_shape(&value, tile_shape, ctx);
                value
            }
            BuiltinOp::LoadTileLike2D => {
                let input = lower_expr(&args[0], ctx);
                let shape = args
                    .get(1)
                    .and_then(|arg| expr_shape(arg, ctx))
                    .unwrap_or_default();
                let row_tile = ensure_program_dim(ctx, 0);
                let col_tile = ensure_program_dim(ctx, 1);
                let value = emit_instruction(
                    ctx,
                    SsaOpcode::LoadTileLike2D,
                    vec![input, row_tile, col_tile],
                    match shape.as_slice() {
                        [rows, cols] => vec![2, *rows, *cols, 1],
                        _ => vec![],
                    },
                );
                record_tile_shape(&value, shape, ctx);
                value
            }
            BuiltinOp::Reshape | BuiltinOp::Broadcast => {
                let input = lower_expr(&args[0], ctx);
                let shape = args
                    .get(1)
                    .and_then(|expr| expr_shape(expr, ctx))
                    .unwrap_or_default();
                let opcode = match op {
                    BuiltinOp::Reshape => SsaOpcode::Reshape,
                    BuiltinOp::Broadcast => SsaOpcode::Broadcast,
                    _ => unreachable!(),
                };
                let value = emit_instruction(ctx, opcode, vec![input], shape.clone());
                record_tile_shape(&value, shape, ctx);
                value
            }
            BuiltinOp::Mma => {
                let uses: Vec<SsaValue> = args.iter().map(|arg| lower_expr(arg, ctx)).collect();
                let acc_shape = uses
                    .get(2)
                    .and_then(|value| tile_shape_of_value(*value, ctx))
                    .unwrap_or_else(|| vec![1, 1]);
                let lhs_shape = uses
                    .first()
                    .and_then(|value| tile_shape_of_value(*value, ctx))
                    .unwrap_or_else(|| vec![1, 1]);
                let tile_m = acc_shape.first().copied().unwrap_or(1);
                let tile_n = acc_shape.get(1).copied().unwrap_or(1);
                let tile_k = lhs_shape.get(1).copied().unwrap_or(1);
                let value = emit_instruction(ctx, SsaOpcode::Mma, uses, vec![tile_m, tile_n, tile_k]);
                record_tile_shape(&value, vec![tile_m, tile_n], ctx);
                value
            }
            other => {
                let uses: Vec<SsaValue> = args.iter().map(|arg| lower_expr(arg, ctx)).collect();
                let immediates: Vec<i64> = uses
                    .iter()
                    .filter_map(|value| match value {
                        SsaValue::Const(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                let opcode = match other {
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
                    BuiltinOp::ScalarDiv => SsaOpcode::ScalarDiv,
                    BuiltinOp::ProgramId
                    | BuiltinOp::LoadTile
                    | BuiltinOp::LoadTileLike2D
                    | BuiltinOp::Mma
                    | BuiltinOp::Constant
                    | BuiltinOp::ShapeDim => unreachable!(),
                };
                let value = emit_instruction(ctx, opcode, uses.clone(), immediates);
                if let Some(shape) = infer_result_tile_shape(*other, &uses, args, ctx) {
                    record_tile_shape(&value, shape, ctx);
                }
                value
            }
        },
    }
}

fn emit_instruction(
    ctx: &mut LowerCtx<'_>,
    opcode: SsaOpcode,
    uses: Vec<SsaValue>,
    immediates: Vec<i64>,
) -> SsaValue {
    let def = SsaValue::Local(ctx.next_local);
    ctx.next_local += 1;
    let inst = SsaInstruction {
        def,
        opcode,
        uses,
        immediates,
    };
    if let SsaValue::Local(idx) = def {
        ctx.defs.insert(idx, inst.clone());
    }
    ctx.instructions.push(inst);
    def
}

fn eval_i64(expr: &Expr, ctx: &LowerCtx<'_>) -> i64 {
    match expr {
        Expr::ScalarI64(v) => *v,
        Expr::ScalarF32(v) => *v as i64,
        Expr::Var(name) => ctx
            .locals
            .get(name)
            .and_then(|value| match value {
                SsaValue::Const(v) => Some(*v),
                _ => None,
            })
            .or_else(|| ctx.const_values.get(name).copied())
            .unwrap_or(0),
        Expr::Builtin { op, args } if *op == BuiltinOp::ShapeDim => args
            .get(1)
            .map(|arg| eval_i64(arg, ctx))
            .unwrap_or(0),
        _ => 0,
    }
}

fn extract_const_shape(expr: &Expr, ctx: &LowerCtx<'_>) -> Vec<i64> {
    match expr {
        Expr::Shape(ShapeExpr::Tuple(dims)) => {
            dims.iter().map(|dim| resolve_const_shape_dim(dim, ctx)).collect()
        }
        Expr::Shape(dim) => vec![resolve_const_shape_dim(dim, ctx)],
        _ => vec![],
    }
}

fn extract_runtime_shape(expr: &Expr, ctx: &LowerCtx<'_>) -> Vec<SsaValue> {
    match expr {
        Expr::Shape(ShapeExpr::Tuple(dims)) => {
            dims.iter().map(|dim| resolve_runtime_shape_dim(dim, ctx)).collect()
        }
        Expr::Shape(dim) => vec![resolve_runtime_shape_dim(dim, ctx)],
        _ => vec![],
    }
}

fn resolve_const_shape_dim(dim: &ShapeExpr, ctx: &LowerCtx<'_>) -> i64 {
    match dim {
        ShapeExpr::Dynamic => -1,
        ShapeExpr::Constant(v) => *v,
        ShapeExpr::Symbol(name) => ctx.const_values.get(name).copied().unwrap_or(-1),
        ShapeExpr::Tuple(_) => -1,
    }
}

fn resolve_runtime_shape_dim(dim: &ShapeExpr, ctx: &LowerCtx<'_>) -> SsaValue {
    match dim {
        ShapeExpr::Dynamic => SsaValue::Const(-1),
        ShapeExpr::Constant(v) => SsaValue::Const(*v),
        ShapeExpr::Symbol(name) => ctx
            .locals
            .get(name)
            .copied()
            .or_else(|| ctx.const_values.get(name).copied().map(SsaValue::Const))
            .unwrap_or(SsaValue::Const(0)),
        ShapeExpr::Tuple(_) => SsaValue::Const(0),
    }
}

fn param_shape_2d(target: &str, ctx: &LowerCtx<'_>) -> Option<[i64; 2]> {
    let param = ctx.typed.kernel.params.iter().find(|param| param.name == target)?;
    let Type::Tensor { shape, .. } = &param.ty else {
        return None;
    };
    let dims = match shape {
        ShapeExpr::Tuple(dims) if dims.len() == 2 => dims,
        _ => return None,
    };
    Some([
        resolve_const_shape_dim(&dims[0], ctx),
        resolve_const_shape_dim(&dims[1], ctx),
    ])
}

fn find_program_dim(ctx: &LowerCtx<'_>, dim: i64) -> Option<SsaValue> {
    ctx.locals
        .values()
        .copied()
        .find(|value| match value {
            SsaValue::Local(idx) => ctx
                .defs
                .get(idx)
                .map(|inst| {
                    inst.opcode == SsaOpcode::ShapeDim
                        && inst.immediates.first().copied() == Some(dim)
                        && inst
                            .uses
                            .first()
                            .and_then(|base| match base {
                                SsaValue::Local(base_idx) => ctx.defs.get(base_idx),
                                _ => None,
                            })
                            .map(|base| base.opcode == SsaOpcode::ProgramId)
                            .unwrap_or(false)
                })
                .unwrap_or(false),
            _ => false,
        })
}

fn ensure_program_dim(ctx: &mut LowerCtx<'_>, dim: i64) -> SsaValue {
    if let Some(value) = find_program_dim(ctx, dim) {
        return value;
    }

    let program_id = ctx
        .locals
        .values()
        .copied()
        .find(|value| match value {
            SsaValue::Local(idx) => ctx
                .defs
                .get(idx)
                .map(|inst| inst.opcode == SsaOpcode::ProgramId)
                .unwrap_or(false),
            _ => false,
        })
        .unwrap_or_else(|| emit_instruction(ctx, SsaOpcode::ProgramId, vec![], vec![]));
    emit_instruction(ctx, SsaOpcode::ShapeDim, vec![program_id], vec![dim])
}

fn tile_shape_of_value(value: SsaValue, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    match value {
        SsaValue::Local(idx) => ctx.tile_shapes.get(&idx).cloned(),
        _ => None,
    }
}

fn record_tile_shape(value: &SsaValue, shape: Vec<i64>, ctx: &mut LowerCtx<'_>) {
    if let (SsaValue::Local(idx), false) = (value, shape.is_empty()) {
        ctx.tile_shapes.insert(*idx, shape);
    }
}

fn infer_result_tile_shape(
    op: BuiltinOp,
    uses: &[SsaValue],
    args: &[Expr],
    ctx: &LowerCtx<'_>,
) -> Option<Vec<i64>> {
    match op {
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul | BuiltinOp::Div | BuiltinOp::Exp => {
            uses.iter().find_map(|value| tile_shape_of_value(*value, ctx))
        }
        BuiltinOp::ReduceMax | BuiltinOp::ReduceSum => uses
            .first()
            .and_then(|value| tile_shape_of_value(*value, ctx))
            .and_then(|shape| shape.first().copied().map(|rows| vec![rows])),
        BuiltinOp::LoadTileLike2D => args
            .get(1)
            .and_then(|expr| expr_shape(expr, ctx))
            .filter(|shape| !shape.is_empty()),
        _ => None,
    }
}

fn expr_shape(expr: &Expr, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    match expr {
        Expr::Shape(_) => {
            let shape = extract_const_shape(expr, ctx);
            if shape.is_empty() {
                None
            } else {
                Some(shape)
            }
        }
        Expr::Var(name) => type_shape_for_name(name, ctx),
        _ => None,
    }
}

fn type_shape_for_name(name: &str, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    if let Some(ty) = ctx.typed.locals.get(name) {
        return resolve_type_shape(ty, ctx);
    }
    ctx.typed
        .kernel
        .params
        .iter()
        .find(|param| param.name == name)
        .and_then(|param| resolve_type_shape(&param.ty, ctx))
}

fn resolve_type_shape(ty: &Type, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    match ty {
        Type::Tensor { shape, .. } | Type::Tile { shape, .. } => match shape {
            ShapeExpr::Tuple(dims) => Some(
                dims.iter()
                    .map(|dim| resolve_const_shape_dim(dim, ctx))
                    .collect(),
            ),
            other => Some(vec![resolve_const_shape_dim(other, ctx)]),
        },
        Type::Shape | Type::Scalar(_) => None,
    }
}
