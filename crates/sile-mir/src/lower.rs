use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_hir::{BuiltinOp, Expr, ShapeExpr, Stmt};

use crate::ir::*;

/// Lower a type-checked HIR kernel to MIR.
pub fn lower_to_mir(typed: &TypedKernel) -> MirFunction {
    let mut ctx = LowerCtx::new(typed);

    // Create entry block
    let entry = ctx.new_block();
    ctx.entry = entry;
    ctx.current_block = entry;

    // Register kernel params as MIR params
    for param in &typed.kernel.params {
        let value = ctx.fresh_value();
        let rank = rank_of_param(param);
        let ty = MirType::Buffer { rank };
        ctx.mir_params.push(MirParam {
            value,
            name: param.name.clone(),
            ty: ty.clone(),
        });
        ctx.types.insert(value, ty);
        ctx.locals.insert(param.name.clone(), value);
    }

    // Lower body statements
    for stmt in &typed.kernel.body {
        lower_stmt(stmt, &mut ctx);
    }

    // Seal the current block with return
    ctx.seal_block(MirTerminator::Return);

    ctx.finish()
}

// ── Lowering context ───────────────────────────────────────────────

struct LowerCtx<'a> {
    typed: &'a TypedKernel,
    const_values: HashMap<String, i64>,
    locals: HashMap<String, ValueId>,
    types: HashMap<ValueId, MirType>,
    mir_params: Vec<MirParam>,
    blocks: Vec<MirBlock>,
    current_block: BlockId,
    entry: BlockId,
    current_insts: Vec<MirInst>,
    next_value: u32,
    next_block: u32,
    /// Pending block params for an exit block that hasn't been sealed yet
    pending_exit_params: Option<(BlockId, Vec<ValueId>)>,
}

impl<'a> LowerCtx<'a> {
    fn new(typed: &'a TypedKernel) -> Self {
        Self {
            typed,
            const_values: typed.kernel.const_params.iter().cloned().collect(),
            locals: HashMap::new(),
            types: HashMap::new(),
            mir_params: Vec::new(),
            blocks: Vec::new(),
            current_block: BlockId(0),
            entry: BlockId(0),
            current_insts: Vec::new(),
            next_value: 0,
            next_block: 0,
            pending_exit_params: None,
        }
    }

    fn fresh_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    fn emit(&mut self, op: MirOp, ty: MirType) -> ValueId {
        let result = self.fresh_value();
        self.types.insert(result, ty);
        self.current_insts.push(MirInst { result, op });
        result
    }

    /// Seal the current block with a terminator, push it to blocks list
    fn seal_block(&mut self, terminator: MirTerminator) {
        let params = if let Some((block_id, params)) = &self.pending_exit_params {
            if *block_id == self.current_block {
                let params = params.clone();
                self.pending_exit_params = None;
                params
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let block = MirBlock {
            id: self.current_block,
            params,
            insts: std::mem::take(&mut self.current_insts),
            terminator,
        };
        self.blocks.push(block);
    }

    /// Seal current block and switch to a new block
    #[allow(dead_code)]
    fn seal_and_switch(&mut self, terminator: MirTerminator) -> BlockId {
        self.seal_block(terminator);
        let new_id = self.new_block();
        self.current_block = new_id;
        new_id
    }

    fn finish(self) -> MirFunction {
        MirFunction {
            name: self.typed.kernel.name.clone(),
            params: self.mir_params,
            blocks: self.blocks,
            entry: self.entry,
            types: self.types,
        }
    }
}

// ── Statement lowering ─────────────────────────────────────────────

fn lower_stmt(stmt: &Stmt, ctx: &mut LowerCtx<'_>) {
    match stmt {
        Stmt::Let { name, expr, .. } | Stmt::Assign { name, expr } => {
            let value = lower_expr(expr, ctx);
            ctx.locals.insert(name.clone(), value);
        }
        Stmt::Store { target, value } => {
            lower_store(target, value, ctx);
        }
        Stmt::AtomicAdd {
            target,
            index,
            value,
        } => {
            lower_atomic_add(target, index, value, ctx);
        }
        Stmt::ForLoop {
            var,
            start,
            end,
            body,
        } => {
            lower_for_loop(var, start, end, body, ctx);
        }
    }
}

fn lower_for_loop(var: &str, start: &Expr, end: &Expr, body: &[Stmt], ctx: &mut LowerCtx<'_>) {
    let start_const = try_eval_i64(start, ctx);
    let end_const = try_eval_i64(end, ctx);

    // If bounds are compile-time known and small, unroll.
    if let (Some(start_val), Some(end_val)) = (start_const, end_const) {
        if end_val - start_val <= 32 {
            for i in start_val..end_val {
                let const_val = ctx.emit(MirOp::ConstI64(i), MirType::I64);
                ctx.locals.insert(var.to_string(), const_val);
                for inner in body {
                    lower_stmt(inner, ctx);
                }
            }
            return;
        }
    }

    // Otherwise: generate CFG loop with block parameters
    //
    //   entry_block:
    //     start_val = ...
    //     <collect loop-carried values>
    //     jump → header(start_val, carried_0, carried_1, ...)
    //
    //   header(k, carried_0, carried_1, ...):
    //     end_val = ...
    //     cond = k < end_val
    //     branch cond → body(...), exit(...)
    //
    //   body(k, ...):
    //     <loop body>
    //     k_next = k + 1
    //     jump → header(k_next, new_carried_0, ...)
    //
    //   exit(...):
    //     <continue>

    // Identify loop-carried variables by scanning the loop body
    let carried_names = find_loop_carried_vars(body);

    // Collect initial values for carried variables
    let init_carried: Vec<(String, ValueId)> = carried_names
        .iter()
        .filter_map(|name| ctx.locals.get(name).map(|v| (name.clone(), *v)))
        .collect();

    let header_id = ctx.new_block();
    let body_id = ctx.new_block();
    let exit_id = ctx.new_block();

    // Create header block parameters: [k, carried_0, carried_1, ...]
    let k_param = ctx.fresh_value();
    ctx.types.insert(k_param, MirType::I64);

    let carried_params: Vec<(String, ValueId)> = init_carried
        .iter()
        .map(|(name, init_val)| {
            let param = ctx.fresh_value();
            let ty = ctx.types.get(init_val).cloned().unwrap_or(MirType::Void);
            ctx.types.insert(param, ty);
            (name.clone(), param)
        })
        .collect();

    // Seal entry block → jump to header with initial values
    let start_v = match start_const {
        Some(value) => ctx.emit(MirOp::ConstI64(value), MirType::I64),
        None => lower_expr(start, ctx),
    };
    let mut jump_args = vec![start_v];
    jump_args.extend(init_carried.iter().map(|(_, v)| *v));
    ctx.seal_block(MirTerminator::Jump {
        target: header_id,
        args: jump_args,
    });

    // ── Header block ──
    ctx.current_block = header_id;

    // Set up locals from block params
    ctx.locals.insert(var.to_string(), k_param);
    for (name, param) in &carried_params {
        ctx.locals.insert(name.clone(), *param);
    }

    let end_v = match end_const {
        Some(value) => ctx.emit(MirOp::ConstI64(value), MirType::I64),
        None => lower_expr(end, ctx),
    };
    let cond = ctx.emit(
        MirOp::ICmp {
            op: CmpOp::Lt,
            lhs: k_param,
            rhs: end_v,
        },
        MirType::I64,
    );

    // Branch: loop-carried values pass through to both body and exit
    let body_args: Vec<ValueId> = std::iter::once(k_param)
        .chain(carried_params.iter().map(|(_, v)| *v))
        .collect();
    let exit_args: Vec<ValueId> = carried_params.iter().map(|(_, v)| *v).collect();

    // header block params
    let mut header_params = vec![k_param];
    header_params.extend(carried_params.iter().map(|(_, v)| *v));

    ctx.seal_block(MirTerminator::Branch {
        cond,
        true_target: body_id,
        true_args: body_args.clone(),
        false_target: exit_id,
        false_args: exit_args,
    });

    // Store header block params
    if let Some(header_block) = ctx.blocks.last_mut() {
        header_block.params = header_params;
    }

    // ── Body block ──
    ctx.current_block = body_id;

    // Body block also receives params (same as header passed them)
    let body_k_param = ctx.fresh_value();
    ctx.types.insert(body_k_param, MirType::I64);
    ctx.locals.insert(var.to_string(), body_k_param);

    let body_carried_params: Vec<(String, ValueId)> = carried_params
        .iter()
        .map(|(name, _)| {
            let param = ctx.fresh_value();
            let ty = init_carried
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, v)| ctx.types.get(v))
                .cloned()
                .unwrap_or(MirType::Void);
            ctx.types.insert(param, ty);
            ctx.locals.insert(name.clone(), param);
            (name.clone(), param)
        })
        .collect();

    // Execute loop body
    for inner in body {
        lower_stmt(inner, ctx);
    }

    // k_next = k + 1
    let one = ctx.emit(MirOp::ConstI64(1), MirType::I64);
    let k_next = ctx.emit(
        MirOp::IBinary {
            op: BinOp::Add,
            lhs: body_k_param,
            rhs: one,
        },
        MirType::I64,
    );

    // Collect new carried values (after body execution updated locals)
    let mut back_args = vec![k_next];
    for (name, _) in &carried_params {
        let current = ctx.locals.get(name).copied().unwrap_or(k_next);
        back_args.push(current);
    }

    // Body block params
    let mut body_block_params = vec![body_k_param];
    body_block_params.extend(body_carried_params.iter().map(|(_, v)| *v));

    ctx.seal_block(MirTerminator::Jump {
        target: header_id,
        args: back_args,
    });

    // Store body block params
    if let Some(body_block) = ctx.blocks.last_mut() {
        body_block.params = body_block_params;
    }

    // ── Exit block ──
    ctx.current_block = exit_id;

    // Exit receives the final carried values
    let exit_params: Vec<(String, ValueId)> = carried_params
        .iter()
        .map(|(name, _)| {
            let param = ctx.fresh_value();
            let ty = init_carried
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, v)| ctx.types.get(v))
                .cloned()
                .unwrap_or(MirType::Void);
            ctx.types.insert(param, ty);
            ctx.locals.insert(name.clone(), param);
            (name.clone(), param)
        })
        .collect();

    // Exit block params (no k, just carried values)
    let exit_block_params: Vec<ValueId> = exit_params.iter().map(|(_, v)| *v).collect();

    // We don't seal exit here — the caller will continue emitting into it.
    // Set the params on the block when it eventually gets sealed.
    // We need to remember these params. We'll store them in a pending manner.
    // Actually let's just create the block entry now and keep emitting.
    // The block is "open" — current_insts will accumulate into it.

    // Store exit block params for when it gets sealed
    // We'll use a small hack: create a placeholder block entry
    ctx.blocks.push(MirBlock {
        id: exit_id,
        params: exit_block_params,
        insts: Vec::new(),
        terminator: MirTerminator::Return, // placeholder, will be overwritten
    });

    // But we need current_insts to go into this block.
    // Remove the placeholder and manage manually.
    let placeholder = ctx.blocks.pop().unwrap();
    // We'll handle this by storing the params and using them when seal_block is next called.
    // Let's refactor: track pending block params.

    // Simpler approach: just track that exit block has these params,
    // and when we seal it, inject them.
    ctx.pending_exit_params = Some((exit_id, placeholder.params));
}

/// Find variables that are assigned within the loop body
/// (these need to become block parameters for the loop)
fn find_loop_carried_vars(body: &[Stmt]) -> Vec<String> {
    let mut vars = Vec::new();
    for stmt in body {
        match stmt {
            Stmt::Assign { name, .. } => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Stmt::Let { name, .. } => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            _ => {}
        }
    }
    vars
}

// ── Expression lowering ────────────────────────────────────────────

fn lower_expr(expr: &Expr, ctx: &mut LowerCtx<'_>) -> ValueId {
    match expr {
        Expr::Var(name) => {
            if let Some(v) = ctx.locals.get(name).copied() {
                return v;
            }
            if let Some(v) = ctx.const_values.get(name).copied() {
                return ctx.emit(MirOp::ConstI64(v), MirType::I64);
            }
            ctx.emit(MirOp::ConstI64(0), MirType::I64)
        }
        Expr::ScalarI64(v) => ctx.emit(MirOp::ConstI64(*v), MirType::I64),
        Expr::ScalarF32(v) => ctx.emit(MirOp::ConstF64(*v as f64), MirType::F32),
        Expr::Shape(_) => ctx.emit(MirOp::ConstI64(0), MirType::I64),
        Expr::Builtin { op, args } => lower_builtin(*op, args, ctx),
    }
}

fn lower_builtin(op: BuiltinOp, args: &[Expr], ctx: &mut LowerCtx<'_>) -> ValueId {
    match op {
        BuiltinOp::ProgramId => ctx.emit(MirOp::ProgramId { dim: 0 }, MirType::I64),
        BuiltinOp::ShapeDim => {
            // ShapeDim(ProgramId, dim) → ProgramId { dim }
            if matches!(
                &args[0],
                Expr::Builtin {
                    op: BuiltinOp::ProgramId,
                    ..
                }
            ) {
                let dim = eval_i64(&args[1], ctx);
                ctx.emit(MirOp::ProgramId { dim }, MirType::I64)
            } else {
                let base = lower_expr(&args[0], ctx);
                let dim = eval_i64(&args[1], ctx) as usize;
                ctx.emit(MirOp::ShapeDim { buf: base, dim }, MirType::I64)
            }
        }
        BuiltinOp::Constant => {
            let value = match args.first() {
                Some(Expr::ScalarF32(v)) => *v as f64,
                Some(Expr::ScalarI64(v)) => *v as f64,
                _ => 0.0,
            };
            let shape = args
                .get(1)
                .map(|arg| extract_const_shape(arg, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&shape);
            ctx.emit(
                MirOp::TileConstant { value, rows, cols },
                MirType::Tile { rows, cols },
            )
        }
        BuiltinOp::LoadTile => {
            let base = lower_expr(&args[0], ctx);
            let tile_shape = args
                .get(1)
                .map(|arg| extract_const_shape(arg, ctx))
                .unwrap_or_default();
            let coords = args
                .get(2)
                .map(|arg| extract_runtime_coords(arg, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&tile_shape);
            let (row_coord, col_coord) = coords_to_2d(&coords, ctx);
            let stride_shape_idx = param_stride_dim(&base, ctx);
            ctx.emit(
                MirOp::TileLoad {
                    buf: base,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
                MirType::Tile { rows, cols },
            )
        }
        BuiltinOp::LoadTileLike2D => {
            let input = lower_expr(&args[0], ctx);
            let shape = args
                .get(1)
                .and_then(|arg| expr_shape(arg, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&shape);
            let row_coord = ctx.emit(MirOp::ProgramId { dim: 0 }, MirType::I64);
            let col_coord = ctx.emit(MirOp::ProgramId { dim: 1 }, MirType::I64);
            let stride_shape_idx = param_stride_dim(&input, ctx);
            ctx.emit(
                MirOp::TileLoad {
                    buf: input,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
                MirType::Tile { rows, cols },
            )
        }
        BuiltinOp::Reshape | BuiltinOp::Broadcast => {
            let input = lower_expr(&args[0], ctx);
            let shape = args
                .get(1)
                .and_then(|expr| expr_shape(expr, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&shape);
            if op == BuiltinOp::Broadcast {
                ctx.emit(
                    MirOp::TileBroadcast {
                        value: input,
                        rows,
                        cols,
                    },
                    MirType::Tile { rows, cols },
                )
            } else {
                // Reshape is a no-op at tile level, just update type
                ctx.types.insert(input, MirType::Tile { rows, cols });
                input
            }
        }
        BuiltinOp::Mma => {
            let uses: Vec<ValueId> = args.iter().map(|arg| lower_expr(arg, ctx)).collect();
            let (a, b, acc) = (uses[0], uses[1], uses[2]);
            let acc_shape = tile_shape_of(acc, ctx).unwrap_or((1, 1));
            let a_shape = tile_shape_of(a, ctx).unwrap_or((1, 1));
            let tile_m = acc_shape.0;
            let tile_n = acc_shape.1;
            let tile_k = a_shape.1;
            ctx.emit(
                MirOp::TileMma {
                    a,
                    b,
                    acc,
                    tile_m,
                    tile_n,
                    tile_k,
                },
                MirType::Tile {
                    rows: tile_m,
                    cols: tile_n,
                },
            )
        }
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul | BuiltinOp::Div => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            let bin_op = match op {
                BuiltinOp::Add => BinOp::Add,
                BuiltinOp::Sub => BinOp::Sub,
                BuiltinOp::Mul => BinOp::Mul,
                BuiltinOp::Div => BinOp::Div,
                _ => unreachable!(),
            };
            // Check if tile or scalar
            if let Some((rows, cols)) = tile_shape_of(lhs, ctx).or_else(|| tile_shape_of(rhs, ctx))
            {
                ctx.emit(
                    MirOp::TileBinary {
                        op: bin_op,
                        lhs,
                        rhs,
                        rows,
                        cols,
                    },
                    MirType::Tile { rows, cols },
                )
            } else {
                ctx.emit(
                    MirOp::IBinary {
                        op: bin_op,
                        lhs,
                        rhs,
                    },
                    MirType::I64,
                )
            }
        }
        BuiltinOp::Exp => {
            let operand = lower_expr(&args[0], ctx);
            if let Some((rows, cols)) = tile_shape_of(operand, ctx) {
                ctx.emit(
                    MirOp::TileUnary {
                        op: UnaryOp::Exp,
                        operand,
                        rows,
                        cols,
                    },
                    MirType::Tile { rows, cols },
                )
            } else {
                ctx.emit(
                    MirOp::TileUnary {
                        op: UnaryOp::Exp,
                        operand,
                        rows: 1,
                        cols: 1,
                    },
                    MirType::F32,
                )
            }
        }
        BuiltinOp::ReduceMax | BuiltinOp::ReduceSum => {
            let value = lower_expr(&args[0], ctx);
            let reduce_op = match op {
                BuiltinOp::ReduceMax => ReduceOp::Max,
                BuiltinOp::ReduceSum => ReduceOp::Sum,
                _ => unreachable!(),
            };
            let (in_rows, in_cols) = tile_shape_of(value, ctx).unwrap_or((1, 1));
            let requested_axis = args.get(1).map(|arg| eval_i64(arg, ctx)).unwrap_or(1);
            // 1D tiles are lowered as 1xN, so their only logical axis maps to cols.
            let axis = if in_rows == 1 { 1 } else { requested_axis };
            let (out_rows, out_cols) = if axis == 1 {
                (in_rows, 1)
            } else {
                (1, in_cols)
            };
            ctx.emit(
                MirOp::TileReduce {
                    op: reduce_op,
                    value,
                    axis,
                    in_rows,
                    in_cols,
                },
                MirType::Tile {
                    rows: out_rows,
                    cols: out_cols,
                },
            )
        }
        BuiltinOp::ShapeOf | BuiltinOp::ScalarDiv => {
            let v = lower_expr(&args[0], ctx);
            v
        }
        BuiltinOp::Index => {
            let target = lower_expr(&args[0], ctx);
            let coords = match args.get(1) {
                Some(Expr::Shape(_)) => extract_runtime_coords(&args[1], ctx),
                Some(index) => vec![lower_expr(index, ctx)],
                None => vec![],
            };
            let (row_coord, col_coord) = coords_to_2d(&coords, ctx);
            ctx.emit(
                MirOp::TileExtract {
                    tile: target,
                    row_coord,
                    col_coord,
                },
                MirType::F32,
            )
        }
        BuiltinOp::Store => {
            // Handled in lower_store
            lower_expr(&args[0], ctx)
        }
    }
}

fn lower_store(target: &str, value: &Expr, ctx: &mut LowerCtx<'_>) {
    let output = ctx
        .locals
        .get(target)
        .copied()
        .unwrap_or_else(|| ctx.emit(MirOp::ConstI64(0), MirType::I64));
    let stored = lower_expr(value, ctx);

    let (rows, cols) = tile_shape_of(stored, ctx)
        .or_else(|| param_shape_2d(target, ctx))
        .unwrap_or((1, 1));

    let row_coord = ctx.emit(MirOp::ProgramId { dim: 0 }, MirType::I64);
    let col_coord = if rows > 1 || cols > 1 {
        ctx.emit(MirOp::ProgramId { dim: 1 }, MirType::I64)
    } else {
        ctx.emit(MirOp::ConstI64(0), MirType::I64)
    };

    let stride_shape_idx = param_stride_dim(&output, ctx);

    ctx.emit(
        MirOp::TileStore {
            buf: output,
            value: stored,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        },
        MirType::Void,
    );
}

fn lower_atomic_add(target: &str, index: &Expr, value: &Expr, ctx: &mut LowerCtx<'_>) {
    let output = ctx
        .locals
        .get(target)
        .copied()
        .unwrap_or_else(|| ctx.emit(MirOp::ConstI64(0), MirType::I64));
    let accumulated = lower_expr(value, ctx);
    let coords = match index {
        Expr::Shape(_) => extract_runtime_coords(index, ctx),
        _ => vec![lower_expr(index, ctx)],
    };
    let (row_coord, col_coord) = coords_to_2d(&coords, ctx);
    let stride_shape_idx = param_stride_dim(&output, ctx);

    ctx.emit(
        MirOp::AtomicAdd {
            buf: output,
            value: accumulated,
            row_coord,
            col_coord,
            stride_shape_idx,
        },
        MirType::Void,
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn eval_i64(expr: &Expr, ctx: &LowerCtx<'_>) -> i64 {
    try_eval_i64(expr, ctx).unwrap_or(0)
}

fn try_eval_i64(expr: &Expr, ctx: &LowerCtx<'_>) -> Option<i64> {
    match expr {
        Expr::ScalarI64(v) => Some(*v),
        Expr::ScalarF32(v) => Some(*v as i64),
        Expr::Var(name) => ctx.const_values.get(name).copied(),
        Expr::Builtin {
            op: BuiltinOp::ShapeDim,
            args,
        } => {
            let dim = usize::try_from(try_eval_i64(args.get(1)?, ctx)?).ok()?;
            let shape = expr_shape(args.first()?, ctx)?;
            let value = *shape.get(dim)?;
            (value >= 0).then_some(value)
        }
        Expr::Builtin {
            op: BuiltinOp::Add,
            args,
        } => Some(try_eval_i64(args.first()?, ctx)? + try_eval_i64(args.get(1)?, ctx)?),
        Expr::Builtin {
            op: BuiltinOp::Sub,
            args,
        } => Some(try_eval_i64(args.first()?, ctx)? - try_eval_i64(args.get(1)?, ctx)?),
        Expr::Builtin {
            op: BuiltinOp::Mul,
            args,
        } => Some(try_eval_i64(args.first()?, ctx)? * try_eval_i64(args.get(1)?, ctx)?),
        Expr::Builtin {
            op: BuiltinOp::Div | BuiltinOp::ScalarDiv,
            args,
        } => {
            let lhs = try_eval_i64(args.first()?, ctx)?;
            let rhs = try_eval_i64(args.get(1)?, ctx)?;
            (rhs != 0).then_some(lhs / rhs)
        }
        Expr::Shape(shape) => match shape {
            ShapeExpr::Constant(v) => Some(*v),
            ShapeExpr::Symbol(name) => ctx.const_values.get(name).copied(),
            _ => None,
        },
        _ => None,
    }
}

fn extract_const_shape(expr: &Expr, ctx: &LowerCtx<'_>) -> Vec<i64> {
    match expr {
        Expr::Shape(ShapeExpr::Tuple(dims)) => {
            dims.iter().map(|dim| resolve_shape_dim(dim, ctx)).collect()
        }
        Expr::Shape(dim) => vec![resolve_shape_dim(dim, ctx)],
        _ => vec![],
    }
}

fn resolve_shape_dim(dim: &ShapeExpr, ctx: &LowerCtx<'_>) -> i64 {
    match dim {
        ShapeExpr::Dynamic => -1,
        ShapeExpr::Constant(v) => *v,
        ShapeExpr::Symbol(name) => ctx.const_values.get(name).copied().unwrap_or(-1),
        ShapeExpr::Tuple(_) => -1,
    }
}

fn extract_runtime_coords(expr: &Expr, ctx: &mut LowerCtx<'_>) -> Vec<ValueId> {
    match expr {
        Expr::Shape(ShapeExpr::Tuple(dims)) => dims
            .iter()
            .map(|dim| resolve_runtime_dim(dim, ctx))
            .collect(),
        Expr::Shape(dim) => vec![resolve_runtime_dim(dim, ctx)],
        _ => vec![],
    }
}

fn resolve_runtime_dim(dim: &ShapeExpr, ctx: &mut LowerCtx<'_>) -> ValueId {
    match dim {
        ShapeExpr::Dynamic => ctx.emit(MirOp::ConstI64(-1), MirType::I64),
        ShapeExpr::Constant(v) => ctx.emit(MirOp::ConstI64(*v), MirType::I64),
        ShapeExpr::Symbol(name) => {
            if let Some(v) = ctx.locals.get(name).copied() {
                return v;
            }
            if let Some(v) = ctx.const_values.get(name).copied() {
                return ctx.emit(MirOp::ConstI64(v), MirType::I64);
            }
            ctx.emit(MirOp::ConstI64(0), MirType::I64)
        }
        ShapeExpr::Tuple(_) => ctx.emit(MirOp::ConstI64(0), MirType::I64),
    }
}

fn shape_to_2d(shape: &[i64]) -> (i64, i64) {
    match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn coords_to_2d(coords: &[ValueId], ctx: &mut LowerCtx<'_>) -> (ValueId, ValueId) {
    match coords {
        [] => {
            let zero = ctx.emit(MirOp::ConstI64(0), MirType::I64);
            let pid = ctx.emit(MirOp::ProgramId { dim: 0 }, MirType::I64);
            (zero, pid)
        }
        [col] => {
            let zero = ctx.emit(MirOp::ConstI64(0), MirType::I64);
            (zero, *col)
        }
        [row, col, ..] => (*row, *col),
    }
}

fn tile_shape_of(value: ValueId, ctx: &LowerCtx<'_>) -> Option<(i64, i64)> {
    match ctx.types.get(&value)? {
        MirType::Tile { rows, cols } => Some((*rows, *cols)),
        _ => None,
    }
}

fn param_shape_2d(name: &str, ctx: &LowerCtx<'_>) -> Option<(i64, i64)> {
    let param = ctx.typed.kernel.params.iter().find(|p| p.name == name)?;
    let sile_hir::Type::Tensor { shape, .. } = &param.ty else {
        return None;
    };
    let dims = match shape {
        ShapeExpr::Tuple(dims) if dims.len() == 2 => dims,
        _ => return None,
    };
    Some((
        resolve_shape_dim(&dims[0], ctx),
        resolve_shape_dim(&dims[1], ctx),
    ))
}

fn param_stride_dim(buf: &ValueId, ctx: &LowerCtx<'_>) -> usize {
    // Find which kernel parameter this buf corresponds to
    for (i, param) in ctx.mir_params.iter().enumerate() {
        if param.value == *buf {
            let hir_param = &ctx.typed.kernel.params[i];
            let rank = rank_of_param(hir_param);
            // stride dim is typically 1 for 2D (column stride)
            return if rank >= 2 { 1 } else { 0 };
        }
    }
    1
}

fn rank_of_param(param: &sile_hir::Param) -> usize {
    match &param.ty {
        sile_hir::Type::Tensor { shape, .. } | sile_hir::Type::Tile { shape, .. } => shape.rank(),
        sile_hir::Type::Shape | sile_hir::Type::Scalar(_) => 0,
    }
}

fn expr_shape(expr: &Expr, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    match expr {
        Expr::Shape(_) => {
            let shape = extract_const_shape(expr, ctx);
            if shape.is_empty() { None } else { Some(shape) }
        }
        Expr::Var(name) => type_shape_for_name(name, ctx),
        Expr::Builtin {
            op: BuiltinOp::ShapeOf,
            args,
        } => args.first().and_then(|arg| match arg {
            Expr::Var(name) => type_shape_for_name(name, ctx),
            other => expr_shape(other, ctx),
        }),
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
        .find(|p| p.name == name)
        .and_then(|p| resolve_type_shape(&p.ty, ctx))
}

fn resolve_type_shape(ty: &sile_hir::Type, ctx: &LowerCtx<'_>) -> Option<Vec<i64>> {
    match ty {
        sile_hir::Type::Tensor { shape, .. } | sile_hir::Type::Tile { shape, .. } => match shape {
            ShapeExpr::Tuple(dims) => {
                Some(dims.iter().map(|d| resolve_shape_dim(d, ctx)).collect())
            }
            other => Some(vec![resolve_shape_dim(other, ctx)]),
        },
        sile_hir::Type::Shape | sile_hir::Type::Scalar(_) => None,
    }
}
