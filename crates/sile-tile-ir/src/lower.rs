use std::collections::HashMap;

use sile_hir::typeck::TypedKernel;
use sile_hir::{BuiltinOp, Expr, ShapeExpr, Stmt};

use crate::ir::*;

/// Lower a type-checked HIR kernel to Tile IR.
pub fn lower_to_tile_ir(typed: &TypedKernel) -> TileIrFunction {
    let mut ctx = LowerCtx::new(typed);

    // Create entry block
    let entry = ctx.new_block();
    ctx.entry = entry;
    ctx.current_block = entry;

    // Register kernel params as Tile IR params
    for param in &typed.kernel.params {
        let rank = rank_of_param(param);
        let ty = TileIrType::Buffer { rank };
        let value = ctx.push_param(param.name.clone(), ty, TileIrParamKind::Buffer);
        ctx.locals.insert(param.name.clone(), value);
    }

    // Lower body statements
    for stmt in &typed.kernel.body {
        lower_stmt(stmt, &mut ctx);
    }

    // Seal the current block with return
    ctx.seal_block(TileIrTerminator::Return);

    ctx.finish()
}

// ── Lowering context ───────────────────────────────────────────────

struct LowerCtx<'a> {
    typed: &'a TypedKernel,
    const_values: HashMap<String, i64>,
    locals: HashMap<String, ValueId>,
    types: HashMap<ValueId, TileIrType>,
    mir_params: Vec<TileIrParam>,
    program_id_params: HashMap<i64, ValueId>,
    shape_desc_params: HashMap<ValueId, ValueId>,
    blocks: Vec<TileIrBlock>,
    current_block: BlockId,
    entry: BlockId,
    current_insts: Vec<TileIrInst>,
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
            program_id_params: HashMap::new(),
            shape_desc_params: HashMap::new(),
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

    fn emit(&mut self, op: TileIrOp, ty: TileIrType) -> ValueId {
        let result = self.fresh_value();
        self.types.insert(result, ty);
        self.current_insts.push(TileIrInst { result, op });
        result
    }

    fn push_param(&mut self, name: String, ty: TileIrType, kind: TileIrParamKind) -> ValueId {
        let value = self.fresh_value();
        self.types.insert(value, ty.clone());
        self.mir_params.push(TileIrParam {
            value,
            name,
            ty,
            kind,
        });
        value
    }

    fn get_or_create_program_id_param(&mut self, dim: i64) -> ValueId {
        if let Some(value) = self.program_id_params.get(&dim).copied() {
            return value;
        }
        let value = self.push_param(
            format!("__launch_idx{dim}"),
            TileIrType::I64,
            TileIrParamKind::LaunchIndex { dim },
        );
        self.program_id_params.insert(dim, value);
        value
    }

    fn get_or_create_shape_desc_param(&mut self, source: ValueId) -> ValueId {
        if let Some(value) = self.shape_desc_params.get(&source).copied() {
            return value;
        }
        let source_name = self
            .mir_params
            .iter()
            .find(|param| param.value == source)
            .map(|param| param.name.clone())
            .unwrap_or_else(|| format!("v{}", source.0));
        let rank = match self.types.get(&source) {
            Some(TileIrType::Buffer { rank }) => *rank,
            _ => 1,
        };
        let value = self.push_param(
            format!("__shape_{}", source_name),
            TileIrType::ShapeDesc { rank },
            TileIrParamKind::ShapeDesc { source },
        );
        self.shape_desc_params.insert(source, value);
        value
    }

    /// Seal the current block with a terminator, push it to blocks list
    fn seal_block(&mut self, terminator: TileIrTerminator) {
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
        let block = TileIrBlock {
            id: self.current_block,
            params,
            insts: std::mem::take(&mut self.current_insts),
            terminator,
        };
        self.blocks.push(block);
    }

    /// Seal current block and switch to a new block
    #[allow(dead_code)]
    fn seal_and_switch(&mut self, terminator: TileIrTerminator) -> BlockId {
        self.seal_block(terminator);
        let new_id = self.new_block();
        self.current_block = new_id;
        new_id
    }

    fn finish(self) -> TileIrFunction {
        TileIrFunction {
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
                let const_val = ctx.emit(TileIrOp::ConstI64(i), TileIrType::I64);
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
    ctx.types.insert(k_param, TileIrType::I64);

    let carried_params: Vec<(String, ValueId)> = init_carried
        .iter()
        .map(|(name, init_val)| {
            let param = ctx.fresh_value();
            let ty = ctx.types.get(init_val).cloned().unwrap_or(TileIrType::Void);
            ctx.types.insert(param, ty);
            (name.clone(), param)
        })
        .collect();

    // Seal entry block → jump to header with initial values
    let start_v = match start_const {
        Some(value) => ctx.emit(TileIrOp::ConstI64(value), TileIrType::I64),
        None => lower_expr(start, ctx),
    };
    let mut jump_args = vec![start_v];
    jump_args.extend(init_carried.iter().map(|(_, v)| *v));
    ctx.seal_block(TileIrTerminator::Jump {
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
        Some(value) => ctx.emit(TileIrOp::ConstI64(value), TileIrType::I64),
        None => lower_expr(end, ctx),
    };
    let cond = ctx.emit(
        TileIrOp::ICmp {
            op: CmpOp::Lt,
            lhs: k_param,
            rhs: end_v,
        },
        TileIrType::I64,
    );

    // Branch: loop-carried values pass through to both body and exit
    let body_args: Vec<ValueId> = std::iter::once(k_param)
        .chain(carried_params.iter().map(|(_, v)| *v))
        .collect();
    let exit_args: Vec<ValueId> = carried_params.iter().map(|(_, v)| *v).collect();

    // header block params
    let mut header_params = vec![k_param];
    header_params.extend(carried_params.iter().map(|(_, v)| *v));

    ctx.seal_block(TileIrTerminator::Branch {
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
    ctx.types.insert(body_k_param, TileIrType::I64);
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
                .unwrap_or(TileIrType::Void);
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
    let one = ctx.emit(TileIrOp::ConstI64(1), TileIrType::I64);
    let k_next = ctx.emit(
        TileIrOp::IBinary {
            op: BinOp::Add,
            lhs: body_k_param,
            rhs: one,
        },
        TileIrType::I64,
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

    ctx.seal_block(TileIrTerminator::Jump {
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
                .unwrap_or(TileIrType::Void);
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
    ctx.blocks.push(TileIrBlock {
        id: exit_id,
        params: exit_block_params,
        insts: Vec::new(),
        terminator: TileIrTerminator::Return, // placeholder, will be overwritten
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
                return ctx.emit(TileIrOp::ConstI64(v), TileIrType::I64);
            }
            ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
        }
        Expr::ScalarI64(v) => ctx.emit(TileIrOp::ConstI64(*v), TileIrType::I64),
        Expr::ScalarF32(v) => ctx.emit(TileIrOp::ConstF64(*v as f64), TileIrType::F32),
        Expr::Shape(_) => ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64),
        Expr::Builtin { op, args } => lower_builtin(*op, args, ctx),
    }
}

fn lower_builtin(op: BuiltinOp, args: &[Expr], ctx: &mut LowerCtx<'_>) -> ValueId {
    match op {
        BuiltinOp::ProgramId => ctx.get_or_create_program_id_param(0),
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
                ctx.get_or_create_program_id_param(dim)
            } else {
                let base = lower_expr(&args[0], ctx);
                let dim = eval_i64(&args[1], ctx) as usize;
                let shape = ctx.get_or_create_shape_desc_param(base);
                ctx.emit(TileIrOp::ShapeDim { shape, dim }, TileIrType::I64)
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
                TileIrOp::Splat { value, rows, cols },
                TileIrType::Tile { rows, cols },
            )
        }
        BuiltinOp::LoadTile => {
            let base = lower_expr(&args[0], ctx);
            let _ = ctx.get_or_create_shape_desc_param(base);
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
                TileIrOp::LoadPtrTko {
                    buf: base,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
                TileIrType::Tile { rows, cols },
            )
        }
        BuiltinOp::LoadTileLike2D => {
            let input = lower_expr(&args[0], ctx);
            let _ = ctx.get_or_create_shape_desc_param(input);
            let shape = args
                .get(1)
                .and_then(|arg| expr_shape(arg, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&shape);
            let row_coord = ctx.get_or_create_program_id_param(0);
            let col_coord = ctx.get_or_create_program_id_param(1);
            let stride_shape_idx = param_stride_dim(&input, ctx);
            ctx.emit(
                TileIrOp::LoadPtrTko {
                    buf: input,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
                TileIrType::Tile { rows, cols },
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
                    TileIrOp::Broadcast {
                        value: input,
                        rows,
                        cols,
                    },
                    TileIrType::Tile { rows, cols },
                )
            } else {
                // Reshape is a no-op at tile level, just update type
                ctx.types.insert(input, TileIrType::Tile { rows, cols });
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
                TileIrOp::MmaF {
                    a,
                    b,
                    acc,
                    tile_m,
                    tile_n,
                    tile_k,
                },
                TileIrType::Tile {
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
                let op = match bin_op {
                    BinOp::Add => TileIrOp::AddF {
                        lhs,
                        rhs,
                        rows,
                        cols,
                    },
                    BinOp::Sub => TileIrOp::SubF {
                        lhs,
                        rhs,
                        rows,
                        cols,
                    },
                    BinOp::Mul => TileIrOp::MulF {
                        lhs,
                        rhs,
                        rows,
                        cols,
                    },
                    BinOp::Div => TileIrOp::DivF {
                        lhs,
                        rhs,
                        rows,
                        cols,
                    },
                };
                ctx.emit(op, TileIrType::Tile { rows, cols })
            } else {
                ctx.emit(
                    TileIrOp::IBinary {
                        op: bin_op,
                        lhs,
                        rhs,
                    },
                    TileIrType::I64,
                )
            }
        }
        BuiltinOp::Exp => {
            let operand = lower_expr(&args[0], ctx);
            if let Some((rows, cols)) = tile_shape_of(operand, ctx) {
                ctx.emit(
                    TileIrOp::Exp {
                        operand,
                        rows,
                        cols,
                    },
                    TileIrType::Tile { rows, cols },
                )
            } else {
                ctx.emit(
                    TileIrOp::Exp {
                        operand,
                        rows: 1,
                        cols: 1,
                    },
                    TileIrType::F32,
                )
            }
        }
        BuiltinOp::ReduceMax | BuiltinOp::ReduceSum => {
            let value = lower_expr(&args[0], ctx);
            let (in_rows, in_cols) = tile_shape_of(value, ctx).unwrap_or((1, 1));
            let requested_axis = args.get(1).map(|arg| eval_i64(arg, ctx)).unwrap_or(1);
            // 1D tiles are lowered as 1xN, so their only logical axis maps to cols.
            let axis = if in_rows == 1 { 1 } else { requested_axis };
            let (out_rows, out_cols) = if axis == 1 {
                (in_rows, 1)
            } else {
                (1, in_cols)
            };
            let reduce_op = match op {
                BuiltinOp::ReduceMax => TileIrOp::ReduceMax {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                },
                BuiltinOp::ReduceSum => TileIrOp::ReduceSum {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                },
                _ => unreachable!(),
            };
            ctx.emit(
                reduce_op,
                TileIrType::Tile {
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
                TileIrOp::Extract {
                    tile: target,
                    row_coord,
                    col_coord,
                },
                TileIrType::F32,
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
        .unwrap_or_else(|| ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64));
    let _ = ctx.get_or_create_shape_desc_param(output);
    let stored = lower_expr(value, ctx);

    let (rows, cols) = tile_shape_of(stored, ctx)
        .or_else(|| param_shape_2d(target, ctx))
        .unwrap_or((1, 1));

    let output_rank = match ctx.types.get(&output) {
        Some(TileIrType::Buffer { rank }) => *rank,
        _ => 1,
    };
    let row_coord = if output_rank <= 1 {
        ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
    } else {
        ctx.get_or_create_program_id_param(0)
    };
    let col_coord = if output_rank <= 1 {
        ctx.get_or_create_program_id_param(0)
    } else if rows > 1 || cols > 1 {
        ctx.get_or_create_program_id_param(1)
    } else {
        ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
    };

    let stride_shape_idx = param_stride_dim(&output, ctx);

    ctx.emit(
        TileIrOp::StorePtrTko {
            buf: output,
            value: stored,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        },
        TileIrType::Void,
    );
}

fn lower_atomic_add(target: &str, index: &Expr, value: &Expr, ctx: &mut LowerCtx<'_>) {
    let output = ctx
        .locals
        .get(target)
        .copied()
        .unwrap_or_else(|| ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64));
    let _ = ctx.get_or_create_shape_desc_param(output);
    let accumulated = lower_expr(value, ctx);
    let coords = match index {
        Expr::Shape(_) => extract_runtime_coords(index, ctx),
        _ => vec![lower_expr(index, ctx)],
    };
    let (row_coord, col_coord) = coords_to_2d(&coords, ctx);
    let stride_shape_idx = param_stride_dim(&output, ctx);

    ctx.emit(
        TileIrOp::SileAtomicAdd {
            buf: output,
            value: accumulated,
            row_coord,
            col_coord,
            stride_shape_idx,
        },
        TileIrType::Void,
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
        ShapeExpr::Dynamic => ctx.emit(TileIrOp::ConstI64(-1), TileIrType::I64),
        ShapeExpr::Constant(v) => ctx.emit(TileIrOp::ConstI64(*v), TileIrType::I64),
        ShapeExpr::Symbol(name) => {
            if let Some(v) = ctx.locals.get(name).copied() {
                return v;
            }
            if let Some(v) = ctx.const_values.get(name).copied() {
                return ctx.emit(TileIrOp::ConstI64(v), TileIrType::I64);
            }
            ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
        }
        ShapeExpr::Tuple(_) => ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64),
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
            let zero = ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64);
            let pid = ctx.get_or_create_program_id_param(0);
            (zero, pid)
        }
        [col] => {
            let zero = ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64);
            (zero, *col)
        }
        [row, col, ..] => (*row, *col),
    }
}

fn tile_shape_of(value: ValueId, ctx: &LowerCtx<'_>) -> Option<(i64, i64)> {
    match ctx.types.get(&value)? {
        TileIrType::Tile { rows, cols } => Some((*rows, *cols)),
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
