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
    let mut scalar_index = 0usize;
    let mut scalar_i64_index = 0usize;
    for param in &typed.kernel.params {
        let (ty, kind) = match &param.ty {
            sile_hir::Type::Scalar(elem) => {
                // A float and an integer travel in their own descriptor, so
                // each counts its own slots.
                let (ty, index) = if elem.is_float() {
                    let index = scalar_index;
                    scalar_index += 1;
                    (TileIrType::F32, index)
                } else {
                    let index = scalar_i64_index;
                    scalar_i64_index += 1;
                    (TileIrType::I64, index)
                };
                (ty, TileIrParamKind::Scalar { index })
            }
            _ => (
                TileIrType::Buffer {
                    rank: rank_of_param(param),
                },
                TileIrParamKind::Buffer,
            ),
        };
        let value = ctx.push_param(param.name.clone(), ty, kind);
        ctx.locals.insert(param.name.clone(), value);
    }

    // Lower body statements. A `continue` cannot reach here: the checker
    // rejects one outside a loop, and a loop body lowers its own statements.
    for stmt in &typed.kernel.body {
        let _ = lower_stmt(stmt, &mut ctx);
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
            format!("launch_idx{dim}"),
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
            format!("shape_{}", source_name),
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

    /// Give a sealed block the parameters its incoming edges pass.
    ///
    /// A loop learns the parameters of its header and its body only after the
    /// blocks exist, so it names them by id. Patching whichever block was
    /// sealed last would be wrong: a body that ends in a branch seals the
    /// blocks of that branch too.
    fn set_block_params(&mut self, id: BlockId, params: Vec<ValueId>) {
        if let Some(block) = self.blocks.iter_mut().find(|block| block.id == id) {
            block.params = params;
        }
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

/// What a statement did to the path through the block it is in.
///
/// A `continue` ends that path: the statements after it in the same loop body
/// do not run, so the body stops lowering there and the loop's existing back
/// edge to its header becomes the `continue`. Every other statement lets the
/// next one follow it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Next,
    Continue,
}

fn lower_stmt(stmt: &Stmt, ctx: &mut LowerCtx<'_>) -> Flow {
    match stmt {
        Stmt::Let { name, expr, .. } | Stmt::Assign { name, expr } => {
            let value = lower_expr(expr, ctx);
            ctx.locals.insert(name.clone(), value);
        }
        Stmt::Store {
            target,
            value,
            index,
        } => {
            lower_store(target, value, index.as_ref(), ctx);
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
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            return lower_if(cond, then_body, else_body, ctx);
        }
        Stmt::Continue => return Flow::Continue,
    }
    Flow::Next
}

/// `if cond { .. } else { .. }`.
///
/// The branch is one conditional jump to two blocks, and each block jumps to a
/// join block. A name either arm assigns becomes a block parameter of the join,
/// which is the phi a reader of the IR would write, and it is what makes
///
/// ```text
/// if CAUSAL == 1 { mask_start = ...; }
/// ```
///
/// leave `mask_start` set after the branch.
fn lower_if(cond: &Expr, then_body: &[Stmt], else_body: &[Stmt], ctx: &mut LowerCtx<'_>) -> Flow {
    // A condition over constants is known here, so the arm that is never taken
    // does not reach the IR at all. This is what makes a const parameter do its
    // work: `if CAUSAL == 1` is a straight line when the caller passes one, and
    // nothing at all when it passes zero.
    if let Some(taken) = try_eval_cond(cond, ctx) {
        for stmt in if taken { then_body } else { else_body } {
            if lower_stmt(stmt, ctx) == Flow::Continue {
                return Flow::Continue;
            }
        }
        return Flow::Next;
    }

    // A name either arm assigns is carried to the join. The names the arms bind
    // for themselves are not: a `let` is fresh in the arm it appears in.
    let carried = find_carried_vars(&[then_body, else_body], ctx);
    let init: Vec<(String, ValueId)> = carried
        .iter()
        .filter_map(|name| ctx.locals.get(name).map(|value| (name.clone(), *value)))
        .collect();

    let cond_value = lower_condition(cond, ctx);

    let then_id = ctx.new_block();
    let else_id = ctx.new_block();
    let join_id = ctx.new_block();

    let init_args: Vec<ValueId> = init.iter().map(|(_, value)| *value).collect();
    ctx.seal_block(TileIrTerminator::Branch {
        cond: cond_value,
        true_target: then_id,
        true_args: init_args.clone(),
        false_target: else_id,
        false_args: init_args,
    });

    let then_args = lower_branch(then_id, then_body, &init, join_id, ctx);
    let else_args = lower_branch(else_id, else_body, &init, join_id, ctx);
    debug_assert_eq!(then_args.len(), init.len());
    debug_assert_eq!(else_args.len(), init.len());

    // ── Join block ──
    //
    // Each arm passes its final value for every carried name, so the join reads
    // one value per name whichever arm ran. The block stays open: the statement
    // after the branch is emitted into it, and `pending_exit_params` gives it
    // the parameters the arms jump with.
    ctx.current_block = join_id;
    let join_params: Vec<ValueId> = init
        .iter()
        .map(|(name, value)| {
            let param = ctx.fresh_value();
            let ty = ctx.types.get(value).cloned().unwrap_or(TileIrType::Void);
            ctx.types.insert(param, ty);
            ctx.locals.insert(name.clone(), param);
            param
        })
        .collect();
    ctx.pending_exit_params = Some((join_id, join_params));

    Flow::Next
}

/// Lower one arm of a branch and jump from it to the join.
///
/// The arm receives every carried name as a block parameter, so it reads and
/// writes the names the enclosing scope had. A name the arm does not assign
/// keeps its parameter, and that value is what the arm passes on.
fn lower_branch(
    block_id: BlockId,
    body: &[Stmt],
    init: &[(String, ValueId)],
    join_id: BlockId,
    ctx: &mut LowerCtx<'_>,
) -> Vec<ValueId> {
    ctx.current_block = block_id;
    let arm_params: Vec<ValueId> = init
        .iter()
        .map(|(name, value)| {
            let param = ctx.fresh_value();
            let ty = ctx.types.get(value).cloned().unwrap_or(TileIrType::Void);
            ctx.types.insert(param, ty);
            ctx.locals.insert(name.clone(), param);
            param
        })
        .collect();
    for stmt in body {
        // A `continue` cannot reach here: the checker rejects one inside a
        // branch, because the arm it sits in would end somewhere the other arm
        // does not.
        let _ = lower_stmt(stmt, ctx);
    }
    // The arm ends where the statements left it, which is a nested construct's
    // join when the arm ends with one.
    let args: Vec<ValueId> = init
        .iter()
        .map(|(name, value)| ctx.locals.get(name).copied().unwrap_or(*value))
        .collect();
    ctx.seal_block(TileIrTerminator::Jump {
        target: join_id,
        args: args.clone(),
    });
    // The arm's own block takes the same parameters. The seal above may have
    // sealed a nested construct's join instead, so the parameters go on by id.
    ctx.set_block_params(block_id, arm_params);
    args
}

/// The predicate a branch tests.
///
/// A comparison of two scalars is the predicate itself: one `icmp` or one
/// `fcmp`, which is what the branch reads. Lowering it to a mask and then
/// testing the mask says the same thing in three instructions.
///
/// Every other condition is a value. A mask holds where it is not zero and a
/// count is true where it is not zero, so the test is against zero. The checker
/// has already rejected a condition that is a tile: a per-element choice is
/// what `select` is for.
fn lower_condition(cond: &Expr, ctx: &mut LowerCtx<'_>) -> ValueId {
    if let Expr::Builtin { op, args } = cond
        && let Some(cmp) = compare_op(*op)
        && let (Some(lhs_arg), Some(rhs_arg)) = (args.first(), args.get(1))
    {
        let lhs = lower_expr(lhs_arg, ctx);
        let rhs = lower_expr(rhs_arg, ctx);
        if tile_shape_of(lhs, ctx).is_none() && tile_shape_of(rhs, ctx).is_none() {
            return ctx.emit(TileIrOp::ICmp { op: cmp, lhs, rhs }, TileIrType::I64);
        }
        // A comparison of tiles is per element, so it is a mask the branch
        // tests. The checker rejects this condition, and lowering it as a test
        // of the mask keeps the operands evaluated once.
        let mask = lower_compare(cmp, lhs, rhs, ctx);
        return test_nonzero(mask, ctx);
    }
    test_nonzero(lower_expr(cond, ctx), ctx)
}

/// A value as the predicate a branch tests.
fn test_nonzero(value: ValueId, ctx: &mut LowerCtx<'_>) -> ValueId {
    let zero = if is_float_operand(value, ctx) {
        ctx.emit(TileIrOp::ConstF64(0.0), TileIrType::F32)
    } else {
        ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
    };
    ctx.emit(
        TileIrOp::ICmp {
            op: CmpOp::Ne,
            lhs: value,
            rhs: zero,
        },
        TileIrType::I64,
    )
}

/// The comparison a builtin names, if it names one.
fn compare_op(op: BuiltinOp) -> Option<CmpOp> {
    match op {
        BuiltinOp::Lt => Some(CmpOp::Lt),
        BuiltinOp::Le => Some(CmpOp::Le),
        BuiltinOp::Gt => Some(CmpOp::Gt),
        BuiltinOp::Ge => Some(CmpOp::Ge),
        BuiltinOp::Eq => Some(CmpOp::Eq),
        BuiltinOp::Ne => Some(CmpOp::Ne),
        _ => None,
    }
}

/// Whether a condition is known before the kernel runs.
///
/// A const parameter is a compile-time value, so `if CAUSAL == 1` names a
/// branch that is always taken or never taken. A comparison folds to `1` or
/// `0`, and `&`/`|` combine those, so one rule reads a count and a mask alike.
fn try_eval_cond(expr: &Expr, ctx: &LowerCtx<'_>) -> Option<bool> {
    try_eval_i64(expr, ctx).map(|value| value != 0)
}

fn lower_for_loop(var: &str, start: &Expr, end: &Expr, body: &[Stmt], ctx: &mut LowerCtx<'_>) {
    let start_const = try_eval_i64(start, ctx);
    let end_const = try_eval_i64(end, ctx);

    // Bindings made inside the body are local to the loop, so the locals map
    // is restored when the loop ends. The type checker already rejects a use
    // after the loop; this keeps the lowering from silently shadowing an outer
    // name if a later stage ever accepts one.
    let outer_locals = ctx.locals.clone();

    // If bounds are compile-time known and small, unroll.
    if let (Some(start_val), Some(end_val)) = (start_const, end_const) {
        if end_val - start_val <= 32 {
            for i in start_val..end_val {
                let const_val = ctx.emit(TileIrOp::ConstI64(i), TileIrType::I64);
                ctx.locals.insert(var.to_string(), const_val);
                for inner in body {
                    // An unrolled `continue` ends this copy of the body, so the
                    // remaining statements of this iteration do not run and the
                    // next constant follows.
                    if lower_stmt(inner, ctx) == Flow::Continue {
                        break;
                    }
                }
            }
            // An unrolled body still assigns to outer names, so their final
            // values stay. Names the body introduced are dropped.
            let outer_names: Vec<String> = outer_locals.keys().cloned().collect();
            let mut restored = outer_locals;
            for name in outer_names {
                if let Some(value) = ctx.locals.get(&name).copied() {
                    restored.insert(name, value);
                }
            }
            ctx.locals = restored;
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

    // Identify loop-carried variables by scanning the loop body. Only names
    // that already exist outside the loop are carried; a `let` in the body is
    // a fresh binding on each iteration.
    let carried_names = find_carried_vars(&[body], ctx);

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
    ctx.set_block_params(header_id, header_params);

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

    // Execute loop body. A `continue` stops the body here, and the back edge
    // below carries the values the loop holds at that point, which is what a
    // `continue` does: it ends this pass through the body and starts the next.
    for inner in body {
        if lower_stmt(inner, ctx) == Flow::Continue {
            break;
        }
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
    ctx.set_block_params(body_id, body_block_params);

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

    // Exit block params (no k, just carried values). Everything the body bound
    // for itself goes out of scope here; only the carried names survive, bound
    // to the exit block's parameters.
    ctx.locals = outer_locals;
    for (name, param) in &exit_params {
        ctx.locals.insert(name.clone(), *param);
    }

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

/// Find variables that are assigned within the bodies and already exist
/// outside them. These become block parameters of the block the bodies join.
///
/// A `let` in a body is not one of them: it is a fresh binding, so it has no
/// value to carry in. Treating it as carried made the loop read a value from
/// before the loop and write it back as if the body had assigned it.
fn find_carried_vars(bodies: &[&[Stmt]], ctx: &LowerCtx<'_>) -> Vec<String> {
    let mut vars = Vec::new();
    for body in bodies {
        collect_carried_names(body, ctx, &mut vars);
    }
    vars
}

/// Walk the bodies, including nested loops and branches.
///
/// An assignment inside a nested loop is still an assignment to the outer
/// loop's variable. Scanning only the top level missed it, so the outer loop
/// never carried the value and the backedge dropped the update.
fn collect_carried_names(body: &[Stmt], ctx: &LowerCtx<'_>, vars: &mut Vec<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign { name, .. } => {
                if ctx.locals.contains_key(name) && !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Stmt::ForLoop { body, .. } => collect_carried_names(body, ctx, vars),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_carried_names(then_body, ctx, vars);
                collect_carried_names(else_body, ctx, vars);
            }
            Stmt::Let { .. } | Stmt::Store { .. } | Stmt::AtomicAdd { .. } | Stmt::Continue => {}
        }
    }
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
        // An index tuple's value is its first axis. Callers needing every axis
        // use `extract_runtime_coords`, which reads the inner list directly.
        Expr::TileIndex(elems) => elems
            .first()
            .map(|elem| lower_expr(elem, ctx))
            .unwrap_or_else(|| ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)),
        Expr::Builtin { op, args } => lower_builtin(*op, args, ctx),
    }
}

fn lower_builtin(op: BuiltinOp, args: &[Expr], ctx: &mut LowerCtx<'_>) -> ValueId {
    match op {
        BuiltinOp::ProgramId => {
            // `program_id(axis)` names its axis; the historical `id()` spelling
            // reached here through `ShapeDim`, which carries the same axis in
            // its second argument.
            let dim = args.first().and_then(|arg| try_eval_i64(arg, ctx)).unwrap_or(0);
            ctx.get_or_create_program_id_param(dim)
        }
        BuiltinOp::NumPrograms => {
            // num_programs(axis): the launch extent. Taken from the axis index
            // argument when it is a literal.
            let dim = args.first().and_then(|arg| try_eval_i64(arg, ctx)).unwrap_or(0);
            ctx.emit(TileIrOp::GridDim { dim }, TileIrType::I64)
        }
        BuiltinOp::NumTiles => {
            // num_tiles(buffer, axis): tiles along one axis of that buffer.
            let buffer = lower_expr(&args[0], ctx);
            let shape = ctx.get_or_create_shape_desc_param(buffer);
            let dim = args
                .get(1)
                .and_then(|arg| try_eval_i64(arg, ctx))
                .unwrap_or(0) as usize;
            let tile_extent = load_tile_extent_for_buffer(buffer, dim, ctx);
            ctx.emit(
                TileIrOp::NumTiles {
                    shape,
                    dim,
                    tile_extent,
                },
                TileIrType::I64,
            )
        }
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
        // A partition is a view, not an instruction. It names the buffer, so
        // the value that reaches a later `load` or `store` is the buffer
        // itself; the tile extent travels in the HIR type and is read there.
        // `partition_mut` is the same view, opened for writing.
        BuiltinOp::Partition | BuiltinOp::PartitionMut => lower_expr(&args[0], ctx),
        BuiltinOp::Load => {
            // `part.load([i, j])` is `load_tile(shape, [i, j])` with the shape
            // taken from the partition. The partition lowers to its buffer, so
            // the buffer is `args[0]` and the tile extent comes from the HIR.
            //
            // `y.load()` names no coordinates: the tile is the one the program
            // owns, which is the same tile `load_like` reads.
            let base = lower_expr(&args[0], ctx);
            let _ = ctx.get_or_create_shape_desc_param(base);
            let tile_shape = partition_tile_extent(&args[0], ctx);
            let tile_shape = if tile_shape.is_empty() {
                args.first().and_then(|arg| expr_shape(arg, ctx)).unwrap_or_default()
            } else {
                tile_shape
            };
            let coords = args
                .get(1)
                .map(|arg| extract_runtime_coords(arg, ctx))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&tile_shape);
            let (plane, row_coord, col_coord) = if coords.is_empty() {
                program_tile_coords(buffer_rank(base, ctx), ctx)
            } else {
                let (row, col) = coords_to_2d(&coords, ctx);
                (outer_coords(&coords), row, col)
            };
            let stride_shape_idx = param_stride_dim(&base, ctx);
            ctx.emit(
                TileIrOp::LoadPtrTko {
                    buf: base,
                    outer_coords: plane,
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
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
                    outer_coords: outer_coords(&coords),
                    row_coord,
                    col_coord,
                    rows,
                    cols,
                    stride_shape_idx,
                },
                TileIrType::Tile { rows, cols },
            )
        }
        BuiltinOp::LoadLike => {
            let input = lower_expr(&args[0], ctx);
            let _ = ctx.get_or_create_shape_desc_param(input);
            // `load_tile_like_2d(x, y)` takes the tile shape from the output
            // buffer `y`, which is the second argument. A third argument, when
            // present, is the index and does not carry the shape.
            //
            // `load_tile_mut(z)` names only the buffer, so the tile is the one
            // `z` declares: its own parameter shape.
            let shape = args
                .get(1)
                .and_then(|arg| expr_shape(arg, ctx))
                .or_else(|| args.first().and_then(|arg| expr_shape(arg, ctx)))
                .unwrap_or_default();
            let (rows, cols) = shape_to_2d(&shape);
            let (plane, row_coord, col_coord) = program_tile_coords(buffer_rank(input, ctx), ctx);
            let stride_shape_idx = param_stride_dim(&input, ctx);
            ctx.emit(
                TileIrOp::LoadPtrTko {
                    buf: input,
                    outer_coords: plane,
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
                ctx.emit(
                    TileIrOp::Reshape {
                        value: input,
                        rows,
                        cols,
                    },
                    TileIrType::Tile { rows, cols },
                )
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
        // A maximum and a minimum are one comparison and one select each, so
        // they ride the map node: the map already carries a scalar expression
        // per element, and no target has an instruction that does more.
        BuiltinOp::Max | BuiltinOp::Min => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            lower_extreme(op == BuiltinOp::Max, lhs, rhs, ctx)
        }
        // A mask is `0.0` or `1.0`, so a conjunction of masks is their product
        // and a disjunction is their maximum. Two counts are anded and ored bit
        // by bit instead, and the operands say which of the two an operator
        // means.
        BuiltinOp::And | BuiltinOp::Or => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            if !is_float_operand(lhs, ctx) && !is_float_operand(rhs, ctx) {
                let op = if op == BuiltinOp::And {
                    BinOp::And
                } else {
                    BinOp::Or
                };
                return ctx.emit(TileIrOp::IBinary { op, lhs, rhs }, TileIrType::I64);
            }
            if op == BuiltinOp::And {
                lower_arith(BinOp::Mul, lhs, rhs, ctx)
            } else {
                lower_extreme(true, lhs, rhs, ctx)
            }
        }
        // `ceil_div(a, b)` is `(a + b - 1) / b`. CuTile names it as one
        // operation over counts, and the three integer steps are what every
        // target would have to write, so the lowering writes them once.
        BuiltinOp::CeilDiv => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            let one = ctx.emit(TileIrOp::ConstI64(1), TileIrType::I64);
            let biased = ctx.emit(
                TileIrOp::IBinary {
                    op: BinOp::Add,
                    lhs,
                    rhs,
                },
                TileIrType::I64,
            );
            let adjusted = ctx.emit(
                TileIrOp::IBinary {
                    op: BinOp::Sub,
                    lhs: biased,
                    rhs: one,
                },
                TileIrType::I64,
            );
            ctx.emit(
                TileIrOp::IBinary {
                    op: BinOp::Div,
                    lhs: adjusted,
                    rhs,
                },
                TileIrType::I64,
            )
        }
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul | BuiltinOp::Div | BuiltinOp::Rem => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            let bin_op = match op {
                BuiltinOp::Add => BinOp::Add,
                BuiltinOp::Sub => BinOp::Sub,
                BuiltinOp::Mul => BinOp::Mul,
                BuiltinOp::Div => BinOp::Div,
                BuiltinOp::Rem => BinOp::Rem,
                _ => unreachable!(),
            };
            lower_arith(bin_op, lhs, rhs, ctx)
        }
        // A transpose walks the tile it writes and reads its operand with the
        // two coordinates exchanged, which is a map over the transposed domain.
        BuiltinOp::Transpose => {
            let value = lower_expr(&args[0], ctx);
            let (rows, cols) = tile_shape_of(value, ctx).unwrap_or((1, 1));
            ctx.emit(
                TileIrOp::Map {
                    expr: TileMapExpr::Transpose {
                        value: Box::new(TileMapExpr::Value(value)),
                    },
                    rows: cols,
                    cols: rows,
                },
                TileIrType::Tile {
                    rows: cols,
                    cols: rows,
                },
            )
        }
        // Every unary float operation is pointwise, so it rides the map node:
        // the map already carries a scalar expression per element, and one op
        // per transcendental would grow every traversal of the IR for no gain.
        // A scalar operand makes the map a scalar, which the lowering reads off
        // the result type.
        BuiltinOp::Unary(unary) => {
            let operand = lower_expr(&args[0], ctx);
            let operand_shape = tile_shape_of(operand, ctx);
            let (rows, cols) = operand_shape.unwrap_or((1, 1));
            let ty = match operand_shape {
                Some(_) => TileIrType::Tile { rows, cols },
                None => TileIrType::F32,
            };
            ctx.emit(
                TileIrOp::Map {
                    expr: TileMapExpr::Unary {
                        op: unary,
                        operand: Box::new(TileMapExpr::Value(operand)),
                    },
                    rows,
                    cols,
                },
                ty,
            )
        }
        // A comparison yields `0.0`/`1.0` per element, so it is a map too. The
        // select that consumes it reads the same coordinates, which is what
        // makes `select(gt_tile(a, b), a, b)` a pointwise maximum.
        BuiltinOp::Gt | BuiltinOp::Lt | BuiltinOp::Ge | BuiltinOp::Le | BuiltinOp::Eq
        | BuiltinOp::Ne => {
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            let cmp = compare_op(op).expect("a comparison names a comparison");
            lower_compare(cmp, lhs, rhs, ctx)
        }
        BuiltinOp::Select => {
            let cond = lower_expr(&args[0], ctx);
            let on_true = lower_expr(&args[1], ctx);
            let on_false = lower_expr(&args[2], ctx);
            let shape = tile_shape_of(on_true, ctx)
                .or_else(|| tile_shape_of(on_false, ctx))
                .or_else(|| tile_shape_of(cond, ctx));
            let (rows, cols) = shape.unwrap_or((1, 1));
            ctx.emit(
                TileIrOp::Map {
                    expr: TileMapExpr::Select {
                        cond: Box::new(TileMapExpr::Value(cond)),
                        on_true: Box::new(TileMapExpr::Value(on_true)),
                        on_false: Box::new(TileMapExpr::Value(on_false)),
                    },
                    rows,
                    cols,
                },
                pointwise_result_ty(shape, &[on_true, on_false], ctx),
            )
        }
        // `iota(shape![N])` is the sequence `0, 1, …, N-1`, so its map reads the
        // coordinate it walks and nothing else. CuTile names it for a rank-one
        // shape, which is one row of N columns here.
        BuiltinOp::Iota => {
            let shape = args.first().map(|arg| extract_const_shape(arg, ctx));
            let (rows, cols) = shape_to_2d(&shape.unwrap_or_default());
            ctx.emit(
                TileIrOp::Map {
                    expr: TileMapExpr::Iota,
                    rows,
                    cols,
                },
                TileIrType::Tile { rows, cols },
            )
        }
        // `tile_to_scalar(t)` reads element `(0, 0)`. A one-element tile has
        // one element, so the read is the value.
        BuiltinOp::TileToScalar => {
            let tile = lower_expr(&args[0], ctx);
            let zero = ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64);
            ctx.emit(
                TileIrOp::Extract {
                    tile,
                    row_coord: zero,
                    col_coord: zero,
                },
                TileIrType::F32,
            )
        }
        // `scalar_to_tile(s)` splats one value into a one-element tile.
        BuiltinOp::ScalarToTile => {
            let value = lower_expr(&args[0], ctx);
            let (rows, cols) = (1, 1);
            ctx.emit(
                TileIrOp::Map {
                    expr: TileMapExpr::Value(value),
                    rows,
                    cols,
                },
                TileIrType::Tile { rows, cols },
            )
        }
        // `convert_scalar(x)` is the identity: every element here is f32.
        // `convert_scalar(x)` names a scalar constant as an element value, so
        // its result is a float. `let n: f32 = convert_scalar(N);` is how CuTile
        // writes a count into an arithmetic expression.
        BuiltinOp::ConvertScalar => {
            let value = args.first().and_then(|arg| try_eval_i64(arg, ctx)).unwrap_or(0);
            ctx.emit(TileIrOp::ConstF64(value as f64), TileIrType::F32)
        }
        // `reduce_min` is the same reduction with the other combiner. The tile
        // IR carries the combiner, so the op needs one more variant and no new
        // shape rule.
        BuiltinOp::ReduceMin => {
            let value = lower_expr(&args[0], ctx);
            let (in_rows, in_cols) = tile_shape_of(value, ctx).unwrap_or((1, 1));
            let requested_axis = args.get(1).map(|arg| eval_i64(arg, ctx)).unwrap_or(1);
            let axis = if in_rows == 1 { 1 } else { requested_axis };
            let (out_rows, out_cols) = if axis == 1 {
                (in_rows, 1)
            } else {
                (1, in_cols)
            };
            ctx.emit(
                TileIrOp::ReduceMin {
                    value,
                    axis,
                    in_rows,
                    in_cols,
                },
                TileIrType::Tile {
                    rows: out_rows,
                    cols: out_cols,
                },
            )
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
        BuiltinOp::ShapeOf => {
            // A shape is metadata, not a runtime value. Returning the operand
            // would let a later scalar use read a buffer pointer as if it were
            // an extent.
            ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
        }
        BuiltinOp::ScalarDiv => {
            // Scalar division is integer division, the same operation the
            // binary `Div` performs when neither side is a tile.
            let lhs = lower_expr(&args[0], ctx);
            let rhs = lower_expr(&args[1], ctx);
            ctx.emit(
                TileIrOp::IBinary {
                    op: BinOp::Div,
                    lhs,
                    rhs,
                },
                TileIrType::I64,
            )
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

fn lower_store(target: &str, value: &Expr, index: Option<&Expr>, ctx: &mut LowerCtx<'_>) {
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
    // `y.store(t)` names no coordinates, so the tile is the one this program
    // owns and the launch index addresses it. `part.store(t, [i, j])` names
    // them, and then the launch index plays no part: the store writes the tile
    // at the coordinates the kernel computed.
    let (plane, row_coord, col_coord) = match index {
        Some(index) => {
            let coords = match index {
                Expr::Shape(_) => extract_runtime_coords(index, ctx),
                other => vec![lower_expr(other, ctx)],
            };
            let (row, col) = coords_to_2d(&coords, ctx);
            (outer_coords(&coords), row, col)
        }
        None if output_rank >= 3 => program_tile_coords(output_rank, ctx),
        None if output_rank <= 1 => (
            Vec::new(),
            ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64),
            ctx.get_or_create_program_id_param(0),
        ),
        None => {
            let row = ctx.get_or_create_program_id_param(0);
            let col = if rows > 1 || cols > 1 {
                ctx.get_or_create_program_id_param(1)
            } else {
                ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64)
            };
            (Vec::new(), row, col)
        }
    };

    let stride_shape_idx = param_stride_dim(&output, ctx);

    ctx.emit(
        TileIrOp::StorePtrTko {
            buf: output,
            value: stored,
            outer_coords: plane,
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
            outer_coords: outer_coords(&coords),
            row_coord,
            col_coord,
            stride_shape_idx,
        },
        TileIrType::Void,
    );
}

// ── Helpers ────────────────────────────────────────────────────────

/// An arithmetic operation over two values.
///
/// A tile operand makes it a tile operation, and a scalar chain that touches a
/// float stays float: `rms / n` divides an f32 by a count, and integer division
/// would round the mean down before the square root ever sees it.
fn lower_arith(op: BinOp, lhs: ValueId, rhs: ValueId, ctx: &mut LowerCtx<'_>) -> ValueId {
    let shape = tile_shape_of(lhs, ctx).or_else(|| tile_shape_of(rhs, ctx));
    let Some((rows, cols)) = shape else {
        if is_float_operand(lhs, ctx) || is_float_operand(rhs, ctx) {
            return ctx.emit(TileIrOp::FBinary { op, lhs, rhs }, TileIrType::F32);
        }
        return ctx.emit(TileIrOp::IBinary { op, lhs, rhs }, TileIrType::I64);
    };
    // No target has a fused tile remainder, so it rides the map node the way
    // the unary functions do.
    if op == BinOp::Rem {
        return ctx.emit(
            TileIrOp::Map {
                expr: TileMapExpr::Rem {
                    lhs: Box::new(TileMapExpr::Value(lhs)),
                    rhs: Box::new(TileMapExpr::Value(rhs)),
                },
                rows,
                cols,
            },
            TileIrType::Tile { rows, cols },
        );
    }
    let op = match op {
        BinOp::Add => TileIrOp::AddF { lhs, rhs, rows, cols },
        BinOp::Sub => TileIrOp::SubF { lhs, rhs, rows, cols },
        BinOp::Mul => TileIrOp::MulF { lhs, rhs, rows, cols },
        BinOp::Div => TileIrOp::DivF { lhs, rhs, rows, cols },
        BinOp::Rem => unreachable!("a remainder rides the map node"),
        BinOp::And | BinOp::Or => {
            unreachable!("a bitwise operation is not a tile operation")
        }
    };
    ctx.emit(op, TileIrType::Tile { rows, cols })
}

/// A comparison of two values, as the `0.0`/`1.0` mask this IR reads.
///
/// A tile operand makes the comparison elementwise, so the result is a tile of
/// the same shape. Without one it is a scalar comparison, and the operand type
/// carries through: `j >= mask_start` compares two counts, and the count it
/// yields is what a following division needs.
fn lower_compare(cmp: CmpOp, lhs: ValueId, rhs: ValueId, ctx: &mut LowerCtx<'_>) -> ValueId {
    let shape = tile_shape_of(lhs, ctx).or_else(|| tile_shape_of(rhs, ctx));
    let (rows, cols) = shape.unwrap_or((1, 1));
    ctx.emit(
        TileIrOp::Map {
            expr: TileMapExpr::Cmp {
                op: cmp,
                lhs: Box::new(TileMapExpr::Value(lhs)),
                rhs: Box::new(TileMapExpr::Value(rhs)),
            },
            rows,
            cols,
        },
        pointwise_result_ty(shape, &[lhs, rhs], ctx),
    )
}

/// The larger, or the smaller, of two values, element by element.
///
/// A maximum is one comparison and one select, which is what the map node
/// carries, and no target has an instruction that does more.
fn lower_extreme(max: bool, lhs: ValueId, rhs: ValueId, ctx: &mut LowerCtx<'_>) -> ValueId {
    let shape = tile_shape_of(lhs, ctx).or_else(|| tile_shape_of(rhs, ctx));
    let (rows, cols) = shape.unwrap_or((1, 1));
    let op = if max { CmpOp::Ge } else { CmpOp::Le };
    ctx.emit(
        TileIrOp::Map {
            expr: TileMapExpr::Select {
                cond: Box::new(TileMapExpr::Cmp {
                    op,
                    lhs: Box::new(TileMapExpr::Value(lhs)),
                    rhs: Box::new(TileMapExpr::Value(rhs)),
                }),
                on_true: Box::new(TileMapExpr::Value(lhs)),
                on_false: Box::new(TileMapExpr::Value(rhs)),
            },
            rows,
            cols,
        },
        pointwise_result_ty(shape, &[lhs, rhs], ctx),
    )
}

/// The type a pointwise operation over these operands produces.
///
/// An operation with a tile operand is a tile of that operand's shape. Without
/// one it is a scalar, and the operands say which kind: a count stays a count,
/// and mixing in a float makes the result a float. `min(mask_start,
/// k_seqlen_tiles)` compares two counts, and the smaller one is still a count.
fn pointwise_result_ty(
    shape: Option<(i64, i64)>,
    operands: &[ValueId],
    ctx: &LowerCtx<'_>,
) -> TileIrType {
    match shape {
        Some((rows, cols)) => TileIrType::Tile { rows, cols },
        None if operands.iter().any(|value| is_float_operand(*value, ctx)) => TileIrType::F32,
        None => TileIrType::I64,
    }
}

/// Whether a value is a float, either an element value or a scalar.
fn is_float_operand(value: ValueId, ctx: &LowerCtx<'_>) -> bool {
    matches!(
        ctx.types.get(&value),
        Some(TileIrType::F32 | TileIrType::Tile { .. })
    )
}

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
        Expr::Builtin {
            op: BuiltinOp::Rem,
            args,
        } => {
            let lhs = try_eval_i64(args.first()?, ctx)?;
            let rhs = try_eval_i64(args.get(1)?, ctx)?;
            (rhs != 0).then_some(lhs % rhs)
        }
        // A comparison of two constants is a constant, and a mask is `1` where
        // it holds and `0` where it does not. `&` and `|` combine those the way
        // they combine any two counts, which is what lets `if CAUSAL == 1` and
        // `if (CAUSAL == 1) || (EVEN_K == 0)` both fold to one answer.
        Expr::Builtin {
            op:
                op @ (BuiltinOp::Eq
                | BuiltinOp::Ne
                | BuiltinOp::Lt
                | BuiltinOp::Le
                | BuiltinOp::Gt
                | BuiltinOp::Ge
                | BuiltinOp::And
                | BuiltinOp::Or),
            args,
        } => {
            let lhs = try_eval_i64(args.first()?, ctx)?;
            let rhs = try_eval_i64(args.get(1)?, ctx)?;
            let value = match op {
                BuiltinOp::Eq => lhs == rhs,
                BuiltinOp::Ne => lhs != rhs,
                BuiltinOp::Lt => lhs < rhs,
                BuiltinOp::Le => lhs <= rhs,
                BuiltinOp::Gt => lhs > rhs,
                BuiltinOp::Ge => lhs >= rhs,
                BuiltinOp::And => (lhs & rhs) != 0,
                BuiltinOp::Or => (lhs | rhs) != 0,
                _ => unreachable!("the arm names the operations it reads"),
            };
            Some(value as i64)
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
    // `x.shape()` names the shape `x` was declared with, which the checker put
    // on the binding's type. A dynamic extent has no compile-time value, so a
    // shape that holds one is not a constant shape.
    if let Expr::Builtin {
        op: BuiltinOp::ShapeOf,
        args,
    } = expr
    {
        return args
            .first()
            .and_then(|arg| expr_shape(arg, ctx))
            .filter(|dims| dims.iter().all(|dim| *dim >= 0))
            .unwrap_or_default();
    }
    let shape = match expr {
        Expr::Shape(shape) => shape.clone(),
        // A shape bound to a name (`let ts = shape![1, 8];`) carries its
        // dimensions in the binding's type, the same way a partition does.
        Expr::Var(name) => match ctx.typed.locals.get(name) {
            Some(sile_hir::Type::Shape(shape)) | Some(sile_hir::Type::Tile { shape, .. }) => {
                shape.clone()
            }
            _ => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    match shape {
        ShapeExpr::Tuple(dims) => dims.iter().map(|dim| resolve_shape_dim(dim, ctx)).collect(),
        other => vec![resolve_shape_dim(&other, ctx)],
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

/// Tile extent a partition was declared with.
///
/// The partition lowers to its buffer, so its extent is not in the value table.
/// The HIR type of the partition does carry it, which is why this reads the
/// typed kernel rather than the value the partition produced.
fn partition_tile_extent(partition: &Expr, ctx: &LowerCtx<'_>) -> Vec<i64> {
    // A partition written inline already carries its shape; one bound to a name
    // carries it in the binding's type, which the checker recorded.
    if let Expr::Builtin {
        op: BuiltinOp::Partition,
        args,
    } = partition
        && let Some(shape) = args.get(1)
    {
        let extent = extract_const_shape(shape, ctx);
        if !extent.is_empty() {
            return extent;
        }
    }
    let ty = match partition {
        Expr::Var(name) => ctx.typed.locals.get(name),
        _ => None,
    };
    match ty {
        Some(sile_hir::Type::Partition { shape, .. }) => match shape {
            ShapeExpr::Tuple(dims) => dims.iter().map(|dim| resolve_shape_dim(dim, ctx)).collect(),
            other => vec![resolve_shape_dim(other, ctx)],
        },
        _ => Vec::new(),
    }
}

fn extract_runtime_coords(expr: &Expr, ctx: &mut LowerCtx<'_>) -> Vec<ValueId> {
    match expr {
        // A tile index is a list of runtime scalars, so each element is lowered
        // as a value. Matching only `Shape` here would silently drop any
        // arithmetic the index carries, and the load would fall back to the raw
        // program id.
        Expr::TileIndex(elems) => elems.iter().map(|elem| lower_expr(elem, ctx)).collect(),
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
        // A tile is two-dimensional, so its last two extents name it and the
        // extents above them are the planes the tile sits in. A tile shape must
        // have one of each of those, which the checker enforces; a buffer shape
        // may have any number, and the program that owns the tile picks one.
        [.., rows, cols] => (*rows, *cols),
    }
}

/// The two coordinates that name a tile.
///
/// A tile covers the last two axes of the buffer, so the tile is always the
/// last two coordinates, whatever the rank. [`outer_coords`] reads the rest.
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
        [.., row, col] => (*row, *col),
    }
}

/// The coordinates of the axes above the tile, outermost first.
///
/// A tile covers the last two axes of the buffer, so every axis above them is
/// a plane of planes and its coordinate is one of these. A rank-1 or rank-2
/// index names the tile alone, so the list is empty.
fn outer_coords(coords: &[ValueId]) -> Vec<ValueId> {
    if coords.len() <= 2 {
        Vec::new()
    } else {
        coords[..coords.len() - 2].to_vec()
    }
}

/// The coordinates of the tile the program owns.
///
/// The launch index is the program's position in the grid, and the grid has one
/// axis for each buffer axis the tile does not cover. A rank-2 buffer addresses
/// its tile with `(row, col)`, so the launch index is those two. A rank-3 buffer
/// addresses a plane of tiles with `(plane, row, col)`, and the launch index is
/// those three. The tile is the last two axes in both cases.
///
/// The launch index fills the buffer axes from the outermost inward, because a
/// grid has three axes at most. An axis the grid does not reach is zero: that
/// is the tile the program owns when the buffer is wider than the grid, which
/// is what a tile covering the whole of a trailing axis needs.
fn program_tile_coords(rank: usize, ctx: &mut LowerCtx<'_>) -> (Vec<ValueId>, ValueId, ValueId) {
    if rank < 3 {
        return (
            Vec::new(),
            ctx.get_or_create_program_id_param(0),
            ctx.get_or_create_program_id_param(1),
        );
    }
    let axis_coord = |ctx: &mut LowerCtx<'_>, axis: usize| match axis {
        0 | 1 | 2 => ctx.get_or_create_program_id_param(axis as i64),
        _ => ctx.emit(TileIrOp::ConstI64(0), TileIrType::I64),
    };
    let outer: Vec<ValueId> = (0..rank - 2).map(|axis| axis_coord(ctx, axis)).collect();
    let row_coord = axis_coord(ctx, rank - 2);
    let col_coord = axis_coord(ctx, rank - 1);
    (outer, row_coord, col_coord)
}

/// How many axes the buffer has.
///
/// A value that is not a buffer has no axes to speak of; two is the answer that
/// keeps a non-buffer access addressing one tile with a row and a column, which
/// is what every such access already did.
fn buffer_rank(value: ValueId, ctx: &LowerCtx<'_>) -> usize {
    match ctx.types.get(&value) {
        Some(TileIrType::Buffer { rank }) => *rank,
        _ => 2,
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
    // A rank-3 declared shape names a plane extent and then the tile, the same
    // way a rank-3 tile shape does.
    let dims = match shape {
        ShapeExpr::Tuple(dims) if dims.len() == 2 => dims,
        ShapeExpr::Tuple(dims) if dims.len() == 3 => &dims[1..],
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
            // The row stride is the innermost extent, which is the last shape
            // dimension: index 1 of a rank-2 buffer, index 2 of a rank-3 one.
            return rank.saturating_sub(1);
        }
    }
    1
}

/// Tile extent this buffer is loaded with, along one axis, for `num_tiles`.
///
/// `num_tiles(buf, axis)` divides the buffer's extent along `axis` by the tile's
/// extent along the same axis, so the lowering has to find the tile shape the
/// buffer is used with. A partition names it (`x.partition(shape![BM, BK])`) and
/// so does a direct load (`x.load_tile([BK], [k])`); both spellings describe the
/// same tiling, so both are read. The kernel body is a single forward pass and
/// `num_tiles` may appear before the load, so the search is order-independent.
fn load_tile_extent_for_buffer(buf: ValueId, axis: usize, ctx: &LowerCtx<'_>) -> i64 {
    let Some(index) = ctx.mir_params.iter().position(|param| param.value == buf) else {
        return 1;
    };
    let hir_param = &ctx.typed.kernel.params[index];
    let buffer_rank = rank_of_param(hir_param);
    let mut shapes = Vec::new();
    for stmt in &ctx.typed.kernel.body {
        collect_buffer_tile_shapes(stmt, &hir_param.name, ctx, &mut shapes);
    }
    // The tile covers the buffer's last axes, so a tile shape shorter than the
    // buffer lines up with the trailing axes and each leading axis the tile does
    // not name holds one tile per element.
    let extent = |dims: &[i64]| -> i64 {
        let leading = buffer_rank.saturating_sub(dims.len());
        if axis < leading {
            1
        } else {
            dims.get(axis - leading).copied().unwrap_or(1)
        }
    };
    // One buffer normally uses a single tile shape; take the largest extent so a
    // tile count is never inflated by a smaller probe load.
    shapes
        .iter()
        .map(|dims| extent(dims))
        .max()
        .unwrap_or(1)
}

/// Append the tile shapes `buf_name` is loaded with anywhere in `stmt`.
fn collect_buffer_tile_shapes(
    stmt: &sile_hir::Stmt,
    buf_name: &str,
    ctx: &LowerCtx<'_>,
    out: &mut Vec<Vec<i64>>,
) {
    match stmt {
        sile_hir::Stmt::Let { expr, .. } | sile_hir::Stmt::Assign { expr, .. } => {
            collect_tile_shapes_in_expr(expr, buf_name, ctx, out);
        }
        sile_hir::Stmt::ForLoop { body, .. } => {
            for inner in body {
                collect_buffer_tile_shapes(inner, buf_name, ctx, out);
            }
        }
        sile_hir::Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            for inner in then_body.iter().chain(else_body) {
                collect_buffer_tile_shapes(inner, buf_name, ctx, out);
            }
        }
        sile_hir::Stmt::Store { value, .. } => {
            collect_tile_shapes_in_expr(value, buf_name, ctx, out);
        }
        sile_hir::Stmt::AtomicAdd { index, value, .. } => {
            collect_tile_shapes_in_expr(index, buf_name, ctx, out);
            collect_tile_shapes_in_expr(value, buf_name, ctx, out);
        }
        sile_hir::Stmt::Continue => {}
    }
}

fn collect_tile_shapes_in_expr(
    expr: &sile_hir::Expr,
    buf_name: &str,
    ctx: &LowerCtx<'_>,
    out: &mut Vec<Vec<i64>>,
) {
    let sile_hir::Expr::Builtin { op, args } = expr else {
        return;
    };
    match op {
        // `x.load_tile([BK], [k])`: the tile shape is the second argument.
        BuiltinOp::LoadTile => {
            if matches!(args.first(), Some(sile_hir::Expr::Var(name)) if name == buf_name)
                && let Some(dims) = args.get(1).map(|arg| extract_const_shape(arg, ctx))
                && !dims.is_empty()
            {
                out.push(dims);
            }
        }
        // `part.load([i, j])` and `x.load()`: the tile shape belongs to the
        // partition or to the buffer itself, and the checker recorded it on the
        // binding's type.
        BuiltinOp::Load => {
            let Some(sile_hir::Expr::Var(name)) = args.first() else {
                return;
            };
            // A bare `x.load()` names the buffer's whole declared shape, so it
            // describes one tile and adds nothing to the search.
            let Some(sile_hir::Type::Partition {
                source: Some(source),
                shape,
                ..
            }) = ctx.typed.locals.get(name)
            else {
                return;
            };
            if source != buf_name {
                return;
            }
            let dims = extract_const_shape(&sile_hir::Expr::Shape(shape.clone()), ctx);
            if !dims.is_empty() {
                out.push(dims);
            }
        }
        _ => {}
    }
    // A tile another expression produced is a load too: `num_tiles` may be
    // written in terms of it.
    for arg in args {
        collect_tile_shapes_in_expr(arg, buf_name, ctx, out);
    }
}

fn rank_of_param(param: &sile_hir::Param) -> usize {
    match &param.ty {
        sile_hir::Type::Tensor { shape, .. }
        | sile_hir::Type::Tile { shape, .. }
        | sile_hir::Type::Partition { shape, .. } => shape.rank(),
        sile_hir::Type::Shape(_) | sile_hir::Type::Scalar(_) => 0,
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
        sile_hir::Type::Tensor { shape, .. }
        | sile_hir::Type::Tile { shape, .. }
        | sile_hir::Type::Partition { shape, .. } => match shape {
            ShapeExpr::Tuple(dims) => {
                Some(dims.iter().map(|d| resolve_shape_dim(d, ctx)).collect())
            }
            other => Some(vec![resolve_shape_dim(other, ctx)]),
        },
        // A shape bound to a name carries its extents, so a reshape can name
        // it: `let tile_shape = shape![1, BLOCK_SIZE]; t.reshape(tile_shape)`.
        sile_hir::Type::Shape(ShapeExpr::Tuple(dims)) => {
            Some(dims.iter().map(|d| resolve_shape_dim(d, ctx)).collect())
        }
        sile_hir::Type::Shape(_) | sile_hir::Type::Scalar(_) => None,
    }
}
