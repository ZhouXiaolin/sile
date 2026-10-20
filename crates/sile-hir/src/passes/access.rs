//! The access checks cutile decides while compiling.
//!
//! cutile-rs proves a tile access from three facts — the index's interval, the
//! static extent, and the tile grid — before it emits anything. That proof is
//! this pass: it walks beside the lowering, and every decision the printed text
//! reflects is made here, in [`predicate`]'s algebra. The lowering consumes the
//! decisions and emits; it decides nothing itself.
//!
//! The proof ladder has four rungs, and a goal falls to the first one that
//! holds it:
//!
//! 1. **Inspection** — the index interval against the tile count proves the
//!    access outright (`fold_static`). A violation here is a compile error.
//! 2. **Belief** — a dominating assumption already entails the goal, which is
//!    what a loop over `num_tiles` states about its variable.
//! 3. **Launch** — the goal ranges only over launch-known quantities, so the
//!    host evaluates it before the kernel runs.
//! 4. **Device** — nothing above holds, so the kernel asserts it at run time.

use super::predicate::{Atom, LaunchCheck, Predicate, Term};
use super::{bounds, value_ids};
use crate::{BuiltinOp, Expr, Kernel, ShapeExpr, Stmt, Type};

/// What one view axis is, as the checks see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AxisExtent {
    /// A static extent: the checks decide against its value.
    Static(i64),
    /// A run-time value the prologue assumed.
    Dynamic,
}

/// The facts about one tensor the checks name.
///
/// The lowering holds the tensor's values; this pass holds only what a check
/// reads — which parameter the tensor came from, its tile, and what each view
/// axis is.
#[derive(Clone, Debug)]
pub struct TensorFacts {
    pub name: String,
    /// The tensor's position among the buffer parameters, which is what a
    /// launch check names.
    pub param: Option<usize>,
    /// The declared tile extents, `None` where dynamic.
    pub tile: Vec<Option<i64>>,
    /// What each view axis is.
    pub view_axes: Vec<AxisExtent>,
}

/// What one axis's access check decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisDecision {
    /// Proven by inspection or already believed: nothing is emitted.
    Discharged,
    /// The device asserts this axis at run time.
    Device {
        /// The index is known non-negative, so the assert tests only the upper
        /// bound.
        proven_non_negative: bool,
    },
}

/// What the checks state about a kernel before anything is emitted.
///
/// These are pure facts of the kernel: a pure function of its AST, computed
/// once at check time and carried by the [`crate::typeck::TypedKernel`]. The
/// lowering reads them; it never recomputes them.
#[derive(Clone, Debug)]
pub struct KernelChecks {
    /// The stable id of every binding, so an induction variable is named by
    /// identity rather than by name.
    pub value_ids: value_ids::ValueIds,
    /// The generics the body compares against a literal; cutile prints `1`.
    pub flags: Vec<String>,
    /// Whether the body partitions a buffer by generics a parameter names.
    pub instantiates_generics: bool,
}

impl KernelChecks {
    pub fn of(kernel: &Kernel) -> Self {
        Self {
            value_ids: value_ids::ValueIds::of(kernel),
            flags: flag_generics(kernel),
            instantiates_generics: instantiates_generics(kernel),
        }
    }
}

/// The checks' state, advanced by the lowering as it walks the body.
///
/// Everything here is decided without emitting: an interval, a predicate, a
/// goal. The lowering asks the questions; this struct keeps the answers.
pub struct AccessAnalysis {
    /// The stable id of every binding, so an induction variable is named by
    /// identity rather than by name.
    pub value_ids: value_ids::ValueIds,
    /// The goals already believed at this point in the body.
    assumptions: Vec<Predicate>,
    /// The access checks that failed at compile time.
    pub errors: Vec<String>,
    /// The checks the host evaluates before it launches.
    pub launch_checks: Vec<LaunchCheck>,
    /// The interval of every integer binding.
    bounds: Vec<(String, bounds::Bounds)>,
    /// The canonical linear term of every integer binding.
    terms: Vec<(String, Term)>,
    /// The names bound by enclosing `for` loops.
    loop_vars: Vec<String>,
    /// The loop variables whose trip count is a compile-time constant.
    bounded_loop_vars: Vec<String>,
    /// The bindings whose value is an integer tile.
    int_bindings: Vec<String>,
    /// The scalar parameters of integer type.
    int_scalars: Vec<String>,
    /// The bindings whose value is a component of the block id.
    block_id_bindings: Vec<String>,
    /// Every tensor the checks can name.
    tensors: Vec<TensorFacts>,
    /// The generics the body compares against a literal; cutile prints `1`.
    flags: Vec<String>,
    /// Whether the body partitions a buffer by generics a parameter names.
    instantiates: bool,
}

impl AccessAnalysis {
    pub fn new(checks: &KernelChecks) -> Self {
        Self {
            value_ids: checks.value_ids.clone(),
            assumptions: Vec::new(),
            errors: Vec::new(),
            launch_checks: Vec::new(),
            bounds: Vec::new(),
            terms: Vec::new(),
            loop_vars: Vec::new(),
            bounded_loop_vars: Vec::new(),
            int_bindings: Vec::new(),
            int_scalars: Vec::new(),
            block_id_bindings: Vec::new(),
            tensors: Vec::new(),
            flags: checks.flags.clone(),
            instantiates: checks.instantiates_generics,
        }
    }

    /// The generics the body compares against a literal.
    pub fn flags(&self) -> &[String] {
        &self.flags
    }

    /// Whether the body partitions a buffer by generics a parameter names.
    pub fn instantiates_generics(&self) -> bool {
        self.instantiates
    }

    /// Record a tensor the checks can name.
    pub fn register_tensor(&mut self, facts: TensorFacts) {
        self.tensors.push(facts);
    }

    /// Record that a binding's value is an integer tile.
    pub fn bind_int(&mut self, name: &str) {
        self.int_bindings.push(name.to_string());
    }

    /// Record that a binding's value is one component of the block id.
    pub fn mark_block_id(&mut self, name: &str) {
        self.block_id_bindings.push(name.to_string());
    }

    /// Record what a `let` proves about its binding: an interval, a term.
    pub fn record_binding(&mut self, kernel: &Kernel, name: &str, expr: &Expr) {
        if let Some(bound) = self.bounds_of(kernel, expr) {
            self.bounds.push((name.to_string(), bound));
        }
        if let Some(term) = self.term_of(kernel, expr) {
            self.terms.push((name.to_string(), term));
        }
    }

    /// Record what entering a `for` proves about its variable.
    pub fn enter_loop(&mut self, kernel: &Kernel, var: &str, bound: Option<i64>) {
        self.loop_vars.push(var.to_string());
        let Some(upper) = bound else { return };
        self.bounded_loop_vars.push(var.to_string());
        self.bounds
            .push((var.to_string(), bounds::Bounds { start: 0, end: upper }));
        if let Some(id) = self.value_ids.id(var) {
            self.terms
                .push((var.to_string(), Term::atom(Atom::Iv(id))));
        }
        let _ = kernel;
    }

    /// Forget what a `for` proved once its body is done.
    pub fn exit_loop(&mut self, var: &str) {
        self.loop_vars.retain(|name| name != var);
        self.bounded_loop_vars.retain(|name| name != var);
    }

    /// Whether a name denotes one component of the block id.
    ///
    /// The launch grid bounds every component, so an access indexed by one
    /// needs no check. A value computed from a component has lost that bound.
    pub fn is_block_id_component(&self, name: &str) -> bool {
        self.block_id_bindings.iter().any(|bound| bound == name)
    }

    /// The value of a comparison between two compile-time constants.
    ///
    /// A const generic on either side is its printed value: a flag generic
    /// prints `1`, because the body only ever compares it.
    pub fn literal_comparison(&self, kernel: &Kernel, expr: &Expr) -> Option<bool> {
        let Expr::Builtin { op, args } = expr else {
            return None;
        };
        let relation = match op {
            BuiltinOp::Eq => "equal",
            BuiltinOp::Ne => "not_equal",
            BuiltinOp::Lt => "less_than",
            BuiltinOp::Le => "less_than_or_equal",
            BuiltinOp::Gt => "greater_than",
            BuiltinOp::Ge => "greater_than_or_equal",
            _ => return None,
        };
        let value_of_arg = |arg: &Expr| -> Option<i64> {
            match arg {
                Expr::ScalarI64(value) => Some(*value),
                Expr::Var(name) => {
                    let (_, value) = kernel.const_params.iter().find(|(n, _)| n == name)?;
                    let printed = if self.flags.iter().any(|flag| flag == name) {
                        1
                    } else {
                        *value
                    };
                    Some(printed)
                }
                _ => None,
            }
        };
        let lhs = value_of_arg(args.first()?)?;
        let rhs = value_of_arg(args.get(1)?)?;
        Some(match relation {
            "equal" => lhs == rhs,
            "not_equal" => lhs != rhs,
            "less_than" => lhs < rhs,
            "less_than_or_equal" => lhs <= rhs,
            "greater_than" => lhs > rhs,
            _ => lhs >= rhs,
        })
    }

    /// Whether an operation works on integers.
    ///
    /// A bound is one witness, but not the only one: a block id and a shape
    /// extent are integers whose range cutile leaves unknown, and their
    /// division still prints as `divi`, not `divf`.
    pub fn is_integer_expr(&self, kernel: &Kernel, args: &[Expr]) -> bool {
        fn integer(analysis: &AccessAnalysis, kernel: &Kernel, expr: &Expr) -> bool {
            match expr {
                Expr::ScalarI64(_) => true,
                Expr::Var(name) => {
                    if kernel.const_params.iter().any(|(n, _)| n == name) {
                        return true;
                    }
                    // A scalar parameter of integer type divides as an integer.
                    if analysis.int_scalars.iter().any(|n| n == name) {
                        return true;
                    }
                    if analysis.int_bindings.iter().any(|n| n == name) {
                        return true;
                    }
                    analysis.bounds.iter().any(|(n, _)| n == name)
                }
                Expr::Builtin { op, args } => match op {
                    BuiltinOp::ProgramId | BuiltinOp::ShapeDim => true,
                    BuiltinOp::Add
                    | BuiltinOp::Sub
                    | BuiltinOp::Mul
                    | BuiltinOp::Div
                    | BuiltinOp::Rem
                    | BuiltinOp::CeilDiv
                    | BuiltinOp::Min
                    | BuiltinOp::Max => args
                        .iter()
                        .all(|arg| integer(analysis, kernel, arg)),
                    _ => false,
                },
                _ => false,
            }
        }
        args.iter().all(|arg| integer(self, kernel, arg))
    }

    /// The canonical linear term of an integer expression.
    ///
    /// This is the fragment the predicate algebra reasons over: a sum of atoms
    /// with coefficients, plus a constant. Anything outside the fragment — a
    /// product of two unknowns, a remainder — is not formed, so a goal over it
    /// sinks to a later stage instead of being decided wrongly.
    pub fn term_of(&self, kernel: &Kernel, expr: &Expr) -> Option<Term> {
        match expr {
            Expr::ScalarI64(value) => Some(Term::constant(*value)),
            Expr::Var(name) => {
                if let Some((_, term)) = self.terms.iter().rev().find(|(n, _)| n == name) {
                    return Some(term.clone());
                }
                // An induction variable is a name for its own value: `1 * iv`.
                // The id is the binding's, not the name's, so a shadowed name
                // still names the value the body means.
                if self.loop_vars.iter().any(|var| var == name) {
                    return self
                        .value_ids
                        .id(name)
                        .map(|id| Term::atom(Atom::Iv(id)));
                }
                // A const generic is a JIT-time constant.
                kernel
                    .const_params
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, value)| Term::constant(*value))
            }
            Expr::Builtin { op, args } => {
                let lhs = self.term_of(kernel, args.first()?)?;
                let rhs = match args.get(1) {
                    Some(rhs) => self.term_of(kernel, rhs)?,
                    None => Term::constant(0),
                };
                match op {
                    BuiltinOp::Add => lhs.add(&rhs),
                    BuiltinOp::Sub => lhs.sub(&rhs),
                    BuiltinOp::Mul => {
                        // Only a constant times a term stays linear.
                        match (lhs.as_constant(), rhs.as_constant()) {
                            (Some(k), _) => rhs.mul_const(k),
                            (_, Some(k)) => lhs.mul_const(k),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The interval of an integer expression, when the pass can prove one.
    pub fn bounds_of(&self, kernel: &Kernel, expr: &Expr) -> Option<bounds::Bounds> {
        match expr {
            Expr::ScalarI64(value) => Some(bounds::Bounds::exact(*value)),
            Expr::Var(name) => {
                if let Some((_, bound)) = self.bounds.iter().rev().find(|(n, _)| n == name) {
                    return Some(*bound);
                }
                // A const generic is a known integer.
                kernel
                    .const_params
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, value)| bounds::Bounds::exact(*value))
            }
            // `num_tiles(&p, axis)`: the tile count of that axis, which is
            // `ceil(extent / tile)` and exact when both are static.
            Expr::Builtin {
                op: BuiltinOp::NumTiles,
                args,
            } => {
                let Expr::Var(name) = args.first()? else {
                    return None;
                };
                let axis = match args.get(1)? {
                    Expr::ScalarI64(axis) => *axis as usize,
                    _ => return None,
                };
                let tensor = self.tensors.iter().find(|t| t.name == *name)?;
                let extent = match tensor.view_axes.get(axis)? {
                    AxisExtent::Static(value) => *value,
                    AxisExtent::Dynamic => return None,
                };
                let tile = tensor.tile.get(axis)?.filter(|t| *t > 0)?;
                Some(bounds::Bounds::exact((extent + tile - 1) / tile))
            }
            Expr::Builtin { op, args } => {
                let lhs = self.bounds_of(kernel, args.first()?)?;
                let rhs = match args.get(1) {
                    Some(rhs) => self.bounds_of(kernel, rhs)?,
                    None => bounds::Bounds::exact(0),
                };
                bounds::transfer(*op, lhs, rhs)
            }
            _ => None,
        }
    }

    /// The constant an integer operation folds to, when its interval is one
    /// value.
    pub fn fold_integer_value(
        &self,
        op: BuiltinOp,
        args: &[Expr],
        kernel: &Kernel,
    ) -> Option<i64> {
        let lhs = self.bounds_of(kernel, args.first()?)?;
        let rhs = match args.get(1) {
            Some(rhs) => self.bounds_of(kernel, rhs)?,
            None => bounds::Bounds::exact(0),
        };
        let bounds = bounds::transfer(op, lhs, rhs)?;
        bounds.is_exact().then_some(bounds.start)
    }

    /// Record what a `for` over `num_tiles(&p, axis)` proves about its
    /// variable.
    ///
    /// The loop walks exactly the tiles of that axis, so every value it visits
    /// is inside the grid. Stating that as a predicate — rather than as an
    /// interval — is what lets a goal over tile counts be discharged:
    /// `TileCount` is an atom, and no arithmetic relationship to it is
    /// otherwise known.
    pub fn assume_loop_over_tile_count(&mut self, kernel: &Kernel, var: &str, end: &Expr) {
        let Expr::Builtin {
            op: BuiltinOp::NumTiles,
            args,
        } = end
        else {
            return;
        };
        let Some(Expr::Var(tensor_name)) = args.first() else {
            return;
        };
        let axis = match args.get(1) {
            Some(Expr::ScalarI64(axis)) => *axis as usize,
            _ => return,
        };
        // The name may be a partition rather than the parameter itself, so the
        // tensor it resolved to is what knows which parameter it came from.
        let Some(tensor) = self.tensors.iter().find(|t| t.name == *tensor_name) else {
            return;
        };
        let Some(param) = tensor.param else {
            return;
        };
        let Some(tile) = tensor.tile.get(axis).copied().flatten().filter(|t| *t > 0) else {
            return;
        };
        let Some(iv) = self.value_ids.id(var) else {
            return;
        };
        let index = Term::atom(Atom::Iv(iv));
        let count = Term::atom(Atom::TileCount {
            param,
            axis,
            tile: tile as i32,
        });
        // The loop's own bound: `0 <= i < num_tiles`. Both halves are believed,
        // so an access indexed by `i` on this axis is discharged rather than
        // checked.
        if let Some(upper) = Predicate::lt(&index, &count) {
            self.assumptions.push(upper);
        }
        if let Some(lower) = Predicate::le(&Term::constant(0), &index) {
            self.assumptions.push(lower);
        }
        let _ = kernel;
    }

    /// Whether the goal pair for this axis is already believed.
    ///
    /// Both halves must be in the assumption set: an access needs `0 <= i` and
    /// `i < num_tiles`, and a loop over `num_tiles` states exactly that about
    /// its variable. Anything the assumptions do not name — a value derived
    /// from the variable, an index from another axis — is not decided here.
    fn believed_access(
        &self,
        kernel: &Kernel,
        index_arg: Option<&Expr>,
        axis: usize,
        tile: i64,
        tensor_name: &str,
    ) -> bool {
        let Some(index) = index_arg.and_then(|arg| index_expr(arg, axis)) else {
            return false;
        };
        let Some(term) = self.term_of(kernel, &index) else {
            return false;
        };
        let Some(param) = buffer_param_position(kernel, tensor_name) else {
            return false;
        };
        let count = Term::atom(Atom::TileCount {
            param,
            axis,
            tile: tile as i32,
        });
        let upper = Predicate::lt(&term, &count);
        let lower = Predicate::le(&Term::constant(0), &term);
        [upper, lower]
            .iter()
            .all(|goal| goal.as_ref().is_some_and(|goal| self.assumptions.contains(goal)))
    }

    /// Decide one axis's access check from its index interval and the tile
    /// grid.
    ///
    /// cutile's `fold_static` (`checks/goals.rs`) does this before emitting
    /// anything: the interval of the index, the static extent, and the tile
    /// extent give the number of tiles, and `0 <= start && end < num_tiles`
    /// proves the access. Anything else — a violation, or an index or extent
    /// this IR cannot range — is left to the device check, except that a
    /// violation is a compile error, which is the one case cutile refuses to
    /// compile.
    fn fold_static(
        &mut self,
        kernel: &Kernel,
        index_arg: Option<&Expr>,
        axis: usize,
        extent: i64,
        tile: i64,
    ) {
        let Some(index) = index_arg.and_then(|arg| index_expr(arg, axis)) else {
            return;
        };
        let Some(bound) = self.bounds_of(kernel, &index) else {
            return;
        };
        let num_tiles = (extent + tile - 1) / tile;
        if 0 <= bound.start && bound.end < num_tiles {
            return;
        }
        // The JIT rung: a goal already believed needs no check at all. This is
        // what a loop over `num_tiles` gives its body.
        if let Some(term) = self.term_of(kernel, &index) {
            let upper = Predicate::lt(&term, &Term::constant(num_tiles));
            let lower = Predicate::le(&Term::constant(0), &term);
            let believed = |goal: &Option<Predicate>, assumptions: &[Predicate]| {
                goal.as_ref().is_some_and(|goal| assumptions.contains(goal))
            };
            if believed(&upper, &self.assumptions) && believed(&lower, &self.assumptions) {
                return;
            }
        }
        self.errors.push(format!(
            "Bounds check failed: 0 <= {} && {} < {}",
            bound.start, bound.end, num_tiles
        ));
    }

    /// Move a constant index's access goals to the host.
    ///
    /// A literal index over a dynamic extent is not decided by inspection: the
    /// extent is a launch-time fact, so the goal moves to the host. cutile
    /// makes the same call, and the check leaving the kernel is the whole point
    /// of the launch rung.
    pub fn hoist_literal_indices(
        &mut self,
        kernel: &Kernel,
        tensor_name: &str,
        tile: &[i64],
        view_axes: &[AxisExtent],
        index_arg: Option<&Expr>,
    ) {
        for (axis, value) in literal_indices(index_arg) {
            if view_axes.get(axis) != Some(&AxisExtent::Dynamic) {
                continue;
            }
            let Some(tile_extent) = tile.get(axis).copied().filter(|t| *t > 0) else {
                continue;
            };
            // The parameter position identifies the atom: a name would not
            // survive inlining, and the host resolves by position. The host
            // holds one entry per buffer parameter, in declaration order, so
            // the index counts buffer parameters rather than every parameter.
            let Some(param) = buffer_param_position(kernel, tensor_name) else {
                continue;
            };
            // `ceil(extent / tile)` is not affine, so it is an atom of its
            // own: the host computes the quotient when it evaluates the goal.
            // `TileCount` bakes the tile size into the atom's identity,
            // because the same axis at two tile sizes is two different
            // quantities.
            let count = Term::atom(Atom::TileCount {
                param,
                axis,
                tile: tile_extent as i32,
            });
            let Some(upper) = Predicate::le(&Term::constant(value), &count) else {
                continue;
            };
            let Some(lower) = Predicate::le(&Term::constant(0), &Term::constant(value)) else {
                continue;
            };
            // The lower goal is over no atoms, so it is decided now.
            if lower.eval(&|_| None) != Some(true) {
                self.errors.push(format!("Bounds check failed: 0 <= {value}"));
                continue;
            }
            self.launch_checks.push(LaunchCheck {
                predicate: upper,
                cause: format!(
                    "partition access out of bounds: dim {axis}, block index >= ceil(?/{tile_extent})"
                ),
            });
        }
    }

    /// Decide every axis of one access.
    ///
    /// A literal index is proven by inspection. A believed goal needs nothing.
    /// A static extent is decided now — a violation is a compile error. What
    /// remains is the device's, and [`AxisDecision::Device`] names it; the
    /// lowering emits that assert and nothing else.
    pub fn decide_access(
        &mut self,
        kernel: &Kernel,
        tensor: &TensorFacts,
        tile: &[i64],
        index_arg: Option<&Expr>,
    ) -> Vec<AxisDecision> {
        // An index arrives as a tuple of expressions, or as a shape of symbol
        // names; both name one value per axis.
        let items: Vec<Option<String>> = match index_arg {
            Some(Expr::TileIndex(items)) => items
                .iter()
                .map(|item| match item {
                    Expr::ScalarI64(_) => None,
                    // A bare `pid.k` is bounded by the launch grid.
                    Expr::Builtin {
                        op: BuiltinOp::ProgramId,
                        ..
                    } => None,
                    Expr::Var(name) => Some(name.clone()),
                    other => Some(format!("{other:?}")),
                })
                .collect(),
            Some(Expr::Shape(shape)) => shape_items(shape)
                .iter()
                .map(|item| match item {
                    ShapeExpr::Constant(_) | ShapeExpr::Dynamic => None,
                    // A bare `pid.k` is bounded by the launch grid, which
                    // cutile trusts; a value derived from one is not.
                    ShapeExpr::Symbol(name) if self.is_block_id_component(name) => None,
                    ShapeExpr::Symbol(name) => Some(name.clone()),
                    ShapeExpr::Tuple(_) => Some(format!("{item:?}")),
                })
                .collect(),
            _ => return Vec::new(),
        };
        self.hoist_literal_indices(kernel, &tensor.name, tile, &tensor.view_axes, index_arg);
        let mut decisions = Vec::new();
        for (axis, item) in items.iter().enumerate() {
            // A literal index is proven by inspection; every other index
            // carries no range cutile can use, so its access carries a check.
            let Some(name) = item else {
                continue;
            };
            let Some(tile_extent) = tile.get(axis).copied().filter(|t| *t > 0) else {
                continue;
            };
            // The JIT rung: a goal already believed needs no check. A loop
            // over `num_tiles` walks exactly that axis's tiles, so the access
            // it indexes is inside the grid by construction.
            if self.believed_access(kernel, index_arg, axis, tile_extent, &tensor.name) {
                continue;
            }
            match tensor.view_axes.get(axis) {
                Some(AxisExtent::Static(extent)) => {
                    self.fold_static(kernel, index_arg, axis, *extent, tile_extent);
                    continue;
                }
                Some(AxisExtent::Dynamic) => {}
                None => continue,
            }
            // A loop counter is known non-negative, so its access needs only
            // the upper comparison; a value derived from a block id has no
            // range at all, so its access tests both sides.
            decisions.push(AxisDecision::Device {
                proven_non_negative: self.bounded_loop_vars.iter().any(|bound| bound == name),
            });
        }
        decisions
    }
}

/// Whether a value is read out of a shape rather than computed.
///
/// cutile keeps a shape in metadata, so naming one of its entries does not make
/// a value: the first broadcast of it pays for materialising the shape.
pub mod shape_derived {
    use crate::{Expr, Stmt};

    pub fn is_shape_derived(expr: &Expr) -> bool {
        match expr {
            Expr::Builtin { op, args } => match op {
                crate::BuiltinOp::ShapeDim => true,
                crate::BuiltinOp::Min
                | crate::BuiltinOp::Max
                | crate::BuiltinOp::Add
                | crate::BuiltinOp::Sub
                | crate::BuiltinOp::Mul
                | crate::BuiltinOp::Div
                | crate::BuiltinOp::CeilDiv
                | crate::BuiltinOp::Rem => args.iter().any(is_shape_derived),
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether the binding named `name` holds a shape-derived value.
    pub fn binding_is_shape_derived(body: &[Stmt], name: &str) -> bool {
        fn walk(stmts: &[Stmt], name: &str) -> bool {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { name: bound, expr, .. } | Stmt::Assign { name: bound, expr, .. } => {
                        if bound == name && is_shape_derived(expr) {
                            return true;
                        }
                    }
                    Stmt::ForLoop { body, .. } | Stmt::While { body, .. } | Stmt::Loop { body } => {
                        if walk(body, name) {
                            return true;
                        }
                    }
                    Stmt::If { then_body, else_body, .. } => {
                        if walk(then_body, name) || walk(else_body, name) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        walk(body, name)
    }
}

/// The expression one axis of an index names.
///
/// A tuple index spells every axis. A shape index names one symbol per axis,
/// and a symbol that is a const generic is that constant.
pub fn index_expr(arg: &Expr, axis: usize) -> Option<Expr> {
    match arg {
        Expr::TileIndex(items) => items.get(axis).cloned(),
        // A shape index names one symbol per axis; a symbol is that variable.
        Expr::Shape(shape) => match shape_items(shape).get(axis) {
            Some(ShapeExpr::Symbol(name)) => Some(Expr::Var(name.clone())),
            _ => None,
        },
        _ => None,
    }
}

/// Where a tensor parameter sits among the buffer parameters.
///
/// The launcher holds one argument per buffer, in declaration order, and the
/// scalars travel elsewhere — so the position a launch check names is the
/// buffer's, not the parameter list's.
pub fn buffer_param_position(kernel: &Kernel, tensor_name: &str) -> Option<usize> {
    kernel
        .params
        .iter()
        .filter(|param| !matches!(param.ty, Type::Scalar(_)))
        .position(|param| param.name == tensor_name)
}

/// The axes whose index is a compile-time literal, with its value.
///
/// An index arrives either as a tuple of expressions (`[0i64]`) or as a shape
/// of symbols (`shape![4]`), and a literal can be spelled in both, so both are
/// read here rather than at each use.
pub fn literal_indices(arg: Option<&Expr>) -> Vec<(usize, i64)> {
    match arg {
        Some(Expr::TileIndex(items)) => items
            .iter()
            .enumerate()
            .filter_map(|(axis, item)| match item {
                Expr::ScalarI64(value) => Some((axis, *value)),
                _ => None,
            })
            .collect(),
        Some(Expr::Shape(shape)) => shape_items(shape)
            .iter()
            .enumerate()
            .filter_map(|(axis, item)| match item {
                ShapeExpr::Constant(value) => Some((axis, *value)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Every symbol a shape names, in order, without repeats.
pub fn shape_names(shape: &ShapeExpr) -> Vec<String> {
    fn walk(shape: &ShapeExpr, names: &mut Vec<String>) {
        match shape {
            ShapeExpr::Symbol(name) => {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
            ShapeExpr::Tuple(items) => {
                for item in items {
                    walk(item, names);
                }
            }
            _ => {}
        }
    }
    let mut names = Vec::new();
    walk(shape, &mut names);
    names
}

/// The top-level items of a shape: the shape itself, or its tuple items.
pub fn shape_items(shape: &ShapeExpr) -> Vec<ShapeExpr> {
    match shape {
        ShapeExpr::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    }
}

/// Whether the body compares the generic against a literal.
fn compared_to_literal(body: &[Stmt], name: &str) -> bool {
    fn is_comparison(op: BuiltinOp) -> bool {
        matches!(
            op,
            BuiltinOp::Eq | BuiltinOp::Ne | BuiltinOp::Lt | BuiltinOp::Le | BuiltinOp::Gt | BuiltinOp::Ge
        )
    }
    fn walk_expr(expr: &Expr, name: &str) -> bool {
        match expr {
            Expr::Builtin { op, args } => {
                (is_comparison(*op)
                    && args.iter().any(|a| matches!(a, Expr::Var(v) if v == name))
                    && args.iter().any(|a| matches!(a, Expr::ScalarI64(_))))
                    || args.iter().any(|a| walk_expr(a, name))
            }
            Expr::TileIndex(items) => items.iter().any(|a| walk_expr(a, name)),
            _ => false,
        }
    }
    fn walk_stmt(stmt: &Stmt, name: &str) -> bool {
        match stmt {
            Stmt::Let { expr, .. } | Stmt::Assign { expr, .. } => walk_expr(expr, name),
            Stmt::Store { value, index, .. } => {
                walk_expr(value, name) || index.as_ref().is_some_and(|i| walk_expr(i, name))
            }
            Stmt::AtomicAdd { index, value, .. } => {
                walk_expr(index, name) || walk_expr(value, name)
            }
            Stmt::Print { args, .. } => args.iter().any(|arg| walk_expr(arg, name)),
            Stmt::ForLoop { start, end, step, body, .. } => {
                walk_expr(start, name)
                    || walk_expr(end, name)
                    || step.as_ref().is_some_and(|s| walk_expr(s, name))
                    || body.iter().any(|s| walk_stmt(s, name))
            }
            Stmt::While { cond, body } => {
                walk_expr(cond, name) || body.iter().any(|s| walk_stmt(s, name))
            }
            Stmt::Loop { body } => body.iter().any(|s| walk_stmt(s, name)),
            Stmt::If { cond, then_body, else_body } => {
                walk_expr(cond, name)
                    || then_body.iter().any(|s| walk_stmt(s, name))
                    || else_body.iter().any(|s| walk_stmt(s, name))
            }
            Stmt::Break | Stmt::Return | Stmt::Continue => false,
        }
    }
    body.iter().any(|stmt| walk_stmt(stmt, name))
}

/// The const generics cutile prints as `1` instead of their value.
///
/// A flag generic is one the body compares against a literal — cutile's own are
/// `CAUSAL == 1i32` and `EVEN_K == 0i32` — and cutile's front end prints it as
/// `1` on the branch where the comparison holds. Nothing else distinguishes a
/// flag from an axis here: both arrive as an `i64` const generic.
fn flag_generics(kernel: &Kernel) -> Vec<String> {
    kernel
        .const_params
        .iter()
        .map(|(name, _)| name.clone())
        .filter(|name| compared_to_literal(&kernel.body, name))
        .collect()
}

/// Whether the body partitions a buffer by generics that a parameter's own
/// declared extents also name.
///
/// Such a partition instantiates the kernel's const generics, and cutile prints
/// every one of them before the body runs. A generic only the partition names
/// is not part of that instantiation.
fn instantiates_generics(kernel: &Kernel) -> bool {
    fn walk(stmt: &Stmt, declared: &[String]) -> bool {
        match stmt {
            Stmt::Let { expr, .. } => matches_partition(expr, declared),
            Stmt::ForLoop { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Loop { body } => body.iter().any(|stmt| walk(stmt, declared)),
            Stmt::If { then_body, else_body, .. } => {
                then_body.iter().any(|stmt| walk(stmt, declared))
                    || else_body.iter().any(|stmt| walk(stmt, declared))
            }
            _ => false,
        }
    }
    fn matches_partition(expr: &Expr, declared: &[String]) -> bool {
        let Expr::Builtin { op, args } = expr else {
            return false;
        };
        if !matches!(
            op,
            BuiltinOp::Partition | BuiltinOp::PartitionMut
        ) {
            return false;
        }
        let Some(Expr::Shape(shape)) = args.get(1) else {
            return false;
        };
        let mut names = Vec::new();
        fn collect(shape: &ShapeExpr, names: &mut Vec<String>) {
            match shape {
                ShapeExpr::Symbol(name) => names.push(name.clone()),
                ShapeExpr::Tuple(items) => {
                    for item in items {
                        collect(item, names);
                    }
                }
                _ => {}
            }
        }
        collect(shape, &mut names);
        names.iter().any(|name| declared.contains(name))
    }
    let declared: Vec<String> = kernel
        .params
        .iter()
        .filter_map(|param| match &param.ty {
            Type::Tensor { shape, .. } => Some(shape.clone()),
            _ => None,
        })
        .flat_map(|shape| shape_names(&shape))
        .collect();
    kernel.body.iter().any(|stmt| walk(stmt, &declared))
}
