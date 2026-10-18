use std::collections::BTreeMap;
use std::fmt;

use crate::types::{ElemType, ShapeExpr, Type};
use crate::{BuiltinOp, Expr, Kernel, ParamKind, Stmt};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    message: String,
}

impl TypeError {
    pub fn unsupported_expr(kind: &str) -> Self {
        Self {
            message: format!("unsupported expression kind: {kind}"),
        }
    }

    pub fn unsupported_builtin(name: impl Into<String>) -> Self {
        Self {
            message: format!("unsupported builtin: {}", name.into()),
        }
    }

    pub fn invalid_kernel(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn invalid_pipeline(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn undefined_name(name: &str) -> Self {
        Self {
            message: format!("undefined name `{name}`"),
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TypeError {}

#[derive(Clone, Debug)]
pub struct TypedKernel {
    pub kernel: Kernel,
    pub locals: BTreeMap<String, Type>,
}

/// Check the kernel body and record the type of every binding.
///
/// The checker is deliberately small, but it is not permissive: a name must be
/// bound before it is used, a store must target an output parameter, and a
/// loop body is checked with its induction variable in scope. Every one of
/// those was previously accepted and then silently lowered as zero, which
/// turned a type error into a wrong answer at run time.
pub fn check_kernel(kernel: &Kernel) -> Result<TypedKernel, TypeError> {
    let mut locals = BTreeMap::new();
    for param in &kernel.params {
        let ty = param.ty.clone();
        locals.insert(param.name.clone(), ty);
    }
    for (name, _) in &kernel.const_params {
        locals.insert(name.clone(), Type::Scalar(ElemType::F32));
    }

    check_block(&kernel.body, &mut locals, kernel, false)?;

    Ok(TypedKernel {
        kernel: kernel.clone(),
        locals,
    })
}

fn check_block(
    body: &[Stmt],
    locals: &mut BTreeMap<String, Type>,
    kernel: &Kernel,
    in_loop: bool,
) -> Result<(), TypeError> {
    for stmt in body {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let ty = infer_expr(expr, locals)?;
                locals.insert(name.clone(), ty);
            }
            // An assignment updates a binding that already exists. Accepting a
            // new name here would make `ac = ...` a silent declaration, and the
            // misspelt `acc` would then be a different variable from the one
            // the kernel reads later.
            Stmt::Assign { name, expr } => {
                let Some(bound) = locals.get(name).cloned() else {
                    return Err(TypeError::undefined_name(name));
                };
                let assigned = infer_expr(expr, locals)?;
                // An assignment updates the binding, so the value must have the
                // binding's category. `acc = 1.0` on a tile used to be accepted,
                // and lowering then stored the scalar as if it were a tile: the
                // store read the scalar constant through a tile pointer.
                if bound.category() != assigned.category() {
                    return Err(TypeError::invalid_kernel(format!(
                        "`{name}` is bound as {:?}, but the assignment has type {:?}",
                        bound.category(),
                        assigned.category()
                    )));
                }
            }
            Stmt::Store {
                target,
                value,
                index,
            } => {
                let target_ty = locals
                    .get(target)
                    .ok_or_else(|| TypeError::undefined_name(target))?
                    .clone();
                // A partition names the buffer it views, so a store through it
                // writes that buffer. Following the name is what keeps
                // `out_part.store(t, [i, j])` a checked write: the view itself
                // is not a parameter and would otherwise look like an input.
                let stored_name = match &target_ty {
                    Type::Tensor { .. } => target.clone(),
                    Type::Partition {
                        source: Some(source), ..
                    } => source.clone(),
                    _ => {
                        return Err(TypeError::invalid_kernel(format!(
                            "store target `{target}` is not a tensor"
                        )));
                    }
                };
                let is_output = kernel
                    .params
                    .iter()
                    .any(|param| param.name == stored_name && param.kind == ParamKind::Output);
                if !is_output {
                    return Err(TypeError::invalid_kernel(format!(
                        "store target `{target}` is not a mutable output parameter"
                    )));
                }
                if let Some(index) = index {
                    // Coordinates address the buffer element by element, so
                    // each must be a scalar or a shape the lowering reads axis
                    // by axis. A tile here has no single address.
                    let index_ty = infer_expr(index, locals)?;
                    if !matches!(index_ty, Type::Scalar(_) | Type::Shape(_)) {
                        return Err(TypeError::invalid_kernel(
                            "a store index must be a scalar or a shape".to_string(),
                        ));
                    }
                }
                let value_ty = infer_expr(value, locals)?;
                if !matches!(value_ty, Type::Tile { .. }) {
                    return Err(TypeError::invalid_kernel(format!(
                        "store into `{target}` expects a tile, got {:?}",
                        value_ty.category()
                    )));
                }
            }
            Stmt::AtomicAdd {
                target,
                index,
                value,
            } => {
                let target_ty = locals
                    .get(target)
                    .ok_or_else(|| TypeError::undefined_name(target))?;
                if !matches!(target_ty, Type::Tensor { .. }) {
                    return Err(TypeError::invalid_kernel(format!(
                        "atomic_add target `{target}` is not a tensor"
                    )));
                }
                // An atomic add writes memory exactly as a store does, so its
                // target must be a mutable output too. Without this check the
                // kernel could accumulate into an input buffer.
                let is_output = kernel
                    .params
                    .iter()
                    .any(|param| param.name == *target && param.kind == ParamKind::Output);
                if !is_output {
                    return Err(TypeError::invalid_kernel(format!(
                        "atomic_add target `{target}` is not a mutable output parameter"
                    )));
                }
                // The index addresses one element, so it must be a scalar (or a
                // shape, which the lowering reads axis by axis). A tile here
                // lowered its pointer as the index and wrote to an address the
                // program never named.
                let index_ty = infer_expr(index, locals)?;
                if !matches!(index_ty, Type::Scalar(_) | Type::Shape(_)) {
                    return Err(TypeError::invalid_kernel(
                        "an atomic_add index must be a scalar".to_string(),
                    ));
                }
                let value_ty = infer_expr(value, locals)?;
                if !matches!(value_ty, Type::Scalar(_) | Type::Tile { .. }) {
                    return Err(TypeError::invalid_kernel(format!(
                        "atomic_add expects a scalar or a one-element tile, got {:?}",
                        value_ty.category()
                    )));
                }
            }
            Stmt::ForLoop {
                var,
                start,
                end,
                body,
            } => {
                infer_expr(start, locals)?;
                infer_expr(end, locals)?;
                // The induction variable is a scalar inside the body only, so
                // it is scoped here rather than left in the kernel's locals.
                let outer = locals.clone();
                locals.insert(var.clone(), Type::Scalar(ElemType::F32));
                let result = check_block(body, locals, kernel, true);
                // A `let` in the body is a fresh binding on each iteration. It
                // is not a loop-carried value, and it does not exist after the
                // loop. Leaving it in scope made the body's `let t = ...`
                // readable after the loop, where the lowered value is the last
                // iteration's rather than an error.
                *locals = outer;
                result?;
            }
            // `continue` has no type, so it has nothing to infer. It does have
            // to be inside a loop: outside one there is no iteration to end, and
            // the lowering would have no header to branch to.
            Stmt::Continue => {
                if !in_loop {
                    return Err(TypeError::invalid_kernel(
                        "`continue` is only meaningful inside a loop".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn infer_expr(expr: &Expr, locals: &BTreeMap<String, Type>) -> Result<Type, TypeError> {
    match expr {
        Expr::Builtin { op, args } => infer_builtin(*op, args, locals),
        Expr::Var(name) => locals
            .get(name)
            .cloned()
            .ok_or_else(|| TypeError::undefined_name(name)),
        Expr::Shape(shape) => Ok(Type::Shape(shape.clone())),
        Expr::ScalarI64(_) => Ok(Type::Scalar(ElemType::F32)),
        Expr::ScalarF32(_) => Ok(Type::Scalar(ElemType::F32)),
        // A tile index is a list of runtime scalars. Its individual elements are
        // checked, so an unbound name in an index is caught here.
        Expr::TileIndex(elems) => {
            for elem in elems {
                let ty = infer_expr(elem, locals)?;
                if !matches!(ty, Type::Scalar(_) | Type::Shape(_)) {
                    return Err(TypeError::invalid_kernel(
                        "a tile index element must be a scalar".to_string(),
                    ));
                }
            }
            Ok(Type::Shape(ShapeExpr::symbol("S")))
        }
    }
}

fn infer_builtin(
    op: BuiltinOp,
    args: &[Expr],
    locals: &BTreeMap<String, Type>,
) -> Result<Type, TypeError> {
    // Every argument is checked, even the ones whose type the result does not
    // depend on. Skipping them is how an unbound name survived to lowering.
    check_arity(op, args.len())?;
    let arg_types = args
        .iter()
        .map(|arg| infer_expr(arg, locals))
        .collect::<Result<Vec<_>, _>>()?;

    let tile_shape = |index: usize| -> ShapeExpr {
        match arg_types.get(index) {
            Some(Type::Tile { shape, .. }) => shape.clone(),
            _ => ShapeExpr::symbol("S"),
        }
    };

    match op {
        BuiltinOp::ProgramId | BuiltinOp::NumPrograms | BuiltinOp::NumTiles => {
            Ok(Type::Scalar(ElemType::F32))
        }
        BuiltinOp::LoadTile => Ok(Type::tile(
            ElemType::F32,
            declared_shape(args, 1, locals),
        )),
        BuiltinOp::LoadTileLike2D => Ok(Type::tile(
            ElemType::F32,
            // `load_tile_mut(z)` names only the buffer, so the tile shape is
            // the buffer's own tile shape: the output parameter declares it.
            declared_shape(args, 1, locals).if_symbolic(|| buffer_declared_shape(args, 0, locals)),
        )),
        // A partition is a view, not a tile: it holds the buffer and the tile
        // extent until a load names the coordinates. Its own type carries that
        // extent, so a later `load` can report the tile it produces.
        //
        // The view records the buffer it came from. `partition_mut` is the same
        // view: CuTile names mutability in the method, and the buffer's own
        // parameter decides whether a store through the view is legal.
        BuiltinOp::Partition | BuiltinOp::PartitionMut => {
            let shape = declared_shape(args, 1, locals);
            check_tile_shape(&shape)?;
            Ok(Type::partition_of(
                ElemType::F32,
                shape,
                partition_source(args, 0),
            ))
        }
        // `y.load()` names no coordinates, so the tile is the buffer's own
        // declared shape, read the same way `load_tile_mut` reads it.
        BuiltinOp::Load => Ok(Type::tile(
            ElemType::F32,
            partition_shape(args, 0, locals).if_symbolic(|| buffer_declared_shape(args, 0, locals)),
        )),
        BuiltinOp::Constant => {
            let shape = declared_shape(args, 1, locals);
            check_tile_shape(&shape)?;
            Ok(Type::tile(ElemType::F32, shape))
        }
        BuiltinOp::Reshape | BuiltinOp::Broadcast => {
            let shape = declared_shape(args, 1, locals);
            check_tile_shape(&shape)?;
            Ok(Type::tile(ElemType::F32, shape))
        }
        // A transpose exchanges the two axes of its operand, so its shape is
        // the operand's with the last two extents swapped.
        BuiltinOp::Transpose => Ok(Type::tile(
            ElemType::F32,
            transpose_shape(&tile_shape(0)),
        )),
        BuiltinOp::ReduceMax | BuiltinOp::ReduceMin | BuiltinOp::ReduceSum => Ok(Type::tile(
            ElemType::F32,
            reduce_shape(&tile_shape(0), reduce_axis(args)),
        )),
        // Pointwise operations keep the operand's shape. When neither operand
        // is a tile the result is a scalar, not a tile: treating every
        // arithmetic result as a tile let a scalar expression be stored into an
        // output buffer and lowered as a one-element tile.
        BuiltinOp::Max
        | BuiltinOp::Min
        | BuiltinOp::Add
        | BuiltinOp::Sub
        | BuiltinOp::Mul
        | BuiltinOp::Div
        | BuiltinOp::Rem => {
            // Two tiles must agree on their shape. Lowering takes the first
            // operand's shape and reads the second at the same coordinates, so
            // mismatched tiles silently read past one of them.
            if let (Some(Type::Tile { shape: lhs, .. }), Some(Type::Tile { shape: rhs, .. })) =
                (arg_types.first(), arg_types.get(1))
                && lhs != rhs
            {
                return Err(TypeError::invalid_kernel(format!(
                    "pointwise operands have different shapes: {lhs} and {rhs}"
                )));
            }
            match (arg_types.first(), arg_types.get(1)) {
                (Some(Type::Tile { shape, .. }), _) => Ok(Type::tile(ElemType::F32, shape.clone())),
                (_, Some(Type::Tile { shape, .. })) => Ok(Type::tile(ElemType::F32, shape.clone())),
                _ => Ok(Type::Scalar(ElemType::F32)),
            }
        }
        // A one-operand float operation keeps the shape of its operand, and a
        // scalar operand stays a scalar.
        BuiltinOp::Unary(_) => Ok(arg_types
            .first()
            .filter(|ty| matches!(ty, Type::Tile { .. }))
            .cloned()
            .unwrap_or(Type::Scalar(ElemType::F32))),
        // A comparison is elementwise, so it has the wider operand's shape. The
        // values are `0.0` and `1.0`, which keeps the result a tile of the one
        // element type the pipeline carries.
        BuiltinOp::Gt
        | BuiltinOp::Lt
        | BuiltinOp::Ge
        | BuiltinOp::Le
        | BuiltinOp::Eq
        | BuiltinOp::Ne => match (arg_types.first(), arg_types.get(1)) {
            (Some(Type::Tile { shape, .. }), _) => Ok(Type::tile(ElemType::F32, shape.clone())),
            (_, Some(Type::Tile { shape, .. })) => Ok(Type::tile(ElemType::F32, shape.clone())),
            _ => Ok(Type::Scalar(ElemType::F32)),
        },
        // `select(cond, a, b)` keeps the shape of its values. The condition is
        // read at the same coordinates, so a condition of another shape would
        // test a different element than the one selected.
        BuiltinOp::Select => match (arg_types.get(1), arg_types.get(2)) {
            (Some(Type::Tile { shape, .. }), _) => Ok(Type::tile(ElemType::F32, shape.clone())),
            (_, Some(Type::Tile { shape, .. })) => Ok(Type::tile(ElemType::F32, shape.clone())),
            _ => Ok(Type::Scalar(ElemType::F32)),
        },
        // `tile_to_scalar` reads the one element a tile holds. A tile of more
        // than one element has no single value, so it is rejected here rather
        // than lowered to whichever element the layout happens to put first.
        BuiltinOp::TileToScalar => match arg_types.first() {
            Some(Type::Tile { shape, .. }) => match shape {
                // Rank zero is `scalar_to_tile`'s one element; rank one is what
                // a full reduction leaves.
                ShapeExpr::Tuple(dims) if dims.len() <= 1 => Ok(Type::Scalar(ElemType::F32)),
                ShapeExpr::Constant(1) => Ok(Type::Scalar(ElemType::F32)),
                // A symbolic shape is not known yet, and the tile a reduction
                // produces is one element wide.
                ShapeExpr::Symbol(_) => Ok(Type::Scalar(ElemType::F32)),
                _ => Err(TypeError::invalid_kernel(format!(
                    "tile_to_scalar expects a one-element tile, got {shape}"
                ))),
            },
            _ => Err(TypeError::invalid_kernel(
                "tile_to_scalar expects a tile".to_string(),
            )),
        },
        // `scalar_to_tile(s)` is the inverse: one element, in a one-element
        // tile. CuTile writes the shape as `shape![]`, which is rank zero.
        BuiltinOp::ScalarToTile => Ok(Type::tile(
            ElemType::F32,
            ShapeExpr::tuple(std::iter::empty()),
        )),
        // `convert_scalar(x)` names a scalar constant. Every element here is
        // f32, so the conversion is the identity.
        BuiltinOp::ConvertScalar => Ok(Type::Scalar(ElemType::F32)),
        // `mma(a, b, acc)` has the accumulator's shape.
        BuiltinOp::Mma => Ok(Type::tile(ElemType::F32, tile_shape(2))),
        // Reading one element of a tile yields a scalar; reading an element of
        // anything else has no meaning in this DSL and would lower to an
        // extract from a non-tile.
        BuiltinOp::Index => {
            match arg_types.first() {
                Some(Type::Tile { .. }) => Ok(Type::Scalar(ElemType::F32)),
                _ => Err(TypeError::invalid_kernel(
                    "an element read expects a tile".to_string(),
                )),
            }
        }
        // `shape()` is a shape, not a number. Treating it as a scalar let
        // `let s = a.shape();` bind a value that lowering turns into zero.
        // `x.shape()` is a shape expression whose dimensions are the buffer's.
        BuiltinOp::ShapeOf => Ok(Type::Shape(buffer_declared_shape(args, 0, locals))),
        // `ceil_div(a, b)` rounds a division of counts up, so its result is a
        // count, not a tile.
        BuiltinOp::ScalarDiv | BuiltinOp::ShapeDim | BuiltinOp::CeilDiv => {
            Ok(Type::Scalar(ElemType::F32))
        }
        BuiltinOp::Store => Err(TypeError::unsupported_builtin(
            "Store is a statement, not an expression",
        )),
    }
}

/// Reject a call with the wrong number of arguments.
///
/// Lowering reads fixed positions and falls back to a default when one is
/// missing: `mma(a)` would read `args[1]` and panic, and `reduce_sum(t)` would
/// default its axis. Both are kernel errors that belong at type check time,
/// with the message naming the builtin rather than a panic inside lowering.
fn check_arity(op: BuiltinOp, actual: usize) -> Result<(), TypeError> {
    let (min, max) = match op {
        BuiltinOp::ProgramId => (0, 1),
        BuiltinOp::NumPrograms => (0, 1),
        BuiltinOp::NumTiles => (2, 2),
        BuiltinOp::LoadTile => (3, 3),
        // `load_like(x, y)` takes its shape from `y`; `load_tile_mut(z)` takes
        // it from `z`'s own declaration, so one argument is enough.
        BuiltinOp::LoadTileLike2D => (1, 3),
        BuiltinOp::Partition | BuiltinOp::PartitionMut => (2, 2),
        // `part.load([i, j])` names coordinates; `y.load()` names only the
        // buffer and loads the tile the program owns, so the index is optional.
        BuiltinOp::Load => (1, 2),
        BuiltinOp::Store => (1, 1),
        BuiltinOp::Max
        | BuiltinOp::Min
        | BuiltinOp::Add
        | BuiltinOp::Sub
        | BuiltinOp::Mul
        | BuiltinOp::Div
        | BuiltinOp::Rem
        | BuiltinOp::CeilDiv => (2, 2),
        BuiltinOp::Unary(_) => (1, 1),
        BuiltinOp::Gt | BuiltinOp::Lt | BuiltinOp::Ge | BuiltinOp::Le | BuiltinOp::Eq
        | BuiltinOp::Ne => (2, 2),
        BuiltinOp::Select => (3, 3),
        BuiltinOp::TileToScalar => (1, 1),
        BuiltinOp::ScalarToTile => (1, 1),
        BuiltinOp::ConvertScalar => (1, 1),
        BuiltinOp::ReduceMax | BuiltinOp::ReduceMin | BuiltinOp::ReduceSum => (1, 2),
        BuiltinOp::Reshape | BuiltinOp::Broadcast => (2, 2),
        BuiltinOp::Transpose => (1, 1),
        BuiltinOp::ShapeOf | BuiltinOp::ShapeDim => (1, 2),
        BuiltinOp::Mma => (3, 3),
        BuiltinOp::Constant => (2, 2),
        BuiltinOp::ScalarDiv => (2, 2),
        BuiltinOp::Index => (2, 2),
    };
    if (min..=max).contains(&actual) {
        return Ok(());
    }
    let expected = if min == max {
        format!("{min}")
    } else {
        format!("{min} to {max}")
    };
    Err(TypeError::invalid_kernel(format!(
        "{op:?} expects {expected} argument(s), got {actual}"
    )))
}

/// The shape an operation asks for in its argument at `index`.
///
/// A load, a constant, a reshape and a broadcast all name their result shape.
/// The argument is either an array literal, which the HIR carries as a `Shape`,
/// or a `shape()` call on a buffer, which names that buffer's own shape. When
/// it is neither, the shape is unknown but the result is still a tile.
fn declared_shape(args: &[Expr], index: usize, locals: &BTreeMap<String, Type>) -> ShapeExpr {
    match args.get(index) {
        Some(Expr::Shape(shape)) => shape.clone(),
        // `x.load_like(y)` names the tile by naming another buffer, so the
        // shape is that buffer's own.
        Some(Expr::Builtin {
            op: BuiltinOp::ShapeOf,
            args: inner,
        }) => buffer_declared_shape(inner, 0, locals),
        Some(Expr::Var(_)) => buffer_declared_shape(args, index, locals),
        _ => ShapeExpr::symbol("S"),
    }
}

/// The tile shape a buffer parameter declares.
///
/// `load_tile_mut(z)` names the output buffer and nothing else, so the tile it
/// loads is the one the parameter was declared with.
fn buffer_declared_shape(
    args: &[Expr],
    index: usize,
    locals: &BTreeMap<String, Type>,
) -> ShapeExpr {
    match args.get(index) {
        Some(Expr::Var(name)) => match locals.get(name) {
            Some(Type::Tensor { shape, .. }) | Some(Type::Tile { shape, .. }) => shape.clone(),
            // A local bound to a shape literal keeps its dimensions in the type.
            Some(Type::Shape(shape)) => shape.clone(),
            _ => ShapeExpr::symbol("S"),
        },
        _ => ShapeExpr::symbol("S"),
    }
}

/// The tile extent a partition was declared with.
///
/// `part.load([i, j])` has no shape argument of its own: the partition carries
/// it. Reading the partition's type is what lets the loaded tile be typed
/// without repeating the shape at the call site.
fn partition_shape(args: &[Expr], index: usize, locals: &BTreeMap<String, Type>) -> ShapeExpr {
    match args.get(index) {
        Some(Expr::Var(name)) => match locals.get(name) {
            Some(Type::Partition { shape, .. }) => shape.clone(),
            _ => ShapeExpr::symbol("S"),
        },
        Some(Expr::Builtin {
            op: BuiltinOp::Partition | BuiltinOp::PartitionMut,
            args,
        }) => declared_shape(args, 1, locals),
        _ => ShapeExpr::symbol("S"),
    }
}

/// The buffer a partition view names.
///
/// `out.partition_mut(shape)` writes `out`, so the view has to remember that
/// name: the store through the view is a write to `out`, and `out` is the
/// parameter the checker asks about.
fn partition_source(args: &[Expr], index: usize) -> String {
    match args.get(index) {
        Some(Expr::Var(name)) => name.clone(),
        _ => String::new(),
    }
}

/// The axis a reduction names, defaulting to the last one.
///
/// `reduce_sum(t)` with no axis reduces the innermost dimension, which is the
/// same default the lowering uses.
fn reduce_axis(args: &[Expr]) -> i64 {
    match args.get(1) {
        Some(Expr::ScalarI64(axis)) => *axis,
        _ => 1,
    }
}

/// Reject a tile shape that covers more than one plane.
///
/// A tile lowers as one pair of extents, so a shape of rank three or more
/// describes planes above the tile, and each of those planes must hold one
/// element. Any other leading extent lowers as if the tile were one plane,
/// which is a wrong answer rather than a rejection.
fn check_tile_shape(shape: &ShapeExpr) -> Result<(), TypeError> {
    let ShapeExpr::Tuple(dims) = shape else {
        return Ok(());
    };
    if dims.len() <= 2 {
        return Ok(());
    }
    // A symbolic extent is not known yet, and a constant parameter may stand
    // for one plane. Every other leading extent is known to name more than one
    // plane, and lowering would read the tile as if it were the first of them.
    let one_plane = dims[..dims.len() - 2].iter().all(|dim| match dim {
        ShapeExpr::Constant(1) | ShapeExpr::Symbol(_) => true,
        ShapeExpr::Constant(_) | ShapeExpr::Dynamic | ShapeExpr::Tuple(_) => false,
    });
    if one_plane {
        return Ok(());
    }
    Err(TypeError::invalid_kernel(format!(
        "a tile covers one plane, so the leading extents of {shape} must be 1"
    )))
}

/// The shape a reduction along `axis` leaves.
///
/// A rank-1 tile lowers as one row, so its reduction is a single scalar-shaped
/// tile. A rank-2 tile keeps the other axis: reducing axis 1 leaves one column
/// per row, and reducing axis 0 leaves one row per column. The extent of the
/// kept axis is known whenever the reduced shape was, which is what lets a
/// following broadcast compare against it instead of against a symbolic `S`.
/// The shape of a transposed tile: the two axes exchanged.
///
/// A tile is two-dimensional, so the swap is between its last two extents; any
/// extent above them names a plane the tile sits in, and a transpose does not
/// move it.
fn transpose_shape(shape: &ShapeExpr) -> ShapeExpr {
    let ShapeExpr::Tuple(dims) = shape else {
        return ShapeExpr::symbol("S");
    };
    let mut dims = dims.clone();
    let count = dims.len();
    if count >= 2 {
        dims.swap(count - 1, count - 2);
    }
    ShapeExpr::Tuple(dims)
}

fn reduce_shape(shape: &ShapeExpr, axis: i64) -> ShapeExpr {
    let ShapeExpr::Tuple(dims) = shape else {
        return ShapeExpr::symbol("S");
    };
    match (dims.as_slice(), axis) {
        // A rank-1 tile is one row, and its only axis is the column axis.
        ([cols], _) => ShapeExpr::tuple([cols.clone()]),
        ([rows, _], 1) => ShapeExpr::tuple([rows.clone()]),
        ([_, cols], 0) => ShapeExpr::tuple([cols.clone()]),
        _ => ShapeExpr::symbol("S"),
    }
}
