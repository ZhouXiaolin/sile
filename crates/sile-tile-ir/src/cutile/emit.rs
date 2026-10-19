//! Building the cutile-shaped program from a typed kernel.
//!
//! This file grows one construct at a time. Each step is checked against the
//! golden text in `crates/sile/tests/fixtures/tileir/`, so a step is either
//! exact or visibly absent.
//!
//! Nothing here may be reordered or folded away: a value's number is its
//! position in the walk, so "cleaning up" the output renumbers everything after
//! it. `crates/sile-tile-ir/TILEIR_ALIGNMENT.md` records the measured rules.

use sile_hir::{ElemType, Kernel, ParamKind, ShapeExpr, Type};

use super::{Format, Function, Module, Op, ScalarType, Stmt, TileElement, ValueId};

/// One tensor view extent: a literal, or the value an `assume` produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    Literal(i64),
    Value(ValueId),
}

/// The strides a buffer of `shape` has in row-major order.
fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    let mut acc = 1i64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = acc;
        acc *= shape[axis];
    }
    strides
}

/// The cutile element type of a HIR element type.
///
/// HIR spells an integer `i64` because that is what its front end carries;
/// cutile's kernels and its tile block ids are `i32`, so that is what the text
/// says. The width is a fact about this IR, not about the value.
fn scalar_of(elem: ElemType) -> ScalarType {
    match elem {
        ElemType::F32 => ScalarType::F32,
        ElemType::F16 => ScalarType::F16,
        ElemType::I64 => ScalarType::I32,
        ElemType::Bool => ScalarType::I1,
    }
}

/// The declared extents of a buffer, `None` where the shape is dynamic.
fn declared_extents(shape: &ShapeExpr, consts: &[(String, i64)]) -> Vec<Option<i64>> {
    let items = match shape {
        ShapeExpr::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    };
    items
        .into_iter()
        .map(|item| match item {
            ShapeExpr::Dynamic => None,
            // A negative extent is CuTile's spelling of a dynamic one, so the
            // constant form of it is dynamic too.
            ShapeExpr::Constant(value) => Some(value).filter(|value| *value >= 0),
            ShapeExpr::Symbol(name) => consts
                .iter()
                .find(|(const_name, _)| *const_name == name)
                .map(|(_, value)| *value),
            ShapeExpr::Tuple(_) => None,
        })
        .collect()
}

struct Builder {
    next_value: u32,
    body: Vec<Stmt>,
    args: Vec<(ValueId, super::Type)>,
    /// Every tensor parameter, by name, with what the body needs to reach it.
    tensors: Vec<(String, Tensor)>,
    /// The scalar parameters, by name: their entry value and element type.
    scalars: Vec<(String, ValueId, ScalarType)>,
    /// The tile a binding names, for every name the body has bound so far.
    values: Vec<(String, ValueId, Option<super::Type>)>,
    /// The type of every binding, as `sile-hir`'s checker recorded it.
    locals: std::collections::BTreeMap<String, Type>,
    /// The interval of every integer binding known at compile time.
    ///
    /// cutile folds an integer expression whose interval is a single value into
    /// a `constant`, and discards the operation it had already allocated for it.
    /// Reproducing the discarded ids needs the same intervals.
    bounds: Vec<(String, super::bounds::Bounds)>,
    /// The tensors whose `shape()` has already become SSA values.
    ///
    /// cutile's `get_tensor_shape_meta` hands back the shape stored in the
    /// tensor's type metadata and emits nothing (`compile_intrinsic.rs:2197`),
    /// so only the first asker pays. The saving is exactly one round of that
    /// tensor's extents, measured against cutile's own output.
    shape_materialised: Vec<String>,
    /// Shapes bound to a name, which emit no values of their own.
    shape_bindings: Vec<(String, ShapeExpr)>,
    /// The three values `get_tile_block_id` produced, once the kernel asked.
    program_id: Option<Vec<ValueId>>,
    /// The `constant` a const generic already printed, by name.
    const_constants: Vec<(String, ValueId)>,
    /// The id each const generic got in the prologue. A body operand that is a
    /// generic reads this value, which the prologue printed once.
    prologue_generics: Vec<(String, ValueId)>,
    /// The offset a mutable partition added to every store through it.
    store_offset: Option<ValueId>,
    /// The block id the mutable partition fetched, which its stores index by.
    store_block: Option<ValueId>,
    /// Whether any partition has been built yet.
    /// The const generics a partition already fetched inside the body. A later
    /// integer fold reads them from here instead of printing them again, which
    /// is why `K / BK` in `gemm` costs one id and no operands.
    fetched_generics: Vec<String>,
    /// The names bound by enclosing `for` loops. An index that is one of these
    /// carries no proof against a dynamic extent, so the access needs a runtime
    /// bound check.
    loop_vars: Vec<String>,
    /// What each enclosing `for` carries, innermost last. An `if` inside a loop
    /// yields those values too: the region has to hand them on.
    loop_carried: Vec<Vec<String>>,
    /// The loop variables whose iteration range is a compile-time constant. An
    /// access indexed by one is proven non-negative; the others still test the
    /// lower bound.
    bounded_loop_vars: Vec<String>,
    /// Scratch space for a shape expression that appears in value position.
    shapes: Vec<(String, ShapeExpr)>,
}

/// What the body knows about one tensor parameter.
#[derive(Clone)]
struct Tensor {
    /// The view a `partition` already built, when this tensor is one.
    partition_view: Option<ValueId>,
    /// The view the prologue built for it.
    view: ValueId,
    /// The token `make_tensor_view` was given, which every access reuses.
    token: ValueId,
    /// The extents the kernel declared — the tile a `load` reads, and the
    /// shape a `broadcast` to `y.shape()` names.
    tile: Vec<Option<i64>>,
    element: ScalarType,
    strides: Vec<i64>,
    /// Whether the view spans the whole buffer (a mutable parameter) or only
    /// its declared tile.
    whole_buffer: bool,
    /// The extents the tensor view was built with, as it prints them.
    view_shape: Vec<Extent>,
}

/// Whether a name denotes one component of the block id.
///
/// The launch grid bounds every component, so an access indexed by one needs no
/// check. A value computed from a component has lost that bound.
fn is_block_id_component(builder: &Builder, name: &str) -> bool {
    let Some((_, value, _)) = builder.values.iter().find(|(bound, _, _)| bound == name) else {
        return false;
    };
    builder
        .program_id
        .as_ref()
        .is_some_and(|ids| ids.contains(value))
}

/// Whether a value is read out of a shape rather than computed.
///
/// cutile keeps a shape in metadata, so naming one of its entries does not make
/// a value: the first broadcast of it pays for materialising the shape.
mod shape_derived {
    use sile_hir::{BuiltinOp, Expr, Stmt};

    pub fn is_shape_derived(expr: &Expr) -> bool {
        match expr {
            Expr::Builtin { op, args } => match op {
                BuiltinOp::ShapeDim => true,
                BuiltinOp::Min
                | BuiltinOp::Max
                | BuiltinOp::Add
                | BuiltinOp::Sub
                | BuiltinOp::Mul
                | BuiltinOp::Div
                | BuiltinOp::CeilDiv
                | BuiltinOp::Rem => args.iter().any(is_shape_derived),
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

/// What a condition folds to.
enum Folded {
    /// Decided while compiling: the comparison was folded away.
    Known(bool),
    /// A runtime value the region branches on.
    Value(ValueId),
}

/// Fold a condition the way cutile does.
///
/// A comparison against a literal whose left side is a const generic is decided
/// while compiling: its left operand is named, the compared id is allocated and
/// dropped, and the predicate prints as an `i1` constant. A boolean op with a
/// decided side folds to the other side, and spends no id.
fn fold_condition(
    builder: &mut Builder,
    expr: &sile_hir::Expr,
    kernel: &Kernel,
) -> Folded {
    use sile_hir::{BuiltinOp, Expr};
    if let Expr::Builtin { op, args } = expr {
        match op {
            BuiltinOp::And | BuiltinOp::Or => {
                let left = fold_condition(builder, &args[0], kernel);
                let decisive = matches!(op, BuiltinOp::Or);
                match left {
                    Folded::Known(value) if value == decisive => {
                        // `true || x` and `false && x` are decided by the left.
                        return if matches!(op, BuiltinOp::Or) {
                            Folded::Known(true)
                        } else {
                            Folded::Known(false)
                        };
                    }
                    Folded::Known(_) => return fold_condition(builder, &args[1], kernel),
                    Folded::Value(_) => {}
                }
                let value = fold_condition(builder, &args[1], kernel);
                return value;
            }
            _ => {}
        }
    }
    if let Some(known) = literal_comparison(builder, expr, kernel) {
        let Expr::Builtin { args, .. } = expr else {
            unreachable!()
        };
        // The value the comparison is decided against is named, then the
        // comparison itself is folded away.
        let _ = emit_expr(builder, &args[1], kernel);
        builder.discard();
        builder.i32_const_i1(known);
        return Folded::Known(known);
    }
    match emit_expr(builder, expr, kernel) {
        (value, _) => Folded::Value(value),
    }
}

/// The value of a comparison between two compile-time constants.
fn literal_comparison(
    builder: &Builder,
    expr: &sile_hir::Expr,
    kernel: &Kernel,
) -> Option<bool> {
    use sile_hir::Expr;
    let Expr::Builtin { op, args } = expr else {
        return None;
    };
    let relation = match op {
        sile_hir::BuiltinOp::Eq => "equal",
        sile_hir::BuiltinOp::Ne => "not_equal",
        sile_hir::BuiltinOp::Lt => "less_than",
        sile_hir::BuiltinOp::Le => "less_than_or_equal",
        sile_hir::BuiltinOp::Gt => "greater_than",
        sile_hir::BuiltinOp::Ge => "greater_than_or_equal",
        _ => return None,
    };
    let _ = builder;
    let value_of_arg = |arg: &Expr| -> Option<i64> {
        match arg {
            Expr::ScalarI64(value) => Some(*value),
            Expr::Var(name) => {
                let (_, value) = kernel.const_params.iter().find(|(n, _)| n == name)?;
                let printed = if flag_generics(kernel).iter().any(|flag| flag == name) {
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

/// The runtime bound of a partition access, when cutile cannot discharge it.
///
/// `placement.rs` proves an access at compile time from the index's range. A
/// loop counter carries a range, so its access is proven; a block id carries
/// none, so cutile leaves it to the launch grid. An index whose range exceeds
/// the tile count of a *dynamic* extent is the one case that costs ops:
/// `ceil(extent / tile)` is only known at run time, so the check is emitted.
fn check_partition_access(
    builder: &mut Builder,
    tensor: &Tensor,
    tile: &[i64],
    indices: &[ValueId],
    index_arg: Option<&sile_hir::Expr>,
    kernel: &Kernel,
) {
    // An index arrives as a tuple of expressions, or as a shape of symbol
    // names; both name one value per axis.
    let items: Vec<Option<String>> = match index_arg {
        Some(sile_hir::Expr::TileIndex(items)) => items
            .iter()
            .map(|item| match item {
                sile_hir::Expr::ScalarI64(_) => None,
                // A bare `pid.k` is bounded by the launch grid.
                sile_hir::Expr::Builtin {
                    op: sile_hir::BuiltinOp::ProgramId,
                    ..
                } => None,
                sile_hir::Expr::Var(name) => Some(name.clone()),
                other => Some(format!("{other:?}")),
            })
            .collect(),
        Some(sile_hir::Expr::Shape(shape)) => shape_items(shape)
            .iter()
            .map(|item| match item {
                ShapeExpr::Constant(_) | ShapeExpr::Dynamic => None,
                // A bare `pid.k` is bounded by the launch grid, which cutile
                // trusts; a value derived from one is not.
                ShapeExpr::Symbol(name) if is_block_id_component(builder, name) => None,
                ShapeExpr::Symbol(name) => Some(name.clone()),
                ShapeExpr::Tuple(_) => Some(format!("{item:?}")),
            })
            .collect(),
        _ => return,
    };
    for (axis, item) in items.iter().enumerate() {
        // A literal index is proven by inspection; every other index carries no
        // range cutile can use, so its access carries a check.
        let Some(name) = item else {
            continue;
        };
        // A loop counter is known non-negative, so its access needs only the
        // upper comparison; a value derived from a block id has no range at
        // all, so its access tests both sides.
        let proven_non_negative = builder.bounded_loop_vars.iter().any(|bound| bound == name);
        // The extent as the view prints it: a literal is proven at compile
        // time, a dynamic one is a value the prologue assumed.
        let extent = match tensor.view_shape.get(axis) {
            Some(Extent::Value(value)) => *value,
            _ => continue,
        };
        let Some(tile_extent) = tile.get(axis).copied().filter(|t| *t > 0) else {
            continue;
        };
        // `ceil(extent / tile)`, with the extent a run-time value. The tile
        // extent is named again here, inside the check.
        let tile_value = builder.i32_const(tile_extent);
        let quotient = builder.value();
        builder.push_op(Op {
            results: vec![quotient],
            format: Format::Plain,
            opcode: "divi",
            operands: vec![extent, tile_value],
            result_types: vec![Builder::i32_tile()],
            attrs: Vec::new(),
            text: format!("{extent}, {tile_value} signed"),
            operand_types: None,
            indent: 0,
        });
        let remainder = builder.value();
        builder.push_op(Op {
            results: vec![remainder],
            format: Format::Plain,
            opcode: "remi",
            operands: vec![extent, tile_value],
            result_types: vec![Builder::i32_tile()],
            attrs: Vec::new(),
            text: format!("{extent}, {tile_value} signed"),
            operand_types: None,
            indent: 0,
        });
        let one = builder.i32_const(1);
        let nonempty = builder.value();
        builder.push_op(Op {
            results: vec![nonempty],
            format: Format::Plain,
            opcode: "mini",
            operands: vec![remainder, one],
            result_types: vec![Builder::i32_tile()],
            attrs: Vec::new(),
            text: format!("{remainder}, {one} signed"),
            operand_types: None,
            indent: 0,
        });
        let count = builder.value();
        builder.push_op(Op {
            results: vec![count],
            format: Format::Plain,
            opcode: "addi",
            operands: vec![quotient, nonempty],
            result_types: vec![Builder::i32_tile()],
            attrs: Vec::new(),
            text: format!("{quotient}, {nonempty}"),
            operand_types: None,
            indent: 0,
        });
        let index = indices[axis];
        let in_range = builder.value();
        builder.push_op(Op {
            results: vec![in_range],
            format: Format::Plain,
            opcode: "cmpi",
            operands: vec![index, count],
            result_types: vec![super::Type::Tile {
                shape: Vec::new(),
                element: TileElement::Scalar(ScalarType::I1),
            }],
            attrs: Vec::new(),
            text: format!("less_than {index}, {count}, signed"),
            operand_types: Some(vec![Builder::i32_tile()]),
            indent: 0,
        });
        // The lower bound: an index cutile cannot prove non-negative is tested
        // against zero as well, and the two predicates are combined.
        let guard = if proven_non_negative {
            in_range
        } else {
            let zero = builder.i32_const(0);
            let non_negative = builder.value();
            builder.push_op(Op {
                results: vec![non_negative],
                format: Format::Plain,
                opcode: "cmpi",
                operands: vec![index, zero],
                result_types: vec![super::Type::Tile {
                    shape: Vec::new(),
                    element: TileElement::Scalar(ScalarType::I1),
                }],
                attrs: Vec::new(),
                text: format!("greater_than_or_equal {index}, {zero}, signed"),
                operand_types: Some(vec![Builder::i32_tile()]),
                indent: 0,
            });
            let both = builder.value();
            builder.push_op(Op {
                results: vec![both],
                format: Format::Plain,
                opcode: "andi",
                operands: vec![non_negative, in_range],
                result_types: vec![super::Type::Tile {
                    shape: Vec::new(),
                    element: TileElement::Scalar(ScalarType::I1),
                }],
                attrs: Vec::new(),
                text: format!("{non_negative}, {in_range}"),
                operand_types: None,
                indent: 0,
            });
            both
        };
        builder.push_op(Op {
            results: Vec::new(),
            format: Format::Assert,
            opcode: "assert",
            operands: vec![in_range],
            result_types: vec![super::Type::Tile {
                shape: Vec::new(),
                element: TileElement::Scalar(ScalarType::I1),
            }],
            attrs: Vec::new(),
            text: format!(
                "{guard}, \"partition access out of bounds: dim {axis}, block index >= ceil(?/{tile_extent}){suffix}\"",
                suffix = if proven_non_negative { "" } else { " or index < 0" },
            ),
            operand_types: None,
            indent: 0,
        });
        let _ = kernel;
    }
}

/// The extents a tensor view prints with.
fn view_extents(tensor: &Tensor) -> Vec<Extent> {
    tensor.view_shape.clone()
}

impl Builder {
    fn new(_kernel: &Kernel) -> Self {
        Self {
            next_value: 0,
            body: Vec::new(),
            args: Vec::new(),
            tensors: Vec::new(),
            scalars: Vec::new(),
            values: Vec::new(),
            locals: std::collections::BTreeMap::new(),
            bounds: Vec::new(),
            shape_materialised: Vec::new(),
            shape_bindings: Vec::new(),
            program_id: None,
            const_constants: Vec::new(),
            prologue_generics: Vec::new(),
            store_offset: None,
            store_block: None,
            fetched_generics: Vec::new(),
            loop_vars: Vec::new(),
            loop_carried: Vec::new(),
            bounded_loop_vars: Vec::new(),
            shapes: Vec::new(),
        }
    }

    fn tensor(&self, name: &str) -> &Tensor {
        &self
            .tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .unwrap_or_else(|| panic!("no tensor parameter named `{name}`"))
            .1
    }

    fn type_of(&self, name: &str) -> Type {
        self.locals
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("no type recorded for `{name}`"))
    }

    fn value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    fn arg(&mut self, ty: super::Type) -> ValueId {
        let value = self.value();
        self.args.push((value, ty));
        value
    }

    /// `%t = constant <i1: v> : tile<i1>`
    fn i32_const_i1(&mut self, value: bool) -> ValueId {
        let id = self.value();
        let ty = super::Type::Tile {
            shape: Vec::new(),
            element: TileElement::Scalar(ScalarType::I1),
        };
        self.push_op(Op {
            results: vec![id],
            format: Format::Constant,
            opcode: "constant",
            operands: Vec::new(),
            result_types: vec![ty],
            attrs: Vec::new(),
            text: format!("<i1: {}>", if value { 1 } else { 0 }),
            operand_types: None,
            indent: 0,
        });
        id
    }

    fn i32_tile() -> super::Type {
        super::Type::Tile {
            shape: Vec::new(),
            element: TileElement::Scalar(ScalarType::I32),
        }
    }

    /// `%t = constant <i32: v> : tile<i32>`
    fn i32_const(&mut self, value: i64) -> ValueId {
        let result = self.value();
        self.push_op(Op {
            results: vec![result],
            format: Format::Constant,
            opcode: "constant",
            operands: Vec::new(),
            result_types: vec![Self::i32_tile()],
            attrs: Vec::new(),
            text: format!("<i32: {value}>"),
            operand_types: None,
            indent: 0,
        });
        result
    }

    /// `%t = constant <f32: v> : tile<Nxf32>`
    fn f32_const(&mut self, value: f32, extents: &[i64]) -> ValueId {
        let result = self.value();
        let _shape = extents
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("x");
        self.push_op(Op {
            results: vec![result],
            format: Format::Constant,
            opcode: "constant",
            operands: Vec::new(),
            result_types: vec![super::Type::Tile {
                shape: extents.to_vec(),
                element: TileElement::Scalar(ScalarType::F32),
            }],
            attrs: Vec::new(),
            text: format!(
                "<f32: {}>",
                if value.fract() == 0.0 {
                    format!("{value:.1}")
                } else {
                    value.to_string()
                }
            ),
            operand_types: None,
            indent: 0,
        });
        result
    }

    fn assume(&mut self, subject: ValueId) -> ValueId {
        let result = self.value();
        self.push_op(Op {
            results: vec![result],
            format: Format::Assume,
            opcode: "assume",
            operands: vec![subject],
            result_types: vec![Self::i32_tile()],
            attrs: Vec::new(),
            text: format!("bounded<0, ?>, {subject}"),
            operand_types: None,
            indent: 0,
        });
        result
    }

    fn make_token(&mut self) -> ValueId {
        let result = self.value();
        self.push_op(Op {
            results: vec![result],
            format: Format::MakeToken,
            opcode: "make_token",
            operands: Vec::new(),
            result_types: vec![super::Type::Token],
            attrs: Vec::new(),
            text: String::new(),
            operand_types: None,
            indent: 0,
        });
        result
    }

    fn make_tensor_view(&mut self, ptr: ValueId, view: super::TensorView) -> ValueId {
        let result = self.value();
        let dynamic = view.shape.iter().any(|e| matches!(e, Extent::Value(_)));
        let shape = view
            .shape
            .iter()
            .map(|e| match e {
                Extent::Literal(v) => v.to_string(),
                Extent::Value(v) => v.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let strides = view
            .strides
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mut text = format!("{ptr}, shape = [{shape}], strides = [{strides}]");
        let view_ty = super::Type::TensorView(view);
        if dynamic {
            text.push_str(&format!(" : {} -> {view_ty}", Self::i32_tile()));
        }
        self.push_op(Op {
            results: vec![result],
            format: Format::MakeTensorView,
            opcode: "make_tensor_view",
            operands: vec![ptr],
            result_types: vec![view_ty],
            attrs: Vec::new(),
            text,
            operand_types: None,
            indent: 0,
        });
        result
    }

    fn push_op(&mut self, op: Op) {
        self.body.push(Stmt::Op(op));
    }

    fn push_stmt(&mut self, stmt: Stmt) {
        self.body.push(stmt);
    }

    /// The id of an operation that was allocated and then proved constant.
    ///
    /// cutile allocates the operation, runs its interval transfer, and returns a
    /// fresh `constant` when the interval is a single value — the operation is
    /// never appended, so its id is printed nowhere. The gap is part of the
    /// contract.
    fn discard(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    /// `%t = get_tile_block_id : tile<i32>` — three results — then one
    /// `assume` per result. Every memory access asks for it again.
    fn block_id(&mut self) -> Vec<ValueId> {
        let base = self.next_value;
        self.next_value += 3;
        let ids = (0..3).map(|i| ValueId(base + i)).collect::<Vec<_>>();
        self.push_op(Op {
            results: ids.clone(),
            format: Format::GetTileBlockId,
            opcode: "get_tile_block_id",
            operands: Vec::new(),
            result_types: vec![Self::i32_tile(); 3],
            attrs: Vec::new(),
            text: String::new(),
            operand_types: None,
            indent: 0,
        });
        ids.into_iter().map(|id| self.assume(id)).collect()
    }

    fn reshape_typed(
        &mut self,
        source: ValueId,
        source_ty: &super::Type,
        to: &[i64],
    ) -> (ValueId, super::Type) {
        let element = match source_ty {
            super::Type::Tile { element, .. } => *element,
            _ => TileElement::Scalar(ScalarType::F32),
        };
        let result = self.value();
        let to_ty = super::Type::Tile {
            shape: to.to_vec(),
            element,
        };
        self.push_op(Op {
            results: vec![result],
            format: Format::Plain,
            opcode: "reshape",
            operands: vec![source],
            result_types: vec![to_ty.clone()],
            attrs: Vec::new(),
            text: format!("{source}"),
            operand_types: Some(vec![source_ty.clone()]),
            indent: 0,
        });
        (result, to_ty)
    }


    fn broadcast(
        &mut self,
        source: ValueId,
        from: &[i64],
        to: &[i64],
        element: TileElement,
    ) -> (ValueId, super::Type) {
        let result = self.value();
        let from_ty = super::Type::Tile {
            shape: from.to_vec(),
            element,
        };
        let to_ty = super::Type::Tile {
            shape: to.to_vec(),
            element,
        };
        self.push_op(Op {
            results: vec![result],
            format: Format::Plain,
            opcode: "broadcast",
            operands: vec![source],
            result_types: vec![to_ty.clone()],
            attrs: Vec::new(),
            text: format!("{source}"),
            operand_types: Some(vec![from_ty]),
            indent: 0,
        });
        (result, to_ty)
    }

    /// `%t = make_partition_view %v : partition_view<...>`
    ///
    /// A read view carries `padding_value = zero`; a write view does not.
    fn make_partition_view(&mut self, tensor: &Tensor, tile: &[i64], padding_zero: bool) -> ValueId {
        let view = super::TensorView {
            element: tensor.element,
            shape: view_extents(tensor),
            strides: tensor.strides.clone(),
        };
        let ty = super::Type::PartitionView(super::PartitionView {
            tile_shape: tile.to_vec(),
            padding_zero,
            tensor: view.clone(),
        });
        let result = self.value();
        let source = tensor.view;
        self.push_op(Op {
            results: vec![result],
            format: Format::MakePartitionView,
            opcode: "make_partition_view",
            operands: vec![source],
            result_types: vec![ty.clone()],
            attrs: Vec::new(),
            text: format!("{source}"),
            operand_types: None,
            indent: 0,
        });
        result
    }

    /// `%t, %u = load_view_tko weak %view[i, j] token = %tok : ...`
    fn load_view_tko(
        &mut self,
        view: ValueId,
        indices: &[ValueId],
        token: ValueId,
        tile: &[i64],
        element: ScalarType,
        view_ty: super::Type,
    ) -> (ValueId, super::Type) {
        let base = self.next_value;
        self.next_value += 2;
        let results = vec![ValueId(base), ValueId(base + 1)];
        let tile_ty = super::Type::Tile {
            shape: tile.to_vec(),
            element: TileElement::Scalar(element),
        };
        let index_text = indices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        self.push_op(Op {
            results: results.clone(),
            format: Format::LoadViewTko,
            opcode: "load_view_tko",
            operands: [&[view][..], indices, &[token][..]].concat(),
            result_types: vec![tile_ty.clone(), super::Type::Token],
            attrs: Vec::new(),
            text: format!("weak {view}[{index_text}] token = {token}"),
            operand_types: Some(vec![view_ty, Self::i32_tile()]),
            indent: 0,
        });
        (results[0], tile_ty)
    }

    /// `%t = store_view_tko weak %value, %view[i, j] token = %tok : ...`
    fn store_view_tko(
        &mut self,
        value: ValueId,
        view: ValueId,
        indices: &[ValueId],
        token: ValueId,
        value_ty: super::Type,
        view_ty: super::Type,
    ) {
        let result = self.value();
        let index_text = indices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        self.push_op(Op {
            results: vec![result],
            format: Format::StoreViewTko,
            opcode: "store_view_tko",
            operands: [&[value, view][..], indices, &[token][..]].concat(),
            result_types: vec![super::Type::Token],
            attrs: Vec::new(),
            text: format!("weak {value}, {view}[{index_text}] token = {token}"),
            operand_types: Some(vec![value_ty, view_ty, Self::i32_tile()]),
            indent: 0,
        });
    }
}

/// The const generics cutile prints as `1` instead of their value.
///
/// A flag generic is one the body compares against a literal — cutile's own are
/// `CAUSAL == 1i32` and `EVEN_K == 0i32` — and cutile's front end prints it as
/// `1` on the branch where the comparison holds. Nothing else distinguishes a
/// flag from an axis here: both arrive as an `i64` const generic.
/// Whether the body partitions a buffer by generics that a parameter's own
/// declared extents also name.
///
/// Such a partition instantiates the kernel's const generics, and cutile prints
/// every one of them before the body runs. A generic only the partition names
/// is not part of that instantiation.
fn instantiates_generics(kernel: &Kernel) -> bool {
    fn walk(stmt: &sile_hir::Stmt, kernel: &Kernel, declared: &[String]) -> bool {
        match stmt {
            sile_hir::Stmt::Let { expr, .. } => matches_partition(expr, declared),
            sile_hir::Stmt::ForLoop { body, .. }
            | sile_hir::Stmt::While { body, .. }
            | sile_hir::Stmt::Loop { body } => {
                body.iter().any(|stmt| walk(stmt, kernel, declared))
            }
            sile_hir::Stmt::If { then_body, else_body, .. } => {
                then_body.iter().any(|stmt| walk(stmt, kernel, declared))
                    || else_body.iter().any(|stmt| walk(stmt, kernel, declared))
            }
            _ => false,
        }
    }
    fn matches_partition(expr: &sile_hir::Expr, declared: &[String]) -> bool {
        let sile_hir::Expr::Builtin { op, args } = expr else {
            return false;
        };
        if !matches!(
            op,
            sile_hir::BuiltinOp::Partition | sile_hir::BuiltinOp::PartitionMut
        ) {
            return false;
        }
        let Some(sile_hir::Expr::Shape(shape)) = args.get(1) else {
            return false;
        };
        let mut names = Vec::new();
        fn collect(shape: &sile_hir::ShapeExpr, names: &mut Vec<String>) {
            match shape {
                sile_hir::ShapeExpr::Symbol(name) => names.push(name.clone()),
                sile_hir::ShapeExpr::Tuple(items) => {
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
            sile_hir::Type::Tensor { shape, .. } => Some(shape.clone()),
            _ => None,
        })
        .flat_map(|shape| shape_names(&shape))
        .collect();
    kernel
        .body
        .iter()
        .any(|stmt| walk(stmt, kernel, &declared))
}

fn flag_generics(kernel: &Kernel) -> Vec<String> {
    kernel
        .const_params
        .iter()
        .map(|(name, _)| name.clone())
        .filter(|name| compared_to_literal(&kernel.body, name))
        .collect()
}

fn compared_to_literal(body: &[sile_hir::Stmt], name: &str) -> bool {
    use sile_hir::BuiltinOp;
    fn is_comparison(op: BuiltinOp) -> bool {
        matches!(
            op,
            BuiltinOp::Eq | BuiltinOp::Ne | BuiltinOp::Lt | BuiltinOp::Le | BuiltinOp::Gt | BuiltinOp::Ge
        )
    }
    fn walk_expr(expr: &sile_hir::Expr, name: &str) -> bool {
        match expr {
            sile_hir::Expr::Builtin { op, args } => {
                (is_comparison(*op)
                    && args.iter().any(|a| matches!(a, sile_hir::Expr::Var(v) if v == name))
                    && args
                        .iter()
                        .any(|a| matches!(a, sile_hir::Expr::ScalarI64(_))))
                    || args.iter().any(|a| walk_expr(a, name))
            }
            sile_hir::Expr::TileIndex(items) => items.iter().any(|a| walk_expr(a, name)),
            _ => false,
        }
    }
    fn walk_stmt(stmt: &sile_hir::Stmt, name: &str) -> bool {
        use sile_hir::Stmt;
        match stmt {
            Stmt::Let { expr, .. } | Stmt::Assign { expr, .. } => walk_expr(expr, name),
            Stmt::Store { value, index, .. } => {
                walk_expr(value, name) || index.as_ref().is_some_and(|i| walk_expr(i, name))
            }
            Stmt::AtomicAdd { index, value, .. } => {
                walk_expr(index, name) || walk_expr(value, name)
            }
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


/// The element type of a HIR element type, as a tile spells it.
fn tile_of(elem: ElemType, shape: &ShapeExpr, consts: &[(String, i64)]) -> super::Type {
    let dims = declared_extents(shape, consts)
        .into_iter()
        .map(|extent| extent.unwrap_or(-1))
        .collect::<Vec<_>>();
    super::Type::Tile {
        shape: dims,
        element: TileElement::Scalar(scalar_of(elem)),
    }
}

/// The `constant` fetches of one shape argument, in source order.
///
/// A literal extent costs one `constant`; a const generic costs one carrying
/// its value; a dynamic extent costs one carrying `-1`. Every call is a fresh
/// fetch: cutile asks for the shape again for each operation that takes one.
fn shape_consts(builder: &mut Builder, shape: &ShapeExpr, consts: &[(String, i64)]) -> Vec<ValueId> {
    let items = match shape {
        ShapeExpr::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    };
    items
        .iter()
        .map(|item| match item {
            ShapeExpr::Constant(value) => builder.i32_const(*value),
            ShapeExpr::Dynamic => builder.i32_const(-1),
            ShapeExpr::Symbol(name) => {
                let value = consts
                    .iter()
                    .find(|(const_name, _)| const_name == name)
                    .map(|(_, value)| *value)
                    .unwrap_or(-1);
                let id = builder.i32_const(value);
                // `convert_scalar` reads back the most recent `constant` of
                // that generic, so the table keeps only the latest.
                if consts.iter().any(|(n, _)| n == name) {
                    builder.const_constants.retain(|(n, _)| n != name);
                    builder.const_constants.push((name.clone(), id));
        builder.prologue_generics.push((name.clone(), id));
                }
                id
            }
            ShapeExpr::Tuple(_) => builder.i32_const(-1),
        })
        .collect()
}

/// The numbers a shape names, in order, with const generics resolved.
fn shape_extents(shape: &ShapeExpr, consts: &[(String, i64)]) -> Vec<i64> {
    let items = match shape {
        ShapeExpr::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    };
    items
        .into_iter()
        .map(|item| match item {
            ShapeExpr::Dynamic => -1,
            ShapeExpr::Constant(value) => value,
            ShapeExpr::Symbol(name) => consts
                .iter()
                .find(|(const_name, _)| const_name == &name)
                .map(|(_, value)| *value)
                .unwrap_or(-1),
            ShapeExpr::Tuple(_) => -1,
        })
        .collect()
}

/// The `2r` shape arguments of a batch of expressions: each extent of each one,
/// twice over.

/// The program cutile prints for `kernel`.
pub fn emit(kernel: &Kernel, locals: &std::collections::BTreeMap<String, Type>) -> Module {
    let mut builder = Builder::new(kernel);
    builder.locals = locals.clone();
    declare_params(&mut builder, kernel);
    // A kernel whose body partitions a buffer by generics that its own
    // parameter types also name instantiates those generics up front: cutile
    // prints all of them once, before the body's first op. A shape built only
    // from generics the partition alone names (`BLOCK_SIZE` in `rms_norm`)
    // does not.
    if instantiates_generics(kernel) {
        builder.fetched_generics = kernel
            .const_params
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let flags = flag_generics(kernel);
        for (name, value) in &kernel.const_params {
            // A flag generic prints `1`: the body only ever compares it.
            let printed = if flags.iter().any(|flag| flag == name) {
                1
            } else {
                *value
            };
            let id = builder.i32_const(printed);
            // These are the ids a later operand reads: the group is the last
            // time the body names each generic.
            builder.prologue_generics.retain(|(n, _)| n != name);
            builder.prologue_generics.push((name.clone(), id));
        }
    }
    for stmt in &kernel.body {
        emit_stmt(&mut builder, stmt, kernel);
    }
    builder.body.push(Stmt::Return);

    Module {
        name: kernel.origin.module.clone(),
        functions: vec![Function {
            name: format!("{}_entry", kernel.name),
            di_name: kernel.name.clone(),
            args: builder.args,
            body: builder.body,
        }],
    }
}

/// The entry parameters, in declaration order, and then one view per tensor.
fn declare_params(builder: &mut Builder, kernel: &Kernel) {
    let mut tensors: Vec<(String, Tensor, Vec<ValueId>, Vec<Option<i64>>, ElemType)> =
        Vec::new();
    for param in &kernel.params {
        match &param.ty {
            Type::Tensor { elem, shape } | Type::Tile { elem, shape } => {
                let tile = declared_extents(shape, &kernel.const_params);
                // A mutable parameter's view spans the whole buffer, so every
                // axis of that view is dynamic; its declared extents are the
                // tile a load or a store covers.
                let view_extents = if param.kind == ParamKind::Output {
                    vec![None; tile.len()]
                } else {
                    tile.clone()
                };
                let strides = param.strides.clone().unwrap_or_else(|| {
                    let dims = view_extents.iter().map(|e| e.unwrap_or(1)).collect::<Vec<_>>();
                    row_major_strides(&dims)
                });
                let ptr = builder.arg(super::Type::Tile {
                    shape: Vec::new(),
                    element: TileElement::Pointer(scalar_of(*elem)),
                });
                let shape_args = (0..tile.len())
                    .map(|_| builder.arg(Builder::i32_tile()))
                    .collect::<Vec<_>>();
                for _ in 0..tile.len() {
                    builder.arg(Builder::i32_tile());
                }
                if param.kind == ParamKind::Output {
                    for _ in 0..tile.len() * 2 {
                        builder.arg(Builder::i32_tile());
                    }
                }
                tensors.push((
                    param.name.clone(),
                    Tensor {
                        view: ptr,
                        token: ptr,
                        tile,
                        element: scalar_of(*elem),
                        strides,
                        whole_buffer: param.kind == ParamKind::Output,
                        partition_view: None,
                        view_shape: Vec::new(),
                    },
                    shape_args,
                    view_extents,
                    *elem,
                ));
            }
            Type::Scalar(elem) => {
                let value = builder.arg(super::Type::Tile {
                    shape: Vec::new(),
                    element: TileElement::Scalar(scalar_of(*elem)),
                });
                builder.scalars.push((param.name.clone(), value, scalar_of(*elem)));
            }
            Type::Partition { .. } | Type::Shape(_) => {}
        }
    }

    // ── Constant generics ──────────────────────────────────────────
    //
    // Every const generic prints, in declaration order, whether or not the body
    // reads it: cutile emits one `constant` per generic. A flag generic is one
    // the body compares against a literal, and cutile prints `1` for it.
    let flags = flag_generics(kernel);
    for (name, value) in &kernel.const_params {
        let id = if flags.iter().any(|flag| flag == name) {
            builder.i32_const(1)
        } else {
            builder.i32_const(*value)
        };
        // A later `convert_scalar` reads this value back, not a new one.
        builder.const_constants.push((name.clone(), id));
        builder.prologue_generics.push((name.clone(), id));
    }

    // ── One view per tensor parameter ──────────────────────────────
    //
    // The extents are assumed first and the view then names the assumed value.
    for (name, tensor, shape_args, view_extents, _) in tensors {
        let shape = view_extents
            .iter()
            .zip(shape_args.iter())
            .map(|(extent, value)| match extent {
                Some(v) => Extent::Literal(*v),
                None => Extent::Value(builder.assume(*value)),
            })
            .collect::<Vec<_>>();
        let token = builder.make_token();
        let view = builder.make_tensor_view(
            tensor.view,
            super::TensorView {
                element: tensor.element,
                shape: shape.clone(),
                strides: tensor.strides.clone(),
            },
        );
        builder.tensors.push((
            name,
            Tensor {
                view,
                token,
                view_shape: shape,
                ..tensor
            },
        ));
    }
}

// ── The body ───────────────────────────────────────────────────────────

fn emit_stmt(builder: &mut Builder, stmt: &sile_hir::Stmt, kernel: &Kernel) {
    use sile_hir::Stmt;
    match stmt {
        Stmt::Let { name, expr, .. } => {
            // A shape binding is metadata; it emits nothing, so it is kept
            // aside rather than given an id.
            if let sile_hir::Expr::Shape(shape) = expr {
                builder.shape_bindings.push((name.clone(), shape.clone()));
                return;
            }
            let (value, ty) = emit_expr(builder, expr, kernel);
            builder.values.push((name.clone(), value, Some(ty)));
            if let Some(bound) = bounds_of(builder, kernel, expr) {
                builder.bounds.push((name.clone(), bound));
            }
            // A partition binds a view of an existing tensor: record it as a
            // tensor of its own so a later `load` can read through it.
            if let sile_hir::Expr::Builtin { op, args } = expr {
                if matches!(
                    op,
                    sile_hir::BuiltinOp::Partition | sile_hir::BuiltinOp::PartitionMut
                ) {
                    if let (sile_hir::Expr::Var(source), Some(shape)) = (
                        &args[0],
                        args.get(1).and_then(|shape| shape_of_expr(builder, shape)),
                    ) {
                        let mut derived = builder.tensor(source).clone();
                        let tile = shape_extents(&shape, &kernel.const_params);
                        derived.view = value;
                        derived.partition_view = Some(value);
                        derived.tile = tile.iter().map(|e| Some(*e)).collect();
                        // A partitioned read is padded like the buffer it came
                        // from: only a whole mutable buffer is unpadded.
                        derived.whole_buffer = false;
                        builder.tensors.push((name.clone(), derived));
                    }
                }
            }
        }
        Stmt::Assign { name, expr } => {
            let (value, ty) = emit_expr(builder, expr, kernel);
            // The innermost binding of the name is the one being assigned.
            match builder.values.iter_mut().rev().find(|(bound, _, _)| bound == name) {
                Some(entry) => {
                    entry.1 = value;
                    entry.2 = Some(ty);
                }
                None => builder.values.push((name.clone(), value, Some(ty))),
            }
        }
        Stmt::Store { target, value, index } => {
            let (stored, stored_ty) = emit_expr(builder, value, kernel);
            // The index the kernel named, with the partition's offset added to
            // its column when the target is a mutable partition.
            let store_index = match index {
                Some(sile_hir::Expr::Shape(shape)) => shape_items(shape)
                    .iter()
                    .map(|item| match item {
                        ShapeExpr::Symbol(name) if builder.values.iter().any(|(n, _, _)| n == name) => {
                            value_of(builder, name)
                        }
                        other => shape_consts(builder, other, &kernel.const_params)
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| builder.i32_const(0)),
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            store_tile(builder, target, stored, stored_ty, store_index, kernel);
        }
                // `for j in 0..n`: start and step are constants, the bound is a value.
        // The loop variable and its accumulator are region parameters.
        sile_hir::Stmt::ForLoop { var, start, end, step, body } => {
            // cutile evaluates the bound first, then the step: the folded
            // `K / BK` prints before the `1`.
            let (start_value, _) = emit_expr(builder, start, kernel);
            let (end_value, _) = emit_expr(builder, end, kernel);
            let step_value = match step {
                Some(step) => emit_expr(builder, step, kernel).0,
                None => builder.i32_const(1),
            };
            // The accumulator: every binding the body assigns.
            // What the loop carries: every name the body assigns, in the order
            // the enclosing scope declared them, once each.
            let assigned = body
                .iter()
                .filter_map(|stmt| match stmt {
                    sile_hir::Stmt::Assign { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            // The region lists them in reverse declaration order: the value
            // bound last comes first.
            let carried = builder
                .values
                .iter()
                .rev()
                .map(|(name, _, _)| name.clone())
                .filter(|name| assigned.contains(name))
                .fold(Vec::new(), |mut kept: Vec<String>, name| {
                    if !kept.contains(&name) {
                        kept.push(name);
                    }
                    kept
                });
            let mut inits = Vec::new();
            for name in &carried {
                let value = value_of(builder, name);
                let ty = emitted_type_of(builder, name)
                    .unwrap_or_else(|| Builder::i32_tile());
                inits.push((value, ty));
            }
            // The loop variable takes the first id, then the accumulator the
            // loop carries; the loop's own result comes after both.
            // The loop variable, the accumulator, then the assumption on the
            // variable; the loop's own result comes after all three.
            let var_id = builder.value();
            let vars = carried
                .iter()
                .map(|_| builder.value())
                .collect::<Vec<_>>();
            // The bound is the number of iterations, which is the loop's own
            // end value less its start.
            let bound = match (bounds_of(builder, kernel, start), bounds_of(builder, kernel, end)) {
                // The variable runs from zero to the last iteration, which is
                // one less than the trip count.
                (Some(start), Some(end)) => (end.end - start.start - 1).to_string(),
                // A trip count the compiler cannot know prints `?`.
                _ => "?".to_string(),
            };
            let assumed_id = builder.value();
            let header_placeholder = format!(
                "    {}for {} in ({} to {}, step {}) : tile<i32>{}{} {{",
                if carried.is_empty() {
                    String::new()
                } else {
                    "@RESULT@ = ".to_string()
                },
                var_id,
                start_value,
                end_value,
                step_value,
                if inits.is_empty() {
                    String::new()
                } else {
                    format!(
                        " iter_values({})",
                        vars.iter()
                            .zip(inits.iter())
                            .map(|(v, (init, _))| format!("{v} = {init}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
                if inits.is_empty() {
                    String::new()
                } else {
                    format!(
                        " -> ({})",
                        inits
                            .iter()
                            .map(|(_, ty)| ty.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
            );
            builder.push_stmt(super::Stmt::Raw(header_placeholder));
            let before = builder.body.len();
            // The loop variable is assumed bounded as soon as the body opens.
            // The loop variable is assumed bounded as soon as the body opens;
            // the assumption does not spend an id of its own here.
            builder.push_op(Op {
                results: vec![assumed_id],
                format: Format::Assume,
                opcode: "assume",
                operands: vec![var_id],
                result_types: vec![Builder::i32_tile()],
                attrs: Vec::new(),
                text: format!("bounded<0, {bound}>, {var_id}"),
                operand_types: None,
                indent: 0,
            });
            let _ = assumed_id;
            builder
                .values
                .push((var.clone(), assumed_id, Some(Builder::i32_tile())));
            builder.loop_vars.push(var.clone());
            builder.loop_carried.push(carried.clone());
            if bound != "?" {
                builder.bounded_loop_vars.push(var.clone());
            }
            for (name, (_, ty)) in carried.iter().zip(inits.iter()) {
                builder
                    .values
                    .push((name.clone(), vars[carried.iter().position(|c| c == name).unwrap()], Some(ty.clone())));
            }
            // The region's body prints two spaces deeper than the loop itself.
            for stmt in body {
                emit_stmt(builder, stmt, kernel);
            }
            builder.loop_vars.retain(|name| name != var);
            builder.loop_carried.pop();
            builder.bounded_loop_vars.retain(|name| name != var);
            indent_from(builder, before);
            let final_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            let _ = &final_values;
            // A loop with nothing to carry prints no result and spares the id.
            // One result per carried value, allocated after the body: a
            // region's results follow every op they dominate.
            let result_ids = if carried.is_empty() {
                Vec::new()
            } else {
                carried.iter().map(|_| builder.value()).collect::<Vec<_>>()
            };
            let trailer = if final_values.is_empty() {
                "      continue\n    }".to_string()
            } else {
                format!(
                    "      continue {} : {}\n    }}",
                    final_values
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    inits
                        .iter()
                        .map(|(_, ty)| ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            };
            builder.push_stmt(super::Stmt::Raw(trailer));
            // After the loop, the carried name denotes the loop's result.
            for (name, result_id) in carried.iter().zip(result_ids.iter()) {
                if let Some(entry) = builder.values.iter_mut().rev().find(|(b, _, _)| b == name) {
                    entry.1 = *result_id;
                }
            }
            for stmt in builder.body.iter_mut().rev() {
                if let super::Stmt::Raw(text) = stmt {
                    if text.contains("@RESULT@") {
                        if !result_ids.is_empty() {
                            *text = text.replace(
                                "@RESULT@",
                                &result_ids
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join(", "),
                            );
                        }
                        break;
                    }
                }
            }
        }
        // `if cond { .. } else { .. }`: a region with two yields. The condition is
        // a value; the results are reserved before either branch runs, so their
        // ids sit past every op the branches print.
        sile_hir::Stmt::If { cond, then_body, else_body } => {
            // `if CAUSAL == 1`: comparing a flag generic against its own literal
            // is decided while compiling, so cutile emits the predicate as a
            // constant and pastes the branch in place of a region.
            let cond_value = match fold_condition(builder, cond, kernel) {
                Folded::Known(taken) => {
                    // The branch is decided, so cutile pastes it in place of a
                    // region.
                    let body = if taken { then_body } else { else_body };
                    for stmt in body {
                        emit_stmt(builder, stmt, kernel);
                    }
                    return;
                }
                Folded::Value(value) => value,
            };
            // What the region yields: the values the enclosing loop carries, in
            // the same order, then every name this branch assigns that the loop
            // does not carry. A region nested in a loop must hand the loop's
            // values on, or the loop loses them.
            let assigned = then_body
                .iter()
                .filter_map(|stmt| match stmt {
                    sile_hir::Stmt::Assign { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let outer = builder.loop_carried.last().cloned().unwrap_or_default();
            let carried = outer
                .iter()
                .cloned()
                .chain(
                    assigned
                        .iter()
                        .filter(|name| !outer.contains(name))
                        .cloned(),
                )
                .collect::<Vec<_>>();
            let types = carried
                .iter()
                .filter_map(|name| emitted_type_of(builder, name))
                .collect::<Vec<_>>();
            let types_text = types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            // The header names results the body has not printed yet, so it is
            // written last: the region's ids follow every op it dominates.
            let header_index = builder.body.len();
            builder.push_stmt(super::Stmt::Raw(String::new()));
            let before = builder.body.len();
            // The values the names held before the branch ran: that is what an
            // `if` without an `else` yields on the untaken side.
            let pre_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            for stmt in then_body {
                emit_stmt(builder, stmt, kernel);
            }
            let then_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            // A branch that changes a value carries the old one on the other
            // side, so cutile prints an `else` even when the kernel has none.
            let changes = !assigned.iter().all(|name| !carried.contains(name));
            let has_else = !else_body.is_empty() || changes;
            let yield_index = builder.body.len();
            builder.push_stmt(super::Stmt::Raw(String::new()));
            for stmt in else_body {
                emit_stmt(builder, stmt, kernel);
            }
            let else_values = if else_body.is_empty() {
                pre_values
            } else {
                carried
                    .iter()
                    .map(|name| value_of(builder, name))
                    .collect::<Vec<_>>()
            };
            let result_ids = carried
                .iter()
                .map(|_| builder.value())
                .collect::<Vec<_>>();
            let ids_text = result_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let join = |values: &[ValueId]| {
                values
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            if let super::Stmt::Raw(text) = &mut builder.body[header_index] {
                *text = format!("    {ids_text} = if {cond_value} -> ({types_text}) {{");
            }
            if let super::Stmt::Raw(text) = &mut builder.body[yield_index] {
                // `indent_from` adds the region's own indentation to every
                // line below the header, so these start at column zero.
                *text = if !has_else {
                    format!("    yield {} : {types_text}\n  }}", join(&then_values))
                } else {
                    format!(
                        "    yield {} : {types_text}\n  }} else {{\n    yield {} : {types_text}\n  }}",
                        join(&then_values),
                        join(&else_values)
                    )
                };
            }
            // The region's body prints two spaces deeper than the `if`.
            indent_from(builder, before);
            // After the region the carried names denote its results.
            for (name, result_id) in carried.iter().zip(result_ids.iter()) {
                if let Some(entry) = builder.values.iter_mut().rev().find(|(b, _, _)| b == name) {
                    entry.1 = *result_id;
                }
            }
        }
        // A loop's `continue` is the trailer the loop already printed.
        sile_hir::Stmt::Continue | sile_hir::Stmt::Break => {}
        other => panic!("the cutile emitter has no arm for {other:?}"),
    }
}

/// The type a binding was emitted with, when the emitter knows it.
///
/// `locals` holds what the checker recorded, which is the expression's static
/// type; a `reduce` or a `reshape` changes the shape, so the emitted type is the
/// one the printed text has to name.
/// The shape an expression denotes, when it denotes one.
///
/// A `shape![..]` literal is a shape. A name bound to one is a shape. Anything
/// else is a value.
fn shape_of_expr(builder: &Builder, expr: &sile_hir::Expr) -> Option<ShapeExpr> {
    use sile_hir::Expr;
    match expr {
        Expr::Shape(shape) => Some(shape.clone()),
        Expr::Var(name) => builder
            .shape_bindings
            .iter()
            .rev()
            .find(|(bound, _)| bound == name)
            .map(|(_, shape)| shape.clone()),
        Expr::Builtin { op: sile_hir::BuiltinOp::ShapeOf, args } => match &args[0] {
            Expr::Var(buffer) => builder.tensors.iter().rev().find(|(n, _)| n == buffer).map(
                |(_, tensor)| shape_from_extents(&tensor.tile),
            ),
            _ => None,
        },
        _ => None,
    }
}

fn emitted_type_of(builder: &Builder, name: &str) -> Option<super::Type> {
    builder
        .values
        .iter()
        .rev()
        .find(|(bound, _, _)| bound == name)
        .and_then(|(_, _, ty)| ty.clone())
}

fn value_of(builder: &Builder, name: &str) -> ValueId {
    builder
        .values
        .iter()
        .rev()
        .find(|(bound, _, _)| bound == name)
        .map(|(_, value, _)| *value)
        .or_else(|| {
            builder
                .scalars
                .iter()
                .find(|(scalar, _, _)| scalar == name)
                .map(|(_, value, _)| *value)
        })
        .unwrap_or_else(|| panic!("`{name}` has no value in this program"))
}

fn emit_expr(
    builder: &mut Builder,
    expr: &sile_hir::Expr,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    use sile_hir::Expr;
    match expr {
        Expr::Var(name) => {
            // A const generic names a number, not a binding: cutile prints it as
            // a fresh `constant` wherever the body uses it as a value.
            if let Some((_, generic)) = kernel
                .const_params
                .iter()
                .find(|(generic, _)| generic == name)
                .filter(|_| emitted_type_of(builder, name).is_none())
                .filter(|_| !builder.scalars.iter().any(|(scalar, _, _)| scalar == name))
            {
                // A flag generic prints `1`: cutile compares it against a
                // literal, and the comparison reads the flag, not its value.
                let printed = if flag_generics(kernel).iter().any(|flag| flag == name) {
                    1
                } else {
                    *generic
                };
                let value = builder.i32_const(printed);
                // The latest `constant` of this generic is what a later
                // `convert_scalar` reads back.
                builder.const_constants.retain(|(n, _)| n != name);
                builder.const_constants.push((name.clone(), value));
                return (value, Builder::i32_tile());
            }
            let value = value_of(builder, name);
            // The type belongs to the value this name denotes *now*: a name
            // rebound to a different shape must not lend its new type to an
            // older use.
            match builder
                .values
                .iter()
                .rev()
                .find(|(_, id, _)| *id == value)
                .and_then(|(_, _, ty)| ty.clone())
            {
                Some(ty) => (value, ty),
                None => {
                    // A scalar parameter has no `values` entry; its type is the
                    // rank-zero tile it prints as.
                    if let Some((_, _, scalar)) =
                        builder.scalars.iter().find(|(_, id, _)| *id == value)
                    {
                        return (
                            value,
                            super::Type::Tile {
                                shape: Vec::new(),
                                element: TileElement::Scalar(*scalar),
                            },
                        );
                    }
                    (value, result_type_of(builder, expr, kernel))
                }
            }
        }
        Expr::Builtin { op, args } => emit_builtin(builder, *op, args, kernel),
        // A float literal in value position is a rank-0 `constant`.
        Expr::ScalarF32(value) => (
            builder.f32_const(*value, &[]),
            super::Type::Tile {
                shape: Vec::new(),
                element: TileElement::Scalar(ScalarType::F32),
            },
        ),
        // An integer literal in value position is a `constant`.
        Expr::ScalarI64(value) => (builder.i32_const(*value), Builder::i32_tile()),
        // A shape is metadata: binding one costs nothing and emits nothing.
        Expr::Shape(shape) => {
            builder.shapes.push((String::new(), shape.clone()));
            (ValueId(0), super::Type::Token)
        }
        other => panic!("the cutile emitter has no arm for {other:?}"),
    }
}


fn emit_builtin(
    builder: &mut Builder,
    op: sile_hir::BuiltinOp,
    args: &[sile_hir::Expr],
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    use sile_hir::{BuiltinOp, Expr};
    match op {
        // `a.broadcast(y.shape())` over a scalar: the target shape is asked for
        // twice, the scalar is reshaped to all-ones, and the broadcast names
        // the reshape. Rank is all that decides the counting.
        BuiltinOp::Broadcast => {
            let source = &args[0];
            let shape = &args[1];
            // `y.shape()` costs twice what a literal shape costs: cutile asks
            // the view for its extents, then asks again for the same numbers.
            let (target, rounds) = match shape {
                // A shape literal, or a name bound to one.
                _ if shape_of_expr(builder, shape).is_some()
                    && !matches!(
                        shape,
                        Expr::Builtin { op: BuiltinOp::ShapeOf, .. }
                    ) =>
                {
                    // A variable already bound to that shape is named once. A
                    // literal is named twice by a scalar source and once by a
                    // tile source; the source decides, below.
                    let rounds = match shape {
                        Expr::Shape(_) => 1usize,
                        _ => 2usize,
                    };
                    (shape_of_expr(builder, shape).unwrap(), rounds)
                }
                Expr::Builtin { op: BuiltinOp::ShapeOf, args } => {
                    let Expr::Var(buffer) = &args[0] else {
                        panic!("`shape_of` of a non-buffer: {args:?}");
                    };
                    let extents = shape_from_extents(&builder.tensor(buffer).tile);
                    // Asking a buffer for its shape materialises it, so a later
                    // `load_like` on the same buffer pays no extra round.
                    // A shape already asked for is one round cheaper: the round
                    // that first materialised it was paid elsewhere.
                    let rounds = if builder.shape_materialised.iter().any(|t| t == buffer) {
                        3usize
                    } else {
                        builder.shape_materialised.push(buffer.clone());
                        4usize
                    };
                    (extents, rounds)
                }
                other => panic!("no shape for a broadcast: {other:?}"),
            };
            // The source is computed first: its own constants come before the
            // target shape the broadcast names.
            let (scalar, source_ty) = emit_expr(builder, source, kernel);
            let _ = scalar;
            let extents = shape_extents(&target, &kernel.const_params);
            let rank = extents.len();
            // A tile broadcasts in place: the target's extents, the tile's own,
            // then the target's again. A scalar needs a reshape first, below.
            // A rank-0 source is a scalar and needs the reshape; anything of
            // rank one or more broadcasts in place.
            let source_is_tile = match &source_ty {
                super::Type::Tile { shape, .. } => !shape.is_empty(),
                _ => false,
            };
            if source_is_tile {
                let (source_value, source_ty) = (scalar, source_ty);
                let source_extents = match &source_ty {
                    super::Type::Tile { shape, .. } => shape.clone(),
                    _ => Vec::new(),
                };
                // A `y.shape()` target is asked for twice with the source's
                // extents between the two asks; a `shape![...]` literal is
                // named once, after the source.
                if matches!(shape, Expr::Builtin { op: BuiltinOp::ShapeOf, .. }) {
                    for value in &extents {
                        builder.i32_const(*value);
                    }
                }
                for value in &source_extents {
                    builder.i32_const(*value);
                }
                for value in &extents {
                    builder.i32_const(*value);
                }
                let result = builder.value();
                let element = match &source_ty {
                    super::Type::Tile { element, .. } => *element,
                    _ => TileElement::Scalar(ScalarType::F32),
                };
                let result_ty = super::Type::Tile {
                    shape: shape_extents(&target, &kernel.const_params),
                    element,
                };
                builder.push_op(Op {
                    results: vec![result],
                    format: Format::Plain,
                    opcode: "broadcast",
                    operands: vec![source_value],
                    result_types: vec![result_ty.clone()],
                    attrs: Vec::new(),
                    text: format!("{source_value}"),
                    operand_types: Some(vec![source_ty]),
                    indent: 0,
                });
                return (result, result_ty);
            }
            // A tile source under `y.shape()`: the target extents print first,
            // then the source's, then the target's again.
            // The target extents, `rounds` times over. Only the last round is
            // consumed: the rest are printed and dropped. A source that is a
            // bare scalar parameter costs one round more than one already held
            // as a value: the parameter's own shape has to be materialised
            // before the literal can name it.
            // The sources whose shape the kernel never named before cost one
            // round more: a scalar parameter, and a value read out of a shape,
            // which cutile keeps as metadata rather than as a value.
            let source_is_parameter = match source {
                Expr::Var(name) => {
                    builder.scalars.iter().any(|(scalar, _, _)| scalar == name)
                        || shape_derived::binding_is_shape_derived(&kernel.body, name)
                }
                expr => shape_derived::is_shape_derived(expr),
            };
            let rounds = if source_is_parameter && matches!(shape, Expr::Shape(_)) {
                rounds + 1
            } else {
                rounds
            };
            for _ in 0..rounds {
                for value in &extents {
                    builder.i32_const(*value);
                }
            }
            // The scalar is reshaped to a rank-`r` tile of ones.
            let ones = vec![1i64; rank];
            // The reshape names its target, so its own `r+1` extents follow.
            for _ in 0..ones.len() + 1 {
                builder.i32_const(1);
            }
            let element = match &source_ty {
                super::Type::Tile { element, .. } => *element,
                _ => TileElement::Scalar(ScalarType::F32),
            };
            let scalar_ty = super::Type::Tile {
                shape: Vec::new(),
                element,
            };
            let (reshaped, _) = builder.reshape_typed(scalar, &scalar_ty, &ones);
            // Then the shape the broadcast names: `r` ones and `r` extents.
            for _ in 0..rank {
                builder.i32_const(1);
            }
            for value in &extents {
                builder.i32_const(*value);
            }
            let target_shape = declared_extents(&target, &kernel.const_params)
                .into_iter()
                .map(|e| e.unwrap_or(-1))
                .collect::<Vec<_>>();
            builder.broadcast(reshaped, &ones, &target_shape, element)
        }
        // `constant(v, shape)`: cutile asks the view for its extents, then names
        // the result tile. A shape already asked for is not asked again.
        BuiltinOp::Constant => {
            // The value decides the element type: `true` gives an `i1` tile,
            // `1.5` an `f32` one, `1` an `i32` one.
            let (text, element) = match &args[0] {
                // A whole float prints with its point: `2.0`, not `2`.
                Expr::ScalarF32(value) => (
                    if value.fract() == 0.0 {
                        format!("{value:.1}")
                    } else {
                        value.to_string()
                    },
                    ScalarType::F32,
                ),
                Expr::ScalarBool(value) => (
                    format!("{}", if *value { 1 } else { 0 }),
                    ScalarType::I1,
                ),
                Expr::ScalarI64(value) => (format!("{value}"), ScalarType::I32),
                other => panic!("`constant` of a non-literal: {other:?}"),
            };
            let (shape, buffer) = match args.last() {
                Some(Expr::Builtin { op: BuiltinOp::ShapeOf, args }) => match &args[0] {
                    Expr::Var(buffer) => (None, Some(buffer.clone())),
                    other => panic!("`shape_of` of a non-buffer: {other:?}"),
                },
                Some(other) => match shape_of_expr(builder, other) {
                    Some(shape) => (Some(shape), None),
                    None => panic!("no shape for a constant: {other:?}"),
                },
                None => panic!("a constant with no shape"),
            };
            let extents = match (&shape, &buffer) {
                (Some(shape), _) => shape_extents(shape, &kernel.const_params),
                (None, Some(buffer)) => {
                    if !builder.shape_materialised.iter().any(|t| t == buffer) {
                        let tile = builder.tensor(buffer).tile.clone();
                        shape_consts(builder, &shape_from_extents(&tile), &kernel.const_params);
                        builder.shape_materialised.push(buffer.clone());
                    }
                    builder
                        .tensor(buffer)
                        .tile
                        .iter()
                        .map(|e| e.unwrap_or(-1))
                        .collect()
                }
                _ => panic!("a constant with no shape"),
            };
            let result = builder.value();
            let ty = super::Type::Tile {
                shape: extents.clone(),
                element: TileElement::Scalar(element),
            };
            builder.push_op(Op {
                results: vec![result],
                format: Format::Constant,
                opcode: "constant",
                operands: Vec::new(),
                result_types: vec![super::Type::Tile {
                    shape: extents.clone(),
                    element: TileElement::Scalar(element),
                }],
                attrs: Vec::new(),
                text: format!("<{element}: {text}>"),
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul if is_integer_expr(builder, kernel, args) => {
            match fold_integer(builder, op, args, kernel) {
                Some(folded) => folded,
                None => binary(builder, args, integer_opcode(op), kernel),
            }
        }
        BuiltinOp::Add => binary(builder, args, "addf", kernel),
        // `a & b` over masks: one op, and the printer states only the result
        // type, never the operand types.
        BuiltinOp::And | BuiltinOp::Or | BuiltinOp::Xor => {
            let opcode = match op {
                BuiltinOp::And => "andi",
                BuiltinOp::Or => "ori",
                _ => "xori",
            };
            let (lhs, lhs_ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode,
                operands: vec![lhs, rhs],
                result_types: vec![lhs_ty.clone()],
                attrs: Vec::new(),
                text: format!("{lhs}, {rhs}"),
                operand_types: None,
                indent: 0,
            });
            (result, lhs_ty)
        }
        // `max_tile(a, b)` / `min_tile(a, b)`: `maxf` states its rounding mode
        // and prints no operand types; `minf` is not used by these kernels.
        BuiltinOp::Max | BuiltinOp::Min => {
            let integer = is_integer_expr(builder, kernel, args);
            let opcode = match (matches!(op, BuiltinOp::Max), integer) {
                (true, false) => "maxf",
                (true, true) => "maxi",
                (false, false) => "minf",
                (false, true) => "mini",
            };
            let (lhs, lhs_ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode,
                operands: vec![lhs, rhs],
                result_types: vec![lhs_ty.clone()],
                attrs: Vec::new(),
                // An integer minimum states its signedness; the float ones state
                // a rounding mode.
                text: if integer {
                    format!("{lhs}, {rhs} signed")
                } else {
                    format!("{lhs}, {rhs} {{rounding_mode = 0}}")
                },
                operand_types: None,
                indent: 0,
            });
            (result, lhs_ty)
        }
        BuiltinOp::Mul => binary(builder, args, "mulf", kernel),
        BuiltinOp::Sub => binary(builder, args, "subf", kernel),
        // An integer operation whose interval is a single value never prints:
        // cutile allocates the id, proves the interval exact, drops the
        // operation and emits a `constant` instead. The dropped id stays a hole.
        // An integer operation whose interval is a single value never prints:
        // cutile allocates the id, proves the interval exact, drops the
        // operation and emits a `constant` instead. The dropped id stays a hole.
        // Floating point never folds, so only an integer expression qualifies.
        BuiltinOp::Div | BuiltinOp::Rem | BuiltinOp::CeilDiv
            if is_integer_expr(builder, kernel, args) =>
        {
            match fold_integer(builder, op, args, kernel) {
                Some(folded) => folded,
                None => binary_suffixed(
                    builder,
                    args,
                    integer_opcode(op),
                    integer_suffix(op),
                    kernel,
                ),
            }
        }
        // A float division prints as `divf`; only the integer path above ever
        // reaches an integer opcode. `true_div` states the two attributes that
        // make it approximate.
        BuiltinOp::Div => binary(builder, args, "divf", kernel),
        BuiltinOp::TrueDiv => {
            let (lhs, ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "divf",
                operands: vec![lhs, rhs],
                result_types: vec![ty.clone()],
                attrs: Vec::new(),
                text: format!("{lhs}, {rhs} rounding<approx> flush_to_zero"),
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        BuiltinOp::Rem => binary(builder, args, "remf", kernel),
        // `ceil_div(a, b)` prints as `divi ... rounding positive_inf`. A const
        // generic operand reads the value the prologue named, not a fresh
        // `constant`.
        BuiltinOp::CeilDiv => {
            let operand = |builder: &mut Builder, expr: &sile_hir::Expr| -> ValueId {
                if let sile_hir::Expr::Var(name) = expr {
                    if kernel.const_params.iter().any(|(generic, _)| generic == name) {
                        if let Some((_, id)) =
                            builder.prologue_generics.iter().find(|(n, _)| n == name)
                        {
                            return *id;
                        }
                    }
                }
                emit_expr(builder, expr, kernel).0
            };
            let lhs = operand(builder, &args[0]);
            let rhs = operand(builder, &args[1]);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "divi",
                operands: vec![lhs, rhs],
                result_types: vec![Builder::i32_tile()],
                attrs: Vec::new(),
                text: format!("{lhs}, {rhs} signed rounding positive_inf"),
                operand_types: None,
                indent: 0,
            });
            (result, Builder::i32_tile())
        }
        // `gt_tile(a, b)`: one predicate, ordered, with its comparison spelled.
        BuiltinOp::Gt => compare(builder, args, "greater_than", kernel),
        BuiltinOp::Lt => compare(builder, args, "less_than", kernel),
        BuiltinOp::Ge => compare(builder, args, "greater_equal", kernel),
        BuiltinOp::Le => compare(builder, args, "less_equal", kernel),
        BuiltinOp::Eq => compare(builder, args, "equal", kernel),
        BuiltinOp::Ne => compare(builder, args, "not_equal", kernel),
        // `select(cond, a, b)`: the condition first, then the two candidates.
        BuiltinOp::Select => {
            let (cond, cond_ty) = emit_expr(builder, &args[0], kernel);
            let (then, ty) = emit_expr(builder, &args[1], kernel);
            let (other, _) = emit_expr(builder, &args[2], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "select",
                operands: vec![cond, then, other],
                result_types: Vec::new(),
                attrs: Vec::new(),
                text: format!(
                    "{cond}, {then}, {other} : {}, {}",
                    cond_ty,
                    ty
                ),
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        // `t.reshape(shape![...])`: cutile names the target's extents first,
        // then the reshape itself.
        BuiltinOp::Reshape => {
            let (source, source_ty) = emit_expr(builder, &args[0], kernel);
            let target = match args.get(1).and_then(|shape| shape_of_expr(builder, shape)) {
                Some(shape) => shape_extents(&shape, &kernel.const_params),
                None => panic!("no shape for a reshape: {:?}", args.get(1)),
            };
            // The reshape names the source tile's extents, then the target's.
            let source_extents = match &source_ty {
                super::Type::Tile { shape, .. } => shape.clone(),
                _ => Vec::new(),
            };
            for extent in &source_extents {
                builder.i32_const(*extent);
            }
            for extent in &target {
                builder.i32_const(*extent);
            }
            builder.reshape_typed(source, &source_ty, &target)
        }
        // `t.transpose()`: an axis permutation of the source tile. The extent
        // constants come from the source tile's own shape.
        BuiltinOp::Transpose => {
            let (source, ty) = emit_expr(builder, &args[0], kernel);
            let source_extents = match &ty {
                super::Type::Tile { shape, element } => (shape.clone(), *element),
                _ => (Vec::new(), TileElement::Scalar(ScalarType::F32)),
            };
            for extent in &source_extents.0 {
                builder.i32_const(*extent);
            }
            let permutation: Vec<usize> = (0..source_extents.0.len()).rev().collect();
            let permuted = permutation
                .iter()
                .map(|axis| source_extents.0[*axis])
                .collect::<Vec<_>>();
            let result = builder.value();
            let result_ty = super::Type::Tile {
                shape: permuted,
                element: source_extents.1,
            };
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "permute",
                operands: vec![source],
                result_types: vec![result_ty.clone()],
                attrs: Vec::new(),
                text: format!(
                    "{source} [{}]",
                    permutation
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                operand_types: Some(vec![ty]),
                indent: 0,
            });
            (result, result_ty)
        }
        BuiltinOp::Unary(kind) => {
            let (source, ty) = emit_expr(builder, &args[0], kernel);
            let opcode = match kind {
                sile_hir::UnaryOp::Exp => "exp",
                sile_hir::UnaryOp::Exp2 => "exp2",
                sile_hir::UnaryOp::Log => "log",
                sile_hir::UnaryOp::Log2 => "log2",
                sile_hir::UnaryOp::Sqrt => "sqrt",
                sile_hir::UnaryOp::Rsqrt => "rsqrt",
                sile_hir::UnaryOp::Abs => "absf",
                sile_hir::UnaryOp::Neg => "negf",
                other => panic!("the cutile emitter has no arm for Unary({other:?})"),
            };
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode,
                operands: vec![source],
                result_types: vec![ty.clone()],
                attrs: Vec::new(),
                // `sqrt` states its rounding mode; the others name themselves.
                text: match kind {
                    sile_hir::UnaryOp::Sqrt => format!("{source} rounding<negative_inf>"),
                    _ => format!("{source}"),
                },
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        // `x.partition(shape)`: cutile compiles the shape argument (its
        // constants print) and then takes the tile shape from the argument's
        // static type, so the compiled value is printed and dropped. The view
        // itself is one op.
        BuiltinOp::Partition | BuiltinOp::PartitionMut => {
            let Expr::Var(name) = &args[0] else {
                panic!("`partition` of a non-buffer: {args:?}");
            };
            let tensor = builder.tensor(name).clone();
            let tile = match args.get(1).and_then(|shape| shape_of_expr(builder, shape)) {
                Some(shape) => shape_extents(&shape, &kernel.const_params),
                None => match args.get(1) {
                    Some(other) => panic!("no shape for a partition: {other:?}"),
                    None => Vec::new(),
                },
            };
            // The shape argument is compiled in full: the generic's own
            // constants, then the tile it names, then the source view's extents
            // twice — once as the view's metadata, once as the argument — and
            // only then the view. All of it prints; the tile shape itself is
            // taken from the argument's static type.
            // The first partition instantiates the kernel's generics again, so
            // every one of them prints; a later one prints only the generics its
            // own shape names.
            // A shape prints its own symbols in its own order.
            // A partition whose tile shape reuses a generic that a tensor's
            // declared extents also name instantiates the kernel's generics: the
            // first such partition prints all of them in declaration order. A
            // shape built only from generics the partition alone names
            // (`BLOCK_SIZE` in `rms_norm`) does not.
            let from_declaration = match args.get(1).and_then(|shape| shape_of_expr(builder, shape))
            {
                Some(shape) => {
                    let names = shape_names(&shape);
                    let declared: Vec<String> = kernel
                        .params
                        .iter()
                        .filter_map(|param| match &param.ty {
                            sile_hir::Type::Tensor { shape, .. } => Some(shape.clone()),
                            _ => None,
                        })
                        .flat_map(|shape| shape_names(&shape))
                        .collect();
                    names.iter().any(|name| declared.contains(name))
                }
                None => false,
            };
            let _ = from_declaration;
            // The tile the argument names.
            shape_consts(
                builder,
                &ShapeExpr::Tuple(tile.iter().map(|d| ShapeExpr::Constant(*d)).collect()),
                &kernel.const_params,
            );
            // The source's extents, twice: the view's metadata, then the
            // argument the dropped shape value carried.
            for _ in 0..2 {
                shape_consts(builder, &shape_from_extents(&tensor.tile), &kernel.const_params);
            }
            // A mutable source is written whole, so it is not padded.
            let padding_zero = !tensor.whole_buffer;
            let mut view = tensor.clone();
            view.tile = tile.iter().map(|e| Some(*e)).collect();
            let id = builder.make_partition_view(&view, &tile, padding_zero);
            // A *mutable* partition offsets every access by the tile block it
            // was built in: cutile asks for the block id again and multiplies
            // the block's row by the tile's row extent.
            if matches!(op, BuiltinOp::PartitionMut) {
                // The block id here is not assumed: cutile reads it straight
                // into the offset arithmetic.
                let base = builder.next_value;
                builder.next_value += 3;
                let ids = (0..3).map(|i| ValueId(base + i)).collect::<Vec<_>>();
                builder.push_op(Op {
                    results: ids.clone(),
                    format: Format::GetTileBlockId,
                    opcode: "get_tile_block_id",
                    operands: Vec::new(),
                    result_types: vec![Builder::i32_tile(); 3],
                    attrs: Vec::new(),
                    text: String::new(),
                    operand_types: None,
                    indent: 0,
                });
                // The offset on each axis is the block id times the ratio of the
                // buffer's extent to the tile's: cutile computes
                // `ceil(outer / nested)` per axis and skips the multiply when
                // the ratio is one.
                let outer = tile_shape_of_extents(&tensor.tile);
                let mut last = None;
                for (axis, nested) in tile.iter().enumerate() {
                    let outer_dim = outer.get(axis).copied().filter(|e| *e > 0);
                    let ratio = match outer_dim {
                        Some(outer_dim) if *nested > 0 => {
                            (outer_dim + nested - 1) / nested
                        }
                        _ => 1,
                    };
                    let offset = if ratio == 1 {
                        ids[axis]
                    } else {
                        let ratio_value = builder.i32_const(ratio);
                        let product = builder.value();
                        builder.push_op(Op {
                            results: vec![product],
                            format: Format::Plain,
                            opcode: "muli",
                            operands: vec![ids[axis], ratio_value],
                            result_types: vec![Builder::i32_tile()],
                            attrs: Vec::new(),
                            text: format!("{}, {ratio_value}", ids[axis]),
                            operand_types: None,
                            indent: 0,
                        });
                        product
                    };
                    last = Some(offset);
                }
                builder.store_offset = last;
                // The store's row index is this block id, not the prologue's.
                builder.store_block = ids.first().copied();
            }
            let ty = super::Type::PartitionView(super::PartitionView {
                tile_shape: tile.clone(),
                padding_zero,
                tensor: super::TensorView {
                    element: tensor.element,
                    shape: view_extents(&tensor),
                    strides: tensor.strides.clone(),
                },
            });
            (id, ty)
        }
        // Reading a one-element tile back as a scalar names the tile's rank as
        // a `constant` and then the same value; the tile itself is unchanged.
        // `tile_to_scalar(t)`: the tile's single value as a scalar. The op
        // prints nothing, but the value it yields is a rank-zero tile.
        BuiltinOp::TileToScalar => {
            let (value, ty) = emit_expr(builder, &args[0], kernel);
            let element = match &ty {
                super::Type::Tile { element, .. } => *element,
                _ => TileElement::Scalar(ScalarType::F32),
            };
            (
                value,
                super::Type::Tile {
                    shape: Vec::new(),
                    element,
                },
            )
        }
        // A scalar tile is the same value to cutile: the conversion names
        // nothing, because the tile and the scalar print alike.
        BuiltinOp::ScalarToTile => emit_expr(builder, &args[0], kernel),
        // `convert_scalar(N)`: an integer naming a float. `N` is a const
        // generic whose `constant` the prologue already printed, so the cast
        // reads that value back instead of printing a second one.
        BuiltinOp::ConvertScalar => {
            let source = match &args[0] {
                Expr::Var(name) => match builder.const_constants.iter().find(|(n, _)| n == name) {
                    Some((_, id)) => *id,
                    None => {
                        let id = builder.i32_const(
                            *kernel
                                .const_params
                                .iter()
                                .find(|(n, _)| n == name)
                                .map(|(_, v)| v)
                                .unwrap_or(&0),
                        );
                        builder.const_constants.push((name.clone(), id));
        builder.prologue_generics.push((name.clone(), id));
                        id
                    }
                },
                other => emit_expr(builder, other, kernel).0,
            };
            let result = builder.value();
            let ty = super::Type::Tile {
                shape: Vec::new(),
                element: TileElement::Scalar(ScalarType::F32),
            };
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "itof",
                operands: vec![source],
                result_types: vec![ty.clone()],
                attrs: Vec::new(),
                text: format!("{source} signed"),
                operand_types: Some(vec![Builder::i32_tile()]),
                indent: 0,
            });
            (result, ty)
        }
        // `mma(a, b, acc)`: one op, three tile operands, in that order.
        BuiltinOp::Mma => {
            let (lhs, lhs_ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, rhs_ty) = emit_expr(builder, &args[1], kernel);
            let (acc, acc_ty) = emit_expr(builder, &args[2], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "mmaf",
                operands: vec![lhs, rhs, acc],
                result_types: Vec::new(),
                attrs: Vec::new(),
                // `mmaf` states no result type after the arrow.
                text: format!(
                    "{lhs}, {rhs}, {acc} : {lhs_ty}, {rhs_ty}, {acc_ty}"
                ),
                operand_types: None,
                indent: 0,
            });
            (result, acc_ty)
        }
        // `iota(shape![N])`: one op naming the element type, then the tile it
        // spans. The length is named once more by the caller's own shape.
        BuiltinOp::Iota => {
            let shape = match args.first() {
                Some(shape) => shape_of_expr(builder, shape)
                    .unwrap_or(ShapeExpr::Tuple(Vec::new())),
                None => ShapeExpr::Tuple(Vec::new()),
            };
            let extents = shape_extents(&shape, &kernel.const_params);
            let result = builder.value();
            let ty = super::Type::Tile {
                shape: extents.clone(),
                element: TileElement::Scalar(ScalarType::I32),
            };
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "iota",
                operands: Vec::new(),
                result_types: vec![ty.clone()],
                attrs: Vec::new(),
                text: String::new(),
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        // `q.shape()[axis]`: cutile asks the view for its rank first — the
        // extents themselves are already the values the view was built with, so
        // reading one costs nothing more.
        BuiltinOp::ShapeDim => {
            let Expr::Var(name) = &args[0] else {
                panic!("`shape_dim` of a non-buffer: {args:?}");
            };
            let axis = match args.get(1) {
                Some(Expr::ScalarI64(axis)) => *axis as usize,
                _ => 0,
            };
            // `shape()` materialises every declared extent — printing `-1` for a
            // dynamic axis — and the subscript prints its own index. The value
            // the subscript yields is the extent the view already assumed, so
            // reading it costs no further op.
            let tensor = builder.tensor(name).clone();
            shape_consts(builder, &shape_from_extents(&tensor.tile), &kernel.const_params);
            builder.i32_const(axis as i64);
            let value = match tensor.view_shape.get(axis) {
                Some(Extent::Value(value)) => *value,
                Some(Extent::Literal(value)) => builder.i32_const(*value),
                None => builder.i32_const(-1),
            };
            (value, Builder::i32_tile())
        }
        // `num_tiles(&p, axis)`: cutile asks the partition view for the shape of
        // its index space, which is two run-time values, and takes one of them.
        BuiltinOp::NumTiles => {
            let Expr::Var(name) = &args[0] else {
                panic!("`num_tiles` of a non-buffer: {args:?}");
            };
            let tensor = builder.tensor(name).clone();
            let axis = match args.get(1) {
                Some(Expr::ScalarI64(axis)) => *axis as usize,
                _ => 0,
            };
            // The axis is an ordinary argument: cutile names it first.
            builder.i32_const(axis as i64);
            let view_ty = super::Type::PartitionView(super::PartitionView {
                tile_shape: tensor.tile.iter().map(|e| e.unwrap_or(-1)).collect(),
                padding_zero: !tensor.whole_buffer,
                tensor: super::TensorView {
                    element: tensor.element,
                    shape: view_extents(&tensor),
                    strides: tensor.strides.clone(),
                },
            });
            let base = builder.next_value;
            builder.next_value += 2;
            let ids = vec![ValueId(base), ValueId(base + 1)];
            builder.push_op(Op {
                results: ids.clone(),
                format: Format::Plain,
                opcode: "get_index_space_shape",
                operands: vec![tensor.view],
                result_types: vec![Builder::i32_tile()],
                attrs: Vec::new(),
                text: format!("{}", tensor.view),
                operand_types: Some(vec![view_ty]),
                indent: 0,
            });
            (ids[axis], Builder::i32_tile())
        }
        // `get_tile_block_id()`: three values, each assumed bounded. The kernel
        // names the whole tuple once and then reads its elements, so the block
        // id is fetched once and every later use reads the same three values.
        BuiltinOp::ProgramId => {
            if builder.program_id.is_none() {
                let ids = builder.block_id();
                builder.program_id = Some(ids);
            }
            let ids = builder.program_id.clone().unwrap_or_default();
            let index = match args.first() {
                Some(Expr::ScalarI64(index)) => *index as usize,
                _ => 0,
            };
            match ids.get(index).copied() {
                Some(id) => (id, Builder::i32_tile()),
                None => (ValueId(0), Builder::i32_tile()),
            }
        }
        // `fma(a, b, c)` lowers to `mulf a, b` then `addf c, product`: the
        // accumulator is the left operand of the sum.
        // `fma(a, b, c)`: one op, three operands, in that order.
        BuiltinOp::Fma => {
            let (a, ty) = emit_expr(builder, &args[0], kernel);
            let (b, _) = emit_expr(builder, &args[1], kernel);
            let (c, _) = emit_expr(builder, &args[2], kernel);
            let result = builder.value();
            builder.push_op(Op {
                results: vec![result],
                format: Format::Plain,
                opcode: "fma",
                operands: vec![a, b, c],
                result_types: vec![ty.clone()],
                attrs: Vec::new(),
                text: format!("{a}, {b}, {c}"),
                operand_types: None,
                indent: 0,
            });
            (result, ty)
        }
        BuiltinOp::ReduceMax => reduce(builder, args, "maxf", "-inf", kernel),
        BuiltinOp::ReduceSum => reduce(builder, args, "addf", "0", kernel),
        BuiltinOp::LoadLike => load_like(builder, args, kernel),
        BuiltinOp::Load => load(builder, args, kernel),
        other => panic!("the cutile emitter has no arm for {other:?}"),
    }
}

fn shape_from_extents(extents: &[Option<i64>]) -> ShapeExpr {
    ShapeExpr::Tuple(
        extents
            .iter()
            .map(|extent| match extent {
                Some(value) => ShapeExpr::Constant(*value),
                None => ShapeExpr::Dynamic,
            })
            .collect(),
    )
}

/// `reduce %t dim=d identities=[i] : tile<...> -> tile<...>` with a region.
///
/// The two operands of the region body get the ids immediately after the
/// result, and the body's own op takes the one after them: the numbering of a
/// reduce is result, then `a`, then `b`, then the combining op.
fn reduce(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    combine: &'static str,
    identity: &'static str,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    let (source, source_ty) = emit_expr(builder, &args[0], kernel);
    let axis = match args.get(1) {
        Some(sile_hir::Expr::ScalarI64(axis)) => *axis,
        _ => 0,
    };
    let shape = match &source_ty {
        super::Type::Tile { shape, .. } => shape.clone(),
        _ => Vec::new(),
    };
    let element = match &source_ty {
        super::Type::Tile { element, .. } => *element,
        _ => TileElement::Scalar(ScalarType::F32),
    };
    let reduced = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis as usize)
        .map(|(_, extent)| *extent)
        .collect::<Vec<_>>();
    // The region's two parameters are allocated first; the result follows them.
    let arg_a = builder.value();
    let arg_b = builder.value();
    let body = builder.value();
    let result = builder.value();
    let lhs = super::Type::Tile {
        shape: Vec::new(),
        element,
    };
    // `maxf` states its rounding mode; `addf` does not.
    let rounding = if combine == "maxf" {
        " {rounding_mode = 0}"
    } else {
        ""
    };
    let text = format!(
        "    {result} = reduce {source} dim={axis} identities=[{identity}] : {source_ty} -> {result_ty} {{\n\
         \x20   ^bb0({arg_a}: {lhs}, {arg_b}: {lhs}):\n\
         \x20     {body} = {combine} {arg_a}, {arg_b}{rounding} : {lhs}\n\
         \x20     yield {body} : {lhs}\n\
         \x20   }}",
        result_ty = super::Type::Tile {
            shape: reduced.clone(),
            element,
        },
    );
    builder.push_stmt(Stmt::Raw(text));
    (
        result,
        super::Type::Tile {
            shape: reduced,
            element,
        },
    )
}

/// The integer spelling of a comparison relation.
fn integer_relation(relation: &'static str) -> &'static str {
    match relation {
        "greater_than" => "greater_than",
        "greater_equal" => "greater_than_or_equal",
        "less_than" => "less_than",
        "less_equal" => "less_than_or_equal",
        "equal" => "equal",
        "not_equal" => "not_equal",
        other => other,
    }
}

/// `cmpf <relation> ordered a, b : tile<Nxf32> -> tile<Nxi1>`
fn compare(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    relation: &'static str,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    let (lhs, ty) = emit_expr(builder, &args[0], kernel);
    let (rhs, _) = emit_expr(builder, &args[1], kernel);
    let shape = match &ty {
        super::Type::Tile { shape, .. } => shape.clone(),
        _ => Vec::new(),
    };
    // An integer comparison states its signedness and drops `ordered`; the
    // float one keeps `ordered` and has no signedness to state.
    let integer = matches!(
        &ty,
        super::Type::Tile {
            element: TileElement::Scalar(ScalarType::I32 | ScalarType::I64),
            ..
        }
    );
    let result = builder.value();
    builder.push_op(Op {
        results: vec![result],
        format: Format::Plain,
        opcode: if integer { "cmpi" } else { "cmpf" },
        operands: vec![lhs, rhs],
        result_types: vec![super::Type::Tile {
            shape: shape.clone(),
            element: TileElement::Scalar(ScalarType::I1),
        }],
        attrs: Vec::new(),
        text: if integer {
            format!("{} {lhs}, {rhs}, signed", integer_relation(relation))
        } else {
            format!("{relation} ordered {lhs}, {rhs}")
        },
        operand_types: Some(vec![ty.clone()]),
        indent: 0,
    });
    (
        result,
        super::Type::Tile {
            shape,
            element: TileElement::Scalar(ScalarType::I1),
        },
    )
}

/// Print the statements added after `from` two spaces deeper.
fn indent_from(builder: &mut Builder, from: usize) {
    for stmt in builder.body.iter_mut().skip(from) {
        match stmt {
            super::Stmt::Raw(text) => {
                *text = text
                    .lines()
                    .map(|line| format!("  {line}"))
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            super::Stmt::Op(op) => op.indent += 2,
            super::Stmt::Return => {}
        }
    }
}

/// Whether an operation works on integers.
/// Whether a division's operands are integers.
///
/// A bound is one witness, but not the only one: a block id and a shape extent
/// are integers whose range cutile leaves unknown, and their division still
/// prints as `divi`, not `divf`.
fn is_integer_expr(builder: &Builder, kernel: &Kernel, args: &[sile_hir::Expr]) -> bool {
    fn integer(builder: &Builder, kernel: &Kernel, expr: &sile_hir::Expr) -> bool {
        use sile_hir::Expr;
        match expr {
            Expr::ScalarI64(_) => true,
            Expr::Var(name) => {
                if kernel.const_params.iter().any(|(n, _)| n == name) {
                    return true;
                }
                // A scalar parameter of integer type divides as an integer.
                if let Some((_, _, scalar)) = builder.scalars.iter().find(|(n, _, _)| n == name) {
                    return matches!(scalar, ScalarType::I32 | ScalarType::I64);
                }
                match builder
                    .values
                    .iter()
                    .rev()
                    .find(|(bound, _, _)| bound == name)
                    .and_then(|(_, _, ty)| ty.clone())
                {
                    Some(super::Type::Tile { element, .. }) => {
                        matches!(element, TileElement::Scalar(ScalarType::I32))
                    }
                    _ => builder.bounds.iter().any(|(n, _)| n == name),
                }
            }
            Expr::Builtin { op, args } => match op {
                sile_hir::BuiltinOp::ProgramId | sile_hir::BuiltinOp::ShapeDim => true,
                sile_hir::BuiltinOp::Add
                | sile_hir::BuiltinOp::Sub
                | sile_hir::BuiltinOp::Mul
                | sile_hir::BuiltinOp::Div
                | sile_hir::BuiltinOp::Rem
                | sile_hir::BuiltinOp::CeilDiv
                | sile_hir::BuiltinOp::Min
                | sile_hir::BuiltinOp::Max => args
                    .iter()
                    .all(|arg| integer(builder, kernel, arg)),
                _ => false,
            },
            _ => false,
        }
    }
    args.iter().all(|arg| integer(builder, kernel, arg))
}

/// The attribute an integer opcode prints after its operands.
///
/// Divide, remainder and minimum state their signedness; add and multiply do
/// not. A ceiling divide additionally states its rounding direction.
fn integer_suffix(op: sile_hir::BuiltinOp) -> &'static str {
    match op {
        sile_hir::BuiltinOp::Rem | sile_hir::BuiltinOp::Div => "signed",
        sile_hir::BuiltinOp::CeilDiv => "signed rounding positive_inf",
        _ => "",
    }
}

fn integer_opcode(op: sile_hir::BuiltinOp) -> &'static str {
    match op {
        sile_hir::BuiltinOp::Add => "addi",
        sile_hir::BuiltinOp::Sub => "subi",
        sile_hir::BuiltinOp::Mul => "muli",
        sile_hir::BuiltinOp::Rem => "remi",
        sile_hir::BuiltinOp::CeilDiv | sile_hir::BuiltinOp::Div => "divi",
        _ => "addi",
    }
}

/// The interval of an integer expression, when the emitter can prove one.
fn bounds_of(
    builder: &Builder,
    kernel: &Kernel,
    expr: &sile_hir::Expr,
) -> Option<super::bounds::Bounds> {
    use sile_hir::Expr;
    match expr {
        Expr::ScalarI64(value) => Some(super::bounds::Bounds::exact(*value)),
        Expr::Var(name) => {
            if let Some((_, bound)) = builder.bounds.iter().rev().find(|(n, _)| n == name) {
                return Some(*bound);
            }
            // A const generic is a known integer.
            kernel
                .const_params
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, value)| super::bounds::Bounds::exact(*value))
        }
        // `num_tiles(&p, axis)`: the tile count of that axis, which is
        // `ceil(extent / tile)` and exact when both are static.
        Expr::Builtin {
            op: sile_hir::BuiltinOp::NumTiles,
            args,
        } => {
            let Expr::Var(name) = args.first()? else {
                return None;
            };
            let axis = match args.get(1)? {
                Expr::ScalarI64(axis) => *axis as usize,
                _ => return None,
            };
            let tensor = builder.tensors.iter().find(|(n, _)| n == name)?.1.clone();
            let extent = match tensor.view_shape.get(axis)? {
                Extent::Literal(value) => *value,
                Extent::Value(_) => return None,
            };
            let tile = tensor.tile.get(axis)?.filter(|t| *t > 0)?;
            Some(super::bounds::Bounds::exact((extent + tile - 1) / tile))
        }
        Expr::Builtin { op, args } => {
            let lhs = bounds_of(builder, kernel, args.first()?)?;
            let rhs = match args.get(1) {
                Some(rhs) => bounds_of(builder, kernel, rhs)?,
                None => super::bounds::Bounds::exact(0),
            };
            super::bounds::transfer(*op, lhs, rhs)
        }
        _ => None,
    }
}

/// An integer operation that folds: the id is allocated, then dropped, and a
/// `constant` takes its place.
fn fold_integer(
    builder: &mut Builder,
    op: sile_hir::BuiltinOp,
    args: &[sile_hir::Expr],
    kernel: &Kernel,
) -> Option<(ValueId, super::Type)> {
    let lhs = bounds_of(builder, kernel, args.first()?)?;
    let rhs = match args.get(1) {
        Some(rhs) => bounds_of(builder, kernel, rhs)?,
        None => super::bounds::Bounds::exact(0),
    };
    let bounds = super::bounds::transfer(op, lhs, rhs)?;
    if !bounds.is_exact() {
        return None;
    }
    // Every operand prints unless the body already fetched that generic; only
    // the folded operation itself always disappears, leaving its id as a hole.
    for arg in args {
        // A literal operand is named once, not once per use.
        if let sile_hir::Expr::ScalarI64(_) = arg {
            continue;
        }
        if let sile_hir::Expr::Var(name) = arg {
            if builder.fetched_generics.iter().any(|fetched| fetched == name) {
                continue;
            }
        }
        let _ = emit_expr(builder, arg, kernel);
    }
    builder.discard();
    let value = builder.i32_const(bounds.start);
    Some((value, Builder::i32_tile()))
}

fn binary(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    opcode: &'static str,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    binary_suffixed(builder, args, opcode, "", kernel)
}

/// The same, with the trailing attribute an opcode states after its operands.
fn binary_suffixed(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    opcode: &'static str,
    suffix: &'static str,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    // A const generic names the value the prologue printed once, so an operand
    // that is one reads that id instead of spending a new `constant`.
    let operand = |builder: &mut Builder, expr: &sile_hir::Expr| -> (ValueId, super::Type) {
        if let sile_hir::Expr::Var(name) = expr {
            if let Some((_, id)) = builder.prologue_generics.iter().find(|(n, _)| n == name) {
                return (*id, Builder::i32_tile());
            }
        }
        emit_expr(builder, expr, kernel)
    };
    let (lhs, ty) = operand(builder, &args[0]);
    let (rhs, _) = operand(builder, &args[1]);
    // Integer operands take the integer opcode: `addi`, not `addf`. Only the
    // float spellings have an integer twin, and only these three do.
    let integer_twin = match opcode {
        "addf" => Some("addi"),
        "subf" => Some("subi"),
        "mulf" => Some("muli"),
        _ => None,
    };
    let opcode = match (integer_twin, &ty) {
        (Some(twin), super::Type::Tile { element: TileElement::Scalar(ScalarType::I32), .. }) => {
            twin
        }
        _ => opcode,
    };
    let result = builder.value();
    builder.push_op(Op {
        results: vec![result],
        format: Format::Plain,
        opcode,
        operands: vec![lhs, rhs],
        result_types: vec![ty.clone()],
        attrs: Vec::new(),
        text: if suffix.is_empty() {
            format!("{lhs}, {rhs}")
        } else {
            format!("{lhs}, {rhs} {suffix}")
        },
        operand_types: None,
        indent: 0,
    });
    (result, ty)
}

/// The type of an expression whose value the emitter already has.
fn result_type_of(builder: &Builder, expr: &sile_hir::Expr, kernel: &Kernel) -> super::Type {
    use sile_hir::Expr;
    match expr {
        Expr::Var(name) => match builder.locals.get(name) {
            Some(Type::Tile { elem, shape }) => tile_of(*elem, shape, &kernel.const_params),
            Some(Type::Tensor { elem, shape }) => tile_of(*elem, shape, &kernel.const_params),
            _ => super::Type::Token,
        },
        _ => panic!("no recorded type for {expr:?}"),
    }
}

fn tile_shape_of(builder: &Builder, name: &str, kernel: &Kernel) -> Vec<i64> {
    match builder.type_of(name) {
        Type::Tile { shape, .. } | Type::Tensor { shape, .. } => {
            declared_extents(&shape, &kernel.const_params)
                .into_iter()
                .map(|e| e.unwrap_or(-1))
                .collect()
        }
        other => panic!("`{name}` is not a tile: {other:?}"),
    }
}

/// `x.load_like(y)`: the tile of `x` sized like `y`.
fn load_like(builder: &mut Builder, args: &[sile_hir::Expr], kernel: &Kernel) -> (ValueId, super::Type) {
    use sile_hir::Expr;
    // `load_tile_mut(z)` names only its source: it reads the tile the buffer
    // already declares, the same way `load` does.
    if let Some(Expr::Var(source)) = args.first() {
        if args.len() == 1 {
            let tile = builder.tensor(source).tile.clone();
            let dims = tile.iter().map(|e| e.unwrap_or(-1)).collect::<Vec<_>>();
            return read_tile(builder, source, &dims, false, kernel);
        }
    }
    let (Expr::Var(source), Expr::Var(like)) = (&args[0], &args[1]) else {
        panic!("`load_like` of non-buffers: {args:?}");
    };
    let tile = tile_shape_of(builder, like, kernel);
    // `load_like` asks its target for `shape()` before it builds the partition
    // view. The first asker pays for that shape; a later one reads it back out
    // of the tensor's type metadata and pays nothing. That is why `saxpy`,
    // whose `broadcast(y.shape())` runs first, has no extra round here while
    // `async_add`, which asks inside `load_like`, has exactly one.
    if !builder.shape_materialised.iter().any(|t| t == like) {
        shape_consts(
            builder,
            &ShapeExpr::Tuple(tile.iter().map(|d| ShapeExpr::Constant(*d)).collect()),
            &kernel.const_params,
        );
        builder.shape_materialised.push(like.clone());
    }
    read_tile(builder, source, &tile, true, kernel)
}

/// `y.load()`: the tile the program owns, of its own declared shape.
fn load(builder: &mut Builder, args: &[sile_hir::Expr], kernel: &Kernel) -> (ValueId, super::Type) {
    use sile_hir::Expr;
    let Expr::Var(source) = &args[0] else {
        panic!("`load` of a non-buffer: {args:?}");
    };
    let tile = builder.tensor(source).tile.clone();
    let dims = tile.iter().map(|e| e.unwrap_or(-1)).collect::<Vec<_>>();
    read_tile_indexed(builder, source, &dims, false, args.get(1), kernel)
}

/// The four steps of a tile read: shape constants, the block id, the partition
/// view, and the load itself.
/// The four steps of a tile read: shape constants, the block id, the partition
/// view, and the load itself.
///
/// The shape is fetched three times over — twice before the block id and once
/// after — and a `load_like` adds the source buffer's extents to every round,
/// which is what makes it cost twice what a plain `load` costs.
fn read_tile(
    builder: &mut Builder,
    source: &str,
    tile: &[i64],
    with_source: bool,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    read_tile_indexed(builder, source, tile, with_source, None, kernel)
}

/// The indices the kernel named, as a shape expression, when it named any.
/// A tensor's declared extents, with a dynamic one standing for itself.
fn tile_shape_of_extents(extents: &[Option<i64>]) -> Vec<i64> {
    extents.iter().map(|e| e.unwrap_or(-1)).collect()
}

/// Every symbol a shape names, in order, without repeats.
fn shape_names(shape: &ShapeExpr) -> Vec<String> {
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

fn shape_items(shape: &ShapeExpr) -> Vec<ShapeExpr> {
    match shape {
        ShapeExpr::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    }
}

fn read_tile_indexed(
    builder: &mut Builder,
    source: &str,
    tile: &[i64],
    with_source: bool,
    index_arg: Option<&sile_hir::Expr>,
    kernel: &Kernel,
) -> (ValueId, super::Type) {
    let tensor = builder.tensor(source).clone();
    let tile_shape = ShapeExpr::Tuple(
        tile.iter().map(|d| ShapeExpr::Constant(*d)).collect::<Vec<_>>(),
    );
    // A read through a partition the body already built reuses that view: it
    // names its tile extents once and takes the block id it already has.
    if let Some(view) = tensor.partition_view {
        // The indices are evaluated first, then the tile's own extents.
        // The indices are the ones the kernel named, not the block id.
        // The indices are ordinary expressions: a loop variable is its
        // assumed value, `program_id(k)` is that component of the block id.
        // The HIR keeps an index as a shape of names, and a name that the body
        // bound is that binding's innermost value — inside a loop, the loop
        // variable rather than the prologue's block id.
        let index_ids = match index_arg {
            Some(sile_hir::Expr::TileIndex(items)) => Some(
                items
                    .iter()
                    .map(|item| emit_expr(builder, item, kernel).0)
                    .collect::<Vec<_>>(),
            ),
            Some(sile_hir::Expr::Shape(shape)) => Some(
                shape_items(shape)
                    .iter()
                    .map(|item| match item {
                        ShapeExpr::Symbol(name)
                            if builder.values.iter().any(|(n, _, _)| n == name) =>
                        {
                            value_of(builder, name)
                        }
                        other => shape_consts(builder, other, &kernel.const_params)
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| builder.i32_const(-1)),
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };
        if let Some(ids) = index_ids {
            if ids.len() >= tile.len() {
                // The tile's own extents follow the indices.
                shape_consts(builder, &tile_shape, &kernel.const_params);
                check_partition_access(builder, &tensor, tile, &ids, index_arg, kernel);
                let view_ty = super::Type::PartitionView(super::PartitionView {
                    tile_shape: tile.to_vec(),
                    padding_zero: true,
                    tensor: super::TensorView {
                        element: tensor.element,
                        shape: view_extents(&tensor),
                        strides: tensor.strides.clone(),
                    },
                });
                return builder.load_view_tko(view, &ids[..tile.len()], tensor.token, tile, tensor.element, view_ty);
            }
        }
        let indices = match builder.program_id.clone() {
            Some(ids) => ids,
            None => builder.block_id(),
        };
        let view_ty = super::Type::PartitionView(super::PartitionView {
            tile_shape: tile.to_vec(),
            padding_zero: true,
            tensor: super::TensorView {
                element: tensor.element,
                shape: view_extents(&tensor),
                strides: tensor.strides.clone(),
            },
        });
        return builder.load_view_tko(view, &indices[..tile.len()], tensor.token, tile, tensor.element, view_ty);
    }
    let source_shape = shape_from_extents(&tensor.tile);
    let round = |builder: &mut Builder| {
        shape_consts(builder, &tile_shape, &kernel.const_params);
        if with_source {
            shape_consts(builder, &source_shape, &kernel.const_params);
        }
    };
    round(builder);
    round(builder);
    let indices = builder.block_id();
    round(builder);
    // A view of a mutable buffer is not padded: its elements are all real,
    // whereas a read-only input may be read past its own extent.
    let padding_zero = !tensor.whole_buffer;
    let view = builder.make_partition_view(&tensor, tile, padding_zero);
    let view_ty = super::Type::PartitionView(super::PartitionView {
        tile_shape: tile.to_vec(),
        padding_zero,
        tensor: super::TensorView {
            element: tensor.element,
            shape: view_extents(&tensor),
            strides: tensor.strides.clone(),
        },
    });
    builder.load_view_tko(view, &indices[..tile.len()], tensor.token, tile, tensor.element, view_ty)
}

fn store_tile(
    builder: &mut Builder,
    target: &str,
    value: ValueId,
    value_ty: super::Type,
    store_index: Vec<ValueId>,
    kernel: &Kernel,
) {
    let tensor = builder.tensor(target).clone();
    let tile = tensor.tile.iter().map(|e| e.unwrap_or(-1)).collect::<Vec<_>>();
    // A store through a partition the body built reuses that view, and every
    // index it names is offset by the block the partition was made in.
    if let Some(view) = tensor.partition_view {
        // The row the partition owns is the offset it carries; the kernel's
        // index adds the column to it.
        let mut indices = store_index.clone();
        if let Some(block) = builder.store_block {
            if indices.len() >= 2 {
                // The store names the tile's shape once more, and the block the
                // partition owns is added to the row the kernel named.
                shape_consts(builder, &ShapeExpr::Tuple(
                    tile.iter().map(|d| ShapeExpr::Constant(*d)).collect(),
                ), &kernel.const_params);
                // The block the partition owns is the row's base, and the index
                // the kernel named is the offset inside it.
                let block_row = block;
                let row = indices[0];
                let sum = builder.value();
                builder.push_op(Op {
                    results: vec![sum],
                    format: Format::Plain,
                    opcode: "addi",
                    operands: vec![block_row, row],
                    result_types: vec![Builder::i32_tile()],
                    attrs: Vec::new(),
                    text: format!("{block_row}, {row}"),
                    operand_types: None,
                    indent: 0,
                });
                indices[0] = sum;
            }
        }
        if let Some(offset) = builder.store_offset {
            if indices.len() >= 2 {
                let column = indices[1];
                let sum = builder.value();
                builder.push_op(Op {
                    results: vec![sum],
                    format: Format::Plain,
                    opcode: "addi",
                    operands: vec![offset, column],
                    result_types: vec![Builder::i32_tile()],
                    attrs: Vec::new(),
                    text: format!("{offset}, {column}"),
                    operand_types: None,
                    indent: 0,
                });
                indices = vec![indices[0], sum];
            }
        }
        let view_ty = super::Type::PartitionView(super::PartitionView {
            tile_shape: tile.clone(),
            padding_zero: false,
            tensor: super::TensorView {
                element: tensor.element,
                shape: view_extents(&tensor),
                strides: tensor.strides.clone(),
            },
        });
        return builder.store_view_tko(value, view, &indices, tensor.token, value_ty, view_ty);
    }
    // The same shape fetches a load makes, then the block id and the write view.
    let tile_shape =
        ShapeExpr::Tuple(tile.iter().map(|d| ShapeExpr::Constant(*d)).collect::<Vec<_>>());
    for _ in 0..2 {
        shape_consts(builder, &tile_shape, &kernel.const_params);
    }
    let indices = builder.block_id();
    shape_consts(builder, &tile_shape, &kernel.const_params);
    let view = builder.make_partition_view(&tensor, &tile, false);
    let view_ty = super::Type::PartitionView(super::PartitionView {
        tile_shape: tile.clone(),
        padding_zero: false,
        tensor: super::TensorView {
            element: tensor.element,
            shape: view_extents(&tensor),
            strides: tensor.strides.clone(),
        },
    });
    builder.store_view_tko(value, view, &indices[..tile.len()], tensor.token, value_ty, view_ty);
}
