//! Building the cutile-shaped program from a typed kernel.
//!
//! The emitter constructs the one Tile IR — [`crate::ir::Module`], the
//! arena-structured port of cutile-ir — and the printer is
//! [`crate::ir::fmt`]. Every step is checked against the golden text in
//! `crates/sile/tests/fixtures/tileir/`, so a step is either exact or visibly
//! absent.
//!
//! Nothing here may be reordered or folded away: a value's number is its
//! allocation position in the module's value arena, so "cleaning up" the
//! output renumbers everything after it. `crates/sile-tile-ir/TILEIR_ALIGNMENT.md`
//! records the measured rules.

use sile_hir::passes::access::{
    shape_derived, shape_items, shape_names, AxisDecision, AxisExtent, TensorFacts,
};
use sile_hir::passes::predicate;
use sile_hir::{ElemType, Kernel, ParamKind, ShapeExpr, Type};

use crate::ir::{
    append_op, build_single_block_region, Attribute, Bounded, DenseElements, FuncType, Location,
    Module as IrModule, OpBuilder, PaddingValue, PointerType, RegionId, ScalarType as IrScalar,
    TileElementType, TileType, Type as IrType, Value as IrValue, ValueProducer, DYNAMIC,
};
use crate::opcode::Opcode;

/// One tensor view extent: a literal, or the value an `assume` produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    Literal(i64),
    Value(IrValue),
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
fn scalar_of(elem: ElemType) -> IrScalar {
    match elem {
        ElemType::F32 => IrScalar::F32,
        ElemType::F16 => IrScalar::F16,
        ElemType::I64 => IrScalar::I32,
        ElemType::Bool => IrScalar::I1,
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
    /// The module under construction. Values are numbered by allocation, which
    /// is the order the printer prints — so the emitter's walk order is the
    /// output's numbering.
    module: IrModule,
    /// The entry region, attached to the `entry` op once the body is done.
    entry_region: RegionId,
    /// The blocks being written, innermost last. The entry block is the bottom
    /// of the stack; a `for` or an `if` pushes its region's block for the
    /// duration of its body.
    blocks: Vec<crate::ir::BlockId>,
    /// The checks' decisions: assumptions, intervals, terms, compile errors,
    /// launch checks. Every judgment the printed text reflects is made there;
    /// the emitter only consumes the answers.
    access: sile_hir::passes::access::AccessAnalysis,
    args: Vec<(IrValue, IrType)>,
    /// Every tensor parameter, by name, with what the body needs to reach it.
    tensors: Vec<(String, Tensor)>,
    /// The scalar parameters, by name: their entry value and element type.
    scalars: Vec<(String, IrValue, IrScalar)>,
    /// The tile a binding names, for every name the body has bound so far.
    values: Vec<(String, IrValue, Option<IrType>)>,
    /// The type of every binding, as `sile-hir`'s checker recorded it.
    locals: std::collections::BTreeMap<String, Type>,
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
    program_id: Option<Vec<IrValue>>,
    /// The `constant` a const generic already printed, by name.
    const_constants: Vec<(String, IrValue)>,
    /// The id each const generic got in the prologue. A body operand that is a
    /// generic reads this value, which the prologue printed once.
    prologue_generics: Vec<(String, IrValue)>,
    /// The offset a mutable partition added to every store through it.
    store_offset: Option<IrValue>,
    /// The block id the mutable partition fetched, which its stores index by.
    store_block: Option<IrValue>,
    /// The const generics a partition already fetched inside the body. A later
    /// integer fold reads them from here instead of printing them again, which
    /// is why `K / BK` in `gemm` costs one id and no operands.
    fetched_generics: Vec<String>,
    /// What each enclosing `for` carries, innermost last. An `if` inside a loop
    /// yields those values too: the region has to hand them on.
    loop_carried: Vec<Vec<String>>,
    /// Scratch space for a shape expression that appears in value position.
    shapes: Vec<(String, ShapeExpr)>,
}

/// What the body knows about one tensor parameter.
#[derive(Clone)]
struct Tensor {
    /// The parameter this tensor came from. A launch check names its operands by
    /// parameter position, so the name is what ties a view back to one.
    name: String,
    /// The view a `partition` already built, when this tensor is one.
    partition_view: Option<IrValue>,
    /// The view the prologue built for it.
    view: IrValue,
    /// The token `make_tensor_view` was given, which every access reuses.
    token: IrValue,
    /// The extents the kernel declared — the tile a `load` reads, and the
    /// shape a `broadcast` to `y.shape()` names.
    tile: Vec<Option<i64>>,
    element: IrScalar,
    strides: Vec<i64>,
    /// Whether the view spans the whole buffer (a mutable parameter) or only
    /// its declared tile.
    whole_buffer: bool,
    /// The extents the tensor view was built with, as it prints them.
    view_shape: Vec<Extent>,
}

/// What a condition folds to.
enum Folded {
    /// Decided while compiling: the comparison was folded away.
    Known(bool),
    /// A runtime value the region branches on.
    Value(IrValue),
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
    // A comparison against a literal decides the branch by the generic's own
    // value, which the IR prints as `1` for a flag. The printed value is what
    // the comparison reads, but it is not what the branch means: folding
    // `if CAUSAL == 1` to the taken arm for every instantiation made the
    // unmasked run execute the masked body.
    if let Some(known) = true_literal_comparison(kernel, expr)
        .or_else(|| builder.access.literal_comparison(kernel, expr))
    {
        let Expr::Builtin { args, .. } = expr else {
            unreachable!()
        };
        let _ = emit_expr(builder, &args[1], kernel);
        builder.discard();
        builder.i32_const_i1(known);
        return Folded::Known(known);
    }
    match emit_expr(builder, expr, kernel) {
        (value, _) => Folded::Value(value),
    }
}

/// A comparison between a const generic and a literal, decided by the
/// generic's declared value rather than by the `1` the IR prints for a flag.
fn true_literal_comparison(kernel: &Kernel, expr: &sile_hir::Expr) -> Option<bool> {
    use sile_hir::{BuiltinOp, Expr};
    let Expr::Builtin { op, args } = expr else {
        return None;
    };
    let value_of = |arg: &Expr| -> Option<i64> {
        match arg {
            Expr::ScalarI64(value) => Some(*value),
            Expr::Var(name) => kernel
                .const_params
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, value)| *value),
            _ => None,
        }
    };
    let lhs = value_of(args.first()?)?;
    let rhs = value_of(args.get(1)?)?;
    Some(match op {
        BuiltinOp::Eq => lhs == rhs,
        BuiltinOp::Ne => lhs != rhs,
        BuiltinOp::Lt => lhs < rhs,
        BuiltinOp::Le => lhs <= rhs,
        BuiltinOp::Gt => lhs > rhs,
        BuiltinOp::Ge => lhs >= rhs,
        _ => return None,
    })
}

/// The extents a view names, as the IR spells them: a dynamic one is `DYNAMIC`.
fn ir_extents(extents: &[Extent]) -> Vec<i64> {
    extents
        .iter()
        .map(|extent| match extent {
            Extent::Literal(value) => *value,
            Extent::Value(_) => DYNAMIC,
        })
        .collect()
}

/// The tile type of a scalar element and a shape.
fn tile_type(element: IrScalar, shape: &[i64]) -> IrType {
    IrType::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(element),
    })
}

/// The scalar a tile type carries, when it carries one.
fn tile_scalar(ty: &IrType) -> Option<IrScalar> {
    match ty {
        IrType::Tile(t) => match &t.element_type {
            TileElementType::Scalar(s) => Some(*s),
            TileElementType::Pointer(_) => None,
        },
        _ => None,
    }
}

/// The shape a tile type carries, empty for a scalar tile.
fn tile_shape(ty: &IrType) -> Vec<i64> {
    match ty {
        IrType::Tile(t) => t.shape.clone(),
        _ => Vec::new(),
    }
}

/// The tensor view type of a tensor, as the IR spells it.
fn tensor_view_type(element: IrScalar, view_shape: &[Extent], strides: &[i64]) -> IrType {
    IrType::TensorView(crate::ir::TensorViewType {
        element_type: element,
        shape: ir_extents(view_shape),
        strides: strides.to_vec(),
    })
}

/// The partition view type of a tensor, as the IR spells it.
fn partition_view_type(
    tile: &[i64],
    element: IrScalar,
    view_shape: &[Extent],
    strides: &[i64],
    padding_zero: bool,
) -> IrType {
    let tensor_view = match tensor_view_type(element, view_shape, strides) {
        IrType::TensorView(tv) => tv,
        _ => unreachable!(),
    };
    IrType::PartitionView(crate::ir::PartitionViewType {
        tile_shape: tile.iter().map(|d| *d as i32).collect(),
        tensor_view,
        dim_map: (0..tile.len() as i32).collect(),
        padding_value: padding_zero.then_some(PaddingValue::Zero),
    })
}

impl Builder {
    fn new(typed: &sile_hir::typeck::TypedKernel) -> Self {
        let mut module = IrModule::new(typed.kernel.origin.module.clone());
        // The entry block exists from the start: parameters allocate their
        // block-arg values into it as `declare_params` walks the list.
        let (entry_region, entry_block, _) = build_single_block_region(&mut module, &[]);
        Self {
            module,
            entry_region,
            blocks: vec![entry_block],
            access: sile_hir::passes::access::AccessAnalysis::new(&typed.checks),
            args: Vec::new(),
            tensors: Vec::new(),
            scalars: Vec::new(),
            values: Vec::new(),
            locals: typed.locals.clone(),
            shape_materialised: Vec::new(),
            shape_bindings: Vec::new(),
            program_id: None,
            const_constants: Vec::new(),
            prologue_generics: Vec::new(),
            store_offset: None,
            store_block: None,
            fetched_generics: Vec::new(),
            loop_carried: Vec::new(),
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

    /// The block being written.
    fn block(&self) -> crate::ir::BlockId {
        *self.blocks.last().unwrap()
    }

    /// Build an operation and append it to the block being written. The
    /// results it returns are numbered by allocation — building *is* naming.
    fn append(&mut self, op: OpBuilder) -> Vec<IrValue> {
        let block = self.block();
        let (id, results) = op.build(&mut self.module);
        append_op(&mut self.module, block, id);
        results
    }

    fn append1(&mut self, op: OpBuilder) -> IrValue {
        self.append(op)[0]
    }

    /// A fresh entry parameter.
    fn arg(&mut self, ty: IrType) -> IrValue {
        let block = self.block();
        let value = self.module.alloc_value(
            ty.clone(),
            ValueProducer::BlockArg {
                block,
                arg_index: self.args.len() as u32,
            },
        );
        self.module.block_mut(block).args.push((value, ty.clone()));
        self.args.push((value, ty));
        value
    }

    /// A `constant` of any element and shape, one splatted value.
    fn const_splat(&mut self, element: IrScalar, extents: &[i64], bytes: Vec<u8>) -> IrValue {
        self.append1(
            OpBuilder::new(Opcode::Constant, Location::Unknown)
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: tile_type(element, extents),
                        shape: extents.to_vec(),
                        data: bytes,
                    }),
                )
                .result(tile_type(element, extents)),
        )
    }

    /// `%t = constant <i1: v> : tile<i1>`
    fn i32_const_i1(&mut self, value: bool) -> IrValue {
        self.const_splat(IrScalar::I1, &[], vec![value as u8])
    }

    /// The rank-zero `tile<i32>` type.
    fn i32_tile() -> IrType {
        tile_type(IrScalar::I32, &[])
    }

    /// `%t = constant <i32: v> : tile<i32>`
    fn i32_const(&mut self, value: i64) -> IrValue {
        self.const_splat(IrScalar::I32, &[], (value as i32).to_le_bytes().to_vec())
    }

    /// `%t = constant <f32: v> : tile<Nxf32>` — one splatted value.
    fn f32_const(&mut self, value: f32, extents: &[i64]) -> IrValue {
        let count: usize = extents.iter().product::<i64>().max(1) as usize;
        let mut data = Vec::with_capacity(count * 4);
        for _ in 0..count {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.const_splat(IrScalar::F32, extents, data)
    }

    /// A two-operand integer op: `signed` states the signedness division,
    /// remainder and minimum print; `rounding` is the non-default rounding a
    /// ceiling divide states.
    fn int_binary(
        &mut self,
        opcode: Opcode,
        lhs: IrValue,
        rhs: IrValue,
        signed: bool,
        rounding: Option<i64>,
    ) -> IrValue {
        let mut op = OpBuilder::new(opcode, Location::Unknown)
            .operand(lhs)
            .operand(rhs);
        if signed {
            op = op.attr("signedness", Attribute::i32(1));
        }
        if let Some(rounding) = rounding {
            op = op.attr("rounding", Attribute::i32(rounding));
        }
        self.append1(op.result(Self::i32_tile()))
    }

    /// A two-operand boolean op over `tile<i1>`.
    fn bool_binary(&mut self, opcode: Opcode, lhs: IrValue, rhs: IrValue) -> IrValue {
        self.append1(
            OpBuilder::new(opcode, Location::Unknown)
                .operand(lhs)
                .operand(rhs)
                .result(tile_type(IrScalar::I1, &[])),
        )
    }

    /// A two-operand op whose result repeats the first operand's type.
    fn same_binary(&mut self, opcode: Opcode, lhs: IrValue, rhs: IrValue, ty: IrType) -> IrValue {
        self.append1(
            OpBuilder::new(opcode, Location::Unknown)
                .operand(lhs)
                .operand(rhs)
                .result(ty),
        )
    }

    /// `%t = cmpi <pred> %lhs, %rhs, signed : tile<i32> -> tile<i1>` — the
    /// rank-zero comparison the access checks are built from.
    fn cmpi(&mut self, pred: i64, lhs: IrValue, rhs: IrValue) -> IrValue {
        self.append1(
            OpBuilder::new(Opcode::CmpI, Location::Unknown)
                .attr("comparison_predicate", Attribute::i32(pred))
                .attr("signedness", Attribute::i32(1))
                .operand(lhs)
                .operand(rhs)
                .result(tile_type(IrScalar::I1, &[])),
        )
    }

    /// `assert %c, "msg" : tile<i1>`
    fn assert_op(&mut self, cond: IrValue, message: &str) {
        self.append(
            OpBuilder::new(Opcode::Assert, Location::Unknown)
                .operand(cond)
                .attr("message", Attribute::String(message.to_string())),
        );
    }

    /// `program_id(axis)`: cutile's `program_id` is a plain Rust function, so
    /// the axis is an `if` chain over a constant. The axis argument compiles
    /// first, then the triple, then one `axis == k` test per arm the chain
    /// walks: each test names `k` and yields a constant predicate, and the
    /// compared id between them is allocated and dropped.
    fn program_id_checked(&mut self, axis: i64) -> IrValue {
        self.i32_const(axis);
        let ids = self.block_id();
        let arms = axis.clamp(0, 2);
        let mut last: Option<IrValue> = None;
        for candidate in 0..=arms {
            self.i32_const(candidate);
            // The comparison is decided while compiling, so its id is spent
            // and its value is a constant.
            self.discard();
            last = Some(self.i32_const_i1(candidate == axis));
        }
        // Only the `else` arm carries the assertion, and it tests the predicate
        // the last comparison already produced.
        if axis >= 2 {
            self.assert_op(last.unwrap(), "program_id: axis must be 0, 1, or 2");
        }
        ids[arms as usize]
    }

    /// `num_programs(axis)`: the same `if` chain as [`Self::program_id_checked`],
    /// over the grid extent triple and with its own assertion text.
    fn num_programs(&mut self, axis: i64) -> IrValue {
        self.i32_const(axis);
        let ids = self.append(
            OpBuilder::new(Opcode::GetNumTileBlocks, Location::Unknown).results([
                Self::i32_tile(),
                Self::i32_tile(),
                Self::i32_tile(),
            ]),
        );
        let assumed = ids.iter().map(|&id| self.assume(id)).collect::<Vec<_>>();
        let arms = axis.clamp(0, 2);
        let mut last: Option<IrValue> = None;
        for candidate in 0..=arms {
            self.i32_const(candidate);
            self.discard();
            last = Some(self.i32_const_i1(candidate == axis));
        }
        if axis >= 2 {
            self.assert_op(last.unwrap(), "num_programs: axis must be 0, 1, or 2");
        }
        assumed[arms as usize]
    }

    /// `%t = assume bounded<0, ?>, %a : tile<i32>`, or `bounded<0, N>` when the
    /// interval is known.
    fn assume_bounded(&mut self, subject: IrValue, ub: Option<i64>) -> IrValue {
        self.append1(
            OpBuilder::new(Opcode::Assume, Location::Unknown)
                .attr(
                    "predicate",
                    Attribute::Bounded(Bounded {
                        lb: Some(0),
                        ub,
                    }),
                )
                .operand(subject)
                .result(Self::i32_tile()),
        )
    }

    fn assume(&mut self, subject: IrValue) -> IrValue {
        self.assume_bounded(subject, None)
    }

    fn make_token(&mut self) -> IrValue {
        self.append1(
            OpBuilder::new(Opcode::MakeToken, Location::Unknown).result(IrType::Token),
        )
    }

    /// `%t = make_tensor_view %p, shape = [...], strides = [...] : ... -> ...`
    ///
    /// A dynamic extent is spelled `DYNAMIC` in the type and named by the next
    /// dynamic operand; a fully static view names none.
    fn make_tensor_view(
        &mut self,
        ptr: IrValue,
        element: IrScalar,
        view_shape: &[Extent],
        strides: &[i64],
    ) -> IrValue {
        let dynamic: Vec<IrValue> = view_shape
            .iter()
            .filter_map(|extent| match extent {
                Extent::Value(value) => Some(*value),
                Extent::Literal(_) => None,
            })
            .collect();
        self.append1(
            OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(ptr)
                .operands(dynamic.iter().copied())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(dynamic.len() as i64),
                        Attribute::i32(0),
                    ]),
                )
                .result(tensor_view_type(element, view_shape, strides)),
        )
    }

    /// `%t = make_partition_view %v : partition_view<...>`
    ///
    /// A read view carries `padding_value = zero`; a write view does not.
    fn make_partition_view(
        &mut self,
        source: IrValue,
        tile: &[i64],
        element: IrScalar,
        view_shape: &[Extent],
        strides: &[i64],
        padding_zero: bool,
    ) -> IrValue {
        self.append1(
            OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                .operand(source)
                .result(partition_view_type(
                    tile,
                    element,
                    view_shape,
                    strides,
                    padding_zero,
                )),
        )
    }

    /// `%t, %u = load_view_tko weak %view[i, j] token = %tok : ...`
    fn load_view_tko(
        &mut self,
        view: IrValue,
        indices: &[IrValue],
        token: IrValue,
        tile: &[i64],
        element: IrScalar,
        view_ty: IrType,
    ) -> (IrValue, IrType) {
        let tile_ty = tile_type(element, tile);
        let results = self.append(
            OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .operand(view)
                .operands(indices.iter().copied())
                .operand(token)
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(indices.len() as i64),
                        Attribute::i32(1),
                    ]),
                )
                .result(tile_ty.clone())
                .result(IrType::Token),
        );
        let _ = view_ty;
        (results[0], tile_ty)
    }

    /// `%t = store_view_tko weak %value, %view[i, j] token = %tok : ...`
    ///
    /// The store's printed type list ends at the token: value, view and index
    /// types come from the operands, and `token` closes the line.
    fn store_view_tko(
        &mut self,
        value: IrValue,
        view: IrValue,
        indices: &[IrValue],
        token: IrValue,
        value_ty: IrType,
        view_ty: IrType,
    ) {
        self.append(
            OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .operand(value)
                .operand(view)
                .operands(indices.iter().copied())
                .operand(token)
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(indices.len() as i64),
                        Attribute::i32(1),
                    ]),
                )
                .result(IrType::Token),
        );
        let _ = (value_ty, view_ty);
    }

    /// The id of an operation that was allocated and then proved constant.
    ///
    /// cutile allocates the operation, runs its interval transfer, and returns a
    /// fresh `constant` when the interval is a single value — the operation is
    /// never appended, so its id is printed nowhere. The gap is part of the
    /// contract: the arena keeps the allocated value, and the hole it leaves in
    /// the numbering is what cutile's text shows. The sentinel producer never
    /// matches a real operation, so the hole stays a hole.
    fn discard(&mut self) -> IrValue {
        self.module.alloc_value(
            IrType::Token,
            ValueProducer::OpResult {
                op: crate::ir::OpId(u32::MAX),
                result_index: 0,
            },
        )
    }

    /// `%t, %u, %w = get_tile_block_id : tile<i32>` — three results — then one
    /// `assume` per result. Every memory access asks for it again.
    fn block_id(&mut self) -> Vec<IrValue> {
        let ids = self.append(
            OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown).results([
                Self::i32_tile(),
                Self::i32_tile(),
                Self::i32_tile(),
            ]),
        );
        ids.iter().map(|&id| self.assume(id)).collect()
    }

    fn reshape_typed(
        &mut self,
        source: IrValue,
        source_ty: &IrType,
        to: &[i64],
    ) -> (IrValue, IrType) {
        let element = match tile_scalar(source_ty) {
            Some(s) => s,
            None => IrScalar::F32,
        };
        let to_ty = tile_type(element, to);
        let result = self.append1(
            OpBuilder::new(Opcode::Reshape, Location::Unknown)
                .operand(source)
                .result(to_ty.clone()),
        );
        (result, to_ty)
    }

    fn broadcast(
        &mut self,
        source: IrValue,
        _from: &[i64],
        to: &[i64],
        element: IrScalar,
    ) -> (IrValue, IrType) {
        let to_ty = tile_type(element, to);
        let result = self.append1(
            OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                .operand(source)
                .result(to_ty.clone()),
        );
        (result, to_ty)
    }
}

/// The element type of a HIR element type, as a tile spells it.
fn tile_of(elem: ElemType, shape: &ShapeExpr, consts: &[(String, i64)]) -> IrType {
    let dims = declared_extents(shape, consts)
        .into_iter()
        .map(|extent| extent.unwrap_or(-1))
        .collect::<Vec<_>>();
    tile_type(scalar_of(elem), &dims)
}

/// The `constant` fetches of one shape argument, in source order.
///
/// A literal extent costs one `constant`; a const generic costs one carrying
/// its value; a dynamic extent costs one carrying `-1`. Every call is a fresh
/// fetch: cutile asks for the shape again for each operation that takes one.
fn shape_consts(builder: &mut Builder, shape: &ShapeExpr, consts: &[(String, i64)]) -> Vec<IrValue> {
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

/// The program cutile prints for `kernel`.
pub fn emit(
    typed: &sile_hir::typeck::TypedKernel,
) -> Result<(IrModule, Vec<predicate::LaunchCheck>), String> {
    let kernel = &typed.kernel;
    let mut builder = Builder::new(typed);
    declare_params(&mut builder, kernel);
    // A kernel whose body partitions a buffer by generics that its own
    // parameter types also name instantiates those generics up front: cutile
    // prints all of them once, before the body's first op. A shape built only
    // from generics the partition alone names (`BLOCK_SIZE` in `rms_norm`)
    // does not.
    if builder.access.instantiates_generics() {
        builder.fetched_generics = kernel
            .const_params
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        for (name, value) in &kernel.const_params {
            // A flag generic prints `1`: the body only ever compares it.
            let printed = if builder.access.flags().iter().any(|flag| flag == name) {
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
    // `return` closes the entry block.
    let block = builder.block();
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut builder.module);
    append_op(&mut builder.module, block, ret);

    // A failed access check is a compile error, not a device assert, so no
    // module is printed for it.
    if let Some(error) = builder.access.errors.first() {
        return Err(error.clone());
    }

    // The entry operation wraps the block the emitter has been writing into.
    let arg_types = builder.args.iter().map(|(_, ty)| ty.clone()).collect();
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr(
            "sym_name",
            Attribute::String(format!("{}_entry", kernel.name)),
        )
        .attr(
            "function_type",
            Attribute::Type(IrType::Func(FuncType {
                inputs: arg_types,
                results: Vec::new(),
            })),
        )
        .attr("di_name", Attribute::String(kernel.name.clone()))
        .region(builder.entry_region)
        .build(&mut builder.module);
    builder.module.functions.push(entry);

    let module = builder.module;
    let launch_checks = builder.access.launch_checks;
    Ok((module, launch_checks))
}

/// What the checks read about one tensor: its parameter, its tile, and what
/// each view axis is. The emitter holds the tensor's values; this is the slice
/// of it a decision needs.
fn tensor_facts(kernel: &Kernel, tensor: &Tensor) -> TensorFacts {
    TensorFacts {
        name: tensor.name.clone(),
        param: sile_hir::passes::access::buffer_param_position(kernel, &tensor.name),
        tile: tensor.tile.clone(),
        view_axes: tensor
            .view_shape
            .iter()
            .map(|extent| match extent {
                Extent::Literal(v) => AxisExtent::Static(*v),
                Extent::Value(_) => AxisExtent::Dynamic,
            })
            .collect(),
    }
}

/// The entry parameters, in declaration order, and then one view per tensor.
fn declare_params(builder: &mut Builder, kernel: &Kernel) {
    let mut tensors: Vec<(String, Tensor, Vec<IrValue>, Vec<Option<i64>>, ElemType)> =
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
                let ptr = builder.arg(IrType::Tile(TileType {
                    shape: Vec::new(),
                    element_type: TileElementType::Pointer(Box::new(PointerType {
                        pointee: scalar_of(*elem),
                    })),
                }));
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
                        name: param.name.clone(),
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
                let value = builder.arg(tile_type(scalar_of(*elem), &[]));
                builder.scalars.push((param.name.clone(), value, scalar_of(*elem)));
                if matches!(scalar_of(*elem), IrScalar::I32) {
                    builder.access.bind_int(&param.name);
                }
            }
            Type::Partition { .. } | Type::Shape(_) => {}
        }
    }

    // ── Constant generics ──────────────────────────────────────────
    //
    // Every const generic prints, in declaration order, whether or not the body
    // reads it: cutile emits one `constant` per generic. A flag generic is one
    // the body compares against a literal, and cutile prints `1` for it.
    for (name, value) in &kernel.const_params {
        let id = if builder.access.flags().iter().any(|flag| flag == name) {
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
            tensor.element,
            &shape,
            &tensor.strides,
        );
        // The checks read the tensor by name: which parameter it came from,
        // its tile, and what each view axis is.
        builder.access.register_tensor(sile_hir::passes::access::TensorFacts {
            param: sile_hir::passes::access::buffer_param_position(kernel, &name),
            tile: tensor.tile.clone(),
            view_axes: view_extents
                .iter()
                .map(|extent| match extent {
                    Some(v) => sile_hir::passes::access::AxisExtent::Static(*v),
                    None => sile_hir::passes::access::AxisExtent::Dynamic,
                })
                .collect(),
            name: name.clone(),
        });
        builder.tensors.push((
            name.clone(),
            Tensor {
                name,
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
            let is_int = matches!(tile_scalar(&ty), Some(IrScalar::I32));
            builder.values.push((name.clone(), value, Some(ty)));
            // What the binding proves — an interval, a term — is the checks'
            // business, and so is whether the value is an integer, and whether
            // it is a component of the block id.
            builder.access.record_binding(kernel, name, expr);
            if is_int {
                builder.access.bind_int(name);
            }
            if matches!(
                expr,
                sile_hir::Expr::Builtin {
                    op: sile_hir::BuiltinOp::ProgramId
                        | sile_hir::BuiltinOp::ProgramIdAxis,
                    ..
                }
            ) {
                builder.access.mark_block_id(name);
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
                        let param =
                            sile_hir::passes::access::buffer_param_position(kernel, &derived.name);
                        let view_axes = derived
                            .view_shape
                            .iter()
                            .map(|extent| match extent {
                                Extent::Literal(v) => {
                                    sile_hir::passes::access::AxisExtent::Static(*v)
                                }
                                Extent::Value(_) => sile_hir::passes::access::AxisExtent::Dynamic,
                            })
                            .collect();
                        builder.access.register_tensor(sile_hir::passes::access::TensorFacts {
                            name: name.clone(),
                            param,
                            tile: derived.tile.clone(),
                            view_axes,
                        });
                        builder.tensors.push((name.clone(), derived));
                    }
                }
            }
        }
        Stmt::Assign { name, expr } => {
            let (value, ty) = emit_expr(builder, expr, kernel);
            let is_int = matches!(tile_scalar(&ty), Some(IrScalar::I32));
            // The innermost binding of the name is the one being assigned.
            match builder.values.iter_mut().rev().find(|(bound, _, _)| bound == name) {
                Some(entry) => {
                    entry.1 = value;
                    entry.2 = Some(ty);
                }
                None => builder.values.push((name.clone(), value, Some(ty))),
            }
            if is_int {
                builder.access.bind_int(name);
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
            // A loop over `num_tiles` walks exactly that axis's tiles, so what
            // it visits is inside the grid by construction.
            builder.access.assume_loop_over_tile_count(kernel, var, end);
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
            let arg_types = std::iter::once(Builder::i32_tile())
                .chain(inits.iter().map(|(_, ty)| ty.clone()))
                .collect::<Vec<_>>();
            let (region_id, body_block, block_args) =
                build_single_block_region(&mut builder.module, &arg_types);
            let var_id = block_args[0];
            let vars = &block_args[1..];
            // The bound is the number of iterations, which is the loop's own
            // end value less its start.
            let bound = match (
                builder.access.bounds_of(kernel, start),
                builder.access.bounds_of(kernel, end),
            ) {
                // The variable runs from zero to the last iteration, which is
                // one less than the trip count.
                (Some(start), Some(end)) => Some(end.end - start.start - 1),
                // A trip count the compiler cannot know prints `?`.
                _ => None,
            };
            builder.blocks.push(body_block);
            // The loop variable is assumed bounded as soon as the body opens;
            // the assumption does not spend an id of its own here.
            let assumed_id = builder.assume_bounded(var_id, bound);
            builder
                .values
                .push((var.clone(), assumed_id, Some(Builder::i32_tile())));
            // What the loop proves about its variable — the belief that it
            // walks one axis's tiles, the interval, the term — is the checks'
            // to keep.
            builder.access.assume_loop_over_tile_count(kernel, var, end);
            builder.access.enter_loop(kernel, var, bound);
            builder.loop_carried.push(carried.clone());
            for (name, (_, ty)) in carried.iter().zip(inits.iter()) {
                builder
                    .values
                    .push((name.clone(), vars[carried.iter().position(|c| c == name).unwrap()], Some(ty.clone())));
            }
            // The region's body prints two spaces deeper than the loop itself.
            for stmt in body {
                emit_stmt(builder, stmt, kernel);
            }
            builder.access.exit_loop(var);
            builder.loop_carried.pop();
            let final_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            // The region's `continue` terminator carries the final values.
            let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
                .operands(final_values.iter().copied())
                .build(&mut builder.module);
            append_op(&mut builder.module, body_block, cont);
            builder.blocks.pop();
            // One result per carried value, allocated after the body: a
            // region's results follow every op they dominate. A loop with
            // nothing to carry prints no results — but the loop itself always
            // prints.
            let outer_block = builder.block();
            let (loop_op, result_values) = OpBuilder::new(Opcode::For, Location::Unknown)
                .operand(start_value)
                .operand(end_value)
                .operand(step_value)
                .operands(inits.iter().map(|(value, _)| *value))
                .results(inits.iter().map(|(_, ty)| ty.clone()))
                .region(region_id)
                .build(&mut builder.module);
            append_op(&mut builder.module, outer_block, loop_op);
            // After the loop, the carried name denotes the loop's result.
            for (name, result_id) in carried.iter().zip(result_values.iter()) {
                if let Some(entry) = builder.values.iter_mut().rev().find(|(b, _, _)| b == name) {
                    entry.1 = *result_id;
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
            // The values the names held before the branch ran: that is what an
            // `if` without an `else` yields on the untaken side.
            let pre_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            // The then region: its body, then the yield that hands the values on.
            let (then_region, then_block, _) =
                build_single_block_region(&mut builder.module, &[]);
            builder.blocks.push(then_block);
            for stmt in then_body {
                emit_stmt(builder, stmt, kernel);
            }
            let then_values = carried
                .iter()
                .map(|name| value_of(builder, name))
                .collect::<Vec<_>>();
            let (then_yield, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operands(then_values.iter().copied())
                .build(&mut builder.module);
            append_op(&mut builder.module, then_block, then_yield);
            builder.blocks.pop();
            // A branch that changes a value carries the old one on the other
            // side, so cutile prints an `else` even when the kernel has none.
            let changes = !assigned.iter().all(|name| !carried.contains(name));
            let has_else = !else_body.is_empty() || changes;
            let else_region = if has_else {
                let (else_region, else_block, _) =
                    build_single_block_region(&mut builder.module, &[]);
                builder.blocks.push(else_block);
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
                let (else_yield, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                    .operands(else_values.iter().copied())
                    .build(&mut builder.module);
                append_op(&mut builder.module, else_block, else_yield);
                builder.blocks.pop();
                Some(else_region)
            } else {
                None
            };
            // The region's results follow every op both branches print.
            let mut op = OpBuilder::new(Opcode::If, Location::Unknown)
                .operand(cond_value)
                .results(types.iter().cloned())
                .region(then_region);
            if let Some(else_region) = else_region {
                op = op.region(else_region);
            }
            let (if_op, result_ids) = op.build(&mut builder.module);
            let block = builder.block();
            append_op(&mut builder.module, block, if_op);
            // After the region the carried names denote its results.
            for (name, result_id) in carried.iter().zip(result_ids.iter()) {
                if let Some(entry) = builder.values.iter_mut().rev().find(|(b, _, _)| b == name) {
                    entry.1 = *result_id;
                }
            }
        }
        // `cuda_tile_print!("fmt {}", a)`: the format string keeps its text and
        // gains one `%` plus the operand's element letter per placeholder.
        sile_hir::Stmt::Print { format, args } => {
            let mut text = String::new();
            let mut operands = Vec::new();
            let mut rest = format.as_str();
            for arg in args {
                let (value, ty) = emit_expr(builder, arg, kernel);
                operands.push(value);
                let letter = match tile_scalar(&ty) {
                    Some(IrScalar::F32) | Some(IrScalar::F16) => "f",
                    _ => "i",
                };
                // The placeholder becomes `%` and the letter; everything before
                // it is the literal text the kernel wrote.
                match rest.split_once("{}") {
                    Some((before, after)) => {
                        text.push_str(before);
                        text.push('%');
                        text.push_str(letter);
                        rest = after;
                    }
                    None => panic!("more print arguments than `{{}}` placeholders: {format}"),
                }
            }
            text.push_str(rest);
            // A print yields a token. The id is real even though nothing reads
            // it, so it takes one like any other op.
            builder.append1(
                OpBuilder::new(Opcode::Print, Location::Unknown)
                    .attr("str", Attribute::String(text))
                    .operands(operands)
                    .result(IrType::Token),
            );
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

fn emitted_type_of(builder: &Builder, name: &str) -> Option<IrType> {
    builder
        .values
        .iter()
        .rev()
        .find(|(bound, _, _)| bound == name)
        .and_then(|(_, _, ty)| ty.clone())
}

fn value_of(builder: &Builder, name: &str) -> IrValue {
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
) -> (IrValue, IrType) {
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
                let printed = if builder.access.flags().iter().any(|flag| flag == name) {
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
                        return (value, tile_type(*scalar, &[]));
                    }
                    (value, result_type_of(builder, expr, kernel))
                }
            }
        }
        Expr::Builtin { op, args } => emit_builtin(builder, *op, args, kernel),
        // A float literal in value position is a rank-0 `constant`.
        Expr::ScalarF32(value) => (builder.f32_const(*value, &[]), tile_type(IrScalar::F32, &[])),
        // An integer literal in value position is a `constant`.
        Expr::ScalarI64(value) => (builder.i32_const(*value), Builder::i32_tile()),
        // A shape is metadata: binding one costs nothing and emits nothing.
        Expr::Shape(shape) => {
            builder.shapes.push((String::new(), shape.clone()));
            (IrValue(u32::MAX), IrType::Token)
        }
        other => panic!("the cutile emitter has no arm for {other:?}"),
    }
}

fn emit_builtin(
    builder: &mut Builder,
    op: sile_hir::BuiltinOp,
    args: &[sile_hir::Expr],
    kernel: &Kernel,
) -> (IrValue, IrType) {
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
            let source_is_tile = !tile_shape(&source_ty).is_empty();
            if source_is_tile {
                let source_extents = tile_shape(&source_ty);
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
                let element = match tile_scalar(&source_ty) {
                    Some(s) => s,
                    None => IrScalar::F32,
                };
                let result_ty = tile_type(
                    element,
                    &shape_extents(&target, &kernel.const_params),
                );
                let result = builder.append1(
                    OpBuilder::new(Opcode::Broadcast, Location::Unknown)
                        .operand(scalar)
                        .result(result_ty.clone()),
                );
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
            let element = match tile_scalar(&source_ty) {
                Some(s) => s,
                None => IrScalar::F32,
            };
            let scalar_ty = tile_type(element, &[]);
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
            // `1.5` an `f32` one, `1` an `i32` one. The literal is matched
            // directly — routing a float through an integer round trip would
            // drop its bit pattern.
            let literal = match &args[0] {
                Expr::ScalarF32(_) | Expr::ScalarBool(_) | Expr::ScalarI64(_) => &args[0],
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
            let count = extents.iter().product::<i64>().max(1) as usize;
            let (result, element) = match literal {
                Expr::ScalarF32(v) => (builder.f32_const(*v, &extents), IrScalar::F32),
                Expr::ScalarBool(v) => (
                    builder.const_splat(IrScalar::I1, &extents, vec![*v as u8; count]),
                    IrScalar::I1,
                ),
                Expr::ScalarI64(v) => (
                    builder.const_splat(
                        IrScalar::I32,
                        &extents,
                        (*v as i32).to_le_bytes().repeat(count),
                    ),
                    IrScalar::I32,
                ),
                _ => unreachable!(),
            };
            (result, tile_type(element, &extents))
        }
        BuiltinOp::Add | BuiltinOp::Sub | BuiltinOp::Mul
            if builder.access.is_integer_expr(kernel, args) =>
        {
            match fold_integer(builder, op, args, kernel) {
                Some(folded) => folded,
                None => binary(builder, args, integer_opcode(op), kernel),
            }
        }
        BuiltinOp::Add => binary(builder, args, Opcode::AddF, kernel),
        // `a & b` over masks: one op, and the printer states only the result
        // type, never the operand types.
        BuiltinOp::And | BuiltinOp::Or | BuiltinOp::Xor => {
            let opcode = match op {
                BuiltinOp::And => Opcode::AndI,
                BuiltinOp::Or => Opcode::OrI,
                _ => Opcode::XOrI,
            };
            let (lhs, lhs_ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let result = builder.same_binary(opcode, lhs, rhs, lhs_ty.clone());
            (result, lhs_ty)
        }
        // `max_tile(a, b)` / `min_tile(a, b)`: `maxf` states its rounding mode
        // and prints no operand types; `minf` is not used by these kernels.
        BuiltinOp::Max | BuiltinOp::Min => {
            let integer = builder.access.is_integer_expr(kernel, args);
            let opcode = match (matches!(op, BuiltinOp::Max), integer) {
                (true, false) => Opcode::MaxF,
                (true, true) => Opcode::MaxI,
                (false, false) => Opcode::MinF,
                (false, true) => Opcode::MinI,
            };
            let (lhs, lhs_ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            // An integer minimum states its signedness; the float ones state
            // a rounding mode.
            let result = if integer {
                builder.append1(
                    OpBuilder::new(opcode, Location::Unknown)
                        .operand(lhs)
                        .operand(rhs)
                        .attr("signedness", Attribute::i32(1))
                        .result(lhs_ty.clone()),
                )
            } else {
                builder.append1(
                    OpBuilder::new(opcode, Location::Unknown)
                        .operand(lhs)
                        .operand(rhs)
                        .attr("rounding_mode", Attribute::i32(0))
                        .result(lhs_ty.clone()),
                )
            };
            (result, lhs_ty)
        }
        BuiltinOp::Mul => binary(builder, args, Opcode::MulF, kernel),
        BuiltinOp::Sub => binary(builder, args, Opcode::SubF, kernel),
        // An integer operation whose interval is a single value never prints:
        // cutile allocates the id, proves the interval exact, drops the
        // operation and emits a `constant` instead. The dropped id stays a hole.
        // Floating point never folds, so only an integer expression qualifies.
        BuiltinOp::Div | BuiltinOp::Rem | BuiltinOp::CeilDiv
            if builder.access.is_integer_expr(kernel, args) =>
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
        BuiltinOp::Div => binary(builder, args, Opcode::DivF, kernel),
        BuiltinOp::TrueDiv => {
            let (lhs, ty) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let result = builder.append1(
                OpBuilder::new(Opcode::DivF, Location::Unknown)
                    .operand(lhs)
                    .operand(rhs)
                    .attr("rounding_mode", Attribute::i32(4))
                    .attr("flush_to_zero", Attribute::Bool(true))
                    .result(ty.clone()),
            );
            (result, ty)
        }
        BuiltinOp::Rem => binary(builder, args, Opcode::RemF, kernel),
        // `ceil_div(a, b)` prints as `divi ... rounding positive_inf`. A const
        // generic operand reads the value the prologue named, not a fresh
        // `constant`.
        BuiltinOp::CeilDiv => {
            let operand = |builder: &mut Builder, expr: &sile_hir::Expr| -> IrValue {
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
            let result = builder.int_binary(Opcode::DivI, lhs, rhs, true, Some(3));
            (result, Builder::i32_tile())
        }
        // `gt_tile(a, b)`: one predicate, ordered, with its comparison spelled.
        BuiltinOp::Gt => compare(builder, args, "greater_than", kernel),
        BuiltinOp::Lt => compare(builder, args, "less_than", kernel),
        BuiltinOp::Ge => compare(builder, args, "greater_than_or_equal", kernel),
        BuiltinOp::Le => compare(builder, args, "less_than_or_equal", kernel),
        BuiltinOp::Eq => compare(builder, args, "equal", kernel),
        BuiltinOp::Ne => compare(builder, args, "not_equal", kernel),
        // `select(cond, a, b)`: the condition first, then the two candidates.
        BuiltinOp::Select => {
            let (cond, _cond_ty) = emit_expr(builder, &args[0], kernel);
            let (then, ty) = emit_expr(builder, &args[1], kernel);
            let (other, _) = emit_expr(builder, &args[2], kernel);
            let result = builder.append1(
                OpBuilder::new(Opcode::Select, Location::Unknown)
                    .operand(cond)
                    .operand(then)
                    .operand(other)
                    .result(ty.clone()),
            );
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
            let source_extents = tile_shape(&source_ty);
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
            let (source_extents, element) = match &ty {
                IrType::Tile(t) => (
                    t.shape.clone(),
                    match &t.element_type {
                        TileElementType::Scalar(s) => *s,
                        TileElementType::Pointer(_) => IrScalar::F32,
                    },
                ),
                _ => (Vec::new(), IrScalar::F32),
            };
            for extent in &source_extents {
                builder.i32_const(*extent);
            }
            let permutation: Vec<usize> = (0..source_extents.len()).rev().collect();
            let permuted = permutation
                .iter()
                .map(|axis| source_extents[*axis])
                .collect::<Vec<_>>();
            let result_ty = tile_type(element, &permuted);
            let result = builder.append1(
                OpBuilder::new(Opcode::Permute, Location::Unknown)
                    .operand(source)
                    .attr(
                        "permutation",
                        Attribute::Array(
                            permutation
                                .iter()
                                .map(|axis| Attribute::i32(*axis as i64))
                                .collect(),
                        ),
                    )
                    .result(result_ty.clone()),
            );
            (result, result_ty)
        }
        BuiltinOp::Unary(kind) => {
            let (source, ty) = emit_expr(builder, &args[0], kernel);
            let opcode = match kind {
                sile_hir::UnaryOp::Exp => Opcode::Exp,
                sile_hir::UnaryOp::Exp2 => Opcode::Exp2,
                sile_hir::UnaryOp::Log => Opcode::Log,
                sile_hir::UnaryOp::Log2 => Opcode::Log2,
                sile_hir::UnaryOp::Sqrt => Opcode::Sqrt,
                sile_hir::UnaryOp::Rsqrt => Opcode::Rsqrt,
                sile_hir::UnaryOp::Abs => Opcode::AbsF,
                sile_hir::UnaryOp::Neg => Opcode::NegF,
                other => panic!("the cutile emitter has no arm for Unary({other:?})"),
            };
            // `sqrt` states its rounding mode; the others name themselves.
            let mut op = OpBuilder::new(opcode, Location::Unknown).operand(source);
            if matches!(kind, sile_hir::UnaryOp::Sqrt) {
                op = op.attr("rounding_mode", Attribute::i32(2));
            }
            let result = builder.append1(op.result(ty.clone()));
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
            let id = builder.make_partition_view(
                tensor.view,
                &tile,
                tensor.element,
                &tensor.view_shape,
                &tensor.strides,
                padding_zero,
            );
            // A *mutable* partition offsets every access by the tile block it
            // was built in: cutile asks for the block id again and multiplies
            // the block's row by the tile's row extent.
            if matches!(op, BuiltinOp::PartitionMut) {
                // The block id here is not assumed: cutile reads it straight
                // into the offset arithmetic.
                let ids = builder.append(
                    OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown).results([
                        Builder::i32_tile(),
                        Builder::i32_tile(),
                        Builder::i32_tile(),
                    ]),
                );
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
                        builder.same_binary(
                            Opcode::MulI,
                            ids[axis],
                            ratio_value,
                            Builder::i32_tile(),
                        )
                    };
                    last = Some(offset);
                }
                builder.store_offset = last;
                // The store's row index is this block id, not the prologue's.
                builder.store_block = ids.first().copied();
            }
            let ty = partition_view_type(
                &tile,
                tensor.element,
                &tensor.view_shape,
                &tensor.strides,
                padding_zero,
            );
            (id, ty)
        }
        // Reading a one-element tile back as a scalar names the tile's rank as
        // a `constant` and then the same value; the tile itself is unchanged.
        // `tile_to_scalar(t)`: the tile's single value as a scalar. The op
        // prints nothing, but the value it yields is a rank-zero tile.
        BuiltinOp::TileToScalar => {
            let (value, ty) = emit_expr(builder, &args[0], kernel);
            let element = match tile_scalar(&ty) {
                Some(s) => s,
                None => IrScalar::F32,
            };
            (value, tile_type(element, &[]))
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
            let ty = tile_type(IrScalar::F32, &[]);
            let result = builder.append1(
                OpBuilder::new(Opcode::IToF, Location::Unknown)
                    .operand(source)
                    .attr("signedness", Attribute::i32(1))
                    .result(ty.clone()),
            );
            (result, ty)
        }
        // `mma(a, b, acc)`: one op, three tile operands, in that order.
        BuiltinOp::Mma => {
            let (lhs, _) = emit_expr(builder, &args[0], kernel);
            let (rhs, _) = emit_expr(builder, &args[1], kernel);
            let (acc, acc_ty) = emit_expr(builder, &args[2], kernel);
            let result = builder.append1(
                OpBuilder::new(Opcode::MmaF, Location::Unknown)
                    .operand(lhs)
                    .operand(rhs)
                    .operand(acc)
                    .result(acc_ty.clone()),
            );
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
            let ty = tile_type(IrScalar::I32, &extents);
            let result = builder.append1(
                OpBuilder::new(Opcode::Iota, Location::Unknown).result(ty.clone()),
            );
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
            // One result per axis: the index-space shape of the view, one
            // extent per rank.
            let rank = tensor.tile.len();
            let ids = builder.append(
                OpBuilder::new(Opcode::GetIndexSpaceShape, Location::Unknown)
                    .operand(tensor.view)
                    .results((0..rank).map(|_| Builder::i32_tile())),
            );
            (ids[axis], Builder::i32_tile())
        }
        // `get_tile_block_id()`: three values, each assumed bounded. The kernel
        // `get_tile_block_id().k`: the tuple is read once and every later read
        // of any component reuses those three values.
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
                None => panic!("program id component {index} does not exist"),
            }
        }
        // `program_id(axis)`: CuTile writes this helper as a plain function
        // with an `if` chain over the axis, so each call reads the triple
        // again and tests the axis. The extra ids are part of the text.
        BuiltinOp::ProgramIdAxis => {
            let index = match args.first() {
                Some(Expr::ScalarI64(index)) => *index,
                _ => 0,
            };
            (builder.program_id_checked(index), Builder::i32_tile())
        }
        // `num_programs(axis)`: the grid extent along one axis. cutile emits
        // the whole triple, checks the axis against 0/1/2, and keeps the
        // selected component.
        BuiltinOp::NumPrograms => {
            let index = match args.first() {
                Some(Expr::ScalarI64(index)) => *index,
                _ => 0,
            };
            (builder.num_programs(index), Builder::i32_tile())
        }
        // `fma(a, b, c)`: one op, three operands, in that order.
        BuiltinOp::Fma => {
            let (a, ty) = emit_expr(builder, &args[0], kernel);
            let (b, _) = emit_expr(builder, &args[1], kernel);
            let (c, _) = emit_expr(builder, &args[2], kernel);
            let result = builder.append1(
                OpBuilder::new(Opcode::Fma, Location::Unknown)
                    .operand(a)
                    .operand(b)
                    .operand(c)
                    .result(ty.clone()),
            );
            (result, ty)
        }
        BuiltinOp::ReduceMax => reduce(builder, args, Opcode::MaxF, -f64::INFINITY, kernel),
        BuiltinOp::ReduceSum => reduce(builder, args, Opcode::AddF, 0.0, kernel),
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
    combine: Opcode,
    identity: f64,
    kernel: &Kernel,
) -> (IrValue, IrType) {
    let (source, source_ty) = emit_expr(builder, &args[0], kernel);
    let axis = match args.get(1) {
        Some(sile_hir::Expr::ScalarI64(axis)) => *axis,
        _ => 0,
    };
    let shape = tile_shape(&source_ty);
    let element = match tile_scalar(&source_ty) {
        Some(s) => s,
        None => IrScalar::F32,
    };
    let reduced = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis as usize)
        .map(|(_, extent)| *extent)
        .collect::<Vec<_>>();
    let lhs = tile_type(element, &[]);
    // The region's two parameters are allocated first; the combining op and the
    // reduce's own result follow them.
    let (region_id, body_block, block_args) = build_single_block_region(
        &mut builder.module,
        &[lhs.clone(), lhs.clone()],
    );
    let arg_a = block_args[0];
    let arg_b = block_args[1];
    builder.blocks.push(body_block);
    // `maxf` states its rounding mode; `addf` does not.
    let mut op = OpBuilder::new(combine, Location::Unknown)
        .operand(arg_a)
        .operand(arg_b);
    if combine == Opcode::MaxF {
        op = op.attr("rounding_mode", Attribute::i32(0));
    }
    let (body_op, body_values) = op.result(lhs.clone()).build(&mut builder.module);
    append_op(&mut builder.module, body_block, body_op);
    let (yield_op, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
        .operand(body_values[0])
        .build(&mut builder.module);
    append_op(&mut builder.module, body_block, yield_op);
    builder.blocks.pop();
    let result_ty = tile_type(element, &reduced);
    let block = builder.block();
    let (reduce_op, results) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
        .operand(source)
        .attr("dim", Attribute::i32(axis))
        .attr(
            "identities",
            Attribute::Array(vec![if combine == Opcode::MaxF {
                Attribute::Float(identity, IrType::Scalar(IrScalar::F32))
            } else {
                Attribute::Integer(identity as i64, IrType::Scalar(IrScalar::I32))
            }]),
        )
        .result(result_ty.clone())
        .region(region_id)
        .build(&mut builder.module);
    append_op(&mut builder.module, block, reduce_op);
    (results[0], result_ty)
}

/// The number of a comparison relation, as the IR spells the predicate.
fn relation_number(relation: &str) -> i64 {
    match relation {
        "equal" => 0,
        "not_equal" => 1,
        "less_than" => 2,
        "less_than_or_equal" => 3,
        "greater_than" => 4,
        "greater_than_or_equal" => 5,
        other => panic!("no comparison predicate named `{other}`"),
    }
}

/// `cmpf <relation> ordered a, b : tile<Nxf32> -> tile<Nxi1>`
fn compare(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    relation: &'static str,
    kernel: &Kernel,
) -> (IrValue, IrType) {
    let (lhs, ty) = emit_expr(builder, &args[0], kernel);
    let (rhs, _) = emit_expr(builder, &args[1], kernel);
    let shape = tile_shape(&ty);
    // An integer comparison states its signedness and drops `ordered`; the
    // float one keeps `ordered` and has no signedness to state.
    let integer = matches!(
        tile_scalar(&ty),
        Some(IrScalar::I32) | Some(IrScalar::I64)
    );
    let pred = relation_number(relation);
    let result_ty = tile_type(IrScalar::I1, &shape);
    let result = if integer {
        builder.append1(
            OpBuilder::new(Opcode::CmpI, Location::Unknown)
                .attr("comparison_predicate", Attribute::i32(pred))
                .attr("signedness", Attribute::i32(1))
                .operand(lhs)
                .operand(rhs)
                .result(result_ty.clone()),
        )
    } else {
        builder.append1(
            OpBuilder::new(Opcode::CmpF, Location::Unknown)
                .attr("comparison_predicate", Attribute::i32(pred))
                .attr("comparison_ordering", Attribute::i32(1))
                .operand(lhs)
                .operand(rhs)
                .result(result_ty.clone()),
        )
    };
    (result, result_ty)
}

/// The attribute an integer opcode prints after its operands.
///
/// Divide, remainder and minimum state their signedness; add and multiply do
/// not. A ceiling divide additionally states its rounding direction.
#[derive(Clone, Copy)]
enum IntSuffix {
    None,
    Signed,
    SignedPositiveInf,
}

fn integer_suffix(op: sile_hir::BuiltinOp) -> IntSuffix {
    match op {
        sile_hir::BuiltinOp::Rem | sile_hir::BuiltinOp::Div => IntSuffix::Signed,
        sile_hir::BuiltinOp::CeilDiv => IntSuffix::SignedPositiveInf,
        _ => IntSuffix::None,
    }
}

fn integer_opcode(op: sile_hir::BuiltinOp) -> Opcode {
    match op {
        sile_hir::BuiltinOp::Add => Opcode::AddI,
        sile_hir::BuiltinOp::Sub => Opcode::SubI,
        sile_hir::BuiltinOp::Mul => Opcode::MulI,
        sile_hir::BuiltinOp::Rem => Opcode::RemI,
        sile_hir::BuiltinOp::CeilDiv | sile_hir::BuiltinOp::Div => Opcode::DivI,
        _ => Opcode::AddI,
    }
}

/// An integer operation that folds: the checks pass the value it folds to, and
/// the emitter allocates the id, drops it, and prints a `constant` in its
/// place.
fn fold_integer(
    builder: &mut Builder,
    op: sile_hir::BuiltinOp,
    args: &[sile_hir::Expr],
    kernel: &Kernel,
) -> Option<(IrValue, IrType)> {
    let value = builder.access.fold_integer_value(op, args, kernel)?;
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
    let value = builder.i32_const(value);
    Some((value, Builder::i32_tile()))
}

fn binary(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    opcode: Opcode,
    kernel: &Kernel,
) -> (IrValue, IrType) {
    binary_suffixed(builder, args, opcode, IntSuffix::None, kernel)
}

/// The same, with the trailing attribute an opcode states after its operands.
fn binary_suffixed(
    builder: &mut Builder,
    args: &[sile_hir::Expr],
    opcode: Opcode,
    suffix: IntSuffix,
    kernel: &Kernel,
) -> (IrValue, IrType) {
    // A const generic names the value the prologue printed once, so an operand
    // that is one reads that id instead of spending a new `constant`.
    let operand = |builder: &mut Builder, expr: &sile_hir::Expr| -> (IrValue, IrType) {
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
        Opcode::AddF => Some(Opcode::AddI),
        Opcode::SubF => Some(Opcode::SubI),
        Opcode::MulF => Some(Opcode::MulI),
        _ => None,
    };
    let opcode = match (integer_twin, tile_scalar(&ty)) {
        (Some(twin), Some(IrScalar::I32)) => twin,
        _ => opcode,
    };
    let mut op = OpBuilder::new(opcode, Location::Unknown)
        .operand(lhs)
        .operand(rhs);
    match (opcode, suffix) {
        (Opcode::DivI, IntSuffix::SignedPositiveInf) => {
            op = op
                .attr("signedness", Attribute::i32(1))
                .attr("rounding", Attribute::i32(3));
        }
        (Opcode::DivI | Opcode::RemI | Opcode::MinI | Opcode::MaxI, IntSuffix::Signed) => {
            op = op.attr("signedness", Attribute::i32(1));
        }
        _ => {}
    }
    let result = builder.append1(op.result(ty.clone()));
    (result, ty)
}

/// The type of an expression whose value the emitter already has.
fn result_type_of(builder: &Builder, expr: &sile_hir::Expr, kernel: &Kernel) -> IrType {
    use sile_hir::Expr;
    match expr {
        Expr::Var(name) => match builder.locals.get(name) {
            Some(Type::Tile { elem, shape }) => tile_of(*elem, shape, &kernel.const_params),
            Some(Type::Tensor { elem, shape }) => tile_of(*elem, shape, &kernel.const_params),
            _ => IrType::Token,
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
fn load_like(builder: &mut Builder, args: &[sile_hir::Expr], kernel: &Kernel) -> (IrValue, IrType) {
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
fn load(builder: &mut Builder, args: &[sile_hir::Expr], kernel: &Kernel) -> (IrValue, IrType) {
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
) -> (IrValue, IrType) {
    read_tile_indexed(builder, source, tile, with_source, None, kernel)
}

/// The indices the kernel named, as a shape expression, when it named any.
/// A tensor's declared extents, with a dynamic one standing for itself.
fn tile_shape_of_extents(extents: &[Option<i64>]) -> Vec<i64> {
    extents.iter().map(|e| e.unwrap_or(-1)).collect()
}

fn read_tile_indexed(
    builder: &mut Builder,
    source: &str,
    tile: &[i64],
    with_source: bool,
    index_arg: Option<&sile_hir::Expr>,
    kernel: &Kernel,
) -> (IrValue, IrType) {
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
                // The checks pass decides each axis; the emitter only prints
                // the assert a `Device` decision names.
                let facts = tensor_facts(kernel, &tensor);
                for (axis, decision) in builder
                    .access
                    .decide_access(kernel, &facts, tile, index_arg)
                    .into_iter()
                    .enumerate()
                {
                    let AxisDecision::Device { proven_non_negative } = decision else {
                        continue;
                    };
                    // `ceil(extent / tile)`, with the extent a run-time value.
                    // The tile extent is named again here, inside the check.
                    let extent = match tensor.view_shape.get(axis) {
                        Some(Extent::Value(value)) => *value,
                        _ => unreachable!("a device check names a dynamic extent"),
                    };
                    let tile_value = builder.i32_const(tile[axis]);
                    let quotient = builder.int_binary(Opcode::DivI, extent, tile_value, true, None);
                    let remainder = builder.int_binary(Opcode::RemI, extent, tile_value, true, None);
                    let one = builder.i32_const(1);
                    let nonempty = builder.int_binary(Opcode::MinI, remainder, one, true, None);
                    let count = builder.int_binary(Opcode::AddI, quotient, nonempty, false, None);
                    let index = ids[axis];
                    let in_range = builder.cmpi(2, index, count);
                    // The lower bound: an index cutile cannot prove
                    // non-negative is tested against zero as well, and the two
                    // predicates are combined.
                    let guard = if proven_non_negative {
                        in_range
                    } else {
                        let zero = builder.i32_const(0);
                        let non_negative = builder.cmpi(5, index, zero);
                        builder.bool_binary(Opcode::AndI, non_negative, in_range)
                    };
                    builder.assert_op(
                        guard,
                        &format!(
                            "partition access out of bounds: dim {axis}, block index >= ceil(?/{}){suffix}",
                            tile[axis],
                            suffix = if proven_non_negative { "" } else { " or index < 0" },
                        ),
                    );
                }
                let view_ty = partition_view_type(
                    tile,
                    tensor.element,
                    &tensor.view_shape,
                    &tensor.strides,
                    true,
                );
                return builder.load_view_tko(view, &ids[..tile.len()], tensor.token, tile, tensor.element, view_ty);
            }
        }
        let indices = match builder.program_id.clone() {
            Some(ids) => ids,
            None => builder.block_id(),
        };
        let view_ty = partition_view_type(
            tile,
            tensor.element,
            &tensor.view_shape,
            &tensor.strides,
            true,
        );
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
    let view = builder.make_partition_view(
        tensor.view,
        tile,
        tensor.element,
        &tensor.view_shape,
        &tensor.strides,
        padding_zero,
    );
    let view_ty = partition_view_type(
        tile,
        tensor.element,
        &tensor.view_shape,
        &tensor.strides,
        padding_zero,
    );
    builder.load_view_tko(view, &indices[..tile.len()], tensor.token, tile, tensor.element, view_ty)
}

fn store_tile(
    builder: &mut Builder,
    target: &str,
    value: IrValue,
    value_ty: IrType,
    store_index: Vec<IrValue>,
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
                indices[0] = builder.same_binary(
                    Opcode::AddI,
                    block_row,
                    row,
                    Builder::i32_tile(),
                );
            }
        }
        if let Some(offset) = builder.store_offset {
            if indices.len() >= 2 {
                let column = indices[1];
                let sum = builder.same_binary(
                    Opcode::AddI,
                    offset,
                    column,
                    Builder::i32_tile(),
                );
                indices = vec![indices[0], sum];
            }
        }
        let view_ty = partition_view_type(
            &tile,
            tensor.element,
            &tensor.view_shape,
            &tensor.strides,
            false,
        );
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
    let view = builder.make_partition_view(
        tensor.view,
        &tile,
        tensor.element,
        &tensor.view_shape,
        &tensor.strides,
        false,
    );
    let view_ty = partition_view_type(
        &tile,
        tensor.element,
        &tensor.view_shape,
        &tensor.strides,
        false,
    );
    builder.store_view_tko(value, view, &indices[..tile.len()], tensor.token, value_ty, view_ty);
}
