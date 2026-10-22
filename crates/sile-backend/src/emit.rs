use std::collections::{HashMap, HashSet};
use std::cell::OnceCell;

use sile_core::{Error, Result};
use sile_lsir as lsir;
use sile_lsir::Function as LsirFunction;

use crate::cpu::codegen::generate as generate_lsir_c_source;
use crate::metal::codegen::generate as generate_lsir_metal_source;
use crate::ptx::codegen::generate as generate_lsir_ptx_source;
use crate::wgsl::codegen::generate as generate_lsir_wgsl_source;
use crate::{BackendArtifact, CodegenTarget};

pub fn run(
    lsir: &LsirFunction,
    target: CodegenTarget,
    policy: &sile_lsir::mapping::MappingPolicy,
) -> Result<BackendArtifact> {
    let artifact = match target {
        CodegenTarget::C => BackendArtifact::CSource(generate_lsir_c_source(lsir)?),
        CodegenTarget::Metal => {
            BackendArtifact::MetalSource(generate_lsir_metal_source(lsir)?)
        }
        CodegenTarget::Wgsl => {
            BackendArtifact::WgslSource(generate_lsir_wgsl_source(lsir, policy)?)
        }
        CodegenTarget::Ptx => BackendArtifact::PtxSource(generate_lsir_ptx_source(lsir)?),
    };
    Ok(artifact)
}

/// Sanitize a name into a valid C/C++/MSL identifier.
///
/// Passes synthesize names that contain characters the target languages
/// reject (for example `v25.c0` or `loop_col22.u`, produced by loop
/// unrolling). Only `[A-Za-z0-9_]` survives; a leading digit is prefixed.
/// An empty result falls back to a numeric name so it stays usable.
fn sanitize_identifier(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, 'v');
    }
    out
}

pub fn build_value_names(func: &lsir::Function) -> HashMap<lsir::ValueId, String> {
    let mut names = HashMap::new();
    for param in &func.params {
        names.insert(param.id, sanitize_identifier(&param.name));
    }
    for block in &func.blocks {
        for param in &block.params {
            names.insert(param.id, sanitize_identifier(&param.name));
        }
        for inst in &block.insts {
            if let (Some(id), Some(name)) = (inst.result, inst.result_name.as_ref()) {
                names.insert(id, sanitize_identifier(name));
            }
        }
    }
    names
}

pub fn value_name(value_names: &HashMap<lsir::ValueId, String>, id: lsir::ValueId) -> String {
    value_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", id.0))
}

pub fn format_operand(
    value_names: &HashMap<lsir::ValueId, String>,
    float_literal: fn(f64) -> String,
    operand: &lsir::Operand,
) -> String {
    match operand {
        lsir::Operand::Value(id) => value_name(value_names, *id),
        lsir::Operand::Const(lsir::Constant::Int(value)) => value.to_string(),
        lsir::Operand::Const(lsir::Constant::Float(value)) => float_literal(*value),
        lsir::Operand::Const(lsir::Constant::Bool(value)) => {
            if *value {
                "true".into()
            } else {
                "false".into()
            }
        }
    }
}

pub fn array_dims(ty: &lsir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let lsir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

pub fn bin_op_symbol(op: lsir::BinOp) -> &'static str {
    match op {
        lsir::BinOp::Add => "+",
        lsir::BinOp::Sub => "-",
        lsir::BinOp::Mul => "*",
        lsir::BinOp::Div => "/",
        lsir::BinOp::Rem => "%",
        lsir::BinOp::And => "&",
        lsir::BinOp::Or => "|",
        lsir::BinOp::Xor => "^",
        lsir::BinOp::Shl => "<<",
        lsir::BinOp::Shr => ">>",
    }
}

pub fn cmp_pred_symbol(pred: lsir::CmpPred) -> &'static str {
    match pred {
        lsir::CmpPred::Eq => "==",
        lsir::CmpPred::Ne => "!=",
        lsir::CmpPred::Slt | lsir::CmpPred::Olt => "<",
        lsir::CmpPred::Sle | lsir::CmpPred::Ole => "<=",
        lsir::CmpPred::Sgt | lsir::CmpPred::Ogt => ">",
        lsir::CmpPred::Sge | lsir::CmpPred::Oge => ">=",
    }
}

/// A float as a C-family literal: `2.0f`, `INFINITY`, `NAN`.
///
/// A finite value is a decimal literal, which every target reads the same way.
/// The three non-finite values are where the targets disagree, so a target
/// whose standard library has no `INFINITY` or `NAN` overrides
/// [`TargetDialect::float_literal`].
pub fn float_literal(value: f64) -> String {
    if value.is_nan() {
        return "NAN".into();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "(-INFINITY)".into()
        } else {
            "INFINITY".into()
        };
    }
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value}f")
    }
}

/// The synthetic shape parameter every backend reads.
pub const SHAPES_PARAM_NAME: &str = lsir::SHAPES_PARAM_NAME;

/// The synthetic scalar parameter, bound after the shape descriptor.
///
/// A scalar kernel argument travels as a load from here rather than as a
/// function parameter, because WGSL cannot pass a value into an entry point.
pub const SCALARS_PARAM_NAME: &str = lsir::SCALARS_PARAM_NAME;

/// True when the parameter is the shape descriptor, not a kernel buffer.
pub fn is_shapes_param(param: &lsir::Param) -> bool {
    param.name == SHAPES_PARAM_NAME
}

/// True when the parameter is the float scalar descriptor, not a kernel buffer.
pub fn is_scalars_param(param: &lsir::Param) -> bool {
    param.name == SCALARS_PARAM_NAME
}

/// The synthetic integer scalar parameter, bound after the float one.
pub const SCALARS_I64_PARAM_NAME: &str = lsir::SCALARS_I64_PARAM_NAME;

/// True when the parameter is the integer scalar descriptor.
pub fn is_scalars_i64_param(param: &lsir::Param) -> bool {
    param.name == SCALARS_I64_PARAM_NAME
}

/// True when the parameter is a descriptor rather than a kernel buffer.
///
/// The descriptors are the synthetic trailing parameters, and every place that
/// counts or picks kernel buffers has to skip all of them.
pub fn is_descriptor_param(param: &lsir::Param) -> bool {
    is_shapes_param(param) || is_scalars_param(param) || is_scalars_i64_param(param)
}

/// The output buffer of a kernel, with the tile it is stored through.
///
/// The tile shape comes from `ParamAbi::tile`, which Tile IR lowering already
/// knew when it emitted `store_ptr_tko`, so no backend re-derives it by scanning
/// the CFG for the last store.
pub struct OutputBuffer<'a> {
    pub param: &'a lsir::Param,
    pub rows: usize,
    pub cols: usize,
}

/// Last output parameter, as the backend signature declares it.
///
/// The output is the last buffer the kernel writes, and the lowering numbers
/// parameters in declaration order. The result is the same parameter whether
/// or not it carries tile ABI metadata: a kernel that only atomic-adds one
/// scalar per program has an output with no tile, and the launch grid still
/// has to come from somewhere.
pub fn find_output_param(func: &lsir::Function) -> Option<&lsir::Param> {
    func.params
        .iter()
        .rev()
        .find(|param| !is_descriptor_param(param))
}

/// Last non-descriptor parameter carrying tile ABI metadata.
pub fn find_output_buffer(func: &lsir::Function) -> Option<OutputBuffer<'_>> {
    func.params
        .iter()
        .rev()
        .find(|param| {
            !is_descriptor_param(param) && param.abi.as_ref().is_some_and(|abi| abi.tile.is_some())
        })
        .and_then(|param| {
            let abi = param.abi.as_ref()?;
            let (rows, cols) = abi.tile?;
            Some(OutputBuffer { param, rows, cols })
        })
}

/// The launch grid a backend that reads one flat index must dispatch.
///
/// A rank-2 output makes a kernel index its tiles by row and column. C and MSL
/// read both from separate launch axes, so a driver that dispatches the grid as
/// given leaves one of them unreachable when it has to be flat. WGSL reads the
/// launch index from one axis only, so the two tile axes have to arrive
/// multiplied together.
///
/// The rule lives here because the codegen and the runtime must agree: if the
/// emitter assumes a flattened grid, the runtime has to send one, and neither
/// side may decide alone.
pub fn flatten_grid(func: &lsir::Function, launch: &sile_core::LaunchConfig) -> [u32; 3] {
    let output_rank = find_output_buffer(func)
        .and_then(|output| output.param.abi.as_ref().map(|abi| abi.rank))
        .unwrap_or(1);
    match output_rank {
        // A rank-1 output puts one tile on each grid slot, so the launch index
        // is the tile index and the grid arrives as the caller wrote it.
        0 | 1 => launch.grid,
        // A rank-3 output addresses a plane of tiles, so all three axes arrive
        // multiplied together and the emitter divides them back out.
        3.. => [
            launch
                .grid
                .iter()
                .fold(1u32, |acc, dim| acc.saturating_mul(*dim)),
            1,
            1,
        ],
        // A rank-2 output addresses one tile with a row and a column. Those two
        // arrive multiplied together; the third axis is not a tile axis and
        // stays as it is.
        _ if launch.grid[1] <= 1 => launch.grid,
        _ => [
            launch.grid[0].saturating_mul(launch.grid[1]),
            1,
            launch.grid[2],
        ],
    }
}

/// The extent of each logical grid axis, as source expressions.
///
/// The logical axes are the output buffer's axes that the tile does not cover,
/// in declaration order. A rank-2 output addresses one tile with `(row, col)`,
/// so its axes are a row count and a column count. A rank-3 output addresses a
/// plane of tiles with `(plane, row, col)`, so a plane count comes first. A
/// rank-1 output has one axis.
///
/// The tile covers the last two axes, so an axis the tile does cover is the
/// buffer's extent divided by the tile's; an axis before those holds one tile
/// per element, and its extent is the buffer's extent unchanged. Both come from
/// the shape descriptor and the tile ABI, so the kernel needs no launch-time
/// value to recover them.
pub fn grid_axis_exprs(
    abi: &lsir::ParamAbi,
    rows: usize,
    cols: usize,
    shapes: &str,
) -> Vec<String> {
    let rank = abi.rank.max(1);
    let offset = abi.shape_offset;
    let mut axes: Vec<String> = (0..rank.saturating_sub(2))
        .map(|axis| format!("{shapes}[{}]", offset + axis))
        .collect();
    if rank >= 2 {
        axes.push(format!("{shapes}[{}] / {}", offset + rank - 2, rows.max(1)));
    }
    axes.push(format!("{shapes}[{}] / {}", offset + rank - 1, cols.max(1)));
    axes
}

/// The stride of logical grid axis `dim` in a flat launch index.
///
/// A flat launch index counts tiles with the last axis varying fastest, so the
/// stride of an axis is the product of the extents after it.
pub fn grid_axis_stride(axes: &[String], dim: usize) -> String {
    axes.iter()
        .skip(dim + 1)
        .fold("1".to_string(), |acc, extent| format!("{acc} * {extent}"))
}

/// How a target spells address-taking and dereferencing.
///
/// C and MSL have pointers, so a load is `*p` and a store is `*p = v`. WGSL has
/// no pointers, so the same IR has to read and write the element directly, and
/// the load of a Gep result is folded into its use.
#[derive(Clone, Copy)]
pub struct PointerSpelling {
    /// Whether the target can take an address, so `&base[i]` is an instruction.
    pub has_address_of: bool,
    /// Whether a load through a pointer dereferences it.
    pub load_deref: bool,
    /// How a flat subscript spells a computed index.
    ///
    /// A target whose array subscript is 32-bit cannot take the 64-bit index
    /// the lowering produces, so it converts at the subscript and leaves every
    /// other value at the width the IR gave it. A constant index never needs
    /// the conversion and is left alone.
    pub index_cast: fn(String) -> String,
}

pub const POINTER_LANGUAGE: PointerSpelling = PointerSpelling {
    has_address_of: true,
    load_deref: true,
    index_cast: std::convert::identity,
};

pub const NO_POINTERS: PointerSpelling = PointerSpelling {
    has_address_of: false,
    load_deref: false,
    index_cast: std::convert::identity,
};

/// How a target spells a three-way choice.
///
/// C and MSL write `cond ? on_true : on_false`. WGSL has no conditional
/// expression, so it calls `select(on_false, on_true, cond)` instead. The two
/// argument orders differ, which is why the spelling is a parameter and not a
/// format string.
#[derive(Clone, Copy, Debug)]
pub enum SelectSpelling {
    Ternary,
    Function {
        /// WGSL resolves a bare integer argument of a call to `i32` before
        /// overload resolution sees the context, so an `i64` result needs its
        /// integer constants wrapped: `select(i64(0), i64(1), cond)`.
        int_const: fn(i64) -> String,
    },
}

/// How a target spells an integer division or remainder.
#[derive(Clone, Copy, Debug)]
pub enum IntDivSpelling {
    /// The operator as the language defines it, for any width. C and MSL
    /// compile a 64-bit `/` and `%` natively.
    Operator,
    /// The division runs in `i32` and the result widens back to `i64`.
    ///
    /// naga wraps a 64-bit `/` or `%` in a helper whose `metal::select` passes
    /// an untyped literal next to a `long long`, which Metal rejects as an
    /// ambiguous call, so a 64-bit division cannot reach the Metal compiler.
    /// Index arithmetic — launch ids, tile coordinates, buffer offsets over
    /// the descriptor — stays far below 2^31, and the operands truncate to
    /// `i32` before the operator rather than after.
    I32Widened,
}

fn select_expr(
    spelling: SelectSpelling,
    cond: String,
    on_true: String,
    on_false: String,
) -> String {
    match spelling {
        SelectSpelling::Ternary => format!("({cond}) ? ({on_true}) : ({on_false})"),
        SelectSpelling::Function { .. } => format!("select({on_false}, {on_true}, {cond})"),
    }
}

/// One Gep in the function: where it starts, its indices, and each index's stride.
///
/// The strides are what a pointer-free target needs. C and MSL write
/// `a[i][j]` and let the type system apply the row width; WGSL flattens the
/// storage to `array<f32, 16>`, so the same access has to spell
/// `a[(i * 8) + j]` itself. An index into an outer dimension therefore carries
/// the product of the dimensions after it, and an index into the innermost
/// dimension carries one.
#[derive(Clone, Debug)]
pub struct GepLink {
    pub base: lsir::Operand,
    pub indices: Vec<lsir::Operand>,
    pub strides: Vec<usize>,
}

/// The type of every value in the function, including an alloca's pointer.
///
/// A Gep's stride comes from the pointee of the value it indexes, so the table
/// has to cover parameters, block parameters and instruction results.
fn value_types(func: &lsir::Function) -> HashMap<lsir::ValueId, lsir::Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            let Some(id) = inst.result else {
                continue;
            };
            types.insert(id, result_type(inst));
        }
    }
    types
}

/// The type of the value an instruction defines.
///
/// An `alloca`'s result is a pointer to the storage it owns; the instruction
/// records the allocation type instead, so the pointer is rebuilt here. Every
/// other instruction's result has the type it recorded.
pub fn result_type(inst: &lsir::Inst) -> lsir::Type {
    match &inst.op {
        lsir::InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => lsir::Type::ptr(addr_space.clone(), alloc_ty.clone()),
        _ => inst.ty.clone(),
    }
}

/// How much one index of a Gep advances, in leaf elements.
///
/// Index `i` steps over the dimensions after it, so a row index of `[2 x [8 x
/// f32]]` advances eight elements and the column index advances one.
fn index_strides(pointee: &lsir::Type, count: usize) -> Vec<usize> {
    let dims = array_dims(pointee);
    (0..count)
        .map(|i| dims.get(i + 1..).map_or(1, |rest| rest.iter().product()))
        .collect()
}

/// Collapse a pointer into the buffer it addresses and every index in its chain.
///
/// A Gep's base can be a Gep, and each link offsets the same buffer by a number
/// of elements, so one chain is one buffer and one summed index. Each index
/// keeps the stride of the link it came from, because a chain can cross array
/// dimensions of different widths.
pub fn walk_gep_chain<'g>(
    ptr: &lsir::Operand,
    gep_elements: &'g HashMap<lsir::ValueId, GepLink>,
) -> Option<(&'g lsir::Operand, Vec<(lsir::Operand, usize)>)> {
    let lsir::Operand::Value(id) = ptr else {
        return None;
    };
    let link = gep_elements.get(id)?;
    let (root, mut prefix) = match walk_gep_chain(&link.base, gep_elements) {
        Some((root, prefix)) => (root, prefix),
        None => (&link.base, Vec::new()),
    };
    prefix.extend(
        link.indices
            .iter()
            .cloned()
            .zip(link.strides.iter().copied()),
    );
    Some((root, prefix))
}

/// One index of a flattened access, scaled by the width of the dimension it
/// indexes into.
pub fn scaled_index(
    index: &lsir::Operand,
    stride: usize,
    format_operand: &dyn Fn(&lsir::Operand) -> String,
    index_cast: fn(String) -> String,
) -> String {
    // A constant index folds into the constant, so a tile access with literal
    // coordinates stays one literal in the emitted source.
    if let lsir::Operand::Const(lsir::Constant::Int(value)) = index {
        return format!("({})", value * stride as i64);
    }
    if stride == 1 {
        return index_cast(format!("({})", format_operand(index)));
    }
    index_cast(format!("({}) * ({stride})", format_operand(index)))
}

/// Spell the element a pointer names, for a target that has no pointers.
///
/// A dialect reaching for the element behind an intrinsic argument gets the
/// same rule the shared load and store use, instead of a second spelling of it.
pub fn element_operand(
    ptr: &lsir::Operand,
    gep_elements: &HashMap<lsir::ValueId, GepLink>,
    value_name: &dyn Fn(lsir::ValueId) -> String,
    format_operand: &dyn Fn(&lsir::Operand) -> String,
    index_cast: fn(String) -> String,
) -> String {
    match walk_gep_chain(ptr, gep_elements) {
        Some((root, indices)) => {
            let terms: Vec<String> = indices
                .iter()
                .map(|(idx, stride)| scaled_index(idx, *stride, format_operand, index_cast))
                .collect();
            match terms.as_slice() {
                [] => format!("{}[0]", format_operand(root)),
                [one] => format!("{}[{one}]", format_operand(root)),
                many => format!("{}[{}]", format_operand(root), many.join(" + ")),
            }
        }
        None => match ptr {
            lsir::Operand::Value(id) => format!("{}[0]", value_name(*id)),
            other => format_operand(other),
        },
    }
}

/// Every Gep in the function, with the stride of each index.
///
/// A target that has no pointers reads this table to spell the element a load or
/// a store addresses, and a dialect needs the same table for an intrinsic
/// argument. Both read the whole function, so both take it from one place
/// instead of one of them re-deriving it during emission.
pub fn collect_gep_elements(func: &lsir::Function) -> HashMap<lsir::ValueId, GepLink> {
    let types = value_types(func);
    let mut gep_elements = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let lsir::InstOp::Gep { base, indices } = &inst.op
                && let Some(result) = inst.result
            {
                let strides = match base {
                    lsir::Operand::Value(id) => match types.get(id) {
                        Some(lsir::Type::Ptr { pointee, .. }) => {
                            index_strides(pointee, indices.len())
                        }
                        _ => vec![1; indices.len()],
                    },
                    lsir::Operand::Const(_) => vec![1; indices.len()],
                };
                gep_elements.insert(
                    result,
                    GepLink {
                        base: (*base).clone(),
                        indices: indices.clone(),
                        strides,
                    },
                );
            }
        }
    }
    gep_elements
}

/// Lower one instruction with the dialect's pointer spelling.
///
/// `gep_elements` maps a Gep result to `(base, indices)`, which a language
/// without pointers needs in order to spell the element a load reads or a store
/// writes. It is empty for a pointer language, which never consults it.
pub fn lower_common_inst_line_with<FValue, FOperand>(
    inst: &lsir::Inst,
    value_name: FValue,
    format_operand: FOperand,
    pointers: PointerSpelling,
    select: SelectSpelling,
    int_div: IntDivSpelling,
    cast_expr: fn(lsir::CastOp, &lsir::Type, String) -> String,
    gep_elements: &mut HashMap<lsir::ValueId, GepLink>,
    folded_geps: &HashMap<lsir::ValueId, (lsir::Operand, Vec<lsir::Operand>)>,
) -> Option<String>
where
    FValue: Fn(lsir::ValueId) -> String,
    FOperand: Fn(&lsir::Operand) -> String,
{
    // The Gep table is collected once in `TargetEmitter::new` for a
    // pointer-free target, so a Gep is already there when its use arrives.

    let element = |ptr: &lsir::Operand| -> String {
        element_operand(
            ptr,
            gep_elements,
            &value_name,
            &format_operand,
            pointers.index_cast,
        )
    };
    // A pointer target reads straight through the geps a single load or store
    // uses, so the access spells `base[i]` and the gep itself leaves no text.
    let addressed = |ptr: &lsir::Operand| -> Option<String> {
        match ptr {
            lsir::Operand::Value(id) => folded_geps.get(id).map(|(base, indices)| {
                let suffix = indices
                    .iter()
                    .map(|idx| format!("[{}]", format_operand(idx)))
                    .collect::<Vec<_>>()
                    .join("");
                format!("{}{}", format_operand(base), suffix)
            }),
            _ => None,
        }
    };

    match &inst.op {
        lsir::InstOp::Bin { op, lhs, rhs } => inst.result.map(|id| {
            match (int_div, &inst.ty, op) {
                (
                    IntDivSpelling::I32Widened,
                    lsir::Type::I64,
                    lsir::BinOp::Div | lsir::BinOp::Rem,
                ) => format!(
                    "{} = i64((i32({}) {} i32({})));",
                    value_name(id),
                    format_operand(lhs),
                    bin_op_symbol(*op),
                    format_operand(rhs)
                ),
                _ => format!(
                    "{} = {} {} {};",
                    value_name(id),
                    format_operand(lhs),
                    bin_op_symbol(*op),
                    format_operand(rhs)
                ),
            }
        }),
        lsir::InstOp::Cmp { pred, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                cmp_pred_symbol(*pred),
                format_operand(rhs)
            )
        }),
        lsir::InstOp::Select {
            cond,
            on_true,
            on_false,
        } => inst.result.map(|id| {
            let arg = |op: &lsir::Operand| -> String {
                match (select, &inst.ty, op) {
                    (
                        SelectSpelling::Function { int_const },
                        lsir::Type::I64,
                        lsir::Operand::Const(lsir::Constant::Int(value)),
                    ) => int_const(*value),
                    _ => format_operand(op),
                }
            };
            format!(
                "{} = {};",
                value_name(id),
                select_expr(select, arg(cond), arg(on_true), arg(on_false))
            )
        }),
        lsir::InstOp::Load { ptr } => inst.result.map(|id| {
            if let Some(address) = addressed(ptr) {
                format!("{} = {};", value_name(id), address)
            } else if pointers.load_deref {
                format!("{} = *({});", value_name(id), format_operand(ptr))
            } else {
                // No pointer to dereference, so the load reads the element.
                format!("{} = {};", value_name(id), element(ptr))
            }
        }),
        lsir::InstOp::Store { ptr, value } => {
            if let Some(address) = addressed(ptr) {
                Some(format!("{} = {};", address, format_operand(value)))
            } else if pointers.load_deref {
                Some(format!("*({}) = {};", format_operand(ptr), format_operand(value)))
            } else {
                Some(format!("{} = {};", element(ptr), format_operand(value)))
            }
        }
        // A conversion is a value the target may have to spell: one language
        // converts where the value is assigned, and another names the type it
        // converts to.
        lsir::InstOp::Cast { op, value, to } => inst.result.map(|id| {
            format!(
                "{} = {};",
                value_name(id),
                cast_expr(*op, to, format_operand(value))
            )
        }),
        // A language with addresses materializes the pointer — unless its only
        // use folded into the access it serves — and one without already
        // recorded the element above and emits nothing.
        lsir::InstOp::Gep { base, indices } if pointers.has_address_of => {
            let folded_away = inst
                .result
                .is_some_and(|id| folded_geps.contains_key(&id));
            if folded_away {
                None
            } else {
                inst.result.map(|id| {
                    let index_suffix = indices
                        .iter()
                        .map(|idx| format!("[{}]", format_operand(idx)))
                        .collect::<Vec<_>>()
                        .join("");
                    format!(
                        "{} = &({}{});",
                        value_name(id),
                        format_operand(base),
                        index_suffix
                    )
                })
            }
        }
        _ => None,
    }
}

pub trait TextCodegen: Sized {
    fn emit_prelude(&mut self);
    fn emit_signature(&mut self);
    fn emit_body(&mut self) -> Result<()>;
    fn finish(self) -> String;
}

pub fn generate_text<E: TextCodegen>(mut emitter: E) -> Result<String> {
    emitter.emit_prelude();
    emitter.emit_signature();
    emitter.emit_body()?;
    Ok(emitter.finish())
}

/// How a target spells one target-language concept.
///
/// Every backend is a dialect over one shared traversal, so adding a target
/// means adding a type name table, an intrinsic spelling and a signature
/// preamble. The traversal, value naming, block ordering and structured-CFG
/// emission live in [`TargetEmitter`] and are written once.
pub trait TargetDialect {
    /// Declaration text for a value of this type, `None` when the target wants
    /// no declaration (the value is read from a built-in).
    /// Declaration text for a value of this type.
    ///
    /// `is_alloca` separates an `alloca`, which owns storage the target must
    /// materialize, from a `gep` of a driver buffer, which borrows storage and
    /// must stay a plain pointer. The two have the same LSIR type, so the
    /// flag is what tells them apart.
    fn var_decl(&self, ty: &lsir::Type, name: &str, is_alloca: bool) -> Option<String>;

    /// Parameter text for one kernel parameter.
    fn param_decl(&self, ty: &lsir::Type, name: &str) -> String;

    /// Text of the function head, up to and including the opening brace.
    ///
    /// Each entry of `params` is `(declaration, name)`, so the dialect decides
    /// whether the two are joined by a space, as in C, or an attribute, as in
    /// MSL.
    fn signature(&self, name: &str, params: &[(String, String)]) -> String;

    /// Lines emitted before the kernel body, after the declarations. The
    /// function is what tells a helper apart from dead weight: a target spells
    /// runtime support only for the intrinsics the body names.
    fn prelude_lines(&self, func: &lsir::Function) -> Vec<String>;

    /// Assignment of an intrinsic result, or a statement when it has none.
    fn intrinsic_stmt(
        &self,
        result: Option<lsir::ValueId>,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
        intrinsic: &lsir::Intrinsic,
        args: &[lsir::Operand],
    ) -> sile_core::Result<Vec<String>>;

    /// A statement per target for operations the shared traversal does not know.
    fn extra_inst_stmt(
        &mut self,
        inst: &lsir::Inst,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
    ) -> sile_core::Result<Vec<String>>;

    /// How this target spells address-taking and dereferencing.
    fn pointers(&self) -> PointerSpelling {
        POINTER_LANGUAGE
    }

    /// How this target spells a three-way choice.
    fn select(&self) -> SelectSpelling {
        SelectSpelling::Ternary
    }

    /// How this target spells an integer division or remainder.
    fn int_div(&self) -> IntDivSpelling {
        IntDivSpelling::Operator
    }

    /// How this target spells a conversion from one scalar type to another.
    ///
    /// An associated function, not a method, so the shared instruction printer
    /// can call it without borrowing the dialect. A language that converts
    /// where the value is assigned writes the value alone; one that has no
    /// implicit conversion names the type it converts to.
    fn cast_expr(_op: lsir::CastOp, _to: &lsir::Type, value: String) -> String
    where
        Self: Sized,
    {
        value
    }

    /// Name this target uses for the shape descriptor.
    ///
    /// Every kernel carries one shape descriptor, and it is the only IR name a
    /// target has an opinion about, so it gets a method instead of making every
    /// target spell a rename rule over arbitrary strings.
    fn shapes_name(&self) -> String {
        SHAPES_PARAM_NAME.to_string()
    }

    /// Name this target uses for the scalar descriptor.
    ///
    /// Renamed for the same reason as the shape descriptor: the IR name is
    /// shared across targets, and a target whose language reserves the leading
    /// underscores has to spell it differently.
    fn scalars_name(&self) -> String {
        SCALARS_PARAM_NAME.to_string()
    }

    /// Name this target uses for the integer scalar descriptor.
    fn scalars_i64_name(&self) -> String {
        SCALARS_I64_PARAM_NAME.to_string()
    }

    /// The statement that gives a block parameter its incoming value.
    ///
    /// A target whose tiles are addresses must copy the elements; see
    /// [`copy_assign`]. A target whose tiles are values assigns them.
    fn param_assign(&self, _ty: &lsir::Type, name: &str, arg: &str) -> String {
        format!("{name} = {arg};")
    }

    /// How this target spells a floating-point constant.
    ///
    /// An associated function, not a method, because the shared operand
    /// formatter needs the spelling without borrowing the dialect.
    fn float_literal(value: f64) -> String
    where
        Self: Sized,
    {
        float_literal(value)
    }

    /// A loop-carried hint the target wants before a simple loop, if any.
    ///
    /// `is_simple_body` reports that the loop has no inner control flow, which
    /// is the only case C asks for. A target with no vectorization pragma keeps
    /// the default of nothing.
    fn vectorize_hint(&self, _is_simple_body: bool) -> Option<String> {
        None
    }

    /// True when the target writes this result at its use, not at the top.
    ///
    /// WGSL writes an intrinsic as `let name = expr;`, so hoisting a declaration
    /// too would define the same name twice. C and MSL assign to a hoisted
    /// variable, so they keep the default of declaring everything up front.
    ///
    /// A target with no pointers also declares nothing for a `gep`: the `gep`
    /// result is the element it points at, spelled where it is consumed, so it
    /// has no value and no type of its own to declare.
    fn declares_at_use(&self, inst: &lsir::Inst) -> bool {
        self.pointers().has_address_of == false && matches!(inst.op, lsir::InstOp::Gep { .. })
            || self.declares_intrinsic_at_use(inst)
    }

    /// True when an intrinsic is a `let`-style binding at its use.
    ///
    /// Separate from [`TargetDialect::declares_at_use`] because the `gep` half
    /// of that answer comes from the pointer language and the intrinsic half is
    /// a per-target choice, so mixing them makes the default wrong for C.
    fn declares_intrinsic_at_use(&self, _inst: &lsir::Inst) -> bool {
        false
    }

    /// Diagnostics that name this target.
    fn messages(&self) -> StructuredCfgMessages;
}

/// The shared emitter: one traversal, one value order, many dialects.
///
/// [`TextCodegen`] is the public entry point. `C` and `Metal` stop implementing
/// it directly and hand this struct a dialect instead, which is what keeps the
/// next target from copying the traversal a third time.
pub struct TargetEmitter<'a, D: TargetDialect> {
    pub func: &'a lsir::Function,
    pub dialect: D,
    /// Values that own storage, so a target can tell real storage apart from a
    /// pointer into a driver buffer.
    pub storage_values: HashSet<lsir::ValueId>,
    pub value_names: HashMap<lsir::ValueId, String>,
    pub declared_values: HashSet<lsir::ValueId>,
    /// Gep results, which a pointer-free target spells as elements.
    ///
    /// Filled by [`lower_common_inst_line_with`], so the emitter must own it and
    /// pass it on every instruction.
    pub gep_elements: HashMap<lsir::ValueId, GepLink>,
    /// Geps whose only use is the load or store reading through them, with the
    /// base and indices to spell at that use.
    ///
    /// A pointer target writes `base[i]` on the spot; materializing the pointer
    /// first is a declaration and a line for nothing. Filled for a target with
    /// addresses only, and empty for a pointer-free one.
    pub folded_geps: HashMap<lsir::ValueId, (lsir::Operand, Vec<lsir::Operand>)>,
    /// Values the traversal actually wrote an assignment for.
    ///
    /// The declaration pass runs before the body, so it cannot see which values
    /// the structured lowering will fold away — a loop header's condition, a
    /// gep a load folded through. The body records them here and the held-back
    /// declarations are filtered by it.
    assigned_values: HashSet<lsir::ValueId>,
    /// The alloca-form counted loops keyed by header, from the emission
    /// pre-pass. Their index updates fold into the `for` header.
    for_loops: HashMap<lsir::BlockId, CountedForLoop>,
    /// The mapping's vectorization judgment. The policy decides whether the
    /// target wants a hint at all; the dialect still spells it.
    vector: lsir::VectorPolicy,
    /// The index allocas of those loops. Their stores fold away, so emission
    /// skips them and their declaration demotes to the scalar itself.
    demoted_ivs: HashSet<lsir::ValueId>,
    /// The step instructions of those loops. Their result only feeds the
    /// skipped store, so they fold away wherever the body put them.
    step_values: HashSet<lsir::ValueId>,
    /// Declarations held back while the body emits, filtered, then written.
    deferred_decls: Vec<(lsir::ValueId, lsir::Type)>,
    defer_decls: bool,
    pub indent: usize,
    out: String,
}

/// The two extents of the storage a tile pointer addresses, when it addresses
/// a tile at all.
pub fn tile_storage_dims(ty: &lsir::Type) -> Option<(usize, usize)> {
    let lsir::Type::Ptr { pointee, .. } = ty else {
        return None;
    };
    let lsir::Type::Array { len: rows, elem } = pointee.as_ref() else {
        return None;
    };
    let lsir::Type::Array { len: cols, .. } = elem.as_ref() else {
        return None;
    };
    Some((*rows, *cols))
}

/// A statement that makes `name` hold the tile `arg` names.
///
/// A tile parameter owns storage and receives its value from storage the loop
/// writes again on the next iteration. Taking the address would make the two
/// names alias, and the parameter would lose the value it carried, so the
/// elements are copied. A target whose tiles are values assigns them directly
/// and never calls this.
pub fn copy_assign(ty: &lsir::Type, name: &str, arg: &str) -> String {
    match tile_storage_dims(ty) {
        Some((rows, 1)) => {
            format!("for (int64_t i = 0; i < {rows}; i++) {name}[i][0] = {arg}[i][0];")
        }
        Some((rows, cols)) => format!(
            "for (int64_t i = 0; i < {rows}; i++) \
             for (int64_t j = 0; j < {cols}; j++) {name}[i][j] = {arg}[i][j];"
        ),
        None => format!("{name} = {arg};"),
    }
}

/// Every value that owns storage: an `alloca`, and a tile block parameter,
/// which is the slot the loop carries that tile in.
pub fn collect_storage_values(func: &lsir::Function) -> HashSet<lsir::ValueId> {
    let mut values = collect_alloca_results(func);
    for block in &func.blocks {
        for param in &block.params {
            if tile_storage_dims(&param.ty).is_some() {
                values.insert(param.id);
            }
        }
        // A tile-level fold writes its output into storage of its own, the way
        // an `alloca` does: the target emits the array, not a pointer to one.
        for inst in &block.insts {
            if matches!(
                inst.op,
                lsir::InstOp::Intrinsic {
                    intrinsic:
                        lsir::Intrinsic::TileReduce { .. } | lsir::Intrinsic::TileMma { .. },
                    ..
                }
            ) && let Some(id) = inst.result
            {
                values.insert(id);
            }
        }
    }
    values
}

/// Every value an `alloca` defines, on any block.
pub fn collect_alloca_results(func: &lsir::Function) -> HashSet<lsir::ValueId> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match inst.op {
            lsir::InstOp::Alloca { .. } => inst.result,
            _ => None,
        })
        .collect()
}

/// The values one instruction reads.
pub fn inst_operands(inst: &lsir::Inst) -> Vec<&lsir::Operand> {
    match &inst.op {
        lsir::InstOp::Gep { base, indices } => {
            let mut all = vec![base];
            all.extend(indices.iter());
            all
        }
        lsir::InstOp::Load { ptr } => vec![ptr],
        lsir::InstOp::Store { ptr, value } => vec![ptr, value],
        lsir::InstOp::AtomicAdd { ptr, value } => vec![ptr, value],
        lsir::InstOp::Memcpy { dst, src, size } => vec![dst, src, size],
        lsir::InstOp::Bin { lhs, rhs, .. } => vec![lhs, rhs],
        lsir::InstOp::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
        lsir::InstOp::Cast { value, .. } => vec![value],
        lsir::InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond, on_true, on_false],
        lsir::InstOp::Call { args, .. } | lsir::InstOp::Intrinsic { args, .. } => {
            args.iter().collect()
        }
        lsir::InstOp::Alloca { .. } => Vec::new(),
    }
}

/// The values a terminator reads.
pub fn terminator_operands(term: &lsir::Terminator) -> Vec<&lsir::Operand> {
    match term {
        lsir::Terminator::Br { args, .. } => args.iter().collect(),
        lsir::Terminator::CondBr {
            cond,
            true_args,
            false_args,
            ..
        } => {
            let mut all = vec![cond];
            all.extend(true_args.iter());
            all.extend(false_args.iter());
            all
        }
        lsir::Terminator::Switch { value, .. } => vec![value],
        lsir::Terminator::Ret { value } => value.iter().collect(),
    }
}

/// The geps whose only use is the load or store reading through them.
///
/// A pointer target spells that access as `base[i]` where the value is used,
/// so the gep itself leaves no text. Anything else — a pointer that lives
/// across statements, one gep feeding two accesses — stays a value.
pub fn collect_folded_geps(
    func: &lsir::Function,
) -> HashMap<lsir::ValueId, (lsir::Operand, Vec<lsir::Operand>)> {
    let mut uses: HashMap<lsir::ValueId, usize> = HashMap::new();
    let mut users: HashMap<lsir::ValueId, &lsir::Inst> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let mut note = |used: &lsir::Operand| {
                if let lsir::Operand::Value(id) = used {
                    *uses.entry(*id).or_insert(0) += 1;
                    users.insert(*id, inst);
                }
            };
            for used in inst_operands(inst) {
                note(used);
            }
        }
        for used in terminator_operands(&block.terminator) {
            if let lsir::Operand::Value(id) = used {
                *uses.entry(*id).or_insert(0) += 1;
            }
        }
    }
    let mut folded = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let lsir::InstOp::Gep { base, indices } = &inst.op else {
                continue;
            };
            let Some(result) = inst.result else {
                continue;
            };
            if uses.get(&result).copied() != Some(1) {
                continue;
            }
            if matches!(
                users.get(&result).map(|user| &user.op),
                Some(lsir::InstOp::Load { .. } | lsir::InstOp::Store { .. })
            ) {
                folded.insert(result, (base.clone(), indices.clone()));
            }
        }
    }
    folded
}

impl<'a, D: TargetDialect> TargetEmitter<'a, D> {
    pub fn new(func: &'a lsir::Function, dialect: D, vector: lsir::VectorPolicy) -> Self {
        // A target that has no pointers reads the Gep table from the first
        // instruction, so the whole function is collected once here rather
        // than filled as the Geps go by.
        let gep_elements = if dialect.pointers().has_address_of {
            HashMap::new()
        } else {
            collect_gep_elements(func)
        };
        let folded_geps = if dialect.pointers().has_address_of {
            collect_folded_geps(func)
        } else {
            HashMap::new()
        };
        let mut value_names = build_value_names(func)
            .into_iter()
            .map(|(id, name)| {
                // The descriptors are the IR names a target can reject, so they
                // are the names renamed here.
                let name = if name == SHAPES_PARAM_NAME {
                    dialect.shapes_name()
                } else if name == SCALARS_PARAM_NAME {
                    dialect.scalars_name()
                } else if name == SCALARS_I64_PARAM_NAME {
                    dialect.scalars_i64_name()
                } else {
                    name
                };
                (id, name)
            })
            .collect::<HashMap<_, _>>();
        // A `for` induction spells its alloca's scalar name wherever its uses
        // appear: the loads fold away — the header's into the `for` header,
        // the body's into the scalar they read — so none of their results
        // reach the text.
        let (for_loops, load_index) = find_alloca_for_loops(func);
        let demoted_ivs: HashSet<lsir::ValueId> =
            for_loops.values().map(|loop_| loop_.iv).collect();
        // The step instruction folds into the `for` header when nothing but
        // the loop's own edge reads it: in the memory form that is the latch's
        // store, in the promoted form the back edge's argument. When the body
        // reads the same value — the lowering computes it before the body runs
        // — it is an ordinary instruction and stays.
        let step_values: HashSet<lsir::ValueId> = for_loops
            .values()
            .filter(|loop_| !value_read_in_body(func, loop_.step_value))
            .map(|loop_| loop_.step_value)
            .collect();
        for (load, iv) in &load_index {
            if let Some(name) = value_names.get(iv).cloned() {
                value_names.insert(*load, name);
            }
        }
        Self {
            func,
            dialect,
            storage_values: collect_storage_values(func),
            value_names,
            declared_values: HashSet::new(),
            gep_elements,
            folded_geps,
            assigned_values: HashSet::new(),
            for_loops,
            vector,
            demoted_ivs,
            step_values,
            deferred_decls: Vec::new(),
            defer_decls: false,
            indent: 0,
            out: String::new(),
        }
    }

    fn param_names(&self) -> HashMap<lsir::ValueId, String> {
        self.func
            .params
            .iter()
            .map(|param| {
                let name = self
                    .value_names
                    .get(&param.id)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", param.id.0));
                (param.id, name)
            })
            .collect()
    }

    fn emit_value_decls_inner(&mut self) {
        let mut declared = HashMap::new();
        for block in &self.func.blocks {
            for param in &block.params {
                declared.entry(param.id).or_insert(param.ty.clone());
            }
            for inst in &block.insts {
                if let Some(id) = inst.result
                    && !self.dialect.declares_at_use(inst)
                {
                    declared.entry(id).or_insert(inst.ty.clone());
                }
            }
        }
        if declared.is_empty() {
            return;
        }
        let mut lines = Vec::new();
        let mut seen: Vec<lsir::ValueId> = Vec::new();
        for block in &self.func.blocks {
            for param in &block.params {
                if declared.contains_key(&param.id) && !seen.contains(&param.id) {
                    seen.push(param.id);
                    lines.push((param.id, declared[&param.id].clone()));
                }
            }
            for inst in &block.insts {
                if let Some(id) = inst.result
                    && declared.contains_key(&id)
                    && !seen.contains(&id)
                {
                    seen.push(id);
                    lines.push((id, result_type(inst)));
                }
            }
        }
        for (id, ty) in lines {
            if self.declared_values.insert(id) {
                if self.defer_decls {
                    self.deferred_decls.push((id, ty));
                } else {
                    self.write_decl(id, &ty);
                }
            }
        }
    }

    /// The held-back declarations, filtered by what the body actually wrote.
    ///
    /// The declaration pass cannot see what the structured lowering folds —
    /// a loop header's condition never reaches the text — so the body's
    /// assignments arbitrate. Storage the body fills element-wise and block
    /// params the headers carry stay declared regardless.
    fn flush_decls(&mut self) {
        for (id, ty) in std::mem::take(&mut self.deferred_decls) {
            let assigned = self.assigned_values.contains(&id);
            let storage = self.storage_values.contains(&id);
            let demoted = self.demoted_ivs.contains(&id);
            let block_param = self
                .func
                .blocks
                .iter()
                .any(|block| block.params.iter().any(|param| param.id == id));
            if assigned || storage || demoted || block_param {
                self.write_decl(id, &ty);
            }
        }
    }

    /// The declaration text for one value.
    fn write_decl(&mut self, id: lsir::ValueId, ty: &lsir::Type) {
        let name = self.value_name(id);
        // A demoted `for` induction declares the scalar itself, not the
        // storage-plus-pointer pair: the `for` header updates the scalar.
        let ty = match ty {
            lsir::Type::Ptr { pointee, .. } if self.demoted_ivs.contains(&id) => {
                pointee.as_ref().clone()
            }
            _ => ty.clone(),
        };
        let decl = self
            .dialect
            .var_decl(&ty, &name, self.storage_values.contains(&id));
        if let Some(decl) = decl {
            self.writeln(&decl);
        }
    }

    pub fn value_name(&self, id: lsir::ValueId) -> String {
        value_name(&self.value_names, id)
    }

    pub fn format_operand(&self, operand: &lsir::Operand) -> String {
        format_operand(&self.value_names, D::float_literal, operand)
    }

    pub fn writeln(&mut self, line: &str) {
        self.out.push_str(&"  ".repeat(self.indent));
        self.out.push_str(line);
        self.out.push('\n');
    }
}

impl<D: TargetDialect> TextCodegen for TargetEmitter<'_, D> {
    fn emit_prelude(&mut self) {
        for line in self.dialect.prelude_lines(self.func) {
            self.out.push_str(&line);
            self.out.push('\n');
        }
    }

    fn emit_signature(&mut self) {
        let names = self.param_names();
        let params: Vec<(String, String)> = self
            .func
            .params
            .iter()
            .map(|param| {
                let ty = self.dialect.param_decl(&param.ty, &names[&param.id]);
                (ty, names[&param.id].clone())
            })
            .collect();
        let head = self.dialect.signature(&self.func.name, &params);
        self.out.push_str(&head);
        self.indent = 1;
    }

    fn emit_body(&mut self) -> Result<()> {
        // Declarations follow the signature but depend on the body: the
        // structured lowering folds instructions away, and a value whose
        // assignment never materializes must not be declared. So the body
        // emits first, the declarations are filtered, and the two halves
        // join here in source order.
        self.defer_decls = true;
        self.emit_value_decls_inner();
        let body_start = self.out.len();
        self.out.push('\n');
        let post_dominators = OnceCell::new();
        let dominators = sile_lsir::passes::cfg::dominators(self.func);
        if self.emit_structured_from(self.func.entry, &[], &post_dominators, &dominators)?.is_some() {
            return Err(Error::Compile(
                "structured codegen unexpectedly stopped at a non-terminal block".into(),
            ));
        }
        self.indent = 0;
        self.out.push_str("}\n");
        let body = self.out.split_off(body_start);
        self.flush_decls();
        self.out.push_str(&body);
        Ok(())
    }

    fn finish(self) -> String {
        self.out
    }
}

impl<D: TargetDialect> StructuredEmitter for TargetEmitter<'_, D> {
    fn emit_block_insts(&mut self, block: &lsir::BasicBlock) -> Result<()> {
        self.emit_block_insts_except(block, None)
    }

    fn emit_block_insts_except(
        &mut self,
        block: &lsir::BasicBlock,
        skip_result: Option<lsir::ValueId>,
    ) -> Result<()> {
        for inst in &block.insts {
            if let Some(skip_result) = skip_result
                && inst.result == Some(skip_result)
            {
                continue;
            }
            // A `for` induction's updates fold into the `for` header: the
            // entry init and the step store never reach the text, the step
            // add — wherever the body put it — only fed that store, and a
            // body reload of the index reads the scalar itself.
            if let Some(result) = inst.result
                && self.step_values.contains(&result)
            {
                continue;
            }
            match &inst.op {
                lsir::InstOp::Load {
                    ptr: lsir::Operand::Value(base),
                }
                | lsir::InstOp::Store {
                    ptr: lsir::Operand::Value(base),
                    ..
                } if self.demoted_ivs.contains(base) => continue,
                _ => {}
            }
            self.emit_inst(inst)?;
        }
        Ok(())
    }

    fn emit_block_param_assignments_skipping(
        &mut self,
        target: lsir::BlockId,
        args: &[lsir::Operand],
        skip_params: &[lsir::ValueId],
    ) {
        let Some(block) = self.func.blocks.iter().find(|block| block.id == target) else {
            return;
        };
        for (param, arg) in block.params.iter().zip(args.iter()) {
            if skip_params.contains(&param.id) {
                continue;
            }
            let name = self.value_name(param.id);
            let arg = self.format_operand(arg);
            let line = self.dialect.param_assign(&param.ty, &name, &arg);
            self.writeln(&line);
        }
    }

    fn format_operand(&self, operand: &lsir::Operand) -> String {
        self.format_operand(operand)
    }

    fn writeln(&mut self, line: &str) {
        TargetEmitter::writeln(self, line);
    }

    fn indent_inc(&mut self) {
        self.indent += 1;
    }

    fn indent_dec(&mut self) {
        self.indent -= 1;
    }

    fn vectorize_hint(&mut self, is_simple_body: bool) -> Option<String> {
        // The policy decides whether the target wants a hint at all; the
        // dialect still spells it. A simple body is part of the spell.
        match self.vector {
            lsir::VectorPolicy::None => None,
            lsir::VectorPolicy::OmpSimd if is_simple_body => {
                self.dialect.vectorize_hint(is_simple_body)
            }
            lsir::VectorPolicy::OmpSimd => None,
        }
    }

    fn for_loop_induction(&self, header: lsir::BlockId) -> Option<CountedForLoop> {
        self.for_loops.get(&header).copied()
    }
}

impl<D: TargetDialect> TargetEmitter<'_, D> {
    fn emit_structured_from(
        &mut self,
        start: lsir::BlockId,
        stop_targets: &[lsir::BlockId],
        post_dominators: &OnceCell<PostDominators>,
        dominators: &Dominators,
    ) -> Result<Option<lsir::BlockId>> {
        emit_structured_from(
            self,
            self.func,
            start,
            stop_targets,
            &[],
            None,
            post_dominators,
            dominators,
            self.dialect.messages(),
        )
    }

    /// One instruction: shared lowering first, dialect second.
    ///
    /// The shared line covers every operator both targets spell the same way,
    /// which is most of the instruction set. A dialect only handles what it
    /// spells differently, so the two emitters cannot drift apart in silence.
    fn emit_inst(&mut self, inst: &lsir::Inst) -> Result<()> {
        if let lsir::InstOp::Intrinsic { intrinsic, args } = &inst.op {
            // The dialect borrows `self` immutably, so the closures it needs
            // cannot borrow `self` at the same time. Copy the name table.
            let value_names = self.value_names.clone();
            let dialect = &self.dialect;
            let lines = dialect.intrinsic_stmt(
                inst.result,
                |id| value_name(&value_names, id),
                |op| format_operand(&value_names, D::float_literal, op),
                intrinsic,
                args,
            )?;
            if !lines.is_empty() {
                if let Some(id) = inst.result {
                    self.assigned_values.insert(id);
                }
            }
            for line in lines {
                self.writeln(&line);
            }
            return Ok(());
        }

        let line = {
            let value_names = self.value_names.clone();
            lower_common_inst_line_with(
                inst,
                |id| value_name(&value_names, id),
                |op| format_operand(&value_names, D::float_literal, op),
                self.dialect.pointers(),
                self.dialect.select(),
                self.dialect.int_div(),
                D::cast_expr,
                &mut self.gep_elements,
                &self.folded_geps,
            )
        };
        if let Some(line) = line {
            self.writeln(&line);
            if let Some(id) = inst.result {
                self.assigned_values.insert(id);
            }
            return Ok(());
        }

        // The dialect borrows `self` mutably, so it cannot also borrow the
        // closures that read `self`. Name the values it needs up front.
        let value_names = self.value_names.clone();
        let lines = self.dialect.extra_inst_stmt(
            inst,
            |id| value_name(&value_names, id),
            |op| format_operand(&value_names, D::float_literal, op),
        )?;
        if !lines.is_empty() {
            if let Some(id) = inst.result {
                self.assigned_values.insert(id);
            }
        }
        for line in lines {
            self.writeln(&line);
        }
        Ok(())
    }
}

pub trait StructuredEmitter {
    fn emit_block_insts(&mut self, block: &lsir::BasicBlock) -> Result<()>;
    fn emit_block_insts_except(
        &mut self,
        block: &lsir::BasicBlock,
        skip_result: Option<lsir::ValueId>,
    ) -> Result<()>;
    fn emit_block_param_assignments_skipping(
        &mut self,
        target: lsir::BlockId,
        args: &[lsir::Operand],
        skip_params: &[lsir::ValueId],
    );
    fn emit_block_param_assignments(
        &mut self,
        target: lsir::BlockId,
        args: &[lsir::Operand],
    ) {
        self.emit_block_param_assignments_skipping(target, args, &[]);
    }
    fn format_operand(&self, operand: &lsir::Operand) -> String;

    fn writeln(&mut self, line: &str);
    fn indent_inc(&mut self);
    fn indent_dec(&mut self);

    /// A loop-carried hint the target wants before a simple loop, if any.
    fn vectorize_hint(&mut self, is_simple_body: bool) -> Option<String>;

    /// The alloca-form counted loop whose header this is, if the emission
    /// pre-pass matched one.
    fn for_loop_induction(&self, _header: lsir::BlockId) -> Option<CountedForLoop> {
        None
    }
}

#[derive(Clone, Copy)]
pub struct StructuredCfgMessages {
    pub preheader_must_branch: &'static str,
    pub missing_loop_header: &'static str,
    pub header_must_cond_br: &'static str,
    pub loop_backedge_mismatch: &'static str,
    pub unsupported_cond_br: &'static str,
    pub unsupported_switch: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub fn emit_structured_from<E: StructuredEmitter>(
    emitter: &mut E,
    func: &lsir::Function,
    start: lsir::BlockId,
    stop_targets: &[lsir::BlockId],
    stop_skips: &[(lsir::BlockId, Vec<lsir::ValueId>)],
    stop_skip_inst: Option<lsir::ValueId>,
    post_dominators: &OnceCell<PostDominators>,
    dominators: &Dominators,
    messages: StructuredCfgMessages,
) -> Result<Option<lsir::BlockId>> {
    let mut current = start;
    // An arm whose first block is the join itself (an `if` with no `else`
    // body) has nothing to spell: emitting from the join here would swallow
    // everything the join reaches into one arm, and the caller spells the
    // join right after the two arms anyway.
    if stop_targets.contains(&start) {
        return Ok(Some(start));
    }
    // One call owns one structured region: the walk threads a single chain of
    // blocks (nested ifs and loops are handed to recursive calls), so a block
    // seen twice here means the region does not close and the walk would spin
    // forever. Fail instead of eating memory; the compile layer falls back.
    let mut walked: std::collections::HashSet<lsir::BlockId> = std::collections::HashSet::new();
    loop {
        if !walked.insert(current) {
            let stop_names: Vec<String> = stop_targets
                .iter()
                .filter_map(|id| get_block(func, *id).map(|b| b.name.clone()))
                .collect();
            return Err(Error::Compile(format!(
                "structured walk revisited block {} (stop targets: {}); \
                 the region never reaches a stop target",
                get_block(func, current).map(|b| b.name.as_str()).unwrap_or("?"),
                stop_names.join(", ")
            )));
        }
        let block = get_block(func, current)
            .cloned()
            .ok_or_else(|| Error::Compile(format!("missing LSIR block {:?}", current)))?;

        if let lsir::Terminator::Br { target, args } = &block.terminator
            && stop_targets.contains(target)
        {
            // The structured loop writes the induction step in its own `for`
            // header, so the latch block must not write it a second time. The
            // step instruction is identified by the value it defines; the block
            // that carries it is whichever block is being emitted here, which
            // is the latch and not the loop header the backedge targets.
            let skip_inst = stop_skip_inst.filter(|skipped| {
                block
                    .insts
                    .iter()
                    .any(|inst| inst.result == Some(*skipped))
            });
            if let Some(skip) = skip_inst {
                emitter.emit_block_insts_except(&block, Some(skip))?;
            } else {
                emitter.emit_block_insts(&block)?;
            }
            emitter.emit_block_param_assignments_skipping(
                *target,
                args,
                skip_param_ids_for_target(stop_skips, *target),
            );
            return Ok(Some(*target));
        }

        if is_loop_preheader(func, &block, dominators) {
            current = emit_structured_loop_preheader(
                emitter,
                func,
                &block,
                stop_targets,
                stop_skips,
                post_dominators,
                dominators,
                messages,
            )?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if is_loop_header(func, &block, dominators) {
            current = emit_structured_loop_header(
                emitter,
                func,
                &block,
                stop_targets,
                stop_skips,
                post_dominators,
                dominators,
                messages,
            )?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if matches!(block.terminator, lsir::Terminator::CondBr { .. }) {
            current =
                emit_structured_if_else(
                    emitter,
                    func,
                    &block,
                    stop_targets,
                    stop_skips,
                    post_dominators,
                    dominators,
                    messages,
                )?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        emitter.emit_block_insts(&block)?;
        match &block.terminator {
            lsir::Terminator::Br { target, args } => {
                emitter.emit_block_param_assignments_skipping(
                    *target,
                    args,
                    skip_param_ids_for_target(stop_skips, *target),
                );
                if stop_targets.contains(target) {
                    return Ok(Some(*target));
                }
                current = *target;
            }
            lsir::Terminator::Ret { value: None } => {
                emitter.writeln("return;");
                return Ok(None);
            }
            lsir::Terminator::Ret { value: Some(value) } => {
                emitter.writeln(&format!("return {};", emitter.format_operand(value)));
                return Ok(None);
            }
            lsir::Terminator::CondBr { .. } => {
                return Err(Error::Compile(messages.unsupported_cond_br.into()));
            }
            lsir::Terminator::Switch { .. } => {
                return Err(Error::Compile(messages.unsupported_switch.into()));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_structured_loop_preheader<E: StructuredEmitter>(
    emitter: &mut E,
    func: &lsir::Function,
    preheader: &lsir::BasicBlock,
    outer_stops: &[lsir::BlockId],
    outer_skips: &[(lsir::BlockId, Vec<lsir::ValueId>)],
    post_dominators: &OnceCell<PostDominators>,
    dominators: &Dominators,
    messages: StructuredCfgMessages,
) -> Result<lsir::BlockId> {
    let lsir::Terminator::Br {
        target: header_id,
        args: header_args,
    } = &preheader.terminator
    else {
        return Err(Error::Compile(messages.preheader_must_branch.into()));
    };
    let header = get_block(func, *header_id)
        .cloned()
        .ok_or_else(|| Error::Compile(messages.missing_loop_header.into()))?;
    let lsir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return Err(Error::Compile(messages.header_must_cond_br.into()));
    };

    emitter.emit_block_insts(preheader)?;
    emit_structured_loop_header_impl(
        emitter,
        func,
        &header,
        Some(header_args),
        cond,
        *true_target,
        true_args,
        *false_target,
        false_args,
        outer_stops,
        outer_skips,
        post_dominators,
        dominators,
        messages,
    )
}

fn emit_structured_loop_header<E: StructuredEmitter>(
    emitter: &mut E,
    func: &lsir::Function,
    header: &lsir::BasicBlock,
    outer_stops: &[lsir::BlockId],
    outer_skips: &[(lsir::BlockId, Vec<lsir::ValueId>)],
    post_dominators: &OnceCell<PostDominators>,
    dominators: &Dominators,
    messages: StructuredCfgMessages,
) -> Result<lsir::BlockId> {
    let lsir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return Err(Error::Compile(messages.header_must_cond_br.into()));
    };
    emit_structured_loop_header_impl(
        emitter,
        func,
        header,
        None,
        cond,
        *true_target,
        true_args,
        *false_target,
        false_args,
        outer_stops,
        outer_skips,
        post_dominators,
        dominators,
        messages,
    )
}

fn emit_structured_if_else<E: StructuredEmitter>(
    emitter: &mut E,
    func: &lsir::Function,
    block: &lsir::BasicBlock,
    outer_stops: &[lsir::BlockId],
    outer_skips: &[(lsir::BlockId, Vec<lsir::ValueId>)],
    post_dominators: &OnceCell<PostDominators>,
    dominators: &Dominators,
    messages: StructuredCfgMessages,
) -> Result<lsir::BlockId> {
    let lsir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &block.terminator
    else {
        return Err(Error::Compile(messages.unsupported_cond_br.into()));
    };
    let join = branch_join(func, block.id, post_dominators)
        .ok_or_else(|| Error::Compile(messages.unsupported_cond_br.into()))?;

    let format_cond = |op: &lsir::Operand| emitter.format_operand(op);
    let (cond_expr, skip_result) =
        branch_condition_expr(&format_cond, block, func, cond, false);
    emitter.emit_block_insts_except(block, skip_result)?;
    emitter.writeln(&format!("if ({cond_expr}) {{"));
    emitter.indent_inc();
    emitter.emit_block_param_assignments(*true_target, true_args);
    // An arm may end in a `return`, which leaves it no edge to the join —
    // the arm's text is complete either way, and the join still holds for
    // the arm that reaches it. The skips pass through because an arm can
    // also end on the enclosing loop's header — a `continue` — and a `for`
    // writes its induction step itself, so the arm must not write it too.
    let mut arm_stops = Vec::with_capacity(outer_stops.len() + 1);
    arm_stops.push(join);
    arm_stops.extend_from_slice(outer_stops);
    emit_structured_from(
        emitter,
        func,
        *true_target,
        &arm_stops,
        outer_skips,
        None,
        post_dominators,
        dominators,
        messages,
    )?;
    emitter.indent_dec();
    emitter.writeln("} else {");
    emitter.indent_inc();
    emitter.emit_block_param_assignments(*false_target, false_args);
    emit_structured_from(
        emitter,
        func,
        *false_target,
        &arm_stops,
        outer_skips,
        None,
        post_dominators,
        dominators,
        messages,
    )?;
    emitter.indent_dec();
    emitter.writeln("}");

    Ok(join)
}

#[allow(clippy::too_many_arguments)]
fn emit_structured_loop_header_impl<E: StructuredEmitter>(
    emitter: &mut E,
    func: &lsir::Function,
    header: &lsir::BasicBlock,
    incoming_header_args: Option<&[lsir::Operand]>,
    cond: &lsir::Operand,
    true_target: lsir::BlockId,
    true_args: &[lsir::Operand],
    false_target: lsir::BlockId,
    false_args: &[lsir::Operand],
    outer_stops: &[lsir::BlockId],
    outer_skips: &[(lsir::BlockId, Vec<lsir::ValueId>)],
    post_dominators: &OnceCell<PostDominators>,
    dominators: &Dominators,
    messages: StructuredCfgMessages,
) -> Result<lsir::BlockId> {
    // The body walk stops at this header's latch — and at every stop target
    // the enclosing walk carries. A nested loop whose exit block is the
    // enclosing loop's latch branches straight to the enclosing header, which
    // this walk would otherwise walk straight past, re-emitting the enclosing
    // loop's body forever.
    let mut stops = Vec::with_capacity(outer_stops.len() + 1);
    stops.push(header.id);
    stops.extend_from_slice(outer_stops);
    // The lowering's alloca-form counted loop: the index lives in an alloca
    // the entry initializes and the latch steps, and the header only loads and
    // compares. All three index updates fold into the `for` header; the name
    // table already spells the header load's uses as the induction name.
    if let Some(loop_) = emitter.for_loop_induction(header.id) {
        let cmp_id = operand_value_id(cond)
            .ok_or_else(|| Error::Compile(messages.header_must_cond_br.into()))?;
        let lsir::InstOp::Cmp { pred, lhs, rhs } =
            &get_inst_by_result(func, cmp_id)
                .ok_or_else(|| Error::Compile(messages.header_must_cond_br.into()))?
                .op
        else {
            return Err(Error::Compile(messages.header_must_cond_br.into()));
        };
        let iv = emitter.format_operand(lhs);
        let cond_expr = format!(
            "{} {} {}",
            iv,
            cmp_pred_symbol(*pred),
            emitter.format_operand(rhs)
        );
        let init_expr = match loop_.init {
            LoopInit::Const(value) => format!("{iv} = {value}"),
            LoopInit::Value(id) => format!("{iv} = {}", emitter.format_operand(&lsir::Operand::Value(id))),
        };
        let step_expr = induction_step_expr(&iv, loop_.step);
        // OpenMP needs the initialization in the header, which the alloca
        // form always provides.
        if let Some(vectorize) =
            emitter.vectorize_hint(is_simple_loop_body(func, true_target, header.id))
        {
            emitter.writeln(&vectorize);
        }
        emitter.writeln(&format!(
            "for ({init_expr}; {cond_expr}; {step_expr}) {{"
        ));
        emitter.indent_inc();
        if loop_.iv_is_param {
            // The `for` header owns the index, so the entry edge's init and the
            // latch's step reach the text through it. Every other parameter the
            // header carries still needs its assignment.
            if let Some(args) = incoming_header_args {
                emitter.emit_block_param_assignments_skipping(
                    header.id,
                    args,
                    &[loop_.iv],
                );
            }
        }
        emitter.emit_block_param_assignments(true_target, true_args);
        let mut skips: Vec<(lsir::BlockId, Vec<lsir::ValueId>)> = outer_skips.to_vec();
        if loop_.iv_is_param {
            skips.push((header.id, vec![loop_.iv]));
        }
        let backedge = emit_structured_from(
            emitter,
            func,
            true_target,
            &stops,
            &skips,
            // The step instruction is skipped by the `step_values` gate in
            // `emit_block_insts_except`, which asks whether the body reads it:
            // a step only the edge consumes folds into the `for` header, and
            // one the body uses is emitted like any other instruction.
            None,
            post_dominators,
            dominators,
            messages,
        )?;
        match backedge {
            Some(id) if id == header.id => {}
            // The walk landed on an enclosing loop's header: this loop's text
            // is complete, and the walk that owns the header closes the cycle.
            // The exit block the jump came from was already spelled, so the
            // `false_target` assignments here must not be.
            Some(ancestor) => {
                emitter.indent_dec();
                emitter.writeln("}");
                return Ok(ancestor);
            }
            None => return Err(Error::Compile(messages.loop_backedge_mismatch.into())),
        }
        emitter.indent_dec();
        emitter.writeln("}");
        emitter.emit_block_param_assignments(false_target, false_args);
        return Ok(false_target);
    }

    let format_cond = |op: &lsir::Operand| emitter.format_operand(op);
    let optimized_loop = branch_condition_expr(&format_cond, header, func, cond, true);
    match optimized_loop {
        (cond_expr, skip_result @ Some(_))
            if header_only_contains_skipped_cmp(header, skip_result) =>
        {
            if let Some(args) = incoming_header_args {
                emitter.emit_block_param_assignments(header.id, args);
            }
            emitter.writeln(&format!("while ({cond_expr}) {{"));
            emitter.indent_inc();
            emitter.emit_block_param_assignments(true_target, true_args);
            let backedge = emit_structured_from(
                emitter,
                func,
                true_target,
                &stops,
                outer_skips,
                None,
                post_dominators,
                dominators,
                messages,
            )?;
            match backedge {
                Some(id) if id == header.id => {}
                Some(ancestor) => {
                    emitter.indent_dec();
                    emitter.writeln("}");
                    return Ok(ancestor);
                }
                None => return Err(Error::Compile(messages.loop_backedge_mismatch.into())),
            }
            emitter.indent_dec();
            emitter.writeln("}");
            emitter.emit_block_param_assignments(false_target, false_args);
            Ok(false_target)
        }
        _ => {
            if let Some(args) = incoming_header_args {
                emitter.emit_block_param_assignments(header.id, args);
            }
            emitter.writeln("while (true) {");
            emitter.indent_inc();
            emitter.emit_block_insts(header)?;
            emitter.writeln(&format!("if (!({})) {{", emitter.format_operand(cond)));
            emitter.indent_inc();
            emitter.emit_block_param_assignments(false_target, false_args);
            emitter.writeln("break;");
            emitter.indent_dec();
            emitter.writeln("}");

            emitter.emit_block_param_assignments(true_target, true_args);
            let backedge = emit_structured_from(
                emitter,
                func,
                true_target,
                &stops,
                outer_skips,
                None,
                post_dominators,
                dominators,
                messages,
            )?;
            match backedge {
                Some(id) if id == header.id => {}
                Some(ancestor) => {
                    emitter.indent_dec();
                    emitter.writeln("}");
                    return Ok(ancestor);
                }
                None => return Err(Error::Compile(messages.loop_backedge_mismatch.into())),
            }
            emitter.indent_dec();
            emitter.writeln("}");

            Ok(false_target)
        }
    }
}

/// A counted loop the emitter spells as a `for`.
///
/// The lowering emits every loop in memory form: the index lives in a scalar
/// `alloca`, the entry stores the initial value, the header loads and compares
/// it, and the latch stores the step. The pre-pass collects the loops matching
/// that shape exactly, so emission can fold all three index updates into the
/// `for` header and demote the alloca to a plain scalar — the canonical form
/// an OpenMP loop directive requires.
///
/// The scalar-promotion pass turns that same loop into the block-parameter
/// form: the header takes the index as a parameter, the entry edge passes the
/// initial value and the latch passes the step. Both shapes spell one `for`,
/// and [`CountedForLoop::iv_is_param`] says which one the IR holds.
#[derive(Clone, Copy)]
pub(crate) struct CountedForLoop {
    /// The index: an `alloca`'s value id in the memory form, the header's
    /// block parameter in the promoted form.
    pub iv: lsir::ValueId,
    /// The entry's initial value: a constant from the lowering, or a value
    /// the mapping strips with (the thread id).
    pub init: LoopInit,
    /// The signed step the latch adds.
    pub step: i64,
    /// The step instruction, which the latch must not emit: the `for` header
    /// writes it.
    pub step_value: lsir::ValueId,
    /// Whether `iv` is a header block parameter rather than an allocation.
    pub iv_is_param: bool,
}

/// A counted loop's initial index.
#[derive(Clone, Copy)]
pub(crate) enum LoopInit {
    Const(i64),
    /// A value defined where every loop entry can read it.
    Value(lsir::ValueId),
}

/// The alloca-form counted loops of `func`, keyed by header block.
///
/// Strictly gated: a header holding only an index load and its compare;
/// exactly two stores, an entry-side constant and a step on a load inside the
/// true arm; two predecessors, the entry edge and the latch; and a step
/// direction consistent with the comparison. The body may reload the index
/// freely, but every reload must sit inside the true arm — a load the loop
/// cannot reach would read the index outside the `for`, where the folded init
/// has not run yet. Under these gates the index has no access between its
/// init and the header or between the step and the back edge, so folding the
/// init into the `for` header's initializer and the step into its increment
/// preserves the semantics wherever the two sit. Anything else is not a
/// counted loop and falls back to the `while` spelling.
///
/// Also returns every load of a recognized index, mapped to the index: the
/// emitter spells all of them as the scalar.
pub(crate) fn find_alloca_for_loops(
    func: &lsir::Function,
) -> (
    HashMap<lsir::BlockId, CountedForLoop>,
    HashMap<lsir::ValueId, lsir::ValueId>,
) {
    let mut loads: HashMap<lsir::ValueId, Vec<(lsir::BlockId, lsir::ValueId)>> =
        HashMap::new();
    let mut stores: HashMap<lsir::ValueId, Vec<(lsir::BlockId, &lsir::Operand)>> =
        HashMap::new();
    let mut alloca_ty: HashMap<lsir::ValueId, &lsir::Type> = HashMap::new();

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                lsir::InstOp::Alloca {
                    alloc_ty,
                    addr_space: lsir::AddressSpace::Private,
                } => {
                    if let Some(result) = inst.result {
                        alloca_ty.insert(result, alloc_ty);
                    }
                }
                lsir::InstOp::Load {
                    ptr: lsir::Operand::Value(base),
                } => {
                    if let Some(result) = inst.result {
                        loads
                            .entry(*base)
                            .or_default()
                            .push((block.id, result));
                    }
                }
                lsir::InstOp::Store {
                    ptr: lsir::Operand::Value(base),
                    value,
                } => {
                    stores
                        .entry(*base)
                        .or_default()
                        .push((block.id, value));
                }
                _ => {}
            }
        }
    }

    let mut loops = HashMap::new();
    let mut load_index = HashMap::new();
    for (iv, alloc_ty) in &alloca_ty {
        // A counted index is a scalar integer; arrays are tile storage.
        if !matches!(alloc_ty, lsir::Type::I64) {
            continue;
        }
        let Some(sites) = loads.get(iv) else {
            continue;
        };
        // The header's first instruction is the index load the compare reads.
        // The header's first instruction is the index load the compare reads.
        let mut header_site = None;
        for (block_id, site_load) in sites {
            let Some(first) = get_block(func, *block_id).and_then(|b| b.insts.first()) else {
                continue;
            };
            if first.result == Some(*site_load)
                && matches!(
                    &first.op,
                    lsir::InstOp::Load {
                        ptr: lsir::Operand::Value(base),
                    } if *base == *iv
                )
            {
                header_site = Some((*block_id, *site_load));
                break;
            }
        }
        let Some((header_id, load)) = header_site else {
            continue;
        };
        let Some(store_sites) = stores.get(iv) else {
            continue;
        };
        if store_sites.len() != 2 {
            continue;
        }
        let Some(header) = get_block(func, header_id) else {
            continue;
        };
        // The header carries exactly the index load and its compare, no block
        // params, and branches on the compare.
        if !header.params.is_empty() || header.insts.len() != 2 {
            continue;
        }
        if header.insts[0].result != Some(load) {
            continue;
        }
        let lsir::InstOp::Cmp {
            pred,
            lhs: lsir::Operand::Value(lhs),
            ..
        } = &header.insts[1].op
        else {
            continue;
        };
        if *lhs != load {
            continue;
        }
        let Some(cmp_id) = header.insts[1].result else {
            continue;
        };
        let lsir::Terminator::CondBr {
            cond: lsir::Operand::Value(cond_id),
            true_target,
            ..
        } = &header.terminator
        else {
            continue;
        };
        if *cond_id != cmp_id {
            continue;
        }
        // Every reload of the index must sit inside the true arm: a load the
        // loop cannot reach reads the index outside the `for`, where the
        // folded init has not run yet.
        if !sites.iter().all(|(block_id, _)| {
            block_reaches(func, *true_target, *block_id)
        }) {
            continue;
        }
        // The two predecessors are the entry edge and the latch, told apart
        // by which one the loop itself reaches.
        let mut preds = Vec::new();
        for block in &func.blocks {
            let targets = match &block.terminator {
                lsir::Terminator::Br { target, .. } => vec![*target],
                lsir::Terminator::CondBr {
                    true_target,
                    false_target,
                    ..
                } => vec![*true_target, *false_target],
                _ => Vec::new(),
            };
            if targets.contains(&header_id) {
                preds.push(block.id);
            }
        }
        if preds.len() != 2 {
            continue;
        }
        let Some(latch) = preds
            .iter()
            .copied()
            .find(|pred| block_reaches(func, header_id, *pred))
        else {
            continue;
        };
        let Some(entry_edge) = preds.iter().copied().find(|pred| *pred != latch) else {
            continue;
        };

        // The two stores: an entry-side constant and a step on the load, taken
        // inside the loop's true arm.
        let mut init = None;
        let mut step = None;
        for (site_block, value) in store_sites {
            match value {
                lsir::Operand::Const(lsir::Constant::Int(value)) => {
                    // Entry side: on the way to the header through the entry
                    // edge, so the init runs once per loop entry.
                    if *site_block == entry_edge
                        || block_reaches(func, *site_block, entry_edge)
                    {
                        init = Some(LoopInit::Const(*value));
                    }
                }
                lsir::Operand::Value(value) => {
                    // A step adds a constant to the header's load.
                    let step_shape = get_inst_by_result(func, *value).and_then(|step_inst| {
                        match &step_inst.op {
                            lsir::InstOp::Bin {
                                op: lsir::BinOp::Add,
                                lhs: lsir::Operand::Value(id),
                                rhs: lsir::Operand::Const(lsir::Constant::Int(k)),
                            } if *id == load => Some((lsir::BinOp::Add, *k)),
                            lsir::InstOp::Bin {
                                op: lsir::BinOp::Sub,
                                lhs: lsir::Operand::Value(id),
                                rhs: lsir::Operand::Const(lsir::Constant::Int(k)),
                            } if *id == load => Some((lsir::BinOp::Sub, *k)),
                            _ => None,
                        }
                    });
                    if let Some((op, amount)) = step_shape {
                        // Inside the true arm, inside the loop, and on the way
                        // back to the header.
                        if block_reaches(func, *true_target, *site_block)
                            && block_reaches(func, header_id, *site_block)
                            && block_reaches(func, *site_block, header_id)
                        {
                            step = Some((op, *site_block, amount, *value));
                        }
                        continue;
                    }
                    // Otherwise an entry-side value init, which the mapping's
                    // strips spell: the thread id. It must be defined where
                    // every loop entry can read it — a parameter or the entry
                    // block, which dominates every loop the `for` reaches.
                    if (*site_block == entry_edge
                        || block_reaches(func, *site_block, entry_edge))
                        && defined_in_entry(func, *value)
                    {
                        init = Some(LoopInit::Value(*value));
                    }
                }
                _ => {}
            }
        }
        let Some(init_value) = init else {
            continue;
        };
        let Some((op, latch, amount, step_value)) = step else {
            continue;
        };
        // The step instruction lives next to its store, in the latch, so the
        // emission skip finds both in one block.
        let latch_holds_step = get_block(func, latch).is_some_and(|block| {
            block.insts.iter().any(|inst| inst.result == Some(step_value))
        });
        if !latch_holds_step {
            continue;
        }
        // The back edge must be the loop's true arm: the `for` body is what
        // returns to the header.
        if !block_reaches(func, *true_target, header_id) {
            continue;
        }
        let signed_step = match op {
            lsir::BinOp::Add => amount,
            _ => -amount,
        };
        let direction_ok = matches!(
            (signed_step > 0, pred),
            (true, lsir::CmpPred::Slt | lsir::CmpPred::Sle)
                | (false, lsir::CmpPred::Sgt | lsir::CmpPred::Sge)
        );
        if !direction_ok {
            continue;
        }
        loops.insert(
            header_id,
            CountedForLoop {
                iv: *iv,
                init: init_value,
                step: signed_step,
                step_value,
                iv_is_param: false,
            },
        );
        for (_, load_result) in sites {
            load_index.insert(*load_result, *iv);
        }
    }
    // The promoted form: the scalar-promotion pass moved the index into the
    // header's block parameters, so the same loop reads as a parameter, an
    // edge-carried init and an edge-carried step.
    for (header_id, loop_) in find_promoted_for_loops(func) {
        loops.entry(header_id).or_insert(loop_);
    }
    (loops, load_index)
}

/// Counted loops whose index is a header block parameter.
///
/// The promoted shape is the memory shape after renaming: the header takes the
/// index as its last parameter, the entry edge passes a constant, the latch
/// passes `iv + step`, and the header compares its own index. Which parameter
/// carries the index is decided by the compare, not by position, so a loop
/// that carries other values promotes as well.
fn find_promoted_for_loops(
    func: &lsir::Function,
) -> HashMap<lsir::BlockId, CountedForLoop> {
    let mut loops = HashMap::new();
    for block in &func.blocks {
        if block.params.is_empty() {
            continue;
        }
        let lsir::Terminator::CondBr {
            cond: lsir::Operand::Value(cond_id),
            true_target,
            ..
        } = &block.terminator
        else {
            continue;
        };
        // The header's compare must read one of its own parameters: that is
        // the index this loop counts.
        let Some(cmp_inst) = block.insts.iter().find(|inst| inst.result == Some(*cond_id))
        else {
            continue;
        };
        let lsir::InstOp::Cmp { pred, lhs, .. } = &cmp_inst.op else {
            continue;
        };
        let Some(lhs_id) = operand_value_id(lhs) else {
            continue;
        };
        let Some(index_in_params) = block
            .params
            .iter()
            .position(|param| param.id == lhs_id)
        else {
            continue;
        };
        // The limit itself is not needed here: emission spells whatever the
        // compare reads, constant or not. A dynamic limit is a counted loop
        // all the same.
        if !matches!(cmp_inst.ty, lsir::Type::I1) {
            continue;
        }
        // Two predecessors: the entry edge carrying the init and the latch
        // carrying the step.
        let mut edges: Vec<(lsir::BlockId, &Vec<lsir::Operand>)> = Vec::new();
        for candidate in &func.blocks {
            match &candidate.terminator {
                lsir::Terminator::Br { target, args } if *target == block.id => {
                    edges.push((candidate.id, args));
                }
                lsir::Terminator::CondBr {
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                    ..
                } => {
                    if *true_target == block.id {
                        edges.push((candidate.id, true_args));
                    }
                    if *false_target == block.id {
                        edges.push((candidate.id, false_args));
                    }
                }
                _ => {}
            }
        }
        if edges.len() != 2 {
            continue;
        }
        let mut init = None;
        let mut step = None;
        for (edge_block, args) in &edges {
            let Some(arg) = args.get(index_in_params) else {
                continue;
            };
            match arg {
                lsir::Operand::Const(lsir::Constant::Int(value)) => {
                    init = Some(LoopInit::Const(*value))
                }
                lsir::Operand::Value(value) => {
                    // A step adds a constant to the header's index parameter.
                    let step_shape = get_inst_by_result(func, *value).and_then(|step_inst| {
                        match &step_inst.op {
                            lsir::InstOp::Bin {
                                op: lsir::BinOp::Add,
                                lhs: lsir::Operand::Value(id),
                                rhs: lsir::Operand::Const(lsir::Constant::Int(k)),
                            } if *id == lhs_id => Some((lsir::BinOp::Add, *k)),
                            lsir::InstOp::Bin {
                                op: lsir::BinOp::Sub,
                                lhs: lsir::Operand::Value(id),
                                rhs: lsir::Operand::Const(lsir::Constant::Int(k)),
                            } if *id == lhs_id => Some((lsir::BinOp::Sub, *k)),
                            _ => None,
                        }
                    });
                    if let Some((op, amount)) = step_shape {
                        // The step must run inside the loop: on the way from
                        // the true arm back to the header.
                        if !block_reaches(func, *true_target, *edge_block) {
                            continue;
                        }
                        step = Some((op, *edge_block, amount, *value));
                        continue;
                    }
                    // Otherwise an entry-edge value init, which the mapping's
                    // strips spell: the thread id. It must be defined where
                    // every loop entry can read it, and run outside the loop.
                    if !block_reaches(func, *true_target, *edge_block)
                        && defined_in_entry(func, *value)
                    {
                        init = Some(LoopInit::Value(*value));
                    }
                }
                _ => {}
            }
        }
        let (Some(init_value), Some((op, _latch, amount, step_value))) = (init, step) else {
            continue;
        };
        let signed_step = match op {
            lsir::BinOp::Add => amount,
            _ => -amount,
        };
        let direction_ok = matches!(
            (signed_step > 0, pred),
            (true, lsir::CmpPred::Slt | lsir::CmpPred::Sle)
                | (false, lsir::CmpPred::Sgt | lsir::CmpPred::Sge)
        );
        if !direction_ok {
            continue;
        }
        loops.insert(
            block.id,
            CountedForLoop {
                iv: lhs_id,
                init: init_value,
                step: signed_step,
                step_value,
                iv_is_param: true,
            },
        );
    }
    loops
}

/// Is `value` defined where every loop entry can read it: a parameter, or a
/// definition in the entry block, which dominates every loop the `for` text
/// reaches? The thread id the mapping plants in the entry is the one case.
fn defined_in_entry(func: &lsir::Function, value: lsir::ValueId) -> bool {
    func.params.iter().any(|param| param.id == value)
        || get_block(func, func.entry).is_some_and(|entry| {
            entry.insts.iter().any(|inst| inst.result == Some(value))
        })
}

/// Whether any instruction reads `value` as an ordinary operand.
///
/// A counted loop's step is read by the loop's own edge — the latch's store in
/// the memory form, the back edge's argument in the promoted form. Both fold
/// into the `for` header's increment. A read from anywhere else means the body
/// uses the stepped value, and then the instruction has to run.
fn value_read_in_body(func: &lsir::Function, value: lsir::ValueId) -> bool {
    func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| {
            inst_operands(inst)
                .iter()
                .any(|operand| matches!(operand, lsir::Operand::Value(id) if *id == value))
        })
    })
}

/// Loop step text, as an assignment every target accepts.
///
/// C spells this `++i`, and WGSL has no increment operator, so the shared
/// emitter writes `i = i + step` even for a pointer target: both compile, and
/// one spelling means the loop increment is not another place the dialects
/// disagree.
fn induction_step_expr(name: &str, step: i64) -> String {
    format!("{name} = {name} + {step}")
}

fn is_simple_loop_body(
    func: &lsir::Function,
    body_start: lsir::BlockId,
    header_id: lsir::BlockId,
) -> bool {
    let mut current = body_start;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(block) = get_block(func, current) else {
            return false;
        };
        // The shape walk decides control flow only. A float function in the
        // body is spelled inline where its target puts it, and the vector
        // hint is a permission the compiler takes or leaves: the handwritten
        // softmax measures `omp simd` over `expf` at 5.5x its scalar form,
        // so a call is no reason to deny the hint. Nested control flow still
        // is — the walk below only accepts a straight chain back to the
        // header.
        match &block.terminator {
            lsir::Terminator::Br { target, .. } if *target == header_id => return true,
            lsir::Terminator::Br { target, .. } => current = *target,
            _ => return false,
        }
    }
}

fn skip_param_ids_for_target<'a>(
    skips: &'a [(lsir::BlockId, Vec<lsir::ValueId>)],
    target: lsir::BlockId,
) -> &'a [lsir::ValueId] {
    skips
        .iter()
        .find(|(skip_target, _)| *skip_target == target)
        .map(|(_, ids)| ids.as_slice())
        .unwrap_or(&[])
}

fn get_block(func: &lsir::Function, id: lsir::BlockId) -> Option<&lsir::BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn get_inst_by_result(func: &lsir::Function, id: lsir::ValueId) -> Option<&lsir::Inst> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(id))
}

fn is_loop_preheader(func: &lsir::Function, block: &lsir::BasicBlock, dominators: &Dominators) -> bool {
    let lsir::Terminator::Br { target: header, .. } = block.terminator else {
        return false;
    };
    get_block(func, header)
        .map(|header| is_loop_header(func, header, dominators))
        .unwrap_or(false)
}

pub(crate) fn is_loop_header(func: &lsir::Function, block: &lsir::BasicBlock, dominators: &Dominators) -> bool {
    let lsir::Terminator::CondBr { .. } = block.terminator else {
        return false;
    };
    // A backedge is an edge whose source the header dominates. Reachability
    // alone cannot tell it: one arm of an `if` joins on the branch itself, and
    // in a mapped kernel that arm can wrap around an enclosing loop before it
    // lands, which looks like a cycle to a reachability walk and spelled the
    // join as a `while` loop that swallowed everything after it.
    func.blocks
        .iter()
        .filter(|source| {
            sile_lsir::passes::cfg::successors(&source.terminator).contains(&block.id)
        })
        .any(|source| dominators[&source.id].contains(&block.id))
}

/// The block the two arms of a branch continue at: the join.
///
/// An arm may end at the join directly, or it may run nested structured code
/// first: a tile operation in an arm lowers to a loop nest of its own. The join
/// is then the closest block that every path out of the branch passes through,
/// which is the immediate post-dominator of the branch. Asking for the nearest
/// block both arms reach would name a block inside an arm instead, because the
/// loop that encloses the branch reaches the arm's blocks too.
fn branch_join(
    func: &lsir::Function,
    block_id: lsir::BlockId,
    post_dominators: &OnceCell<PostDominators>,
) -> Option<lsir::BlockId> {
    if let Some(join) = branch_join_post_dominator(func, block_id, post_dominators) {
        return Some(join);
    }
    // A branch whose arm returns has a join that does not post-dominate it:
    // the path through the returning arm never reaches the join, so the set
    // arithmetic above finds nothing. The join is still readable off the
    // arms — an arm that continues names it, and an arm that returns names
    // nothing, so the continuing arm's target is the join. Two continuing
    // arms must name the same one, or the CFG is not structured.
    let lsir::Terminator::CondBr {
        true_target,
        false_target,
        ..
    } = &get_block(func, block_id)?.terminator
    else {
        return None;
    };
    let arm_target = |id: &lsir::BlockId| {
        get_block(func, *id).and_then(|arm| match &arm.terminator {
            lsir::Terminator::Br { target, .. } => Some(*target),
            _ => None,
        })
    };
    match (arm_target(true_target), arm_target(false_target)) {
        (Some(join), None) | (None, Some(join)) => Some(join),
        (Some(a), Some(b)) if a == b => Some(a),
        _ => None,
    }
}

fn branch_join_post_dominator(
    func: &lsir::Function,
    block_id: lsir::BlockId,
    post_dominators: &OnceCell<PostDominators>,
) -> Option<lsir::BlockId> {
    let post_dominators = post_dominators.get_or_init(|| compute_post_dominators(func));
    // A block closer to the branch post-dominates a farther one, so it has the
    // larger set: the join is the closest, which is the largest.
    let mut best: Option<(lsir::BlockId, usize)> = None;
    for candidate in post_dominators.get(&block_id)? {
        if *candidate == block_id {
            continue;
        }
        let covered = post_dominators.get(candidate).map_or(0, HashSet::len);
        if best.is_none_or(|(_, best_covered)| covered > best_covered) {
            best = Some((*candidate, covered));
        }
    }
    best.map(|(join, _)| join)
}

/// The post-dominators of every block, each set including the block itself.
///
/// A block post-dominates a second block when every path from the second block
/// to a return passes through the first. This is the dominator computation on
/// the reversed graph: a block's set is the intersection of its successors'
/// sets, and a block that returns post-dominates only itself.

/// The post-dominator table of one function, computed at most once and only
/// when a branch actually asks for it.
type PostDominators = HashMap<lsir::BlockId, HashSet<lsir::BlockId>>;

/// The dominator table of one function, each set including the block itself.
///
/// The structured walk asks for it at every block it visits — a loop header
/// is a block a backedge returns to — and the dataflow rebuilds the whole
/// table per call, so one table serves the walk instead.
type Dominators = HashMap<lsir::BlockId, HashSet<lsir::BlockId>>;

fn compute_post_dominators(
    func: &lsir::Function,
) -> HashMap<lsir::BlockId, HashSet<lsir::BlockId>> {
    compute_post_dominators_inner(func)
}

fn compute_post_dominators_inner(
    func: &lsir::Function,
) -> HashMap<lsir::BlockId, HashSet<lsir::BlockId>> {
    let returns =
        |block: &lsir::BasicBlock| matches!(block.terminator, lsir::Terminator::Ret { .. });
    if !func.blocks.iter().any(returns) {
        // Nothing returns, so no path ends and no block post-dominates another.
        return func
            .blocks
            .iter()
            .map(|block| (block.id, HashSet::from([block.id])))
            .collect();
    }

    let all: HashSet<lsir::BlockId> = func.blocks.iter().map(|block| block.id).collect();
    let mut post_dominators: HashMap<lsir::BlockId, HashSet<lsir::BlockId>> = func
        .blocks
        .iter()
        .map(|block| {
            let set = if returns(block) {
                HashSet::from([block.id])
            } else {
                all.clone()
            };
            (block.id, set)
        })
        .collect();

    loop {
        let mut changed = false;
        for block in &func.blocks {
            if returns(block) {
                continue;
            }
            let mut new_set: Option<HashSet<lsir::BlockId>> = None;
            for successor in terminator_successors(&block.terminator) {
                let successor_set = post_dominators.get(&successor).cloned().unwrap_or_default();
                new_set = Some(match new_set {
                    None => successor_set,
                    Some(current) => current.intersection(&successor_set).copied().collect(),
                });
            }
            let mut new_set = new_set.unwrap_or_default();
            new_set.insert(block.id);
            if post_dominators.get(&block.id) != Some(&new_set) {
                post_dominators.insert(block.id, new_set);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    post_dominators
}

fn terminator_successors(terminator: &lsir::Terminator) -> Vec<lsir::BlockId> {
    match terminator {
        lsir::Terminator::Br { target, .. } => vec![*target],
        lsir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        lsir::Terminator::Switch { default, cases, .. } => {
            let mut targets = vec![*default];
            targets.extend(cases.iter().map(|(_, target)| *target));
            targets
        }
        lsir::Terminator::Ret { .. } => Vec::new(),
    }
}

/// Does `start` reach `goal` along branches?
///
/// The walk memoizes: a block whose answer is known short-circuits, and a
/// block on the active path reports `false` for the cycle it closes (the
/// answer for the path through it is the caller's to make). The mapped
/// kernels spell CFGs thousands of blocks deep, where a backtracking walk —
/// one that un-marks blocks on its way out — re-explores the same subgraph
/// once per path and blows the stack.
fn block_reaches(
    func: &lsir::Function,
    start: lsir::BlockId,
    goal: lsir::BlockId,
) -> bool {
    let mut done: HashMap<lsir::BlockId, bool> = HashMap::new();
    let mut active: HashSet<lsir::BlockId> = HashSet::new();
    block_reaches_walk(func, start, goal, &mut done, &mut active)
}

fn block_reaches_walk(
    func: &lsir::Function,
    start: lsir::BlockId,
    goal: lsir::BlockId,
    done: &mut HashMap<lsir::BlockId, bool>,
    active: &mut HashSet<lsir::BlockId>,
) -> bool {
    if start == goal {
        return true;
    }
    if let Some(&known) = done.get(&start) {
        return known;
    }
    if !active.insert(start) {
        return false;
    }
    let reaches = match get_block(func, start).map(|block| &block.terminator) {
        Some(lsir::Terminator::Br { target, .. }) => {
            block_reaches_walk(func, *target, goal, done, active)
        }
        Some(lsir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        }) => {
            block_reaches_walk(func, *true_target, goal, done, active)
                || block_reaches_walk(func, *false_target, goal, done, active)
        }
        _ => false,
    };
    active.remove(&start);
    done.insert(start, reaches);
    reaches
}

/// Spell a branch condition as an expression.
///
/// Operands go through the emitter's formatter so skipped definitions (an
/// induction variable's body reload, say) still name the value that survives.
fn branch_condition_expr(
    format_operand: &impl Fn(&lsir::Operand) -> String,
    block: &lsir::BasicBlock,
    func: &lsir::Function,
    cond: &lsir::Operand,
    require_header_only: bool,
) -> (String, Option<lsir::ValueId>) {
    let Some(cond_id) = operand_value_id(cond) else {
        return (format_operand(cond), None);
    };
    let Some(inst) = get_inst_by_result(func, cond_id) else {
        return (format_operand(cond), None);
    };
    if !matches!(inst.op, lsir::InstOp::Cmp { .. }) {
        return (format_operand(cond), None);
    }
    if require_header_only && !header_only_contains_skipped_cmp(block, Some(cond_id)) {
        return (format_operand(cond), None);
    }
    match &inst.op {
        lsir::InstOp::Cmp { pred, lhs, rhs } => (
            format!(
                "{} {} {}",
                format_operand(lhs),
                cmp_pred_symbol(*pred),
                format_operand(rhs)
            ),
            Some(cond_id),
        ),
        _ => (format_operand(cond), None),
    }
}

fn header_only_contains_skipped_cmp(
    block: &lsir::BasicBlock,
    skip_result: Option<lsir::ValueId>,
) -> bool {
    let Some(skip_result) = skip_result else {
        return false;
    };
    block
        .insts
        .iter()
        .all(|inst| inst.result == Some(skip_result))
}

fn operand_value_id(operand: &lsir::Operand) -> Option<lsir::ValueId> {
    match operand {
        lsir::Operand::Value(id) => Some(*id),
        lsir::Operand::Const(_) => None,
    }
}
