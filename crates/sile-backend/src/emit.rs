use std::collections::{HashMap, HashSet};
use std::cell::OnceCell;

use sile_core::{Error, Result};
use sile_llvm_ir as llvm_ir;
use sile_llvm_ir::Function as LlvmIrFunction;

use crate::cpu::codegen::generate as generate_llvm_ir_c_source;
use crate::metal::codegen::generate as generate_llvm_ir_metal_source;
use crate::ptx::codegen::generate as generate_llvm_ir_ptx_source;
use crate::wgsl::codegen::generate as generate_llvm_ir_wgsl_source;
use crate::{BackendArtifact, CodegenTarget};

pub fn run(llvm_ir: &LlvmIrFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    let artifact = match target {
        CodegenTarget::C => BackendArtifact::CSource(generate_llvm_ir_c_source(llvm_ir)?),
        CodegenTarget::Metal => {
            BackendArtifact::MetalSource(generate_llvm_ir_metal_source(llvm_ir)?)
        }
        CodegenTarget::Wgsl => BackendArtifact::WgslSource(generate_llvm_ir_wgsl_source(llvm_ir)?),
        CodegenTarget::Ptx => BackendArtifact::PtxSource(generate_llvm_ir_ptx_source(llvm_ir)?),
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

pub fn build_value_names(func: &llvm_ir::Function) -> HashMap<llvm_ir::ValueId, String> {
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

pub fn value_name(value_names: &HashMap<llvm_ir::ValueId, String>, id: llvm_ir::ValueId) -> String {
    value_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", id.0))
}

pub fn format_operand(
    value_names: &HashMap<llvm_ir::ValueId, String>,
    float_literal: fn(f64) -> String,
    operand: &llvm_ir::Operand,
) -> String {
    match operand {
        llvm_ir::Operand::Value(id) => value_name(value_names, *id),
        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value)) => value.to_string(),
        llvm_ir::Operand::Const(llvm_ir::Constant::Float(value)) => float_literal(*value),
        llvm_ir::Operand::Const(llvm_ir::Constant::Bool(value)) => {
            if *value {
                "true".into()
            } else {
                "false".into()
            }
        }
    }
}

pub fn array_dims(ty: &llvm_ir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let llvm_ir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

pub fn bin_op_symbol(op: llvm_ir::BinOp) -> &'static str {
    match op {
        llvm_ir::BinOp::Add => "+",
        llvm_ir::BinOp::Sub => "-",
        llvm_ir::BinOp::Mul => "*",
        llvm_ir::BinOp::Div => "/",
        llvm_ir::BinOp::Rem => "%",
        llvm_ir::BinOp::And => "&",
        llvm_ir::BinOp::Or => "|",
        llvm_ir::BinOp::Xor => "^",
        llvm_ir::BinOp::Shl => "<<",
        llvm_ir::BinOp::Shr => ">>",
    }
}

pub fn cmp_pred_symbol(pred: llvm_ir::CmpPred) -> &'static str {
    match pred {
        llvm_ir::CmpPred::Eq => "==",
        llvm_ir::CmpPred::Ne => "!=",
        llvm_ir::CmpPred::Slt | llvm_ir::CmpPred::Olt => "<",
        llvm_ir::CmpPred::Sle | llvm_ir::CmpPred::Ole => "<=",
        llvm_ir::CmpPred::Sgt | llvm_ir::CmpPred::Ogt => ">",
        llvm_ir::CmpPred::Sge | llvm_ir::CmpPred::Oge => ">=",
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
pub const SHAPES_PARAM_NAME: &str = llvm_ir::SHAPES_PARAM_NAME;

/// The synthetic scalar parameter, bound after the shape descriptor.
///
/// A scalar kernel argument travels as a load from here rather than as a
/// function parameter, because WGSL cannot pass a value into an entry point.
pub const SCALARS_PARAM_NAME: &str = llvm_ir::SCALARS_PARAM_NAME;

/// True when the parameter is the shape descriptor, not a kernel buffer.
pub fn is_shapes_param(param: &llvm_ir::Param) -> bool {
    param.name == SHAPES_PARAM_NAME
}

/// True when the parameter is the float scalar descriptor, not a kernel buffer.
pub fn is_scalars_param(param: &llvm_ir::Param) -> bool {
    param.name == SCALARS_PARAM_NAME
}

/// The synthetic integer scalar parameter, bound after the float one.
pub const SCALARS_I64_PARAM_NAME: &str = llvm_ir::SCALARS_I64_PARAM_NAME;

/// True when the parameter is the integer scalar descriptor.
pub fn is_scalars_i64_param(param: &llvm_ir::Param) -> bool {
    param.name == SCALARS_I64_PARAM_NAME
}

/// True when the parameter is a descriptor rather than a kernel buffer.
///
/// The descriptors are the synthetic trailing parameters, and every place that
/// counts or picks kernel buffers has to skip all of them.
pub fn is_descriptor_param(param: &llvm_ir::Param) -> bool {
    is_shapes_param(param) || is_scalars_param(param) || is_scalars_i64_param(param)
}

/// The output buffer of a kernel, with the tile it is stored through.
///
/// The tile shape comes from `ParamAbi::tile`, which Tile IR lowering already
/// knew when it emitted `store_ptr_tko`, so no backend re-derives it by scanning
/// the CFG for the last store.
pub struct OutputBuffer<'a> {
    pub param: &'a llvm_ir::Param,
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
pub fn find_output_param(func: &llvm_ir::Function) -> Option<&llvm_ir::Param> {
    func.params
        .iter()
        .rev()
        .find(|param| !is_descriptor_param(param))
}

/// Last non-descriptor parameter carrying tile ABI metadata.
pub fn find_output_buffer(func: &llvm_ir::Function) -> Option<OutputBuffer<'_>> {
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
pub fn flatten_grid(func: &llvm_ir::Function, launch: &sile_core::LaunchConfig) -> [u32; 3] {
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
    abi: &llvm_ir::ParamAbi,
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
    pub base: llvm_ir::Operand,
    pub indices: Vec<llvm_ir::Operand>,
    pub strides: Vec<usize>,
}

/// The type of every value in the function, including an alloca's pointer.
///
/// A Gep's stride comes from the pointee of the value it indexes, so the table
/// has to cover parameters, block parameters and instruction results.
fn value_types(func: &llvm_ir::Function) -> HashMap<llvm_ir::ValueId, llvm_ir::Type> {
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
pub fn result_type(inst: &llvm_ir::Inst) -> llvm_ir::Type {
    match &inst.op {
        llvm_ir::InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => llvm_ir::Type::ptr(addr_space.clone(), alloc_ty.clone()),
        _ => inst.ty.clone(),
    }
}

/// How much one index of a Gep advances, in leaf elements.
///
/// Index `i` steps over the dimensions after it, so a row index of `[2 x [8 x
/// f32]]` advances eight elements and the column index advances one.
fn index_strides(pointee: &llvm_ir::Type, count: usize) -> Vec<usize> {
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
    ptr: &llvm_ir::Operand,
    gep_elements: &'g HashMap<llvm_ir::ValueId, GepLink>,
) -> Option<(&'g llvm_ir::Operand, Vec<(llvm_ir::Operand, usize)>)> {
    let llvm_ir::Operand::Value(id) = ptr else {
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
    index: &llvm_ir::Operand,
    stride: usize,
    format_operand: &dyn Fn(&llvm_ir::Operand) -> String,
    index_cast: fn(String) -> String,
) -> String {
    // A constant index folds into the constant, so a tile access with literal
    // coordinates stays one literal in the emitted source.
    if let llvm_ir::Operand::Const(llvm_ir::Constant::Int(value)) = index {
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
    ptr: &llvm_ir::Operand,
    gep_elements: &HashMap<llvm_ir::ValueId, GepLink>,
    value_name: &dyn Fn(llvm_ir::ValueId) -> String,
    format_operand: &dyn Fn(&llvm_ir::Operand) -> String,
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
            llvm_ir::Operand::Value(id) => format!("{}[0]", value_name(*id)),
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
pub fn collect_gep_elements(func: &llvm_ir::Function) -> HashMap<llvm_ir::ValueId, GepLink> {
    let types = value_types(func);
    let mut gep_elements = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let llvm_ir::InstOp::Gep { base, indices } = &inst.op
                && let Some(result) = inst.result
            {
                let strides = match base {
                    llvm_ir::Operand::Value(id) => match types.get(id) {
                        Some(llvm_ir::Type::Ptr { pointee, .. }) => {
                            index_strides(pointee, indices.len())
                        }
                        _ => vec![1; indices.len()],
                    },
                    llvm_ir::Operand::Const(_) => vec![1; indices.len()],
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
    inst: &llvm_ir::Inst,
    value_name: FValue,
    format_operand: FOperand,
    pointers: PointerSpelling,
    select: SelectSpelling,
    int_div: IntDivSpelling,
    cast_expr: fn(llvm_ir::CastOp, &llvm_ir::Type, String) -> String,
    gep_elements: &mut HashMap<llvm_ir::ValueId, GepLink>,
    folded_geps: &HashMap<llvm_ir::ValueId, (llvm_ir::Operand, Vec<llvm_ir::Operand>)>,
) -> Option<String>
where
    FValue: Fn(llvm_ir::ValueId) -> String,
    FOperand: Fn(&llvm_ir::Operand) -> String,
{
    // The Gep table is collected once in `TargetEmitter::new` for a
    // pointer-free target, so a Gep is already there when its use arrives.

    let element = |ptr: &llvm_ir::Operand| -> String {
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
    let addressed = |ptr: &llvm_ir::Operand| -> Option<String> {
        match ptr {
            llvm_ir::Operand::Value(id) => folded_geps.get(id).map(|(base, indices)| {
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
        llvm_ir::InstOp::Bin { op, lhs, rhs } => inst.result.map(|id| {
            match (int_div, &inst.ty, op) {
                (
                    IntDivSpelling::I32Widened,
                    llvm_ir::Type::I64,
                    llvm_ir::BinOp::Div | llvm_ir::BinOp::Rem,
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
        llvm_ir::InstOp::Cmp { pred, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                cmp_pred_symbol(*pred),
                format_operand(rhs)
            )
        }),
        llvm_ir::InstOp::Select {
            cond,
            on_true,
            on_false,
        } => inst.result.map(|id| {
            let arg = |op: &llvm_ir::Operand| -> String {
                match (select, &inst.ty, op) {
                    (
                        SelectSpelling::Function { int_const },
                        llvm_ir::Type::I64,
                        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value)),
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
        llvm_ir::InstOp::Load { ptr } => inst.result.map(|id| {
            if let Some(address) = addressed(ptr) {
                format!("{} = {};", value_name(id), address)
            } else if pointers.load_deref {
                format!("{} = *({});", value_name(id), format_operand(ptr))
            } else {
                // No pointer to dereference, so the load reads the element.
                format!("{} = {};", value_name(id), element(ptr))
            }
        }),
        llvm_ir::InstOp::Store { ptr, value } => {
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
        llvm_ir::InstOp::Cast { op, value, to } => inst.result.map(|id| {
            format!(
                "{} = {};",
                value_name(id),
                cast_expr(*op, to, format_operand(value))
            )
        }),
        // A language with addresses materializes the pointer — unless its only
        // use folded into the access it serves — and one without already
        // recorded the element above and emits nothing.
        llvm_ir::InstOp::Gep { base, indices } if pointers.has_address_of => {
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
    /// must stay a plain pointer. The two have the same LLVM IR type, so the
    /// flag is what tells them apart.
    fn var_decl(&self, ty: &llvm_ir::Type, name: &str, is_alloca: bool) -> Option<String>;

    /// Parameter text for one kernel parameter.
    fn param_decl(&self, ty: &llvm_ir::Type, name: &str) -> String;

    /// Text of the function head, up to and including the opening brace.
    ///
    /// Each entry of `params` is `(declaration, name)`, so the dialect decides
    /// whether the two are joined by a space, as in C, or an attribute, as in
    /// MSL.
    fn signature(&self, name: &str, params: &[(String, String)]) -> String;

    /// Lines emitted before the kernel body, after the declarations.
    fn prelude_lines(&self) -> Vec<String>;

    /// Assignment of an intrinsic result, or a statement when it has none.
    fn intrinsic_stmt(
        &self,
        result: Option<llvm_ir::ValueId>,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
        intrinsic: &llvm_ir::Intrinsic,
        args: &[llvm_ir::Operand],
    ) -> sile_core::Result<Vec<String>>;

    /// A statement per target for operations the shared traversal does not know.
    fn extra_inst_stmt(
        &mut self,
        inst: &llvm_ir::Inst,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
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
    fn cast_expr(_op: llvm_ir::CastOp, _to: &llvm_ir::Type, value: String) -> String
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
    fn param_assign(&self, _ty: &llvm_ir::Type, name: &str, arg: &str) -> String {
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
    fn declares_at_use(&self, inst: &llvm_ir::Inst) -> bool {
        self.pointers().has_address_of == false && matches!(inst.op, llvm_ir::InstOp::Gep { .. })
            || self.declares_intrinsic_at_use(inst)
    }

    /// True when an intrinsic is a `let`-style binding at its use.
    ///
    /// Separate from [`TargetDialect::declares_at_use`] because the `gep` half
    /// of that answer comes from the pointer language and the intrinsic half is
    /// a per-target choice, so mixing them makes the default wrong for C.
    fn declares_intrinsic_at_use(&self, _inst: &llvm_ir::Inst) -> bool {
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
    pub func: &'a llvm_ir::Function,
    pub dialect: D,
    /// Values that own storage, so a target can tell real storage apart from a
    /// pointer into a driver buffer.
    pub storage_values: HashSet<llvm_ir::ValueId>,
    pub value_names: HashMap<llvm_ir::ValueId, String>,
    pub declared_values: HashSet<llvm_ir::ValueId>,
    /// Gep results, which a pointer-free target spells as elements.
    ///
    /// Filled by [`lower_common_inst_line_with`], so the emitter must own it and
    /// pass it on every instruction.
    pub gep_elements: HashMap<llvm_ir::ValueId, GepLink>,
    /// Geps whose only use is the load or store reading through them, with the
    /// base and indices to spell at that use.
    ///
    /// A pointer target writes `base[i]` on the spot; materializing the pointer
    /// first is a declaration and a line for nothing. Filled for a target with
    /// addresses only, and empty for a pointer-free one.
    pub folded_geps: HashMap<llvm_ir::ValueId, (llvm_ir::Operand, Vec<llvm_ir::Operand>)>,
    /// Values the traversal actually wrote an assignment for.
    ///
    /// The declaration pass runs before the body, so it cannot see which values
    /// the structured lowering will fold away — a loop header's condition, a
    /// gep a load folded through. The body records them here and the held-back
    /// declarations are filtered by it.
    assigned_values: HashSet<llvm_ir::ValueId>,
    /// Declarations held back while the body emits, filtered, then written.
    deferred_decls: Vec<(llvm_ir::ValueId, llvm_ir::Type)>,
    defer_decls: bool,
    pub indent: usize,
    out: String,
}

/// The two extents of the storage a tile pointer addresses, when it addresses
/// a tile at all.
pub fn tile_storage_dims(ty: &llvm_ir::Type) -> Option<(usize, usize)> {
    let llvm_ir::Type::Ptr { pointee, .. } = ty else {
        return None;
    };
    let llvm_ir::Type::Array { len: rows, elem } = pointee.as_ref() else {
        return None;
    };
    let llvm_ir::Type::Array { len: cols, .. } = elem.as_ref() else {
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
pub fn copy_assign(ty: &llvm_ir::Type, name: &str, arg: &str) -> String {
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
pub fn collect_storage_values(func: &llvm_ir::Function) -> HashSet<llvm_ir::ValueId> {
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
                llvm_ir::InstOp::Intrinsic {
                    intrinsic:
                        llvm_ir::Intrinsic::TileReduce { .. } | llvm_ir::Intrinsic::TileMma { .. },
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
pub fn collect_alloca_results(func: &llvm_ir::Function) -> HashSet<llvm_ir::ValueId> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match inst.op {
            llvm_ir::InstOp::Alloca { .. } => inst.result,
            _ => None,
        })
        .collect()
}

/// The values one instruction reads.
pub fn inst_operands(inst: &llvm_ir::Inst) -> Vec<&llvm_ir::Operand> {
    match &inst.op {
        llvm_ir::InstOp::Gep { base, indices } => {
            let mut all = vec![base];
            all.extend(indices.iter());
            all
        }
        llvm_ir::InstOp::Load { ptr } => vec![ptr],
        llvm_ir::InstOp::Store { ptr, value } => vec![ptr, value],
        llvm_ir::InstOp::AtomicAdd { ptr, value } => vec![ptr, value],
        llvm_ir::InstOp::Memcpy { dst, src, size } => vec![dst, src, size],
        llvm_ir::InstOp::Bin { lhs, rhs, .. } => vec![lhs, rhs],
        llvm_ir::InstOp::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
        llvm_ir::InstOp::Cast { value, .. } => vec![value],
        llvm_ir::InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond, on_true, on_false],
        llvm_ir::InstOp::Call { args, .. } | llvm_ir::InstOp::Intrinsic { args, .. } => {
            args.iter().collect()
        }
        llvm_ir::InstOp::Alloca { .. } => Vec::new(),
    }
}

/// The values a terminator reads.
pub fn terminator_operands(term: &llvm_ir::Terminator) -> Vec<&llvm_ir::Operand> {
    match term {
        llvm_ir::Terminator::Br { args, .. } => args.iter().collect(),
        llvm_ir::Terminator::CondBr {
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
        llvm_ir::Terminator::Switch { value, .. } => vec![value],
        llvm_ir::Terminator::Ret { value } => value.iter().collect(),
    }
}

/// The geps whose only use is the load or store reading through them.
///
/// A pointer target spells that access as `base[i]` where the value is used,
/// so the gep itself leaves no text. Anything else — a pointer that lives
/// across statements, one gep feeding two accesses — stays a value.
pub fn collect_folded_geps(
    func: &llvm_ir::Function,
) -> HashMap<llvm_ir::ValueId, (llvm_ir::Operand, Vec<llvm_ir::Operand>)> {
    let mut uses: HashMap<llvm_ir::ValueId, usize> = HashMap::new();
    let mut users: HashMap<llvm_ir::ValueId, &llvm_ir::Inst> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let mut note = |used: &llvm_ir::Operand| {
                if let llvm_ir::Operand::Value(id) = used {
                    *uses.entry(*id).or_insert(0) += 1;
                    users.insert(*id, inst);
                }
            };
            for used in inst_operands(inst) {
                note(used);
            }
        }
        for used in terminator_operands(&block.terminator) {
            if let llvm_ir::Operand::Value(id) = used {
                *uses.entry(*id).or_insert(0) += 1;
            }
        }
    }
    let mut folded = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let llvm_ir::InstOp::Gep { base, indices } = &inst.op else {
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
                Some(llvm_ir::InstOp::Load { .. } | llvm_ir::InstOp::Store { .. })
            ) {
                folded.insert(result, (base.clone(), indices.clone()));
            }
        }
    }
    folded
}

impl<'a, D: TargetDialect> TargetEmitter<'a, D> {
    pub fn new(func: &'a llvm_ir::Function, dialect: D) -> Self {
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
        let value_names = build_value_names(func)
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
            .collect();
        Self {
            func,
            dialect,
            storage_values: collect_storage_values(func),
            value_names,
            declared_values: HashSet::new(),
            gep_elements,
            folded_geps,
            assigned_values: HashSet::new(),
            deferred_decls: Vec::new(),
            defer_decls: false,
            indent: 0,
            out: String::new(),
        }
    }

    fn param_names(&self) -> HashMap<llvm_ir::ValueId, String> {
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
        let mut seen: Vec<llvm_ir::ValueId> = Vec::new();
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
            let block_param = self
                .func
                .blocks
                .iter()
                .any(|block| block.params.iter().any(|param| param.id == id));
            if assigned || storage || block_param {
                self.write_decl(id, &ty);
            }
        }
    }

    /// The declaration text for one value.
    fn write_decl(&mut self, id: llvm_ir::ValueId, ty: &llvm_ir::Type) {
        let name = self.value_name(id);
        let decl = self
            .dialect
            .var_decl(ty, &name, self.storage_values.contains(&id));
        if let Some(decl) = decl {
            self.writeln(&decl);
        }
    }

    pub fn value_name(&self, id: llvm_ir::ValueId) -> String {
        value_name(&self.value_names, id)
    }

    pub fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
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
        for line in self.dialect.prelude_lines() {
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
        if self.emit_structured_from(self.func.entry, &[], &post_dominators)?.is_some() {
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
    fn float_literal(value: f64) -> String {
        D::float_literal(value)
    }

    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> Result<()> {
        self.emit_block_insts_except(block, None)
    }

    fn emit_block_insts_except(
        &mut self,
        block: &llvm_ir::BasicBlock,
        skip_result: Option<llvm_ir::ValueId>,
    ) -> Result<()> {
        for inst in &block.insts {
            if let Some(skip_result) = skip_result
                && inst.result == Some(skip_result)
            {
                continue;
            }
            self.emit_inst(inst)?;
        }
        Ok(())
    }

    fn emit_block_param_assignments_skipping(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
        skip_params: &[llvm_ir::ValueId],
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

    fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
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
        self.dialect.vectorize_hint(is_simple_body)
    }
}

impl<D: TargetDialect> TargetEmitter<'_, D> {
    fn emit_structured_from(
        &mut self,
        start: llvm_ir::BlockId,
        stop_targets: &[llvm_ir::BlockId],
        post_dominators: &OnceCell<PostDominators>,
    ) -> Result<Option<llvm_ir::BlockId>> {
        emit_structured_from(
            self,
            self.func,
            start,
            stop_targets,
            None,
            None,
            post_dominators,
            self.dialect.messages(),
        )
    }

    /// One instruction: shared lowering first, dialect second.
    ///
    /// The shared line covers every operator both targets spell the same way,
    /// which is most of the instruction set. A dialect only handles what it
    /// spells differently, so the two emitters cannot drift apart in silence.
    fn emit_inst(&mut self, inst: &llvm_ir::Inst) -> Result<()> {
        if let llvm_ir::InstOp::Intrinsic { intrinsic, args } = &inst.op {
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
    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> Result<()>;
    fn emit_block_insts_except(
        &mut self,
        block: &llvm_ir::BasicBlock,
        skip_result: Option<llvm_ir::ValueId>,
    ) -> Result<()>;
    fn emit_block_param_assignments_skipping(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
        skip_params: &[llvm_ir::ValueId],
    );
    fn emit_block_param_assignments(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
    ) {
        self.emit_block_param_assignments_skipping(target, args, &[]);
    }
    fn format_operand(&self, operand: &llvm_ir::Operand) -> String;

    /// How this target spells a floating-point constant, for the branch
    /// conditions this emitter formats outside its own name table.
    fn float_literal(value: f64) -> String
    where
        Self: Sized,
    {
        float_literal(value)
    }

    fn writeln(&mut self, line: &str);
    fn indent_inc(&mut self);
    fn indent_dec(&mut self);

    /// A loop-carried hint the target wants before a simple loop, if any.
    fn vectorize_hint(&mut self, is_simple_body: bool) -> Option<String>;
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

pub fn emit_structured_from<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    start: llvm_ir::BlockId,
    stop_targets: &[llvm_ir::BlockId],
    stop_skip_params: Option<(llvm_ir::BlockId, Vec<llvm_ir::ValueId>)>,
    stop_skip_inst: Option<llvm_ir::ValueId>,
    post_dominators: &OnceCell<PostDominators>,
    messages: StructuredCfgMessages,
) -> Result<Option<llvm_ir::BlockId>> {
    let mut current = start;
    loop {
        let block = get_block(func, current)
            .cloned()
            .ok_or_else(|| Error::Compile(format!("missing LLVM IR block {:?}", current)))?;

        if let llvm_ir::Terminator::Br { target, args } = &block.terminator
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
                skip_param_ids_for_target(stop_skip_params.as_ref(), *target),
            );
            return Ok(Some(*target));
        }

        if is_loop_preheader(func, &block) {
            current = emit_structured_loop_preheader(
                emitter,
                func,
                &block,
                post_dominators,
                messages,
            )?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if is_loop_header(func, &block) {
            current = emit_structured_loop_header(emitter, func, &block, post_dominators, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if matches!(block.terminator, llvm_ir::Terminator::CondBr { .. }) {
            current =
                emit_structured_if_else(
                    emitter,
                    func,
                    &block,
                    stop_skip_params.clone(),
                    post_dominators,
                    messages,
                )?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        emitter.emit_block_insts(&block)?;
        match &block.terminator {
            llvm_ir::Terminator::Br { target, args } => {
                emitter.emit_block_param_assignments_skipping(
                    *target,
                    args,
                    skip_param_ids_for_target(stop_skip_params.as_ref(), *target),
                );
                if stop_targets.contains(target) {
                    return Ok(Some(*target));
                }
                current = *target;
            }
            llvm_ir::Terminator::Ret { value: None } => {
                emitter.writeln("return;");
                return Ok(None);
            }
            llvm_ir::Terminator::Ret { value: Some(value) } => {
                emitter.writeln(&format!("return {};", emitter.format_operand(value)));
                return Ok(None);
            }
            llvm_ir::Terminator::CondBr { .. } => {
                return Err(Error::Compile(messages.unsupported_cond_br.into()));
            }
            llvm_ir::Terminator::Switch { .. } => {
                return Err(Error::Compile(messages.unsupported_switch.into()));
            }
        }
    }
}

fn emit_structured_loop_preheader<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    preheader: &llvm_ir::BasicBlock,
    post_dominators: &OnceCell<PostDominators>,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::Br {
        target: header_id,
        args: header_args,
    } = &preheader.terminator
    else {
        return Err(Error::Compile(messages.preheader_must_branch.into()));
    };
    let header = get_block(func, *header_id)
        .cloned()
        .ok_or_else(|| Error::Compile(messages.missing_loop_header.into()))?;
    let llvm_ir::Terminator::CondBr {
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
        post_dominators,
        messages,
    )
}

fn emit_structured_loop_header<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    post_dominators: &OnceCell<PostDominators>,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::CondBr {
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
        post_dominators,
        messages,
    )
}

fn emit_structured_if_else<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    block: &llvm_ir::BasicBlock,
    stop_skip_params: Option<(llvm_ir::BlockId, Vec<llvm_ir::ValueId>)>,
    post_dominators: &OnceCell<PostDominators>,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::CondBr {
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

    let (cond_expr, skip_result) = branch_condition_expr::<E>(func, block, cond, false);
    emitter.emit_block_insts_except(block, skip_result)?;
    emitter.writeln(&format!("if ({cond_expr}) {{"));
    emitter.indent_inc();
    emitter.emit_block_param_assignments(*true_target, true_args);
    // An arm may end in a `return`, which leaves it no edge to the join —
    // the arm's text is complete either way, and the join still holds for
    // the arm that reaches it. The skips pass through because an arm can
    // also end on the enclosing loop's header — a `continue` — and a `for`
    // writes its induction step itself, so the arm must not write it too.
    emit_structured_from(
        emitter,
        func,
        *true_target,
        &[join],
        stop_skip_params.clone(),
        None,
        post_dominators,
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
        &[join],
        stop_skip_params,
        None,
        post_dominators,
        messages,
    )?;
    emitter.indent_dec();
    emitter.writeln("}");

    Ok(join)
}

#[allow(clippy::too_many_arguments)]
fn emit_structured_loop_header_impl<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    incoming_header_args: Option<&[llvm_ir::Operand]>,
    cond: &llvm_ir::Operand,
    true_target: llvm_ir::BlockId,
    true_args: &[llvm_ir::Operand],
    false_target: llvm_ir::BlockId,
    false_args: &[llvm_ir::Operand],
    post_dominators: &OnceCell<PostDominators>,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let optimized_loop = branch_condition_expr::<E>(func, header, cond, true);
    match optimized_loop {
        (cond_expr, skip_result @ Some(_))
            if header_only_contains_skipped_cmp(header, skip_result) =>
        {
            if let Some(for_loop) =
                detect_for_loop_induction(func, header, cond, true_target, true_args)
            {
                let induction_name =
                    emitter.format_operand(&llvm_ir::Operand::Value(for_loop.induction_param));
                if let Some(args) = incoming_header_args {
                    emitter.emit_block_param_assignments_skipping(
                        header.id,
                        args,
                        &[for_loop.induction_param],
                    );
                }
                let init_expr = incoming_header_args
                    .and_then(|args| args.get(for_loop.induction_param_index))
                    .map(|arg| format!("{induction_name} = {}", emitter.format_operand(arg)))
                    .unwrap_or_default();
                let step_expr = induction_step_expr(&induction_name, for_loop.step);
                // Only C has this directive, and a target without one keeps
                // the loop as it is.
                //
                // OpenMP requires the loop's initializer to be `var = init`
                // in the `for` header. When the preheader is inside a branch,
                // the induction value is assigned before the loop and the
                // header is empty, so the directive would make the C compiler
                // reject the whole file. The loop is still correct without it.
                let vectorize = (!init_expr.is_empty())
                    .then(|| {
                        emitter.vectorize_hint(is_simple_loop_body(func, true_target, header.id))
                    })
                    .flatten();
                if let Some(vectorize) = vectorize {
                    emitter.writeln(&vectorize);
                }
                emitter.writeln(&format!(
                    "for ({}; {}; {}) {{",
                    init_expr, cond_expr, step_expr
                ));
                emitter.indent_inc();
                emitter.emit_block_param_assignments(true_target, true_args);
                let backedge = emit_structured_from(
                    emitter,
                    func,
                    true_target,
                    &[header.id],
                    Some((header.id, vec![for_loop.induction_param])),
                    for_loop.step_value_id,
                    post_dominators,
                    messages,
                )?;
                if backedge != Some(header.id) {
                    return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
                }
                emitter.indent_dec();
                emitter.writeln("}");
                emitter.emit_block_param_assignments(false_target, false_args);
                return Ok(false_target);
            }

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
                &[header.id],
                None,
                None,
                post_dominators,
                messages,
            )?;
            if backedge != Some(header.id) {
                return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
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
                &[header.id],
                None,
                None,
                post_dominators,
                messages,
            )?;
            if backedge != Some(header.id) {
                return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
            }
            emitter.indent_dec();
            emitter.writeln("}");

            Ok(false_target)
        }
    }
}

#[derive(Clone, Copy)]
struct ForLoopInduction {
    induction_param: llvm_ir::ValueId,
    induction_param_index: usize,
    step: i64,
    step_value_id: Option<llvm_ir::ValueId>,
}

fn detect_for_loop_induction(
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    cond: &llvm_ir::Operand,
    true_target: llvm_ir::BlockId,
    _true_args: &[llvm_ir::Operand],
) -> Option<ForLoopInduction> {
    let cond_id = operand_value_id(cond)?;
    let cmp_inst = get_inst_by_result(func, cond_id)?;
    let llvm_ir::InstOp::Cmp { pred, lhs, rhs } = &cmp_inst.op else {
        return None;
    };
    let llvm_ir::Operand::Value(loop_param) = lhs else {
        return None;
    };
    if !matches!(
        pred,
        llvm_ir::CmpPred::Slt
            | llvm_ir::CmpPred::Sle
            | llvm_ir::CmpPred::Sgt
            | llvm_ir::CmpPred::Sge
    ) {
        return None;
    }
    let induction_param_index = header
        .params
        .iter()
        .position(|param| param.id == *loop_param)?;
    let backedge_candidates = find_loop_backedge_args(func, header.id, true_target);
    let (step, step_value_id) = backedge_candidates.into_iter().find_map(|args| {
        induction_step_from_backedge_arg(func, *loop_param, args.get(induction_param_index)?)
    })?;
    if step == 0 {
        return None;
    }
    if !matches!(
        (step > 0, pred),
        (true, llvm_ir::CmpPred::Slt | llvm_ir::CmpPred::Sle)
            | (false, llvm_ir::CmpPred::Sgt | llvm_ir::CmpPred::Sge)
    ) {
        return None;
    }
    if !matches!(rhs, llvm_ir::Operand::Const(_) | llvm_ir::Operand::Value(_)) {
        return None;
    }
    Some(ForLoopInduction {
        induction_param: *loop_param,
        induction_param_index,
        step,
        step_value_id,
    })
}

fn find_loop_backedge_args(
    func: &llvm_ir::Function,
    header_id: llvm_ir::BlockId,
    true_target: llvm_ir::BlockId,
) -> Vec<Vec<llvm_ir::Operand>> {
    func.blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            llvm_ir::Terminator::Br { target, args }
                if *target == header_id
                    && block_reaches(func, true_target, block.id, &mut HashSet::new()) =>
            {
                Some(args.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

fn induction_step_from_backedge_arg(
    func: &llvm_ir::Function,
    loop_param: llvm_ir::ValueId,
    arg: &llvm_ir::Operand,
) -> Option<(i64, Option<llvm_ir::ValueId>)> {
    let llvm_ir::Operand::Value(step_value_id) = arg else {
        return None;
    };
    let step_inst = get_inst_by_result(func, *step_value_id)?;
    let llvm_ir::InstOp::Bin { op, lhs, rhs } = &step_inst.op else {
        return None;
    };
    let step_val = *step_value_id;
    match (op, lhs, rhs) {
        (
            llvm_ir::BinOp::Add,
            llvm_ir::Operand::Value(id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
        ) if *id == loop_param => Some((*step, Some(step_val))),
        (
            llvm_ir::BinOp::Add,
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
            llvm_ir::Operand::Value(id),
        ) if *id == loop_param => Some((*step, Some(step_val))),
        (
            llvm_ir::BinOp::Sub,
            llvm_ir::Operand::Value(id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
        ) if *id == loop_param => Some((-*step, Some(step_val))),
        _ => None,
    }
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
    func: &llvm_ir::Function,
    body_start: llvm_ir::BlockId,
    header_id: llvm_ir::BlockId,
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
        for inst in &block.insts {
            // A call, and a float function the target may implement as one,
            // keep the body out of the vectorized form.
            let calls_a_function = match &inst.op {
                llvm_ir::InstOp::Call { .. } => true,
                llvm_ir::InstOp::Intrinsic { intrinsic, .. } => {
                    intrinsic.float_function().is_some()
                        || intrinsic.binary_float_function().is_some()
                        || intrinsic.ternary_float_function().is_some()
                }
                _ => false,
            };
            if calls_a_function {
                return false;
            }
        }
        match &block.terminator {
            llvm_ir::Terminator::Br { target, .. } if *target == header_id => return true,
            llvm_ir::Terminator::Br { target, .. } => current = *target,
            _ => return false,
        }
    }
}

fn skip_param_ids_for_target<'a>(
    skip: Option<&'a (llvm_ir::BlockId, Vec<llvm_ir::ValueId>)>,
    target: llvm_ir::BlockId,
) -> &'a [llvm_ir::ValueId] {
    if let Some((skip_target, ids)) = skip
        && *skip_target == target
    {
        return ids;
    }
    &[]
}

fn get_block(func: &llvm_ir::Function, id: llvm_ir::BlockId) -> Option<&llvm_ir::BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn get_inst_by_result(func: &llvm_ir::Function, id: llvm_ir::ValueId) -> Option<&llvm_ir::Inst> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(id))
}

fn is_loop_preheader(func: &llvm_ir::Function, block: &llvm_ir::BasicBlock) -> bool {
    let llvm_ir::Terminator::Br { target: header, .. } = block.terminator else {
        return false;
    };
    get_block(func, header)
        .map(|header| is_loop_header(func, header))
        .unwrap_or(false)
}

fn is_loop_header(func: &llvm_ir::Function, block: &llvm_ir::BasicBlock) -> bool {
    let llvm_ir::Terminator::CondBr {
        true_target,
        false_target,
        ..
    } = block.terminator
    else {
        return false;
    };
    predecessor_count(func, block.id) > 1
        && (block_reaches(func, true_target, block.id, &mut HashSet::new())
            || block_reaches(func, false_target, block.id, &mut HashSet::new()))
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
    func: &llvm_ir::Function,
    block_id: llvm_ir::BlockId,
    post_dominators: &OnceCell<PostDominators>,
) -> Option<llvm_ir::BlockId> {
    if let Some(join) = branch_join_post_dominator(func, block_id, post_dominators) {
        return Some(join);
    }
    // A branch whose arm returns has a join that does not post-dominate it:
    // the path through the returning arm never reaches the join, so the set
    // arithmetic above finds nothing. The join is still readable off the
    // arms — an arm that continues names it, and an arm that returns names
    // nothing, so the continuing arm's target is the join. Two continuing
    // arms must name the same one, or the CFG is not structured.
    let llvm_ir::Terminator::CondBr {
        true_target,
        false_target,
        ..
    } = &get_block(func, block_id)?.terminator
    else {
        return None;
    };
    let arm_target = |id: &llvm_ir::BlockId| {
        get_block(func, *id).and_then(|arm| match &arm.terminator {
            llvm_ir::Terminator::Br { target, .. } => Some(*target),
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
    func: &llvm_ir::Function,
    block_id: llvm_ir::BlockId,
    post_dominators: &OnceCell<PostDominators>,
) -> Option<llvm_ir::BlockId> {
    let post_dominators = post_dominators.get_or_init(|| compute_post_dominators(func));
    // A block closer to the branch post-dominates a farther one, so it has the
    // larger set: the join is the closest, which is the largest.
    let mut best: Option<(llvm_ir::BlockId, usize)> = None;
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
type PostDominators = HashMap<llvm_ir::BlockId, HashSet<llvm_ir::BlockId>>;

fn compute_post_dominators(
    func: &llvm_ir::Function,
) -> HashMap<llvm_ir::BlockId, HashSet<llvm_ir::BlockId>> {
    let __t = std::time::Instant::now();
    let __blocks = func.blocks.len();
    let __r = compute_post_dominators_inner(func);
    if std::env::var_os("SILE_PD_PROFILE").is_some() {
        eprintln!("[pd] blocks={__blocks} compute={}ms", __t.elapsed().as_millis());
    }
    return __r;
}

fn compute_post_dominators_inner(
    func: &llvm_ir::Function,
) -> HashMap<llvm_ir::BlockId, HashSet<llvm_ir::BlockId>> {
    let returns =
        |block: &llvm_ir::BasicBlock| matches!(block.terminator, llvm_ir::Terminator::Ret { .. });
    if !func.blocks.iter().any(returns) {
        // Nothing returns, so no path ends and no block post-dominates another.
        return func
            .blocks
            .iter()
            .map(|block| (block.id, HashSet::from([block.id])))
            .collect();
    }

    let all: HashSet<llvm_ir::BlockId> = func.blocks.iter().map(|block| block.id).collect();
    let mut post_dominators: HashMap<llvm_ir::BlockId, HashSet<llvm_ir::BlockId>> = func
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
            let mut new_set: Option<HashSet<llvm_ir::BlockId>> = None;
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

fn terminator_successors(terminator: &llvm_ir::Terminator) -> Vec<llvm_ir::BlockId> {
    match terminator {
        llvm_ir::Terminator::Br { target, .. } => vec![*target],
        llvm_ir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        llvm_ir::Terminator::Switch { default, cases, .. } => {
            let mut targets = vec![*default];
            targets.extend(cases.iter().map(|(_, target)| *target));
            targets
        }
        llvm_ir::Terminator::Ret { .. } => Vec::new(),
    }
}

fn block_reaches(
    func: &llvm_ir::Function,
    start: llvm_ir::BlockId,
    goal: llvm_ir::BlockId,
    visiting: &mut HashSet<llvm_ir::BlockId>,
) -> bool {
    if start == goal {
        return true;
    }
    if !visiting.insert(start) {
        return false;
    }
    let reaches = match get_block(func, start).map(|block| &block.terminator) {
        Some(llvm_ir::Terminator::Br { target, .. }) => {
            block_reaches(func, *target, goal, visiting)
        }
        Some(llvm_ir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        }) => {
            block_reaches(func, *true_target, goal, visiting)
                || block_reaches(func, *false_target, goal, visiting)
        }
        _ => false,
    };
    visiting.remove(&start);
    reaches
}

fn branch_condition_expr<E: StructuredEmitter>(
    func: &llvm_ir::Function,
    block: &llvm_ir::BasicBlock,
    cond: &llvm_ir::Operand,
    require_header_only: bool,
) -> (String, Option<llvm_ir::ValueId>) {
    let Some(cond_id) = operand_value_id(cond) else {
        return (format_operand_with_function::<E>(func, cond), None);
    };
    let Some(inst) = get_inst_by_result(func, cond_id) else {
        return (format_operand_with_function::<E>(func, cond), None);
    };
    if !matches!(inst.op, llvm_ir::InstOp::Cmp { .. }) {
        return (format_operand_with_function::<E>(func, cond), None);
    }
    if require_header_only && !header_only_contains_skipped_cmp(block, Some(cond_id)) {
        return (format_operand_with_function::<E>(func, cond), None);
    }
    match &inst.op {
        llvm_ir::InstOp::Cmp { pred, lhs, rhs } => (
            format!(
                "{} {} {}",
                format_operand_with_function::<E>(func, lhs),
                cmp_pred_symbol(*pred),
                format_operand_with_function::<E>(func, rhs)
            ),
            Some(cond_id),
        ),
        _ => (format_operand_with_function::<E>(func, cond), None),
    }
}

fn header_only_contains_skipped_cmp(
    block: &llvm_ir::BasicBlock,
    skip_result: Option<llvm_ir::ValueId>,
) -> bool {
    let Some(skip_result) = skip_result else {
        return false;
    };
    block
        .insts
        .iter()
        .all(|inst| inst.result == Some(skip_result))
}

fn operand_value_id(operand: &llvm_ir::Operand) -> Option<llvm_ir::ValueId> {
    match operand {
        llvm_ir::Operand::Value(id) => Some(*id),
        llvm_ir::Operand::Const(_) => None,
    }
}

fn format_operand_with_function<E: StructuredEmitter>(
    func: &llvm_ir::Function,
    operand: &llvm_ir::Operand,
) -> String {
    let value_names = build_value_names(func);
    format_operand(&value_names, E::float_literal, operand)
}

fn predecessor_count(func: &llvm_ir::Function, target: llvm_ir::BlockId) -> usize {
    func.blocks
        .iter()
        .map(|block| match &block.terminator {
            llvm_ir::Terminator::Br { target: succ, .. } => usize::from(*succ == target),
            llvm_ir::Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => usize::from(*true_target == target) + usize::from(*false_target == target),
            llvm_ir::Terminator::Ret { .. } | llvm_ir::Terminator::Switch { .. } => 0,
        })
        .sum()
}
