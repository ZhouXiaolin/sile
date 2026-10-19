use crate::emit::{
    self, OutputBuffer, StructuredCfgMessages, TargetDialect, array_dims, find_output_buffer,
    find_output_param,
};
use sile_llvm_ir as llvm_ir;

const SHAPES_PARAM_NAME: &str = "__sile_shapes";

pub fn generate(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let dialect = MetalDialect::new(func)?;
    emit::generate_text(emit::TargetEmitter::new(func, dialect))
}

/// The MSL dialect: one spelling table, plus the tile arithmetic that MSL reads
/// from buffer shapes.
///
/// `grid_dim` and `block_id` divide the flat thread index into tiles, which is
/// what the tile ABI promises the target can reconstruct.
struct MetalDialect<'a> {
    func: &'a llvm_ir::Function,
    output: Option<OutputBuffer<'a>>,
    /// The output even when it stores no tile, so the launch index still has a
    /// place to come from.
    output_param: Option<&'a llvm_ir::Param>,
}

impl<'a> MetalDialect<'a> {
    fn new(func: &'a llvm_ir::Function) -> sile_core::Result<Self> {
        Ok(Self {
            func,
            output: find_output_buffer(func),
            output_param: find_output_param(func),
        })
    }

    fn shapes_param_name(&self) -> sile_core::Result<&str> {
        self.func
            .params
            .iter()
            .find(|param| param.name == SHAPES_PARAM_NAME)
            .map(|param| param.name.as_str())
            .ok_or_else(|| {
                sile_core::Error::Compile(
                    "Metal backend requires explicit __sile_shapes parameter in LLVM IR".into(),
                )
            })
    }

    /// Extent of logical grid axis `dim`.
    ///
    /// Metal dispatches one thread per tile, so the axes are the tile counts
    /// along the output's axes. The count comes from the shape descriptor and
    /// the tile ABI, so no launch-time value is needed.
    fn grid_dim_expr(&self, dim: u8) -> sile_core::Result<String> {
        let Some(output) = self.output.as_ref() else {
            // The output has no tile, so one program owns one grid slot and
            // the grid is the flat launch size.
            return Ok(match dim {
                0 => "(int64_t)gid.x".to_string(),
                _ => "1".to_string(),
            });
        };
        let Some(abi) = output.param.abi.as_ref() else {
            return Err(sile_core::Error::Compile(
                "missing output ABI for Metal grid_dim".into(),
            ));
        };
        let shapes = self.shapes_param_name()?;
        let axes = emit::grid_axis_exprs(abi, output.rows, output.cols, &shapes);
        Ok(axes
            .get(dim as usize)
            .cloned()
            .unwrap_or_else(|| "1".to_string()))
    }

    /// Program index along logical grid axis `dim`.
    ///
    /// One thread owns one tile, so the thread's flat position in the grid
    /// counts tiles with the last axis varying fastest. Dividing by the strides
    /// after `dim` and taking the remainder recovers the index along `dim`,
    /// whether the output has one axis, two or three.
    fn logical_block_id_expr(&self, dim: u8) -> sile_core::Result<String> {
        let index = |dim: u8| -> String {
            match dim {
                0 => "(int64_t)gid.x".into(),
                1 => "(int64_t)gid.y".into(),
                2 => "(int64_t)gid.z".into(),
                _ => unreachable!("guarded by the caller"),
            }
        };
        // A kernel whose output has no tile (an atomic accumulation, for
        // example) dispatches one program per grid slot, so the launch index is
        // the program id. Splitting it by the output's extent would put every
        // program on the same row.
        if self.output_param.is_none() {
            return Ok(index(dim));
        }
        let Some(output) = self.output.as_ref() else {
            // The output stores no tile: it is written one element at a time,
            // so one program owns one grid slot.
            return Ok(match dim {
                0 => index(0),
                1 | 2 => "0".into(),
                _ => index(dim),
            });
        };
        // The axes and the rank come from the buffer that stores the tile, not
        // from the last parameter: a kernel whose last parameter is an input
        // has that input's shape where the output's belongs.
        let Some(abi) = output.param.abi.as_ref() else {
            return Ok(index(dim));
        };
        if abi.rank <= 1 {
            return Ok(match dim {
                0 => index(0),
                1 | 2 => "0".into(),
                _ => {
                    return Err(sile_core::Error::Compile(format!(
                        "unsupported block_id dimension in Metal codegen: {dim}"
                    )));
                }
            });
        }
        let shapes = self.shapes_param_name()?;
        let axes = emit::grid_axis_exprs(abi, output.rows, output.cols, &shapes);
        let Some(extent) = axes.get(dim as usize) else {
            return Ok("0".to_string());
        };
        let stride = emit::grid_axis_stride(&axes, dim as usize);
        Ok(format!("((int64_t)gid.x / ({stride}) % ({extent}))"))
    }
}

impl TargetDialect for MetalDialect<'_> {
    /// A tile is an address here, so a parameter takes a copy of the elements
    /// rather than the address of storage the next iteration overwrites.
    fn param_assign(&self, ty: &llvm_ir::Type, name: &str, arg: &str) -> String {
        emit::copy_assign(ty, name, arg)
    }

    fn var_decl(&self, ty: &llvm_ir::Type, name: &str, _is_alloca: bool) -> Option<String> {
        // MSL has no function-local pointer storage, so a private pointer binds
        // to a `thread` local array. Unlike C, every private pointer behaves the
        // same way here, whether it came from an `alloca` or a `gep`.
        match ty {
            llvm_ir::Type::Ptr {
                addr_space: llvm_ir::AddressSpace::Private,
                pointee,
            } => {
                let storage_name = format!("{name}_storage");
                Some(format!(
                    "{}; {};",
                    metal_storage_decl(pointee, &storage_name),
                    metal_private_ptr_bind_decl(pointee, name, &storage_name)
                ))
            }
            _ => Some(format!("{};", metal_var_decl(ty, name))),
        }
    }

    fn param_decl(&self, ty: &llvm_ir::Type, _name: &str) -> String {
        metal_type(ty)
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        let mut out = format!("kernel void sile_kernel_{name}(\n");
        for (idx, (ty, param_name)) in params.iter().enumerate() {
            out.push_str(&format!("    {ty} {param_name} [[buffer({idx})]],\n"));
        }
        out.push_str("    uint3 gid [[thread_position_in_grid]]) {\n");
        out
    }

    fn prelude_lines(&self) -> Vec<String> {
        vec![
            "#include <metal_stdlib>".to_string(),
            "using namespace metal;".to_string(),
            String::new(),
        ]
    }

    fn vectorize_hint(&self, _is_simple_body: bool) -> Option<String> {
        // MSL lets the driver schedule this loop; a pragma is not available.
        None
    }

    fn intrinsic_stmt(
        &self,
        result: Option<llvm_ir::ValueId>,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
        intrinsic: &llvm_ir::Intrinsic,
        args: &[llvm_ir::Operand],
    ) -> sile_core::Result<Vec<String>> {
        let result = result.ok_or_else(|| {
            sile_core::Error::Compile(format!(
                "{intrinsic:?} in LLVM IR Metal codegen must produce a result"
            ))
        })?;
        let name = || value_name(result);
        let line = match intrinsic {
            llvm_ir::Intrinsic::BlockId { dim } => {
                format!("{} = {};", name(), self.logical_block_id_expr(*dim)?)
            }
            llvm_ir::Intrinsic::ThreadId { dim } => format!(
                "{} = (int64_t)gid.{};",
                name(),
                match dim {
                    0 => "x",
                    1 => "y",
                    _ => "z",
                }
            ),
            llvm_ir::Intrinsic::GridDim { dim } => {
                format!("{} = {};", name(), self.grid_dim_expr(*dim)?)
            }
            llvm_ir::Intrinsic::Barrier { .. } => {
                return Err(sile_core::Error::Compile(
                    "barrier intrinsic is not yet supported in LLVM IR Metal codegen".into(),
                ));
            }
            llvm_ir::Intrinsic::Exp
            | llvm_ir::Intrinsic::Exp2
            | llvm_ir::Intrinsic::Log
            | llvm_ir::Intrinsic::Log2
            | llvm_ir::Intrinsic::Sqrt
            | llvm_ir::Intrinsic::Abs
            | llvm_ir::Intrinsic::Sin
            | llvm_ir::Intrinsic::Cos
            | llvm_ir::Intrinsic::Tan
            | llvm_ir::Intrinsic::Sinh
            | llvm_ir::Intrinsic::Cosh
            | llvm_ir::Intrinsic::Tanh
            | llvm_ir::Intrinsic::Ceil
            | llvm_ir::Intrinsic::Floor => {
                let [arg] = args else {
                    return Err(sile_core::Error::Compile(
                        "a float function takes one argument".into(),
                    ));
                };
                let function = intrinsic
                    .float_function()
                    .expect("a float function has a name");
                format!("{} = metal::{}({});", name(), function, format_operand(arg))
            }
            llvm_ir::Intrinsic::Pow | llvm_ir::Intrinsic::Atan2 => {
                let [lhs, rhs] = args else {
                    return Err(sile_core::Error::Compile(
                        "a binary float function takes two arguments".into(),
                    ));
                };
                let function = intrinsic
                    .binary_float_function()
                    .expect("a binary float function has a name");
                format!(
                    "{} = metal::{}({}, {});",
                    name(),
                    function,
                    format_operand(lhs),
                    format_operand(rhs)
                )
            }
            llvm_ir::Intrinsic::Fma => {
                let [a, b, c] = args else {
                    return Err(sile_core::Error::Compile(
                        "a ternary float function takes three arguments".into(),
                    ));
                };
                let function = intrinsic
                    .ternary_float_function()
                    .expect("a ternary float function has a name");
                format!(
                    "{} = metal::{}({}, {}, {});",
                    name(),
                    function,
                    format_operand(a),
                    format_operand(b),
                    format_operand(c)
                )
            }
            llvm_ir::Intrinsic::VecLoad { len } => {
                let [ptr, offset] = args else {
                    return Err(sile_core::Error::Compile(
                        "vec_load intrinsic expects [ptr, offset] arguments".into(),
                    ));
                };
                format!(
                    "{} = *((device float{}*)({} + {}));",
                    name(),
                    len,
                    format_operand(ptr),
                    format_operand(offset)
                )
            }
            llvm_ir::Intrinsic::VecStore { len } => {
                let [ptr, offset, value] = args else {
                    return Err(sile_core::Error::Compile(
                        "vec_store intrinsic expects [ptr, offset, value] arguments".into(),
                    ));
                };
                format!(
                    "*((device float{}*)({} + {})) = {};",
                    len,
                    format_operand(ptr),
                    format_operand(offset),
                    format_operand(value)
                )
            }
            llvm_ir::Intrinsic::VecReduceAdd => {
                let [vector] = args else {
                    return Err(sile_core::Error::Compile(
                        "vec_reduce_add intrinsic expects one argument".into(),
                    ));
                };
                let v = format_operand(vector);
                format!("{} = {v}[0] + {v}[1] + {v}[2] + {v}[3];", name())
            }
        };
        Ok(vec![line])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &llvm_ir::Inst,
        _value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
    ) -> sile_core::Result<Vec<String>> {
        // MSL has no `memcpy` builtin and no ordinary function calls, so both
        // become explicit loops or an error.
        match &inst.op {
            llvm_ir::InstOp::Memcpy { dst, src, size } => Ok(vec![
                format!(
                    "for (int64_t copy_i = 0; copy_i < {}; ++copy_i) {{",
                    format_operand(size)
                ),
                format!(
                    "{}[copy_i] = {}[copy_i];",
                    format_operand(dst),
                    format_operand(src)
                ),
                "}".to_string(),
            ]),
            llvm_ir::InstOp::AtomicAdd { ptr, value } => Ok(vec![format!(
                "atomic_fetch_add_explicit((device atomic_float*)({}), {}, memory_order_relaxed);",
                format_operand(ptr),
                format_operand(value)
            )]),
            llvm_ir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR Metal codegen does not support call instructions: {func}"
            ))),
            // An intrinsic never reaches here: the shared emitter dispatches it
            // to `intrinsic_stmt` before this match is consulted.
            llvm_ir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LLVM IR Metal codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // An alloca is the declaration the emitter already wrote, and a
            // `gep` is spelled at its use: neither leaves a statement here.
            llvm_ir::InstOp::Alloca { .. } | llvm_ir::InstOp::Gep { .. } => Ok(vec![]),
            // The shared lowering spells these the same way on every target. A
            // result-less operator reaching here would otherwise be dropped,
            // and the kernel would read an undefined name.
            llvm_ir::InstOp::Bin { .. }
            | llvm_ir::InstOp::Cmp { .. }
            | llvm_ir::InstOp::Cast { .. }
            | llvm_ir::InstOp::Load { .. }
            | llvm_ir::InstOp::Select { .. }
            | llvm_ir::InstOp::Store { .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR Metal codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured Metal loop preheader must end with a branch",
            missing_loop_header: "missing LLVM IR structured Metal loop header",
            header_must_cond_br: "structured Metal loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured Metal loop body did not produce the expected backedge",
            unsupported_cond_br: "LLVM IR Metal codegen only supports structured conditional branches",
            unsupported_switch: "LLVM IR Metal codegen does not yet support switch terminators",
        }
    }
}

fn metal_var_decl(ty: &llvm_ir::Type, name: &str) -> String {
    match ty {
        llvm_ir::Type::I1 => format!("bool {} = false", name),
        llvm_ir::Type::I32 => format!("int32_t {} = 0", name),
        llvm_ir::Type::I64 => format!("int64_t {} = 0", name),
        llvm_ir::Type::F32 => format!("float {} = 0.0f", name),
        llvm_ir::Type::F64 => format!("double {} = 0.0", name),
        llvm_ir::Type::Ptr {
            addr_space,
            pointee,
        } => metal_ptr_zero_decl(addr_space, pointee, name),
        llvm_ir::Type::Vector { len, elem: _ } => {
            format!("float{} {} = {{0}}", len, name)
        }
        other => format!("{} {}", metal_type(other), name),
    }
}

fn metal_storage_decl(ty: &llvm_ir::Type, name: &str) -> String {
    match ty {
        llvm_ir::Type::Array { .. } => {
            let dims = array_dims(ty);
            let base = array_base_type(ty);
            format!(
                "{} {}{}",
                base,
                name,
                dims.iter()
                    .map(|dim| format!("[{}]", dim))
                    .collect::<Vec<_>>()
                    .join("")
            )
        }
        other => format!("{} {}", metal_type(other), name),
    }
}

fn metal_private_ptr_bind_decl(pointee: &llvm_ir::Type, name: &str, storage_name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("thread {base}* {name} = &{storage_name};")
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("thread {base} (*{name}){suffix} = {storage_name};")
    }
}

fn metal_ptr_zero_decl(
    addr_space: &llvm_ir::AddressSpace,
    pointee: &llvm_ir::Type,
    name: &str,
) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    let qualifier = metal_addr_qual(addr_space);
    if dims.is_empty() {
        format!("{qualifier}{base}* {name} = nullptr")
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{qualifier}{base} (*{name}){suffix} = nullptr")
    }
}

fn metal_addr_qual(addr_space: &llvm_ir::AddressSpace) -> &'static str {
    match addr_space {
        llvm_ir::AddressSpace::Generic => "",
        llvm_ir::AddressSpace::Private => "thread ",
        llvm_ir::AddressSpace::Global => "device ",
        llvm_ir::AddressSpace::Constant => "constant ",
        llvm_ir::AddressSpace::Shared => "threadgroup ",
    }
}

fn metal_type(ty: &llvm_ir::Type) -> String {
    match ty {
        llvm_ir::Type::Void => "void".into(),
        llvm_ir::Type::I1 => "bool".into(),
        llvm_ir::Type::I32 => "int32_t".into(),
        llvm_ir::Type::I64 => "int64_t".into(),
        llvm_ir::Type::F16 => "half".into(),
        llvm_ir::Type::F32 => "float".into(),
        llvm_ir::Type::F64 => "double".into(),
        llvm_ir::Type::Ptr {
            addr_space,
            pointee,
        } => format!(
            "{}{}*",
            metal_addr_qual(addr_space),
            metal_scalar_type(pointee)
        ),
        llvm_ir::Type::Vector { len, elem } => format!("{}{}", metal_scalar_type(elem), len),
        llvm_ir::Type::Array { len, elem } => format!("{}[{}]", metal_type(elem), len),
    }
}

fn metal_scalar_type(ty: &llvm_ir::Type) -> String {
    match ty {
        llvm_ir::Type::Array { .. } => array_base_type(ty),
        other => metal_type(other),
    }
}

fn array_base_type(ty: &llvm_ir::Type) -> String {
    let mut current = ty;
    while let llvm_ir::Type::Array { elem, .. } = current {
        current = elem;
    }
    metal_type(current)
}
