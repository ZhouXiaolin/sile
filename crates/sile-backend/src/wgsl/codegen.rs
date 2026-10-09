use crate::emit::{
    self, NO_POINTERS, OutputBuffer, StructuredCfgMessages, TargetDialect, find_output_buffer,
};
use sile_core::Result;
use sile_llvm_ir as llvm_ir;

pub fn generate(func: &llvm_ir::Function) -> Result<String> {
    let dialect = WgslDialect::new(func)?;
    emit::generate_text(emit::TargetEmitter::new(func, dialect))
}

/// The WGSL dialect.
///
/// WGSPL takes no pointer parameters: a kernel buffer is an array bound to a
/// `@binding`, and a private local is a `var`. That constraint is what most of
/// this file is about. Everything else is spelling, which is the payoff of the
/// shared emitter: this is the whole third backend.
struct WgslDialect<'a> {
    func: &'a llvm_ir::Function,
    output: Option<OutputBuffer<'a>>,
}

impl<'a> WgslDialect<'a> {
    fn new(func: &'a llvm_ir::Function) -> Result<Self> {
        Ok(Self {
            func,
            output: find_output_buffer(func),
        })
    }

    /// Binding index of one parameter.
    ///
    /// Kernel buffers bind from zero in parameter order, and the shape
    /// descriptor binds after all of them, so a kernel's bindings stay stable
    /// however many shapes it carries.
    fn binding_index(&self, param: &llvm_ir::Param) -> usize {
        if emit::is_shapes_param(param) {
            return self
                .func
                .params
                .iter()
                .filter(|other| !emit::is_shapes_param(other))
                .count();
        }
        self.func
            .params
            .iter()
            .filter(|other| !emit::is_shapes_param(other))
            .position(|other| other.id == param.id)
            .unwrap_or(0)
    }

    /// `array<f32>` with the address space WGSL requires for this parameter.
    ///
    /// A read-only buffer is `read`, a mutable one is `read_write`, and the
    /// shape descriptor is read-only.
    fn storage_decl(&self, param: &llvm_ir::Param) -> String {
        let access = if param.name == emit::SHAPES_PARAM_NAME {
            "read"
        } else {
            "read_write"
        };
        let elem = match &param.ty {
            llvm_ir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
            _ => "f32".to_string(),
        };
        format!("var<storage, {access}> {}: array<{elem}>;", param.name)
    }

    /// Program index along `dim`.
    ///
    /// WGSL exposes only the global invocation id, so all three axes come from
    /// the one vector.
    fn block_id_expr(&self, dim: u8) -> String {
        let axis = match dim {
            0 => "x",
            1 => "y",
            _ => "z",
        };
        format!("i64(global_invocation_id.{axis})")
    }

    /// Grid extent along `dim`, reconstructed from the output buffer's shape.
    ///
    /// One workgroup item owns one tile, so axis 0 spans the output's tile count
    /// and the other axes stay one deep.
    fn grid_dim_expr(&self, dim: u8) -> Result<String> {
        if dim != 0 {
            return Ok("1".to_string());
        }
        let Some(output) = self.output.as_ref() else {
            return Ok("i64(global_invocation_id.count)".to_string());
        };
        let abi = output.param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile("missing output ABI for WGSL grid_dim".into())
        })?;
        let offset = abi.shape_offset;
        if abi.rank <= 1 {
            return Ok(format!(
                "({SHAPES}[{offset}] / {})",
                output.cols.max(1),
                SHAPES = emit::SHAPES_PARAM_NAME
            ));
        }
        Ok(format!(
            "(({SHAPES}[{offset}] / {rows}) * ({SHAPES}[{}] / {cols}))",
            offset + 1,
            SHAPES = emit::SHAPES_PARAM_NAME,
            rows = output.rows.max(1),
            cols = output.cols.max(1)
        ))
    }
}

/// WGSL name of one LLVM IR type.
fn wgsl_type(ty: &llvm_ir::Type) -> String {
    match ty {
        llvm_ir::Type::Void => "void".to_string(),
        llvm_ir::Type::F16 => "f16".to_string(),
        llvm_ir::Type::I1 => "bool".to_string(),
        llvm_ir::Type::I32 => "i32".to_string(),
        llvm_ir::Type::I64 => "i64".to_string(),
        llvm_ir::Type::F32 => "f32".to_string(),
        llvm_ir::Type::F64 => "f64".to_string(),
        // WGSL arrays are one-dimensional, so a nested array becomes the flat
        // element type. A private alloca whose type is a nested array is
        // spelled by `var_decl` through this same path.
        llvm_ir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llvm_ir::Type::Array { elem, .. } => format!("array<{}>", wgsl_type(elem)),
            other => wgsl_type(other),
        },
        llvm_ir::Type::Array { elem, .. } => format!("array<{}>", wgsl_type(elem)),
        llvm_ir::Type::Vector { elem, len } => format!("vec{}<{}>", len, wgsl_type(elem)),
    }
}

impl TargetDialect for WgslDialect<'_> {
    fn var_decl(
        &self,
        ty: &llvm_ir::Type,
        name: &str,
        _is_alloca: bool,
    ) -> Option<String> {
        Some(format!("var {}: {};", name, wgsl_type(ty)))
    }

    fn param_decl(&self, ty: &llvm_ir::Type, _name: &str) -> String {
        // A buffer parameter contributes a module-level binding, not a
        // parameter list entry. See `prelude_lines`.
        let _ = ty;
        String::new()
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        // All buffers arrive as bindings, so the parameter list is empty apart
        // from the launch index the entry point reads itself.
        let _ = params;
        format!("fn sile_kernel_{name}() {{\n")
    }

    fn prelude_lines(&self) -> Vec<String> {
        let mut lines = vec!["@compute @workgroup_size(1)".to_string()];
        // Kernel buffers bind first, then the shapes descriptor, so a kernel's
        // bindings stay stable no matter how many shapes it carries.
        for param in &self.func.params {
            if emit::is_shapes_param(param) {
                continue;
            }
            lines.push(format!(
                "@group(0) @binding({}) {}",
                self.binding_index(param),
                self.storage_decl(param)
            ));
        }
        for param in &self.func.params {
            if emit::is_shapes_param(param) {
                lines.push(format!(
                    "@group(0) @binding({}) {}",
                    self.binding_index(param),
                    self.storage_decl(param)
                ));
            }
        }
        lines
    }

    fn pointers(&self) -> emit::PointerSpelling {
        // WGSL has no pointers: a `load` reads the element and a `gep` yields
        // the element itself, so the shared instruction lowering must not emit
        // `&` or `*`.
        NO_POINTERS
    }

    fn declares_at_use(&self, inst: &llvm_ir::Inst) -> bool {
        // A WGSL intrinsic is a `let` binding, so it must not be declared twice.
        matches!(inst.op, llvm_ir::InstOp::Intrinsic { .. })
    }

    fn vectorize_hint(&self, _is_simple_body: bool) -> Option<String> {
        // WGSL has no pragma, and the driver already vectorizes a simple loop.
        None
    }

    fn intrinsic_stmt(
        &self,
        result: Option<llvm_ir::ValueId>,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
        intrinsic: &llvm_ir::Intrinsic,
        args: &[llvm_ir::Operand],
    ) -> Result<Vec<String>> {
        let name = || {
            result.map(|id| value_name(id)).unwrap_or_else(|| {
                sile_core::Error::Compile("WGSL intrinsic needs a result".into()).to_string()
            })
        };
        let line = match intrinsic {
            llvm_ir::Intrinsic::BlockId { dim } => {
                format!("let {} = {};", name(), self.block_id_expr(*dim))
            }
            llvm_ir::Intrinsic::ThreadId { dim } => {
                format!("let {} = {};", name(), self.block_id_expr(*dim))
            }
            llvm_ir::Intrinsic::GridDim { dim } => {
                format!("let {} = {};", name(), self.grid_dim_expr(*dim)?)
            }
            llvm_ir::Intrinsic::Barrier { .. } => "workgroupBarrier();".to_string(),
            llvm_ir::Intrinsic::Exp => {
                let [arg] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL exp intrinsic expects one argument".into(),
                    ));
                };
                format!("let {} = exp({});", name(), format_operand(arg))
            }
            llvm_ir::Intrinsic::VecLoad { len } => {
                let [ptr, offset] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL vec_load expects [ptr, offset]".into(),
                    ));
                };
                format!(
                    "let {} = vec{}<f32>({} + i64(0) + {});",
                    name(),
                    len,
                    format_operand(ptr),
                    format_operand(offset)
                )
            }
            llvm_ir::Intrinsic::VecStore { len } => {
                let [ptr, offset, value] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL vec_store expects [ptr, offset, value]".into(),
                    ));
                };
                let _ = len;
                format!("{}[{}] = {};", format_operand(ptr), format_operand(offset), format_operand(value))
            }
            llvm_ir::Intrinsic::VecReduceAdd => {
                let [vector] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL vec_reduce_add expects one argument".into(),
                    ));
                };
                format!(
                    "let {} = {v}[0] + {v}[1] + {v}[2] + {v}[3];",
                    name(),
                    v = format_operand(vector)
                )
            }
        };
        Ok(vec![line])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &llvm_ir::Inst,
        _value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
    ) -> Result<Vec<String>> {
        match &inst.op {
            // WGSL has no `memcpy`; a byte copy becomes an element loop.
            llvm_ir::InstOp::Memcpy { dst, src, size } => Ok(vec![
                format!("for (var copy_i: i64 = 0; copy_i < {}; copy_i++) {{", format_operand(size)),
                format!("{}[copy_i] = {}[copy_i];", format_operand(dst), format_operand(src)),
                "}".to_string(),
            ]),
            // A WGSL atomic add works on integers, not floats, so the float
            // buffer is read-modify-written under an atomic compare exchange.
            llvm_ir::InstOp::AtomicAdd { ptr, value } => Ok(vec![format!(
                "atomicAdd(({}), ({}));",
                format_operand(ptr),
                format_operand(value)
            )]),
            llvm_ir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR WGSL codegen does not support call instructions: {func}"
            ))),
            _ => Ok(vec![]),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured WGSL loop preheader must end with a branch",
            missing_loop_header: "missing LLVM IR WGSL loop header",
            header_must_cond_br: "structured WGSL loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured WGSL loop body did not produce the expected backedge",
            unsupported_cond_br: "LLVM IR WGSL codegen only supports structured conditional branches",
            unsupported_switch: "LLVM IR WGSL codegen does not yet support structured switch lowering",
        }
    }
}
