use crate::emit::{
    self, NO_POINTERS, OutputBuffer, StructuredCfgMessages, TargetDialect, find_output_buffer,
};
use sile_core::Result;
use sile_llvm_ir as llvm_ir;

/// The entry point parameter that carries the launch index.
///
/// WGSL has no implicit launch index: a compute entry point declares one
/// `@builtin(global_invocation_id)` parameter and every axis of the launch grid
/// is a component of it. `workgroup_size(1)` means one workgroup item owns one
/// tile, so the invocation id is the tile index.
const LAUNCH_INDEX: &str = "sile_launch_index";

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
    /// WGSL spelling of the shape descriptor.
    ///
    /// WGSL reserves a leading double underscore, so the shared
    /// `__sile_shapes` name cannot be a WGSL binding. The emitter renames it
    /// through [`TargetDialect::shapes_name`], and this field carries the same
    /// name to the binding declarations and to `grid_dim_expr`.
    shapes_name: String,
}

impl<'a> WgslDialect<'a> {
    fn new(func: &'a llvm_ir::Function) -> Result<Self> {
        Ok(Self {
            func,
            output: find_output_buffer(func),
            shapes_name: "sile_shapes".to_string(),
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
        let access = if emit::is_shapes_param(param) {
            "read"
        } else {
            "read_write"
        };
        let elem = match &param.ty {
            llvm_ir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
            _ => "f32".to_string(),
        };
        let name = if emit::is_shapes_param(param) {
            self.shapes_name.clone()
        } else {
            param.name.clone()
        };
        format!("var<storage, {access}> {name}: array<{elem}>;")
    }

    /// Program index along `dim`.
    ///
    /// A rank-1 output puts one workgroup item on each tile, so the launch
    /// index along x is the tile index directly. A rank-2 output is dispatched
    /// flat, one workgroup item per tile, so the x component carries both axes
    /// and splits into a row count and a column count exactly as in C and MSL.
    fn block_id_expr(&self, dim: u8) -> String {
        let flat = format!("i64({LAUNCH_INDEX}.x)");
        let Some(output) = self.output.as_ref() else {
            return match dim {
                0 => flat,
                _ => "0".to_string(),
            };
        };
        let Some(abi) = output.param.abi.as_ref() else {
            return flat;
        };
        if abi.rank <= 1 {
            return match dim {
                0 => flat,
                _ => "0".to_string(),
            };
        }
        let tiles_n = format!(
            "({shapes}[{}] / {})",
            abi.shape_offset + 1,
            output.cols.max(1),
            shapes = self.shapes_name
        );
        match dim {
            0 => format!("({flat} / {tiles_n})"),
            1 => format!("({flat} % {tiles_n})"),
            _ => "0".to_string(),
        }
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
            return Ok(format!("i64({LAUNCH_INDEX}.x)"));
        };
        let abi = output.param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile("missing output ABI for WGSL grid_dim".into())
        })?;
        let offset = abi.shape_offset;
        if abi.rank <= 1 {
            return Ok(format!(
                "({SHAPES}[{offset}] / {})",
                output.cols.max(1),
                SHAPES = self.shapes_name
            ));
        }
        Ok(format!(
            "(({SHAPES}[{offset}] / {rows}) * ({SHAPES}[{}] / {cols}))",
            offset + 1,
            SHAPES = self.shapes_name,
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
        llvm_ir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
        // WGSL arrays are one-dimensional: a nested array is the flat element
        // type with the total element count. A private alloca that holds a
        // tile is spelled through this same path, and a Gep chain already
        // summed its indices into one element offset, so the flat storage and
        // the flat index agree.
        llvm_ir::Type::Array { .. } => {
            let base = array_base_type(ty);
            let count = array_element_count(ty);
            format!("array<{base}, {count}>")
        }
        llvm_ir::Type::Vector { elem, len } => format!("vec{}<{}>", len, wgsl_type(elem)),
    }
}

/// Element type at the bottom of a (possibly nested) array.
fn array_base_type(ty: &llvm_ir::Type) -> String {
    let mut current = ty;
    while let llvm_ir::Type::Array { elem, .. } = current {
        current = elem;
    }
    wgsl_type(current)
}

/// Total number of elements in a (possibly nested) array.
fn array_element_count(ty: &llvm_ir::Type) -> usize {
    let mut count = 1usize;
    let mut current = ty;
    while let llvm_ir::Type::Array { len, elem } = current {
        count *= *len;
        current = elem;
    }
    count
}

impl TargetDialect for WgslDialect<'_> {
    fn shapes_name(&self) -> String {
        self.shapes_name.clone()
    }

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
        // from the launch index. WGSL names that index in the signature as an
        // `@builtin`, which is the only way a compute entry point reads it.
        //
        // The entry attributes belong here, against the function they decorate.
        // WGSL attributes attach to the declaration that follows them, so an
        // entry attribute printed before the buffer bindings decorates those
        // bindings instead, and the kernel stops being an entry point.
        let _ = params;
        format!(
            "@compute @workgroup_size(1)\n\
             fn sile_kernel_{name}(@builtin(global_invocation_id) {LAUNCH_INDEX}: vec3<u32>) {{\n"
        )
    }

    fn prelude_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
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

    fn declares_intrinsic_at_use(&self, inst: &llvm_ir::Inst) -> bool {
        // A WGSL intrinsic is a `let` binding, so it must not also be declared.
        matches!(inst.op, llvm_ir::InstOp::Intrinsic { .. })
    }

    fn select(&self) -> emit::SelectSpelling {
        // WGSL has no conditional expression; a choice is a call.
        emit::SelectSpelling::Function
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
