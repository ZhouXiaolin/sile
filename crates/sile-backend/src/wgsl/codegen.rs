use std::collections::HashMap;

use crate::emit::{
    self, NO_POINTERS, OutputBuffer, StructuredCfgMessages, TargetDialect, element_operand,
    find_output_buffer,
};
use sile_core::Result;
use sile_llvm_ir as llvm_ir;
use sile_llvm_ir::{Operand, ValueId};

/// The entry point parameter that carries the launch index.
///
/// WGSL has no implicit launch index: a compute entry point declares one
/// `@builtin(global_invocation_id)` parameter and every axis of the launch grid
/// is a component of it. `workgroup_size(1)` means one workgroup item owns one
/// tile, so the invocation id is the tile index.
const LAUNCH_INDEX: &str = "sile_launch_index";

/// WGSL spelling of the scalar descriptor.
///
/// The IR name `__sile_scalars` starts with a double underscore, which WGSL
/// reserves, so the binding is declared under a legal name and the emitter
/// renames the parameter through [`TargetDialect::scalars_name`].
const SCALARS_NAME: &str = "sile_scalars";

/// WGSL spelling of the integer scalar descriptor.
const SCALARS_I64_NAME: &str = "sile_scalars_i64";

/// WGSL spelling of one array subscript.
///
/// WGSL subscripts take `i32` or `u32`, and the lowering makes every index
/// `i64`, so a computed index narrows here. Every other value keeps the width
/// the IR gave it; this is the one place the two disagree.
fn wgsl_index(index: String) -> String {
    format!("i32({index})")
}

pub fn generate(func: &llvm_ir::Function) -> Result<String> {
    let dialect = WgslDialect::new(func)?;
    emit::generate_text(emit::TargetEmitter::new(func, dialect))
}

/// The kernel buffers that carry a float atomic add, in binding order.
///
/// WGSL has no atomic add on a float, so each such buffer is bound a second
/// time as `atomic<i32>` and the add is a compare exchange over the bit
/// pattern. The runtime has to bind the same storage twice, and this tells it
/// which buffers, so the codegen and the runtime cannot disagree about the
/// layout.
pub fn atomic_alias_bindings(func: &llvm_ir::Function) -> Vec<usize> {
    let dialect = WgslDialect::new(func).expect("WGSL dialect over a typed function");
    dialect
        .atomic_buffer_params()
        .iter()
        .map(|param| dialect.binding_index(param))
        .collect()
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
    /// Every Gep in the function, as `(base, indices)`.
    ///
    /// WGSL has no pointers, so a `vec_load` argument that the shared emitter
    /// sees as a pointer is an element here. The rule that spells it lives once
    /// in the shared emitter as `element_operand`, and this table is the input
    /// it reads.
    gep_elements: HashMap<ValueId, emit::GepLink>,
    /// Number of float atomic sites emitted so far.
    ///
    /// Each site names its own compare-exchange locals, and WGSL declares them
    /// in the function scope, so two sites cannot share a name.
    atomic_sites: usize,
    /// WGSL spelling of the shape descriptor.
    ///
    /// WGSL reserves a leading double underscore, so the shared
    /// `__sile_shapes` name cannot be a WGSL binding. The emitter renames it
    /// through [`TargetDialect::shapes_name`], and this field carries the same
    /// name to the binding declarations and to `grid_dim_expr`.
    shapes_name: String,
}

impl<'a> WgslDialect<'a> {
    /// The element an intrinsic's pointer argument names.
    ///
    /// A `gep` result is already an element, so the shared spelling applies.
    /// A kernel buffer argument is not: it is the whole array, and the index
    /// that follows addresses into it. `element_operand` appends `[0]` for a
    /// value with no Gep history, which is right for a `load` of a pointer and
    /// wrong here, so this drops it.
    fn buffer_element(
        &self,
        ptr: &llvm_ir::Operand,
        value_name: &dyn Fn(llvm_ir::ValueId) -> String,
        format_operand: &dyn Fn(&llvm_ir::Operand) -> String,
    ) -> String {
        let spelled = element_operand(
            ptr,
            &self.gep_elements,
            value_name,
            format_operand,
            wgsl_index,
        );
        match ptr {
            llvm_ir::Operand::Value(id) if self.gep_elements.contains_key(id) => spelled,
            _ => format_operand(ptr),
        }
    }

    /// First binding number the atomic aliases occupy.
    ///
    /// They come after every kernel buffer and the shapes descriptor, so a
    /// kernel with no atomic add binds nothing new and the numbers a caller
    /// already passes keep their meaning.
    fn alias_binding_base(&self) -> usize {
        self.func.params.len()
    }

    /// The buffers a float atomic add writes through.
    ///
    /// A kernel that adds nothing binds no alias, so the aliases come from the
    /// instruction stream rather than from the parameter list. The root of the
    /// Gep chain is the buffer, and a chain is already collected once for the
    /// whole function.
    fn atomic_buffer_params(&self) -> Vec<&llvm_ir::Param> {
        let names: Vec<String> = self
            .func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.op {
                llvm_ir::InstOp::AtomicAdd { ptr, .. } => self.atomic_buffer_name(ptr),
                _ => None,
            })
            .collect();
        self.func
            .params
            .iter()
            .filter(|param| names.iter().any(|name| *name == param.name))
            .collect()
    }

    /// Name of the buffer an atomic add's pointer root addresses.
    fn atomic_buffer_name(&self, ptr: &llvm_ir::Operand) -> Option<String> {
        let (root, _) = emit::walk_gep_chain(ptr, &self.gep_elements)?;
        match root {
            llvm_ir::Operand::Value(id) => self
                .func
                .params
                .iter()
                .find(|param| param.id == *id && !emit::is_shapes_param(param))
                .map(|param| param.name.clone()),
            _ => None,
        }
    }

    /// Name this atomic compare-exchange site writes its locals under.
    ///
    /// WGSL declares the compare-exchange operands in the function scope, so
    /// two sites in one kernel cannot share a name. A counter is the cheapest
    /// unique suffix, and it is read in the order the sites are emitted.
    fn next_atomic_cas_name(&mut self) -> String {
        self.atomic_sites += 1;
        format!("sile_atomic_{}", self.atomic_sites)
    }

    /// The `atomic<i32>` binding and index this atomic add operates on.
    ///
    /// WGSL has no atomic add on a float, so the same buffer is bound a second
    /// time as `atomic<i32>` and the add is a compare exchange over the bit
    /// pattern. The pointer operand names the buffer and an element index, and
    /// the alias binding is the same buffer under the integer spelling.
    fn atomic_target(
        &self,
        ptr: &llvm_ir::Operand,
        value_name: impl Fn(ValueId) -> String,
        format_operand: impl Fn(&Operand) -> String,
    ) -> (String, String) {
        let (root, indices) =
            emit::walk_gep_chain(ptr, &self.gep_elements).unwrap_or((ptr, Vec::new()));
        let index = indices
            .iter()
            .map(|(index, stride)| emit::scaled_index(index, *stride, &format_operand, wgsl_index))
            .collect::<Vec<_>>()
            .join(" + ");
        // The alias binding is named after the kernel parameter it covers, so
        // the CAS and the prelude agree without the value table.
        let buffer = self.atomic_buffer_name(root).unwrap_or_else(|| match root {
            llvm_ir::Operand::Value(id) => value_name(*id),
            other => format_operand(other),
        });
        (format!("{buffer}_atomic"), format!("({index})"))
    }

    fn new(func: &'a llvm_ir::Function) -> Result<Self> {
        Ok(Self {
            func,
            output: find_output_buffer(func),
            gep_elements: emit::collect_gep_elements(func),
            atomic_sites: 0,
            shapes_name: "sile_shapes".to_string(),
        })
    }

    /// Binding index of one parameter.
    ///
    /// Kernel buffers bind from zero in parameter order, and the two
    /// descriptors bind after all of them, the shapes first and the scalars
    /// second, so a kernel's bindings stay stable however many shapes or
    /// scalars it carries.
    fn binding_index(&self, param: &llvm_ir::Param) -> usize {
        let buffers = self
            .func
            .params
            .iter()
            .filter(|other| !emit::is_descriptor_param(other))
            .count();
        if emit::is_shapes_param(param) {
            return buffers;
        }
        if emit::is_scalars_param(param) {
            return buffers + 1;
        }
        if emit::is_scalars_i64_param(param) {
            return buffers + 2;
        }
        self.func
            .params
            .iter()
            .filter(|other| !emit::is_descriptor_param(other))
            .position(|other| other.id == param.id)
            .unwrap_or(0)
    }

    /// `array<f32>` with the address space WGSL requires for this parameter.
    ///
    /// A read-only buffer is `read`, a mutable one is `read_write`, and both
    /// descriptors are read-only.
    fn storage_decl(&self, param: &llvm_ir::Param) -> String {
        let access = if emit::is_descriptor_param(param) {
            "read"
        } else {
            "read_write"
        };
        let elem = match &param.ty {
            llvm_ir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
            // Every parameter the lowering produces is a pointer: a buffer or
            // one of the descriptors. A plain value here would mean the ABI
            // grew a parameter kind this declaration cannot spell.
            other => panic!("a WGSL storage parameter holds a buffer pointer, not {other:?}"),
        };
        let name = if emit::is_shapes_param(param) {
            self.shapes_name.clone()
        } else if emit::is_scalars_param(param) {
            SCALARS_NAME.to_string()
        } else if emit::is_scalars_i64_param(param) {
            SCALARS_I64_NAME.to_string()
        } else {
            param.name.clone()
        };
        format!("var<storage, {access}> {name}: array<{elem}>;")
    }

    /// Program index along logical grid axis `dim`.
    ///
    /// One workgroup item owns one tile, and the dispatch flattens the grid
    /// onto the x axis of the invocation id. Dividing the flat index by the
    /// strides after `dim` and taking the remainder recovers the index along
    /// `dim`, whether the output has one axis, two or three. Reading `y` here
    /// instead would make every program of a rank-2 kernel the same row:
    /// `dispatch_workgroups(n, 1, 1)` leaves y at zero.
    fn block_id_expr(&self, dim: u8) -> String {
        let flat = format!("i32({LAUNCH_INDEX}.x)");
        let Some(output) = self.output.as_ref() else {
            return match dim {
                0 => format!("i64({flat})"),
                _ => "i64(0)".to_string(),
            };
        };
        let Some(abi) = output.param.abi.as_ref() else {
            return format!("i64({flat})");
        };
        if abi.rank <= 1 {
            return match dim {
                0 => format!("i64({flat})"),
                _ => "i64(0)".to_string(),
            };
        }
        let axes = self.i32_grid_axes(abi, output.rows, output.cols);
        let Some(extent) = axes.get(dim as usize) else {
            return "i64(0)".to_string();
        };
        let stride = emit::grid_axis_stride(&axes, dim as usize);
        format!("i64((({flat} / ({stride})) % ({extent})))")
    }

    /// The grid-axis extents spelled in `i32`.
    ///
    /// A launch grid never approaches 2^31 tiles, so the flat-index split
    /// needs no 64-bit arithmetic — and naga wraps a 64-bit division in a
    /// `metal::select` with an untyped literal, which Metal rejects as an
    /// ambiguous call. The read widens first, so the division itself is
    /// 32-bit; `block_id_expr` widens the split result back.
    fn i32_grid_axes(&self, abi: &llvm_ir::ParamAbi, rows: usize, cols: usize) -> Vec<String> {
        emit::grid_axis_exprs(abi, rows, cols, &self.shapes_name)
            .iter()
            .map(|extent| {
                let rest = extent
                    .strip_prefix(self.shapes_name.as_str())
                    .unwrap_or_else(|| panic!("a grid axis reads the shape descriptor"));
                let index_end = rest.find(']').unwrap_or_else(|| {
                    panic!("a grid axis reads one shape descriptor entry: {extent}")
                });
                let (read, tail) = rest.split_at(index_end + 1);
                format!("i32({}{read}){tail}", self.shapes_name)
            })
            .collect()
    }

    /// Extent of logical grid axis `dim`.
    ///
    /// One workgroup item owns one tile, so the axis is the output's tile count
    /// along it. The count comes from the shape descriptor and the tile ABI.
    fn grid_dim_expr(&self, dim: u8) -> Result<String> {
        let Some(output) = self.output.as_ref() else {
            return Ok(match dim {
                0 => format!("i64({LAUNCH_INDEX}.x)"),
                _ => "i64(1)".to_string(),
            });
        };
        let abi = output.param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile("missing output ABI for WGSL grid_dim".into())
        })?;
        let axes = self.i32_grid_axes(abi, output.rows, output.cols);
        Ok(axes
            .get(dim as usize)
            .map(|extent| format!("i64({extent})"))
            .unwrap_or_else(|| "i64(1)".to_string()))
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
    /// WGSL has no infinity and no NaN constant, so a non-finite value is the
    /// bit pattern it stands for. `bitcast` is a constant expression, so this
    /// stays a literal wherever a literal is allowed.
    fn float_literal(value: f64) -> String {
        if value.is_finite() {
            return emit::float_literal(value);
        }
        format!("bitcast<f32>({:#010x}u)", (value as f32).to_bits())
    }

    fn shapes_name(&self) -> String {
        self.shapes_name.clone()
    }

    fn scalars_name(&self) -> String {
        SCALARS_NAME.to_string()
    }

    fn scalars_i64_name(&self) -> String {
        SCALARS_I64_NAME.to_string()
    }

    fn var_decl(&self, ty: &llvm_ir::Type, name: &str, _is_alloca: bool) -> Option<String> {
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
        // A half storage is an optional WGSL capability, and the `f16` type it
        // names is only in scope under its enable directive, which must come
        // before every declaration. The runtime asks the adapter for the
        // feature the kernel needs.
        if self.func.params.iter().any(|param| {
            matches!(
                &param.ty,
                llvm_ir::Type::Ptr { pointee, .. }
                    if **pointee == llvm_ir::Type::F16
            )
        }) {
            lines.push("enable f16;".to_string());
        }
        // Kernel buffers bind first, then the two descriptors, so a kernel's
        // bindings stay stable no matter how many shapes or scalars it carries.
        for param in &self.func.params {
            if emit::is_descriptor_param(param) {
                continue;
            }
            lines.push(format!(
                "@group(0) @binding({}) {}",
                self.binding_index(param),
                self.storage_decl(param)
            ));
        }
        for param in &self.func.params {
            if emit::is_descriptor_param(param) {
                lines.push(format!(
                    "@group(0) @binding({}) {}",
                    self.binding_index(param),
                    self.storage_decl(param)
                ));
            }
        }
        // A float atomic add needs the same storage bound once more as an
        // integer atomic, which is the only spelling WGSL has. One binding per
        // buffer that carries an atomic add, numbered after the shapes
        // descriptor so a kernel's bindings stay stable.
        let mut alias = self.alias_binding_base();
        for param in &self.atomic_buffer_params() {
            lines.push(format!(
                "@group(0) @binding({}) var<storage, read_write> {}_atomic: array<atomic<i32>>;",
                alias, param.name
            ));
            alias += 1;
        }
        lines
    }

    fn pointers(&self) -> emit::PointerSpelling {
        emit::PointerSpelling {
            index_cast: wgsl_index,
            ..NO_POINTERS
        }
    }

    fn int_div(&self) -> emit::IntDivSpelling {
        emit::IntDivSpelling::I32Widened
    }

    fn declares_intrinsic_at_use(&self, inst: &llvm_ir::Inst) -> bool {
        // A WGSL intrinsic is a `let` binding, so it must not also be declared.
        matches!(inst.op, llvm_ir::InstOp::Intrinsic { .. })
    }

    fn select(&self) -> emit::SelectSpelling {
        // WGSL has no conditional expression; a choice is a call. A bare
        // integer argument resolves to `i32` before the context is seen, so
        // the constants of an `i64` choice carry their type.
        emit::SelectSpelling::Function {
            int_const: |value| format!("i64({value})"),
        }
    }

    fn cast_expr(op: llvm_ir::CastOp, to: &llvm_ir::Type, value: String) -> String {
        // WGSL has no implicit conversion, so a conversion names the type it
        // produces. A bitcast reinterprets the bits instead of converting them.
        match op {
            llvm_ir::CastOp::Bitcast => format!("bitcast<{}>({value})", wgsl_type(to)),
            _ => format!("{}({value})", wgsl_type(to)),
        }
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
                // WGSL names the absolute value `abs`; the shared table says
                // `fabs`, the C and MSL spelling.
                let function = match intrinsic {
                    llvm_ir::Intrinsic::Abs => "abs",
                    _ => intrinsic
                        .float_function()
                        .expect("a float function has a name"),
                };
                format!("let {} = {}({});", name(), function, format_operand(arg))
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
                    "let {} = {}({}, {});",
                    name(),
                    function,
                    format_operand(lhs),
                    format_operand(rhs)
                )
            }
            // WGSL names the fused multiply-add `fma`, the shared table's
            // spelling, so no per-target rename applies here.
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
                    "let {} = {}({}, {}, {});",
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
                        "WGSL vec_load expects [ptr, offset]".into(),
                    ));
                };
                // WGSL cannot cast a storage array to a vector, so each lane
                // is read as one element and the constructor builds the vector.
                // The IR keeps the vector type because C and MSL read all four
                // lanes in one instruction.
                let element =
                    self.buffer_element(ptr, &|id| value_name(id), &|op| format_operand(op));
                let offset = format_operand(offset);
                let lanes = (0..*len)
                    .map(|lane| format!("{element}[{offset} + i64({lane})]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("let {} = vec{len}<f32>({lanes});", name())
            }
            llvm_ir::Intrinsic::VecStore { len } => {
                let [ptr, offset, value] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL vec_store expects [ptr, offset, value]".into(),
                    ));
                };
                let _ = len;
                format!(
                    "{}[{}] = {};",
                    self.buffer_element(ptr, &|id| value_name(id), &|op| format_operand(op)),
                    format_operand(offset),
                    format_operand(value)
                )
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
            // WGSL has no reduction directive, so a tile-level fold is the
            // loop it would have become anyway. Keeping the shape here means
            // the lowering does not decide how the target folds.
            llvm_ir::Intrinsic::TileReduce {
                reduce,
                extent,
                inner,
                outer,
            } => {
                let [source] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL tile_reduce expects one argument".into(),
                    ));
                };
                let src = self.buffer_element(
                    source,
                    &|id| value_name(id),
                    &|op| format_operand(op),
                );
                let dst = name();
                let span = *extent * *inner;
                // The identity is the first element of each fold: every
                // combining operation here is one the first element absorbs.
                let combine = match reduce {
                    llvm_ir::TileReduceOp::Sum => |acc: String, v: String| format!("{acc} + {v}"),
                    llvm_ir::TileReduceOp::Product => {
                        |acc: String, v: String| format!("{acc} * {v}")
                    }
                    llvm_ir::TileReduceOp::Max => {
                        |acc: String, v: String| format!("select({v}, {acc}, {acc} > {v})")
                    }
                    llvm_ir::TileReduceOp::Min => {
                        |acc: String, v: String| format!("select({v}, {acc}, {acc} < {v})")
                    }
                };
                // WGSL indexes an array with an `i32`, so the fold's counters
                // are `i32` rather than the `i64` the rest of the lowering uses.
                let mut lines = vec![format!("var {dst}: array<f32, {}>;", outer)];
                lines.push(format!(
                    "for (var ro: i32 = 0; ro < {outer}; ro = ro + 1) {{"
                ));
                lines.push(format!("  var racc = {src}[ro * {span}];"));
                lines.push(format!(
                    "  for (var ri: i32 = 1; ri < {span}; ri = ri + 1) {{"
                ));
                lines.push(format!(
                    "    let rv = {src}[ro * {span} + ri];"
                ));
                lines.push(format!("    racc = {};", combine("racc".into(), "rv".into())));
                lines.push("  }".to_string());
                lines.push(format!("  {dst}[ro] = racc;"));
                lines.push("}".to_string());
                return Ok(lines);
            }
            // WGSL has no matrix instruction, so the multiply is the loop
            // nest the target's own compiler vectorises. The tiles arrive as
            // `var` arrays, which are indexed with an `i32`.
            llvm_ir::Intrinsic::TileMma { m, n, k, rhs_cols } => {
                let [lhs, rhs, ..] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL tile_mma expects at least two arguments".into(),
                    ));
                };
                let acc = args.get(2);
                let buffer = |operand: &llvm_ir::Operand| {
                    self.buffer_element(operand, &|id| value_name(id), &|op| format_operand(op))
                };
                let (lhs, rhs) = (buffer(lhs), buffer(rhs));
                let dst = name();
                let (m, n, k, rhs_cols) = (*m, *n, *k, *rhs_cols);
                let seed = match acc {
                    Some(_) => format!("{dst}[mo * {n} + no]"),
                    None => "0.0f".to_string(),
                };
                let mut lines = vec![format!("var {dst}: array<f32, {}>;", m * n)];
                if let Some(accumulator) = acc {
                    let source = buffer(accumulator);
                    lines.push(format!(
                        "for (var mi: i32 = 0; mi < {}; mi = mi + 1) {{",
                        m * n
                    ));
                    lines.push(format!("  {dst}[mi] = {source}[mi];"));
                    lines.push("}".to_string());
                }
                lines.push(format!(
                    "for (var mo: i32 = 0; mo < {m}; mo = mo + 1) {{"
                ));
                lines.push(format!(
                    "  for (var no: i32 = 0; no < {n}; no = no + 1) {{"
                ));
                lines.push(format!("    var racc = {seed};"));
                lines.push(format!(
                    "    for (var ko: i32 = 0; ko < {k}; ko = ko + 1) {{"
                ));
                lines.push(format!(
                    "      racc = racc + {lhs}[mo * {k} + ko] * {rhs}[ko * {rhs_cols} + no];"
                ));
                lines.push("    }".to_string());
                lines.push(format!("    {dst}[mo * {n} + no] = racc;"));
                lines.push("  }".to_string());
                lines.push("}".to_string());
                return Ok(lines);
            }
        };
        Ok(vec![line])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &llvm_ir::Inst,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
    ) -> Result<Vec<String>> {
        // A pointer argument spells the element it addresses, the same way a
        // load reads it, so the shared spelling is reused rather than repeated.
        let element = |ptr: &llvm_ir::Operand| {
            self.buffer_element(ptr, &|id| value_name(id), &|op| format_operand(op))
        };
        match &inst.op {
            // WGSL has no `memcpy`; a byte copy becomes an element loop.
            llvm_ir::InstOp::Memcpy { dst, src, size } => Ok(vec![
                format!(
                    "for (var copy_i: i64 = 0; copy_i < {}; copy_i++) {{",
                    format_operand(size)
                ),
                format!("{}[copy_i] = {}[copy_i];", element(dst), element(src)),
                "}".to_string(),
            ]),
            // WGSL has no atomic add on floats, so the same storage is bound
            // a second time as `atomic<i32>` and the add is a compare exchange
            // loop over the bit pattern. One storage buffer, two bindings,
            // which is the only spelling this target has.
            llvm_ir::InstOp::AtomicAdd { ptr, value } => {
                let (buffer, index) = self.atomic_target(ptr, &value_name, &format_operand);
                let cas = self.next_atomic_cas_name();
                Ok(vec![
                    format!("var {cas}_comparand: i32 = 0;"),
                    format!(
                        "var {cas}_desired: i32 = bitcast<i32>({});",
                        format_operand(value)
                    ),
                    "loop {".to_string(),
                    format!(
                        "let {cas}_exchange = atomicCompareExchangeWeak(&{buffer}[{index}], {cas}_comparand, {cas}_desired);"
                    ),
                    format!("if ({cas}_exchange.exchanged) {{ break; }}"),
                    format!("{cas}_comparand = {cas}_exchange.old_value;"),
                    format!(
                        "{cas}_desired = bitcast<i32>(bitcast<f32>({cas}_comparand) + {});",
                        format_operand(value)
                    ),
                    "}".to_string(),
                ])
            }
            llvm_ir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR WGSL codegen does not support call instructions: {func}"
            ))),
            // An intrinsic never reaches here: the shared emitter dispatches it
            // to `intrinsic_stmt` before this match is consulted.
            llvm_ir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LLVM IR WGSL codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // An alloca is the `var` declaration the emitter already wrote, and
            // a `gep` is spelled at its use: neither leaves a statement here.
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
                "LLVM IR WGSL codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
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
