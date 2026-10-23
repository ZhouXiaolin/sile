use std::collections::HashMap;

use crate::emit::{self, NO_POINTERS, StructuredCfgMessages, TargetDialect, element_operand};
use sile_core::Result;
use sile_lsir as lsir;
use sile_lsir::{Operand, ValueId};

/// The entry point parameter that carries the launch index.
///
/// WGSL has no implicit launch index: a compute entry point declares one
/// `@builtin(workgroup_id)` parameter and every axis of the launch grid is a
/// component of it. One workgroup owns one tile, so the workgroup id is the
/// tile index, whatever the workgroup size.
const LAUNCH_INDEX: &str = "sile_launch_index";

/// The entry point parameter that carries the thread's slot in its workgroup.
///
/// The mapping strips the tile's element runs across this index; the IR's
/// `thread_id` is spelled from it.
const LOCAL_INDEX: &str = "sile_local_index";

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
/// Every integer value spells 32-bit (see `wgsl_type`), so a computed index is
/// already an `i32`; the cast stays as the guard for the `u32` builtins and
/// bare literals a subscript can still meet.
fn wgsl_index(index: String) -> String {
    format!("i32({index})")
}

pub fn generate(func: &lsir::Function, policy: &sile_lsir::mapping::MappingPolicy) -> Result<String> {
    let dialect = WgslDialect::new(func, policy.launch.count())?;
    emit::generate_text(emit::TargetEmitter::new(
        func,
        dialect,
        policy.vector,
    ))
}

/// The kernel buffers that carry a float atomic add, in binding order.
///
/// WGSL has no atomic add on a float, so each such buffer is bound a second
/// time as `atomic<i32>` and the add is a compare exchange over the bit
/// pattern. The runtime has to bind the same storage twice, and this tells it
/// which buffers, so the codegen and the runtime cannot disagree about the
/// layout.
pub fn atomic_alias_bindings(func: &lsir::Function) -> Vec<usize> {
    let dialect =
        WgslDialect::new(func, 1).expect("WGSL dialect over a typed function");
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
    func: &'a lsir::Function,
    /// Threads that cooperate on one tile, from the mapping policy. The entry
    /// point declares it as its workgroup size, and the thread id is spelled
    /// against it.
    threads: u32,
    /// Every shared allocation the mapping introduced, as `(value name,
    /// pointee type)`. WGSL only holds shared storage at module scope, so
    /// these print among the prelude declarations instead of in the body.
    shared_allocas: Vec<(String, lsir::Type)>,
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
        ptr: &lsir::Operand,
        value_name: &dyn Fn(lsir::ValueId) -> String,
        format_operand: &dyn Fn(&lsir::Operand) -> String,
    ) -> String {
        let spelled = element_operand(
            ptr,
            &self.gep_elements,
            value_name,
            format_operand,
            wgsl_index,
        );
        match ptr {
            lsir::Operand::Value(id) if self.gep_elements.contains_key(id) => spelled,
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
    fn atomic_buffer_params(&self) -> Vec<&lsir::Param> {
        let names: Vec<String> = self
            .func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.op {
                lsir::InstOp::AtomicAdd { ptr, .. } => self.atomic_buffer_name(ptr),
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
    fn atomic_buffer_name(&self, ptr: &lsir::Operand) -> Option<String> {
        let (root, _) = emit::walk_gep_chain(ptr, &self.gep_elements)?;
        match root {
            lsir::Operand::Value(id) => self
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
        ptr: &lsir::Operand,
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
            lsir::Operand::Value(id) => value_name(*id),
            other => format_operand(other),
        });
        (format!("{buffer}_atomic"), format!("({index})"))
    }

    fn new(func: &'a lsir::Function, threads: u32) -> Result<Self> {
        // The spelling demotes every integer to `i32`, which is sound only
        // for the descriptors the host packs 32-bit. An `i64` element buffer
        // would bind `array<i32>` over 8-byte host elements — refused here
        // rather than misspelled.
        for param in &func.params {
            if emit::is_descriptor_param(param) {
                continue;
            }
            if let lsir::Type::Ptr { pointee, .. } = &param.ty
                && **pointee == lsir::Type::I64
            {
                return Err(sile_core::Error::Compile(format!(
                    "WGSL spells 32-bit integers, so the i64 buffer `{}` has no spelling",
                    param.name
                )));
            }
        }
        // Shared storage prints in the prelude under the same names the body
        // spells, so the table is the one the emitter builds.
        let names = emit::build_value_names(func);
        let shared_allocas = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.op {
                lsir::InstOp::Alloca {
                    alloc_ty,
                    addr_space: lsir::AddressSpace::Shared,
                } => inst
                    .result
                    .and_then(|id| names.get(&id).cloned())
                    .map(|name| (name, alloc_ty.clone())),
                _ => None,
            })
            .collect();
        Ok(Self {
            func,
            threads,
            shared_allocas,
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
    fn binding_index(&self, param: &lsir::Param) -> usize {
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
    fn storage_decl(&self, param: &lsir::Param) -> String {
        let access = if emit::is_descriptor_param(param) {
            "read"
        } else {
            "read_write"
        };
        let elem = match &param.ty {
            lsir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
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

    /// Program index along launch grid axis `dim`.
    ///
    /// The runtime dispatches the caller's grid as given, so each axis of
    /// `workgroup_id` is one program axis — the same terms the CPU spells its
    /// `gid` in. The grid a kernel reads is the grid the launcher derived; no
    /// flat-index split here, because the source of the split (the tile sizes
    /// the launcher divided the buffer extents by) lives on the host alone.
    fn block_id_expr(&self, dim: u8) -> String {
        let axis = ["x", "y", "z"][dim as usize];
        format!("i32({LAUNCH_INDEX}.{axis})")
    }
}

/// WGSL name of one LSIR type.
fn wgsl_type(ty: &lsir::Type) -> String {
    match ty {
        lsir::Type::Void => "void".to_string(),
        lsir::Type::F16 => "f16".to_string(),
        lsir::Type::I1 => "bool".to_string(),
        lsir::Type::I32 => "i32".to_string(),
        // The WGSL spelling demotes the lowering's 64-bit integers to `i32`
        // (MAPPING_DESIGN §5): no kernel value needs the width, and i64 is an
        // optional capability that naga lowers into wrapped multi-instruction
        // arithmetic.
        lsir::Type::I64 => "i32".to_string(),
        lsir::Type::F32 => "f32".to_string(),
        lsir::Type::F64 => "f64".to_string(),
        lsir::Type::Ptr { pointee, .. } => wgsl_type(pointee.as_ref()),
        // WGSL arrays are one-dimensional: a nested array is the flat element
        // type with the total element count. A private alloca that holds a
        // tile is spelled through this same path, and a Gep chain already
        // summed its indices into one element offset, so the flat storage and
        // the flat index agree.
        lsir::Type::Array { .. } => {
            let base = array_base_type(ty);
            let count = array_element_count(ty);
            format!("array<{base}, {count}>")
        }
        lsir::Type::Vector { elem, len } => format!("vec{}<{}>", len, wgsl_type(elem)),
    }
}

/// Element type at the bottom of a (possibly nested) array.
fn array_base_type(ty: &lsir::Type) -> String {
    let mut current = ty;
    while let lsir::Type::Array { elem, .. } = current {
        current = elem;
    }
    wgsl_type(current)
}

/// Total number of elements in a (possibly nested) array.
fn array_element_count(ty: &lsir::Type) -> usize {
    let mut count = 1usize;
    let mut current = ty;
    while let lsir::Type::Array { len, elem } = current {
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

    fn var_decl(&self, ty: &lsir::Type, name: &str, _is_alloca: bool) -> Option<String> {
        // Shared storage only exists at module scope in WGSL, so a shared
        // alloca prints among the prelude declarations, not here.
        if matches!(ty, lsir::Type::Ptr { addr_space: lsir::AddressSpace::Shared, .. }) {
            return None;
        }
        Some(format!("var {}: {};", name, wgsl_type(ty)))
    }

    fn param_decl(&self, ty: &lsir::Type, _name: &str) -> String {
        // A buffer parameter contributes a module-level binding, not a
        // parameter list entry. See `prelude_lines`.
        let _ = ty;
        String::new()
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        // All buffers arrive as bindings, so the parameter list is empty apart
        // from the two launch indices. WGSL names both in the signature as
        // `@builtin`s, which is the only way a compute entry point reads them:
        // the workgroup id is the tile index, and the local invocation id is
        // the thread's slot in the tile's work.
        //
        // The entry attributes belong here, against the function they decorate.
        // WGSL attributes attach to the declaration that follows them, so an
        // entry attribute printed before the buffer bindings decorates those
        // bindings instead, and the kernel stops being an entry point.
        let _ = params;
        format!(
            "@compute @workgroup_size({})\n\
             fn sile_kernel_{name}(@builtin(workgroup_id) {LAUNCH_INDEX}: vec3<u32>, \
             @builtin(local_invocation_id) {LOCAL_INDEX}: vec3<u32>) {{\n",
            self.threads
        )
    }

    fn prelude_lines(&self, _func: &lsir::Function) -> Vec<String> {
        let mut lines = Vec::new();
        // A half storage is an optional WGSL capability, and the `f16` type it
        // names is only in scope under its enable directive, which must come
        // before every declaration. The runtime asks the adapter for the
        // feature the kernel needs.
        if self.func.params.iter().any(|param| {
            matches!(
                &param.ty,
                lsir::Type::Ptr { pointee, .. }
                    if **pointee == lsir::Type::F16
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
        // Shared storage is module scope in WGSL: every thread of the
        // workgroup sees one array, which is what the mapping's tree
        // reduction and shared multiply panels live in.
        for (name, pointee) in &self.shared_allocas {
            lines.push(format!("var<workgroup> {name}: {};", wgsl_type(pointee)));
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

    fn declares_intrinsic_at_use(&self, inst: &lsir::Inst) -> bool {
        // A WGSL intrinsic is a `let` binding, so it must not also be declared.
        matches!(inst.op, lsir::InstOp::Intrinsic { .. })
    }

    fn select(&self) -> emit::SelectSpelling {
        // WGSL has no conditional expression; a choice is a call. A bare
        // integer argument resolves to `i32`, which is what every choice
        // spells now that the i64 IR type demotes (see `wgsl_type`).
        emit::SelectSpelling::Function {
            int_const: |value| value.to_string(),
        }
    }

    fn cast_expr(op: lsir::CastOp, to: &lsir::Type, value: String) -> String {
        // WGSL has no implicit conversion, so a conversion names the type it
        // produces. A bitcast reinterprets the bits instead of converting them.
        match op {
            lsir::CastOp::Bitcast => format!("bitcast<{}>({value})", wgsl_type(to)),
            _ => format!("{}({value})", wgsl_type(to)),
        }
    }

    fn vectorize_hint(&self, _is_simple_body: bool) -> Option<String> {
        // WGSL has no pragma, and the driver already vectorizes a simple loop.
        None
    }

    fn intrinsic_stmt(
        &self,
        result: Option<lsir::ValueId>,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
        intrinsic: &lsir::Intrinsic,
        args: &[lsir::Operand],
    ) -> Result<Vec<String>> {
        let name = || {
            result.map(|id| value_name(id)).unwrap_or_else(|| {
                sile_core::Error::Compile("WGSL intrinsic needs a result".into()).to_string()
            })
        };
        let line = match intrinsic {
            lsir::Intrinsic::BlockId { dim } => {
                format!("let {} = {};", name(), self.block_id_expr(*dim))
            }
            lsir::Intrinsic::ThreadId { dim } => {
                // The mapping's flat thread index: this workgroup's x slot.
                // Only dimension 0 exists under the workgroup mapping.
                debug_assert!(*dim == 0, "the workgroup mapping only uses thread axis 0");
                format!("let {} = i32({LOCAL_INDEX}.x);", name())
            }
            lsir::Intrinsic::GridDim { .. } => {
                // The launch grid lives on the host: the launcher derives it
                // from the output shapes before the runtime dispatches. A
                // kernel that needs it compiles for the CPU, whose launcher
                // writes the grid into thread-local state.
                return Err(sile_core::Error::Compile(
                    "WGSL cannot spell grid_dim: the launch grid is host-side state".into(),
                ));
            }
            lsir::Intrinsic::Barrier { .. } => "workgroupBarrier();".to_string(),
            lsir::Intrinsic::Exp
            | lsir::Intrinsic::Exp2
            | lsir::Intrinsic::Log
            | lsir::Intrinsic::Log2
            | lsir::Intrinsic::Sqrt
            | lsir::Intrinsic::Abs
            | lsir::Intrinsic::Sin
            | lsir::Intrinsic::Cos
            | lsir::Intrinsic::Tan
            | lsir::Intrinsic::Sinh
            | lsir::Intrinsic::Cosh
            | lsir::Intrinsic::Tanh
            | lsir::Intrinsic::Ceil
            | lsir::Intrinsic::Floor => {
                let [arg] = args else {
                    return Err(sile_core::Error::Compile(
                        "a float function takes one argument".into(),
                    ));
                };
                // WGSL names the absolute value `abs`; the shared table says
                // `fabs`, the C and MSL spelling.
                let function = match intrinsic {
                    lsir::Intrinsic::Abs => "abs",
                    _ => intrinsic
                        .float_function()
                        .expect("a float function has a name"),
                };
                format!("let {} = {}({});", name(), function, format_operand(arg))
            }
            lsir::Intrinsic::Pow | lsir::Intrinsic::Atan2 => {
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
            lsir::Intrinsic::Fma => {
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
            lsir::Intrinsic::VecLoad { len } => {
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
                    .map(|lane| format!("{element}[{offset} + {lane}]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("let {} = vec{len}<f32>({lanes});", name())
            }
            lsir::Intrinsic::VecStore { len } => {
                let [ptr, offset, value] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL vec_store expects [ptr, offset, value]".into(),
                    ));
                };
                // An array element is one scalar, so the vector fans out
                // into one store per lane.
                let element =
                    self.buffer_element(ptr, &|id| value_name(id), &|op| format_operand(op));
                let offset = format_operand(offset);
                let value = format_operand(value);
                (0..*len)
                    .map(|lane| format!("{element}[{offset} + {lane}] = {value}[{lane}];"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            lsir::Intrinsic::VecReduceAdd => {
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
            lsir::Intrinsic::TileReduce {
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
                    lsir::TileReduceOp::Sum => |acc: String, v: String| format!("{acc} + {v}"),
                    lsir::TileReduceOp::Product => {
                        |acc: String, v: String| format!("{acc} * {v}")
                    }
                    lsir::TileReduceOp::Max => {
                        |acc: String, v: String| format!("select({v}, {acc}, {acc} > {v})")
                    }
                    lsir::TileReduceOp::Min => {
                        |acc: String, v: String| format!("select({v}, {acc}, {acc} < {v})")
                    }
                };
                // The fold's counters are `i32` — the width every value
                // spells (see `wgsl_type`) — and array subscripts require it.
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
            lsir::Intrinsic::TileMma { m, n, k, rhs_cols } => {
                let [lhs, rhs, ..] = args else {
                    return Err(sile_core::Error::Compile(
                        "WGSL tile_mma expects at least two arguments".into(),
                    ));
                };
                let acc = args.get(2);
                let buffer = |operand: &lsir::Operand| {
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
                // An f16 panel widens at the multiply: WGSL has no implicit
                // conversion, and a half product added to the f32 accumulator
                // would not parse. The widening also keeps the contract the
                // f16 examples state — no half arithmetic, only half storage.
                lines.push(format!(
                    "      racc = racc + f32({lhs}[mo * {k} + ko]) * f32({rhs}[ko * {rhs_cols} + no]);"
                ));
                lines.push("    }".to_string());
                lines.push(format!("    {dst}[mo * {n} + no] = racc;"));
                lines.push("  }".to_string());
                lines.push("}".to_string());
                return Ok(lines);
            }
            // The SIMD-group intrinsics are the Metal mapping's plants. The
            // parallel contract rejects them before emit; this arm keeps the
            // rejection loud if the pipeline changes under it.
            lsir::Intrinsic::SimdReduce { .. }
            | lsir::Intrinsic::SimdLaneId
            | lsir::Intrinsic::SimdGroupId
            | lsir::Intrinsic::SimdGroupMma { .. } => {
                return Err(sile_core::Error::Compile(
                    "WGSL cannot spell the SIMD-group intrinsics: the WGSL mapping never plants them"
                        .into(),
                ));
            }
        };
        Ok(vec![line])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &lsir::Inst,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
    ) -> Result<Vec<String>> {
        // A pointer argument spells the element it addresses, the same way a
        // load reads it, so the shared spelling is reused rather than repeated.
        let element = |ptr: &lsir::Operand| {
            self.buffer_element(ptr, &|id| value_name(id), &|op| format_operand(op))
        };
        match &inst.op {
            // WGSL has no `memcpy`; a byte copy becomes an element loop.
            lsir::InstOp::Memcpy { dst, src, size } => Ok(vec![
                format!(
                    "for (var copy_i: i32 = 0; copy_i < {}; copy_i++) {{",
                    format_operand(size)
                ),
                format!("{}[copy_i] = {}[copy_i];", element(dst), element(src)),
                "}".to_string(),
            ]),
            // WGSL has no atomic add on floats, so the same storage is bound
            // a second time as `atomic<i32>` and the add is a compare exchange
            // loop over the bit pattern. One storage buffer, two bindings,
            // which is the only spelling this target has.
            lsir::InstOp::AtomicAdd { ptr, value } => {
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
            lsir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LSIR WGSL codegen does not support call instructions: {func}"
            ))),
            // An intrinsic never reaches here: the shared emitter dispatches it
            // to `intrinsic_stmt` before this match is consulted.
            lsir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LSIR WGSL codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // An alloca is the `var` declaration the emitter already wrote, and
            // a `gep` is spelled at its use: neither leaves a statement here.
            lsir::InstOp::Alloca { .. } | lsir::InstOp::Gep { .. } => Ok(vec![]),
            // The shared lowering spells these the same way on every target. A
            // result-less operator reaching here would otherwise be dropped,
            // and the kernel would read an undefined name.
            lsir::InstOp::Bin { .. }
            | lsir::InstOp::Cmp { .. }
            | lsir::InstOp::Cast { .. }
            | lsir::InstOp::Load { .. }
            | lsir::InstOp::Select { .. }
            | lsir::InstOp::Store { .. } => Err(sile_core::Error::Compile(format!(
                "LSIR WGSL codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured WGSL loop preheader must end with a branch",
            missing_loop_header: "missing LSIR WGSL loop header",
            header_must_cond_br: "structured WGSL loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured WGSL loop body did not produce the expected backedge",
            unsupported_cond_br: "LSIR WGSL codegen only supports structured conditional branches",
            unsupported_switch: "LSIR WGSL codegen does not yet support structured switch lowering",
        }
    }
}
