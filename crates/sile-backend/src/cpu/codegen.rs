use crate::emit::{
    self, StructuredCfgMessages, array_dims, find_output_buffer, is_descriptor_param,
    is_scalars_i64_param, is_scalars_param, is_shapes_param,
};
use sile_llvm_ir as llvm_ir;

pub fn generate(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let mut code = emit::generate_text(CTargetEmitter::new(func, CDialect::new()))?;
    let wrapper = generate_wrapper(func)?;
    code.push('\n');
    code.push_str(&wrapper);
    Ok(code)
}

/// The C emitter: shared traversal plus the C dialect.
type CTargetEmitter<'a> = emit::TargetEmitter<'a, CDialect>;

/// How C spells each SILE concept.
///
/// The traversal that walks blocks, names values and emits structured control
/// flow lives in `emit::TargetEmitter`. This is only C's part of the deal, so
/// a new target writes one of these instead of copying the walk.
struct CDialect;

impl CDialect {
    fn new() -> Self {
        Self
    }
}

impl emit::TargetDialect for CDialect {
    /// A tile is an address here, so a parameter takes a copy of the elements
    /// rather than the address of storage the next iteration overwrites.
    fn param_assign(&self, ty: &llvm_ir::Type, name: &str, arg: &str) -> String {
        emit::copy_assign(ty, name, arg)
    }

    fn var_decl(&self, ty: &llvm_ir::Type, name: &str, is_alloca: bool) -> Option<String> {
        // An `alloca` is the only C local that owns storage. A `gep` into a
        // driver buffer has the same type but borrows, so it stays a pointer.
        match ty {
            llvm_ir::Type::Ptr {
                addr_space: llvm_ir::AddressSpace::Private,
                pointee,
            } if is_alloca => {
                let storage_name = format!("{name}_storage");
                Some(format!(
                    "{}; {};",
                    c_storage_decl(pointee, &storage_name),
                    c_ptr_storage_bind_decl(pointee, name, &storage_name)
                ))
            }
            _ => Some(format!("{};", c_var_decl(ty, name))),
        }
    }

    fn param_decl(&self, ty: &llvm_ir::Type, _name: &str) -> String {
        c_param_type(ty)
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        let joined = params
            .iter()
            .map(|(ty, param_name)| format!("{ty} {param_name}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("void sile_llvm_ir_{name}({joined}) {{\n")
    }

    /// C needs its headers, its per-thread launch state and its vector helpers,
    /// in the order the kernel body depends on them.
    fn prelude_lines(&self) -> Vec<String> {
        let mut lines = vec![
            "#include <stdint.h>".to_string(),
            "#include <stdbool.h>".to_string(),
            "#include <omp.h>".to_string(),
            "#include <math.h>".to_string(),
            "#include <string.h>".to_string(),
            String::new(),
        ];
        // Grid extents only exist on the CPU host side, so the launcher writes
        // them before each kernel call. The grid is flattened here, so axes 1
        // and 2 are always one tile deep.
        for axis in 0..3 {
            lines.push(format!("static _Thread_local int64_t sile_gid_{axis} = 0;"));
        }
        for axis in 0..3 {
            lines.push(format!(
                "static _Thread_local int64_t sile_grid_{axis} = 1;"
            ));
        }
        lines.push(String::new());
        lines.push("#define llir_block_id_0() (sile_gid_0)".to_string());
        lines.push("#define llir_block_id_1() (sile_gid_1)".to_string());
        lines.push("#define llir_block_id_2() (sile_gid_2)".to_string());
        lines.push("#define llir_grid_dim_0() (sile_grid_0)".to_string());
        lines.push("#define llir_grid_dim_1() (sile_grid_1)".to_string());
        lines.push("#define llir_grid_dim_2() (sile_grid_2)".to_string());
        lines.push("#define llir_thread_id_0() (0)".to_string());
        lines.push("#define llir_thread_id_1() (0)".to_string());
        lines.push("#define llir_thread_id_2() (0)".to_string());
        lines.push("#define llir_barrier() ((void)0)".to_string());
        lines.push(String::new());
        // Vector load/reduce helpers, used by the vectorized reduce loops.
        lines.push("typedef struct { float v[4]; } llir_float4;".to_string());
        lines.push(
            "static inline llir_float4 llir_vec_load_4(const float* ptr, int64_t offset) {"
                .to_string(),
        );
        lines.push("    llir_float4 r;".to_string());
        lines.push(
            "    r.v[0] = ptr[offset]; r.v[1] = ptr[offset+1]; r.v[2] = ptr[offset+2]; r.v[3] = ptr[offset+3];"
                .to_string(),
        );
        lines.push("    return r;".to_string());
        lines.push("}".to_string());
        lines.push("static inline float llir_vec_reduce_add(llir_float4 v) {".to_string());
        lines.push("    return v.v[0] + v.v[1] + v.v[2] + v.v[3];".to_string());
        lines.push("}".to_string());
        lines.push(String::new());
        lines
    }

    fn vectorize_hint(&self, is_simple_body: bool) -> Option<String> {
        is_simple_body.then(|| "#pragma omp simd".to_string())
    }

    fn intrinsic_stmt(
        &self,
        result: Option<llvm_ir::ValueId>,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
        intrinsic: &llvm_ir::Intrinsic,
        args: &[llvm_ir::Operand],
    ) -> sile_core::Result<Vec<String>> {
        let joined = args
            .iter()
            .map(|arg| format_operand(arg))
            .collect::<Vec<_>>()
            .join(", ");
        let call = format!("{}({joined})", intrinsic_name(intrinsic));
        Ok(vec![match result {
            Some(id) => format!("{} = {call};", value_name(id)),
            None => format!("{call};"),
        }])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &llvm_ir::Inst,
        value_name: impl Fn(llvm_ir::ValueId) -> String,
        format_operand: impl Fn(&llvm_ir::Operand) -> String,
    ) -> sile_core::Result<Vec<String>> {
        // C is the only target that keeps an alloca as a real local array, so
        // it is the only one that skips the instruction and copies bytes.
        match &inst.op {
            llvm_ir::InstOp::Alloca { .. } => Ok(vec![]),
            llvm_ir::InstOp::Memcpy { dst, src, size } => Ok(vec![format!(
                "memcpy({}, {}, {});",
                format_operand(dst),
                format_operand(src),
                format_operand(size)
            )]),
            llvm_ir::InstOp::AtomicAdd { ptr, value } => Ok(vec![
                "#pragma omp atomic update".to_string(),
                format!("(*{}) += {};", format_operand(ptr), format_operand(value)),
            ]),
            llvm_ir::InstOp::Call { func, args } => {
                let joined = args
                    .iter()
                    .map(|arg| format_operand(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(vec![match inst.result {
                    Some(id) => format!("{} = {func}({joined});", value_name(id)),
                    None => format!("{func}({joined});"),
                }])
            }
            // An intrinsic never reaches here: the shared emitter dispatches it
            // to `intrinsic_stmt` before this match is consulted.
            llvm_ir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LLVM IR C codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // The shared lowering spells these the same way on every target. A
            // result-less operator reaching here would otherwise be dropped,
            // and the kernel would read an undefined name.
            llvm_ir::InstOp::Bin { .. }
            | llvm_ir::InstOp::Cmp { .. }
            | llvm_ir::InstOp::Cast { .. }
            | llvm_ir::InstOp::Gep { .. }
            | llvm_ir::InstOp::Load { .. }
            | llvm_ir::InstOp::Select { .. }
            | llvm_ir::InstOp::Store { .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR C codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured C loop preheader must end with a branch",
            missing_loop_header: "missing LLVM IR C loop header",
            header_must_cond_br: "structured C loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured C loop body did not produce the expected backedge",
            unsupported_cond_br: "LLVM IR C codegen only supports structured conditional branches",
            unsupported_switch: "LLVM IR C codegen does not yet support structured switch lowering",
        }
    }
}

fn generate_wrapper(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let mut out = String::new();
    let output = find_output_buffer(func);

    out.push_str(&format!("void sile_kernel_{}(\n", func.name));
    out.push_str("    void** buffers,\n");
    out.push_str("    int64_t num_threadgroups,\n");
    out.push_str("    int64_t threads_per_group,\n");
    out.push_str("    const int64_t* shapes,\n");
    out.push_str("    int64_t num_shapes,\n");
    out.push_str("    const float* scalars,\n");
    out.push_str("    const int64_t* scalars_i64\n");
    out.push_str(") {\n");

    let mut buffer_idx = 0usize;
    for param in &func.params {
        if is_descriptor_param(param) {
            continue;
        }
        let qualifier = c_param_type(&param.ty);
        out.push_str(&format!(
            "  {qualifier} {} = ({qualifier})buffers[{buffer_idx}];\n",
            param.name
        ));
        buffer_idx += 1;
    }
    out.push_str("  (void)num_shapes;\n");
    out.push('\n');

    // The launch grid is the tile count, which is what the caller dispatched.
    // Deriving it from a buffer's extent instead is wrong the moment a kernel
    // does not tile the buffer it reads: `parallel_reduce` launches one program
    // per 1000-element tile and atomically adds them into a one-element output,
    // so the output's extent is one and every other tile would be dropped.
    // Metal and WGSL already honor the grid, so this makes the three agree.
    out.push_str("  int64_t sile_total_tiles = num_threadgroups;\n");
    out.push('\n');

    // A tiled output is dispatched flat, one program per tile, so the single
    // index splits into the output's logical grid axes the same way MSL and
    // WGSL split it. Those axes are the output's axes the tile does not cover:
    // a rank-2 output addresses one tile with a row and a column, and a rank-3
    // output addresses a plane of tiles, so a plane axis comes first. Each
    // count is the buffer's extent along that axis divided by the tile's, and
    // comes from the shape descriptor and the tile ABI.
    //
    // Only a *tiled* output splits. A kernel that stores no tile owns one grid
    // slot per program, and its launch index is the program id: `rms_norm`
    // writes a row through a `[1, BLOCK_SIZE]` view, so reading a column count
    // out of that store would split an index that was never flat.
    let axes = output
        .as_ref()
        .map(|output| {
            let abi = output.param.abi.as_ref();
            let offset = abi.map(|abi| abi.shape_offset).unwrap_or(0);
            let rank = abi.map(|abi| abi.rank).unwrap_or(2).max(1);
            let mut axes: Vec<String> = (0..rank.saturating_sub(2))
                .map(|axis| format!("shapes[{}]", offset + axis))
                .collect();
            if rank >= 2 {
                axes.push(format!(
                    "shapes[{}] / {}",
                    offset + rank - 2,
                    output.rows.max(1)
                ));
            }
            axes.push(format!(
                "shapes[{}] / {}",
                offset + rank - 1,
                output.cols.max(1)
            ));
            axes
        })
        .unwrap_or_default();
    for (axis, extent) in axes.iter().enumerate() {
        out.push_str(&format!("  int64_t sile_tiles_{axis} = {extent};\n"));
    }

    // Publish the tile counts so `GridDim(axis)` inside the kernel reads the
    // same bound the split below uses.
    for axis in 0..3 {
        let value = axes
            .get(axis)
            .map(|_| format!("sile_tiles_{axis}"))
            .unwrap_or_else(|| "1".to_string());
        out.push_str(&format!("  sile_grid_{axis} = {value};\n"));
    }

    out.push_str("  #pragma omp parallel for schedule(static)\n");
    out.push_str("  for (int64_t tg = 0; tg < num_threadgroups; ++tg) {\n");
    out.push_str("    int64_t base = tg * threads_per_group;\n");
    out.push_str("    for (int64_t t = 0; t < threads_per_group; ++t) {\n");
    out.push_str("      int64_t sile_pid = base + t;\n");
    out.push_str("      if (sile_pid < sile_total_tiles) {\n");
    if axes.is_empty() {
        // No tiled output: one program per grid slot, so the launch index is
        // the program id and the other axes stay at zero.
        out.push_str("        sile_gid_0 = sile_pid;\n");
        out.push_str("        sile_gid_1 = 0;\n");
        out.push_str("        sile_gid_2 = 0;\n");
    } else {
        // A flat launch index counts tiles with the last axis varying fastest,
        // so each axis is the remainder against its own extent, taken in turn.
        //
        // A grid has three axes and a buffer may have more, so an axis above
        // the third has no launch index to land in. Its tile count is folded
        // away and its remainder dropped, which keeps every axis below it
        // right: a tile covers the whole of every axis above the last two, and
        // a kernel that tiles the last axis as well is one a three-axis launch
        // cannot describe, so that axis counts one and the fold costs nothing.
        out.push_str("        int64_t sile_tile = sile_pid;\n");
        for axis in (0..axes.len()).rev() {
            if axis < 3 {
                out.push_str(&format!(
                    "        sile_gid_{axis} = sile_tile % sile_tiles_{axis};\n"
                ));
            }
            out.push_str(&format!("        sile_tile /= sile_tiles_{axis};\n"));
        }
        for axis in axes.len()..3 {
            out.push_str(&format!("        sile_gid_{axis} = 0;\n"));
        }
    }
    out.push_str(&format!(
        "        sile_llvm_ir_{}({});\n",
        func.name,
        func.params
            .iter()
            .map(|param| {
                if is_shapes_param(param) {
                    "shapes".to_string()
                } else if is_scalars_param(param) {
                    "scalars".to_string()
                } else if is_scalars_i64_param(param) {
                    "scalars_i64".to_string()
                } else {
                    param.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("      }\n");
    out.push_str("    }\n");
    out.push_str("  }\n");
    out.push_str("}\n");

    Ok(out)
}

fn c_param_type(ty: &llvm_ir::Type) -> String {
    match ty {
        llvm_ir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llvm_ir::Type::F32 => "float*".to_string(),
            other => format!("{}*", c_type(other)),
        },
        other => c_type(other),
    }
}

fn c_var_decl(ty: &llvm_ir::Type, name: &str) -> String {
    match ty {
        llvm_ir::Type::I1 => format!("bool {}", name),
        llvm_ir::Type::I32 => format!("int32_t {}", name),
        llvm_ir::Type::I64 => format!("int64_t {}", name),
        llvm_ir::Type::F32 => format!("float {}", name),
        llvm_ir::Type::F64 => format!("double {}", name),
        llvm_ir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llvm_ir::Type::Array { .. } => c_ptr_decl(pointee, name),
            other => format!("{}* {} = NULL", c_type(other), name),
        },
        other => format!("{} {}", c_type(other), name),
    }
}

fn c_storage_decl(ty: &llvm_ir::Type, name: &str) -> String {
    match ty {
        llvm_ir::Type::Array { len, elem } => match elem.as_ref() {
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
            elem => format!("{} {}[{}]", c_type(elem), name, len),
        },
        other => format!("{} {}", c_type(other), name),
    }
}

fn c_ptr_decl(pointee: &llvm_ir::Type, name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{}* {} = NULL", base, name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){} = NULL", base, name, suffix)
    }
}

fn c_ptr_storage_bind_decl(pointee: &llvm_ir::Type, name: &str, storage_name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{}* {} = &{}", base, name, storage_name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){} = {}", base, name, suffix, storage_name)
    }
}

fn c_type(ty: &llvm_ir::Type) -> String {
    match ty {
        llvm_ir::Type::Void => "void".to_string(),
        llvm_ir::Type::I1 => "bool".to_string(),
        llvm_ir::Type::I32 => "int32_t".to_string(),
        llvm_ir::Type::I64 => "int64_t".to_string(),
        // The half storage is the C float of that width, not its bit pattern:
        // a load widens and a store narrows, which is the arithmetic the tile
        // around it assumes.
        llvm_ir::Type::F16 => "_Float16".to_string(),
        llvm_ir::Type::F32 => "float".to_string(),
        llvm_ir::Type::F64 => "double".to_string(),
        llvm_ir::Type::Ptr { pointee, .. } => format!("{}*", c_type(pointee)),
        llvm_ir::Type::Vector { len: _, elem: _ } => "llir_float4".to_string(),
        llvm_ir::Type::Array { len, elem } => format!("{}[{}]", c_type(elem), len),
    }
}

fn array_base_type(ty: &llvm_ir::Type) -> String {
    let mut current = ty;
    while let llvm_ir::Type::Array { elem, .. } = current {
        current = elem;
    }
    c_type(current)
}

fn intrinsic_name(intrinsic: &llvm_ir::Intrinsic) -> String {
    match intrinsic {
        llvm_ir::Intrinsic::ThreadId { dim } => format!("llir_thread_id_{}", dim),
        llvm_ir::Intrinsic::BlockId { dim } => format!("llir_block_id_{}", dim),
        llvm_ir::Intrinsic::GridDim { dim } => format!("llir_grid_dim_{}", dim),
        llvm_ir::Intrinsic::Barrier { .. } => "llir_barrier".to_string(),
        // The C library spells the float form of each one-argument float
        // function with an `f` on the end: `expf`, `log2f`, and the rest.
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
        | llvm_ir::Intrinsic::Floor => format!(
            "{}f",
            intrinsic
                .float_function()
                .expect("a float function has a name")
        ),
        llvm_ir::Intrinsic::Pow | llvm_ir::Intrinsic::Atan2 => format!(
            "{}f",
            intrinsic
                .binary_float_function()
                .expect("a binary float function has a name")
        ),
        llvm_ir::Intrinsic::VecLoad { len } => format!("llir_vec_load_{}", len),
        llvm_ir::Intrinsic::VecStore { len } => format!("llir_vec_store_{}", len),
        llvm_ir::Intrinsic::VecReduceAdd => "llir_vec_reduce_add".to_string(),
    }
}
