use crate::emit::{
    self, StructuredCfgMessages, array_dims, find_output_buffer, is_descriptor_param,
    is_scalars_i64_param, is_scalars_param, is_shapes_param,
};
use sile_lsir as lsir;

pub fn generate(func: &lsir::Function) -> sile_core::Result<String> {
    let mut code = emit::generate_text(CTargetEmitter::new(
        func,
        CDialect::new(),
        lsir::VectorPolicy::OmpSimd,
    ))?;
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
    fn param_assign(&self, ty: &lsir::Type, name: &str, arg: &str) -> String {
        emit::copy_assign(ty, name, arg)
    }

    fn var_decl(&self, ty: &lsir::Type, name: &str, is_alloca: bool) -> Option<String> {
        // An `alloca` is the only C local that owns storage. A `gep` into a
        // driver buffer has the same type but borrows, so it stays a pointer.
        match ty {
            lsir::Type::Ptr {
                addr_space: lsir::AddressSpace::Private,
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

    fn param_decl(&self, ty: &lsir::Type, _name: &str) -> String {
        c_param_type(ty)
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        let joined = params
            .iter()
            .map(|(ty, param_name)| format!("{ty} {param_name}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("void sile_lsir_{name}({joined}) {{\n")
    }

    /// C needs its headers, its per-thread launch state and — only when the
    /// body spells a vector intrinsic — its vector helpers.
    fn prelude_lines(&self, func: &lsir::Function) -> Vec<String> {
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
        let spells_vectors = func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(
                    &inst.op,
                    lsir::InstOp::Intrinsic {
                        intrinsic: lsir::Intrinsic::VecLoad { .. }
                            | lsir::Intrinsic::VecStore { .. }
                            | lsir::Intrinsic::VecReduceAdd,
                        ..
                    }
                )
            })
        });
        if spells_vectors {
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
            lines.push(
                "static inline void llir_vec_store_4(float* ptr, int64_t offset, llir_float4 v) {"
                    .to_string(),
            );
            lines.push(
                "    ptr[offset] = v.v[0]; ptr[offset+1] = v.v[1]; ptr[offset+2] = v.v[2]; ptr[offset+3] = v.v[3];"
                    .to_string(),
            );
            lines.push("}".to_string());
            lines.push("static inline float llir_vec_reduce_add(llir_float4 v) {".to_string());
            lines.push("    return v.v[0] + v.v[1] + v.v[2] + v.v[3];".to_string());
            lines.push("}".to_string());
            lines.push(String::new());
        }
        lines
    }

    fn vectorize_hint(&self, is_simple_body: bool) -> Option<String> {
        is_simple_body.then(|| "#pragma omp simd".to_string())
    }

    fn intrinsic_stmt(
        &self,
        result: Option<lsir::ValueId>,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
        intrinsic: &lsir::Intrinsic,
        args: &[lsir::Operand],
    ) -> sile_core::Result<Vec<String>> {
        // A tile-level fold keeps its shape here rather than in the lowering:
        // this is the layer that knows whether the target has a reduction
        // instruction or wants a loop.
        if let lsir::Intrinsic::TileReduce {
            reduce,
            extent,
            inner,
            outer,
        } = intrinsic
        {
            let [source] = args else {
                return Err(sile_core::Error::Compile(
                    "tile_reduce expects one argument".into(),
                ));
            };
            let id = result.ok_or_else(|| {
                sile_core::Error::Compile("tile_reduce needs a result".into())
            })?;
            let dst = value_name(id);
            let src = format_operand(source);
            // The reduction a target can fuse into one directive, or a loop
            // nest over the folded axis.
            // `max`/`min` are not C operators, so the fold spells them as a
            // conditional the compiler turns back into a max instruction.
            let combine = match reduce {
                lsir::TileReduceOp::Sum => "racc + rv",
                lsir::TileReduceOp::Product => "racc * rv",
                lsir::TileReduceOp::Max => "(racc > rv ? racc : rv)",
                lsir::TileReduceOp::Min => "(racc < rv ? racc : rv)",
            };
            let (extent, inner, outer) = (*extent, *inner, *outer);
            let span = extent * inner;
            // A fold of one contiguous run is exactly what an OpenMP reduction
            // clause is for: the compiler is free to vectorise it, which is the
            // whole point of keeping the reduction whole until here.
            // OpenMP names the arithmetic clauses by their operator.
            let clause = match reduce {
                lsir::TileReduceOp::Sum => "+",
                lsir::TileReduceOp::Product => "*",
                lsir::TileReduceOp::Max => "max",
                lsir::TileReduceOp::Min => "min",
            };
            let mut lines = Vec::new();
            if outer == 1 {
                // One row folds flat. The general spelling's `ro * span`
                // index defeats the compiler's data-reference analysis here
                // — its trip count is a known one, and the vectoriser gives
                // up on the loads before the reduction clause applies.
                // The brace keeps two flat folds in one function from both
                // declaring `racc` in the function's own scope.
                lines.push("{".to_string());
                lines.push(format!("  float racc = {src}[0];"));
                lines.push(format!("  #pragma omp simd reduction({clause}:racc)"));
                lines.push(format!("  for (int64_t ri = 1; ri < {span}; ++ri) {{"));
                lines.push(format!("    float rv = {src}[ri];"));
                lines.push(format!("    racc = {combine};"));
                lines.push("  }".to_string());
                lines.push(format!("  {dst}[0] = racc;"));
                lines.push("}".to_string());
                return Ok(lines);
            }
            lines.push(format!("for (int64_t ro = 0; ro < {outer}; ++ro) {{"));
            lines.push(format!("  float racc = {src}[ro * {span}];"));
            lines.push(format!("  #pragma omp simd reduction({clause}:racc)"));
            lines.push(format!("  for (int64_t ri = 1; ri < {span}; ++ri) {{"));
            lines.push(format!("    float rv = {src}[ro * {span} + ri];"));
            lines.push(format!("    racc = {combine};"));
            lines.push("  }".to_string());
            lines.push(format!("  {dst}[ro] = racc;"));
            lines.push("}".to_string());
            return Ok(lines);
        }
        // A tile-level multiply becomes the loop nest the C compiler turns into
        // its own vector code. The tiles arrive as pointers to their storage.
        if let lsir::Intrinsic::TileMma { m, n, k, rhs_cols } = intrinsic {
            let id = result.ok_or_else(|| {
                sile_core::Error::Compile("tile_mma needs a result".into())
            })?;
            let dst = value_name(id);
            let lhs = format_operand(&args[0]);
            let rhs = format_operand(&args[1]);
            let acc = args.get(2).map(|arg| format_operand(arg));
            let (m, n, k, rhs_cols) = (*m, *n, *k, *rhs_cols);
            let mut lines = Vec::new();
            lines.push(format!("for (int64_t mo = 0; mo < {m}; ++mo) {{"));
            lines.push(format!("  for (int64_t no = 0; no < {n}; ++no) {{"));
            lines.push(match &acc {
                Some(acc) => format!("    float racc = {acc}[mo * {n} + no];"),
                None => "    float racc = 0.0f;".to_string(),
            });
            lines.push(format!("    for (int64_t ko = 0; ko < {k}; ++ko) {{"));
            lines.push(format!(
                "      racc += {lhs}[mo * {k} + ko] * {rhs}[ko * {rhs_cols} + no];"
            ));
            lines.push("    }".to_string());
            lines.push(format!("    {dst}[mo * {n} + no] = racc;"));
            lines.push("  }".to_string());
            lines.push("}".to_string());
            return Ok(lines);
        }
        // The fused multiply-add spells as plain multiply and add: the
        // baseline C target has no FMA instruction, and the `fmaf` library
        // call the literal spelling produces sits in the way of the
        // vectorizer. The compiler contracts the pair back into a hardware
        // FMA wherever the build target has one.
        if let lsir::Intrinsic::Fma = intrinsic {
            let [a, b, c] = args else {
                return Err(sile_core::Error::Compile(
                    "a fused multiply-add takes three arguments".into(),
                ));
            };
            let id = result.ok_or_else(|| {
                sile_core::Error::Compile("a fused multiply-add needs a result".into())
            })?;
            return Ok(vec![format!(
                "{} = {} * {} + {};",
                value_name(id),
                format_operand(a),
                format_operand(b),
                format_operand(c)
            )]);
        }
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
        inst: &lsir::Inst,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
    ) -> sile_core::Result<Vec<String>> {
        // C is the only target that keeps an alloca as a real local array, so
        // it is the only one that skips the instruction and copies bytes.
        match &inst.op {
            lsir::InstOp::Alloca { .. } => Ok(vec![]),
            lsir::InstOp::Memcpy { dst, src, size } => Ok(vec![format!(
                "memcpy({}, {}, {});",
                format_operand(dst),
                format_operand(src),
                format_operand(size)
            )]),
            lsir::InstOp::AtomicAdd { ptr, value } => Ok(vec![
                "#pragma omp atomic update".to_string(),
                format!("(*{}) += {};", format_operand(ptr), format_operand(value)),
            ]),
            lsir::InstOp::Call { func, args } => {
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
            lsir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LSIR C codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // The shared lowering spells these the same way on every target —
            // or folds a gep away when the load or store reading through it is
            // its only use. A result-less operator reaching here would
            // otherwise be dropped, and the kernel would read an undefined
            // name.
            lsir::InstOp::Gep { .. } => Ok(vec![]),
            lsir::InstOp::Bin { .. }
            | lsir::InstOp::Cmp { .. }
            | lsir::InstOp::Cast { .. }
            | lsir::InstOp::Load { .. }
            | lsir::InstOp::Select { .. }
            | lsir::InstOp::Store { .. } => Err(sile_core::Error::Compile(format!(
                "LSIR C codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured C loop preheader must end with a branch",
            missing_loop_header: "missing LSIR C loop header",
            header_must_cond_br: "structured C loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured C loop body did not produce the expected backedge",
            unsupported_cond_br: "LSIR C codegen only supports structured conditional branches",
            unsupported_switch: "LSIR C codegen does not yet support structured switch lowering",
        }
    }
}

fn generate_wrapper(func: &lsir::Function) -> sile_core::Result<String> {
    let mut out = String::new();
    let output = find_output_buffer(func);

    out.push_str(&format!("void sile_kernel_{}(\n", func.name));
    out.push_str("    void** buffers,\n");
    out.push_str("    int64_t num_threadgroups,\n");
    out.push_str("    int64_t threads_per_group,\n");
    out.push_str("    const int64_t* shapes,\n");
    out.push_str("    int64_t num_shapes,\n");
    out.push_str("    const float* scalars,\n");
    out.push_str("    const int64_t* scalars_i64,\n");
    out.push_str("    const int64_t* grid\n");
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

    // How many axes the launch index splits into: the output's rank, since a
    // flat index splits along every axis the output addresses. The extents come
    // from the grid below.
    let axes = output
        .as_ref()
        .map(|output| {
            let rank = output
                .param
                .abi
                .as_ref()
                .map(|abi| abi.rank)
                .unwrap_or(2)
                .max(1);
            vec![0usize; rank]
        })
        .unwrap_or_default();

    // The grid is the launch's, not this kernel's guess. A store tile says how
    // much one program writes; it does not say how many programs there are. The
    // host partitions the buffer, so the two can differ — `rms_norm` stores one
    // `[1, BLOCK_SIZE]` tile per program while the host hands it whole rows —
    // and deriving the split from the store would then drop work or read past
    // the buffer. The launcher knows the grid, so it passes it in.
    for axis in 0..3 {
        out.push_str(&format!("  int64_t sile_tiles_{axis} = grid[{axis}];\n"));
    }

    // Publish the tile counts so `GridDim(axis)` inside the kernel reads the
    // same bound the split below uses.
    for axis in 0..3 {
        out.push_str(&format!("  sile_grid_{axis} = sile_tiles_{axis};\n"));
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
        "        sile_lsir_{}({});\n",
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

fn c_param_type(ty: &lsir::Type) -> String {
    match ty {
        lsir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            lsir::Type::F32 => "float*".to_string(),
            other => format!("{}*", c_type(other)),
        },
        other => c_type(other),
    }
}

fn c_var_decl(ty: &lsir::Type, name: &str) -> String {
    match ty {
        lsir::Type::I1 => format!("bool {}", name),
        lsir::Type::I32 => format!("int32_t {}", name),
        lsir::Type::I64 => format!("int64_t {}", name),
        lsir::Type::F32 => format!("float {}", name),
        lsir::Type::F64 => format!("double {}", name),
        lsir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            lsir::Type::Array { .. } => c_ptr_decl(pointee, name),
            other => format!("{}* {} = NULL", c_type(other), name),
        },
        other => format!("{} {}", c_type(other), name),
    }
}

fn c_storage_decl(ty: &lsir::Type, name: &str) -> String {
    match ty {
        lsir::Type::Array { len, elem } => match elem.as_ref() {
            lsir::Type::Array { .. } => {
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

fn c_ptr_decl(pointee: &lsir::Type, name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.len() <= 1 {
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

fn c_ptr_storage_bind_decl(pointee: &lsir::Type, name: &str, storage_name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        // A scalar slot's storage is the variable itself, so the pointer the
        // loads and stores name is its address.
        return format!("{}* {} = &{}", c_type(pointee), name, storage_name);
    }
    if dims.len() <= 1 {
        format!("{}* {} = {}", base, name, storage_name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){} = {}", base, name, suffix, storage_name)
    }
}

fn c_type(ty: &lsir::Type) -> String {
    match ty {
        lsir::Type::Void => "void".to_string(),
        lsir::Type::I1 => "bool".to_string(),
        lsir::Type::I32 => "int32_t".to_string(),
        lsir::Type::I64 => "int64_t".to_string(),
        // The half storage is the C float of that width, not its bit pattern:
        // a load widens and a store narrows, which is the arithmetic the tile
        // around it assumes.
        lsir::Type::F16 => "_Float16".to_string(),
        lsir::Type::F32 => "float".to_string(),
        lsir::Type::F64 => "double".to_string(),
        lsir::Type::Ptr { pointee, .. } => format!("{}*", c_type(pointee)),
        lsir::Type::Vector { len: _, elem: _ } => "llir_float4".to_string(),
        lsir::Type::Array { len, elem } => format!("{}[{}]", c_type(elem), len),
    }
}

fn array_base_type(ty: &lsir::Type) -> String {
    let mut current = ty;
    while let lsir::Type::Array { elem, .. } = current {
        current = elem;
    }
    c_type(current)
}

fn intrinsic_name(intrinsic: &lsir::Intrinsic) -> String {
    match intrinsic {
        lsir::Intrinsic::ThreadId { dim } => format!("llir_thread_id_{}", dim),
        lsir::Intrinsic::BlockId { dim } => format!("llir_block_id_{}", dim),
        lsir::Intrinsic::GridDim { dim } => format!("llir_grid_dim_{}", dim),
        lsir::Intrinsic::Barrier { .. } => "llir_barrier".to_string(),
        // The C library spells the float form of each one-argument float
        // function with an `f` on the end: `expf`, `log2f`, and the rest.
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
        | lsir::Intrinsic::Floor => format!(
            "{}f",
            intrinsic
                .float_function()
                .expect("a float function has a name")
        ),
        lsir::Intrinsic::Pow | lsir::Intrinsic::Atan2 => format!(
            "{}f",
            intrinsic
                .binary_float_function()
                .expect("a binary float function has a name")
        ),
        lsir::Intrinsic::Fma => format!(
            "{}f",
            intrinsic
                .ternary_float_function()
                .expect("a ternary float function has a name")
        ),
        lsir::Intrinsic::VecLoad { len } => format!("llir_vec_load_{}", len),
        lsir::Intrinsic::VecStore { len } => format!("llir_vec_store_{}", len),
        lsir::Intrinsic::VecReduceAdd => "llir_vec_reduce_add".to_string(),
        // Both tile-level forms are expanded into loops by `intrinsic_stmt`, so
        // no helper carries their name.
        lsir::Intrinsic::TileReduce { .. } | lsir::Intrinsic::TileMma { .. } => {
            unreachable!("a tile-level intrinsic is expanded, not called")
        }
    }
}
