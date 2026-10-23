use crate::emit::{self, StructuredCfgMessages, TargetDialect, array_dims};
use sile_lsir as lsir;
use std::collections::HashMap;

pub fn generate(func: &lsir::Function) -> sile_core::Result<String> {
    emit::generate_text(emit::TargetEmitter::new(
        func,
        MetalDialect::new(func),
        lsir::VectorPolicy::None,
    ))
}

/// The MSL dialect: one spelling table, plus the tile arithmetic that MSL reads
/// from buffer shapes.
///
/// The launch indices arrive as per-axis builtins — one threadgroup owns one
/// tile, so the threadgroup's position in the grid is the tile index, and the
/// dispatch shape itself answers a `grid_dim` query. No ABI reconstruction.
struct MetalDialect<'a> {
    func: &'a lsir::Function,
    /// Address space of every pointer-typed instruction result, so a vector
    /// load spells the cast its pointer actually names: a relay over private
    /// tile storage must not cast through a `device` pointer.
    ptr_spaces: HashMap<lsir::ValueId, lsir::AddressSpace>,
}

impl<'a> MetalDialect<'a> {
    fn new(func: &'a lsir::Function) -> Self {
        let ptr_spaces = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| {
                let id = inst.result?;
                match &inst.ty {
                    lsir::Type::Ptr { addr_space, .. } => Some((id, addr_space.clone())),
                    _ => None,
                }
            })
            .collect();
        Self { func, ptr_spaces }
    }

    /// Address-space qualifier of a vector access's pointer, `device` for a
    /// kernel buffer argument, which no instruction defines.
    fn vec_qualifier(&self, ptr: &lsir::Operand) -> &'static str {
        match ptr {
            lsir::Operand::Value(id) => self
                .ptr_spaces
                .get(id)
                .map(metal_addr_qual)
                .unwrap_or("device "),
            lsir::Operand::Const(_) => "device ",
        }
    }
}

impl TargetDialect for MetalDialect<'_> {
    /// A tile is an address here, so a parameter takes a copy of the elements
    /// rather than the address of storage the next iteration overwrites.
    fn param_assign(&self, ty: &lsir::Type, name: &str, arg: &str) -> String {
        emit::copy_assign(ty, name, arg)
    }

    fn var_decl(&self, ty: &lsir::Type, name: &str, _is_alloca: bool) -> Option<String> {
        // MSL has no function-local pointer storage, so a private or
        // threadgroup pointer binds to a local array in its own address
        // space. Unlike C, every storage pointer behaves the same way here,
        // whether it came from an `alloca` or a `gep`.
        match ty {
            lsir::Type::Ptr { addr_space, pointee }
                if matches!(
                    addr_space,
                    lsir::AddressSpace::Private | lsir::AddressSpace::Shared
                ) =>
            {
                let qualifier = metal_addr_qual(addr_space);
                let storage_name = format!("{name}_storage");
                Some(format!(
                    "{}; {};",
                    metal_storage_decl(qualifier, pointee, &storage_name),
                    metal_ptr_bind_decl(qualifier, pointee, name, &storage_name)
                ))
            }
            _ => Some(format!("{};", metal_var_decl(ty, name))),
        }
    }

    fn param_decl(&self, ty: &lsir::Type, _name: &str) -> String {
        metal_type(ty)
    }

    fn signature(&self, name: &str, params: &[(String, String)]) -> String {
        // One threadgroup owns one tile, so the launch indices arrive as
        // separate builtins, one per axis: the threadgroup's position in the
        // grid is the tile index, the thread's slot inside the threadgroup is
        // what the mapping's strips index with, and the dispatch shape itself
        // answers a grid_dim query. Under a serial dispatch the threadgroup
        // holds one thread and the builtins still read the launch as written.
        let mut out = format!("kernel void sile_kernel_{name}(\n");
        for (idx, (ty, param_name)) in params.iter().enumerate() {
            out.push_str(&format!("    {ty} {param_name} [[buffer({idx})]],\n"));
        }
        out.push_str(
            "    uint3 tgid [[threadgroup_position_in_grid]],\n    \
             uint3 tid [[thread_position_in_threadgroup]],\n    \
             uint simd_lid [[thread_index_in_simdgroup]],\n    \
             uint simd_gid [[simdgroup_index_in_threadgroup]],\n    \
             uint3 tgpog [[threadgroups_per_grid]]) {\n",
        );
        out
    }

    fn prelude_lines(&self, _func: &lsir::Function) -> Vec<String> {
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
        result: Option<lsir::ValueId>,
        value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
        intrinsic: &lsir::Intrinsic,
        args: &[lsir::Operand],
    ) -> sile_core::Result<Vec<String>> {
        // A vector store is a plain assignment and takes no result; a barrier
        // orders the tile's threads and takes none either; the staged
        // SIMD-group multiply lands its tile in the out buffer, so it names
        // nothing; every other intrinsic names one.
        let named = match intrinsic {
            lsir::Intrinsic::VecStore { .. }
            | lsir::Intrinsic::Barrier { .. }
            | lsir::Intrinsic::SimdGroupMma { .. } => None,
            _ => Some(result.ok_or_else(|| {
                sile_core::Error::Compile(format!(
                    "{intrinsic:?} in LSIR Metal codegen must produce a result"
                ))
            })?),
        };
        let name = || value_name(named.expect("every other intrinsic names its result"));
        let line = match intrinsic {
            lsir::Intrinsic::BlockId { dim } => {
                format!("{} = (int64_t)tgid.{};", name(), metal_launch_axis(*dim)?)
            }
            lsir::Intrinsic::ThreadId { dim } => {
                format!("{} = (int64_t)tid.{};", name(), metal_launch_axis(*dim)?)
            }
            lsir::Intrinsic::GridDim { dim } => {
                format!("{} = (int64_t)tgpog.{};", name(), metal_launch_axis(*dim)?)
            }
            // The hardware fold across one SIMD-group. The mapped tree plants
            // it where every lane of the group arrives together, and a
            // partially filled group folds its active lanes.
            lsir::Intrinsic::SimdReduce { reduce } => {
                let [value] = args else {
                    return Err(sile_core::Error::Compile(
                        "simd_reduce intrinsic expects one argument".into(),
                    ));
                };
                let function = match reduce {
                    lsir::TileReduceOp::Sum => "simd_sum",
                    lsir::TileReduceOp::Product => "simd_product",
                    lsir::TileReduceOp::Max => "simd_max",
                    lsir::TileReduceOp::Min => "simd_min",
                };
                format!("{} = {}({});", name(), function, format_operand(value))
            }
            // The SIMD-group coordinates come from the spelled builtins, never
            // `tid % 32`: the hardware owns the map from thread index to
            // SIMD-group.
            lsir::Intrinsic::SimdLaneId => {
                format!("{} = (int64_t)simd_lid;", name())
            }
            lsir::Intrinsic::SimdGroupId => {
                format!("{} = (int64_t)simd_gid;", name())
            }
            // The mapping plants one barrier per tree step and shared-panel
            // handoff, all inside one tile's threadgroup: the scope MSL's
            // threadgroup barrier names.
            lsir::Intrinsic::Barrier { .. } => {
                "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()
            }
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
                let function = intrinsic
                    .float_function()
                    .expect("a float function has a name");
                format!("{} = metal::{}({});", name(), function, format_operand(arg))
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
                    "{} = metal::{}({}, {});",
                    name(),
                    function,
                    format_operand(lhs),
                    format_operand(rhs)
                )
            }
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
                    "{} = metal::{}({}, {}, {});",
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
                        "vec_load intrinsic expects [ptr, offset] arguments".into(),
                    ));
                };
                let ptr_text = format!("({} + {})", format_operand(ptr), format_operand(offset));
                if self.vec_qualifier(ptr) == "device " {
                    // A kernel buffer's base is aligned and its offset is a
                    // whole vector, so the packed cast is safe.
                    format!("{} = *((device float{}*){ptr_text});", name(), len)
                } else {
                    // Private and threadgroup storage carry no alignment
                    // promise, so the elements gather one by one — the same
                    // spelling WGSL uses.
                    let items = (0..*len)
                        .map(|k| format!("{ptr_text}[{k}]"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{} = float{}({items});", name(), len)
                }
            }
            lsir::Intrinsic::VecStore { len } => {
                let [ptr, offset, value] = args else {
                    return Err(sile_core::Error::Compile(
                        "vec_store intrinsic expects [ptr, offset, value] arguments".into(),
                    ));
                };
                let value_text = format_operand(value);
                let ptr_text = format!("({} + {})", format_operand(ptr), format_operand(offset));
                if self.vec_qualifier(ptr) == "device " {
                    format!("*((device float{}*){ptr_text}) = {value_text};", len)
                } else {
                    let mut lines = Vec::new();
                    for k in 0..*len {
                        lines.push(format!("{ptr_text}[{k}] = {value_text}[{k}];"));
                    }
                    return Ok(lines);
                }
            }
            lsir::Intrinsic::VecReduceAdd => {
                let [vector] = args else {
                    return Err(sile_core::Error::Compile(
                        "vec_reduce_add intrinsic expects one argument".into(),
                    ));
                };
                let v = format_operand(vector);
                format!("{} = {v}[0] + {v}[1] + {v}[2] + {v}[3];", name())
            }
            // A tile-level fold is a loop here, the same way WGSL spells it:
            // MSL has no reduction directive either.
            lsir::Intrinsic::TileReduce {
                reduce,
                extent,
                inner,
                outer,
            } => {
                let [source] = args else {
                    return Err(sile_core::Error::Compile(
                        "tile_reduce intrinsic expects one argument".into(),
                    ));
                };
                let src = format_operand(source);
                let dst = name();
                let span = *extent * *inner;
                let combine = match reduce {
                    lsir::TileReduceOp::Sum => "racc + rv",
                    lsir::TileReduceOp::Product => "racc * rv",
                    lsir::TileReduceOp::Max => "max(racc, rv)",
                    lsir::TileReduceOp::Min => "min(racc, rv)",
                };
                return Ok(vec![
                    format!("for (int ro = 0; ro < {outer}; ++ro) {{"),
                    format!("  float racc = {src}[ro * {span}];"),
                    format!("  for (int ri = 1; ri < {span}; ++ri) {{"),
                    format!("    float rv = {src}[ro * {span} + ri];"),
                    format!("    racc = {combine};"),
                    "  }".to_string(),
                    format!("  {dst}[ro] = racc;"),
                    "}".to_string(),
                ]);
            }
            // The serial rung of the mma ladder: plain FMA loops over the
            // panels. The strip policy reaches here only after the parallel
            // mapping declined, and a correct slow kernel is what a fallback
            // owes.
            lsir::Intrinsic::TileMma {
                m,
                n,
                k,
                rhs_cols,
            } => {
                let [lhs, rhs, ..] = args else {
                    return Err(sile_core::Error::Compile(
                        "tile_mma intrinsic expects at least two arguments".into(),
                    ));
                };
                let lhs = format_operand(lhs);
                let rhs = format_operand(rhs);
                let acc = args.get(2).map(format_operand);
                let dst = name();
                let (m, n, k, rhs_cols) = (*m, *n, *k, *rhs_cols);
                let seed = match &acc {
                    Some(_acc) => format!("{dst}[mo * {n} + no]"),
                    None => "0.0f".to_string(),
                };
                // The loop variables spell as `int`, never `int64_t`: AIR
                // miscompiles the read-modify-write tile update when the
                // subscripts are 64-bit (probe: the accumulator collapses to
                // one element). WGSL already spells these loops as i32, so
                // both targets agree.
                let mut lines = Vec::new();
                if let Some(acc) = &acc {
                    lines.push(format!(
                        "for (int mi = 0; mi < {}; ++mi) {{ {dst}[mi] = {acc}[mi]; }}",
                        m * n
                    ));
                }
                lines.push(format!("for (int mo = 0; mo < {m}; ++mo) {{"));
                lines.push(format!("  for (int no = 0; no < {n}; ++no) {{"));
                lines.push(format!("    float racc = {seed};"));
                lines.push(format!("    for (int ko = 0; ko < {k}; ++ko) {{"));
                lines.push(format!(
                    "      racc += {lhs}[mo * {k} + ko] * {rhs}[ko * {rhs_cols} + no];"
                ));
                lines.push("    }".to_string());
                lines.push(format!("    {dst}[mo * {n} + no] = racc;"));
                lines.push("  }".to_string());
                lines.push("}".to_string());
                return Ok(lines);
            }
            // The SIMD-group rung of the mma ladder: the staged multiply the
            // mapping left whole, spelled as 8x8 `simdgroup_matrix` fragments
            // partitioned across the workgroup's SIMD-groups. A group's
            // fragment block seeds from the staged accumulator (or zero),
            // multiplies the k fragments in — the B fragment transposed out
            // of the row-major panel by the load — and stores the block back
            // to the staging buffer the copy-out strip reads. All of it sits
            // behind one guard so a group with no block idles instead of
            // addressing past the tile.
            lsir::Intrinsic::SimdGroupMma {
                m,
                n,
                k,
                rhs_cols,
                row_groups,
                col_groups,
            } => {
                let [lhs, rhs, rest @ ..] = args else {
                    return Err(sile_core::Error::Compile(
                        "simd_group_mma intrinsic expects at least three arguments".into(),
                    ));
                };
                let (acc, out) = match rest {
                    [acc, out] => (Some(acc), out),
                    [out] => (None, out),
                    _ => {
                        return Err(sile_core::Error::Compile(
                            "simd_group_mma intrinsic expects [lhs, rhs, out] or [lhs, rhs, acc, out]"
                                .into(),
                        ));
                    }
                };
                let lhs = format_operand(lhs);
                let rhs = format_operand(rhs);
                let acc = acc.map(|operand| format_operand(operand));
                let out = format_operand(out);
                let (m, n, k, rhs_cols) = (*m as u32, *n as u32, *k as u32, *rhs_cols as u32);
                let frag_m = m / 8 / row_groups;
                let frag_n = n / 8 / col_groups;
                let frags = frag_m * frag_n;
                let k_frags = k / 8;
                // The array's name rides on the out buffer's: one staging
                // buffer per site, so one fragment set per site, with no
                // counter to keep deterministic.
                let acc_arr = format!("{out}_acc");
                let mut lines = Vec::new();
                lines.push(format!("if (simd_gid < {}ul) {{", row_groups * col_groups));
                lines.push(format!(
                    "  const int br = (int)(simd_gid / {col_groups}ul) * {frag_m} * 8;"
                ));
                lines.push(format!(
                    "  const int bc = (int)(simd_gid % {col_groups}ul) * {frag_n} * 8;"
                ));
                lines.push(format!("  simdgroup_float8x8 {acc_arr}[{frags}];"));
                match &acc {
                    Some(acc) => lines.push(format!(
                        "  for (int fi = 0; fi < {frags}; ++fi) {{ simdgroup_load({acc_arr}[fi], {acc}, (ulong){n}, ulong2(bc + (fi % {frag_n}) * 8, br + (fi / {frag_n}) * 8), false); }}"
                    )),
                    None => lines.push(format!(
                        "  for (int fi = 0; fi < {frags}; ++fi) {{ {acc_arr}[fi] = simdgroup_float8x8(0.0f); }}"
                    )),
                };
                lines.push("  simdgroup_float8x8 fa;".to_string());
                lines.push("  simdgroup_float8x8 fb;".to_string());
                lines.push(format!("  for (int kf = 0; kf < {k_frags}; ++kf) {{"));
                lines.push(format!("    for (int fi = 0; fi < {frags}; ++fi) {{"));
                lines.push(format!(
                    "      const int row0 = br + (fi / {frag_n}) * 8;"
                ));
                lines.push(format!(
                    "      const int col0 = bc + (fi % {frag_n}) * 8;"
                ));
                lines.push(format!(
                    "      simdgroup_load(fa, {lhs}, (ulong){k}, ulong2(kf * 8, row0), false);"
                ));
                lines.push(format!(
                    "      simdgroup_load(fb, {rhs}, (ulong){rhs_cols}, ulong2(col0, kf * 8), true);"
                ));
                lines.push(format!(
                    "      simdgroup_multiply_accumulate({acc_arr}[fi], fa, fb, {acc_arr}[fi]);"
                ));
                lines.push("    }".to_string());
                lines.push("  }".to_string());
                lines.push(format!("  for (int fi = 0; fi < {frags}; ++fi) {{"));
                lines.push(format!(
                    "    const int row0 = br + (fi / {frag_n}) * 8;"
                ));
                lines.push(format!(
                    "    const int col0 = bc + (fi % {frag_n}) * 8;"
                ));
                lines.push(format!(
                    "    simdgroup_store({acc_arr}[fi], {out}, (ulong){n}, ulong2(col0, row0), false);"
                ));
                lines.push("  }".to_string());
                lines.push("}".to_string());
                return Ok(lines);
            }
        };
        Ok(vec![line])
    }

    fn extra_inst_stmt(
        &mut self,
        inst: &lsir::Inst,
        _value_name: impl Fn(lsir::ValueId) -> String,
        format_operand: impl Fn(&lsir::Operand) -> String,
    ) -> sile_core::Result<Vec<String>> {
        // MSL has no `memcpy` builtin and no ordinary function calls, so both
        // become explicit loops or an error.
        match &inst.op {
            lsir::InstOp::Memcpy { dst, src, size } => Ok(vec![
                format!(
                    "for (int copy_i = 0; copy_i < {}; ++copy_i) {{",
                    format_operand(size)
                ),
                format!(
                    "{}[copy_i] = {}[copy_i];",
                    format_operand(dst),
                    format_operand(src)
                ),
                "}".to_string(),
            ]),
            lsir::InstOp::AtomicAdd { ptr, value } => Ok(vec![format!(
                "atomic_fetch_add_explicit((device atomic_float*)({}), {}, memory_order_relaxed);",
                format_operand(ptr),
                format_operand(value)
            )]),
            lsir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LSIR Metal codegen does not support call instructions: {func}"
            ))),
            // An intrinsic never reaches here: the shared emitter dispatches it
            // to `intrinsic_stmt` before this match is consulted.
            lsir::InstOp::Intrinsic { .. } => Err(sile_core::Error::Compile(
                "LSIR Metal codegen cannot spell an intrinsic in extra_inst_stmt".into(),
            )),
            // An alloca is the declaration the emitter already wrote, and a
            // `gep` is spelled at its use: neither leaves a statement here.
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
                "LSIR Metal codegen cannot spell the instruction {:?}; the shared lowering was expected to take it",
                inst.op
            ))),
        }
    }

    fn messages(&self) -> StructuredCfgMessages {
        StructuredCfgMessages {
            preheader_must_branch: "structured Metal loop preheader must end with a branch",
            missing_loop_header: "missing LSIR structured Metal loop header",
            header_must_cond_br: "structured Metal loop header must end with a conditional branch",
            loop_backedge_mismatch: "structured Metal loop body did not produce the expected backedge",
            unsupported_cond_br: "LSIR Metal codegen only supports structured conditional branches",
            unsupported_switch: "LSIR Metal codegen does not yet support switch terminators",
        }
    }
}

/// The MSL component of launch-axis `dim`.
fn metal_launch_axis(dim: u8) -> sile_core::Result<&'static str> {
    match dim {
        0 => Ok("x"),
        1 => Ok("y"),
        2 => Ok("z"),
        other => Err(sile_core::Error::Compile(format!(
            "Metal codegen only supports launch axes 0..=2, got {other}"
        ))),
    }
}

fn metal_var_decl(ty: &lsir::Type, name: &str) -> String {
    match ty {
        lsir::Type::I1 => format!("bool {} = false", name),
        lsir::Type::I32 => format!("int32_t {} = 0", name),
        lsir::Type::I64 => format!("int64_t {} = 0", name),
        lsir::Type::F32 => format!("float {} = 0.0f", name),
        lsir::Type::F64 => format!("double {} = 0.0", name),
        lsir::Type::Ptr {
            addr_space,
            pointee,
        } => metal_ptr_zero_decl(addr_space, pointee, name),
        lsir::Type::Vector { len, elem: _ } => {
            format!("float{} {} = {{0}}", len, name)
        }
        other => format!("{} {}", metal_type(other), name),
    }
}

fn metal_storage_decl(qualifier: &str, ty: &lsir::Type, name: &str) -> String {
    match ty {
        lsir::Type::Array { .. } => {
            let dims = array_dims(ty);
            let base = array_base_type(ty);
            format!(
                "{qualifier}{base} {name}{}",
                dims.iter()
                    .map(|dim| format!("[{dim}]"))
                    .collect::<Vec<_>>()
                    .join("")
            )
        }
        other => format!("{qualifier}{} {name}", metal_type(other)),
    }
}

fn metal_ptr_bind_decl(
    qualifier: &str,
    pointee: &lsir::Type,
    name: &str,
    storage_name: &str,
) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{qualifier}{base}* {name} = &{storage_name};")
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{qualifier}{base} (*{name}){suffix} = {storage_name};")
    }
}

fn metal_ptr_zero_decl(
    addr_space: &lsir::AddressSpace,
    pointee: &lsir::Type,
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

fn metal_addr_qual(addr_space: &lsir::AddressSpace) -> &'static str {
    match addr_space {
        lsir::AddressSpace::Generic => "",
        lsir::AddressSpace::Private => "thread ",
        lsir::AddressSpace::Global => "device ",
        lsir::AddressSpace::Constant => "constant ",
        lsir::AddressSpace::Shared => "threadgroup ",
    }
}

fn metal_type(ty: &lsir::Type) -> String {
    match ty {
        lsir::Type::Void => "void".into(),
        lsir::Type::I1 => "bool".into(),
        lsir::Type::I32 => "int32_t".into(),
        lsir::Type::I64 => "int64_t".into(),
        lsir::Type::F16 => "half".into(),
        lsir::Type::F32 => "float".into(),
        lsir::Type::F64 => "double".into(),
        lsir::Type::Ptr {
            addr_space,
            pointee,
        } => format!(
            "{}{}*",
            metal_addr_qual(addr_space),
            metal_scalar_type(pointee)
        ),
        lsir::Type::Vector { len, elem } => format!("{}{}", metal_scalar_type(elem), len),
        lsir::Type::Array { len, elem } => format!("{}[{}]", metal_type(elem), len),
    }
}

fn metal_scalar_type(ty: &lsir::Type) -> String {
    match ty {
        lsir::Type::Array { .. } => array_base_type(ty),
        other => metal_type(other),
    }
}

fn array_base_type(ty: &lsir::Type) -> String {
    let mut current = ty;
    while let lsir::Type::Array { elem, .. } = current {
        current = elem;
    }
    metal_type(current)
}
