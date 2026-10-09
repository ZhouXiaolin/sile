use crate::emit::{self, StructuredCfgMessages, find_output_buffer, array_dims};
use sile_llvm_ir as llvm_ir;

const SHAPES_PARAM_NAME: &str = "__sile_shapes";

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
            lines.push(format!("static _Thread_local int64_t sile_grid_{axis} = 1;"));
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
        lines.push("static inline llir_float4 llir_vec_load_4(const float* ptr, int64_t offset) {".to_string());
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
            _ => Ok(vec![]),
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


fn emit_tile_count_logic(
    out: &mut String,
    abi: &llvm_ir::ParamAbi,
    tile_rows: usize,
    tile_cols: usize,
) {
    if abi.rank == 1 {
        // The extent is in elements, and one program owns `tile_cols` of them,
        // or `num_programs` reports the element count instead of the tile count.
        out.push_str(&format!(
            "  int64_t sile_total_tiles = shapes[{}] / {};\n",
            abi.shape_offset,
            tile_cols.max(1)
        ));
    } else {
        out.push_str(&format!(
            "  int64_t sile_tiles_n = shapes[{}] / {};\n",
            abi.shape_offset + 1,
            tile_cols
        ));
        out.push_str(&format!(
            "  int64_t sile_total_tiles = (shapes[{}] / {}) * sile_tiles_n;\n",
            abi.shape_offset, tile_rows
        ));
    }
}

fn generate_wrapper(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let mut out = String::new();
    let output = find_output_buffer(func);
    let output_rank = output
        .as_ref()
        .and_then(|out| out.param.abi.as_ref().map(|abi| abi.rank))
        .or_else(|| {
            // No tile ABI: fall back to rank only, so the wrapper still compiles.
            func.params
                .iter()
                .filter(|param| !is_shapes_param(param))
                .last()
                .and_then(|param| param.abi.as_ref().map(|abi| abi.rank))
        });

    out.push_str(&format!("void sile_kernel_{}(\n", func.name));
    out.push_str("    void** buffers,\n");
    out.push_str("    int64_t num_threadgroups,\n");
    out.push_str("    int64_t threads_per_group,\n");
    out.push_str("    const int64_t* shapes,\n");
    out.push_str("    int64_t num_shapes\n");
    out.push_str(") {\n");

    let mut buffer_idx = 0usize;
    for param in &func.params {
        if is_shapes_param(param) {
            continue;
        }
        let qualifier = "float*";
        out.push_str(&format!(
            "  {qualifier} {} = ({qualifier})buffers[{buffer_idx}];\n",
            param.name
        ));
        buffer_idx += 1;
    }
    out.push_str("  (void)num_shapes;\n");
    out.push('\n');

    if let Some(output) = &output {
        let abi = output.param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile(
                "LLVM IR CPU wrapper requires output parameter ABI metadata".into(),
            )
        })?;
        emit_tile_count_logic(&mut out, abi, output.rows, output.cols);
    } else {
        // No tile ABI at all: cover the buffer with one thread per tile-sized
        // slice, taken from the first buffer's leading extent.
        let first_extent = func
            .params
            .first()
            .filter(|param| !is_shapes_param(param))
            .or_else(|| func.params.iter().find(|param| !is_shapes_param(param)))
            .and_then(|param| param.abi.as_ref())
            .map(|abi| format!("shapes[{}]", abi.shape_offset))
            .unwrap_or_else(|| "num_threadgroups * threads_per_group".into());
        out.push_str(&format!("  int64_t sile_total_tiles = {first_extent};\n"));
    }
    out.push('\n');

    // Publish the tile count so `GridDim(0)` inside the kernel reads the same
    // bound the pid loop uses. The CPU grid is flattened, so axes 1 and 2 stay
    // one tile deep.
    out.push_str("  sile_grid_0 = sile_total_tiles;\n");
    out.push_str("  sile_grid_1 = 1;\n");
    out.push_str("  sile_grid_2 = 1;\n");

    out.push_str("  #pragma omp parallel for schedule(static)\n");
    out.push_str("  for (int64_t tg = 0; tg < num_threadgroups; ++tg) {\n");
    out.push_str("    int64_t base = tg * threads_per_group;\n");
    out.push_str("    for (int64_t t = 0; t < threads_per_group; ++t) {\n");
    out.push_str("      int64_t sile_pid = base + t;\n");
    out.push_str("      if (sile_pid < sile_total_tiles) {\n");
    match output_rank.unwrap_or(1) {
        0 | 1 => {
            out.push_str("        sile_gid_0 = sile_pid;\n");
            out.push_str("        sile_gid_1 = 0;\n");
        }
        _ => {
            out.push_str("        sile_gid_0 = sile_pid / sile_tiles_n;\n");
            out.push_str("        sile_gid_1 = sile_pid % sile_tiles_n;\n");
        }
    }
    out.push_str("        sile_gid_2 = 0;\n");
    out.push_str(&format!(
        "        sile_llvm_ir_{}({});\n",
        func.name,
        func.params
            .iter()
            .map(|param| {
                if is_shapes_param(param) {
                    "shapes".to_string()
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

fn is_shapes_param(param: &llvm_ir::Param) -> bool {
    param.name == SHAPES_PARAM_NAME
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
        llvm_ir::Type::F16 => "uint16_t".to_string(),
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
        llvm_ir::Intrinsic::Exp => "expf".to_string(),
        llvm_ir::Intrinsic::VecLoad { len } => format!("llir_vec_load_{}", len),
        llvm_ir::Intrinsic::VecStore { len } => format!("llir_vec_store_{}", len),
        llvm_ir::Intrinsic::VecReduceAdd => "llir_vec_reduce_add".to_string(),
    }
}


#[cfg(test)]
mod tests {
    use super::generate;
    use sile_llvm_ir::{
        AddressSpace, BasicBlock, BlockId, Constant, Function, Inst, InstOp, Operand, Terminator,
        Type, ValueId,
    };

    #[test]
    fn gep_private_ptrs_do_not_materialize_fake_storage_bindings() {
        let func = Function {
            name: "ptr_decl".into(),
            params: vec![],
            blocks: vec![BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![
                    Inst {
                        result: Some(ValueId(0)),
                        result_name: Some("tile".into()),
                        ty: Type::ptr(
                            AddressSpace::Private,
                            Type::array(2, Type::array(2, Type::F32)),
                        ),
                        op: InstOp::Alloca {
                            alloc_ty: Type::array(2, Type::array(2, Type::F32)),
                            addr_space: AddressSpace::Private,
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(1)),
                        result_name: Some("elt".into()),
                        ty: Type::ptr(AddressSpace::Private, Type::F32),
                        op: InstOp::Gep {
                            base: Operand::Value(ValueId(0)),
                            indices: vec![
                                Operand::Const(Constant::Int(0)),
                                Operand::Const(Constant::Int(1)),
                            ],
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::Ret { value: None },
            }],
            entry: BlockId(0),
            metadata: vec![],
        };

        let c = generate(&func).unwrap();
        assert!(c.contains("float tile_storage[2][2];"));
        assert!(c.contains("float* elt = NULL;"));
        assert!(!c.contains("float elt_storage;"));
        assert!(!c.contains("float* elt = &elt_storage;"));
    }
}
