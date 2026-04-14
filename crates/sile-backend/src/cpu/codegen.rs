use std::collections::{HashMap, HashSet};

use crate::emit::{
    self, StructuredCfgMessages, StructuredEmitter, TextCodegen, array_dims,
    build_value_names, infer_tile_plan, value_name as llir_value_name,
};
use sile_llvm_ir as llvm_ir;

const SHAPES_PARAM_NAME: &str = "__sile_shapes";

pub fn generate(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let mut code = emit::generate_text(CCodegen::new(func))?;
    let wrapper = generate_wrapper(func)?;
    code.push('\n');
    code.push_str(&wrapper);
    Ok(code)
}

struct CCodegen<'a> {
    func: &'a llvm_ir::Function,
    value_names: HashMap<llvm_ir::ValueId, String>,
    alloca_results: HashSet<llvm_ir::ValueId>,
    declared_values: HashSet<llvm_ir::ValueId>,
    indent: usize,
    out: String,
}

impl<'a> CCodegen<'a> {
    fn new(func: &'a llvm_ir::Function) -> Self {
        Self {
            func,
            value_names: build_value_names(func),
            alloca_results: collect_alloca_results(func),
            declared_values: HashSet::new(),
            indent: 0,
            out: String::new(),
        }
    }

    fn emit_prelude(&mut self) {
        self.out.push_str("#include <stdint.h>\n");
        self.out.push_str("#include <stdbool.h>\n");
        self.out.push_str("#include <omp.h>\n");
        self.out.push_str("#include <math.h>\n");
        self.out.push_str("#include <string.h>\n\n");

        self.out
            .push_str("static _Thread_local int64_t sile_gid_0 = 0;\n");
        self.out
            .push_str("static _Thread_local int64_t sile_gid_1 = 0;\n");
        self.out
            .push_str("static _Thread_local int64_t sile_gid_2 = 0;\n");
        self.out.push('\n');

        self.emit_runtime_helpers();
    }

    fn emit_runtime_helpers(&mut self) {
        self.out
            .push_str("#define llir_block_id_0() (sile_gid_0)\n");
        self.out
            .push_str("#define llir_block_id_1() (sile_gid_1)\n");
        self.out
            .push_str("#define llir_block_id_2() (sile_gid_2)\n");
        self.out.push_str("#define llir_thread_id_0() (0)\n");
        self.out.push_str("#define llir_thread_id_1() (0)\n");
        self.out.push_str("#define llir_thread_id_2() (0)\n");
        self.out.push_str("#define llir_barrier() ((void)0)\n");
        self.out.push('\n');

        // Vector load/reduce helpers (used by vectorized reduce loop)
        self.out.push_str("typedef struct { float v[4]; } llir_float4;\n");
        self.out.push_str("static inline llir_float4 llir_vec_load_4(const float* ptr, int64_t offset) {\n");
        self.out.push_str("    llir_float4 r;\n");
        self.out.push_str("    r.v[0] = ptr[offset]; r.v[1] = ptr[offset+1]; r.v[2] = ptr[offset+2]; r.v[3] = ptr[offset+3];\n");
        self.out.push_str("    return r;\n");
        self.out.push_str("}\n");
        self.out.push_str("static inline float llir_vec_reduce_add(llir_float4 v) {\n");
        self.out.push_str("    return v.v[0] + v.v[1] + v.v[2] + v.v[3];\n");
        self.out.push_str("}\n");
        self.out.push('\n');
    }

    fn emit_signature(&mut self) {
        self.out.push_str(&format!(
            "void sile_llvm_ir_{}({}) {{\n",
            self.func.name,
            self.func
                .params
                .iter()
                .map(|param| format!(
                    "{} {}",
                    c_param_type(&param.ty),
                    self.value_names
                        .get(&param.id)
                        .cloned()
                        .unwrap_or_else(|| format!("v{}", param.id.0))
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        self.indent = 1;
    }

    fn emit_body(&mut self) -> sile_core::Result<()> {
        self.emit_value_decls();
        self.writeln("");
        if self.emit_structured_from(self.func.entry, &[])?.is_some() {
            return Err(sile_core::Error::Compile(
                "structured C codegen unexpectedly stopped at a non-terminal block".into(),
            ));
        }

        self.indent = 0;
        self.out.push_str("}\n");
        Ok(())
    }

    fn emit_value_decls(&mut self) {
        let mut emitted_any = false;
        for block in &self.func.blocks {
            for param in &block.params {
                if self.declared_values.insert(param.id) {
                    self.emit_decl(param.id, &param.ty);
                    emitted_any = true;
                }
            }
            for inst in &block.insts {
                if let Some(id) = inst.result {
                    // Declare all computed values at the top to avoid
                    // use-before-declaration issues when structured CFG
                    // emission reorders blocks
                    if self.declared_values.insert(id) {
                        self.emit_decl(id, &inst.ty);
                        emitted_any = true;
                    }
                }
            }
        }

        if emitted_any {
            self.writeln("");
        }
    }

    fn emit_decl(&mut self, id: llvm_ir::ValueId, ty: &llvm_ir::Type) {
        let name = self
            .value_names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("v{}", id.0));
        match ty {
            llvm_ir::Type::Ptr {
                addr_space: llvm_ir::AddressSpace::Private,
                pointee,
            } if self.alloca_results.contains(&id) => {
                let storage_name = format!("{}_storage", name);
                self.writeln(&format!("{};", c_storage_decl(pointee, &storage_name)));
                self.writeln(&format!(
                    "{};",
                    c_ptr_storage_bind_decl(pointee, &name, &storage_name)
                ));
            }
            _ => {
                self.writeln(&format!("{};", c_var_decl(ty, &name)));
            }
        }
    }

    fn emit_inst(&mut self, inst: &llvm_ir::Inst) -> sile_core::Result<()> {
        if let Some(line) = emit::lower_common_inst_line(
            inst,
            |id| self.value_name(id),
            |op| self.format_operand(op),
        ) {
            if let Some(id) = inst.result {
                let line = self.line_with_declaration(id, &inst.ty, &line);
                self.writeln(&line);
            } else {
                self.writeln(&line);
            }
            return Ok(());
        }

        match &inst.op {
            llvm_ir::InstOp::Alloca { .. } => Ok(()),
            llvm_ir::InstOp::Call { func, args } => {
                if let Some(id) = inst.result {
                    let name = self.value_name(id);
                    let line = format!(
                        "{} = {}({});",
                        name,
                        func,
                        args.iter()
                            .map(|arg| self.format_operand(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    let line = self.line_with_declaration(id, &inst.ty, &line);
                    self.writeln(&line);
                } else {
                    self.writeln(&format!(
                        "{}({});",
                        func,
                        args.iter()
                            .map(|arg| self.format_operand(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                Ok(())
            }
            llvm_ir::InstOp::Intrinsic { intrinsic, args } => {
                let expr = format!(
                    "{}({})",
                    intrinsic_name(intrinsic),
                    args.iter()
                        .map(|arg| self.format_operand(arg))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if let Some(id) = inst.result {
                    let line = format!("{} = {};", self.value_name(id), expr);
                    let line = self.line_with_declaration(id, &inst.ty, &line);
                    self.writeln(&line);
                } else {
                    self.writeln(&format!("{};", expr));
                }
                Ok(())
            }
            llvm_ir::InstOp::Memcpy { dst, src, size } => {
                self.writeln(&format!(
                    "memcpy({}, {}, {});",
                    self.format_operand(dst),
                    self.format_operand(src),
                    self.format_operand(size)
                ));
                Ok(())
            }
            llvm_ir::InstOp::AtomicAdd { ptr, value } => {
                self.writeln("#pragma omp atomic update");
                self.writeln(&format!(
                    "(*{}) += {};",
                    self.format_operand(ptr),
                    self.format_operand(value)
                ));
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> sile_core::Result<()> {
        self.emit_block_insts_except(block, None)
    }

    fn emit_block_insts_except(
        &mut self,
        block: &llvm_ir::BasicBlock,
        skip_result: Option<llvm_ir::ValueId>,
    ) -> sile_core::Result<()> {
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

    fn emit_structured_from(
        &mut self,
        start: llvm_ir::BlockId,
        stop_targets: &[llvm_ir::BlockId],
    ) -> sile_core::Result<Option<llvm_ir::BlockId>> {
        emit::emit_structured_from(
            self,
            self.func,
            start,
            stop_targets,
            None,
            None,
            StructuredCfgMessages {
                preheader_must_branch: "structured C loop preheader must end with a branch",
                missing_loop_header: "missing LLVM IR C loop header",
                header_must_cond_br: "structured C loop header must end with a conditional branch",
                loop_backedge_mismatch: "structured C loop body did not produce the expected backedge",
                unsupported_cond_br: "LLVM IR C codegen only supports structured conditional branches",
                unsupported_switch: "LLVM IR C codegen does not yet support structured switch lowering",
            },
        )
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
            self.writeln(&format!("{name} = {};", self.format_operand(arg)));
        }
    }

    fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
        emit::format_operand(&self.value_names, operand)
    }

    fn value_name(&self, id: llvm_ir::ValueId) -> String {
        llir_value_name(&self.value_names, id)
    }

    fn writeln(&mut self, line: &str) {
        self.out
            .push_str(&format!("{}{}\n", "  ".repeat(self.indent), line));
    }

    fn line_with_declaration(
        &mut self,
        id: llvm_ir::ValueId,
        ty: &llvm_ir::Type,
        line: &str,
    ) -> String {
        if self.declared_values.contains(&id) {
            return line.to_string();
        }
        let name = self.value_name(id);
        let prefix = format!("{name} = ");
        let Some(rhs) = line.strip_prefix(&prefix) else {
            self.declared_values.insert(id);
            return line.to_string();
        };
        self.declared_values.insert(id);
        format!("{} = {}", c_decl_head(ty, &name), rhs)
    }
}

impl TextCodegen for CCodegen<'_> {
    fn emit_prelude(&mut self) {
        CCodegen::emit_prelude(self);
    }

    fn emit_signature(&mut self) {
        CCodegen::emit_signature(self);
    }

    fn emit_body(&mut self) -> sile_core::Result<()> {
        CCodegen::emit_body(self)
    }

    fn finish(self) -> String {
        self.out
    }
}

impl StructuredEmitter for CCodegen<'_> {
    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> sile_core::Result<()> {
        CCodegen::emit_block_insts(self, block)
    }

    fn emit_block_insts_except(
        &mut self,
        block: &llvm_ir::BasicBlock,
        skip_result: Option<llvm_ir::ValueId>,
    ) -> sile_core::Result<()> {
        CCodegen::emit_block_insts_except(self, block, skip_result)
    }

    fn emit_block_param_assignments_skipping(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
        skip_params: &[llvm_ir::ValueId],
    ) {
        CCodegen::emit_block_param_assignments_skipping(self, target, args, skip_params);
    }

    fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
        CCodegen::format_operand(self, operand)
    }

    fn writeln(&mut self, line: &str) {
        CCodegen::writeln(self, line);
    }

    fn indent_inc(&mut self) {
        self.indent += 1;
    }

    fn indent_dec(&mut self) {
        self.indent -= 1;
    }
}

fn emit_tile_count_logic(
    out: &mut String,
    abi: &llvm_ir::ParamAbi,
    tile_rows: usize,
    tile_cols: usize,
) {
    if abi.rank == 1 {
        out.push_str(&format!(
            "  int64_t sile_total_tiles = shapes[{}];\n",
            abi.shape_offset
        ));
    } else {
        out.push_str(&format!(
            "  int64_t sile_tiles_n = shapes[{}] / {};\n",
            abi.shape_offset + 1,
            tile_cols
        ));
        out.push_str(&format!(
            "  int64_t sile_total_tiles = (shapes[{}] / {}) * sile_tiles_n;\n",
            abi.shape_offset,
            tile_rows
        ));
    }
}

fn generate_wrapper(func: &llvm_ir::Function) -> sile_core::Result<String> {
    let mut out = String::new();
    let tile_plan = infer_tile_plan(func);
    let output_rank = tile_plan
        .and_then(|plan| func.params.get(plan.output_param))
        .and_then(|param| param.abi.as_ref().map(|abi| abi.rank))
        .or_else(|| {
            // Fallback: find output buffer param by ABI metadata directly
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

    if let Some(plan) = tile_plan {
        let output_param = &func.params[plan.output_param];
        let abi = output_param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile(
                "LLVM IR CPU wrapper requires output parameter ABI metadata".into(),
            )
        })?;
        emit_tile_count_logic(&mut out, abi, plan.rows, plan.cols);
    } else if let Some(output_param) = func
        .params
        .iter()
        .filter(|param| !is_shapes_param(param))
        .last()
        .filter(|param| param.abi.is_some())
    {
        let abi = output_param.abi.as_ref().unwrap();
        // Use default tile dims when plan is not inferred
        if abi.rank == 1 {
            // For 1D, use the first buffer param's shape dimension
            let first_buf = func
                .params
                .iter()
                .filter(|param| !is_shapes_param(param))
                .next()
                .and_then(|p| p.abi.as_ref())
                .unwrap_or(abi);
            out.push_str(&format!(
                "  int64_t sile_total_tiles = shapes[{}];\n",
                first_buf.shape_offset
            ));
        } else {
            emit_tile_count_logic(&mut out, abi, 32, 16);
        }
    } else {
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

fn c_decl_head(ty: &llvm_ir::Type, name: &str) -> String {
    match ty {
        llvm_ir::Type::I1 => format!("bool {}", name),
        llvm_ir::Type::I32 => format!("int32_t {}", name),
        llvm_ir::Type::I64 => format!("int64_t {}", name),
        llvm_ir::Type::F32 => format!("float {}", name),
        llvm_ir::Type::F64 => format!("double {}", name),
        llvm_ir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llvm_ir::Type::Array { .. } => c_ptr_decl_head(pointee, name),
            other => format!("{}* {}", c_type(other), name),
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

fn c_ptr_decl_head(pointee: &llvm_ir::Type, name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{}* {}", base, name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){}", base, name, suffix)
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
        llvm_ir::Intrinsic::Barrier { .. } => "llir_barrier".to_string(),
        llvm_ir::Intrinsic::Exp => "expf".to_string(),
        llvm_ir::Intrinsic::VecLoad { len } => format!("llir_vec_load_{}", len),
        llvm_ir::Intrinsic::VecStore { len } => format!("llir_vec_store_{}", len),
        llvm_ir::Intrinsic::VecReduceAdd => "llir_vec_reduce_add".to_string(),
    }
}

fn collect_alloca_results(func: &llvm_ir::Function) -> HashSet<llvm_ir::ValueId> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match inst.op {
            llvm_ir::InstOp::Alloca { .. } => inst.result,
            _ => None,
        })
        .collect()
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
