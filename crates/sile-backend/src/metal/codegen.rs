use std::collections::HashMap;

use crate::emit::{
    self, StructuredCfgMessages, StructuredEmitter, TextCodegen, TilePlan, array_dims,
    block_param_assignments, build_value_names, format_operand as format_llir_operand,
    infer_tile_plan, value_name as llir_value_name,
};
use sile_llvm_ir as llvm_ir;

const SHAPES_PARAM_NAME: &str = "__sile_shapes";

pub fn generate(func: &llvm_ir::Function) -> sile_core::Result<String> {
    emit::generate_text(MetalCodegen::new(func))
}

struct MetalCodegen<'a> {
    func: &'a llvm_ir::Function,
    tile_plan: Option<TilePlan>,
    value_names: HashMap<llvm_ir::ValueId, String>,
    indent: usize,
    out: String,
}

impl<'a> MetalCodegen<'a> {
    fn new(func: &'a llvm_ir::Function) -> Self {
        Self {
            func,
            tile_plan: infer_tile_plan(func),
            value_names: build_value_names(func),
            indent: 0,
            out: String::new(),
        }
    }

    fn emit_prelude(&mut self) {
        self.out.push_str("#include <metal_stdlib>\n");
        self.out.push_str("using namespace metal;\n\n");
    }

    fn emit_signature(&mut self) {
        self.out
            .push_str(&format!("kernel void sile_kernel_{}(\n", self.func.name));
        for (idx, param) in self.func.params.iter().enumerate() {
            self.out.push_str(&format!(
                "    {} [[buffer({})]],\n",
                metal_param_decl(param),
                idx,
            ));
        }
        self.out
            .push_str("    uint3 gid [[thread_position_in_grid]]) {\n");
        self.indent = 1;
    }

    fn emit_body(&mut self) -> sile_core::Result<()> {
        self.emit_value_decls();
        self.writeln("");
        self.emit_structured_from(self.func.entry, &[])?;

        self.indent = 0;
        self.out.push_str("}\n");
        Ok(())
    }

    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> sile_core::Result<()> {
        for inst in &block.insts {
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
            StructuredCfgMessages {
                preheader_must_branch: "structured loop preheader must end with a branch",
                missing_loop_header: "missing LLVM IR structured loop header",
                header_must_cond_br: "structured loop header must end with a conditional branch",
                loop_backedge_mismatch: "structured loop body did not produce the expected backedge",
                unsupported_cond_br: "LLVM IR Metal codegen only supports structured conditional branches",
                unsupported_switch: "LLVM IR Metal codegen does not yet support switch terminators",
            },
        )
    }

    fn emit_value_decls(&mut self) {
        let func = self.func;
        let _ = emit::emit_value_decls(func, |id, ty| self.emit_decl(id, ty));
    }

    fn emit_decl(&mut self, id: llvm_ir::ValueId, ty: &llvm_ir::Type) {
        let name = self.value_name(id);
        match ty {
            llvm_ir::Type::Ptr {
                addr_space: llvm_ir::AddressSpace::Private,
                pointee,
            } => {
                let storage_name = format!("{}_storage", name);
                self.writeln(&format!("{};", metal_storage_decl(pointee, &storage_name)));
                self.writeln(&metal_private_ptr_bind_decl(pointee, &name, &storage_name));
            }
            _ => self.writeln(&format!("{};", metal_var_decl(ty, &name))),
        }
    }

    fn emit_inst(&mut self, inst: &llvm_ir::Inst) -> sile_core::Result<()> {
        if let Some(line) = emit::lower_common_inst_line(
            inst,
            |id| self.value_name(id),
            |op| self.format_operand(op),
        ) {
            self.writeln(&line);
            return Ok(());
        }

        match &inst.op {
            llvm_ir::InstOp::Alloca { .. } => Ok(()),
            llvm_ir::InstOp::Memcpy { dst, src, size } => {
                self.writeln(&format!(
                    "for (int64_t copy_i = 0; copy_i < {}; ++copy_i) {{",
                    self.format_operand(size)
                ));
                self.indent += 1;
                self.writeln(&format!(
                    "{}[copy_i] = {}[copy_i];",
                    self.format_operand(dst),
                    self.format_operand(src)
                ));
                self.indent -= 1;
                self.writeln("}");
                Ok(())
            }
            llvm_ir::InstOp::AtomicAdd { ptr, value } => {
                self.writeln(&format!(
                    "atomic_fetch_add_explicit((device atomic_float*)({}), {}, memory_order_relaxed);",
                    self.format_operand(ptr),
                    self.format_operand(value)
                ));
                Ok(())
            }
            llvm_ir::InstOp::Call { func, .. } => Err(sile_core::Error::Compile(format!(
                "LLVM IR Metal codegen does not support call instructions: {func}"
            ))),
            llvm_ir::InstOp::Intrinsic { intrinsic, args } => {
                self.emit_intrinsic(inst.result, intrinsic, args)
            }
            _ => Ok(()),
        }
    }

    fn emit_intrinsic(
        &mut self,
        result: Option<llvm_ir::ValueId>,
        intrinsic: &llvm_ir::Intrinsic,
        args: &[llvm_ir::Operand],
    ) -> sile_core::Result<()> {
        match intrinsic {
            llvm_ir::Intrinsic::BlockId { dim } => {
                let Some(id) = result else {
                    return Err(sile_core::Error::Compile(
                        "block_id intrinsic must produce a result".into(),
                    ));
                };
                let axis = self.logical_block_id_expr(*dim)?;
                self.writeln(&format!("{} = {};", self.value_name(id), axis));
                Ok(())
            }
            llvm_ir::Intrinsic::ThreadId { .. } => Err(sile_core::Error::Compile(
                "thread_id intrinsic is not yet supported in LLVM IR Metal codegen".into(),
            )),
            llvm_ir::Intrinsic::Barrier { .. } => Err(sile_core::Error::Compile(
                "barrier intrinsic is not yet supported in LLVM IR Metal codegen".into(),
            )),
            llvm_ir::Intrinsic::Exp => {
                let Some(id) = result else {
                    return Err(sile_core::Error::Compile(
                        "exp intrinsic must produce a result".into(),
                    ));
                };
                let [arg] = args else {
                    return Err(sile_core::Error::Compile(
                        "exp intrinsic expects one argument".into(),
                    ));
                };
                self.writeln(&format!(
                    "{} = metal::exp({});",
                    self.value_name(id),
                    self.format_operand(arg)
                ));
                Ok(())
            }
        }
    }

    fn emit_block_param_assignments(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
    ) {
        for (name, arg) in block_param_assignments(self.func, &self.value_names, target, args) {
            self.writeln(&format!("{name} = {arg};"));
        }
    }

    fn logical_block_id_expr(&self, dim: u8) -> sile_core::Result<String> {
        let Some(plan) = self.tile_plan else {
            return Ok(match dim {
                0 => "(int64_t)gid.x".into(),
                1 => "(int64_t)gid.y".into(),
                2 => "(int64_t)gid.z".into(),
                _ => {
                    return Err(sile_core::Error::Compile(format!(
                        "unsupported block_id dimension in Metal codegen: {dim}"
                    )));
                }
            });
        };
        let output_param = self.func.params.get(plan.output_param).ok_or_else(|| {
            sile_core::Error::Compile("missing output parameter for Metal tile plan".into())
        })?;
        let output_rank = output_param.abi.as_ref().map(|abi| abi.rank).unwrap_or(1);
        if output_rank <= 1 {
            return Ok(match dim {
                0 => "(int64_t)gid.x".into(),
                1 | 2 => "0".into(),
                _ => {
                    return Err(sile_core::Error::Compile(format!(
                        "unsupported block_id dimension in Metal codegen: {dim}"
                    )));
                }
            });
        }
        let abi = output_param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile("missing output ABI for Metal tile plan".into())
        })?;
        let shapes = self.shapes_param_name()?;
        let tiles_n = format!("({shapes}[{}] / {})", abi.shape_offset + 1, plan.cols);
        Ok(match dim {
            0 => format!("((int64_t)gid.x / {tiles_n})"),
            1 => format!("((int64_t)gid.x % {tiles_n})"),
            2 => "0".into(),
            _ => {
                return Err(sile_core::Error::Compile(format!(
                    "unsupported block_id dimension in Metal codegen: {dim}"
                )));
            }
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

    fn value_name(&self, id: llvm_ir::ValueId) -> String {
        llir_value_name(&self.value_names, id)
    }

    fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
        format_llir_operand(&self.value_names, operand)
    }

    fn writeln(&mut self, line: &str) {
        self.out
            .push_str(&format!("{}{}\n", "  ".repeat(self.indent), line));
    }
}

impl TextCodegen for MetalCodegen<'_> {
    fn emit_prelude(&mut self) {
        MetalCodegen::emit_prelude(self);
    }

    fn emit_signature(&mut self) {
        MetalCodegen::emit_signature(self);
    }

    fn emit_body(&mut self) -> sile_core::Result<()> {
        MetalCodegen::emit_body(self)
    }

    fn finish(self) -> String {
        self.out
    }
}

impl StructuredEmitter for MetalCodegen<'_> {
    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> sile_core::Result<()> {
        MetalCodegen::emit_block_insts(self, block)
    }

    fn emit_block_param_assignments(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
    ) {
        MetalCodegen::emit_block_param_assignments(self, target, args);
    }

    fn format_operand(&self, operand: &llvm_ir::Operand) -> String {
        MetalCodegen::format_operand(self, operand)
    }

    fn writeln(&mut self, line: &str) {
        MetalCodegen::writeln(self, line);
    }

    fn indent_inc(&mut self) {
        self.indent += 1;
    }

    fn indent_dec(&mut self) {
        self.indent -= 1;
    }
}

fn metal_param_decl(param: &llvm_ir::Param) -> String {
    let ty = match &param.ty {
        llvm_ir::Type::Ptr {
            addr_space: llvm_ir::AddressSpace::Global,
            pointee,
        } => format!("device {}* {}", metal_scalar_type(pointee), param.name),
        llvm_ir::Type::Ptr {
            addr_space: llvm_ir::AddressSpace::Constant,
            pointee,
        } => format!("constant {}* {}", metal_scalar_type(pointee), param.name),
        other => format!("{} {}", metal_type(other), param.name),
    };
    ty
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
