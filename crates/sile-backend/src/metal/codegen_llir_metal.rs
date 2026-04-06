use std::collections::{HashMap, HashSet};

use crate::passes::emit::shared::{
    array_dims, bin_op_symbol, block_param_assignments, build_param_indices, build_value_names,
    cmp_pred_symbol, format_operand as format_llir_operand, value_name as llir_value_name,
};
use sile_llir as llir;

pub fn generate(func: &llir::Function) -> sile_core::Result<String> {
    let mut ctx = MetalCodegen {
        func,
        tile_plan: infer_tile_plan(func),
        value_names: build_value_names(func),
        param_indices: build_param_indices(func),
        indent: 0,
        out: String::new(),
    };

    ctx.emit_prelude();
    ctx.emit_signature();
    ctx.emit_body()?;

    Ok(ctx.out)
}

struct MetalCodegen<'a> {
    func: &'a llir::Function,
    tile_plan: Option<TilePlan>,
    value_names: HashMap<llir::ValueId, String>,
    param_indices: HashMap<llir::ValueId, usize>,
    indent: usize,
    out: String,
}

#[derive(Clone, Copy)]
struct TilePlan {
    output_param: usize,
    cols: usize,
}

impl<'a> MetalCodegen<'a> {
    fn emit_prelude(&mut self) {
        self.out.push_str("#include <metal_stdlib>\n");
        self.out.push_str("using namespace metal;\n\n");
    }

    fn emit_signature(&mut self) {
        self.out
            .push_str(&format!("kernel void sile_kernel_{}(\n", self.func.name));
        for (idx, param) in self.func.params.iter().enumerate() {
            let comma = if idx + 1 == self.func.params.len() {
                ",\n"
            } else {
                ",\n"
            };
            self.out.push_str(&format!(
                "    {} [[buffer({})]]{}",
                metal_param_decl(param),
                idx,
                comma
            ));
        }
        self.out.push_str(&format!(
            "    const device int64_t* shapes [[buffer({})]],\n",
            self.func.params.len()
        ));
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

    fn emit_block_insts(&mut self, block: &llir::BasicBlock) -> sile_core::Result<()> {
        for inst in &block.insts {
            self.emit_inst(inst)?;
        }
        Ok(())
    }

    fn emit_structured_from(
        &mut self,
        start: llir::BlockId,
        stop_targets: &[llir::BlockId],
    ) -> sile_core::Result<Option<llir::BlockId>> {
        let mut current = start;
        loop {
            let block = self.get_block(current).cloned().ok_or_else(|| {
                sile_core::Error::Compile(format!("missing LLIR block {:?}", current))
            })?;

            if let llir::Terminator::Br { target, args } = &block.terminator {
                if stop_targets.contains(target) {
                    self.emit_block_insts(&block)?;
                    self.emit_block_param_assignments(*target, args);
                    return Ok(Some(*target));
                }
            }

            if self.is_loop_preheader(&block) {
                current = self.emit_structured_loop_preheader(&block)?;
                if stop_targets.contains(&current) {
                    return Ok(Some(current));
                }
                continue;
            }

            if matches!(block.terminator, llir::Terminator::CondBr { .. }) {
                current = self.emit_structured_loop_header(&block)?;
                if stop_targets.contains(&current) {
                    return Ok(Some(current));
                }
                continue;
            }

            self.emit_block_insts(&block)?;
            match &block.terminator {
                llir::Terminator::Br { target, args } => {
                    self.emit_block_param_assignments(*target, args);
                    if stop_targets.contains(target) {
                        return Ok(Some(*target));
                    }
                    current = *target;
                }
                llir::Terminator::Ret { value: None } => {
                    self.writeln("return;");
                    return Ok(None);
                }
                llir::Terminator::Ret { value: Some(value) } => {
                    self.writeln(&format!("return {};", self.format_operand(value)));
                    return Ok(None);
                }
                llir::Terminator::CondBr { .. } => {
                    return Err(sile_core::Error::Compile(
                        "LLIR Metal codegen only supports structured conditional branches".into(),
                    ));
                }
                llir::Terminator::Switch { .. } => {
                    return Err(sile_core::Error::Compile(
                        "LLIR Metal codegen does not yet support switch terminators".into(),
                    ));
                }
            }
        }
    }

    fn emit_structured_loop_preheader(
        &mut self,
        preheader: &llir::BasicBlock,
    ) -> sile_core::Result<llir::BlockId> {
        let llir::Terminator::Br {
            target: header_id,
            args: header_args,
        } = &preheader.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured loop preheader must end with a branch".into(),
            ));
        };
        let header = self.get_block(*header_id).cloned().ok_or_else(|| {
            sile_core::Error::Compile("missing LLIR structured loop header".into())
        })?;
        let llir::Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } = &header.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured loop header must end with a conditional branch".into(),
            ));
        };

        self.emit_block_insts(preheader)?;
        self.emit_block_param_assignments(*header_id, header_args);
        self.emit_structured_loop_header_impl(
            &header,
            cond,
            *true_target,
            true_args,
            *false_target,
            false_args,
        )
    }

    fn emit_structured_loop_header(
        &mut self,
        header: &llir::BasicBlock,
    ) -> sile_core::Result<llir::BlockId> {
        let llir::Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } = &header.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured loop header must end with a conditional branch".into(),
            ));
        };
        self.emit_structured_loop_header_impl(
            header,
            cond,
            *true_target,
            true_args,
            *false_target,
            false_args,
        )
    }

    fn emit_structured_loop_header_impl(
        &mut self,
        header: &llir::BasicBlock,
        cond: &llir::Operand,
        true_target: llir::BlockId,
        true_args: &[llir::Operand],
        false_target: llir::BlockId,
        false_args: &[llir::Operand],
    ) -> sile_core::Result<llir::BlockId> {
        self.writeln("while (true) {");
        self.indent += 1;
        self.emit_block_insts(header)?;
        self.writeln(&format!("if (!({})) {{", self.format_operand(cond)));
        self.indent += 1;
        self.emit_block_param_assignments(false_target, false_args);
        self.writeln("break;");
        self.indent -= 1;
        self.writeln("}");

        self.emit_block_param_assignments(true_target, true_args);
        let backedge = self.emit_structured_from(true_target, &[header.id])?;
        if backedge != Some(header.id) {
            return Err(sile_core::Error::Compile(
                "structured loop body did not produce the expected backedge".into(),
            ));
        }
        self.indent -= 1;
        self.writeln("}");

        Ok(false_target)
    }

    fn is_loop_preheader(&self, block: &llir::BasicBlock) -> bool {
        let llir::Terminator::Br { target: header, .. } = block.terminator else {
            return false;
        };
        matches!(
            self.get_block(header).map(|block| &block.terminator),
            Some(llir::Terminator::CondBr { .. })
        )
    }

    fn emit_value_decls(&mut self) {
        let mut declared = HashSet::new();

        for block in &self.func.blocks {
            for param in &block.params {
                if declared.insert(param.id) {
                    self.emit_decl(param.id, &param.ty);
                }
            }
            for inst in &block.insts {
                if let Some(id) = inst.result {
                    if declared.insert(id) {
                        self.emit_decl(id, &inst.ty);
                    }
                }
            }
        }
    }

    fn emit_decl(&mut self, id: llir::ValueId, ty: &llir::Type) {
        let name = self.value_name(id);
        match ty {
            llir::Type::Ptr {
                addr_space: llir::AddressSpace::Private,
                pointee,
            } => {
                let storage_name = format!("{}_storage", name);
                self.writeln(&format!("{};", metal_storage_decl(pointee, &storage_name)));
                self.writeln(&metal_private_ptr_bind_decl(pointee, &name, &storage_name));
            }
            _ => self.writeln(&format!("{};", metal_var_decl(ty, &name))),
        }
    }

    fn emit_inst(&mut self, inst: &llir::Inst) -> sile_core::Result<()> {
        match &inst.op {
            llir::InstOp::ShapeDim { buf, dim } => {
                if let Some(id) = inst.result {
                    let expr = self.emit_shape_dim(buf, *dim)?;
                    self.writeln(&format!("{} = {};", self.value_name(id), expr));
                }
                Ok(())
            }
            llir::InstOp::Alloca { .. } => Ok(()),
            llir::InstOp::Bin { op, lhs, rhs } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {} {} {};",
                        self.value_name(id),
                        self.format_operand(lhs),
                        bin_op_symbol(*op),
                        self.format_operand(rhs)
                    ));
                }
                Ok(())
            }
            llir::InstOp::Cmp { pred, lhs, rhs } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {} {} {};",
                        self.value_name(id),
                        self.format_operand(lhs),
                        cmp_pred_symbol(*pred),
                        self.format_operand(rhs)
                    ));
                }
                Ok(())
            }
            llir::InstOp::Select {
                cond,
                on_true,
                on_false,
            } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = ({}) ? ({}) : ({});",
                        self.value_name(id),
                        self.format_operand(cond),
                        self.format_operand(on_true),
                        self.format_operand(on_false)
                    ));
                }
                Ok(())
            }
            llir::InstOp::Load { ptr } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = *({});",
                        self.value_name(id),
                        self.format_operand(ptr)
                    ));
                }
                Ok(())
            }
            llir::InstOp::Store { ptr, value } => {
                self.writeln(&format!(
                    "*({}) = {};",
                    self.format_operand(ptr),
                    self.format_operand(value)
                ));
                Ok(())
            }
            llir::InstOp::Cast { value, .. } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {};",
                        self.value_name(id),
                        self.format_operand(value)
                    ));
                }
                Ok(())
            }
            llir::InstOp::Gep { base, indices } => {
                if let Some(id) = inst.result {
                    let index_suffix = indices
                        .iter()
                        .map(|idx| format!("[{}]", self.format_operand(idx)))
                        .collect::<Vec<_>>()
                        .join("");
                    self.writeln(&format!(
                        "{} = &({}{});",
                        self.value_name(id),
                        self.format_operand(base),
                        index_suffix
                    ));
                }
                Ok(())
            }
            llir::InstOp::Memcpy { dst, src, size } => {
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
            llir::InstOp::Call { func, args } => self.emit_call(inst.result, func, args),
            llir::InstOp::Intrinsic { intrinsic, args } => {
                self.emit_intrinsic(inst.result, intrinsic, args)
            }
        }
    }

    fn emit_call(
        &mut self,
        _result: Option<llir::ValueId>,
        func: &str,
        args: &[llir::Operand],
    ) -> sile_core::Result<()> {
        match func {
            "tile_load_2d_f32" => self.emit_tile_load(args),
            "tile_store_2d_f32" => self.emit_tile_store(args),
            other => Err(sile_core::Error::Compile(format!(
                "unsupported LLIR helper call for Metal codegen: {other}"
            ))),
        }
    }

    fn emit_intrinsic(
        &mut self,
        result: Option<llir::ValueId>,
        intrinsic: &llir::Intrinsic,
        args: &[llir::Operand],
    ) -> sile_core::Result<()> {
        match intrinsic {
            llir::Intrinsic::BlockId { dim } => {
                let Some(id) = result else {
                    return Err(sile_core::Error::Compile(
                        "block_id intrinsic must produce a result".into(),
                    ));
                };
                let axis = self.logical_block_id_expr(*dim)?;
                self.writeln(&format!("{} = {};", self.value_name(id), axis));
                Ok(())
            }
            llir::Intrinsic::ThreadId { .. } => Err(sile_core::Error::Compile(
                "thread_id intrinsic is not yet supported in LLIR Metal codegen".into(),
            )),
            llir::Intrinsic::Barrier { .. } => Err(sile_core::Error::Compile(
                "barrier intrinsic is not yet supported in LLIR Metal codegen".into(),
            )),
            llir::Intrinsic::Exp => {
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

    fn emit_shape_dim(&self, buf: &llir::Operand, dim: usize) -> sile_core::Result<String> {
        let param_idx = self.param_index(buf).ok_or_else(|| {
            sile_core::Error::Compile(
                "shape_dim currently requires a direct kernel parameter operand".into(),
            )
        })?;
        let rank = self
            .func
            .params
            .get(param_idx)
            .and_then(|param| param.abi.as_ref().map(|abi| abi.rank))
            .unwrap_or(1);
        if dim >= rank {
            return Err(sile_core::Error::Compile(format!(
                "shape_dim requested dim {dim} for parameter rank {rank}"
            )));
        }
        let offset = self
            .func
            .params
            .get(param_idx)
            .and_then(|param| param.abi.as_ref().map(|abi| abi.shape_offset + dim))
            .ok_or_else(|| {
                sile_core::Error::Compile(
                    "shape_dim requires LLIR parameter ABI metadata in Metal codegen".into(),
                )
            })?;
        Ok(format!("shapes[{offset}]"))
    }

    fn emit_tile_load(&mut self, args: &[llir::Operand]) -> sile_core::Result<()> {
        let [dst, buf, row_tile, col_tile, rows, cols, stride_idx] = args else {
            return Err(sile_core::Error::Compile(
                "tile_load_2d_f32 expects seven arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile_load rows")?;
        let cols = const_usize(cols, "tile_load cols")?;
        let stride_idx = const_usize(stride_idx, "tile_load stride dim")?;
        let dst_name = self.format_operand(dst);
        let buf_name = self.format_operand(buf);
        let row_tile = self.format_operand(row_tile);
        let col_tile = self.format_operand(col_tile);
        let rank = self.buffer_rank(buf, stride_idx);

        self.emit_nested_tile_loop("load", rows, cols, |ctx, r, c| {
            if rank <= 1 {
                ctx.writeln(&format!(
                    "{dst_name}[{r}][{c}] = {buf_name}[({col_tile} * {cols}) + {c}];"
                ));
            } else {
                let stride = ctx.shape_dim_expr(buf, stride_idx);
                ctx.writeln(&format!(
                    "{dst_name}[{r}][{c}] = {buf_name}[(({row_tile} * {rows}) + {r}) * {stride} + (({col_tile} * {cols}) + {c})];"
                ));
            }
        });
        Ok(())
    }

    fn emit_tile_store(&mut self, args: &[llir::Operand]) -> sile_core::Result<()> {
        let [buf, value, row_tile, col_tile, rows, cols, stride_idx] = args else {
            return Err(sile_core::Error::Compile(
                "tile_store_2d_f32 expects seven arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile_store rows")?;
        let cols = const_usize(cols, "tile_store cols")?;
        let stride_idx = const_usize(stride_idx, "tile_store stride dim")?;
        let buf_name = self.format_operand(buf);
        let value_name = self.format_operand(value);
        let row_tile = self.format_operand(row_tile);
        let col_tile = self.format_operand(col_tile);
        let rank = self.buffer_rank(buf, stride_idx);

        self.emit_nested_tile_loop("store", rows, cols, |ctx, r, c| {
            if rank <= 1 {
                ctx.writeln(&format!(
                    "{buf_name}[({col_tile} * {cols}) + {c}] = {value_name}[{r}][{c}];"
                ));
            } else {
                let stride = ctx.shape_dim_expr(buf, stride_idx);
                ctx.writeln(&format!(
                    "{buf_name}[(({row_tile} * {rows}) + {r}) * {stride} + (({col_tile} * {cols}) + {c})] = {value_name}[{r}][{c}];"
                ));
            }
        });
        Ok(())
    }

    fn emit_nested_tile_loop<F>(&mut self, prefix: &str, rows: usize, cols: usize, mut body: F)
    where
        F: FnMut(&mut Self, &str, &str),
    {
        let row_var = format!("{prefix}_r");
        let col_var = format!("{prefix}_c");
        self.writeln(&format!(
            "for (int {row_var} = 0; {row_var} < {rows}; ++{row_var}) {{"
        ));
        self.indent += 1;
        self.writeln(&format!(
            "for (int {col_var} = 0; {col_var} < {cols}; ++{col_var}) {{"
        ));
        self.indent += 1;
        body(self, &row_var, &col_var);
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
    }

    fn emit_block_param_assignments(&mut self, target: llir::BlockId, args: &[llir::Operand]) {
        for (name, arg) in block_param_assignments(self.func, &self.value_names, target, args) {
            self.writeln(&format!("{name} = {arg};"));
        }
    }

    fn buffer_rank(&self, operand: &llir::Operand, stride_idx: usize) -> usize {
        self.param_index(operand)
            .and_then(|idx| self.func.params.get(idx))
            .and_then(|param| param.abi.as_ref().map(|abi| abi.rank))
            .unwrap_or_else(|| if stride_idx >= 1 { 2 } else { 1 })
    }

    fn shape_dim_expr(&self, operand: &llir::Operand, dim: usize) -> String {
        self.param_index(operand)
            .and_then(|param_idx| self.func.params.get(param_idx))
            .and_then(|param| {
                param
                    .abi
                    .as_ref()
                    .map(|abi| format!("shapes[{}]", abi.shape_offset + dim))
            })
            .unwrap_or_else(|| "1".into())
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
        let tiles_n = format!("(shapes[{}] / {})", abi.shape_offset + 1, plan.cols);
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

    fn param_index(&self, operand: &llir::Operand) -> Option<usize> {
        match operand {
            llir::Operand::Value(id) => self.param_indices.get(id).copied(),
            _ => None,
        }
    }

    fn value_name(&self, id: llir::ValueId) -> String {
        llir_value_name(&self.value_names, id)
    }

    fn format_operand(&self, operand: &llir::Operand) -> String {
        format_llir_operand(&self.value_names, operand)
    }

    fn get_block(&self, id: llir::BlockId) -> Option<&llir::BasicBlock> {
        self.func.blocks.iter().find(|block| block.id == id)
    }

    fn writeln(&mut self, line: &str) {
        self.out
            .push_str(&format!("{}{}\n", "  ".repeat(self.indent), line));
    }
}

fn metal_param_decl(param: &llir::Param) -> String {
    let ty = match &param.ty {
        llir::Type::Ptr {
            addr_space: llir::AddressSpace::Global,
            pointee,
        } => format!("device {}* {}", metal_scalar_type(pointee), param.name),
        llir::Type::Ptr {
            addr_space: llir::AddressSpace::Constant,
            pointee,
        } => format!("constant {}* {}", metal_scalar_type(pointee), param.name),
        other => format!("{} {}", metal_type(other), param.name),
    };
    ty
}

fn metal_var_decl(ty: &llir::Type, name: &str) -> String {
    match ty {
        llir::Type::I1 => format!("bool {} = false", name),
        llir::Type::I32 => format!("int32_t {} = 0", name),
        llir::Type::I64 => format!("int64_t {} = 0", name),
        llir::Type::F32 => format!("float {} = 0.0f", name),
        llir::Type::F64 => format!("double {} = 0.0", name),
        llir::Type::Ptr {
            addr_space,
            pointee,
        } => metal_ptr_zero_decl(addr_space, pointee, name),
        other => format!("{} {}", metal_type(other), name),
    }
}

fn metal_storage_decl(ty: &llir::Type, name: &str) -> String {
    match ty {
        llir::Type::Array { .. } => {
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

fn metal_private_ptr_bind_decl(pointee: &llir::Type, name: &str, storage_name: &str) -> String {
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
    addr_space: &llir::AddressSpace,
    pointee: &llir::Type,
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

fn metal_addr_qual(addr_space: &llir::AddressSpace) -> &'static str {
    match addr_space {
        llir::AddressSpace::Generic => "",
        llir::AddressSpace::Private => "thread ",
        llir::AddressSpace::Global => "device ",
        llir::AddressSpace::Constant => "constant ",
        llir::AddressSpace::Shared => "threadgroup ",
    }
}

fn metal_type(ty: &llir::Type) -> String {
    match ty {
        llir::Type::Void => "void".into(),
        llir::Type::I1 => "bool".into(),
        llir::Type::I32 => "int32_t".into(),
        llir::Type::I64 => "int64_t".into(),
        llir::Type::F16 => "half".into(),
        llir::Type::F32 => "float".into(),
        llir::Type::F64 => "double".into(),
        llir::Type::Ptr {
            addr_space,
            pointee,
        } => format!(
            "{}{}*",
            metal_addr_qual(addr_space),
            metal_scalar_type(pointee)
        ),
        llir::Type::Vector { len, elem } => format!("{}{}", metal_scalar_type(elem), len),
        llir::Type::Array { len, elem } => format!("{}[{}]", metal_type(elem), len),
    }
}

fn metal_scalar_type(ty: &llir::Type) -> String {
    match ty {
        llir::Type::Array { .. } => array_base_type(ty),
        other => metal_type(other),
    }
}

fn array_base_type(ty: &llir::Type) -> String {
    let mut current = ty;
    while let llir::Type::Array { elem, .. } = current {
        current = elem;
    }
    metal_type(current)
}

fn const_usize(operand: &llir::Operand, what: &str) -> sile_core::Result<usize> {
    match operand {
        llir::Operand::Const(llir::Constant::Int(value)) if *value >= 0 => Ok(*value as usize),
        _ => Err(sile_core::Error::Compile(format!(
            "{what} must be a non-negative integer constant"
        ))),
    }
}

fn infer_tile_plan(func: &llir::Function) -> Option<TilePlan> {
    let inst_by_result = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect::<HashMap<_, _>>();
    let value_types = build_value_type_map(func);

    for block in &func.blocks {
        for inst in &block.insts {
            let llir::InstOp::Store {
                ptr: llir::Operand::Value(ptr_id),
                value: llir::Operand::Value(value_id),
            } = &inst.op
            else {
                continue;
            };
            let Some(ptr_inst) = inst_by_result.get(ptr_id) else {
                continue;
            };
            let llir::InstOp::Gep {
                base: llir::Operand::Value(buf_id),
                ..
            } = &ptr_inst.op
            else {
                continue;
            };
            let Some(output_param) = func.params.iter().position(|param| param.id == *buf_id)
            else {
                continue;
            };
            let Some((_rows, cols)) =
                infer_tile_dims_from_scalar_store(*value_id, &inst_by_result, &value_types)
            else {
                continue;
            };
            return Some(TilePlan { output_param, cols });
        }
    }

    None
}

fn build_value_type_map(func: &llir::Function) -> HashMap<llir::ValueId, llir::Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(id) = inst.result {
                types.insert(id, inst.ty.clone());
            }
        }
    }
    types
}

fn infer_tile_dims_from_scalar_store(
    value_id: llir::ValueId,
    inst_by_result: &HashMap<llir::ValueId, &llir::Inst>,
    value_types: &HashMap<llir::ValueId, llir::Type>,
) -> Option<(usize, usize)> {
    let load_inst = inst_by_result.get(&value_id)?;
    let llir::InstOp::Load {
        ptr: llir::Operand::Value(tile_ptr_id),
    } = &load_inst.op
    else {
        return None;
    };
    let tile_gep = inst_by_result.get(tile_ptr_id)?;
    let llir::InstOp::Gep {
        base: llir::Operand::Value(tile_id),
        ..
    } = &tile_gep.op
    else {
        return None;
    };
    let tile_ty = value_types.get(tile_id)?;
    tile_shape_from_type(tile_ty)
}

fn tile_shape_from_type(ty: &llir::Type) -> Option<(usize, usize)> {
    let llir::Type::Ptr {
        addr_space: llir::AddressSpace::Private,
        pointee,
    } = ty
    else {
        return None;
    };
    let dims = array_dims(pointee);
    match dims.as_slice() {
        [cols] => Some((1, *cols)),
        [rows, cols, ..] => Some((*rows, *cols)),
        _ => None,
    }
}
