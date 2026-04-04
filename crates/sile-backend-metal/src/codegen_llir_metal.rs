use std::collections::{HashMap, HashSet};

use sile_llir as llir;

pub fn generate(func: &llir::Function) -> sile_core::Result<String> {
    let mut ctx = MetalCodegen {
        func,
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
    value_names: HashMap<llir::ValueId, String>,
    param_indices: HashMap<llir::ValueId, usize>,
    indent: usize,
    out: String,
}

struct StructuredLoop {
    entry: llir::BlockId,
    header: llir::BlockId,
    body: llir::BlockId,
    exit: llir::BlockId,
}

impl StructuredLoop {
    fn analyze(func: &llir::Function) -> Option<Self> {
        let entry = func.blocks.first()?;
        let llir::Terminator::Br { target: header, .. } = &entry.terminator else {
            return None;
        };
        let header_block = func.blocks.iter().find(|block| block.id == *header)?;
        let llir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        } = &header_block.terminator
        else {
            return None;
        };
        let body_block = func.blocks.iter().find(|block| block.id == *true_target)?;
        let llir::Terminator::Br {
            target: back_target,
            ..
        } = &body_block.terminator
        else {
            return None;
        };
        if *back_target != *header {
            return None;
        }
        let exit_block = func.blocks.iter().find(|block| block.id == *false_target)?;
        if !matches!(exit_block.terminator, llir::Terminator::Ret { .. }) {
            return None;
        }
        Some(Self {
            entry: entry.id,
            header: *header,
            body: *true_target,
            exit: *false_target,
        })
    }
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

        if let Some(loop_cfg) = StructuredLoop::analyze(self.func) {
            self.emit_structured_loop(&loop_cfg)?;
        } else if self.func.blocks.len() == 1 {
            let block = self.func.blocks[0].clone();
            self.emit_linear_block(&block)?;
        } else {
            return Err(sile_core::Error::Compile(
                "LLIR Metal codegen currently supports single-block kernels or a single structured loop"
                    .into(),
            ));
        }

        self.indent = 0;
        self.out.push_str("}\n");
        Ok(())
    }

    fn emit_linear_block(&mut self, block: &llir::BasicBlock) -> sile_core::Result<()> {
        for inst in &block.insts {
            self.emit_inst(inst)?;
        }
        match &block.terminator {
            llir::Terminator::Ret { value: None } => self.writeln("return;"),
            llir::Terminator::Ret { value: Some(value) } => {
                self.writeln(&format!("return {};", self.format_operand(value)))
            }
            _ => {
                return Err(sile_core::Error::Compile(
                    "LLIR Metal linear codegen only supports return terminators".into(),
                ));
            }
        }
        Ok(())
    }

    fn emit_structured_loop(&mut self, loop_cfg: &StructuredLoop) -> sile_core::Result<()> {
        let entry = self
            .get_block(loop_cfg.entry)
            .cloned()
            .ok_or_else(|| sile_core::Error::Compile("missing LLIR loop entry block".into()))?;
        let header = self
            .get_block(loop_cfg.header)
            .cloned()
            .ok_or_else(|| sile_core::Error::Compile("missing LLIR loop header block".into()))?;
        let body = self
            .get_block(loop_cfg.body)
            .cloned()
            .ok_or_else(|| sile_core::Error::Compile("missing LLIR loop body block".into()))?;
        let exit = self
            .get_block(loop_cfg.exit)
            .cloned()
            .ok_or_else(|| sile_core::Error::Compile("missing LLIR loop exit block".into()))?;

        for inst in &entry.insts {
            self.emit_inst(inst)?;
        }
        let llir::Terminator::Br {
            target: entry_target,
            args: entry_args,
        } = &entry.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured LLIR entry must end with a branch".into(),
            ));
        };
        self.emit_block_param_assignments(*entry_target, entry_args);

        self.writeln("while (true) {");
        self.indent += 1;
        for inst in &header.insts {
            self.emit_inst(inst)?;
        }
        let llir::Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } = &header.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured LLIR header must end with a conditional branch".into(),
            ));
        };

        self.writeln(&format!("if (!({})) {{", self.format_operand(cond)));
        self.indent += 1;
        self.emit_block_param_assignments(*false_target, false_args);
        self.writeln("break;");
        self.indent -= 1;
        self.writeln("}");

        self.emit_block_param_assignments(*true_target, true_args);
        for inst in &body.insts {
            self.emit_inst(inst)?;
        }
        let llir::Terminator::Br {
            target: body_target,
            args: body_args,
        } = &body.terminator
        else {
            return Err(sile_core::Error::Compile(
                "structured LLIR body must end with a branch".into(),
            ));
        };
        self.emit_block_param_assignments(*body_target, body_args);
        self.indent -= 1;
        self.writeln("}");

        for inst in &exit.insts {
            self.emit_inst(inst)?;
        }
        match &exit.terminator {
            llir::Terminator::Ret { value: None } => self.writeln("return;"),
            llir::Terminator::Ret { value: Some(value) } => {
                self.writeln(&format!("return {};", self.format_operand(value)))
            }
            _ => {
                return Err(sile_core::Error::Compile(
                    "structured LLIR exit must end with return".into(),
                ));
            }
        }

        Ok(())
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
                        metal_bin_op(*op),
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
                        metal_cmp_pred(*pred),
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
            "tile_splat_f32" => self.emit_tile_splat(args),
            "tile_load_2d_f32" => self.emit_tile_load(args),
            "tile_store_2d_f32" => self.emit_tile_store(args),
            "tile_add_f32" => self.emit_tile_binary(args, "+"),
            "tile_sub_f32" => self.emit_tile_binary(args, "-"),
            "tile_mul_f32" => self.emit_tile_binary(args, "*"),
            "tile_div_f32" => self.emit_tile_binary(args, "/"),
            "tile_exp_f32" => self.emit_tile_unary(args, "metal::exp"),
            "tile_neg_f32" => self.emit_tile_unary(args, "-"),
            "tile_broadcast_f32" => self.emit_tile_broadcast(args),
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
                let axis = match dim {
                    0 => "gid.x",
                    1 => "gid.y",
                    2 => "gid.z",
                    _ => {
                        return Err(sile_core::Error::Compile(format!(
                            "unsupported block_id dimension in Metal codegen: {dim}"
                        )));
                    }
                };
                self.writeln(&format!("{} = (int64_t){};", self.value_name(id), axis));
                Ok(())
            }
            llir::Intrinsic::ThreadId { .. } => Err(sile_core::Error::Compile(
                "thread_id intrinsic is not yet supported in LLIR Metal codegen".into(),
            )),
            llir::Intrinsic::Barrier { .. } => Err(sile_core::Error::Compile(
                "barrier intrinsic is not yet supported in LLIR Metal codegen".into(),
            )),
            llir::Intrinsic::MatmulFragment => self.emit_matmul_fragment(args),
            llir::Intrinsic::ReduceAdd => self.emit_reduce(args, false),
            llir::Intrinsic::ReduceMax => self.emit_reduce(args, true),
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

    fn emit_tile_splat(&mut self, args: &[llir::Operand]) -> sile_core::Result<()> {
        let [dst, value, rows, cols] = args else {
            return Err(sile_core::Error::Compile(
                "tile_splat_f32 expects four arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile_splat rows")?;
        let cols = const_usize(cols, "tile_splat cols")?;
        let dst = self.format_operand(dst);
        let value = self.format_operand(value);
        self.emit_nested_tile_loop("splat", rows, cols, |ctx, r, c| {
            ctx.writeln(&format!("{dst}[{r}][{c}] = {value};"));
        });
        Ok(())
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

    fn emit_tile_binary(&mut self, args: &[llir::Operand], op: &str) -> sile_core::Result<()> {
        let [dst, lhs, rhs, rows, cols] = args else {
            return Err(sile_core::Error::Compile(
                "tile binary helper expects five arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile binary rows")?;
        let cols = const_usize(cols, "tile binary cols")?;
        let dst = self.format_operand(dst);
        let lhs = self.format_operand(lhs);
        let rhs = self.format_operand(rhs);
        self.emit_nested_tile_loop("tile_bin", rows, cols, |ctx, r, c| {
            ctx.writeln(&format!(
                "{dst}[{r}][{c}] = {lhs}[{r}][{c}] {op} {rhs}[{r}][{c}];"
            ));
        });
        Ok(())
    }

    fn emit_tile_unary(&mut self, args: &[llir::Operand], op: &str) -> sile_core::Result<()> {
        let [dst, src, rows, cols] = args else {
            return Err(sile_core::Error::Compile(
                "tile unary helper expects four arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile unary rows")?;
        let cols = const_usize(cols, "tile unary cols")?;
        let dst = self.format_operand(dst);
        let src = self.format_operand(src);
        self.emit_nested_tile_loop("tile_un", rows, cols, |ctx, r, c| {
            if op == "-" {
                ctx.writeln(&format!("{dst}[{r}][{c}] = -{src}[{r}][{c}];"));
            } else {
                ctx.writeln(&format!("{dst}[{r}][{c}] = {op}({src}[{r}][{c}]);"));
            }
        });
        Ok(())
    }

    fn emit_tile_broadcast(&mut self, args: &[llir::Operand]) -> sile_core::Result<()> {
        let [dst, src, rows, cols] = args else {
            return Err(sile_core::Error::Compile(
                "tile_broadcast_f32 expects four arguments".into(),
            ));
        };
        let rows = const_usize(rows, "tile broadcast rows")?;
        let cols = const_usize(cols, "tile broadcast cols")?;
        let dst = self.format_operand(dst);
        let src = self.format_operand(src);
        self.emit_nested_tile_loop("tile_bcast", rows, cols, |ctx, r, c| {
            ctx.writeln(&format!("{dst}[{r}][{c}] = {src}[{r}][0];"));
        });
        Ok(())
    }

    fn emit_matmul_fragment(&mut self, args: &[llir::Operand]) -> sile_core::Result<()> {
        let [dst, a, b, acc, tile_m, tile_n, tile_k] = args else {
            return Err(sile_core::Error::Compile(
                "matmul_fragment expects seven arguments".into(),
            ));
        };
        let tile_m = const_usize(tile_m, "matmul tile_m")?;
        let tile_n = const_usize(tile_n, "matmul tile_n")?;
        let tile_k = const_usize(tile_k, "matmul tile_k")?;
        let dst = self.format_operand(dst);
        let a = self.format_operand(a);
        let b = self.format_operand(b);
        let acc = self.format_operand(acc);

        self.writeln(&format!(
            "for (int mma_r = 0; mma_r < {tile_m}; ++mma_r) {{"
        ));
        self.indent += 1;
        self.writeln(&format!(
            "for (int mma_c = 0; mma_c < {tile_n}; ++mma_c) {{"
        ));
        self.indent += 1;
        self.writeln(&format!("{dst}[mma_r][mma_c] = {acc}[mma_r][mma_c];"));
        self.writeln(&format!(
            "for (int mma_k = 0; mma_k < {tile_k}; ++mma_k) {{"
        ));
        self.indent += 1;
        self.writeln(&format!(
            "{dst}[mma_r][mma_c] += {a}[mma_r][mma_k] * {b}[mma_k][mma_c];"
        ));
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        Ok(())
    }

    fn emit_reduce(&mut self, args: &[llir::Operand], is_max: bool) -> sile_core::Result<()> {
        let [dst, src, axis, in_rows, in_cols] = args else {
            return Err(sile_core::Error::Compile(
                "reduce intrinsic expects five arguments".into(),
            ));
        };
        let axis = const_usize(axis, "reduce axis")?;
        let in_rows = const_usize(in_rows, "reduce rows")?;
        let in_cols = const_usize(in_cols, "reduce cols")?;
        let dst = self.format_operand(dst);
        let src = self.format_operand(src);

        match axis {
            1 => {
                self.writeln(&format!(
                    "for (int red_r = 0; red_r < {in_rows}; ++red_r) {{"
                ));
                self.indent += 1;
                let init = if is_max {
                    format!("{src}[red_r][0]")
                } else {
                    "0.0f".into()
                };
                self.writeln(&format!("{dst}[red_r][0] = {init};"));
                self.writeln(&format!(
                    "for (int red_c = {}; red_c < {in_cols}; ++red_c) {{",
                    if is_max { 1 } else { 0 }
                ));
                self.indent += 1;
                if is_max {
                    self.writeln(&format!(
                        "{dst}[red_r][0] = metal::max({dst}[red_r][0], {src}[red_r][red_c]);"
                    ));
                } else {
                    self.writeln(&format!("{dst}[red_r][0] += {src}[red_r][red_c];"));
                }
                self.indent -= 1;
                self.writeln("}");
                self.indent -= 1;
                self.writeln("}");
            }
            0 => {
                self.writeln(&format!(
                    "for (int red_c = 0; red_c < {in_cols}; ++red_c) {{"
                ));
                self.indent += 1;
                let init = if is_max {
                    format!("{src}[0][red_c]")
                } else {
                    "0.0f".into()
                };
                self.writeln(&format!("{dst}[0][red_c] = {init};"));
                self.writeln(&format!(
                    "for (int red_r = {}; red_r < {in_rows}; ++red_r) {{",
                    if is_max { 1 } else { 0 }
                ));
                self.indent += 1;
                if is_max {
                    self.writeln(&format!(
                        "{dst}[0][red_c] = metal::max({dst}[0][red_c], {src}[red_r][red_c]);"
                    ));
                } else {
                    self.writeln(&format!("{dst}[0][red_c] += {src}[red_r][red_c];"));
                }
                self.indent -= 1;
                self.writeln("}");
                self.indent -= 1;
                self.writeln("}");
            }
            other => {
                return Err(sile_core::Error::Compile(format!(
                    "unsupported reduction axis in LLIR Metal codegen: {other}"
                )));
            }
        }
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
        let Some(block) = self.get_block(target) else {
            return;
        };
        let assignments: Vec<_> = block
            .params
            .iter()
            .zip(args.iter())
            .map(|(param, arg)| (param.id, self.format_operand(arg)))
            .collect();
        for (param_id, arg) in assignments {
            self.writeln(&format!("{} = {};", self.value_name(param_id), arg));
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

    fn param_index(&self, operand: &llir::Operand) -> Option<usize> {
        match operand {
            llir::Operand::Value(id) => self.param_indices.get(id).copied(),
            _ => None,
        }
    }

    fn value_name(&self, id: llir::ValueId) -> String {
        self.value_names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("v{}", id.0))
    }

    fn format_operand(&self, operand: &llir::Operand) -> String {
        match operand {
            llir::Operand::Value(id) => self.value_name(*id),
            llir::Operand::Const(llir::Constant::Int(value)) => value.to_string(),
            llir::Operand::Const(llir::Constant::Float(value)) => float_literal(*value),
            llir::Operand::Const(llir::Constant::Bool(value)) => {
                if *value {
                    "true".into()
                } else {
                    "false".into()
                }
            }
        }
    }

    fn get_block(&self, id: llir::BlockId) -> Option<&llir::BasicBlock> {
        self.func.blocks.iter().find(|block| block.id == id)
    }

    fn writeln(&mut self, line: &str) {
        self.out
            .push_str(&format!("{}{}\n", "  ".repeat(self.indent), line));
    }
}

fn build_value_names(func: &llir::Function) -> HashMap<llir::ValueId, String> {
    let mut names = HashMap::new();
    for param in &func.params {
        names.insert(param.id, param.name.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            names.insert(param.id, param.name.clone());
        }
        for inst in &block.insts {
            if let (Some(id), Some(name)) = (inst.result, inst.result_name.as_ref()) {
                names.insert(id, name.clone());
            }
        }
    }
    names
}

fn build_param_indices(func: &llir::Function) -> HashMap<llir::ValueId, usize> {
    func.params
        .iter()
        .enumerate()
        .map(|(idx, param)| (param.id, idx))
        .collect()
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
        format!("{base}* {name} = &{storage_name};")
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{base} (*{name}){suffix} = {storage_name};")
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
        llir::AddressSpace::Generic | llir::AddressSpace::Private => "",
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

fn array_dims(ty: &llir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let llir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

fn array_base_type(ty: &llir::Type) -> String {
    let mut current = ty;
    while let llir::Type::Array { elem, .. } = current {
        current = elem;
    }
    metal_type(current)
}

fn metal_bin_op(op: llir::BinOp) -> &'static str {
    match op {
        llir::BinOp::Add => "+",
        llir::BinOp::Sub => "-",
        llir::BinOp::Mul => "*",
        llir::BinOp::Div => "/",
        llir::BinOp::And => "&",
        llir::BinOp::Or => "|",
    }
}

fn metal_cmp_pred(pred: llir::CmpPred) -> &'static str {
    match pred {
        llir::CmpPred::Eq => "==",
        llir::CmpPred::Ne => "!=",
        llir::CmpPred::Slt | llir::CmpPred::Olt => "<",
        llir::CmpPred::Sle | llir::CmpPred::Ole => "<=",
        llir::CmpPred::Sgt | llir::CmpPred::Ogt => ">",
        llir::CmpPred::Sge | llir::CmpPred::Oge => ">=",
    }
}

fn const_usize(operand: &llir::Operand, what: &str) -> sile_core::Result<usize> {
    match operand {
        llir::Operand::Const(llir::Constant::Int(value)) if *value >= 0 => Ok(*value as usize),
        _ => Err(sile_core::Error::Compile(format!(
            "{what} must be a non-negative integer constant"
        ))),
    }
}

fn float_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value}f")
    }
}
