use std::collections::HashMap;

use sile_hir::ParamKind;
use sile_lir::ir::*;
use sile_lir::{KernelAbi, ValueInfo, ValueInfoTable};

#[derive(Clone, Copy)]
struct TilePlan {
    output_param: usize,
    rows: i64,
    cols: i64,
}

struct StructuredLoop {
    entry: String,
    header: String,
    body: String,
    exit: String,
}

impl StructuredLoop {
    fn analyze(func: &Function) -> Option<Self> {
        let entry = func.blocks.first()?;
        let Terminator::Br { target: header } = &entry.terminator else {
            return None;
        };
        let header_block = func.get_block(header)?;
        let Terminator::CondBr {
            true_target,
            false_target,
            ..
        } = &header_block.terminator
        else {
            return None;
        };
        let body_block = func.get_block(true_target)?;
        let Terminator::Br {
            target: back_target,
        } = &body_block.terminator
        else {
            return None;
        };
        if back_target != header {
            return None;
        }
        let exit_block = func.get_block(false_target)?;
        if !matches!(exit_block.terminator, Terminator::Ret(_)) {
            return None;
        }
        Some(Self {
            entry: entry.label.clone(),
            header: header.clone(),
            body: true_target.clone(),
            exit: false_target.clone(),
        })
    }
}

pub fn generate(
    func: &Function,
    abi: &KernelAbi,
    value_info: &ValueInfoTable,
) -> sile_core::Result<String> {
    let mut ctx = MetalCodegen {
        func,
        abi,
        tile_plan: infer_tile_plan(func),
        inst_info: &value_info.instructions,
        inst_shapes: instruction_shapes(value_info),
        param_names: Vec::new(),
        inst_names: Vec::new(),
        indent: 0,
        out: String::new(),
    };

    ctx.emit_prologue();
    ctx.emit_kernel_signature();
    ctx.emit_kernel_body();

    Ok(ctx.out)
}

struct MetalCodegen<'a> {
    func: &'a Function,
    abi: &'a KernelAbi,
    tile_plan: Option<TilePlan>,
    inst_info: &'a [ValueInfo],
    inst_shapes: Vec<Option<Vec<i64>>>,
    param_names: Vec<String>,
    inst_names: Vec<String>,
    indent: usize,
    out: String,
}

impl<'a> MetalCodegen<'a> {
    fn emit_prologue(&mut self) {
        self.out.push_str("#include <metal_stdlib>\n");
        self.out.push_str("using namespace metal;\n");
        self.out.push_str("\n");
    }

    fn emit_kernel_signature(&mut self) {
        let fn_name = format!("kernel void sile_kernel_{}", self.func.name);
        self.out.push_str(&format!("{}(\n", fn_name));

        for (i, param) in self.abi.params.iter().enumerate() {
            let qualifier = match param.kind {
                ParamKind::Input => "const ",
                ParamKind::Output => "",
            };
            self.out.push_str(&format!(
                "    {}device float* buf_{} [[buffer({})]]",
                qualifier, i, i
            ));
            self.out.push_str(",\n");
        }

        let buffer_count = self.abi.params.len();
        self.out.push_str(&format!(
            "    device int64_t* shapes [[buffer({})]],\n",
            buffer_count
        ));
        self.out
            .push_str("    uint2 gid [[thread_position_in_grid]],\n");
        self.out
            .push_str("    uint2 tid [[thread_position_in_threadgroup]],\n");
        self.out
            .push_str("    uint2 tgsize [[threads_per_threadgroup]]\n");
        self.out.push_str(") {\n");
        self.indent = 1;
        self.writeln("(void)tid;");
        self.writeln("(void)tgsize;");
    }

    fn emit_kernel_body(&mut self) {
        for i in 0..self.abi.params.len() {
            self.param_names.push(format!("buf_{}", i));
        }

        let total_insts: usize = self
            .func
            .blocks
            .iter()
            .map(|block| block.instructions.len())
            .sum();
        for i in 0..total_insts {
            self.inst_names.push(format!("v{}", i));
        }

        self.writeln("uint sile_pid = gid.x;");

        for (param_idx, param) in self.abi.params.iter().enumerate() {
            for dim in 0..param.rank {
                let shape_idx = self.abi.shape_layout.offsets[param_idx] + dim;
                self.writeln(&format!(
                    "int64_t {}_dim_{} = shapes[{}];",
                    self.param_names[param_idx], dim, shape_idx
                ));
            }
        }

        self.writeln("");
        self.emit_block_param_decls();

        if let Some(plan) = self.tile_plan {
            let output_name = self.param_names[plan.output_param].clone();
            let output_rank = self.abi.params[plan.output_param].rank;
            if output_rank == 1 {
                self.writeln(&format!(
                    "int64_t sile_total_tiles = {}_dim_0 / {};",
                    output_name, plan.cols
                ));
            } else {
                self.writeln(&format!(
                    "int64_t sile_tiles_n = {}_dim_1 / {};",
                    output_name, plan.cols
                ));
                self.writeln(&format!(
                    "int64_t sile_total_tiles = ({}_dim_0 / {}) * sile_tiles_n;",
                    output_name, plan.rows
                ));
            }
            self.writeln("if (sile_pid < sile_total_tiles) {");
        } else {
            let scalar_extent = if self.abi.params.is_empty() {
                "0".to_string()
            } else {
                format!("{}_dim_0", self.param_names[0])
            };
            self.writeln(&format!("if (sile_pid < {}) {{", scalar_extent));
        }
        self.indent += 1;

        let inst_offsets = instruction_offsets(self.func);
        if let Some(loop_cfg) = StructuredLoop::analyze(self.func) {
            self.emit_structured_loop(&loop_cfg, &inst_offsets);
        } else {
            for block in &self.func.blocks {
                let base_offset = inst_offsets.get(&block.label).copied().unwrap_or_default();
                self.emit_block_instructions(block, base_offset);
            }
        }

        self.indent -= 1;
        self.writeln("}");
        self.indent = 0;
        self.out.push_str("}\n");
    }

    fn emit_block_instructions(&mut self, block: &BasicBlock, base_offset: usize) {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let global_idx = base_offset + idx;
            let code = emit_instruction(
                inst,
                &self.param_names,
                &self.inst_names,
                self.abi,
                self.tile_plan,
                &self.inst_shapes,
                global_idx,
            );
            for line in code.lines() {
                if line.is_empty() {
                    self.writeln("");
                } else {
                    self.writeln(line);
                }
            }
        }
    }

    fn emit_structured_loop(
        &mut self,
        loop_cfg: &StructuredLoop,
        inst_offsets: &HashMap<String, usize>,
    ) {
        let entry_block = self
            .func
            .get_block(&loop_cfg.entry)
            .cloned()
            .expect("entry block");
        let header_block = self
            .func
            .get_block(&loop_cfg.header)
            .cloned()
            .expect("header block");
        let body_block = self
            .func
            .get_block(&loop_cfg.body)
            .cloned()
            .expect("body block");
        let exit_block = self
            .func
            .get_block(&loop_cfg.exit)
            .cloned()
            .expect("exit block");

        let entry_offset = inst_offsets
            .get(&loop_cfg.entry)
            .copied()
            .unwrap_or_default();
        let header_offset = inst_offsets
            .get(&loop_cfg.header)
            .copied()
            .unwrap_or_default();
        let body_offset = inst_offsets
            .get(&loop_cfg.body)
            .copied()
            .unwrap_or_default();
        let exit_offset = inst_offsets
            .get(&loop_cfg.exit)
            .copied()
            .unwrap_or_default();

        self.emit_block_instructions(&entry_block, entry_offset);
        self.emit_phi_assignments(&loop_cfg.entry, &loop_cfg.header);

        let cond_name = match &header_block.terminator {
            Terminator::CondBr { cond, .. } => {
                resolve_value_name(cond, &self.param_names, &self.inst_names)
            }
            _ => unreachable!("structured loop header must branch"),
        };

        self.writeln("while (true) {");
        self.indent += 1;
        self.emit_block_instructions(&header_block, header_offset);
        self.writeln(&format!("if (!({})) {{", cond_name));
        self.indent += 1;
        self.emit_phi_assignments(&loop_cfg.header, &loop_cfg.exit);
        self.writeln("break;");
        self.indent -= 1;
        self.writeln("}");
        self.emit_phi_assignments(&loop_cfg.header, &loop_cfg.body);
        self.emit_block_instructions(&body_block, body_offset);
        self.emit_phi_assignments(&loop_cfg.body, &loop_cfg.header);
        self.indent -= 1;
        self.writeln("}");

        self.emit_block_instructions(&exit_block, exit_offset);
    }

    fn emit_phi_assignments(&mut self, pred_label: &str, target_label: &str) {
        let Some(target_block) = self.func.get_block(target_label) else {
            return;
        };
        let phi_nodes = target_block.phi_nodes.clone();

        for phi in &phi_nodes {
            let Some((incoming, _)) = phi.incoming.iter().find(|(_, pred)| pred == pred_label)
            else {
                continue;
            };
            let Some(info) = self.value_info_for_name(&phi.dest).cloned() else {
                continue;
            };
            match info {
                ValueInfo::Tile { rows, cols, .. } => {
                    let dst_arr = format!("{}_tlocal", phi.dest);
                    match incoming {
                        Value::Const(Constant::Int(v)) => {
                            self.writeln(&format!(
                                "for (int {0}r = 0; {0}r < {1}; ++{0}r)",
                                phi.dest, rows
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int {0}c = 0; {0}c < {1}; ++{0}c)",
                                phi.dest, cols
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}r][{}c] = {};",
                                dst_arr, phi.dest, phi.dest, v
                            ));
                            self.indent -= 1;
                            self.indent -= 1;
                        }
                        Value::Const(Constant::Float(v)) => {
                            self.writeln(&format!(
                                "for (int {0}r = 0; {0}r < {1}; ++{0}r)",
                                phi.dest, rows
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int {0}c = 0; {0}c < {1}; ++{0}c)",
                                phi.dest, cols
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}r][{}c] = {}f;",
                                dst_arr,
                                phi.dest,
                                phi.dest,
                                float_literal(*v)
                            ));
                            self.indent -= 1;
                            self.indent -= 1;
                        }
                        Value::Const(Constant::Bool(v)) => {
                            let literal = if *v { "1" } else { "0" };
                            self.writeln(&format!(
                                "for (int {0}r = 0; {0}r < {1}; ++{0}r)",
                                phi.dest, rows
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int {0}c = 0; {0}c < {1}; ++{0}c)",
                                phi.dest, cols
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}r][{}c] = {};",
                                dst_arr, phi.dest, phi.dest, literal
                            ));
                            self.indent -= 1;
                            self.indent -= 1;
                        }
                        _ => {
                            let src_arr = resolve_tile_array_name(
                                incoming,
                                &self.param_names,
                                &self.inst_names,
                            );
                            self.writeln(&format!(
                                "for (int {0}r = 0; {0}r < {1}; ++{0}r)",
                                phi.dest, rows
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int {0}c = 0; {0}c < {1}; ++{0}c)",
                                phi.dest, cols
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}r][{}c] = {}[{}r][{}c];",
                                dst_arr, phi.dest, phi.dest, src_arr, phi.dest, phi.dest
                            ));
                            self.indent -= 1;
                            self.indent -= 1;
                        }
                    }
                }
                ValueInfo::Scalar { .. } => {
                    let incoming_name =
                        resolve_value_name(incoming, &self.param_names, &self.inst_names);
                    self.writeln(&format!("{} = {};", phi.dest, incoming_name));
                }
                _ => {
                    let incoming_name =
                        resolve_value_name(incoming, &self.param_names, &self.inst_names);
                    self.writeln(&format!("{} = {};", phi.dest, incoming_name));
                }
            }
        }
    }

    fn emit_block_param_decls(&mut self) {
        let mut inst_idx = 0usize;
        let mut emitted = false;
        for block in &self.func.blocks {
            for inst in &block.instructions {
                if matches!(inst, Instruction::BlockParam) {
                    let name = self
                        .inst_names
                        .get(inst_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("v{}", inst_idx));
                    if let Some(info) = self.inst_info.get(inst_idx) {
                        match info {
                            ValueInfo::Tile { rows, cols, .. } => {
                                self.writeln(&format!(
                                    "threadgroup float {}_tlocal[{}][{}];",
                                    name, rows, cols
                                ));
                            }
                            ValueInfo::Scalar { .. } => {
                                self.writeln(&format!("float {} = 0.0f;", name));
                            }
                            _ => {
                                self.writeln(&format!("int {} = 0;", name));
                            }
                        }
                        emitted = true;
                    }
                }
                inst_idx += 1;
            }
        }
        if emitted {
            self.writeln("");
        }
    }

    fn value_info_for_name(&self, name: &str) -> Option<&ValueInfo> {
        let idx = name.strip_prefix('v')?.parse::<usize>().ok()?;
        self.inst_info.get(idx)
    }

    fn writeln(&mut self, line: &str) {
        let indent = "  ".repeat(self.indent);
        self.out.push_str(&format!("{}{}\n", indent, line));
    }
}

fn infer_tile_plan(func: &Function) -> Option<TilePlan> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::TileStore2D {
                buf: Value::Param(output_param),
                rows,
                cols,
                ..
            } = inst
            {
                return Some(TilePlan {
                    output_param: *output_param,
                    rows: *rows,
                    cols: *cols,
                });
            }
        }
    }
    None
}

fn instruction_shapes(value_info: &ValueInfoTable) -> Vec<Option<Vec<i64>>> {
    value_info
        .instructions
        .iter()
        .map(|info| match info {
            ValueInfo::Tile { rows, cols, .. } => Some(vec![*rows, *cols]),
            _ => None,
        })
        .collect()
}

fn instruction_offsets(func: &Function) -> HashMap<String, usize> {
    let mut offsets = HashMap::new();
    let mut next = 0usize;
    for block in &func.blocks {
        offsets.insert(block.label.clone(), next);
        next += block.instructions.len();
    }
    offsets
}

fn emit_instruction(
    inst: &Instruction,
    param_names: &[String],
    inst_names: &[String],
    abi: &KernelAbi,
    tile_plan: Option<TilePlan>,
    inst_shapes: &[Option<Vec<i64>>],
    inst_idx: usize,
) -> String {
    let result_name = inst_names
        .get(inst_idx)
        .cloned()
        .unwrap_or_else(|| format!("v{}", inst_idx));
    let result_shape = inst_shapes.get(inst_idx).and_then(|shape| shape.clone());

    match inst {
        Instruction::BlockParam => String::new(),
        Instruction::Add(lhs, rhs) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                return emit_tile_binary(
                    &result_name,
                    rows,
                    cols,
                    "+",
                    lhs,
                    rhs,
                    param_names,
                    inst_names,
                );
            }
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} + {};", result_name, lhs_name, rhs_name)
        }
        Instruction::Sub(lhs, rhs) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                return emit_tile_binary(
                    &result_name,
                    rows,
                    cols,
                    "-",
                    lhs,
                    rhs,
                    param_names,
                    inst_names,
                );
            }
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} - {};", result_name, lhs_name, rhs_name)
        }
        Instruction::Mul(lhs, rhs) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                return emit_tile_binary(
                    &result_name,
                    rows,
                    cols,
                    "*",
                    lhs,
                    rhs,
                    param_names,
                    inst_names,
                );
            }
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} * {};", result_name, lhs_name, rhs_name)
        }
        Instruction::Div(lhs, rhs) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                return emit_tile_binary(
                    &result_name,
                    rows,
                    cols,
                    "/",
                    lhs,
                    rhs,
                    param_names,
                    inst_names,
                );
            }
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} / {};", result_name, lhs_name, rhs_name)
        }
        Instruction::Exp(value) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                let value_name = resolve_value_name(value, param_names, inst_names);
                return emit_tile_unary(&result_name, rows, cols, "metal::exp", &value_name);
            }
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("float {} = metal::exp({});", result_name, value_name)
        }
        Instruction::FNeg(value) => {
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("float {} = -{};", result_name, value_name)
        }
        Instruction::FMax(lhs, rhs) => {
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!(
                "float {} = metal::max({}, {});",
                result_name, lhs_name, rhs_name
            )
        }
        Instruction::FMin(lhs, rhs) => {
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!(
                "float {} = metal::min({}, {});",
                result_name, lhs_name, rhs_name
            )
        }
        Instruction::GetTileCoord { dim } => {
            let plan = tile_plan.expect("tile coord requires tile plan");
            let output_name = &param_names[plan.output_param];
            let output_rank = abi.params[plan.output_param].rank;
            if output_rank == 1 {
                format!("int {} = sile_pid;", result_name)
            } else {
                match dim {
                    0 => format!(
                        "int {} = sile_pid / ({}_dim_1 / {});",
                        result_name, output_name, plan.cols
                    ),
                    _ => format!(
                        "int {} = sile_pid % ({}_dim_1 / {});",
                        result_name, output_name, plan.cols
                    ),
                }
            }
        }
        Instruction::TileAlloc { rows, cols, init } => {
            let init_val = float_literal(*init);
            format!(
                "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c)\n    {0}_tlocal[{0}r][{0}c] = {3}f;",
                result_name, rows, cols, init_val
            )
        }
        Instruction::TileLoad2D {
            buf,
            rows,
            cols,
            row_tile,
            col_tile,
            stride_shape_idx,
        } => {
            let buf_name = resolve_value_name(buf, param_names, inst_names);
            let row_name = resolve_value_name(row_tile, param_names, inst_names);
            let col_name = resolve_value_name(col_tile, param_names, inst_names);
            let param_rank = buffer_rank(buf, abi);
            if param_rank == Some(1) {
                format!(
                    "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c) {{\n    int sile_index = {3} * {2} + {0}c;\n    {0}_tlocal[{0}r][{0}c] = {4}[sile_index];\n  }}",
                    result_name, rows, cols, col_name, buf_name
                )
            } else {
                let stride = buffer_dim_expr(buf, *stride_shape_idx, param_names, abi);
                format!(
                    "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c) {{\n    int sile_row = {3} * {1} + {0}r;\n    int sile_col = {5} * {2} + {0}c;\n    {0}_tlocal[{0}r][{0}c] = {4}[sile_row * {6} + sile_col];\n  }}",
                    result_name, rows, cols, row_name, buf_name, col_name, stride
                )
            }
        }
        Instruction::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            let a_name = resolve_value_name(a, param_names, inst_names);
            let b_name = resolve_value_name(b, param_names, inst_names);
            let acc_name = resolve_value_name(acc, param_names, inst_names);
            format!(
                "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c) {{\n    {0}_tlocal[{0}r][{0}c] = {6}_tlocal[{0}r][{0}c];\n    for (int {0}k = 0; {0}k < {3}; ++{0}k)\n      {0}_tlocal[{0}r][{0}c] += {4}_tlocal[{0}r][{0}k] * {5}_tlocal[{0}k][{0}c];\n  }}",
                result_name, tile_m, tile_n, tile_k, a_name, b_name, acc_name
            )
        }
        Instruction::TileReduceMax {
            value,
            axis,
            rows,
            cols,
        } => emit_tile_reduce(
            &result_name,
            value,
            *axis,
            *rows,
            *cols,
            "metal::max",
            param_names,
            inst_names,
        ),
        Instruction::TileReduceSum {
            value,
            axis,
            rows,
            cols,
        } => emit_tile_reduce(
            &result_name,
            value,
            *axis,
            *rows,
            *cols,
            "+",
            param_names,
            inst_names,
        ),
        Instruction::TileBroadcast { value, rows, cols } => {
            let value_name = resolve_tile_array_name(value, param_names, inst_names);
            format!(
                "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c)\n    {0}_tlocal[{0}r][{0}c] = {3}[{0}r][0];",
                result_name, rows, cols, value_name
            )
        }
        Instruction::TileStore2D {
            buf,
            value,
            rows,
            cols,
            row_tile,
            col_tile,
            stride_shape_idx,
        } => {
            let buf_name = resolve_value_name(buf, param_names, inst_names);
            let row_name = resolve_value_name(row_tile, param_names, inst_names);
            let col_name = resolve_value_name(col_tile, param_names, inst_names);
            let param_rank = buffer_rank(buf, abi);
            let src_arr = resolve_tile_array_name(value, param_names, inst_names);
            if param_rank == Some(1) {
                format!(
                    "for (int stor_r = 0; stor_r < {}; ++stor_r)\n  for (int stor_c = 0; stor_c < {}; ++stor_c) {{\n    int sile_index = {} * {} + stor_c;\n    {}[sile_index] = {}[stor_r][stor_c];\n  }}",
                    rows, cols, col_name, cols, buf_name, src_arr
                )
            } else {
                let stride = buffer_dim_expr(buf, *stride_shape_idx, param_names, abi);
                format!(
                    "for (int stor_r = 0; stor_r < {}; ++stor_r)\n  for (int stor_c = 0; stor_c < {}; ++stor_c) {{\n    int sile_row = {} * {} + stor_r;\n    int sile_col = {} * {} + stor_c;\n    {}[sile_row * {} + sile_col] = {}[stor_r][stor_c];\n  }}",
                    rows, cols, row_name, rows, col_name, cols, buf_name, stride, src_arr
                )
            }
        }
        _ => String::new(),
    }
}

fn buffer_dim_expr(value: &Value, dim: usize, param_names: &[String], abi: &KernelAbi) -> String {
    match value {
        Value::Param(param_idx)
            if *param_idx < abi.params.len() && dim < abi.params[*param_idx].rank =>
        {
            format!("{}_dim_{}", param_names[*param_idx], dim)
        }
        _ => "1".to_string(),
    }
}

fn resolve_value_name(value: &Value, param_names: &[String], inst_names: &[String]) -> String {
    match value {
        Value::Param(i) => param_names
            .get(*i)
            .cloned()
            .unwrap_or_else(|| format!("buf_{}", i)),
        Value::Const(Constant::Int(v)) => format!("{}", v),
        Value::Const(Constant::Float(v)) => format!("{}", v),
        Value::Const(Constant::Bool(v)) => {
            if *v {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        Value::Inst(i) => inst_names
            .get(*i)
            .cloned()
            .unwrap_or_else(|| format!("v{}", i)),
        Value::ShapeDim(i) => format!("shapes[{}]", i),
    }
}

fn resolve_tile_array_name(value: &Value, param_names: &[String], inst_names: &[String]) -> String {
    let name = resolve_value_name(value, param_names, inst_names);
    if matches!(value, Value::Inst(_)) {
        format!("{}_tlocal", name)
    } else {
        name
    }
}

fn float_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{:.1}", value)
    } else {
        format!("{}", value)
    }
}

fn shape2(shape: &Option<Vec<i64>>) -> Option<[i64; 2]> {
    let dims = shape.as_ref()?;
    match dims.as_slice() {
        [rows, cols] => Some([*rows, *cols]),
        _ => None,
    }
}

fn emit_tile_binary(
    result_name: &str,
    rows: i64,
    cols: i64,
    op: &str,
    lhs: &Value,
    rhs: &Value,
    param_names: &[String],
    inst_names: &[String],
) -> String {
    let lhs_name = resolve_value_name(lhs, param_names, inst_names);
    let rhs_name = resolve_value_name(rhs, param_names, inst_names);
    let lhs_arr = if matches!(lhs, Value::Inst(_)) {
        format!("{}_tlocal", lhs_name)
    } else {
        lhs_name
    };
    let rhs_arr = if matches!(rhs, Value::Inst(_)) {
        format!("{}_tlocal", rhs_name)
    } else {
        rhs_name
    };
    format!(
        "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c)\n    {0}_tlocal[{0}r][{0}c] = {3}[{0}r][{0}c] {5} {4}[{0}r][{0}c];",
        result_name, rows, cols, lhs_arr, rhs_arr, op
    )
}

fn emit_tile_unary(
    result_name: &str,
    rows: i64,
    cols: i64,
    func: &str,
    value_name: &str,
) -> String {
    let val_arr = if value_name.starts_with("v") && value_name[1..].parse::<usize>().is_ok() {
        format!("{}_tlocal", value_name)
    } else {
        value_name.to_string()
    };
    format!(
        "threadgroup float {0}_tlocal[{1}][{2}];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r)\n  for (int {0}c = 0; {0}c < {2}; ++{0}c)\n    {0}_tlocal[{0}r][{0}c] = {3}({4}[{0}r][{0}c]);",
        result_name, rows, cols, func, val_arr
    )
}

fn buffer_rank(value: &Value, abi: &KernelAbi) -> Option<usize> {
    match value {
        Value::Param(param_idx) => abi.params.get(*param_idx).map(|param| param.rank),
        _ => None,
    }
}

fn emit_tile_reduce(
    result_name: &str,
    value: &Value,
    axis: i64,
    rows: i64,
    cols: i64,
    op: &str,
    param_names: &[String],
    inst_names: &[String],
) -> String {
    let value_name = resolve_value_name(value, param_names, inst_names);
    let value_arr = if matches!(value, Value::Inst(_)) {
        format!("{}_tlocal", value_name)
    } else {
        value_name
    };
    if axis == 1 {
        if op == "metal::max" {
            format!(
                "threadgroup float {0}_tlocal[{1}][1];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r) {{\n  {0}_tlocal[{0}r][0] = {2}[{0}r][0];\n  for (int {0}c = 1; {0}c < {3}; ++{0}c)\n    {0}_tlocal[{0}r][0] = metal::max({0}_tlocal[{0}r][0], {2}[{0}r][{0}c]);\n}}",
                result_name, rows, value_arr, cols
            )
        } else {
            format!(
                "threadgroup float {0}_tlocal[{1}][1];\nfor (int {0}r = 0; {0}r < {1}; ++{0}r) {{\n  {0}_tlocal[{0}r][0] = 0.0f;\n  for (int {0}c = 0; {0}c < {3}; ++{0}c)\n    {0}_tlocal[{0}r][0] += {2}[{0}r][{0}c];\n}}",
                result_name, rows, value_arr, cols
            )
        }
    } else {
        if op == "metal::max" {
            format!(
                "threadgroup float {0}_tlocal[1][{1}];\nfor (int {0}c = 0; {0}c < {1}; ++{0}c) {{\n  {0}_tlocal[0][{0}c] = {2}[0][{0}c];\n  for (int {0}r = 1; {0}r < {3}; ++{0}r)\n    {0}_tlocal[0][{0}c] = metal::max({0}_tlocal[0][{0}c], {2}[{0}r][{0}c]);\n}}",
                result_name, cols, value_arr, rows
            )
        } else {
            format!(
                "threadgroup float {0}_tlocal[1][{1}];\nfor (int {0}c = 0; {0}c < {1}; ++{0}c) {{\n  {0}_tlocal[0][{0}c] = 0.0f;\n  for (int {0}r = 0; {0}r < {3}; ++{0}r)\n    {0}_tlocal[0][{0}c] += {2}[{0}r][{0}c];\n}}",
                result_name, cols, value_arr, rows
            )
        }
    }
}
