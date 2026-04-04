use sile_hir::ParamKind;
use sile_lir::ir::*;
use sile_lir::{KernelAbi, ValueInfo, ValueInfoTable};

#[derive(Clone, Copy)]
struct TilePlan {
    output_param: usize,
    rows: i64,
    cols: i64,
}

pub fn generate(
    func: &Function,
    abi: &KernelAbi,
    value_info: &ValueInfoTable,
) -> sile_core::Result<String> {
    let mut ctx = CCodegen {
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
    ctx.emit_wrapper_signature();
    ctx.emit_wrapper_body();

    Ok(ctx.out)
}

struct CCodegen<'a> {
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

impl<'a> CCodegen<'a> {
    fn emit_prologue(&mut self) {
        self.out.push_str("#include <stdint.h>\n");
        self.out.push_str("#include <math.h>\n");
        self.out.push_str("#include <omp.h>\n");
        self.out.push_str("\n");
    }

    fn emit_wrapper_signature(&mut self) {
        let fn_name = format!("sile_kernel_{}", self.func.name);
        self.out.push_str(&format!("void {}(\n", fn_name));
        self.out.push_str("    void** buffers,\n");
        self.out.push_str("    int64_t num_threadgroups,\n");
        self.out.push_str("    int64_t threads_per_group,\n");
        self.out.push_str("    const int64_t* shapes,\n");
        self.out.push_str("    int64_t num_shapes\n");
        self.out.push_str(") {\n");
        self.indent = 1;
    }

    fn emit_wrapper_body(&mut self) {
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

        for (i, param) in self.abi.params.iter().enumerate() {
            let qualifier = match param.kind {
                ParamKind::Input => "const ",
                ParamKind::Output => "",
            };
            self.writeln(&format!(
                "{}float* {} = ({}float*)buffers[{}];",
                qualifier, self.param_names[i], qualifier, i
            ));
        }
        self.writeln("");

        for (param_idx, param) in self.abi.params.iter().enumerate() {
            for dim in 0..param.rank {
                let shape_idx = self.abi.shape_layout.offsets[param_idx] + dim;
                self.writeln(&format!(
                    "int64_t {}_dim_{} = shapes[{}];",
                    self.param_names[param_idx], dim, shape_idx
                ));
            }
        }
        self.writeln("(void)num_shapes;");
        self.writeln("");

        self.emit_block_param_decls();

        self.writeln("#pragma omp parallel for schedule(static)");
        self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
        self.indent += 1;
        self.writeln("int64_t base = tg * threads_per_group;");
        self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
        self.indent += 1;
        self.writeln("int64_t sile_pid = base + t;");

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

        let mut inst_offset = 0;
        if let Some(entry_label) = self.func.blocks.first().map(|block| block.label.clone()) {
            self.writeln(&format!("goto {};", entry_label));
            self.writeln("");
        }
        for block in &self.func.blocks {
            self.emit_block(block, inst_offset);
            inst_offset += block.instructions.len();
        }
        self.writeln("sile_block_end:;");

        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");

        self.indent = 0;
        self.out.push_str("}\n");
    }

    fn emit_block(&mut self, block: &BasicBlock, base_offset: usize) {
        self.writeln(&format!("{}:", block.label));
        self.indent += 1;
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
        self.emit_terminator(block);
        self.indent -= 1;
        self.writeln("");
    }

    fn emit_terminator(&mut self, block: &BasicBlock) {
        match &block.terminator {
            Terminator::Br { target } => {
                self.emit_phi_assignments(&block.label, target);
                self.writeln(&format!("goto {};", target));
            }
            Terminator::CondBr {
                cond,
                true_target,
                false_target,
            } => {
                let cond_name = resolve_value_name(cond, &self.param_names, &self.inst_names);
                self.writeln(&format!("if ({}) {{", cond_name));
                self.indent += 1;
                self.emit_phi_assignments(&block.label, true_target);
                self.writeln(&format!("goto {};", true_target));
                self.indent -= 1;
                self.writeln("} else {");
                self.indent += 1;
                self.emit_phi_assignments(&block.label, false_target);
                self.writeln(&format!("goto {};", false_target));
                self.indent -= 1;
                self.writeln("}");
            }
            Terminator::Switch { .. } => {
                self.writeln("goto sile_block_end;");
            }
            Terminator::Ret(_) => {
                self.writeln("goto sile_block_end;");
            }
        }
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
                    let row_idx = format!("{}_r", phi.dest);
                    let col_idx = format!("{}_c", phi.dest);
                    match incoming {
                        Value::Const(Constant::Int(v)) => {
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                row_idx, row_idx, rows, row_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                col_idx, col_idx, cols, col_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}][{}] = {};",
                                phi.dest, row_idx, col_idx, v
                            ));
                            self.indent -= 1;
                            self.writeln("}");
                            self.indent -= 1;
                            self.writeln("}");
                        }
                        Value::Const(Constant::Float(v)) => {
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                row_idx, row_idx, rows, row_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                col_idx, col_idx, cols, col_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}][{}] = {}f;",
                                phi.dest,
                                row_idx,
                                col_idx,
                                float_literal(*v)
                            ));
                            self.indent -= 1;
                            self.writeln("}");
                            self.indent -= 1;
                            self.writeln("}");
                        }
                        Value::Const(Constant::Bool(v)) => {
                            let literal = if *v { "1" } else { "0" };
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                row_idx, row_idx, rows, row_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                col_idx, col_idx, cols, col_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}][{}] = {};",
                                phi.dest, row_idx, col_idx, literal
                            ));
                            self.indent -= 1;
                            self.writeln("}");
                            self.indent -= 1;
                            self.writeln("}");
                        }
                        _ => {
                            let incoming_name =
                                resolve_value_name(incoming, &self.param_names, &self.inst_names);
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                row_idx, row_idx, rows, row_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "for (int64_t {} = 0; {} < {}; ++{}) {{",
                                col_idx, col_idx, cols, col_idx
                            ));
                            self.indent += 1;
                            self.writeln(&format!(
                                "{}[{}][{}] = {}[{}][{}];",
                                phi.dest, row_idx, col_idx, incoming_name, row_idx, col_idx
                            ));
                            self.indent -= 1;
                            self.writeln("}");
                            self.indent -= 1;
                            self.writeln("}");
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
                                self.writeln(&format!("float {}[{}][{}];", name, rows, cols));
                            }
                            ValueInfo::Scalar { .. } => {
                                self.writeln(&format!("float {} = 0.0f;", name));
                            }
                            _ => {
                                self.writeln(&format!("int64_t {} = 0;", name));
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
        Instruction::Alloca { .. } => String::new(),
        Instruction::Load { ptr, .. } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            format!("float {} = {}[sile_pid];", result_name, ptr_name)
        }
        Instruction::Store { ptr, value, .. } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("{}[sile_pid] = {};", ptr_name, value_name)
        }
        Instruction::Gep { ptr, indices } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            let index_exprs: Vec<String> = indices
                .iter()
                .map(|value| resolve_value_name(value, param_names, inst_names))
                .collect();
            if index_exprs.is_empty() {
                format!("float* {} = {} + sile_pid;", result_name, ptr_name)
            } else {
                format!(
                    "float* {} = {} + {};",
                    result_name,
                    ptr_name,
                    index_exprs.join(" + ")
                )
            }
        }
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
        Instruction::FNeg(value) => {
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("float {} = -{};", result_name, value_name)
        }
        Instruction::FMax(lhs, rhs) => {
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = fmaxf({}, {});", result_name, lhs_name, rhs_name)
        }
        Instruction::FMin(lhs, rhs) => {
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = fminf({}, {});", result_name, lhs_name, rhs_name)
        }
        Instruction::Exp(value) => {
            if let Some([rows, cols]) = shape2(&result_shape) {
                let value_name = resolve_value_name(value, param_names, inst_names);
                return emit_tile_unary(&result_name, rows, cols, "expf", &value_name);
            }
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("float {} = expf({});", result_name, value_name)
        }
        Instruction::Icmp(op, lhs, rhs) | Instruction::Fcmp(op, lhs, rhs) => {
            let lhs_name = resolve_value_name(lhs, param_names, inst_names);
            let rhs_name = resolve_value_name(rhs, param_names, inst_names);
            let cmp = cmp_op_to_c(op);
            format!("int {} = {} {} {};", result_name, lhs_name, cmp, rhs_name)
        }
        Instruction::Trunc(value, _) | Instruction::FPToSI(value, _) => {
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("int {} = (int){};", result_name, value_name)
        }
        Instruction::ZExt(value, _) => {
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("int64_t {} = (int64_t){};", result_name, value_name)
        }
        Instruction::SIToFP(value, _) | Instruction::BitCast(value, _) => {
            let value_name = resolve_value_name(value, param_names, inst_names);
            format!("float {} = (float){};", result_name, value_name)
        }
        Instruction::Call { func, args, .. } => {
            let args: Vec<String> = args
                .iter()
                .map(|value| resolve_value_name(value, param_names, inst_names))
                .collect();
            format!("float {} = {}({});", result_name, func, args.join(", "))
        }
        Instruction::GetTileCoord { dim } => {
            let plan = tile_plan.expect("tile coord requires tile plan");
            let output_name = &param_names[plan.output_param];
            let output_rank = abi.params[plan.output_param].rank;
            if output_rank == 1 {
                format!("int64_t {} = sile_pid;", result_name)
            } else {
                match dim {
                    0 => format!(
                        "int64_t {} = sile_pid / ({}_dim_1 / {});",
                        result_name, output_name, plan.cols
                    ),
                    _ => format!(
                        "int64_t {} = sile_pid % ({}_dim_1 / {});",
                        result_name, output_name, plan.cols
                    ),
                }
            }
        }
        Instruction::TileAlloc { rows, cols, init } => {
            let init_literal = float_literal(*init);
            let row_idx = format!("{}_r", result_name);
            let col_idx = format!("{}_c", result_name);
            format!(
                "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][{}] = {}f;\n  }}\n}}",
                result_name,
                rows,
                cols,
                row_idx,
                row_idx,
                rows,
                row_idx,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                row_idx,
                col_idx,
                init_literal
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
            let row_idx = format!("{}_r", result_name);
            let col_idx = format!("{}_c", result_name);
            let param_rank = buffer_rank(buf, abi);
            if param_rank == Some(1) {
                format!(
                    "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    int64_t sile_index = {} * {} + {};\n    {}[{}][{}] = {}[sile_index];\n  }}\n}}",
                    result_name,
                    rows,
                    cols,
                    row_idx,
                    row_idx,
                    rows,
                    row_idx,
                    col_idx,
                    col_idx,
                    cols,
                    col_idx,
                    col_name,
                    cols,
                    col_idx,
                    result_name,
                    row_idx,
                    col_idx,
                    buf_name
                )
            } else {
                let stride = buffer_dim_expr(buf, *stride_shape_idx, param_names, abi);
                format!(
                    "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    int64_t sile_row = {} * {} + {};\n    int64_t sile_col = {} * {} + {};\n    {}[{}][{}] = {}[sile_row * {} + sile_col];\n  }}\n}}",
                    result_name,
                    rows,
                    cols,
                    row_idx,
                    row_idx,
                    rows,
                    row_idx,
                    col_idx,
                    col_idx,
                    cols,
                    col_idx,
                    row_name,
                    rows,
                    row_idx,
                    col_name,
                    cols,
                    col_idx,
                    result_name,
                    row_idx,
                    col_idx,
                    buf_name,
                    stride
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
            let row_idx = format!("{}_r", result_name);
            let col_idx = format!("{}_c", result_name);
            let k_idx = format!("{}_k", result_name);
            format!(
                "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][{}] = {}[{}][{}];\n    for (int64_t {} = 0; {} < {}; ++{}) {{\n      {}[{}][{}] += {}[{}][{}] * {}[{}][{}];\n    }}\n  }}\n}}",
                result_name,
                tile_m,
                tile_n,
                row_idx,
                row_idx,
                tile_m,
                row_idx,
                col_idx,
                col_idx,
                tile_n,
                col_idx,
                result_name,
                row_idx,
                col_idx,
                acc_name,
                row_idx,
                col_idx,
                k_idx,
                k_idx,
                tile_k,
                k_idx,
                result_name,
                row_idx,
                col_idx,
                a_name,
                row_idx,
                k_idx,
                b_name,
                k_idx,
                col_idx
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
            "fmaxf",
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
            let value_name = resolve_value_name(value, param_names, inst_names);
            let row_idx = format!("{}_r", result_name);
            let col_idx = format!("{}_c", result_name);
            format!(
                "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][{}] = {}[{}][0];\n  }}\n}}",
                result_name,
                rows,
                cols,
                row_idx,
                row_idx,
                rows,
                row_idx,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                row_idx,
                col_idx,
                value_name,
                row_idx
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
            let value_name = resolve_value_name(value, param_names, inst_names);
            let row_name = resolve_value_name(row_tile, param_names, inst_names);
            let col_name = resolve_value_name(col_tile, param_names, inst_names);
            let row_idx = format!("{}_r", result_name);
            let col_idx = format!("{}_c", result_name);
            let param_rank = buffer_rank(buf, abi);
            if param_rank == Some(1) {
                format!(
                    "for (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    int64_t sile_index = {} * {} + {};\n    {}[sile_index] = {}[{}][{}];\n  }}\n}}",
                    row_idx,
                    row_idx,
                    rows,
                    row_idx,
                    col_idx,
                    col_idx,
                    cols,
                    col_idx,
                    col_name,
                    cols,
                    col_idx,
                    buf_name,
                    value_name,
                    row_idx,
                    col_idx
                )
            } else {
                let stride = buffer_dim_expr(buf, *stride_shape_idx, param_names, abi);
                format!(
                    "for (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    int64_t sile_row = {} * {} + {};\n    int64_t sile_col = {} * {} + {};\n    {}[sile_row * {} + sile_col] = {}[{}][{}];\n  }}\n}}",
                    row_idx,
                    row_idx,
                    rows,
                    row_idx,
                    col_idx,
                    col_idx,
                    cols,
                    col_idx,
                    row_name,
                    rows,
                    row_idx,
                    col_name,
                    cols,
                    col_idx,
                    buf_name,
                    stride,
                    value_name,
                    row_idx,
                    col_idx
                )
            }
        }
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

fn cmp_op_to_c(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Slt | CmpOp::Ult | CmpOp::Olt => "<",
        CmpOp::Sle | CmpOp::Ule | CmpOp::Ole => "<=",
        CmpOp::Sgt | CmpOp::Ugt | CmpOp::Ogt => ">",
        CmpOp::Sge | CmpOp::Uge | CmpOp::Oge => ">=",
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
    let row_idx = format!("{}_r", result_name);
    let col_idx = format!("{}_c", result_name);
    format!(
        "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][{}] = {}[{}][{}] {} {}[{}][{}];\n  }}\n}}",
        result_name,
        rows,
        cols,
        row_idx,
        row_idx,
        rows,
        row_idx,
        col_idx,
        col_idx,
        cols,
        col_idx,
        result_name,
        row_idx,
        col_idx,
        lhs_name,
        row_idx,
        col_idx,
        op,
        rhs_name,
        row_idx,
        col_idx
    )
}

fn emit_tile_unary(
    result_name: &str,
    rows: i64,
    cols: i64,
    func: &str,
    value_name: &str,
) -> String {
    let row_idx = format!("{}_r", result_name);
    let col_idx = format!("{}_c", result_name);
    format!(
        "float {}[{}][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][{}] = {}({}[{}][{}]);\n  }}\n}}",
        result_name,
        rows,
        cols,
        row_idx,
        row_idx,
        rows,
        row_idx,
        col_idx,
        col_idx,
        cols,
        col_idx,
        result_name,
        row_idx,
        col_idx,
        func,
        value_name,
        row_idx,
        col_idx
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
    if axis == 1 {
        let row_idx = format!("{}_r", result_name);
        let col_idx = format!("{}_c", result_name);
        if op == "fmaxf" {
            format!(
                "float {}[{}][1];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  {}[{}][0] = {}[{}][0];\n  for (int64_t {} = 1; {} < {}; ++{}) {{\n    {}[{}][0] = fmaxf({}[{}][0], {}[{}][{}]);\n  }}\n}}",
                result_name,
                rows,
                row_idx,
                row_idx,
                rows,
                row_idx,
                result_name,
                row_idx,
                value_name,
                row_idx,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                row_idx,
                result_name,
                row_idx,
                value_name,
                row_idx,
                col_idx
            )
        } else {
            format!(
                "float {}[{}][1];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  {}[{}][0] = 0.0f;\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[{}][0] = {}[{}][0] + {}[{}][{}];\n  }}\n}}",
                result_name,
                rows,
                row_idx,
                row_idx,
                rows,
                row_idx,
                result_name,
                row_idx,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                row_idx,
                result_name,
                row_idx,
                value_name,
                row_idx,
                col_idx
            )
        }
    } else {
        let col_idx = format!("{}_c", result_name);
        let row_idx = format!("{}_r", result_name);
        if op == "fmaxf" {
            format!(
                "float {}[1][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  {}[0][{}] = {}[0][{}];\n  for (int64_t {} = 1; {} < {}; ++{}) {{\n    {}[0][{}] = fmaxf({}[0][{}], {}[{}][{}]);\n  }}\n}}",
                result_name,
                cols,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                col_idx,
                value_name,
                col_idx,
                row_idx,
                row_idx,
                rows,
                row_idx,
                result_name,
                col_idx,
                result_name,
                col_idx,
                value_name,
                row_idx,
                col_idx
            )
        } else {
            format!(
                "float {}[1][{}];\nfor (int64_t {} = 0; {} < {}; ++{}) {{\n  {}[0][{}] = 0.0f;\n  for (int64_t {} = 0; {} < {}; ++{}) {{\n    {}[0][{}] = {}[0][{}] + {}[{}][{}];\n  }}\n}}",
                result_name,
                cols,
                col_idx,
                col_idx,
                cols,
                col_idx,
                result_name,
                col_idx,
                row_idx,
                row_idx,
                rows,
                row_idx,
                result_name,
                col_idx,
                result_name,
                col_idx,
                value_name,
                row_idx,
                col_idx
            )
        }
    }
}
