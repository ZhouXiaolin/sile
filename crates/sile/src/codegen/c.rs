use crate::lir::ir::*;

pub struct KernelGenInfo {
    pub name: String,
    pub num_buffers: usize,
    pub buffer_kinds: Vec<BufferKind>,
    pub num_shapes: usize,
}

#[derive(Clone, Copy)]
pub enum BufferKind {
    Input,
    Output,
}

pub fn generate(func: &Function, info: &KernelGenInfo) -> crate::Result<String> {
    let mut ctx = CCodegen {
        func,
        info,
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
    info: &'a KernelGenInfo,
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
        let fn_name = format!("sile_kernel_{}", self.info.name);
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
        // Build param_names: buffer variable names
        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let name = match kind {
                BufferKind::Input => format!("buf_{}", i),
                BufferKind::Output => format!("buf_{}", i),
            };
            self.param_names.push(name);
        }

        // Build inst_names: temporary variable names for each instruction
        let total_insts: usize = self.func.blocks.iter().map(|b| b.instructions.len()).sum();
        for i in 0..total_insts {
            self.inst_names.push(format!("v{}", i));
        }

        // 1. Unpack buffers into typed pointers
        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let qualifier = match kind {
                BufferKind::Input => "const ",
                BufferKind::Output => "",
            };
            self.writeln(&format!(
                "{}float* {} = ({}float*)buffers[{}];",
                qualifier, self.param_names[i], qualifier, i
            ));
        }
        self.writeln("");

        // 2. Unpack shapes
        self.writeln("int64_t n = shapes[0];");
        if self.info.num_shapes > 1 {
            self.writeln("int64_t m = shapes[1];");
        }
        if self.info.num_shapes > 2 {
            self.writeln("int64_t k = shapes[2];");
        }
        self.writeln("");

        // 3. Emit OpenMP parallel for over thread groups
        self.writeln("#pragma omp parallel for schedule(static)");
        self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
        self.indent += 1;

        // 4. Inner loop over threads per group
        self.writeln("int64_t base = tg * threads_per_group;");
        self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
        self.indent += 1;

        self.writeln("int64_t i = base + t;");
        self.writeln("if (i < n) {");
        self.indent += 1;

        // 5. Emit body block instructions
        let mut inst_offset = 0;
        for block in &self.func.blocks {
            if block.label == "body" {
                self.emit_block(block, inst_offset);
            }
            inst_offset += block.instructions.len();
        }

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
        for (idx, inst) in block.instructions.iter().enumerate() {
            let global_idx = base_offset + idx;
            let c_code = emit_instruction(inst, &self.param_names, &self.inst_names, global_idx);
            if !c_code.is_empty() {
                self.writeln(&c_code);
            }
        }
    }

    fn writeln(&mut self, line: &str) {
        let indent_str = "  ".repeat(self.indent);
        self.out.push_str(&format!("{}{}\n", indent_str, line));
    }
}

fn emit_instruction(
    inst: &Instruction,
    param_names: &[String],
    inst_names: &[String],
    inst_idx: usize,
) -> String {
    let result_name = if inst_idx < inst_names.len() {
        inst_names[inst_idx].clone()
    } else {
        format!("v{}", inst_idx)
    };

    match inst {
        Instruction::Alloca { .. } => String::new(),
        Instruction::Load { ptr, .. } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            format!("float {} = {}[i];", result_name, ptr_name)
        }
        Instruction::Store { ptr, value, .. } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            let val_name = resolve_value_name(value, param_names, inst_names);
            format!("{}[i] = {};", ptr_name, val_name)
        }
        Instruction::Gep { ptr, indices } => {
            let ptr_name = resolve_value_name(ptr, param_names, inst_names);
            let index_exprs: Vec<String> = indices
                .iter()
                .map(|v| resolve_value_name(v, param_names, inst_names))
                .collect();
            if index_exprs.is_empty() {
                format!("float* {} = {} + i;", result_name, ptr_name)
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
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} + {};", result_name, l, r)
        }
        Instruction::Sub(lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} - {};", result_name, l, r)
        }
        Instruction::Mul(lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} * {};", result_name, l, r)
        }
        Instruction::Div(lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = {} / {};", result_name, l, r)
        }
        Instruction::FMax(lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = fmaxf({}, {});", result_name, l, r)
        }
        Instruction::FMin(lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            format!("float {} = fminf({}, {});", result_name, l, r)
        }
        Instruction::Exp(val) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("float {} = expf({});", result_name, v)
        }
        Instruction::FNeg(val) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("float {} = -{};", result_name, v)
        }
        Instruction::Icmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            let c_op = cmp_op_to_c(op);
            format!("int {} = {} {} {};", result_name, l, c_op, r)
        }
        Instruction::Fcmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, param_names, inst_names);
            let r = resolve_value_name(rhs, param_names, inst_names);
            let c_op = cmp_op_to_c(op);
            format!("int {} = {} {} {};", result_name, l, c_op, r)
        }
        Instruction::Call { func, args, .. } => {
            let args_str: Vec<String> = args
                .iter()
                .map(|v| resolve_value_name(v, param_names, inst_names))
                .collect();
            format!("float {} = {}({});", result_name, func, args_str.join(", "))
        }
        Instruction::Trunc(val, _ty) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("int {} = (int){};", result_name, v)
        }
        Instruction::ZExt(val, _ty) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("int64_t {} = (int64_t){};", result_name, v)
        }
        Instruction::SIToFP(val, _ty) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("float {} = (float){};", result_name, v)
        }
        Instruction::FPToSI(val, _ty) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("int {} = (int){};", result_name, v)
        }
        Instruction::BitCast(val, _ty) => {
            let v = resolve_value_name(val, param_names, inst_names);
            format!("float {} = {};", result_name, v)
        }
    }
}

fn resolve_value_name(value: &Value, param_names: &[String], inst_names: &[String]) -> String {
    match value {
        Value::Param(i) => {
            if *i < param_names.len() {
                param_names[*i].clone()
            } else {
                format!("buf_{}", i)
            }
        }
        Value::Const(c) => match c {
            Constant::Int(v) => format!("{}", v),
            Constant::Float(v) => format!("{}", v),
            Constant::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
        },
        Value::Inst(i) => {
            if *i < inst_names.len() {
                inst_names[*i].clone()
            } else {
                format!("v{}", i)
            }
        }
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
