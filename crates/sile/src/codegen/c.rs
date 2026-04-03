use crate::lir::ir::*;
use crate::scheduling::{ParallelRegion, ReductionOp, ScheduleAnnotation, SimdRegion};

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

pub fn generate(
    func: &Function,
    annotations: &ScheduleAnnotation,
    info: &KernelGenInfo,
) -> crate::Result<String> {
    let mut ctx = CCodegen {
        func,
        annotations,
        info,
        value_names: Vec::new(),
        temp_counter: 0,
        indent: 0,
        out: String::new(),
    };

    ctx.emit_prologue();
    ctx.emit_wrapper_signature();
    ctx.emit_wrapper_body();
    ctx.emit_epilogue();

    Ok(ctx.out)
}

struct CCodegen<'a> {
    func: &'a Function,
    annotations: &'a ScheduleAnnotation,
    info: &'a KernelGenInfo,
    value_names: Vec<String>,
    temp_counter: usize,
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

    fn emit_epilogue(&mut self) {}

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
        self.emit_buffer_unpack();
        self.emit_shape_unpack();

        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let name = match kind {
                BufferKind::Input => format!("in_{}", i),
                BufferKind::Output => format!("out_{}", i),
            };
            self.value_names.push(name);
        }

        for region in &self.annotations.regions {
            self.emit_parallel_region(region);
        }

        for block in &self.func.blocks {
            let covered = self.annotations.regions.iter().any(|r| {
                let blocks = match r {
                    ParallelRegion::ParallelFor { body_blocks, .. }
                    | ParallelRegion::ParallelReduction { body_blocks, .. } => body_blocks,
                };
                blocks.contains(&block.label)
            });
            if !covered && block.label != "entry" {
                self.emit_block(block);
            }
        }

        self.indent = 0;
        self.out.push_str("}\n");
    }

    fn emit_buffer_unpack(&mut self) {
        for (i, kind) in self.info.buffer_kinds.iter().enumerate() {
            let name = match kind {
                BufferKind::Input => format!("in_{}", i),
                BufferKind::Output => format!("out_{}", i),
            };
            let qualifier = match kind {
                BufferKind::Input => "const ",
                BufferKind::Output => "",
            };
            self.writeln(&format!(
                "{}float* {} = ({}float*)buffers[{}];",
                qualifier, name, qualifier, i
            ));
        }
        self.writeln("");
    }

    fn emit_shape_unpack(&mut self) {
        self.writeln("int64_t n = shapes[0];");
        if self.info.num_shapes > 1 {
            self.writeln("int64_t m = shapes[1];");
        }
        if self.info.num_shapes > 2 {
            self.writeln("int64_t k = shapes[2];");
        }
        self.writeln("");
    }

    fn emit_parallel_region(&mut self, region: &ParallelRegion) {
        match region {
            ParallelRegion::ParallelFor {
                body_blocks,
                simd_regions,
                ..
            } => {
                self.writeln(
                    "#pragma omp parallel for num_threads(num_threadgroups) schedule(static)",
                );
                self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
                self.indent += 1;

                self.writeln("int64_t base = tg * threads_per_group;");

                if simd_regions.is_empty() {
                    self.writeln("#pragma omp simd");
                }
                self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
                self.indent += 1;
                self.writeln("int64_t i = base + t;");
                self.writeln("if (i < n) {");
                self.indent += 1;

                for label in body_blocks {
                    if let Some(block) = self.func.get_block(label) {
                        self.emit_block(block);
                    }
                }

                self.indent -= 1;
                self.writeln("}");
                self.indent -= 1;
                self.writeln("}");

                for simd in simd_regions {
                    self.emit_simd_region(simd);
                }

                self.indent -= 1;
                self.writeln("}");
            }
            ParallelRegion::ParallelReduction {
                reduction_op,
                body_blocks,
                ..
            } => {
                let op_str = match reduction_op {
                    ReductionOp::Max => "max",
                    ReductionOp::Sum => "+",
                    ReductionOp::Min => "min",
                    ReductionOp::Product => "*",
                };
                self.writeln(&format!(
                    "#pragma omp parallel for num_threads(num_threadgroups) schedule(static) reduction({}:acc)",
                    op_str
                ));
                self.writeln("for (int64_t tg = 0; tg < num_threadgroups; ++tg) {");
                self.indent += 1;
                self.writeln("int64_t base = tg * threads_per_group;");
                self.writeln("float acc = 0.0f;");

                for label in body_blocks {
                    if let Some(block) = self.func.get_block(label) {
                        self.emit_block(block);
                    }
                }

                self.indent -= 1;
                self.writeln("}");
            }
        }
    }

    fn emit_simd_region(&mut self, region: &SimdRegion) {
        self.writeln("#pragma omp simd");
        self.writeln("for (int64_t t = 0; t < threads_per_group; ++t) {");
        self.indent += 1;

        for label in &region.body_blocks {
            if let Some(block) = self.func.get_block(label) {
                self.emit_block(block);
            }
        }

        self.indent -= 1;
        self.writeln("}");
    }

    fn emit_block(&mut self, block: &BasicBlock) {
        for phi in &block.phi_nodes {
            let dest = &phi.dest;
            let src = self.value_name(&phi.incoming[0].0);
            self.writeln(&format!("{} = {};", dest, src));
        }

        for inst in &block.instructions {
            let c_code = emit_instruction(inst, &self.value_names);
            if !c_code.is_empty() {
                self.writeln(&c_code);
            }
        }
    }

    fn value_name(&mut self, value: &Value) -> String {
        match value {
            Value::Param(i) => {
                if *i < self.value_names.len() {
                    self.value_names[*i].clone()
                } else {
                    format!("param{}", i)
                }
            }
            Value::Const(c) => match c {
                Constant::Int(v) => format!("{}", v),
                Constant::Float(v) => format!("{:.1f}", v),
                Constant::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
            },
            Value::Inst(i) => {
                while self.value_names.len() <= *i {
                    let name = format!("t{}", self.temp_counter);
                    self.temp_counter += 1;
                    self.value_names.push(name);
                }
                self.value_names[*i].clone()
            }
        }
    }

    fn writeln(&mut self, line: &str) {
        let indent_str = "  ".repeat(self.indent);
        self.out.push_str(&format!("{}{}\n", indent_str, line));
    }
}

fn emit_instruction(inst: &Instruction, value_names: &[String]) -> String {
    match inst {
        Instruction::Alloca { ty } => {
            let c_ty = c_type(ty);
            let name = format!("local_{}", value_names.len());
            format!("{} {};", c_ty, name)
        }
        Instruction::Load { ptr, .. } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let name = format!("t{}", value_names.len());
            format!("float {} = {}[i];", name, ptr_name)
        }
        Instruction::Store { ptr, value, .. } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let val_name = resolve_value_name(value, value_names);
            format!("{}[i] = {};", ptr_name, val_name)
        }
        Instruction::Gep { ptr, indices } => {
            let ptr_name = resolve_value_name(ptr, value_names);
            let index_exprs: Vec<String> = indices
                .iter()
                .map(|v| resolve_value_name(v, value_names))
                .collect();
            let name = format!("t{}", value_names.len());
            if index_exprs.len() == 1 {
                format!("float* {} = {} + {};", name, ptr_name, index_exprs[0])
            } else {
                format!(
                    "float* {} = {} + {};",
                    name,
                    ptr_name,
                    index_exprs.join(" + ")
                )
            }
        }
        Instruction::Add(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} + {};", next_temp(value_names), l, r)
        }
        Instruction::Sub(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} - {};", next_temp(value_names), l, r)
        }
        Instruction::Mul(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} * {};", next_temp(value_names), l, r)
        }
        Instruction::Div(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = {} / {};", next_temp(value_names), l, r)
        }
        Instruction::FMax(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = fmaxf({}, {});", next_temp(value_names), l, r)
        }
        Instruction::FMin(lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            format!("{} = fminf({}, {});", next_temp(value_names), l, r)
        }
        Instruction::Exp(val) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = expf({});", next_temp(value_names), v)
        }
        Instruction::Icmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            let c_op = cmp_op_to_c(op);
            format!("{} = {} {} {};", next_temp(value_names), l, c_op, r)
        }
        Instruction::Fcmp(op, lhs, rhs) => {
            let l = resolve_value_name(lhs, value_names);
            let r = resolve_value_name(rhs, value_names);
            let c_op = cmp_op_to_c(op);
            format!("{} = {} {} {};", next_temp(value_names), l, c_op, r)
        }
        Instruction::Call { func, args, .. } => {
            let args_str: Vec<String> = args
                .iter()
                .map(|v| resolve_value_name(v, value_names))
                .collect();
            format!("{}({});", func, args_str.join(", "))
        }
        Instruction::Trunc(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (int){};", next_temp(value_names), v)
        }
        Instruction::ZExt(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (unsigned){};", next_temp(value_names), v)
        }
        Instruction::SIToFP(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (float){};", next_temp(value_names), v)
        }
        Instruction::FPToSI(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = (int){};", next_temp(value_names), v)
        }
        Instruction::BitCast(val, _ty) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = {};", next_temp(value_names), v)
        }
        Instruction::FNeg(val) => {
            let v = resolve_value_name(val, value_names);
            format!("{} = -{};", next_temp(value_names), v)
        }
    }
}

fn resolve_value_name(value: &Value, value_names: &[String]) -> String {
    match value {
        Value::Param(i) => {
            if *i < value_names.len() {
                value_names[*i].clone()
            } else {
                format!("param{}", i)
            }
        }
        Value::Const(c) => match c {
            Constant::Int(v) => format!("{}", v),
            Constant::Float(v) => format!("{:.1f}", v),
            Constant::Bool(v) => format!("{}", if *v { 1 } else { 0 }),
        },
        Value::Inst(i) => {
            if *i < value_names.len() {
                value_names[*i].clone()
            } else {
                format!("t{}", i)
            }
        }
    }
}

fn next_temp(value_names: &[String]) -> String {
    format!("t{}", value_names.len())
}

fn c_type(ty: &Type) -> String {
    match ty {
        Type::Void => "void".into(),
        Type::Int(it) => match it {
            IntegerType::I8 => "int8_t".into(),
            IntegerType::I16 => "int16_t".into(),
            IntegerType::I32 => "int32_t".into(),
            IntegerType::I64 => "int64_t".into(),
        },
        Type::Float(ft) => match ft {
            FloatType::F16 => "float".into(),
            FloatType::F32 => "float".into(),
            FloatType::F64 => "double".into(),
        },
        Type::Pointer(inner) => format!("{}*", c_type(inner)),
        Type::Vector(elem, len) => format!("{}[{}]", c_type(elem), len),
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
