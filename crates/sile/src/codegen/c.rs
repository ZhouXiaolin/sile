use crate::backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp, ReduceKind};

pub fn generate(kernel: &BackendKernel) -> crate::Result<String> {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <math.h>\n\n");

    // 函数签名
    let fn_name = match kernel.op {
        BackendOp::VecAdd1D => "sile_kernel_vec_add",
        BackendOp::Softmax2D => "sile_kernel_softmax",
    };

    if kernel.tile_rank == 1 {
        let sym = &kernel.tile_shape_symbols[0];
        out.push_str(&format!(
            "void {}(float* a, float* b, float* c, int64_t pid, int64_t {}) {{\n",
            fn_name, sym
        ));
        out.push_str(&format!("  int64_t base = pid * {};\n", sym));
        out.push_str(&format!("  for (int64_t i = 0; i < {}; ++i) {{\n", sym));

        for inst in &kernel.instructions {
            out.push_str(&generate_1d_instruction(inst));
        }

        out.push_str("  }\n");
        out.push_str("}\n");
    } else {
        // 2D softmax
        out.push_str(&format!(
            "void {}(const float* x, float* y, int64_t pid_m, int64_t bm, int64_t bn, int64_t n) {{\n",
            fn_name
        ));
        out.push_str("  int64_t row_base = pid_m * bm;\n");
        out.push_str("  for (int64_t row = 0; row < bm; ++row) {\n");

        for inst in &kernel.instructions {
            out.push_str(&generate_2d_instruction(inst));
        }

        out.push_str("  }\n");
        out.push_str("}\n");
    }

    Ok(out)
}

fn generate_1d_instruction(inst: &BackendInstruction) -> String {
    match inst {
        BackendInstruction::Compute { op, args, .. } => match op.as_str() {
            "add" if args.len() >= 2 => {
                format!("    c[base + i] = a[base + i] + b[base + i];\n")
            }
            _ => String::new(),
        },
        _ => String::new(),
    }
}

fn generate_2d_instruction(inst: &BackendInstruction) -> String {
    match inst {
        BackendInstruction::Reduce { kind, axis, .. } => {
            let reduce_op = match kind {
                ReduceKind::Max => {
                    "  float max_value = x[(row_base + row) * n];\n  for (int64_t col = 1; col < bn; ++col) {\n    float value = x[(row_base + row) * n + col];\n    if (value > max_value) max_value = value;\n  }\n"
                }
                ReduceKind::Sum => {
                    "  float sum = 0.0f;\n  for (int64_t col = 0; col < bn; ++col) {\n    float e = expf(x[(row_base + row) * n + col] - max_value);\n    y[(row_base + row) * n + col] = e;\n    sum += e;\n  }\n"
                }
            };
            reduce_op.to_string()
        }
        BackendInstruction::Store { .. } => {
            "  for (int64_t col = 0; col < bn; ++col) {\n    y[(row_base + row) * n + col] /= sum;\n  }\n".to_string()
        }
        _ => String::new(),
    }
}
