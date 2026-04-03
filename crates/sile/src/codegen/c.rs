use crate::backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp, ReduceKind};

pub fn generate(kernel: &BackendKernel) -> crate::Result<String> {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <math.h>\n#include <stdlib.h>\n\n");

    let fn_name = match kernel.op {
        BackendOp::VecAdd1D => "sile_kernel_vec_add",
        BackendOp::Softmax2D => "sile_kernel_softmax",
        BackendOp::MatMul2D => "sile_kernel_matmul",
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
    } else if kernel.tile_rank == 2 {
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
    } else {
        // 3D MatMul2D - generate parameterized matmul kernel
        out.push_str(&format!(
            "void {}(const float* a, const float* b, float* c, int64_t pid_m, int64_t pid_n, int64_t m, int64_t n, int64_t k, int64_t bm, int64_t bn, int64_t bk) {{\n",
            fn_name
        ));
        out.push_str("  int64_t row = pid_m * bm;\n");
        out.push_str("  int64_t col = pid_n * bn;\n");
        out.push_str("  float* acc = (float*)calloc(bm * bn, sizeof(float));\n");
        out.push_str("  for (int64_t kk = 0; kk < k; kk += bk) {\n");
        out.push_str("    for (int64_t i = 0; i < bm; ++i) {\n");
        out.push_str("      for (int64_t j = 0; j < bn; ++j) {\n");
        out.push_str("        for (int64_t l = 0; l < bk; ++l) {\n");
        out.push_str(
            "          acc[i * bn + j] += a[(row + i) * k + kk + l] * b[(kk + l) * n + col + j];\n",
        );
        out.push_str("        }\n");
        out.push_str("      }\n");
        out.push_str("    }\n");
        out.push_str("  }\n");
        out.push_str("  for (int64_t i = 0; i < bm; ++i) {\n");
        out.push_str("    for (int64_t j = 0; j < bn; ++j) {\n");
        out.push_str("      c[(row + i) * n + col + j] = acc[i * bn + j];\n");
        out.push_str("    }\n");
        out.push_str("  }\n");
        out.push_str("  free(acc);\n");
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
        BackendInstruction::Reduce { kind, .. } => {
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
