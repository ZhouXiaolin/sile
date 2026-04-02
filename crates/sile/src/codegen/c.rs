use crate::backend_ir::ir::{BackendKernel, BackendOp};

pub fn generate(kernel: &BackendKernel) -> crate::Result<String> {
    match kernel.op {
        BackendOp::VecAdd1D => generate_vec_add(kernel),
        BackendOp::Softmax2D => generate_softmax(kernel),
    }
}

fn generate_vec_add(kernel: &BackendKernel) -> crate::Result<String> {
    let tile_size_symbol = &kernel.tile_shape_symbols[0];
    Ok(format!(
        "#include <stdint.h>\n#include <stddef.h>\n\n\
         void sile_kernel_vec_add(float* a, float* b, float* c, int64_t pid, int64_t {tile_size_symbol}) {{\n\
         \x20   int64_t base = pid * {tile_size_symbol};\n\
         \x20   for (int64_t i = 0; i < {tile_size_symbol}; ++i) {{\n\
         \x20       c[base + i] = a[base + i] + b[base + i];\n\
         \x20   }}\n\
         }}\n"
    ))
}

fn generate_softmax(_kernel: &BackendKernel) -> crate::Result<String> {
    Ok(
        "#include <math.h>\n#include <stdint.h>\n\n\
         void sile_kernel_softmax(const float* x, float* y, int64_t pid_m, int64_t bm, int64_t bn, int64_t n) {\n\
         \x20   int64_t row_base = pid_m * bm;\n\
         \x20   for (int64_t row = 0; row < bm; ++row) {\n\
         \x20       float max_value = x[(row_base + row) * n];\n\
         \x20       for (int64_t col = 1; col < bn; ++col) {\n\
         \x20           float value = x[(row_base + row) * n + col];\n\
         \x20           if (value > max_value) max_value = value;\n\
         \x20       }\n\
         \x20       float sum = 0.0f;\n\
         \x20       for (int64_t col = 0; col < bn; ++col) {\n\
         \x20           float e = expf(x[(row_base + row) * n + col] - max_value);\n\
         \x20           y[(row_base + row) * n + col] = e;\n\
         \x20           sum += e;\n\
         \x20       }\n\
         \x20       for (int64_t col = 0; col < bn; ++col) {\n\
         \x20           y[(row_base + row) * n + col] /= sum;\n\
         \x20       }\n\
         \x20   }\n\
         }\n"
            .to_string(),
    )
}
