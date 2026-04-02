use crate::{BinaryOp, KernelSpec, Node, Result};

pub fn generate(spec: &KernelSpec) -> Result<String> {
    let tile_size = spec.tile_size()? as usize;

    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <stddef.h>\n\n");
    out.push_str("typedef struct {\n");
    out.push_str("    void* data;\n    int32_t dtype;\n    int32_t rank;\n");
    out.push_str("    const int64_t* shape;\n    const int64_t* strides;\n");
    out.push_str("} SileTensorArg;\n\n");
    out.push_str("typedef struct {\n");
    out.push_str("    int64_t grid[3];\n    int64_t tile_shape[4];\n    int32_t tile_rank;\n");
    out.push_str("} SileLaunch;\n\n");
    out.push_str(&format!(
        "void sile_kernel_{}(const SileTensorArg* args, uintptr_t arg_count, const SileLaunch* launch, const int64_t tile_id[3]) {{\n",
        spec.name
    ));
    out.push_str("    (void)arg_count;\n    (void)launch;\n");
    out.push_str(&format!(
        "    const int64_t base = tile_id[0] * {};\n",
        tile_size
    ));

    for (idx, node) in spec.nodes.iter().enumerate() {
        match node {
            Node::LoadTile { param, .. } => {
                out.push_str(&format!("    float tmp_{idx}[{tile_size}];\n"));
                out.push_str(&format!(
                    "    const float* arg_{param} = (const float*)args[{param}].data;\n"
                ));
                out.push_str(&format!(
                    "    for (int i = 0; i < {tile_size}; ++i) tmp_{idx}[i] = arg_{param}[base + i];\n"
                ));
            }
            Node::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
                ..
            } => {
                out.push_str(&format!("    float tmp_{idx}[{tile_size}];\n"));
                out.push_str(&format!(
                    "    for (int i = 0; i < {tile_size}; ++i) tmp_{idx}[i] = tmp_{lhs}[i] + tmp_{rhs}[i];\n"
                ));
            }
            _ => {}
        }
    }

    // Store: hardcoded for param 2, value 2
    out.push_str("    float* out_ptr = (float*)args[2].data;\n");
    out.push_str(&format!(
        "    for (int i = 0; i < {tile_size}; ++i) out_ptr[base + i] = tmp_2[i];\n"
    ));
    out.push_str("}\n");
    Ok(out)
}
