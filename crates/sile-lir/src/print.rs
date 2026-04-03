use std::fmt::Write;

use sile_hir::ParamKind;

use crate::{ExecutableKernel, ParamPassing, ValueInfo};

pub fn format_executable_kernel(kernel: &ExecutableKernel) -> String {
    let mut out = String::new();
    writeln!(&mut out, "kernel @{}", kernel.name).unwrap();
    writeln!(&mut out, "abi:").unwrap();

    for param in &kernel.abi.params {
        let kind = match param.kind {
            ParamKind::Input => "input",
            ParamKind::Output => "output",
        };
        let passing = match param.passing {
            ParamPassing::Buffer => "buffer",
        };
        let offset = kernel.abi.shape_layout.offsets[param.index];
        writeln!(
            &mut out,
            "  arg{} {} {}<f32, rank={}> shape_offset={}",
            param.index, kind, passing, param.rank, offset
        )
        .unwrap();
    }

    writeln!(
        &mut out,
        "  total_dims={}",
        kernel.abi.shape_layout.total_dims
    )
    .unwrap();
    writeln!(
        &mut out,
        "  launch program_id_dims={}",
        kernel.abi.launch.program_id_dims
    )
    .unwrap();
    writeln!(&mut out, "func:").unwrap();

    let mut inst_idx = 0usize;
    for block in &kernel.func.blocks {
        writeln!(&mut out, "{}:", block.label).unwrap();
        for inst in &block.instructions {
            let info = kernel
                .value_info
                .instructions
                .get(inst_idx)
                .map(format_value_info)
                .unwrap_or_else(|| "void".to_string());
            writeln!(&mut out, "  %{}:{} = {:?}", inst_idx, info, inst).unwrap();
            inst_idx += 1;
        }
        writeln!(&mut out, "  {:?}", block.terminator).unwrap();
    }

    out
}

fn format_value_info(info: &ValueInfo) -> String {
    match info {
        ValueInfo::Buffer { rank, .. } => format!("buffer<f32, rank={rank}>"),
        ValueInfo::Scalar { .. } => "scalar<f32>".to_string(),
        ValueInfo::Index => "index".to_string(),
        ValueInfo::Shape => "shape".to_string(),
        ValueInfo::Tile { rows, cols, .. } => format!("tile<f32, {rows}x{cols}>"),
        ValueInfo::Void => "void".to_string(),
    }
}
