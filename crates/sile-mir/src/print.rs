use std::fmt::Write;

use crate::ir::*;

pub fn format_mir(func: &MirFunction) -> String {
    let mut out = String::new();
    writeln!(&mut out, "func @{}(", func.name).unwrap();
    for (i, param) in func.params.iter().enumerate() {
        let comma = if i + 1 < func.params.len() { "," } else { "" };
        writeln!(
            &mut out,
            "  {} {}: {}{}",
            param.value, param.name, param.ty, comma
        )
        .unwrap();
    }
    writeln!(&mut out, ") {{").unwrap();

    for block in &func.blocks {
        let entry_marker = if block.id == func.entry {
            " // entry"
        } else {
            ""
        };
        if block.params.is_empty() {
            writeln!(&mut out, "  {}:{}", block.id, entry_marker).unwrap();
        } else {
            let params: Vec<String> = block
                .params
                .iter()
                .map(|v| {
                    let ty = func
                        .types
                        .get(v)
                        .map(|t| format!("{t}"))
                        .unwrap_or("?".into());
                    format!("{v}: {ty}")
                })
                .collect();
            writeln!(
                &mut out,
                "  {}({}):{}",
                block.id,
                params.join(", "),
                entry_marker
            )
            .unwrap();
        }

        for inst in &block.insts {
            let ty = func
                .types
                .get(&inst.result)
                .map(|t| format!("{t}"))
                .unwrap_or("?".into());
            writeln!(
                &mut out,
                "    {}:{} = {}",
                inst.result,
                ty,
                format_op(&inst.op)
            )
            .unwrap();
        }

        writeln!(&mut out, "    {}", format_terminator(&block.terminator)).unwrap();
    }

    writeln!(&mut out, "}}").unwrap();
    out
}

fn format_op(op: &MirOp) -> String {
    match op {
        MirOp::TileLoad {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!("tile_load {buf}, [{row_coord}, {col_coord}], shape=[{rows}, {cols}], stride_dim={stride_shape_idx}")
        }
        MirOp::TileStore {
            buf,
            value,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!("tile_store {buf}, {value}, [{row_coord}, {col_coord}], shape=[{rows}, {cols}], stride_dim={stride_shape_idx}")
        }
        MirOp::TileConstant { value, rows, cols } => {
            format!("tile_constant {value}, shape=[{rows}, {cols}]")
        }
        MirOp::TileBinary {
            op,
            lhs,
            rhs,
            rows,
            cols,
        } => {
            format!("tile_{op} {lhs}, {rhs}, shape=[{rows}, {cols}]")
        }
        MirOp::TileUnary {
            op,
            operand,
            rows,
            cols,
        } => {
            format!("tile_{op:?} {operand}, shape=[{rows}, {cols}]")
        }
        MirOp::TileMma {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            format!("tile_mma {a}, {b}, {acc}, mnk=[{tile_m}, {tile_n}, {tile_k}]")
        }
        MirOp::TileReduce {
            op,
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!("tile_reduce_{op:?} {value}, axis={axis}, in_shape=[{in_rows}, {in_cols}]")
        }
        MirOp::TileBroadcast { value, rows, cols } => {
            format!("tile_broadcast {value}, shape=[{rows}, {cols}]")
        }
        MirOp::IBinary { op, lhs, rhs } => {
            format!("i_{op} {lhs}, {rhs}")
        }
        MirOp::ICmp { op, lhs, rhs } => {
            format!("icmp_{op} {lhs}, {rhs}")
        }
        MirOp::ConstI64(v) => format!("const_i64 {v}"),
        MirOp::ConstF64(v) => format!("const_f64 {v}"),
        MirOp::ProgramId { dim } => format!("program_id dim={dim}"),
        MirOp::ShapeDim { buf, dim } => format!("shape_dim {buf}, dim={dim}"),
    }
}

fn format_terminator(term: &MirTerminator) -> String {
    match term {
        MirTerminator::Jump { target, args } => {
            if args.is_empty() {
                format!("jump {target}")
            } else {
                let args: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
                format!("jump {target}({})", args.join(", "))
            }
        }
        MirTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            let t_args = if true_args.is_empty() {
                format!("{true_target}")
            } else {
                let a: Vec<String> = true_args.iter().map(|v| format!("{v}")).collect();
                format!("{true_target}({})", a.join(", "))
            };
            let f_args = if false_args.is_empty() {
                format!("{false_target}")
            } else {
                let a: Vec<String> = false_args.iter().map(|v| format!("{v}")).collect();
                format!("{false_target}({})", a.join(", "))
            };
            format!("branch {cond}, {t_args}, {f_args}")
        }
        MirTerminator::Return => "return".to_string(),
    }
}
