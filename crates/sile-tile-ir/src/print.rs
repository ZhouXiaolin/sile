use std::fmt::Write;

use crate::ir::*;

pub fn format_tile_ir(func: &TileIrFunction) -> String {
    let mut out = String::new();
    writeln!(&mut out, "module {{").unwrap();
    writeln!(&mut out, "  entry @{}(", func.name).unwrap();
    for (i, param) in func.params.iter().enumerate() {
        let comma = if i + 1 < func.params.len() { "," } else { "" };
        writeln!(
            &mut out,
            "    {} {{{}}} : {}{}",
            param.value,
            format_param_attrs(param),
            format_tile_ir_type(&param.ty),
            comma
        )
        .unwrap();
    }
    writeln!(&mut out, "  ) {{").unwrap();

    for block in &func.blocks {
        let entry_marker = if block.id == func.entry {
            "  // entry"
        } else {
            ""
        };
        if block.params.is_empty() {
            writeln!(&mut out, "    ^{}:{}", block.id, entry_marker).unwrap();
        } else {
            let params = block
                .params
                .iter()
                .map(|v| {
                    let ty = func
                        .types
                        .get(v)
                        .map(format_tile_ir_type)
                        .unwrap_or("?".into());
                    format!("{v}: {ty}")
                })
                .collect::<Vec<_>>();
            writeln!(
                &mut out,
                "    ^{}({}):{}",
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
                .map(format_tile_ir_type)
                .unwrap_or("?".into());
            writeln!(
                &mut out,
                "      {} = {} : {}",
                inst.result,
                format_tile_ir_op(&inst.op),
                ty
            )
            .unwrap();
        }

        writeln!(
            &mut out,
            "      {}",
            format_tile_ir_terminator(&block.terminator)
        )
        .unwrap();
    }

    writeln!(&mut out, "  }}").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
}

fn format_tile_ir_op(op: &TileIrOp) -> String {
    match op {
        TileIrOp::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!(
                "load_ptr_tko {buf}, [{row_coord}, {col_coord}] {{shape = [{rows}, {cols}], stride_dim = {stride_shape_idx}}}"
            )
        }
        TileIrOp::StorePtrTko {
            buf,
            value,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!(
                "store_ptr_tko {buf}, {value}, [{row_coord}, {col_coord}] {{shape = [{rows}, {cols}], stride_dim = {stride_shape_idx}}}"
            )
        }
        TileIrOp::SileAtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            stride_shape_idx,
        } => {
            format!(
                "sile.atomic_add {buf}, {value}, [{row_coord}, {col_coord}] {{stride_dim = {stride_shape_idx}}}"
            )
        }
        TileIrOp::Splat { value, .. } => {
            format!("constant <f32: {}>", format_float_literal(*value))
        }
        TileIrOp::AddF { lhs, rhs, .. } => format!("addf {lhs}, {rhs}"),
        TileIrOp::SubF { lhs, rhs, .. } => format!("subf {lhs}, {rhs}"),
        TileIrOp::MulF { lhs, rhs, .. } => format!("mulf {lhs}, {rhs}"),
        TileIrOp::DivF { lhs, rhs, .. } => format!("divf {lhs}, {rhs}"),
        TileIrOp::NegF { operand, .. } => format!("negf {operand}"),
        TileIrOp::Exp { operand, .. } => format!("exp {operand}"),
        TileIrOp::MmaF {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            format!("mmaf {a}, {b}, {acc} {{m = {tile_m}, n = {tile_n}, k = {tile_k}}}")
        }
        TileIrOp::ReduceSum {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{axis = {axis}, combiner = \"sum\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::ReduceMax {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{axis = {axis}, combiner = \"max\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::Broadcast { value, rows, cols } => {
            format!("broadcast {value} {{shape = [{rows}, {cols}]}}")
        }
        TileIrOp::Map { expr, rows, cols } => {
            format!(
                "map {} {{shape = [{rows}, {cols}]}}",
                format_tile_map_expr(expr)
            )
        }
        TileIrOp::Extract {
            tile,
            row_coord,
            col_coord,
        } => {
            format!("extract {tile}[{row_coord}, {col_coord}]")
        }
        TileIrOp::IBinary { op, lhs, rhs } => {
            format!("{} {lhs}, {rhs}", format_tile_binary_op(*op, false))
        }
        TileIrOp::ICmp { op, lhs, rhs } => {
            format!("cmpi \"{op}\" {lhs}, {rhs}")
        }
        TileIrOp::ConstI64(v) => format!("constant <i64: {v}>"),
        TileIrOp::ConstF64(v) => format!("constant <f64: {}>", format_float_literal(*v)),
        TileIrOp::ShapeDim { shape, dim } => format!("sile.shape_dim {shape} {{dim = {dim}}}"),
    }
}

fn format_tile_map_expr(expr: &TileMapExpr) -> String {
    match expr {
        TileMapExpr::Value(value) => value.to_string(),
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            rows,
            cols,
            stride_shape_idx,
        } => format!(
            "load_ptr_tko({buf}, [{row_coord}, {col_coord}], shape=[{rows}, {cols}], stride_dim={stride_shape_idx})"
        ),
        TileMapExpr::Splat { value } => format!("constant({})", format_float_literal(*value)),
        TileMapExpr::Add { lhs, rhs } => {
            format!(
                "addf({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Sub { lhs, rhs } => {
            format!(
                "subf({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Mul { lhs, rhs } => {
            format!(
                "mulf({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Div { lhs, rhs } => {
            format!(
                "divf({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Neg { operand } => format!("negf({})", format_tile_map_expr(operand)),
        TileMapExpr::Exp { operand } => format!("exp({})", format_tile_map_expr(operand)),
        TileMapExpr::Broadcast {
            value,
            src_rows,
            src_cols,
        } => format!(
            "broadcast({}, src_shape=[{src_rows}, {src_cols}])",
            format_tile_map_expr(value)
        ),
    }
}

fn format_tile_ir_terminator(term: &TileIrTerminator) -> String {
    match term {
        TileIrTerminator::Jump { target, args } => {
            if args.is_empty() {
                format!("cf.br ^{target}")
            } else {
                let args = args.iter().map(ToString::to_string).collect::<Vec<_>>();
                format!("cf.br ^{target}({})", args.join(", "))
            }
        }
        TileIrTerminator::Branch {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            let true_args = true_args
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let false_args = false_args
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("cf.cond_br {cond}, ^{true_target}({true_args}), ^{false_target}({false_args})")
        }
        TileIrTerminator::Return => "return".to_string(),
    }
}

fn format_tile_ir_type(ty: &TileIrType) -> String {
    match ty {
        TileIrType::I64 => "tile<i64>".into(),
        TileIrType::F32 => "tile<f32>".into(),
        TileIrType::ShapeDesc { rank } => format!("shape_desc<{rank}>"),
        TileIrType::Tile { rows, cols } => format!("tile<{rows}x{cols}xf32>"),
        TileIrType::Buffer { .. } => "tile<ptr<f32>>".into(),
        TileIrType::Void => "none".into(),
    }
}

fn format_tile_binary_op(op: BinOp, is_float: bool) -> &'static str {
    match (op, is_float) {
        (BinOp::Add, true) => "addf",
        (BinOp::Sub, true) => "subf",
        (BinOp::Mul, true) => "mulf",
        (BinOp::Div, true) => "divf",
        (BinOp::Add, false) => "addi",
        (BinOp::Sub, false) => "subi",
        (BinOp::Mul, false) => "muli",
        (BinOp::Div, false) => "divi",
    }
}

fn buffer_rank_attr(ty: &TileIrType) -> usize {
    match ty {
        TileIrType::Buffer { rank } => *rank,
        _ => 0,
    }
}

fn format_param_attrs(param: &TileIrParam) -> String {
    match &param.kind {
        TileIrParamKind::Buffer => {
            format!(
                "sym_name = \"{}\", rank = {}",
                param.name,
                buffer_rank_attr(&param.ty)
            )
        }
        TileIrParamKind::LaunchIndex { dim } => {
            format!(
                "sym_name = \"{}\", kind = \"launch.index\", dim = {}",
                param.name, dim
            )
        }
        TileIrParamKind::ShapeDesc { source } => {
            format!(
                "sym_name = \"{}\", kind = \"shape.desc\", source = {}",
                param.name, source
            )
        }
    }
}

fn format_float_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}
