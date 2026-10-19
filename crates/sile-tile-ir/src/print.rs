use std::fmt::Write;

use crate::ir::*;

/// The name a unary float operation prints under. The name is the operation:
/// `exp`, `log2`, and the rest are the spellings every target uses.
fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "negf",
        UnaryOp::Exp => "exp",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Log => "log",
        UnaryOp::Log2 => "log2",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Rsqrt => "rsqrt",
        UnaryOp::Abs => "absf",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Sinh => "sinh",
        UnaryOp::Cosh => "cosh",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Floor => "floor",
    }
}

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
            let op_text = format_tile_ir_op(&inst.op);
            // An mmaf names its operand types itself, in the CuTile order;
            // the accumulator type among them is the result.
            if matches!(inst.op, TileIrOp::MmaF { .. }) {
                writeln!(&mut out, "      {} = {}", inst.result, op_text).unwrap();
                continue;
            }
            // A select names the condition's type next to the result's, the
            // way the CuTile printer spells it.
            if let TileIrOp::Select { cond, .. } = &inst.op {
                let cond_ty = func
                    .types
                    .get(cond)
                    .map(format_tile_ir_type)
                    .unwrap_or("?".into());
                let ty = func
                    .types
                    .get(&inst.result)
                    .map(format_tile_ir_type)
                    .unwrap_or("?".into());
                writeln!(&mut out, "      {} = {} : {}, {}", inst.result, op_text, cond_ty, ty)
                    .unwrap();
                continue;
            }
            let ty = func
                .types
                .get(&inst.result)
                .map(format_tile_ir_type)
                .unwrap_or("?".into());
            writeln!(&mut out, "      {} = {} : {}", inst.result, op_text, ty).unwrap();
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

/// The coordinates an access names, as a list.
///
/// The outer coordinates come first and the tile's own two come last, which is
/// the order a kernel writes them in.
fn format_coords(outer_coords: &[ValueId], row_coord: &ValueId, col_coord: &ValueId) -> String {
    let mut parts: Vec<String> = outer_coords.iter().map(|coord| coord.to_string()).collect();
    parts.push(row_coord.to_string());
    parts.push(col_coord.to_string());
    format!("[{}]", parts.join(", "))
}

fn format_tile_ir_op(op: &TileIrOp) -> String {
    match op {
        TileIrOp::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!(
                "load_ptr_tko {buf}, {} {{shape = [{rows}, {cols}], stride_dim = {stride_shape_idx}}}",
                format_coords(outer_coords, row_coord, col_coord)
            )
        }
        TileIrOp::StorePtrTko {
            buf,
            value,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => {
            format!(
                "store_ptr_tko {buf}, {value}, {} {{shape = [{rows}, {cols}], stride_dim = {stride_shape_idx}}}",
                format_coords(outer_coords, row_coord, col_coord)
            )
        }
        TileIrOp::SileAtomicAdd {
            buf,
            value,
            row_coord,
            col_coord,
            outer_coords,
            stride_shape_idx,
        } => {
            format!(
                "sile.atomic_add {buf}, {value}, {} {{stride_dim = {stride_shape_idx}}}",
                format_coords(outer_coords, row_coord, col_coord)
            )
        }
        TileIrOp::Splat { value, .. } => {
            format!("constant <f32: {}>", format_float_literal(*value))
        }
        TileIrOp::AddF { lhs, rhs, .. } => format!("addf {lhs}, {rhs}"),
        TileIrOp::SubF { lhs, rhs, .. } => format!("subf {lhs}, {rhs}"),
        TileIrOp::MulF { lhs, rhs, .. } => format!("mulf {lhs}, {rhs}"),
        TileIrOp::DivF { lhs, rhs, .. } => format!("divf {lhs}, {rhs}"),
        // The spellings the CuTile printer uses: `fma $lhs, $rhs, $acc`,
        // `select $cond, $val_if_true, $val_if_false`, `maxf`, `minf`.
        TileIrOp::Fma {
            lhs,
            rhs,
            acc,
            ..
        } => format!("fma {lhs}, {rhs}, {acc}"),
        TileIrOp::Select {
            cond,
            on_true,
            on_false,
            ..
        } => format!("select {cond}, {on_true}, {on_false}"),
        TileIrOp::MaxF { lhs, rhs, .. } => format!("maxf {lhs}, {rhs}"),
        TileIrOp::MinF { lhs, rhs, .. } => format!("minf {lhs}, {rhs}"),
        TileIrOp::UnaryF { op, operand, .. } => format!("{} {operand}", unary_name(*op)),
        TileIrOp::MmaF {
            a,
            b,
            acc,
            tile_m,
            tile_n,
            tile_k,
        } => {
            // The shapes ride on the operand types, the way the CuTile
            // printer spells them: there is no m/n/k attribute to keep in
            // step with the types.
            format!(
                "mmaf {a}, {b}, {acc} : tile<{tile_m}x{tile_k}xf32>, tile<{tile_k}x{tile_n}xf32>, tile<{tile_m}x{tile_n}xf32>"
            )
        }
        TileIrOp::ReduceSum {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{dim = {axis}, combiner = \"sum\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::ReduceMax {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{dim = {axis}, combiner = \"max\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::ReduceMin {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{dim = {axis}, combiner = \"min\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::ReduceProd {
            value,
            axis,
            in_rows,
            in_cols,
        } => {
            format!(
                "reduce {value} {{dim = {axis}, combiner = \"prod\", input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::Scan {
            value,
            axis,
            in_rows,
            in_cols,
            reverse,
            combiner,
            ..
        } => {
            let combiner = match combiner {
                crate::ir::ReduceCombiner::Sum => "sum",
                crate::ir::ReduceCombiner::Prod => "prod",
                crate::ir::ReduceCombiner::Max => "max",
                crate::ir::ReduceCombiner::Min => "min",
            };
            format!(
                "scan {value} {{dim = {axis}, combiner = \"{combiner}\", reverse = {reverse}, input_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::Cat {
            lhs,
            rhs,
            axis,
            lhs_rows,
            lhs_cols,
            rhs_rows,
            rhs_cols,
        } => {
            format!(
                "cat {lhs}, {rhs} {{dim = {axis}, lhs_shape = [{lhs_rows}, {lhs_cols}], rhs_shape = [{rhs_rows}, {rhs_cols}]}}"
            )
        }
        TileIrOp::ExtractSlice {
            source,
            row_idx,
            col_idx,
            in_rows,
            in_cols,
            out_rows,
            out_cols,
        } => {
            format!(
                "extract {source} {{index = [{row_idx}, {col_idx}], input_shape = [{in_rows}, {in_cols}], result_shape = [{out_rows}, {out_cols}]}}"
            )
        }
        TileIrOp::InsertSlice {
            source,
            destination,
            row_idx,
            col_idx,
            in_rows,
            in_cols,
            ..
        } => {
            format!(
                "insert {source} into {destination} {{index = [{row_idx}, {col_idx}], subtile_shape = [{in_rows}, {in_cols}]}}"
            )
        }
        TileIrOp::Broadcast { value, rows, cols } => {
            format!("broadcast {value} {{shape = [{rows}, {cols}]}}")
        }
        TileIrOp::Reshape { value, rows, cols } => {
            format!("reshape {value} {{shape = [{rows}, {cols}]}}")
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
        TileIrOp::FBinary { op, lhs, rhs } => {
            format!("{} {lhs}, {rhs}", format_tile_binary_op(*op, true))
        }
        TileIrOp::ICmp { op, lhs, rhs } => {
            format!("cmpi {} {lhs}, {rhs}, signed", comparison_predicate_name(*op))
        }
        TileIrOp::ConstI64(v) => format!("constant <i64: {v}>"),
        // The value is held wide, but the tile type system only carries f32
        // scalars: the element spelling names the type the result has.
        TileIrOp::ConstF64(v) => format!("constant <f32: {}>", format_float_literal(*v)),
        TileIrOp::ShapeDim { shape, dim } => format!("sile.shape_dim {shape} {{dim = {dim}}}"),
        TileIrOp::GridDim { dim } => format!("sile.grid_dim {{dim = {dim}}}"),
        TileIrOp::NumTiles {
            shape,
            dim,
            tile_extent,
        } => format!("sile.num_tiles {shape} {{dim = {dim}, tile_extent = {tile_extent}}}"),
    }
}

fn format_tile_map_expr(expr: &TileMapExpr) -> String {
    match expr {
        TileMapExpr::Value(value) => value.to_string(),
        TileMapExpr::Iota => "iota".to_string(),
        TileMapExpr::LoadPtrTko {
            buf,
            row_coord,
            col_coord,
            outer_coords,
            rows,
            cols,
            stride_shape_idx,
        } => format!(
            "load_ptr_tko({buf}, {}, shape=[{rows}, {cols}], stride_dim={stride_shape_idx})",
            format_coords(outer_coords, row_coord, col_coord)
        ),
        TileMapExpr::Splat { value } => format!("constant({})", format_float_literal(*value)),
        TileMapExpr::Cmp { op, lhs, rhs } => {
            format!(
                "cmpf(\"{op}\", {}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
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
        TileMapExpr::Rem { lhs, rhs } => format!(
            "rem({}, {})",
            format_tile_map_expr(lhs),
            format_tile_map_expr(rhs)
        ),
        TileMapExpr::Transpose { value } => {
            format!("transpose({})", format_tile_map_expr(value))
        }
        TileMapExpr::Unary { op, operand } => {
            format!("{}({})", unary_name(*op), format_tile_map_expr(operand))
        }
        TileMapExpr::Pow { lhs, rhs } => {
            format!(
                "powf({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Atan2 { lhs, rhs } => {
            format!(
                "atan2({}, {})",
                format_tile_map_expr(lhs),
                format_tile_map_expr(rhs)
            )
        }
        TileMapExpr::Fma { a, b, c } => format!(
            "fma({}, {}, {})",
            format_tile_map_expr(a),
            format_tile_map_expr(b),
            format_tile_map_expr(c)
        ),
        TileMapExpr::Select {
            cond,
            on_true,
            on_false,
        } => format!(
            "select({}, {}, {})",
            format_tile_map_expr(cond),
            format_tile_map_expr(on_true),
            format_tile_map_expr(on_false)
        ),
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
        TileIrType::Buffer { elem, .. } => {
            let elem = match elem {
                BufferElem::F32 => "f32",
                BufferElem::F16 => "f16",
            };
            format!("tile<ptr<{elem}>>")
        }
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
        (BinOp::Rem, true) => "remf",
        (BinOp::Rem, false) => "remi",
        (BinOp::And, _) => "and",
        (BinOp::Or, _) => "or",
        (BinOp::Xor, _) => "xori",
        (BinOp::Shl, _) => "shli",
        (BinOp::Shr, _) => "shri",
    }
}

fn buffer_rank_attr(ty: &TileIrType) -> usize {
    match ty {
        TileIrType::Buffer { rank, .. } => *rank,
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
        TileIrParamKind::Scalar { index } => {
            format!(
                "sym_name = \"{}\", kind = \"scalar\", index = {}",
                param.name, index
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

/// The predicate spelling the CuTile printer uses, per its name table.
fn comparison_predicate_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "less_than",
        CmpOp::Le => "less_than_or_equal",
        CmpOp::Gt => "greater_than",
        CmpOp::Ge => "greater_than_or_equal",
        CmpOp::Eq => "equal",
        CmpOp::Ne => "not_equal",
    }
}
