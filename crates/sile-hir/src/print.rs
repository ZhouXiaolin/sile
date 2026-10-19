//! Text form of the kernel IR.
//!
//! The form is a readable Rust-shaped listing: one statement per line, nested
//! blocks indented, and every expression in prefix form. It exists to make the
//! HIR comparable across the pipeline and against the layers below it, not to
//! round-trip.

use std::fmt::Write;

use crate::kernel::{BuiltinOp, Expr, Kernel, ParamKind, Stmt, UnaryOp};
use crate::types::{ElemType, ShapeExpr, Type};

pub fn format_kernel(kernel: &Kernel) -> String {
    let mut out = String::new();
    writeln!(out, "hir kernel @{}", kernel.name).unwrap();
    if !kernel.origin.module.is_empty() {
        writeln!(
            out,
            "  origin module = {}, entry = {}, source bytes = {}",
            kernel.origin.module,
            kernel.origin.entry_index,
            kernel.origin.source.len()
        )
        .unwrap();
    }
    if !kernel.const_params.is_empty() {
        let consts = kernel
            .const_params
            .iter()
            .map(|(name, value)| format!("{name} = {value}"))
            .collect::<Vec<_>>();
        writeln!(out, "  consts {}", consts.join(", ")).unwrap();
    }
    for param in &kernel.params {
        let kind = match param.kind {
            ParamKind::Input => "in",
            ParamKind::Output => "out",
        };
        writeln!(
            out,
            "  param {kind} {}: {}",
            param.name,
            format_type(&param.ty)
        )
        .unwrap();
    }
    writeln!(out, "  body {{").unwrap();
    for stmt in &kernel.body {
        format_stmt(stmt, 2, &mut out);
    }
    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

fn format_stmt(stmt: &Stmt, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    match stmt {
        Stmt::Let { name, ty, expr } => match ty {
            Some(ty) => writeln!(
                out,
                "{pad}{name}: {} = {}",
                format_type(ty),
                format_expr(expr)
            ),
            None => writeln!(out, "{pad}{name} = {}", format_expr(expr)),
        }
        .unwrap(),
        Stmt::Assign { name, expr } => {
            writeln!(out, "{pad}{name} = {}", format_expr(expr)).unwrap();
        }
        Stmt::Store {
            target,
            value,
            index,
        } => match index {
            Some(index) => writeln!(
                out,
                "{pad}store {target}[{}] = {}",
                format_expr(index),
                format_expr(value)
            ),
            None => writeln!(out, "{pad}store {target} = {}", format_expr(value)),
        }
        .unwrap(),
        Stmt::AtomicAdd {
            target,
            index,
            value,
        } => {
            writeln!(
                out,
                "{pad}atomic_add {target}[{}], {}",
                format_expr(index),
                format_expr(value)
            )
            .unwrap();
        }
        Stmt::ForLoop {
            var,
            start,
            end,
            step,
            body,
        } => {
            match step {
                Some(step) => writeln!(
                    out,
                    "{pad}for {var} in {}..{} step {} {{",
                    format_expr(start),
                    format_expr(end),
                    format_expr(step)
                ),
                None => writeln!(
                    out,
                    "{pad}for {var} in {}..{} {{",
                    format_expr(start),
                    format_expr(end)
                ),
            }
            .unwrap();
            for inner in body {
                format_stmt(inner, depth + 1, out);
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        Stmt::While { cond, body } => {
            writeln!(out, "{pad}while {} {{", format_expr(cond)).unwrap();
            for inner in body {
                format_stmt(inner, depth + 1, out);
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        Stmt::Loop { body } => {
            writeln!(out, "{pad}loop {{").unwrap();
            for inner in body {
                format_stmt(inner, depth + 1, out);
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            writeln!(out, "{pad}if {} {{", format_expr(cond)).unwrap();
            for inner in then_body {
                format_stmt(inner, depth + 1, out);
            }
            if else_body.is_empty() {
                writeln!(out, "{pad}}}").unwrap();
            } else {
                writeln!(out, "{pad}}} else {{").unwrap();
                for inner in else_body {
                    format_stmt(inner, depth + 1, out);
                }
                writeln!(out, "{pad}}}").unwrap();
            }
        }
        Stmt::Break => writeln!(out, "{pad}break").unwrap(),
        Stmt::Continue => writeln!(out, "{pad}continue").unwrap(),
        Stmt::Return => writeln!(out, "{pad}return").unwrap(),
    }
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Shape(shape) => format!("shape![{}]", format_shape(shape)),
        Expr::ScalarI64(value) => format!("{value}i64"),
        Expr::ScalarF32(value) => format!("{value}f32"),
        Expr::ScalarBool(value) => format!("{value}"),
        Expr::TileIndex(coords) => {
            let parts = coords.iter().map(format_expr).collect::<Vec<_>>();
            format!("[{}]", parts.join(", "))
        }
        Expr::Builtin { op, args } => {
            let name = builtin_name(*op);
            if args.is_empty() {
                name.to_string()
            } else {
                let parts = args.iter().map(format_expr).collect::<Vec<_>>();
                format!("{name}({})", parts.join(", "))
            }
        }
    }
}

/// The name a unary float operation prints under, which is the name every
/// layer below spells it with.
pub fn format_unary_name(op: UnaryOp) -> &'static str {
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

fn builtin_name(op: BuiltinOp) -> &'static str {
    match op {
        BuiltinOp::ProgramId => "program_id",
        BuiltinOp::NumPrograms => "num_programs",
        BuiltinOp::NumTiles => "num_tiles",
        BuiltinOp::LoadTile => "load_tile",
        BuiltinOp::LoadLike => "load_like",
        BuiltinOp::Partition => "partition",
        BuiltinOp::PartitionMut => "partition_mut",
        BuiltinOp::Load => "load",
        BuiltinOp::Store => "store",
        BuiltinOp::Max => "max_tile",
        BuiltinOp::Min => "min_tile",
        BuiltinOp::CeilDiv => "ceil_div",
        BuiltinOp::Add => "addf",
        BuiltinOp::Sub => "subf",
        BuiltinOp::Mul => "mulf",
        BuiltinOp::Div => "divf",
        BuiltinOp::TrueDiv => "true_div",
        BuiltinOp::Rem => "remf",
        BuiltinOp::Unary(op) => format_unary_name(op),
        BuiltinOp::Gt => "gt_tile",
        BuiltinOp::Lt => "lt_tile",
        BuiltinOp::Ge => "ge_tile",
        BuiltinOp::Le => "le_tile",
        BuiltinOp::Eq => "eq_tile",
        BuiltinOp::Ne => "ne_tile",
        BuiltinOp::Select => "select",
        BuiltinOp::Fma => "fma",
        BuiltinOp::Iota => "iota",
        BuiltinOp::And => "andi",
        BuiltinOp::Or => "ori",
        BuiltinOp::TileToScalar => "tile_to_scalar",
        BuiltinOp::ScalarToTile => "scalar_to_tile",
        BuiltinOp::ConvertScalar => "convert_scalar",
        BuiltinOp::ReduceMax => "reduce_max",
        BuiltinOp::ReduceMin => "reduce_min",
        BuiltinOp::ReduceSum => "reduce_sum",
        BuiltinOp::ReduceProd => "reduce_prod",
        BuiltinOp::Cat => "cat",
        BuiltinOp::ExtractSlice => "extract",
        BuiltinOp::InsertSlice => "insert",
        BuiltinOp::ScanSum => "scan_sum",
        BuiltinOp::ScanProd => "scan_prod",
        BuiltinOp::Reshape => "reshape",
        BuiltinOp::Broadcast => "broadcast",
        BuiltinOp::Transpose => "transpose",
        BuiltinOp::ShapeOf => "shape_of",
        BuiltinOp::Mma => "mma",
        BuiltinOp::Constant => "constant",
        BuiltinOp::ScalarDiv => "scalar_div",
        BuiltinOp::ShapeDim => "shape_dim",
        BuiltinOp::Index => "index",
        BuiltinOp::Pow => "pow",
        BuiltinOp::Atan2 => "atan2",
        BuiltinOp::Xor => "xori",
        BuiltinOp::Shl => "shli",
        BuiltinOp::Shr => "shri",
    }
}

pub fn format_type(ty: &Type) -> String {
    match ty {
        Type::Tensor { elem, shape } => {
            format!("Tensor<{}, {{ [{}] }}>", format_elem(*elem), format_shape(shape))
        }
        Type::Tile { elem, shape } => {
            format!("Tile<{}, {{ [{}] }}>", format_elem(*elem), format_shape(shape))
        }
        Type::Partition { elem, shape, .. } => {
            format!("Partition<{}, {{ [{}] }}>", format_elem(*elem), format_shape(shape))
        }
        Type::Shape(shape) => format!("Shape<[{}]>", format_shape(shape)),
        Type::Scalar(elem) => format!("Scalar<{}>", format_elem(*elem)),
    }
}

fn format_elem(elem: ElemType) -> &'static str {
    match elem {
        ElemType::F32 => "f32",
        ElemType::F16 => "f16",
        ElemType::I64 => "i64",
        ElemType::Bool => "bool",
    }
}

fn format_shape(shape: &ShapeExpr) -> String {
    match shape {
        ShapeExpr::Dynamic => "-1".to_string(),
        ShapeExpr::Constant(value) => value.to_string(),
        ShapeExpr::Symbol(name) => name.clone(),
        ShapeExpr::Tuple(items) => items
            .iter()
            .map(format_shape)
            .collect::<Vec<_>>()
            .join(", "),
    }
}
