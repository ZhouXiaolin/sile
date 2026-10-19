//! Printing a [`crate::cutile::Module`] as cutile-rs prints it.
//!
//! Every rule here was read off real cutile output; see
//! `crates/sile-tile-ir/TILEIR_ALIGNMENT.md`, which also holds the table of op
//! spellings this file has to reproduce.

use std::fmt::Write as _;

use super::{Module, Op, Stmt};

/// The text of a whole program.
pub fn format(module: &Module) -> String {
    let mut out = String::new();
    writeln!(&mut out, "cuda_tile.module @{} {{", module.name).unwrap();
    for function in &module.functions {
        format_function(function, &mut out);
    }
    writeln!(&mut out, "}}").unwrap();
    out
}

fn format_function(function: &super::Function, out: &mut String) {
    let args = function
        .args
        .iter()
        .map(|(value, ty)| format!("{value}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        out,
        "  entry @{}({}) di_name = {:?} {{",
        function.name, args, function.di_name
    )
    .unwrap();
    for stmt in &function.body {
        format_stmt(stmt, out);
    }
    writeln!(out, "  }}").unwrap();
}

fn format_stmt(stmt: &Stmt, out: &mut String) {
    match stmt {
        Stmt::Op(op) => format_op(op, out),
        Stmt::Raw(text) => {
            writeln!(out, "{text}").unwrap();
        }
        Stmt::Return => writeln!(out, "    return").unwrap(),
    }
}

fn format_op(op: &Op, out: &mut String) {
    let results = op
        .results
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    // An op with no results prints no `=`: `assert` is the one such op.
    if op.results.is_empty() {
        write!(out, "{}    {}", " ".repeat(op.indent), op.opcode).unwrap();
    } else {
        write!(out, "{}    {results} = {}", " ".repeat(op.indent), op.opcode).unwrap();
    }
    if !op.text.is_empty() {
        write!(out, " {}", op.text).unwrap();
    }
    if !op.attrs.is_empty() {
        let attrs = op
            .attrs
            .iter()
            .map(|attr| format!("{} = {}", attr.name, attr.value))
            .collect::<Vec<_>>()
            .join(", ");
        write!(out, " {{{attrs}}}").unwrap();
    }
    if op.format == super::Format::MakeTensorView && !op.text.is_empty() && op.text.contains(" -> ") {
        writeln!(out).unwrap();
        return;
    }
    match &op.operand_types {
        Some(types) => {
            let types = types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let results = op
                .result_types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, " : {types} -> {results}").unwrap();
        }
        None => {
            if let Some(ty) = op.result_types.first() {
                writeln!(out, " : {ty}").unwrap();
            } else {
                writeln!(out).unwrap();
            }
        }
    }
}
