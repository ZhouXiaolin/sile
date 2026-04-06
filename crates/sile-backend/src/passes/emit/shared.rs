use std::collections::HashMap;

use sile_llir as llir;

pub fn build_value_names(func: &llir::Function) -> HashMap<llir::ValueId, String> {
    let mut names = HashMap::new();
    for param in &func.params {
        names.insert(param.id, param.name.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            names.insert(param.id, param.name.clone());
        }
        for inst in &block.insts {
            if let (Some(id), Some(name)) = (inst.result, inst.result_name.as_ref()) {
                names.insert(id, name.clone());
            }
        }
    }
    names
}

pub fn build_param_indices(func: &llir::Function) -> HashMap<llir::ValueId, usize> {
    func.params
        .iter()
        .enumerate()
        .map(|(idx, param)| (param.id, idx))
        .collect()
}

pub fn value_name(value_names: &HashMap<llir::ValueId, String>, id: llir::ValueId) -> String {
    value_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", id.0))
}

pub fn format_operand(
    value_names: &HashMap<llir::ValueId, String>,
    operand: &llir::Operand,
) -> String {
    match operand {
        llir::Operand::Value(id) => value_name(value_names, *id),
        llir::Operand::Const(llir::Constant::Int(value)) => value.to_string(),
        llir::Operand::Const(llir::Constant::Float(value)) => float_literal(*value),
        llir::Operand::Const(llir::Constant::Bool(value)) => {
            if *value {
                "true".into()
            } else {
                "false".into()
            }
        }
    }
}

pub fn block_param_assignments(
    func: &llir::Function,
    value_names: &HashMap<llir::ValueId, String>,
    target: llir::BlockId,
    args: &[llir::Operand],
) -> Vec<(String, String)> {
    let Some(block) = func.blocks.iter().find(|block| block.id == target) else {
        return Vec::new();
    };
    block
        .params
        .iter()
        .zip(args.iter())
        .map(|(param, arg)| {
            (
                value_name(value_names, param.id),
                format_operand(value_names, arg),
            )
        })
        .collect()
}

pub fn array_dims(ty: &llir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let llir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

pub fn bin_op_symbol(op: llir::BinOp) -> &'static str {
    match op {
        llir::BinOp::Add => "+",
        llir::BinOp::Sub => "-",
        llir::BinOp::Mul => "*",
        llir::BinOp::Div => "/",
        llir::BinOp::And => "&",
        llir::BinOp::Or => "|",
    }
}

pub fn cmp_pred_symbol(pred: llir::CmpPred) -> &'static str {
    match pred {
        llir::CmpPred::Eq => "==",
        llir::CmpPred::Ne => "!=",
        llir::CmpPred::Slt | llir::CmpPred::Olt => "<",
        llir::CmpPred::Sle | llir::CmpPred::Ole => "<=",
        llir::CmpPred::Sgt | llir::CmpPred::Ogt => ">",
        llir::CmpPred::Sge | llir::CmpPred::Oge => ">=",
    }
}

pub fn float_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value}f")
    }
}
