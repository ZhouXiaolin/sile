use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::*;

pub fn format_function(func: &Function) -> String {
    let mut out = String::new();
    let names = build_value_names(func);
    let block_names = build_block_names(func);

    writeln!(
        &mut out,
        "define {} @{}({}) {{",
        format_type(&Type::Void),
        func.name,
        format_params(&func.params, &names)
    )
    .unwrap();

    for block in &func.blocks {
        let block_name = block_names
            .get(&block.id)
            .cloned()
            .unwrap_or_else(|| block.name.clone());
        if block.params.is_empty() {
            writeln!(&mut out, "{}:", block_name).unwrap();
        } else {
            writeln!(
                &mut out,
                "{}({}):",
                block_name,
                format_block_params(&block.params, &names)
            )
            .unwrap();
        }

        for inst in &block.insts {
            let line = format_inst(inst, &names);
            if !line.is_empty() {
                writeln!(&mut out, "  {}", line).unwrap();
            }
        }

        writeln!(
            &mut out,
            "  {}",
            format_terminator(&block.terminator, &names, &block_names)
        )
        .unwrap();
    }

    writeln!(&mut out, "}}").unwrap();
    out
}

fn build_value_names(func: &Function) -> HashMap<ValueId, String> {
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

fn build_block_names(func: &Function) -> HashMap<BlockId, String> {
    func.blocks
        .iter()
        .map(|block| (block.id, block.name.clone()))
        .collect()
}

fn format_params(params: &[Param], names: &HashMap<ValueId, String>) -> String {
    params
        .iter()
        .map(|param| {
            let abi = param.abi.as_ref().map(format_param_abi).unwrap_or_default();
            format!(
                "{} %{}",
                format_type(&param.ty),
                names
                    .get(&param.id)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", param.id.0))
            ) + &abi
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_block_params(params: &[BlockParam], names: &HashMap<ValueId, String>) -> String {
    params
        .iter()
        .map(|param| {
            format!(
                "%{}: {}",
                names
                    .get(&param.id)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", param.id.0)),
                format_type(&param.ty)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_inst(inst: &Inst, names: &HashMap<ValueId, String>) -> String {
    let body = match &inst.op {
        InstOp::ShapeDim { buf, dim } => {
            format!("shape.dim {}, {}", format_operand(buf, names), dim)
        }
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => {
            format!(
                "alloca {}, addrspace({})",
                format_type(alloc_ty),
                format_addr_space(addr_space)
            )
        }
        InstOp::Gep { base, indices } => format!(
            "gep {}, [{}]",
            format_operand(base, names),
            indices
                .iter()
                .map(|idx| format_operand(idx, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        InstOp::Load { ptr } => format!("load {}", format_operand(ptr, names)),
        InstOp::Store { ptr, value } => format!(
            "store {}, {}",
            format_operand(ptr, names),
            format_operand(value, names)
        ),
        InstOp::Memcpy { dst, src, size } => format!(
            "memcpy {}, {}, {}",
            format_operand(dst, names),
            format_operand(src, names),
            format_operand(size, names)
        ),
        InstOp::Bin { op, lhs, rhs } => format!(
            "{} {}, {}",
            format_bin_op(*op),
            format_operand(lhs, names),
            format_operand(rhs, names)
        ),
        InstOp::Cmp { pred, lhs, rhs } => format!(
            "icmp {} {}, {}",
            format_cmp_pred(*pred),
            format_operand(lhs, names),
            format_operand(rhs, names)
        ),
        InstOp::Cast { op, value, to } => format!(
            "{} {} to {}",
            format_cast_op(*op),
            format_operand(value, names),
            format_type(to)
        ),
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => format!(
            "select {}, {}, {}",
            format_operand(cond, names),
            format_operand(on_true, names),
            format_operand(on_false, names)
        ),
        InstOp::Call { func, args } => format!(
            "call @{}({})",
            func,
            args.iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        InstOp::Intrinsic { intrinsic, args } => format!(
            "intrinsic {}({})",
            format_intrinsic(intrinsic),
            args.iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };

    let metadata = if inst.metadata.is_empty() {
        String::new()
    } else {
        format!(
            " [{}]",
            inst.metadata
                .iter()
                .map(format_metadata)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    match inst.result {
        Some(id) => format!(
            "%{} = {}{}",
            names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("v{}", id.0)),
            body,
            metadata
        ),
        None => format!("{}{}", body, metadata),
    }
}

fn format_terminator(
    term: &Terminator,
    names: &HashMap<ValueId, String>,
    block_names: &HashMap<BlockId, String>,
) -> String {
    match term {
        Terminator::Br { target, args } => format!(
            "br label %{}({})",
            block_names
                .get(target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", target.0)),
            args.iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => format!(
            "condbr {}, label %{}({}), label %{}({})",
            format_operand(cond, names),
            block_names
                .get(true_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", true_target.0)),
            true_args
                .iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", "),
            block_names
                .get(false_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", false_target.0)),
            false_args
                .iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Terminator::Switch {
            value,
            default,
            cases,
        } => format!(
            "switch {}, label %{} [{}]",
            format_operand(value, names),
            block_names
                .get(default)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", default.0)),
            cases
                .iter()
                .map(|(literal, target)| format!(
                    "{} -> %{}",
                    literal,
                    block_names
                        .get(target)
                        .cloned()
                        .unwrap_or_else(|| format!("bb{}", target.0))
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Terminator::Ret { value } => match value {
            Some(value) => format!("ret {}", format_operand(value, names)),
            None => "ret".to_string(),
        },
    }
}

fn format_operand(operand: &Operand, names: &HashMap<ValueId, String>) -> String {
    match operand {
        Operand::Value(id) => format!(
            "%{}",
            names
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("v{}", id.0))
        ),
        Operand::Const(constant) => format_constant(constant),
    }
}

fn format_constant(constant: &Constant) -> String {
    match constant {
        Constant::Int(value) => value.to_string(),
        Constant::Float(value) => {
            if value.fract() == 0.0 {
                format!("{value:.1}")
            } else {
                value.to_string()
            }
        }
        Constant::Bool(value) => {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
    }
}

fn format_type(ty: &Type) -> String {
    match ty {
        Type::Void => "void".to_string(),
        Type::I1 => "i1".to_string(),
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::F16 => "f16".to_string(),
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::Ptr {
            addr_space,
            pointee,
        } => {
            format!(
                "ptr<{}, {}>",
                format_addr_space(addr_space),
                format_type(pointee)
            )
        }
        Type::Vector { len, elem } => format!("<{} x {}>", len, format_type(elem)),
        Type::Array { len, elem } => format!("[{} x {}]", len, format_type(elem)),
    }
}

fn format_addr_space(addr_space: &AddressSpace) -> &'static str {
    match addr_space {
        AddressSpace::Generic => "generic",
        AddressSpace::Global => "global",
        AddressSpace::Constant => "constant",
        AddressSpace::Shared => "shared",
        AddressSpace::Private => "private",
    }
}

fn format_bin_op(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

fn format_cmp_pred(pred: CmpPred) -> &'static str {
    match pred {
        CmpPred::Eq => "eq",
        CmpPred::Ne => "ne",
        CmpPred::Slt => "slt",
        CmpPred::Sle => "sle",
        CmpPred::Sgt => "sgt",
        CmpPred::Sge => "sge",
        CmpPred::Olt => "olt",
        CmpPred::Ole => "ole",
        CmpPred::Ogt => "ogt",
        CmpPred::Oge => "oge",
    }
}

fn format_cast_op(op: CastOp) -> &'static str {
    match op {
        CastOp::Bitcast => "bitcast",
        CastOp::Sext => "sext",
        CastOp::Zext => "zext",
        CastOp::Trunc => "trunc",
        CastOp::Sitofp => "sitofp",
        CastOp::Fptosi => "fptosi",
    }
}

fn format_intrinsic(intrinsic: &Intrinsic) -> String {
    match intrinsic {
        Intrinsic::ThreadId { dim } => format!("thread_id.{}", dim),
        Intrinsic::BlockId { dim } => format!("block_id.{}", dim),
        Intrinsic::Barrier { scope } => format!("barrier.{}", format_scope(scope)),
        Intrinsic::MatmulFragment => "matmul_fragment".to_string(),
        Intrinsic::ReduceAdd => "reduce_add".to_string(),
        Intrinsic::ReduceMax => "reduce_max".to_string(),
    }
}

fn format_scope(scope: &SyncScope) -> &'static str {
    match scope {
        SyncScope::Workgroup => "workgroup",
        SyncScope::Device => "device",
    }
}

fn format_metadata(metadata: &Metadata) -> String {
    match metadata {
        Metadata::Parallel => "parallel".to_string(),
        Metadata::Reduction => "reduction".to_string(),
        Metadata::VectorizeWidth(width) => format!("vectorize_width={width}"),
        Metadata::Unroll(factor) => format!("unroll={factor}"),
        Metadata::Alignment(bytes) => format!("align={bytes}"),
        Metadata::NoAlias => "noalias".to_string(),
        Metadata::ReadOnly => "readonly".to_string(),
        Metadata::WriteOnly => "writeonly".to_string(),
    }
}

fn format_param_abi(abi: &ParamAbi) -> String {
    format!(" [rank={}, shape_offset={}]", abi.rank, abi.shape_offset)
}
