use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::*;

pub fn format_llvm_ir(func: &Function) -> String {
    let mut out = String::new();
    let names = build_value_names(func);
    let block_names = build_block_names(func);
    let types = build_value_types(func);

    writeln!(
        &mut out,
        "define {} @{}({}) {{",
        format_type(&Type::Void),
        func.name,
        format_params(&func.params, &names)
    )
    .unwrap();
    for param in &func.params {
        if let Some(comment) = format_param_comment(param, &names) {
            writeln!(&mut out, "  ; {}", comment).unwrap();
        }
    }

    for block in &func.blocks {
        let block_name = block_names
            .get(&block.id)
            .cloned()
            .unwrap_or_else(|| block.name.clone());
        writeln!(&mut out, "{}:", block_name).unwrap();
        if !block.params.is_empty() {
            writeln!(
                &mut out,
                "  ; args({})",
                format_block_params(&block.params, &names)
            )
            .unwrap();
        }

        for inst in &block.insts {
            let line = format_inst(inst, &names, &types);
            if !line.is_empty() {
                writeln!(&mut out, "  {}", line).unwrap();
            }
        }

        writeln!(
            &mut out,
            "  {}",
            format_terminator(&block.terminator, &names, &block_names, &types)
        )
        .unwrap();
    }

    writeln!(&mut out, "}}").unwrap();
    out
}

pub fn format_function(func: &Function) -> String {
    format_llvm_ir(func)
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

fn build_value_types(func: &Function) -> HashMap<ValueId, Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(id) = inst.result {
                types.insert(id, inst.ty.clone());
            }
        }
    }
    types
}

fn format_params(params: &[Param], names: &HashMap<ValueId, String>) -> String {
    params
        .iter()
        .map(|param| {
            format!(
                "{} %{}",
                format_type(&param.ty),
                names
                    .get(&param.id)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", param.id.0))
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_param_comment(param: &Param, names: &HashMap<ValueId, String>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(abi) = &param.abi {
        parts.push(format_param_abi(abi));
    }
    if let Type::Ptr { pointee, .. } = &param.ty {
        parts.push(format!("elem={}", format_type(pointee)));
    }
    if parts.is_empty() {
        None
    } else {
        Some(format!(
            "sile.param %{} [{}]",
            names
                .get(&param.id)
                .cloned()
                .unwrap_or_else(|| format!("v{}", param.id.0)),
            parts.join(", ")
        ))
    }
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

fn format_inst(
    inst: &Inst,
    names: &HashMap<ValueId, String>,
    types: &HashMap<ValueId, Type>,
) -> String {
    let body = match &inst.op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => format!(
            "alloca {}, addrspace({})",
            format_type(alloc_ty),
            format_addr_space_number(addr_space)
        ),
        InstOp::Gep { base, indices } => format!(
            "getelementptr {}, {} {}, {}",
            format_pointee_type(base, types),
            format_operand_type(base, types),
            format_operand(base, names),
            indices
                .iter()
                .map(|idx| format!("i64 {}", format_operand(idx, names)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        InstOp::Load { ptr } => format!(
            "load {}, {} {}",
            format_load_store_value_type(ptr, types),
            format_operand_type(ptr, types),
            format_operand(ptr, names)
        ),
        InstOp::Store { ptr, value } => format!(
            "store {} {}, {} {}",
            format_load_store_value_type(ptr, types),
            format_operand(value, names),
            format_operand_type(ptr, types),
            format_operand(ptr, names)
        ),
        InstOp::AtomicAdd { ptr, value } => format!(
            "atomicrmw {} {} {}, {} {} monotonic",
            format_atomic_rmw_op(value, types),
            format_operand_type(ptr, types),
            format_operand(ptr, names),
            format_operand_type(value, types),
            format_operand(value, names)
        ),
        InstOp::Memcpy { dst, src, size } => format!(
            "call void @llvm.memcpy.inline({} {}, {} {}, i64 {}, i1 false)",
            format_operand_type(dst, types),
            format_operand(dst, names),
            format_operand_type(src, types),
            format_operand(src, names),
            format_operand(size, names),
        ),
        InstOp::Bin { op, lhs, rhs } => format!(
            "{} {}, {}",
            format_bin_op(*op, &inst.ty),
            format_operand(lhs, names),
            format_operand(rhs, names)
        ),
        InstOp::Cmp { pred, lhs, rhs } => format!(
            "{} {} {}, {}",
            format_cmp_mnemonic(*pred),
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
            "select i1 {}, {} {}, {} {}",
            format_operand(cond, names),
            format_operand_type(on_true, types),
            format_operand(on_true, names),
            format_operand_type(on_false, types),
            format_operand(on_false, names)
        ),
        InstOp::Call { func, args } => format!(
            "call {} @{}({})",
            format_type(&inst.ty),
            func,
            args.iter()
                .map(|arg| format!(
                    "{} {}",
                    format_operand_type(arg, types),
                    format_operand(arg, names)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        InstOp::Intrinsic { intrinsic, args } => {
            format_intrinsic(intrinsic, &inst.ty, args, names, types)
        }
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
    types: &HashMap<ValueId, Type>,
) -> String {
    match term {
        Terminator::Br { target, args } => {
            let target_name = block_names
                .get(target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", target.0));
            let args_suffix = format_branch_args_suffix("args", args, names);
            format!("br label %{target_name}{args_suffix}")
        }
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => format!(
            "br i1 {}, label %{}, label %{}{}{}",
            format_operand(cond, names),
            block_names
                .get(true_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", true_target.0)),
            block_names
                .get(false_target)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", false_target.0)),
            format_branch_args_suffix("true_args", true_args, names),
            format_branch_args_suffix("false_args", false_args, names)
        ),
        Terminator::Switch {
            value,
            default,
            cases,
        } => format!(
            "switch {} {}, label %{} [{}]",
            format_operand_type(value, types),
            format_operand(value, names),
            block_names
                .get(default)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", default.0)),
            cases
                .iter()
                .map(|(literal, target)| format!(
                    "{} , label %{}",
                    literal,
                    block_names
                        .get(target)
                        .cloned()
                        .unwrap_or_else(|| format!("bb{}", target.0))
                ))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        Terminator::Ret { value } => match value {
            Some(value) => format!(
                "ret {} {}",
                format_operand_type(value, types),
                format_operand(value, names)
            ),
            None => "ret void".to_string(),
        },
    }
}

fn format_branch_args_suffix(
    label: &str,
    args: &[Operand],
    names: &HashMap<ValueId, String>,
) -> String {
    if args.is_empty() {
        String::new()
    } else {
        format!(
            " ; {label}({})",
            args.iter()
                .map(|arg| format_operand(arg, names))
                .collect::<Vec<_>>()
                .join(", ")
        )
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

fn format_operand_type(operand: &Operand, types: &HashMap<ValueId, Type>) -> String {
    match operand {
        Operand::Value(id) => types
            .get(id)
            .map(format_type)
            .unwrap_or_else(|| "i64".to_string()),
        Operand::Const(constant) => format_constant_type(constant).to_string(),
    }
}

fn format_constant_type(constant: &Constant) -> &'static str {
    match constant {
        Constant::Int(_) => "i64",
        Constant::Float(_) => "f64",
        Constant::Bool(_) => "i1",
    }
}

fn format_pointee_type(operand: &Operand, types: &HashMap<ValueId, Type>) -> String {
    match operand {
        Operand::Value(id) => types
            .get(id)
            .and_then(pointee_type)
            .map(format_type)
            .unwrap_or_else(|| "i8".to_string()),
        Operand::Const(_) => "i8".to_string(),
    }
}

fn format_load_store_value_type(operand: &Operand, types: &HashMap<ValueId, Type>) -> String {
    match operand {
        Operand::Value(id) => types
            .get(id)
            .and_then(pointee_type)
            .map(format_type)
            .unwrap_or_else(|| "i8".to_string()),
        Operand::Const(_) => "i8".to_string(),
    }
}

fn pointee_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr { pointee, .. } => Some(pointee),
        _ => None,
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
        Type::Ptr { addr_space, .. } => format_llvm_ptr_type(addr_space),
        Type::Vector { len, elem } => format!("<{} x {}>", len, format_type(elem)),
        Type::Array { len, elem } => format!("[{} x {}]", len, format_type(elem)),
    }
}

fn format_llvm_ptr_type(addr_space: &AddressSpace) -> String {
    match addr_space {
        AddressSpace::Generic => "ptr".into(),
        _ => format!("ptr addrspace({})", format_addr_space_number(addr_space)),
    }
}

fn format_addr_space_number(addr_space: &AddressSpace) -> u8 {
    match addr_space {
        AddressSpace::Generic => 0,
        AddressSpace::Global => 1,
        AddressSpace::Shared => 3,
        AddressSpace::Constant => 4,
        AddressSpace::Private => 5,
    }
}

fn format_bin_op(op: BinOp, ty: &Type) -> &'static str {
    match (op, is_float_type(ty)) {
        (BinOp::Add, true) => "fadd",
        (BinOp::Sub, true) => "fsub",
        (BinOp::Mul, true) => "fmul",
        (BinOp::Div, true) => "fdiv",
        (BinOp::Add, false) => "add",
        (BinOp::Sub, false) => "sub",
        (BinOp::Mul, false) => "mul",
        (BinOp::Div, false) => "sdiv",
        (BinOp::And, _) => "and",
        (BinOp::Or, _) => "or",
    }
}

fn is_float_type(ty: &Type) -> bool {
    matches!(ty, Type::F16 | Type::F32 | Type::F64)
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

fn format_cmp_mnemonic(pred: CmpPred) -> &'static str {
    match pred {
        CmpPred::Olt | CmpPred::Ole | CmpPred::Ogt | CmpPred::Oge => "fcmp",
        _ => "icmp",
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

fn format_intrinsic(
    intrinsic: &Intrinsic,
    result_ty: &Type,
    args: &[Operand],
    names: &HashMap<ValueId, String>,
    types: &HashMap<ValueId, Type>,
) -> String {
    match intrinsic {
        Intrinsic::ThreadId { dim } => {
            format!("call i32 @llvm.nvvm.tid.{}()", format_dim_suffix(*dim))
        }
        Intrinsic::BlockId { dim } => {
            format!("call i32 @llvm.nvvm.ctaid.{}()", format_dim_suffix(*dim))
        }
        Intrinsic::Barrier { .. } => "call void @llvm.nvvm.barrier0()".to_string(),
        Intrinsic::Exp => {
            let arg = args.first().expect("exp intrinsic expects one argument");
            format!(
                "call {} @llvm.exp.{}({} {})",
                format_type(result_ty),
                format_exp_suffix(result_ty),
                format_operand_type(arg, types),
                format_operand(arg, names)
            )
        }
        Intrinsic::VecLoad { len } => {
            let ptr = &args[0];
            let offset = &args[1];
            format!(
                "call {} @llvm.vec.load.{}({} {}, {} {})",
                format_type(result_ty),
                len,
                format_operand_type(ptr, types),
                format_operand(ptr, names),
                format_operand_type(offset, types),
                format_operand(offset, names)
            )
        }
        Intrinsic::VecStore { len } => {
            let ptr = &args[0];
            let offset = &args[1];
            let value = &args[2];
            format!(
                "call void @llvm.vec.store.{}({} {}, {} {}, {} {})",
                len,
                format_operand_type(ptr, types),
                format_operand(ptr, names),
                format_operand_type(offset, types),
                format_operand(offset, names),
                format_operand_type(value, types),
                format_operand(value, names)
            )
        }
        Intrinsic::VecReduceAdd => {
            let vector = &args[0];
            format!(
                "call {} @llvm.vec.reduce.add({} {})",
                format_type(result_ty),
                format_operand_type(vector, types),
                format_operand(vector, names)
            )
        }
    }
}

fn format_dim_suffix(dim: u8) -> &'static str {
    match dim {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "x",
    }
}

fn format_exp_suffix(ty: &Type) -> &'static str {
    match ty {
        Type::F64 => "f64",
        _ => "f32",
    }
}

fn format_atomic_rmw_op(value: &Operand, types: &HashMap<ValueId, Type>) -> &'static str {
    if operand_is_float(value, types) {
        "fadd"
    } else {
        "add"
    }
}

fn operand_is_float(operand: &Operand, types: &HashMap<ValueId, Type>) -> bool {
    match operand {
        Operand::Value(id) => types.get(id).map(is_float_type).unwrap_or(false),
        Operand::Const(Constant::Float(_)) => true,
        Operand::Const(_) => false,
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
    format!("rank={}, shape_offset={}", abi.rank, abi.shape_offset)
}
