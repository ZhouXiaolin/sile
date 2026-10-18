use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use crate::ir::*;

pub fn format_llvm_ir(func: &Function) -> String {
    let mut out = String::new();
    let names = build_value_names(func);
    let block_names = build_block_names(func);
    let types = build_value_types(func);
    let incoming = build_incoming(func);
    let mut footer = ModuleFooter::default();

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
        // A block parameter is defined by the block label, not by an
        // instruction, so the label has to carry it.
        writeln!(&mut out, "{}:", block_name).unwrap();
        for (index, param) in block.params.iter().enumerate() {
            writeln!(
                &mut out,
                "  {}",
                format_block_param_phi(block.id, index, param, &incoming, &names, &block_names)
            )
            .unwrap();
        }

        for inst in &block.insts {
            let line = format_inst(inst, &names, &types, &mut footer);
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
    out.push_str(&footer.format());
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

/// The value each block receives from each predecessor.
///
/// LLVM's textual IR cannot name a block parameter: its lexer reads a label only
/// when a colon follows the name at once, so `header(%i: i64):` is not a label
/// at all. The same control flow is written with `phi` instructions, which is
/// the form LLVM's own writer produces for an in-memory block argument.
fn build_incoming(func: &Function) -> HashMap<BlockId, Vec<(BlockId, Vec<Operand>)>> {
    let mut incoming: HashMap<BlockId, Vec<(BlockId, Vec<Operand>)>> = HashMap::new();
    for block in &func.blocks {
        match &block.terminator {
            Terminator::Br { target, args } => {
                incoming.entry(*target).or_default().push((block.id, args.clone()));
            }
            Terminator::CondBr {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                incoming
                    .entry(*true_target)
                    .or_default()
                    .push((block.id, true_args.clone()));
                incoming
                    .entry(*false_target)
                    .or_default()
                    .push((block.id, false_args.clone()));
            }
            Terminator::Switch { .. } | Terminator::Ret { .. } => {}
        }
    }
    incoming
}

/// One block parameter, as the phi that carries it.
fn format_block_param_phi(
    block: BlockId,
    index: usize,
    param: &BlockParam,
    incoming: &HashMap<BlockId, Vec<(BlockId, Vec<Operand>)>>,
    names: &HashMap<ValueId, String>,
    block_names: &HashMap<BlockId, String>,
) -> String {
    let name = names
        .get(&param.id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", param.id.0));
    let ty = format_type(&param.ty);
    let Some(edges) = incoming.get(&block) else {
        // A block with no predecessor receives nothing: only the entry block is
        // like that, and nothing passes it a value.
        return format!("%{name} = freeze {ty} undef");
    };
    let edges = edges
        .iter()
        .map(|(predecessor, args)| {
            let value = args
                .get(index)
                .map(|arg| format_operand(arg, names))
                .unwrap_or_else(|| "undef".to_string());
            let predecessor_name = block_names
                .get(predecessor)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", predecessor.0));
            format!("[ {value}, %{predecessor_name} ]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("%{name} = phi {ty} {edges}")
}

fn format_inst(
    inst: &Inst,
    names: &HashMap<ValueId, String>,
    types: &HashMap<ValueId, Type>,
    footer: &mut ModuleFooter,
) -> String {
    // An `alloca` takes its alignment before the address space, so it is the one
    // memory instruction whose alignment is not written last.
    let alloca_alignment = if matches!(&inst.op, InstOp::Alloca { .. }) {
        format_alignment_attribute(&inst.metadata)
    } else {
        String::new()
    };
    let body = match &inst.op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => format!(
            "alloca {}{}, addrspace({})",
            format_type(alloc_ty),
            alloca_alignment,
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
            "{} {} {}, {}",
            format_bin_op(*op, &inst.ty),
            format_type(&inst.ty),
            format_operand(lhs, names),
            format_operand(rhs, names)
        ),
        InstOp::Cmp { pred, lhs, rhs } => format!(
            "{} {} {} {}, {}",
            format_cmp_mnemonic(*pred),
            format_cmp_pred(*pred),
            shared_operand_type(lhs, rhs, types),
            format_operand(lhs, names),
            format_operand(rhs, names)
        ),
        InstOp::Cast { op, value, to } => format!(
            "{} {} {} to {}",
            format_cast_op(*op),
            format_operand_type(value, types),
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
            format_type(&inst.ty),
            format_operand(on_true, names),
            format_type(&inst.ty),
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
            format_intrinsic(intrinsic, &inst.ty, args, names, types, footer)
        }
    };

    let attributes = if matches!(&inst.op, InstOp::Alloca { .. }) {
        String::new()
    } else {
        format_alignment_attribute(&inst.metadata)
    };
    let attachments = format_metadata_attachments(&inst.metadata, footer);
    let suffix = format!("{attributes}{attachments}");

    match inst.result {
        Some(id) => format!(
            "%{} = {}{}",
            names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("v{}", id.0)),
            body,
            suffix
        ),
        None => format!("{}{}", body, suffix),
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
            // A case literal carries the type of the value it is compared with.
            cases
                .iter()
                .map(|(literal, target)| format!(
                    "{} {}, label %{}",
                    format_operand_type(value, types),
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

/// The type two operands of one instruction share.
///
/// A constant in this IR carries no width, so its type comes from the operand
/// beside it: the same `0.0` is a `float` next to a float and a `double` next
/// to a double. Reading the constant's own type instead made every comparison
/// against a float constant a `double` one.
fn shared_operand_type(lhs: &Operand, rhs: &Operand, types: &HashMap<ValueId, Type>) -> String {
    match (lhs, rhs) {
        (Operand::Value(_), _) | (_, Operand::Value(_)) => {
            let valued = if matches!(lhs, Operand::Value(_)) { lhs } else { rhs };
            format_operand_type(valued, types)
        }
        _ => format_operand_type(lhs, types),
    }
}

fn format_constant_type(constant: &Constant) -> &'static str {
    match constant {
        Constant::Int(_) => "i64",
        Constant::Float(_) => "double",
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
        // LLVM names its floating-point types after their width in the IEEE
        // sense, not after the bit count: `half`, `float`, `double`.
        Type::F16 => "half".to_string(),
        Type::F32 => "float".to_string(),
        Type::F64 => "double".to_string(),
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
        (BinOp::Rem, true) => "frem",
        (BinOp::Add, false) => "add",
        (BinOp::Sub, false) => "sub",
        (BinOp::Mul, false) => "mul",
        (BinOp::Div, false) => "sdiv",
        (BinOp::Rem, false) => "srem",
        (BinOp::And, _) => "and",
        (BinOp::Or, _) => "or",
        (BinOp::Xor, _) => "xor",
        (BinOp::Shl, _) => "shl",
        (BinOp::Shr, _) => "ashr",
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
        CastOp::Fext => "fpext",
        CastOp::Ftrunc => "fptrunc",
    }
}

fn format_intrinsic(
    intrinsic: &Intrinsic,
    result_ty: &Type,
    args: &[Operand],
    names: &HashMap<ValueId, String>,
    types: &HashMap<ValueId, Type>,
    footer: &mut ModuleFooter,
) -> String {
    let operand = |index: usize| -> (String, String) {
        let arg = &args[index];
        (
            format_operand_type(arg, types),
            format_operand(arg, names),
        )
    };
    match intrinsic {
        Intrinsic::ThreadId { dim } => format_runtime_call(
            &format!("sile.thread.id.{}", format_dim_suffix(*dim)),
            result_ty,
            &[],
            footer,
        ),
        Intrinsic::BlockId { dim } => format_runtime_call(
            &format!("sile.block.id.{}", format_dim_suffix(*dim)),
            result_ty,
            &[],
            footer,
        ),
        Intrinsic::GridDim { dim } => format_runtime_call(
            &format!("sile.grid.dim.{}", format_dim_suffix(*dim)),
            result_ty,
            &[],
            footer,
        ),
        Intrinsic::Barrier { .. } => "call void @llvm.nvvm.barrier0()".to_string(),
        Intrinsic::Exp
        | Intrinsic::Exp2
        | Intrinsic::Log
        | Intrinsic::Log2
        | Intrinsic::Sqrt
        | Intrinsic::Abs
        | Intrinsic::Sin
        | Intrinsic::Cos
        | Intrinsic::Tan
        | Intrinsic::Sinh
        | Intrinsic::Cosh
        | Intrinsic::Tanh
        | Intrinsic::Ceil
        | Intrinsic::Floor => {
            let function = intrinsic
                .float_function()
                .expect("a float function has a name");
            let arg = args.first().expect("a float function takes one argument");
            format!(
                "call {} @llvm.{}.{}({} {})",
                format_type(result_ty),
                function,
                format_exp_suffix(result_ty),
                format_operand_type(arg, types),
                format_operand(arg, names)
            )
        }
        Intrinsic::Pow | Intrinsic::Atan2 => {
            let function = intrinsic
                .binary_float_function()
                .expect("a binary float function has a name");
            let lhs = args.first().expect("a binary float function takes two arguments");
            let rhs = args.get(1).expect("a binary float function takes two arguments");
            format!(
                "call {} @llvm.{}.{}({} {}, {} {})",
                format_type(result_ty),
                function,
                format_exp_suffix(result_ty),
                format_operand_type(lhs, types),
                format_operand(lhs, names),
                format_operand_type(rhs, types),
                format_operand(rhs, names)
            )
        }
        Intrinsic::VecLoad { len } => format_runtime_call(
            &format!("sile.vec.load.{len}"),
            result_ty,
            &[operand(0), operand(1)],
            footer,
        ),
        Intrinsic::VecStore { len } => format_runtime_call(
            &format!("sile.vec.store.{len}"),
            result_ty,
            &[operand(0), operand(1), operand(2)],
            footer,
        ),
        Intrinsic::VecReduceAdd => format_runtime_call(
            "sile.vec.reduce.add",
            result_ty,
            &[operand(0)],
            footer,
        ),
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

/// The parts of a module that are collected while it is written: the metadata
/// nodes it references and the declarations its calls need.
///
/// LLVM spells alignment as an instruction attribute and every other hint as a
/// metadata node. This IR carries both in one list, so the printer splits them:
/// the module stays one `llvm-as` accepts, and no hint is dropped.
#[derive(Default)]
struct ModuleFooter {
    metadata: Vec<Metadata>,
    declares: BTreeSet<String>,
}

impl ModuleFooter {
    /// The number that names `metadata` in the printed module.
    fn metadata_number(&mut self, metadata: &Metadata) -> usize {
        if let Some(index) = self.metadata.iter().position(|node| node == metadata) {
            return index;
        }
        self.metadata.push(metadata.clone());
        self.metadata.len() - 1
    }

    /// Record a declaration the module needs.
    fn declare(&mut self, declaration: String) {
        self.declares.insert(declaration);
    }

    /// The declarations and metadata nodes, in a stable order.
    fn format(&self) -> String {
        let mut out = String::new();
        for declaration in &self.declares {
            writeln!(&mut out, "{declaration}").unwrap();
        }
        for (number, node) in self.metadata.iter().enumerate() {
            writeln!(&mut out, "!{number} = {}", format_metadata_node(node)).unwrap();
        }
        out
    }
}

/// A call to a primitive the runtime supplies, and the declaration it needs.
///
/// These are not LLVM intrinsics: no portable LLVM intrinsic names a workgroup
/// id or a vector load. Naming them under `sile.` keeps the module valid LLVM —
/// a call to a declared function — and says plainly that the runtime supplies
/// them, which is what the three backends do.
fn format_runtime_call(
    name: &str,
    result_ty: &Type,
    args: &[(String, String)],
    footer: &mut ModuleFooter,
) -> String {
    let parameters = args
        .iter()
        .map(|(ty, _)| ty.clone())
        .collect::<Vec<_>>()
        .join(", ");
    footer.declare(format!(
        "declare {} @{name}({parameters})",
        format_type(result_ty)
    ));
    let operands = args
        .iter()
        .map(|(ty, value)| format!("{ty} {value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("call {} @{name}({operands})", format_type(result_ty))
}

/// Whether LLVM accepts this hint as an instruction attribute.
///
/// Alignment is the only one: LLVM has an `align` attribute on the memory
/// instructions, and no attribute for the rest.
fn is_instruction_attribute(metadata: &Metadata) -> bool {
    matches!(metadata, Metadata::Alignment(_))
}

/// The name this hint is attached under.
fn metadata_name(metadata: &Metadata) -> &'static str {
    match metadata {
        Metadata::Parallel => "sile.parallel",
        Metadata::Reduction => "sile.reduction",
        Metadata::VectorizeWidth(_) => "sile.vectorize_width",
        Metadata::Unroll(_) => "sile.unroll",
        Metadata::Alignment(_) => "sile.align",
        Metadata::NoAlias => "sile.noalias",
        Metadata::ReadOnly => "sile.readonly",
        Metadata::WriteOnly => "sile.writeonly",
    }
}

/// The definition of one metadata node.
fn format_metadata_node(metadata: &Metadata) -> String {
    let name = metadata_name(metadata);
    match metadata {
        Metadata::VectorizeWidth(width) => format!("!{{!\"{name}\", i32 {width}}}"),
        Metadata::Unroll(factor) => format!("!{{!\"{name}\", i32 {factor}}}"),
        Metadata::Alignment(bytes) => format!("!{{!\"{name}\", i32 {bytes}}}"),
        _ => format!("!{{!\"{name}\"}}"),
    }
}

/// The alignment attribute, which every memory instruction takes last.
fn format_alignment_attribute(metadata: &[Metadata]) -> String {
    let mut out = String::new();
    for entry in metadata {
        if let Metadata::Alignment(bytes) = entry {
            write!(&mut out, ", align {bytes}").unwrap();
        }
    }
    out
}

/// The named metadata that follows the body, as `!sile.unroll !0`.
fn format_metadata_attachments(metadata: &[Metadata], footer: &mut ModuleFooter) -> String {
    let mut out = String::new();
    for entry in metadata {
        if is_instruction_attribute(entry) {
            continue;
        }
        let number = footer.metadata_number(entry);
        write!(&mut out, ", !{} !{}", metadata_name(entry), number).unwrap();
    }
    out
}

fn format_param_abi(abi: &ParamAbi) -> String {
    format!("rank={}, shape_offset={}", abi.rank, abi.shape_offset)
}
