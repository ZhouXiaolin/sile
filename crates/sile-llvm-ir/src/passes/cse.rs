use std::collections::HashMap;

use crate::{BinOp, Constant, Function, Inst, InstOp, Operand, Terminator, ValueId};

/// Local common-subexpression elimination for side-effect-free LLVM IR ops.
pub fn run(mut func: Function) -> Function {
    for block in &mut func.blocks {
        let mut expr_to_value = HashMap::<String, ValueId>::new();
        let mut replacements = HashMap::<ValueId, ValueId>::new();
        let mut new_insts = Vec::with_capacity(block.insts.len());

        for inst in &block.insts {
            let rewritten = rewrite_inst(inst.clone(), &replacements);

            if rewritten.metadata.is_empty() {
                if let Some(key) = expr_key(&rewritten.op) {
                    if let Some(existing) = expr_to_value.get(&key).copied() {
                        if let Some(result) = rewritten.result {
                            replacements.insert(result, existing);
                            continue;
                        }
                    } else if let Some(result) = rewritten.result {
                        expr_to_value.insert(key, result);
                    }
                }
            }

            new_insts.push(rewritten);
        }

        block.insts = new_insts;
        block.terminator = rewrite_terminator(block.terminator.clone(), &replacements);
    }

    func
}

fn expr_key(op: &InstOp) -> Option<String> {
    match op {
        InstOp::Gep { base, indices } => Some(format!(
            "gep:{}:[{}]",
            operand_key(base),
            indices
                .iter()
                .map(operand_key)
                .collect::<Vec<_>>()
                .join(",")
        )),
        InstOp::Bin { op, lhs, rhs } => {
            let (lhs, rhs) = canonical_bin_pair(*op, operand_key(lhs), operand_key(rhs));
            Some(format!("bin:{op:?}:{lhs}:{rhs}"))
        }
        InstOp::Cmp { pred, lhs, rhs } => Some(format!(
            "cmp:{pred:?}:{}:{}",
            operand_key(lhs),
            operand_key(rhs)
        )),
        InstOp::Cast { op, value, to } => {
            Some(format!("cast:{op:?}:{}:{to:?}", operand_key(value)))
        }
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => Some(format!(
            "select:{}:{}:{}",
            operand_key(cond),
            operand_key(on_true),
            operand_key(on_false)
        )),
        InstOp::Intrinsic { intrinsic, args } if is_pure_intrinsic(intrinsic) => Some(format!(
            "intrinsic:{intrinsic:?}:[{}]",
            args.iter().map(operand_key).collect::<Vec<_>>().join(",")
        )),
        InstOp::Alloca { .. }
        | InstOp::Load { .. }
        | InstOp::Store { .. }
        | InstOp::AtomicAdd { .. }
        | InstOp::Memcpy { .. }
        | InstOp::Call { .. }
        | InstOp::Intrinsic { .. } => None,
    }
}

fn is_pure_intrinsic(intrinsic: &crate::Intrinsic) -> bool {
    !matches!(intrinsic, crate::Intrinsic::Barrier { .. })
}

fn operand_key(operand: &Operand) -> String {
    match operand {
        Operand::Value(value) => format!("%{}", value.0),
        Operand::Const(constant) => const_key(constant),
    }
}

fn const_key(constant: &Constant) -> String {
    match constant {
        Constant::Int(v) => format!("i{v}"),
        Constant::Float(v) => format!("f{}", v.to_bits()),
        Constant::Bool(v) => format!("b{}", u8::from(*v)),
    }
}

fn canonical_bin_pair(op: BinOp, lhs: String, rhs: String) -> (String, String) {
    if matches!(op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or) && lhs > rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn rewrite_inst(inst: Inst, replacements: &HashMap<ValueId, ValueId>) -> Inst {
    Inst {
        op: rewrite_op(inst.op, replacements),
        ..inst
    }
}

fn rewrite_op(op: InstOp, replacements: &HashMap<ValueId, ValueId>) -> InstOp {
    let rewrite = |operand: Operand| rewrite_operand(operand, replacements);
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty,
            addr_space,
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: rewrite(base),
            indices: indices.into_iter().map(rewrite).collect(),
        },
        InstOp::Load { ptr } => InstOp::Load { ptr: rewrite(ptr) },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: rewrite(ptr),
            value: rewrite(value),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: rewrite(ptr),
            value: rewrite(value),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: rewrite(dst),
            src: rewrite(src),
            size: rewrite(size),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred,
            lhs: rewrite(lhs),
            rhs: rewrite(rhs),
        },
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op,
            value: rewrite(value),
            to,
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: rewrite(cond),
            on_true: rewrite(on_true),
            on_false: rewrite(on_false),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func,
            args: args.into_iter().map(rewrite).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic,
            args: args.into_iter().map(rewrite).collect(),
        },
    }
}

fn rewrite_terminator(
    terminator: Terminator,
    replacements: &HashMap<ValueId, ValueId>,
) -> Terminator {
    let rewrite = |operand: Operand| rewrite_operand(operand, replacements);
    match terminator {
        Terminator::Br { target, args } => Terminator::Br {
            target,
            args: args.into_iter().map(rewrite).collect(),
        },
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => Terminator::CondBr {
            cond: rewrite(cond),
            true_target,
            true_args: true_args.into_iter().map(rewrite).collect(),
            false_target,
            false_args: false_args.into_iter().map(rewrite).collect(),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: rewrite(value),
            default,
            cases,
        },
        Terminator::Ret { value } => Terminator::Ret {
            value: value.map(rewrite),
        },
    }
}

fn rewrite_operand(operand: Operand, replacements: &HashMap<ValueId, ValueId>) -> Operand {
    match operand {
        Operand::Value(value) => Operand::Value(resolve_replacement(value, replacements)),
        Operand::Const(_) => operand,
    }
}

fn resolve_replacement(value: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut current = value;
    while let Some(next) = replacements.get(&current).copied() {
        if next == current {
            break;
        }
        current = next;
    }
    current
}
