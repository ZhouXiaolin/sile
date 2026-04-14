use std::collections::HashMap;

use crate::{Function, InstOp, Operand, ValueId};

/// Eliminate allocas that are only used by stores and loads with the alloca
/// as the *direct* pointer (no intervening GEP).
///
/// Pattern matched per alloca:
///   alloca a
///   store a, value        (single store)
///   ... load a ...        (one or more loads)
///
/// All loads from `a` are replaced by the stored `value`.
pub fn run(mut func: Function) -> Function {
    let mut replacements: HashMap<ValueId, Operand> = HashMap::new();

    for alloca_block in &func.blocks {
        for alloca_inst in &alloca_block.insts {
            let Some(alloca_result) = alloca_inst.result else {
                continue;
            };
            if !matches!(alloca_inst.op, InstOp::Alloca { .. }) {
                continue;
            }

            // Find the single store and all direct loads of this alloca.
            let mut store_value: Option<Operand> = None;
            let mut store_block: Option<usize> = None;
            let mut store_idx: Option<usize> = None;
            let mut load_results: Vec<(usize, usize, ValueId)> = Vec::new(); // (block_idx, inst_idx, result)
            let mut too_complex = false;

            for (block_idx, block) in func.blocks.iter().enumerate() {
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    match &inst.op {
                        InstOp::Store {
                            ptr: Operand::Value(ptr),
                            value,
                        } if *ptr == alloca_result => {
                            if store_value.is_some() {
                                too_complex = true; // multiple stores
                                break;
                            }
                            store_value = Some(value.clone());
                            store_block = Some(block_idx);
                            store_idx = Some(inst_idx);
                        }
                        InstOp::Load {
                            ptr: Operand::Value(ptr),
                        } if *ptr == alloca_result => {
                            if let Some(result) = inst.result {
                                load_results.push((block_idx, inst_idx, result));
                            }
                        }
                        _ => {
                            // If alloca appears as any other operand (e.g. GEP base),
                            // this alloca is too complex to simplify.
                            if uses_value(&inst.op, alloca_result) {
                                too_complex = true;
                                break;
                            }
                        }
                    }
                }
                if too_complex {
                    break;
                }
            }

            if too_complex || store_value.is_none() || load_results.is_empty() {
                continue;
            }

            let sv = store_value.unwrap();
            let sb = store_block.unwrap();
            let si = store_idx.unwrap();

            for (lb, li, lr) in &load_results {
                let dominates = if *lb == sb { si < *li } else { sb < *lb };
                if dominates {
                    replacements.insert(*lr, sv.clone());
                }
            }
        }
    }

    if replacements.is_empty() {
        return func;
    }

    for block in &mut func.blocks {
        block.insts = std::mem::take(&mut block.insts)
            .into_iter()
            .map(|inst| crate::Inst {
                op: remap_inst_op(inst.op.clone(), &replacements),
                ..inst
            })
            .filter(|inst| inst.result.map_or(true, |r| !replacements.contains_key(&r)))
            .collect();

        block.terminator = remap_terminator(block.terminator.clone(), &replacements);
    }

    crate::passes::dce::run(func)
}

fn uses_value(op: &InstOp, id: ValueId) -> bool {
    match op {
        InstOp::Alloca { .. } => false,
        InstOp::Gep { base, indices } => {
            matches!(base, Operand::Value(v) if *v == id)
                || indices
                    .iter()
                    .any(|o| matches!(o, Operand::Value(v) if *v == id))
        }
        InstOp::Load { ptr } => matches!(ptr, Operand::Value(v) if *v == id),
        InstOp::Store { ptr, value } => {
            matches!(ptr, Operand::Value(v) if *v == id)
                || matches!(value, Operand::Value(v) if *v == id)
        }
        InstOp::AtomicAdd { ptr, value } => {
            matches!(ptr, Operand::Value(v) if *v == id)
                || matches!(value, Operand::Value(v) if *v == id)
        }
        InstOp::Memcpy { dst, src, size } => {
            matches!(dst, Operand::Value(v) if *v == id)
                || matches!(src, Operand::Value(v) if *v == id)
                || matches!(size, Operand::Value(v) if *v == id)
        }
        InstOp::Bin { lhs, rhs, .. } => {
            matches!(lhs, Operand::Value(v) if *v == id)
                || matches!(rhs, Operand::Value(v) if *v == id)
        }
        InstOp::Cmp { lhs, rhs, .. } => {
            matches!(lhs, Operand::Value(v) if *v == id)
                || matches!(rhs, Operand::Value(v) if *v == id)
        }
        InstOp::Cast { value, .. } => matches!(value, Operand::Value(v) if *v == id),
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => {
            matches!(cond, Operand::Value(v) if *v == id)
                || matches!(on_true, Operand::Value(v) if *v == id)
                || matches!(on_false, Operand::Value(v) if *v == id)
        }
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => args
            .iter()
            .any(|o| matches!(o, Operand::Value(v) if *v == id)),
    }
}

fn remap_inst_op(op: InstOp, replacements: &HashMap<ValueId, Operand>) -> InstOp {
    match op {
        InstOp::Alloca {
            alloc_ty,
            addr_space,
        } => InstOp::Alloca {
            alloc_ty,
            addr_space,
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: remap(base, replacements),
            indices: indices
                .into_iter()
                .map(|o| remap(o, replacements))
                .collect(),
        },
        InstOp::Load { ptr } => InstOp::Load {
            ptr: remap(ptr, replacements),
        },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: remap(ptr, replacements),
            value: remap(value, replacements),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: remap(ptr, replacements),
            value: remap(value, replacements),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: remap(dst, replacements),
            src: remap(src, replacements),
            size: remap(size, replacements),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op,
            lhs: remap(lhs, replacements),
            rhs: remap(rhs, replacements),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred,
            lhs: remap(lhs, replacements),
            rhs: remap(rhs, replacements),
        },
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op,
            value: remap(value, replacements),
            to,
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: remap(cond, replacements),
            on_true: remap(on_true, replacements),
            on_false: remap(on_false, replacements),
        },
        InstOp::Call { func: f, args } => InstOp::Call {
            func: f,
            args: args.into_iter().map(|o| remap(o, replacements)).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic,
            args: args.into_iter().map(|o| remap(o, replacements)).collect(),
        },
    }
}

fn remap_terminator(
    terminator: crate::Terminator,
    replacements: &HashMap<ValueId, Operand>,
) -> crate::Terminator {
    use crate::Terminator;
    match terminator {
        Terminator::Br { target, args } => Terminator::Br {
            target,
            args: args.into_iter().map(|o| remap(o, replacements)).collect(),
        },
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => Terminator::CondBr {
            cond: remap(cond, replacements),
            true_target,
            true_args: true_args
                .into_iter()
                .map(|o| remap(o, replacements))
                .collect(),
            false_target,
            false_args: false_args
                .into_iter()
                .map(|o| remap(o, replacements))
                .collect(),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: remap(value, replacements),
            default,
            cases,
        },
        Terminator::Ret { value } => Terminator::Ret {
            value: value.map(|o| remap(o, replacements)),
        },
    }
}

fn remap(operand: Operand, replacements: &HashMap<ValueId, Operand>) -> Operand {
    match operand {
        Operand::Value(id) => replacements.get(&id).cloned().unwrap_or(Operand::Value(id)),
        Operand::Const(_) => operand,
    }
}
