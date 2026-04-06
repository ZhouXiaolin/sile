use std::collections::HashSet;

use crate::{Function, InstOp, Operand, Terminator, ValueId};

/// Dead code elimination for side-effect-free LLIR instructions.
pub fn run(mut func: Function) -> Function {
    loop {
        let live = collect_live_values(&func);
        let mut changed = false;

        for block in &mut func.blocks {
            let before = block.insts.len();
            block.insts.retain(|inst| match inst.result {
                Some(result) if !live.contains(&result) => is_side_effecting(&inst.op),
                _ => true,
            });
            changed |= before != block.insts.len();
        }

        if !changed {
            break;
        }
    }

    func
}

fn collect_live_values(func: &Function) -> HashSet<ValueId> {
    let mut live = HashSet::new();
    let mut worklist = Vec::new();

    for block in &func.blocks {
        for operand in terminator_operands(&block.terminator) {
            if let Operand::Value(value) = operand {
                worklist.push(*value);
            }
        }
        for inst in &block.insts {
            if is_side_effecting(&inst.op) {
                for operand in inst_operands(&inst.op) {
                    if let Operand::Value(value) = operand {
                        worklist.push(*value);
                    }
                }
            }
        }
    }

    while let Some(value) = worklist.pop() {
        if !live.insert(value) {
            continue;
        }

        if let Some(def) = find_def_inst(func, value) {
            for operand in inst_operands(&def.op) {
                if let Operand::Value(dep) = operand {
                    worklist.push(*dep);
                }
            }
        }
    }

    live
}

fn find_def_inst(func: &Function, value: ValueId) -> Option<&crate::Inst> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(value))
}

fn is_side_effecting(op: &InstOp) -> bool {
    match op {
        InstOp::Store { .. } | InstOp::Memcpy { .. } | InstOp::Call { .. } => true,
        InstOp::Intrinsic { intrinsic, .. } => {
            matches!(intrinsic, crate::Intrinsic::Barrier { .. })
        }
        _ => false,
    }
}

fn inst_operands(op: &InstOp) -> Vec<&Operand> {
    match op {
        InstOp::ShapeDim { buf, .. } => vec![buf],
        InstOp::Alloca { .. } => Vec::new(),
        InstOp::Gep { base, indices } => {
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(base);
            operands.extend(indices.iter());
            operands
        }
        InstOp::Load { ptr } => vec![ptr],
        InstOp::Store { ptr, value } => vec![ptr, value],
        InstOp::Memcpy { dst, src, size } => vec![dst, src, size],
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
        InstOp::Cast { value, .. } => vec![value],
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond, on_true, on_false],
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => args.iter().collect(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::Br { args, .. } => args.iter().collect(),
        Terminator::CondBr {
            cond,
            true_args,
            false_args,
            ..
        } => {
            let mut operands = Vec::with_capacity(true_args.len() + false_args.len() + 1);
            operands.push(cond);
            operands.extend(true_args.iter());
            operands.extend(false_args.iter());
            operands
        }
        Terminator::Switch { value, .. } => vec![value],
        Terminator::Ret { value } => value.iter().collect(),
    }
}
