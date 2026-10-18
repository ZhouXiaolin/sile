//! Loop-invariant code motion.
//!
//! The lowering computes a kernel's launch base, tile offset and shape strides
//! before a loop, but it also recomputes part of that arithmetic inside the
//! loop body. This pass moves an instruction out of a loop when every operand
//! it reads is already available before the loop. Only instructions that are
//! safe to execute on every path are moved, so the pass cannot introduce a
//! fault that the original program would not have executed.

use std::collections::{HashMap, HashSet};

use crate::passes::cfg::{dominators, natural_loops, preheader, NaturalLoop};
use crate::{BlockId, CastOp, Function, Inst, InstOp, Operand, ValueId};

pub fn run(mut func: Function) -> Function {
    loop {
        let loops = natural_loops(&func);
        let mut moved = false;
        for loop_blocks in loops {
            if hoist_loop(&mut func, &loop_blocks) {
                moved = true;
                break;
            }
        }
        if !moved {
            break;
        }
    }
    func
}

fn hoist_loop(func: &mut Function, loop_blocks: &NaturalLoop) -> bool {
    let Some(preheader_id) = preheader(func, loop_blocks.header, &loop_blocks.blocks) else {
        return false;
    };
    let dominators = dominators(func);
    let mut def_block = definitions_by_block(func);

    // A value defined before the preheader is available to every hoisted
    // instruction; a value defined inside the loop becomes available only once
    // its own instruction has been hoisted, which this loop records.
    let mut invariant: HashSet<ValueId> = HashSet::new();
    for (value, block) in &def_block {
        if !loop_blocks.blocks.contains(block) {
            invariant.insert(*value);
        }
    }

    let mut moved_any = false;
    loop {
        let mut moved_this_round = false;
        let block_ids: Vec<BlockId> = func
            .blocks
            .iter()
            .filter(|block| loop_blocks.blocks.contains(&block.id))
            .map(|block| block.id)
            .collect();
        for block_id in block_ids {
            let Some(block_index) = func.blocks.iter().position(|block| block.id == block_id) else {
                continue;
            };
            let mut keep = Vec::with_capacity(func.blocks[block_index].insts.len());
            let mut hoisted: Vec<Inst> = Vec::new();
            for inst in func.blocks[block_index].insts.drain(..) {
                let Some(result) = inst.result else {
                    keep.push(inst);
                    continue;
                };
                if !can_hoist(&inst.op, &invariant)
                    || !operands_available(
                        &inst.op,
                        preheader_id,
                        &def_block,
                        &dominators,
                    )
                {
                    keep.push(inst);
                    continue;
                }
                invariant.insert(result);
                def_block.insert(result, preheader_id);
                hoisted.push(inst);
            }
            func.blocks[block_index].insts = keep;
            if !hoisted.is_empty() {
                moved_this_round = true;
                moved_any = true;
                if let Some(preheader_block) = func
                    .blocks
                    .iter_mut()
                    .find(|block| block.id == preheader_id)
                {
                    preheader_block.insts.extend(hoisted);
                }
            }
        }
        if !moved_this_round {
            break;
        }
    }
    moved_any
}

/// The block that defines every value the function defines.
fn definitions_by_block(func: &Function) -> HashMap<ValueId, BlockId> {
    let mut definitions = HashMap::new();
    for param in &func.params {
        definitions.insert(param.id, func.entry);
    }
    for block in &func.blocks {
        for param in &block.params {
            definitions.insert(param.id, block.id);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                definitions.insert(result, block.id);
            }
        }
    }
    definitions
}

fn can_hoist(op: &InstOp, invariant: &HashSet<ValueId>) -> bool {
    match op {
        InstOp::Bin { op, lhs, rhs } => {
            // A division or a remainder can fault, so it stays where the
            // program reached it.
            !matches!(op, crate::BinOp::Div | crate::BinOp::Rem)
                && operands_invariant(&[lhs, rhs], invariant)
        }
        InstOp::Cmp { lhs, rhs, .. } => operands_invariant(&[lhs, rhs], invariant),
        InstOp::Gep { base, indices } => {
            let mut operands = vec![base];
            operands.extend(indices.iter());
            operands_invariant(&operands, invariant)
        }
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => operands_invariant(&[cond, on_true, on_false], invariant),
        InstOp::Cast { op, value, .. } => {
            // An out-of-range float-to-int conversion is poison, and moving it
            // where the program never executed it would change the answer.
            !matches!(op, CastOp::Fptosi)
                && operands_invariant(&[value], invariant)
        }
        _ => false,
    }
}

fn operands_invariant(operands: &[&Operand], invariant: &HashSet<ValueId>) -> bool {
    operands.iter().all(|operand| match operand {
        Operand::Const(_) => true,
        Operand::Value(value) => invariant.contains(value),
    })
}

/// Whether every value an instruction reads is defined before the preheader.
///
/// A value defined outside the loop but used inside it dominates the loop
/// header, hence the preheader; the check is still explicit because this pass
/// also reads the IR before a verifier has confirmed it.
fn operands_available(
    op: &InstOp,
    preheader: BlockId,
    def_block: &HashMap<ValueId, BlockId>,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> bool {
    let mut operands = Vec::new();
    collect_operands(op, &mut operands);
    operands.iter().all(|operand| match operand {
        Operand::Const(_) => true,
        Operand::Value(value) => {
            let Some(block) = def_block.get(value) else {
                return false;
            };
            *block == preheader
                || dominators
                    .get(&preheader)
                    .is_some_and(|preheader_dominators| preheader_dominators.contains(block))
        }
    })
}

fn collect_operands(op: &InstOp, operands: &mut Vec<Operand>) {
    match op {
        InstOp::Alloca { .. } => {}
        InstOp::Gep { base, indices } => {
            operands.push(base.clone());
            operands.extend(indices.iter().cloned());
        }
        InstOp::Load { ptr } => operands.push(ptr.clone()),
        InstOp::Store { ptr, value } => {
            operands.push(ptr.clone());
            operands.push(value.clone());
        }
        InstOp::AtomicAdd { ptr, value } => {
            operands.push(ptr.clone());
            operands.push(value.clone());
        }
        InstOp::Memcpy { dst, src, size } => {
            operands.push(dst.clone());
            operands.push(src.clone());
            operands.push(size.clone());
        }
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => {
            operands.push(lhs.clone());
            operands.push(rhs.clone());
        }
        InstOp::Cast { value, .. } => operands.push(value.clone()),
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => {
            operands.push(cond.clone());
            operands.push(on_true.clone());
            operands.push(on_false.clone());
        }
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => {
            operands.extend(args.iter().cloned());
        }
    }
}
