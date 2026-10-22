//! Loop-invariant code motion.
//!
//! The lowering computes a kernel's launch base, tile offset and shape strides
//! before a loop, but it also recomputes part of that arithmetic inside the
//! loop body. This pass moves an instruction out of a loop when every operand
//! it reads is already available before the loop. Only instructions that are
//! safe to execute on every path are moved, so the pass cannot introduce a
//! fault that the original program would not have executed.

use std::collections::{HashMap, HashSet};

use crate::passes::cfg::{
    dominators, natural_loops, preheader, reverse_postorder, NaturalLoop,
};
use crate::{BlockId, CastOp, Function, Inst, InstOp, Operand, ValueId};

pub fn run(mut func: Function) -> Function {
    // Hoisting moves an instruction to a block that already exists, so the
    // CFG — and with it the loop set and the dominator relation — is the same
    // before and after. Both are computed once for the whole run.
    let dominators = dominators(&func);
    let mut loops = natural_loops(&func);
    // Innermost first. Hoisting an inner loop's operands to its preheader puts
    // their definitions inside the outer loop, and the outer loop's own sweep
    // reads them there; the reverse order would leave the inner hoists for a
    // second visit the pass no longer makes.
    loops.sort_by(|left, right| left.blocks.len().cmp(&right.blocks.len()));
    for loop_blocks in &loops {
        hoist_loop(&mut func, loop_blocks, &dominators);
    }
    func
}

fn hoist_loop(
    func: &mut Function,
    loop_blocks: &NaturalLoop,
    dominators: &HashMap<BlockId, HashSet<BlockId>>,
) -> bool {
    let Some(preheader_id) = preheader(func, loop_blocks.header, &loop_blocks.blocks) else {
        return false;
    };
    let mut def_block = definitions_by_block(func);

    // A value defined before the preheader is available to every hoisted
    // instruction; a value hoisted out of an inner loop becomes available the
    // moment this sweep passes the instruction that used it.
    let mut invariant: HashSet<ValueId> = HashSet::new();
    for (value, block) in &def_block {
        if !loop_blocks.blocks.contains(block) {
            invariant.insert(*value);
        }
    }

    // One sweep, in reverse postorder over the loop's blocks. Every value an
    // instruction names is defined in a block the sweep reached earlier — a
    // dominator always precedes its dependents in this order — or earlier in
    // the same block. The one exception is a value carried around a back edge,
    // and those travel in block params, which are never hoisted. A single pass
    // therefore sees each definition before every use it can enable.
    let mut order: Vec<BlockId> = reverse_postorder(func)
        .into_iter()
        .filter(|block_id| loop_blocks.blocks.contains(block_id))
        .collect();
    for block in &func.blocks {
        if loop_blocks.blocks.contains(&block.id) && !order.contains(&block.id) {
            order.push(block.id);
        }
    }

    let mut moved_any = false;
    for block_id in order {
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
            if !can_hoist(&inst.op)
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

fn can_hoist(op: &InstOp) -> bool {
    match op {
        InstOp::Bin { op, .. } => {
            // A division or a remainder can fault, so it stays where the
            // program reached it.
            !matches!(op, crate::BinOp::Div | crate::BinOp::Rem)
        }
        InstOp::Cmp { .. } | InstOp::Gep { .. } | InstOp::Select { .. } => true,
        InstOp::Cast { op, .. } => {
            // An out-of-range float-to-int conversion is poison, and moving it
            // where the program never executed it would change the answer.
            !matches!(op, CastOp::Fptosi)
        }
        _ => false,
    }
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
