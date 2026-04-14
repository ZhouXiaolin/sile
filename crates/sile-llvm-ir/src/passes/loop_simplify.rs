use std::collections::{HashMap, HashSet};

use crate::{
    BasicBlock, BlockId, BlockParam, Constant, Function, Metadata, Operand, Terminator, ValueId,
};

/// Conservative loop-oriented canonicalization.
///
/// Current behavior:
/// - Fold degenerate conditional branches (`condbr` with identical targets or
///   compile-time boolean condition) into plain branches.
/// - Remove duplicate loop metadata (`Parallel`, `Reduction`, `Unroll`, etc.)
///   so later passes/codegen can reason on a normalized annotation set.
/// - Insert preheader blocks for loop headers whose only non-backedge entry
///   comes from a `CondBr`, enabling LICM to recognize the loop.
pub fn run(mut func: Function) -> Function {
    for block in &mut func.blocks {
        block.terminator = simplify_terminator(block.terminator.clone());
        dedup_metadata(&mut block.insts);
    }
    dedup_function_metadata(&mut func.metadata);
    insert_preheaders(&mut func);
    func
}

fn simplify_terminator(terminator: Terminator) -> Terminator {
    match terminator {
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            if true_target == false_target {
                return Terminator::Br {
                    target: true_target,
                    args: true_args,
                };
            }

            if let Operand::Const(Constant::Bool(flag)) = cond {
                if flag {
                    return Terminator::Br {
                        target: true_target,
                        args: true_args,
                    };
                }
                return Terminator::Br {
                    target: false_target,
                    args: false_args,
                };
            }

            Terminator::CondBr {
                cond,
                true_target,
                true_args,
                false_target,
                false_args,
            }
        }
        other => other,
    }
}

fn dedup_metadata(insts: &mut [crate::Inst]) {
    for inst in insts {
        let mut seen = HashSet::<MetadataKey>::new();
        inst.metadata
            .retain(|meta| seen.insert(MetadataKey::from(meta)));
    }
}

fn dedup_function_metadata(metadata: &mut Vec<Metadata>) {
    let mut seen = HashSet::<MetadataKey>::new();
    metadata.retain(|meta| seen.insert(MetadataKey::from(meta)));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MetadataKey {
    Parallel,
    Reduction,
    VectorizeWidth(u32),
    Unroll(u32),
    Alignment(u32),
    NoAlias,
    ReadOnly,
    WriteOnly,
}

impl From<&Metadata> for MetadataKey {
    fn from(value: &Metadata) -> Self {
        match value {
            Metadata::Parallel => Self::Parallel,
            Metadata::Reduction => Self::Reduction,
            Metadata::VectorizeWidth(width) => Self::VectorizeWidth(*width),
            Metadata::Unroll(width) => Self::Unroll(*width),
            Metadata::Alignment(align) => Self::Alignment(*align),
            Metadata::NoAlias => Self::NoAlias,
            Metadata::ReadOnly => Self::ReadOnly,
            Metadata::WriteOnly => Self::WriteOnly,
        }
    }
}

fn successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        Terminator::Switch { default, cases, .. } => {
            let mut s = vec![*default];
            s.extend(cases.iter().map(|(_, b)| *b));
            s
        }
        Terminator::Ret { .. } => vec![],
    }
}

/// Insert a `Br`-terminated preheader block between each CondBr predecessor
/// and a loop header, so that LICM's `find_structured_loops` can recognise
/// the loop.
fn insert_preheaders(func: &mut Function) {
    // Build predecessor map
    let mut predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    let block_map: HashMap<BlockId, &BasicBlock> = func.blocks.iter().map(|b| (b.id, b)).collect();
    for block in &func.blocks {
        for succ in successors(&block.terminator) {
            predecessors.entry(succ).or_default().push(block.id);
        }
    }

    let mut next_block_id = func.blocks.iter().map(|b| b.id.0).max().unwrap_or(0) + 1;
    let mut next_value_id = func
        .blocks
        .iter()
        .flat_map(|b| {
            b.insts
                .iter()
                .filter_map(|i| i.result)
                .chain(b.params.iter().map(|p| p.id))
        })
        .map(|v| v.0)
        .max()
        .unwrap_or(0)
        + 1;

    let mut new_blocks: Vec<BasicBlock> = Vec::new();
    // Collect (cond_br_block_id, header_id, new_preheader_id) triples.
    let mut redirects: Vec<(BlockId, BlockId, BlockId)> = Vec::new();

    for block in &func.blocks {
        let Some(preds) = predecessors.get(&block.id) else {
            continue;
        };
        if preds.len() < 2 {
            continue;
        }

        // Check for at least one backedge (Br targeting this block from a block
        // that is also reachable from this block).
        let has_backedge = preds.iter().any(|&pred_id| {
            let Some(pred) = block_map.get(&pred_id) else {
                return false;
            };
            matches!(pred.terminator, Terminator::Br { .. })
                && block_reaches(func, block.id, pred_id, &mut HashSet::new())
        });
        if !has_backedge {
            continue;
        }

        // Find CondBr predecessors that target this header.
        for &pred_id in preds {
            let Some(pred) = block_map.get(&pred_id) else {
                continue;
            };
            let Terminator::CondBr {
                true_target,
                false_target,
                ..
            } = &pred.terminator
            else {
                continue;
            };
            let targets_header =
                (*true_target == block.id) as usize + (*false_target == block.id) as usize;
            if targets_header != 1 {
                continue;
            }

            let preheader_id = BlockId(next_block_id);
            next_block_id += 1;

            // Create preheader params matching header params.
            let preheader_params: Vec<BlockParam> = block
                .params
                .iter()
                .map(|p| {
                    let new_id = ValueId(next_value_id);
                    next_value_id += 1;
                    BlockParam {
                        id: new_id,
                        name: format!("{}_pre", p.name),
                        ty: p.ty.clone(),
                    }
                })
                .collect();

            let args: Vec<Operand> = preheader_params
                .iter()
                .map(|p| Operand::Value(p.id))
                .collect();

            new_blocks.push(BasicBlock {
                id: preheader_id,
                name: format!("{}_preheader", block.name),
                params: preheader_params,
                insts: vec![],
                terminator: Terminator::Br {
                    target: block.id,
                    args,
                },
            });

            redirects.push((pred_id, block.id, preheader_id));
        }
    }

    // Apply redirects to existing CondBr terminators.
    for (cond_br_id, old_header, new_preheader) in &redirects {
        let block = func
            .blocks
            .iter_mut()
            .find(|b| b.id == *cond_br_id)
            .unwrap();
        match &mut block.terminator {
            Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => {
                if *true_target == *old_header {
                    *true_target = *new_preheader;
                } else if *false_target == *old_header {
                    *false_target = *new_preheader;
                }
            }
            _ => unreachable!(),
        }
    }

    func.blocks.extend(new_blocks);
}

fn block_reaches(
    func: &Function,
    from: BlockId,
    target: BlockId,
    visited: &mut HashSet<BlockId>,
) -> bool {
    if from == target {
        return true;
    }
    if !visited.insert(from) {
        return false;
    }
    let block_map: HashMap<BlockId, &BasicBlock> = func.blocks.iter().map(|b| (b.id, b)).collect();
    let Some(block) = block_map.get(&from) else {
        return false;
    };
    for succ in successors(&block.terminator) {
        if block_reaches(func, succ, target, visited) {
            return true;
        }
    }
    false
}
