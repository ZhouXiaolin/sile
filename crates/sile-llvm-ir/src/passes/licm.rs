use std::collections::{HashMap, HashSet};

use crate::{
    AddressSpace, BasicBlock, BlockId, Function, Inst, InstOp, Intrinsic, Operand, Terminator,
    Type, ValueId,
};

/// Conservative loop-invariant code motion for structured loops.
///
/// The current implementation only hoists:
/// - Pure scalar ops (`gep`, `bin`, `cmp`, `cast`, `select`, pure intrinsics)
/// - Loads from invariant pointers in constant address space
///
/// This intentionally matches the current Tile IR -> LLVM IR lowering shape,
/// where dynamic shape descriptors and launch-domain math are expressed as
/// explicit scalar SSA inside structured loops.
pub fn run(mut func: Function) -> Function {
    loop {
        let loops = find_structured_loops(&func);
        if loops.is_empty() {
            break;
        }

        let value_types = build_value_type_map(&func);
        let mut changed = false;
        for loop_info in loops {
            changed |= hoist_loop_invariants(&mut func, &value_types, &loop_info);
        }

        if !changed {
            break;
        }
    }

    func
}

#[derive(Clone, Debug)]
struct StructuredLoop {
    preheader: BlockId,
    blocks: Vec<BlockId>,
}

fn find_structured_loops(func: &Function) -> Vec<StructuredLoop> {
    let block_map = block_map(func);
    let mut loops = func
        .blocks
        .iter()
        .filter_map(|block| {
            let Terminator::Br { target: header, .. } = &block.terminator else {
                return None;
            };
            let header_block = block_map.get(header)?;
            let Terminator::CondBr {
                true_target,
                false_target,
                ..
            } = &header_block.terminator
            else {
                return None;
            };

            // Reject backedges: if the candidate preheader is reachable from
            // the loop body it is inside the loop, not before it.
            if block_reaches_target(func, *true_target, block.id, &mut HashSet::new()) {
                return None;
            }

            let mut blocks = vec![*header];
            let mut seen = HashSet::from([*header, block.id]); // exclude preheader from body
            let mut stack = vec![*true_target];
            while let Some(current) = stack.pop() {
                if current == *header || current == *false_target || !seen.insert(current) {
                    continue;
                }
                let Some(current_block) = block_map.get(&current) else {
                    continue;
                };
                blocks.push(current);
                stack.extend(successors(&current_block.terminator));
            }
            Some(StructuredLoop {
                preheader: block.id,
                blocks,
            })
        })
        .collect::<Vec<_>>();

    loops.sort_by_key(|loop_info| loop_info.blocks.len());
    loops
}

fn hoist_loop_invariants(
    func: &mut Function,
    value_types: &HashMap<ValueId, Type>,
    loop_info: &StructuredLoop,
) -> bool {
    let loop_block_set = loop_info.blocks.iter().copied().collect::<HashSet<_>>();
    let loop_defined = loop_defined_values(func, &loop_block_set);
    if loop_defined.is_empty() {
        return false;
    }

    let hoistable = collect_hoistable_values(func, value_types, &loop_block_set, &loop_defined);
    if hoistable.is_empty() {
        return false;
    }

    let mut moved = Vec::new();
    for block in &mut func.blocks {
        if !loop_block_set.contains(&block.id) {
            continue;
        }

        let mut retained = Vec::with_capacity(block.insts.len());
        for inst in block.insts.drain(..) {
            if inst
                .result
                .is_some_and(|result| hoistable.contains(&result))
            {
                moved.push(inst);
            } else {
                retained.push(inst);
            }
        }
        block.insts = retained;
    }

    if moved.is_empty() {
        return false;
    }

    let Some(preheader) = func
        .blocks
        .iter_mut()
        .find(|block| block.id == loop_info.preheader)
    else {
        return false;
    };
    preheader.insts.extend(moved);
    true
}

fn collect_hoistable_values(
    func: &Function,
    value_types: &HashMap<ValueId, Type>,
    loop_block_set: &HashSet<BlockId>,
    loop_defined: &HashSet<ValueId>,
) -> HashSet<ValueId> {
    let mut hoisted = HashSet::new();
    loop {
        let mut changed = false;
        for block in &func.blocks {
            if !loop_block_set.contains(&block.id) {
                continue;
            }
            for inst in &block.insts {
                let Some(result) = inst.result else {
                    continue;
                };
                if hoisted.contains(&result) {
                    continue;
                }
                if is_hoistable_inst(inst, value_types, loop_defined, &hoisted) {
                    hoisted.insert(result);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    hoisted
}

fn is_hoistable_inst(
    inst: &Inst,
    value_types: &HashMap<ValueId, Type>,
    loop_defined: &HashSet<ValueId>,
    hoisted: &HashSet<ValueId>,
) -> bool {
    let operand_is_invariant =
        |operand: &Operand| operand_is_invariant(operand, loop_defined, hoisted);
    match &inst.op {
        InstOp::Gep { base, indices } => {
            operand_is_invariant(base) && indices.iter().all(operand_is_invariant)
        }
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => {
            operand_is_invariant(lhs) && operand_is_invariant(rhs)
        }
        InstOp::Cast { value, .. } => operand_is_invariant(value),
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => {
            operand_is_invariant(cond)
                && operand_is_invariant(on_true)
                && operand_is_invariant(on_false)
        }
        InstOp::Intrinsic { intrinsic, args } => {
            is_pure_intrinsic(intrinsic) && args.iter().all(operand_is_invariant)
        }
        InstOp::Load { ptr } => operand_is_invariant(ptr) && is_constant_pointer(ptr, value_types),
        InstOp::Alloca { .. }
        | InstOp::Store { .. }
        | InstOp::AtomicAdd { .. }
        | InstOp::Memcpy { .. }
        | InstOp::Call { .. } => false,
    }
}

fn operand_is_invariant(
    operand: &Operand,
    loop_defined: &HashSet<ValueId>,
    hoisted: &HashSet<ValueId>,
) -> bool {
    match operand {
        Operand::Const(_) => true,
        Operand::Value(value) => !loop_defined.contains(value) || hoisted.contains(value),
    }
}

fn is_constant_pointer(operand: &Operand, value_types: &HashMap<ValueId, Type>) -> bool {
    let Operand::Value(value) = operand else {
        return false;
    };
    matches!(
        value_types.get(value),
        Some(Type::Ptr {
            addr_space: AddressSpace::Constant,
            ..
        })
    )
}

fn is_pure_intrinsic(intrinsic: &Intrinsic) -> bool {
    !matches!(intrinsic, Intrinsic::Barrier { .. })
}

fn loop_defined_values(func: &Function, loop_block_set: &HashSet<BlockId>) -> HashSet<ValueId> {
    let mut defined = HashSet::new();
    for block in &func.blocks {
        if !loop_block_set.contains(&block.id) {
            continue;
        }
        defined.extend(block.params.iter().map(|param| param.id));
        defined.extend(block.insts.iter().filter_map(|inst| inst.result));
    }
    defined
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
            let mut out = Vec::with_capacity(cases.len() + 1);
            out.push(*default);
            out.extend(cases.iter().map(|(_, target)| *target));
            out
        }
        Terminator::Ret { .. } => Vec::new(),
    }
}

fn block_reaches_target(
    func: &Function,
    start: BlockId,
    goal: BlockId,
    visiting: &mut HashSet<BlockId>,
) -> bool {
    if start == goal {
        return true;
    }
    if !visiting.insert(start) {
        return false;
    }
    let reaches = get_block(func, start)
        .map(|block| successors(&block.terminator))
        .map(|succs| {
            succs
                .into_iter()
                .any(|succ| block_reaches_target(func, succ, goal, visiting))
        })
        .unwrap_or(false);
    visiting.remove(&start);
    reaches
}

fn get_block(func: &Function, id: BlockId) -> Option<&BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn block_map(func: &Function) -> HashMap<BlockId, &BasicBlock> {
    func.blocks.iter().map(|block| (block.id, block)).collect()
}

fn build_value_type_map(func: &Function) -> HashMap<ValueId, Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                types.insert(result, inst.ty.clone());
            }
        }
    }
    types
}

#[cfg(test)]
mod tests {
    use crate::{
        AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, Constant, Function, Inst, InstOp,
        Operand, Param, Terminator, Type, ValueId,
    };

    #[test]
    fn hoists_constant_shape_loads_and_scalar_math_out_of_structured_loop() {
        let func = Function {
            name: "licm_shape".into(),
            params: vec![Param {
                id: ValueId(0),
                name: "__sile_shapes".into(),
                ty: Type::ptr(AddressSpace::Constant, Type::I64),
                abi: None,
            }],
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    name: "preheader".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Br {
                        target: BlockId(1),
                        args: vec![Operand::Const(Constant::Int(0))],
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    name: "header".into(),
                    params: vec![BlockParam {
                        id: ValueId(1),
                        name: "i".into(),
                        ty: Type::I64,
                    }],
                    insts: vec![Inst {
                        result: Some(ValueId(2)),
                        result_name: Some("cond".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: crate::CmpPred::Slt,
                            lhs: Operand::Value(ValueId(1)),
                            rhs: Operand::Const(Constant::Int(4)),
                        },
                        metadata: vec![],
                    }],
                    terminator: Terminator::CondBr {
                        cond: Operand::Value(ValueId(2)),
                        true_target: BlockId(2),
                        true_args: vec![Operand::Value(ValueId(1))],
                        false_target: BlockId(3),
                        false_args: vec![],
                    },
                },
                BasicBlock {
                    id: BlockId(2),
                    name: "body".into(),
                    params: vec![BlockParam {
                        id: ValueId(3),
                        name: "body_i".into(),
                        ty: Type::I64,
                    }],
                    insts: vec![
                        Inst {
                            result: Some(ValueId(4)),
                            result_name: Some("shape1_ptr".into()),
                            ty: Type::ptr(AddressSpace::Constant, Type::I64),
                            op: InstOp::Gep {
                                base: Operand::Value(ValueId(0)),
                                indices: vec![Operand::Const(Constant::Int(1))],
                            },
                            metadata: vec![],
                        },
                        Inst {
                            result: Some(ValueId(5)),
                            result_name: Some("shape1".into()),
                            ty: Type::I64,
                            op: InstOp::Load {
                                ptr: Operand::Value(ValueId(4)),
                            },
                            metadata: vec![],
                        },
                        Inst {
                            result: Some(ValueId(6)),
                            result_name: Some("scaled".into()),
                            ty: Type::I64,
                            op: InstOp::Bin {
                                op: BinOp::Mul,
                                lhs: Operand::Value(ValueId(5)),
                                rhs: Operand::Const(Constant::Int(2)),
                            },
                            metadata: vec![],
                        },
                        Inst {
                            result: Some(ValueId(7)),
                            result_name: Some("next".into()),
                            ty: Type::I64,
                            op: InstOp::Bin {
                                op: BinOp::Add,
                                lhs: Operand::Value(ValueId(3)),
                                rhs: Operand::Const(Constant::Int(1)),
                            },
                            metadata: vec![],
                        },
                    ],
                    terminator: Terminator::Br {
                        target: BlockId(1),
                        args: vec![Operand::Value(ValueId(7))],
                    },
                },
                BasicBlock {
                    id: BlockId(3),
                    name: "exit".into(),
                    params: vec![],
                    insts: vec![],
                    terminator: Terminator::Ret { value: None },
                },
            ],
            entry: BlockId(0),
            metadata: vec![],
        };

        let optimized = super::run(func);
        let preheader = optimized
            .blocks
            .iter()
            .find(|block| block.id == BlockId(0))
            .unwrap();
        let body = optimized
            .blocks
            .iter()
            .find(|block| block.id == BlockId(2))
            .unwrap();

        let hoisted = preheader
            .insts
            .iter()
            .filter_map(|inst| inst.result)
            .collect::<Vec<_>>();
        assert_eq!(hoisted, vec![ValueId(4), ValueId(5), ValueId(6)]);
        assert_eq!(
            body.insts
                .iter()
                .filter_map(|inst| inst.result)
                .collect::<Vec<_>>(),
            vec![ValueId(7)]
        );
    }
}
