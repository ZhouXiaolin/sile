use std::collections::{HashMap, HashSet};

use crate::{BlockId, Function, InstOp, Operand, Terminator, ValueId};

pub fn verify_function(func: &Function) -> Result<(), String> {
    let mut block_ids = HashSet::new();
    let mut block_params = HashMap::new();
    let mut value_defs = HashSet::new();
    let mut value_def_sites = HashMap::new();

    for param in &func.params {
        if !value_defs.insert(param.id) {
            return Err(format!("duplicate function param value id {}", param.id.0));
        }
        value_def_sites.insert(param.id, DefSite::FunctionParam);
    }

    for block in &func.blocks {
        if !block_ids.insert(block.id) {
            return Err(format!("duplicate block id {}", block.id.0));
        }
        block_params.insert(block.id, block.params.len());

        for param in &block.params {
            if !value_defs.insert(param.id) {
                return Err(format!("duplicate block param value id {}", param.id.0));
            }
            value_def_sites.insert(param.id, DefSite::BlockParam { block: block.id });
        }
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result {
                if !value_defs.insert(result) {
                    return Err(format!(
                        "duplicate instruction result value id {}",
                        result.0
                    ));
                }
                value_def_sites.insert(
                    result,
                    DefSite::Inst {
                        block: block.id,
                        inst_idx,
                    },
                );
            }
        }
    }

    if !block_ids.contains(&func.entry) {
        return Err(format!("entry block {} does not exist", func.entry.0));
    }

    for block in &func.blocks {
        verify_block_uses(block, &value_defs, &value_def_sites)?;
        verify_terminator(
            &block.terminator,
            block.id,
            &block_params,
            &value_defs,
            &value_def_sites,
        )?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefSite {
    FunctionParam,
    BlockParam { block: BlockId },
    Inst { block: BlockId, inst_idx: usize },
}

fn verify_block_uses(
    block: &crate::BasicBlock,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
) -> Result<(), String> {
    for (inst_idx, inst) in block.insts.iter().enumerate() {
        for operand in inst_operands(&inst.op) {
            verify_operand(
                operand,
                block.id,
                Some(inst_idx),
                value_defs,
                value_def_sites,
            )?;
        }
    }
    Ok(())
}

fn verify_terminator(
    terminator: &Terminator,
    block_id: BlockId,
    block_params: &HashMap<BlockId, usize>,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
) -> Result<(), String> {
    match terminator {
        Terminator::Br { target, args } => {
            verify_branch_target(*target, args.len(), block_params)?;
            for operand in args {
                verify_operand(operand, block_id, None, value_defs, value_def_sites)?;
            }
            Ok(())
        }
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            verify_branch_target(*true_target, true_args.len(), block_params)?;
            verify_branch_target(*false_target, false_args.len(), block_params)?;
            verify_operand(cond, block_id, None, value_defs, value_def_sites)?;
            for operand in true_args.iter().chain(false_args.iter()) {
                verify_operand(operand, block_id, None, value_defs, value_def_sites)?;
            }
            Ok(())
        }
        Terminator::Switch {
            value,
            default,
            cases,
        } => {
            verify_branch_target(*default, 0, block_params)?;
            for (_, target) in cases {
                verify_branch_target(*target, 0, block_params)?;
            }
            verify_operand(value, block_id, None, value_defs, value_def_sites)?;
            Ok(())
        }
        Terminator::Ret { value } => {
            if let Some(operand) = value {
                verify_operand(operand, block_id, None, value_defs, value_def_sites)?;
            }
            Ok(())
        }
    }
}

fn verify_operand(
    operand: &Operand,
    use_block: BlockId,
    use_inst_idx: Option<usize>,
    value_defs: &HashSet<ValueId>,
    value_def_sites: &HashMap<ValueId, DefSite>,
) -> Result<(), String> {
    let Operand::Value(value) = operand else {
        return Ok(());
    };

    if !value_defs.contains(value) {
        return Err(format!("use of undefined value id {}", value.0));
    }

    let Some(def_site) = value_def_sites.get(value).copied() else {
        return Err(format!("missing def-site for value id {}", value.0));
    };

    if let (
        DefSite::Inst {
            block: def_block,
            inst_idx: def_inst_idx,
        },
        Some(use_inst_idx),
    ) = (def_site, use_inst_idx)
    {
        if def_block == use_block && def_inst_idx >= use_inst_idx {
            return Err(format!(
                "value id {} is used before its definition in block {}",
                value.0, use_block.0
            ));
        }
    }

    Ok(())
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

fn verify_branch_target(
    target: BlockId,
    arg_count: usize,
    block_params: &HashMap<BlockId, usize>,
) -> Result<(), String> {
    let expected = block_params
        .get(&target)
        .ok_or_else(|| format!("branch target {} does not exist", target.0))?;
    if *expected != arg_count {
        return Err(format!(
            "branch target {} expects {} args but got {}",
            target.0, expected, arg_count
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressSpace, BasicBlock, BlockParam, CmpPred, Constant, Inst, Param, Type};

    #[test]
    fn rejects_undefined_value_use() {
        let func = Function {
            name: "bad_use".into(),
            params: vec![param(1)],
            blocks: vec![BasicBlock {
                id: BlockId(0),
                name: "bb0".into(),
                params: Vec::new(),
                insts: vec![Inst {
                    result: Some(ValueId(2)),
                    result_name: Some("v2".into()),
                    ty: Type::F32,
                    op: InstOp::Load {
                        ptr: Operand::Value(ValueId(999)),
                    },
                    metadata: Vec::new(),
                }],
                terminator: Terminator::Ret { value: None },
            }],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let err = verify_function(&func).unwrap_err();
        assert!(err.contains("undefined value id 999"));
    }

    #[test]
    fn rejects_same_block_use_before_def() {
        let func = Function {
            name: "forward_ref".into(),
            params: vec![param(1)],
            blocks: vec![BasicBlock {
                id: BlockId(0),
                name: "bb0".into(),
                params: vec![block_param(2)],
                insts: vec![
                    Inst {
                        result: Some(ValueId(3)),
                        result_name: Some("v3".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: CmpPred::Eq,
                            lhs: Operand::Value(ValueId(4)),
                            rhs: Operand::Const(Constant::Int(0)),
                        },
                        metadata: Vec::new(),
                    },
                    Inst {
                        result: Some(ValueId(4)),
                        result_name: Some("v4".into()),
                        ty: Type::I64,
                        op: InstOp::ShapeDim {
                            buf: Operand::Value(ValueId(1)),
                            dim: 0,
                        },
                        metadata: Vec::new(),
                    },
                ],
                terminator: Terminator::Ret { value: None },
            }],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        let err = verify_function(&func).unwrap_err();
        assert!(err.contains("used before its definition"));
    }

    #[test]
    fn accepts_well_formed_function() {
        let func = Function {
            name: "ok".into(),
            params: vec![param(1)],
            blocks: vec![BasicBlock {
                id: BlockId(0),
                name: "bb0".into(),
                params: vec![block_param(2)],
                insts: vec![
                    Inst {
                        result: Some(ValueId(3)),
                        result_name: Some("v3".into()),
                        ty: Type::I64,
                        op: InstOp::ShapeDim {
                            buf: Operand::Value(ValueId(1)),
                            dim: 0,
                        },
                        metadata: Vec::new(),
                    },
                    Inst {
                        result: Some(ValueId(4)),
                        result_name: Some("v4".into()),
                        ty: Type::ptr(AddressSpace::Private, Type::F32),
                        op: InstOp::Alloca {
                            alloc_ty: Type::F32,
                            addr_space: AddressSpace::Private,
                        },
                        metadata: Vec::new(),
                    },
                    Inst {
                        result: None,
                        result_name: None,
                        ty: Type::Void,
                        op: InstOp::Store {
                            ptr: Operand::Value(ValueId(4)),
                            value: Operand::Const(Constant::Float(1.0)),
                        },
                        metadata: Vec::new(),
                    },
                ],
                terminator: Terminator::Ret {
                    value: Some(Operand::Value(ValueId(3))),
                },
            }],
            entry: BlockId(0),
            metadata: Vec::new(),
        };

        verify_function(&func).unwrap();
    }

    fn param(id: u32) -> Param {
        Param {
            id: ValueId(id),
            name: format!("p{id}"),
            ty: Type::ptr(AddressSpace::Global, Type::F32),
            abi: None,
        }
    }

    fn block_param(id: u32) -> BlockParam {
        BlockParam {
            id: ValueId(id),
            name: format!("b{id}"),
            ty: Type::I64,
        }
    }
}
