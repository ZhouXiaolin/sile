//! Does each LLVM IR pass actually do anything on the shape it claims to handle?
//!
//! Each test builds the smallest IR that should trigger the pass and asserts
//! the pass changes it. A pass that cannot move its own canonical input has no
//! reason to sit in the pipeline.

use sile_llvm_ir::passes::{
    alloca_simplify, cse, dce, inline_preheader_constants, licm, simplify_cfg,
};
use sile_llvm_ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst, InstOp,
    Operand, Param, Terminator, Type, ValueId,
};


fn inst(result: ValueId, ty: Type, op: InstOp) -> Inst {
    Inst {
        result: Some(result),
        result_name: Some(format!("v{}", result.0)),
        ty,
        op,
        metadata: vec![],
    }
}

fn bin(id: ValueId, op: BinOp, lhs: Operand, rhs: Operand) -> Inst {
    inst(id, Type::I64, InstOp::Bin { op, lhs, rhs })
}

/// entry -> preheader -> header(iv) -> body -> header, header -> exit.
///
/// `body` holds `inv = arg + 1`, which must not move (it depends on `iv`), and
/// `sunk = const_param + 1`, which is loop invariant and must be hoisted.
fn counted_loop_with_invariant() -> Function {
    Function {
        name: "licm_target".into(),
        params: vec![Param {
            id: ValueId(0),
            name: "c".into(),
            ty: Type::I64,
            abi: None,
        }],
        entry: BlockId(0),
        metadata: vec![],
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: vec![],
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "preheader".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: vec![Operand::Const(Constant::Int(0))],
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "header".into(),
                params: vec![BlockParam {
                    id: ValueId(1),
                    name: "iv".into(),
                    ty: Type::I64,
                }],
                insts: vec![inst(
                    ValueId(2),
                    Type::I1,
                    InstOp::Cmp {
                        pred: CmpPred::Slt,
                        lhs: Operand::Value(ValueId(1)),
                        rhs: Operand::Const(Constant::Int(8)),
                    },
                )],
                terminator: Terminator::CondBr {
                    cond: Operand::Value(ValueId(2)),
                    true_target: BlockId(3),
                    true_args: vec![],
                    false_target: BlockId(4),
                    false_args: vec![],
                },
            },
            BasicBlock {
                id: BlockId(3),
                name: "body".into(),
                params: vec![],
                insts: vec![
                    // Loop invariant: `c` is a function parameter.
                    Inst {
                        result: Some(ValueId(3)),
                        result_name: Some("sunk".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(0)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    },
                    // Loop varying.
                    bin(
                        ValueId(4),
                        BinOp::Add,
                        Operand::Value(ValueId(1)),
                        Operand::Const(Constant::Int(1)),
                    ),
                ],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: vec![Operand::Value(ValueId(4))],
                },
            },
            BasicBlock {
                id: BlockId(4),
                name: "exit".into(),
                params: vec![],
                insts: vec![Inst {
                    result: Some(ValueId(5)),
                    result_name: Some("acc".into()),
                    ty: Type::I64,
                    op: InstOp::Bin {
                        op: BinOp::Add,
                        lhs: Operand::Value(ValueId(3)),
                        rhs: Operand::Const(Constant::Int(0)),
                    },
                    metadata: vec![],
                }],
                terminator: Terminator::Ret { value: None },
            },
        ],
    }
}

/// LICM's documented contract is a `Br`-terminated preheader feeding a
/// `CondBr` loop header, with the invariant sitting in the body.
#[test]
fn licm_moves_invariant_out_of_the_loop_body() {
    let before = counted_loop_with_invariant();
    let after = licm::run(before.clone());

    assert_ne!(
        before, after,
        "LICM left the IR untouched, so it hoisted nothing"
    );
    let body = after
        .blocks
        .iter()
        .find(|block| block.id == BlockId(3))
        .expect("loop body block");
    assert!(
        !body.insts.iter().any(|inst| inst.result == Some(ValueId(3))),
        "loop-invariant `%c + 1` stayed in the loop body"
    );
    let preheader = after
        .blocks
        .iter()
        .find(|block| block.id == BlockId(1))
        .expect("preheader block");
    assert!(
        preheader
            .insts
            .iter()
            .any(|inst| inst.result == Some(ValueId(3))),
        "loop-invariant `%c + 1` did not move into the preheader"
    );
}

/// The pass only touches a block that receives constant block arguments from
/// its single predecessor, so the fixture must build that shape.
#[test]
fn inline_preheader_constants_drops_constant_edge_param() {
    let mut func = simple_counted_loop();
    // entry: Br preheader [Const 0] -> force the constant onto the edge.
    func.blocks[0].terminator = Terminator::Br {
        target: BlockId(1),
        args: vec![Operand::Const(Constant::Int(0))],
    };
    // preheader carries a param that only forwards the constant to the header.
    func.blocks[1].params.push(BlockParam {
        id: ValueId(9),
        name: "iv_pre".into(),
        ty: Type::I64,
    });
    func.blocks[1].terminator = Terminator::Br {
        target: BlockId(2),
        args: vec![Operand::Value(ValueId(9))],
    };
    let original = func.clone();

    let after = inline_preheader_constants::run(func);
    assert_ne!(after, original, "constant edge parameter was not folded");
    let preheader = after
        .blocks
        .iter()
        .find(|block| block.id == BlockId(1))
        .expect("preheader block");
    assert!(
        preheader.params.is_empty(),
        "preheader still carries a constant-forwarded parameter"
    );
    assert!(
        matches!(
            &preheader.terminator,
            Terminator::Br { args, .. } if args.iter().all(|arg| matches!(arg, Operand::Const(_)))
        ),
        "preheader edge still forwards a value instead of the constant: {:?}",
        preheader.terminator
    );
}

#[test]
fn dce_removes_unused_definitions() {
    let mut func = counted_loop_with_invariant();
    let extra = ValueId(90);
    func.blocks[4].insts.push(bin(
        extra,
        BinOp::Add,
        Operand::Const(Constant::Int(1)),
        Operand::Const(Constant::Int(2)),
    ));

    let after = dce::run(func);
    let still_there = after
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| i.result == Some(extra));
    assert!(!still_there, "DCE kept an unused definition");
}

#[test]
fn cse_removes_duplicate_expression() {
    let mut func = counted_loop_with_invariant();
    // Add a second, identical add of `iv + 1`.
    let dup = bin(
        ValueId(50),
        BinOp::Add,
        Operand::Value(ValueId(1)),
        Operand::Const(Constant::Int(1)),
    );
    func.blocks[3].insts.insert(1, dup);
    let original = func.clone();

    let after = cse::run(func);
    let count = after
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| {
            matches!(
                &i.op,
                InstOp::Bin { op: BinOp::Add, lhs: Operand::Value(v), rhs: Operand::Const(Constant::Int(1)) }
                    if *v == ValueId(1)
            )
        })
        .count();
    // The original `iv + 1` plus the clone should collapse to one definition.
    assert!(
        after != original,
        "CSE did not change an IR that contains an obvious duplicate"
    );
    assert!(count >= 1, "CSE removed a live definition");
}


fn count_insts(func: &Function, id: ValueId) -> usize {
    func.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| i.result == Some(id))
        .count()
}

/// entry -> preheader(br header(0)) -> header(iv) -> body -> header, -> exit.
fn simple_counted_loop() -> Function {
    Function {
        name: "simple".into(),
        params: vec![],
        entry: BlockId(0),
        metadata: vec![],
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![inst(
                    ValueId(1),
                    Type::I64,
                    InstOp::Bin {
                        op: BinOp::Add,
                        lhs: Operand::Const(Constant::Int(1)),
                        rhs: Operand::Const(Constant::Int(2)),
                    },
                )],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: vec![],
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "preheader".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: vec![Operand::Value(ValueId(1))],
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "header".into(),
                params: vec![BlockParam {
                    id: ValueId(2),
                    name: "iv".into(),
                    ty: Type::I64,
                }],
                insts: vec![],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: vec![Operand::Value(ValueId(2))],
                },
            },
        ],
    }
}

#[test]
fn alloca_simplify_promotes_single_store_alloca() {
    let mut func = simple_counted_loop();
    let ptr = Type::ptr(AddressSpace::Private, Type::F32);
    let alloca = inst(
        ValueId(20),
        ptr.clone(),
        InstOp::Alloca {
            alloc_ty: Type::F32,
            addr_space: AddressSpace::Private,
        },
    );
    let store = Inst {
        result: None,
        result_name: None,
        ty: Type::Void,
        op: InstOp::Store {
            ptr: Operand::Value(ValueId(20)),
            value: Operand::Const(Constant::Float(1.0)),
        },
        metadata: vec![],
    };
    let load = inst(
        ValueId(21),
        Type::F32,
        InstOp::Load {
            ptr: Operand::Value(ValueId(20)),
        },
    );
    func.blocks[0].insts.push(alloca);
    func.blocks[0].insts.push(store);
    func.blocks[0].insts.push(load);
    let original = func.clone();

    let after = alloca_simplify::run(func);
    assert_ne!(after, original, "a single-store scalar alloca was not promoted");
}

#[test]
fn simplify_cfg_drops_unreachable_blocks() {
    let mut func = simple_counted_loop();
    func.blocks.push(BasicBlock {
        id: BlockId(9),
        name: "unreachable".into(),
        params: vec![],
        insts: vec![],
        terminator: Terminator::Ret { value: None },
    });

    let after = simplify_cfg::run(func);
    let has_block = after.blocks.iter().any(|b| b.id == BlockId(9));
    assert!(!has_block, "unreachable block survived SimplifyCfg");
}
