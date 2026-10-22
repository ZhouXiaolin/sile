//! Scalar promotion: a loop's induction slot leaves private memory and becomes
//! the header's block parameter.

use sile_lsir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst,
    InstOp, Operand, Terminator, Type, ValueId,
};

fn inst(result: u32, ty: Type, op: InstOp) -> Inst {
    Inst {
        result: Some(ValueId(result)),
        result_name: None,
        ty,
        op,
        metadata: Vec::new(),
    }
}

fn private_ptr() -> Type {
    Type::ptr(AddressSpace::Private, Type::I64)
}

/// `entry: i = 0; br h   h: if (i < n) b else x   b: i = i + 1; br h   x: ret`
fn counted_loop() -> Function {
    Function {
        name: "loop".into(),
        params: Vec::new(),
        entry: BlockId(0),
        metadata: Vec::new(),
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: Vec::new(),
                insts: vec![
                    inst(
                        0,
                        private_ptr(),
                        InstOp::Alloca {
                            alloc_ty: Type::I64,
                            addr_space: AddressSpace::Private,
                        },
                    ),
                    inst(
                        1,
                        Type::Void,
                        InstOp::Store {
                            ptr: Operand::Value(ValueId(0)),
                            value: Operand::Const(Constant::Int(0)),
                        },
                    ),
                ],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "header".into(),
                params: Vec::new(),
                insts: vec![
                    inst(
                        2,
                        Type::I64,
                        InstOp::Load {
                            ptr: Operand::Value(ValueId(0)),
                        },
                    ),
                    inst(
                        3,
                        Type::I1,
                        InstOp::Cmp {
                            pred: CmpPred::Slt,
                            lhs: Operand::Value(ValueId(2)),
                            rhs: Operand::Const(Constant::Int(4)),
                        },
                    ),
                ],
                terminator: Terminator::CondBr {
                    cond: Operand::Value(ValueId(3)),
                    true_target: BlockId(2),
                    true_args: Vec::new(),
                    false_target: BlockId(3),
                    false_args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "body".into(),
                params: Vec::new(),
                insts: vec![
                    inst(
                        4,
                        Type::I64,
                        InstOp::Load {
                            ptr: Operand::Value(ValueId(0)),
                        },
                    ),
                    inst(
                        5,
                        Type::I64,
                        InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(4)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                    ),
                    inst(
                        6,
                        Type::Void,
                        InstOp::Store {
                            ptr: Operand::Value(ValueId(0)),
                            value: Operand::Value(ValueId(5)),
                        },
                    ),
                ],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(3),
                name: "exit".into(),
                params: Vec::new(),
                insts: Vec::new(),
                terminator: Terminator::Ret { value: None },
            },
        ],
    }
}

#[test]
fn induction_slot_becomes_a_block_parameter() {
    let out = sile_lsir::run_lsir_pipeline(
        counted_loop(),
        &[sile_lsir::LsirPassKind::ScalarPromote],
    )
    .expect("the promoted function verifies");

    let insts: Vec<&Inst> = out.blocks.iter().flat_map(|block| block.insts.iter()).collect();
    assert!(
        !insts
            .iter()
            .any(|i| matches!(i.op, InstOp::Alloca { .. })),
        "the slot allocation must be gone"
    );
    assert!(
        !insts.iter().any(|i| matches!(
            &i.op,
            InstOp::Load { ptr: Operand::Value(_) } | InstOp::Store { ptr: Operand::Value(_), .. }
        )),
        "no private load or store may remain"
    );

    let header = out
        .blocks
        .iter()
        .find(|block| block.id == BlockId(1))
        .expect("the header block survives");
    assert_eq!(header.params.len(), 1, "the header carries the index");
    assert_eq!(header.params[0].ty, Type::I64);

    // The entry passes the init and the latch passes the step.
    let entry = out
        .blocks
        .iter()
        .find(|block| block.id == BlockId(0))
        .expect("the entry block survives");
    match &entry.terminator {
        Terminator::Br { args, .. } => {
            assert_eq!(args, &[Operand::Const(Constant::Int(0))], "the init is 0")
        }
        other => panic!("the entry must branch to the header, got {other:?}"),
    }
    let body = out
        .blocks
        .iter()
        .find(|block| block.id == BlockId(2))
        .expect("the body block survives");
    match &body.terminator {
        Terminator::Br { args, .. } => {
            assert_eq!(args.len(), 1);
            // The back edge carries the step, not a stored value.
            assert!(matches!(args[0], Operand::Value(_)), "the step is a value");
        }
        other => panic!("the latch must branch to the header, got {other:?}"),
    }
}

#[test]
fn a_slot_used_as_a_gep_is_left_in_memory() {
    let mut func = counted_loop();
    // Indexing the slot makes it an aggregate: the pass cannot follow that.
    func.blocks[2].insts.push(inst(
        7,
        Type::ptr(AddressSpace::Private, Type::I64),
        InstOp::Gep {
            base: Operand::Value(ValueId(0)),
            indices: vec![Operand::Const(Constant::Int(0))],
        },
    ));
    let before = func.clone();
    let out = sile_lsir::run_lsir_pipeline(func, &[sile_lsir::LsirPassKind::ScalarPromote])
        .expect("an unpromoted function is still well formed");
    assert_eq!(out, before, "the function must be unchanged");
}

#[test]
fn a_loop_carried_scalar_merges_on_the_back_edge() {
    // A second slot written in one arm of a branch and read after the join:
    // the join needs a parameter, and both arms must pass a value.
    let mut func = counted_loop();
    let func_blocks = &mut func.blocks;
    func_blocks[3].params.push(BlockParam {
        id: ValueId(100),
        name: "carry".into(),
        ty: Type::I64,
    });
    let out = sile_lsir::run_lsir_pipeline(func, &[sile_lsir::LsirPassKind::ScalarPromote])
        .expect("the result verifies");
    assert!(!out.blocks[3].params.is_empty(), "the join keeps its parameter");
    assert_eq!(out.blocks[3].params[0].id, ValueId(100));
}
