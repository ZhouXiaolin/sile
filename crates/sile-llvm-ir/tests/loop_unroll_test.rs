use sile_llvm_ir::{
    passes::loop_unroll, BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function,
    Inst, InstOp, Operand, Terminator, Type, ValueId,
};

/// Build a canonical counted loop:
///   entry → header(iv, other) → body(iv_body, other_body) → header(next, other_body)
///                              → exit(other)
/// The comparison is `iv < trip_count` (Slt).
fn make_counted_loop(trip_count: i64) -> Function {
    let header_id = BlockId(1);
    let body_id = BlockId(2);
    let exit_id = BlockId(3);

    Function {
        name: "test_counted_loop".into(),
        params: vec![],
        entry: BlockId(0),
        metadata: vec![],
        blocks: vec![
            // entry
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Br {
                    target: header_id,
                    args: vec![
                        Operand::Const(Constant::Int(0)),
                        Operand::Const(Constant::Int(0)),
                    ],
                },
            },
            // header params: [iv: i64, other: i64]
            BasicBlock {
                id: header_id,
                name: "header".into(),
                params: vec![
                    BlockParam {
                        id: ValueId(0),
                        name: "iv".into(),
                        ty: Type::I64,
                    },
                    BlockParam {
                        id: ValueId(1),
                        name: "other".into(),
                        ty: Type::I64,
                    },
                ],
                insts: vec![Inst {
                    result: Some(ValueId(2)),
                    result_name: Some("cond".into()),
                    ty: Type::I1,
                    op: InstOp::Cmp {
                        pred: CmpPred::Slt,
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Const(Constant::Int(trip_count)),
                    },
                    metadata: vec![],
                }],
                terminator: Terminator::CondBr {
                    cond: Operand::Value(ValueId(2)),
                    true_target: body_id,
                    true_args: vec![
                        Operand::Value(ValueId(0)),
                        Operand::Value(ValueId(1)),
                    ],
                    false_target: exit_id,
                    false_args: vec![Operand::Value(ValueId(1))],
                },
            },
            // body params: [iv_body: i64, other_body: i64]
            BasicBlock {
                id: body_id,
                name: "body".into(),
                params: vec![
                    BlockParam {
                        id: ValueId(3),
                        name: "iv_body".into(),
                        ty: Type::I64,
                    },
                    BlockParam {
                        id: ValueId(4),
                        name: "other_body".into(),
                        ty: Type::I64,
                    },
                ],
                insts: vec![
                    Inst {
                        result: Some(ValueId(5)),
                        result_name: Some("tmp".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(0)),
                            rhs: Operand::Value(ValueId(3)),
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(6)),
                        result_name: Some("next".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(0)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::Br {
                    target: header_id,
                    args: vec![
                        Operand::Value(ValueId(6)),
                        Operand::Value(ValueId(4)),
                    ],
                },
            },
            // exit
            BasicBlock {
                id: exit_id,
                name: "exit".into(),
                params: vec![BlockParam {
                    id: ValueId(7),
                    name: "other_exit".into(),
                    ty: Type::I64,
                }],
                insts: vec![],
                terminator: Terminator::Ret { value: None },
            },
        ],
    }
}

#[test]
fn simple_constant_trip_loop_unrolled() {
    let func = make_counted_loop(3);
    let result = loop_unroll::run(func);

    let header_block = result
        .blocks
        .iter()
        .find(|b| b.id == BlockId(1))
        .expect("header block should exist");

    // The original header should now be a pass-through Br to the first body copy
    let Terminator::Br { target: first_copy, args: _ } = &header_block.terminator else {
        panic!("expected pass-through Br terminator on header, got {:?}", header_block.terminator);
    };

    // Verify the first body copy exists
    let copy0 = result
        .blocks
        .iter()
        .find(|b| b.id == *first_copy)
        .expect("first body copy block should exist");
    assert!(
        copy0.name.contains("unroll.0"),
        "expected body copy 0, got: {}",
        copy0.name
    );
    assert_eq!(copy0.params.len(), 2, "body copy should have 2 params");

    // Verify the exit block exists and is linked from the last copy
    let last_copy = result
        .blocks
        .iter()
        .filter(|b| b.name.contains("unroll.2"))
        .last()
        .expect("last body copy (unroll.2) should exist");
    let Terminator::Br {
        target: last_target,
        args: _,
    } = &last_copy.terminator
    else {
        panic!("last copy should branch to exit");
    };
    assert_eq!(*last_target, BlockId(3), "last copy should branch to exit");

    // The exit block should be unchanged
    let exit = result
        .blocks
        .iter()
        .find(|b| b.id == BlockId(3))
        .expect("exit block should exist");
    assert_eq!(exit.name, "exit");

    // The entry block should be unchanged
    let entry = result
        .blocks
        .iter()
        .find(|b| b.id == BlockId(0))
        .expect("entry block should exist");
    let Terminator::Br {
        target: entry_target,
        args: _,
    } = &entry.terminator
    else {
        panic!("entry should branch");
    };
    assert_eq!(*entry_target, BlockId(1), "entry should branch to header");

    // Verify each body copy has proper instructions and params
    for k in 0..3 {
        let copy = result
            .blocks
            .iter()
            .find(|b| b.name.contains(&format!("unroll.{}", k)))
            .unwrap_or_else(|| panic!("body copy {} should exist", k));
        assert_eq!(copy.insts.len(), 2, "copy {} should have 2 instructions", k);
    }
}

#[test]
fn large_trip_loop_not_unrolled() {
    let func = make_counted_loop(10);
    let result = loop_unroll::run(func);

    // Loop with trip count > MAX_UNROLL_TRIP_COUNT should be unchanged
    let header = result
        .blocks
        .iter()
        .find(|b| b.id == BlockId(1))
        .expect("header block should exist");
    assert!(
        matches!(header.terminator, Terminator::CondBr { .. }),
        "header should still have CondBr when loop is not unrolled"
    );
}

#[test]
fn loop_with_work_unrolled_correctly() {
    let func = make_counted_loop(2);
    let result = loop_unroll::run(func);

    // With trip_count=2, we should have:
    //   pass-through header + body.0 + body.1 + entry + exit = 5 blocks

    // Find all body copy blocks
    let copy_count = result
        .blocks
        .iter()
        .filter(|b| b.name.contains("unroll."))
        .count();
    assert_eq!(copy_count, 2, "should have exactly 2 body copies");

    // Body copy 1 (last) should branch to exit
    let copy1 = result
        .blocks
        .iter()
        .find(|b| b.name.contains("unroll.1"))
        .expect("body copy 1 should exist");
    let Terminator::Br {
        target: t,
        args: _,
    } = &copy1.terminator
    else {
        panic!("last copy should branch");
    };
    assert_eq!(*t, BlockId(3), "last copy should go to exit");

    // Body copy 0 should branch to body copy 1
    let copy0 = result
        .blocks
        .iter()
        .find(|b| b.name.contains("unroll.0"))
        .expect("body copy 0 should exist");
    let Terminator::Br {
        target: t0,
        args: _,
    } = &copy0.terminator
    else {
        panic!("copy 0 should branch");
    };
    let copy1_id = copy1.id;
    assert_eq!(*t0, copy1_id, "copy 0 should branch to copy 1");
}
