//! The dominator case of cross-block CSE: a two-block chain where the second
//! block repeats an expression its predecessor already computed.

use sile_lsir::{AddressSpace, BasicBlock, BlockId, Function, Inst, InstOp, Operand, Terminator, Type, ValueId};

fn add(result: u32, lhs: u32) -> Inst {
    Inst {
        result: Some(ValueId(result)),
        result_name: None,
        ty: Type::I64,
        op: InstOp::Bin {
            op: sile_lsir::BinOp::Add,
            lhs: Operand::Value(ValueId(lhs)),
            rhs: Operand::Const(sile_lsir::Constant::Int(1)),
        },
        metadata: Vec::new(),
    }
}

#[test]
fn cse_merges_across_straight_chain() {
    let func = Function {
        name: "chain".into(),
        params: Vec::new(),
        entry: BlockId(0),
        metadata: Vec::new(),
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: Vec::new(),
                insts: vec![Inst {
                    result: Some(ValueId(0)),
                    result_name: None,
                    ty: Type::I64,
                    op: InstOp::Intrinsic {
                        intrinsic: sile_lsir::Intrinsic::BlockId { dim: 0 },
                        args: Vec::new(),
                    },
                    metadata: Vec::new(),
                }],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "b1".into(),
                params: Vec::new(),
                insts: vec![add(1, 0)],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "b2".into(),
                params: Vec::new(),
                insts: vec![add(2, 0)],
                terminator: Terminator::Ret { value: None },
            },
        ],
    };

    let out = sile_lsir::run_lsir_pipeline(func, &[sile_lsir::LsirPassKind::Cse])
        .expect("cse");
    let b2 = out.blocks.iter().find(|b| b.id == BlockId(2)).unwrap();
    assert!(
        b2.insts.is_empty(),
        "b2 should have folded its add into b1's, got {:?}",
        b2.insts
    );
}

#[test]
fn alloca_type_has_space() {
    // Anchor the AddressSpace import so the file compiles standalone.
    let _ = AddressSpace::Private;
}

#[test]
fn cse_merges_when_predecessor_is_last_in_list() {
    // Mirror the saxpy CFG shape: the second copy's add is in a block whose
    // only predecessor sits at the END of the block list.
    let func = Function {
        name: "late_pred".into(),
        params: Vec::new(),
        entry: BlockId(0),
        metadata: Vec::new(),
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: Vec::new(),
                insts: vec![Inst {
                    result: Some(ValueId(0)),
                    result_name: None,
                    ty: Type::I64,
                    op: InstOp::Intrinsic {
                        intrinsic: sile_lsir::Intrinsic::BlockId { dim: 0 },
                        args: Vec::new(),
                    },
                    metadata: Vec::new(),
                }],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "b1".into(),
                params: Vec::new(),
                insts: vec![add(1, 0)],
                terminator: Terminator::Br {
                    target: BlockId(2),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "b2".into(),
                params: Vec::new(),
                insts: vec![add(2, 0)],
                terminator: Terminator::Br {
                    target: BlockId(3),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(3),
                name: "b3_latch".into(),
                params: Vec::new(),
                insts: Vec::new(),
                terminator: Terminator::Br {
                    target: BlockId(4),
                    args: Vec::new(),
                },
            },
            BasicBlock {
                id: BlockId(4),
                name: "b4_preheader_last".into(),
                params: Vec::new(),
                insts: Vec::new(),
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            },
        ],
    };

    let out = sile_lsir::run_lsir_pipeline(func, &[sile_lsir::LsirPassKind::Cse])
        .expect("cse");
    let b2 = out.blocks.iter().find(|b| b.id == BlockId(2)).unwrap();
    assert!(
        b2.insts.is_empty(),
        "b2 should fold into b1 even with the predecessor last, got {:?}",
        b2.insts
    );
}
