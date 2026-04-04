use sile_llir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst,
    InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, Terminator, Type, ValueId,
    format_function,
};

#[test]
fn prints_dynamic_k_matmul_like_cfg() {
    let f32x2x2 = Type::array(2, Type::array(2, Type::F32));
    let private_tile_ptr = Type::ptr(AddressSpace::Private, f32x2x2.clone());

    let func = Function {
        name: "matmul_dynamic_k".into(),
        params: vec![
            Param {
                id: ValueId(0),
                name: "a".into(),
                ty: Type::ptr(AddressSpace::Global, Type::F32),
                abi: Some(ParamAbi {
                    rank: 2,
                    shape_offset: 0,
                }),
            },
            Param {
                id: ValueId(1),
                name: "b".into(),
                ty: Type::ptr(AddressSpace::Global, Type::F32),
                abi: Some(ParamAbi {
                    rank: 2,
                    shape_offset: 2,
                }),
            },
            Param {
                id: ValueId(2),
                name: "c".into(),
                ty: Type::ptr(AddressSpace::Global, Type::F32),
                abi: Some(ParamAbi {
                    rank: 2,
                    shape_offset: 4,
                }),
            },
        ],
        entry: BlockId(0),
        metadata: vec![],
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![
                    Inst {
                        result: Some(ValueId(3)),
                        result_name: Some("m_idx".into()),
                        ty: Type::I32,
                        op: InstOp::Intrinsic {
                            intrinsic: Intrinsic::BlockId { dim: 0 },
                            args: vec![],
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(4)),
                        result_name: Some("n_idx".into()),
                        ty: Type::I32,
                        op: InstOp::Intrinsic {
                            intrinsic: Intrinsic::BlockId { dim: 1 },
                            args: vec![],
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(5)),
                        result_name: Some("acc0".into()),
                        ty: private_tile_ptr.clone(),
                        op: InstOp::Alloca {
                            alloc_ty: f32x2x2.clone(),
                            addr_space: AddressSpace::Private,
                        },
                        metadata: vec![Metadata::Alignment(16)],
                    },
                ],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: vec![Operand::Const(Constant::Int(0)), Operand::Value(ValueId(5))],
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "loop_header".into(),
                params: vec![
                    BlockParam {
                        id: ValueId(6),
                        name: "k".into(),
                        ty: Type::I64,
                    },
                    BlockParam {
                        id: ValueId(7),
                        name: "acc".into(),
                        ty: private_tile_ptr.clone(),
                    },
                ],
                insts: vec![
                    Inst {
                        result: Some(ValueId(8)),
                        result_name: Some("k_end".into()),
                        ty: Type::I64,
                        op: InstOp::ShapeDim {
                            buf: Operand::Value(ValueId(0)),
                            dim: 1,
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(9)),
                        result_name: Some("cond".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: CmpPred::Slt,
                            lhs: Operand::Value(ValueId(6)),
                            rhs: Operand::Value(ValueId(8)),
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::CondBr {
                    cond: Operand::Value(ValueId(9)),
                    true_target: BlockId(2),
                    true_args: vec![Operand::Value(ValueId(6)), Operand::Value(ValueId(7))],
                    false_target: BlockId(3),
                    false_args: vec![Operand::Value(ValueId(7))],
                },
            },
            BasicBlock {
                id: BlockId(2),
                name: "loop_body".into(),
                params: vec![
                    BlockParam {
                        id: ValueId(10),
                        name: "k_body".into(),
                        ty: Type::I64,
                    },
                    BlockParam {
                        id: ValueId(11),
                        name: "acc_body".into(),
                        ty: private_tile_ptr.clone(),
                    },
                ],
                insts: vec![
                    Inst {
                        result: Some(ValueId(12)),
                        result_name: Some("acc_next".into()),
                        ty: private_tile_ptr.clone(),
                        op: InstOp::Intrinsic {
                            intrinsic: Intrinsic::MatmulFragment,
                            args: vec![
                                Operand::Value(ValueId(0)),
                                Operand::Value(ValueId(1)),
                                Operand::Value(ValueId(11)),
                            ],
                        },
                        metadata: vec![Metadata::Unroll(4)],
                    },
                    Inst {
                        result: Some(ValueId(13)),
                        result_name: Some("k_next".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(10)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: vec![Operand::Value(ValueId(13)), Operand::Value(ValueId(12))],
                },
            },
            BasicBlock {
                id: BlockId(3),
                name: "exit".into(),
                params: vec![BlockParam {
                    id: ValueId(14),
                    name: "acc_final".into(),
                    ty: private_tile_ptr,
                }],
                insts: vec![Inst {
                    result: None,
                    result_name: None,
                    ty: Type::Void,
                    op: InstOp::Store {
                        ptr: Operand::Value(ValueId(2)),
                        value: Operand::Value(ValueId(14)),
                    },
                    metadata: vec![Metadata::WriteOnly],
                }],
                terminator: Terminator::Ret { value: None },
            },
        ],
    };

    let printed = format_function(&func);

    assert!(printed.contains("define void @matmul_dynamic_k"));
    assert!(printed.contains("ptr<global, f32> %a [rank=2, shape_offset=0]"));
    assert!(printed.contains("loop_header(%k: i64, %acc: ptr<private, [2 x [2 x f32]]>)"));
    assert!(printed.contains("%k_end = shape.dim %a, 1"));
    assert!(printed.contains("intrinsic matmul_fragment(%a, %b, %acc_body)"));
    assert!(printed.contains("condbr %cond, label %loop_body(%k, %acc), label %exit(%acc)"));
    assert!(printed.contains("[align=16]"));
    assert!(printed.contains("[unroll=4]"));
}
