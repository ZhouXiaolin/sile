use sile_llvm_ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst,
    InstOp, Intrinsic, Metadata, Operand, Param, ParamAbi, Terminator, Type, ValueId,
    format_llvm_ir,
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
            Param {
                id: ValueId(16),
                name: "__sile_shapes".into(),
                ty: Type::ptr(AddressSpace::Constant, Type::I64),
                abi: None,
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
                        result_name: Some("k_end_ptr".into()),
                        ty: Type::ptr(AddressSpace::Constant, Type::I64),
                        op: InstOp::Gep {
                            base: Operand::Value(ValueId(16)),
                            indices: vec![Operand::Const(Constant::Int(1))],
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(9)),
                        result_name: Some("k_end".into()),
                        ty: Type::I64,
                        op: InstOp::Load {
                            ptr: Operand::Value(ValueId(8)),
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(10)),
                        result_name: Some("cond".into()),
                        ty: Type::I1,
                        op: InstOp::Cmp {
                            pred: CmpPred::Slt,
                            lhs: Operand::Value(ValueId(6)),
                            rhs: Operand::Value(ValueId(9)),
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::CondBr {
                    cond: Operand::Value(ValueId(10)),
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
                        id: ValueId(11),
                        name: "k_body".into(),
                        ty: Type::I64,
                    },
                    BlockParam {
                        id: ValueId(12),
                        name: "acc_body".into(),
                        ty: private_tile_ptr.clone(),
                    },
                ],
                insts: vec![
                    Inst {
                        result: Some(ValueId(13)),
                        result_name: Some("tmp_mul".into()),
                        ty: Type::F32,
                        op: InstOp::Bin {
                            op: BinOp::Mul,
                            lhs: Operand::Const(Constant::Float(2.0)),
                            rhs: Operand::Const(Constant::Float(3.0)),
                        },
                        metadata: vec![Metadata::Unroll(4)],
                    },
                    Inst {
                        result: Some(ValueId(14)),
                        result_name: Some("tmp_add".into()),
                        ty: Type::F32,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(13)),
                            rhs: Operand::Const(Constant::Float(1.0)),
                        },
                        metadata: vec![],
                    },
                    Inst {
                        result: Some(ValueId(15)),
                        result_name: Some("k_next".into()),
                        ty: Type::I64,
                        op: InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(ValueId(11)),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                        metadata: vec![],
                    },
                ],
                terminator: Terminator::Br {
                    target: BlockId(1),
                    args: vec![Operand::Value(ValueId(15)), Operand::Value(ValueId(12))],
                },
            },
            BasicBlock {
                id: BlockId(3),
                name: "exit".into(),
                params: vec![BlockParam {
                    id: ValueId(17),
                    name: "acc_final".into(),
                    ty: private_tile_ptr,
                }],
                insts: vec![Inst {
                    result: None,
                    result_name: None,
                    ty: Type::Void,
                    op: InstOp::Store {
                        ptr: Operand::Value(ValueId(2)),
                        value: Operand::Value(ValueId(17)),
                    },
                    metadata: vec![Metadata::WriteOnly],
                }],
                terminator: Terminator::Ret { value: None },
            },
        ],
    };

    let printed = format_llvm_ir(&func);

    assert!(printed.contains("define void @matmul_dynamic_k"));
    assert!(printed.contains("ptr addrspace(1) %a"));
    assert!(printed.contains("ptr addrspace(4) %__sile_shapes"));
    assert!(printed.contains("; sile.param %a [rank=2, shape_offset=0, elem=f32]"));
    assert!(printed.contains("; sile.param %__sile_shapes [elem=i64]"));
    assert!(printed.contains("loop_header:"));
    assert!(printed.contains("; args(%k: i64, %acc: ptr addrspace(5))"));
    assert!(
        printed.contains("%k_end_ptr = getelementptr i64, ptr addrspace(4) %__sile_shapes, i64 1")
    );
    assert!(printed.contains("%k_end = load i64, ptr addrspace(4) %k_end_ptr"));
    assert!(printed.contains("%tmp_mul = fmul 2.0, 3.0"));
    assert!(printed.contains("%tmp_add = fadd %tmp_mul, 1.0"));
    assert!(printed.contains(
        "br i1 %cond, label %loop_body, label %exit ; true_args(%k, %acc) ; false_args(%acc)"
    ));
    assert!(printed.contains("[align=16]"));
    assert!(printed.contains("[unroll=4]"));
}
