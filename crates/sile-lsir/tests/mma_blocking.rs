//! The strip mma's column-blocked spelling.
//!
//! With the workgroup width a whole multiple of the tile's row length, every
//! element a thread owns sits in one column, so the stage spells the
//! contraction outermost, the thread's strip innermost, and the strip's
//! partials into a private slot array whose slots carry constant indices —
//! the only form sroa and scalar promotion can turn into registers. The
//! lowering's result copy (`acc[e] = result[e]`, the result tile's only
//! consumer) is written through, so no copy loop and no result round-trip
//! survive the stage. A row length the width cannot fix falls back to the
//! per-lane shape.
//!
//! The fixture is the shape the gemm lowering produces: panels of 32x16 and
//! 16x16, an accumulator tile, a TileMma of 32x16x16, threads=128 (strips of
//! four), and the copy as a counted loop in the index-alloca shape the
//! mapping spells.

use sile_lsir::{
    mapping, AddressSpace, BasicBlock, BinOp, BlockId, CmpPred, Constant, Function, Inst, InstOp,
    Intrinsic, LaunchShape, MmaPolicy, MappingPolicy, Operand, ReducePolicy, Terminator, Type,
    ValueId, VectorPolicy,
};

const TID: ValueId = ValueId(1);
const LHS: ValueId = ValueId(2);
const RHS: ValueId = ValueId(3);
const CARRY: ValueId = ValueId(4);
const RESULT: ValueId = ValueId(5);
const IV: ValueId = ValueId(6);
const AT: ValueId = ValueId(7);
const CMP: ValueId = ValueId(8);
const FROM: ValueId = ValueId(9);
const LOADED: ValueId = ValueId(10);
const TO: ValueId = ValueId(11);

fn inst(result: ValueId, ty: Type, op: InstOp) -> Inst {
    Inst {
        result: Some(result),
        result_name: None,
        ty,
        op,
        metadata: Vec::new(),
    }
}

fn void(op: InstOp) -> Inst {
    Inst {
        result: None,
        result_name: None,
        ty: Type::Void,
        op,
        metadata: Vec::new(),
    }
}

fn block(id: BlockId, name: &str, insts: Vec<Inst>, terminator: Terminator) -> BasicBlock {
    BasicBlock {
        id,
        name: name.into(),
        params: Vec::new(),
        insts,
        terminator,
    }
}

fn tile_mma_func() -> Function {
    Function {
        name: "gemm_tiles".into(),
        params: Vec::new(),
        blocks: vec![
            block(
                BlockId(0),
                "entry",
                vec![
                    inst(
                        TID,
                        Type::I64,
                        InstOp::Intrinsic {
                            intrinsic: Intrinsic::ThreadId { dim: 0 },
                            args: Vec::new(),
                        },
                    ),
                    inst(
                        LHS,
                        Type::ptr(AddressSpace::Shared, Type::array(32 * 16, Type::F32)),
                        InstOp::Alloca {
                            alloc_ty: Type::array(32 * 16, Type::F32),
                            addr_space: AddressSpace::Shared,
                        },
                    ),
                    inst(
                        RHS,
                        Type::ptr(AddressSpace::Shared, Type::array(16 * 16, Type::F32)),
                        InstOp::Alloca {
                            alloc_ty: Type::array(16 * 16, Type::F32),
                            addr_space: AddressSpace::Shared,
                        },
                    ),
                    inst(
                        CARRY,
                        Type::ptr(AddressSpace::Shared, Type::array(32 * 16, Type::F32)),
                        InstOp::Alloca {
                            alloc_ty: Type::array(32 * 16, Type::F32),
                            addr_space: AddressSpace::Shared,
                        },
                    ),
                    inst(
                        RESULT,
                        Type::ptr(AddressSpace::Private, Type::array(32 * 16, Type::F32)),
                        InstOp::Intrinsic {
                            intrinsic: Intrinsic::TileMma {
                                m: 32,
                                n: 16,
                                k: 16,
                                rhs_cols: 16,
                            },
                            args: vec![
                                Operand::Value(LHS),
                                Operand::Value(RHS),
                                Operand::Value(CARRY),
                            ],
                        },
                    ),
                ],
                Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            ),
            // The lowering's copy: carry[e] = result[e] over the tile, one
            // counted loop in the index-alloca shape the mapping spells.
            block(
                BlockId(1),
                "copy_header",
                vec![
                    inst(
                        AT,
                        Type::I64,
                        InstOp::Load {
                            ptr: Operand::Value(IV),
                        },
                    ),
                    inst(
                        CMP,
                        Type::I1,
                        InstOp::Cmp {
                            pred: CmpPred::Slt,
                            lhs: Operand::Value(AT),
                            rhs: Operand::Const(Constant::Int(32 * 16)),
                        },
                    ),
                ],
                Terminator::CondBr {
                    cond: Operand::Value(CMP),
                    true_target: BlockId(2),
                    true_args: Vec::new(),
                    false_target: BlockId(3),
                    false_args: Vec::new(),
                },
            ),
            block(
                BlockId(2),
                "copy_body",
                vec![
                    inst(
                        FROM,
                        Type::ptr(AddressSpace::Private, Type::F32),
                        InstOp::Gep {
                            base: Operand::Value(RESULT),
                            indices: vec![Operand::Value(AT)],
                        },
                    ),
                    inst(
                        LOADED,
                        Type::F32,
                        InstOp::Load {
                            ptr: Operand::Value(FROM),
                        },
                    ),
                    inst(
                        TO,
                        Type::ptr(AddressSpace::Shared, Type::F32),
                        InstOp::Gep {
                            base: Operand::Value(CARRY),
                            indices: vec![Operand::Value(AT)],
                        },
                    ),
                    void(InstOp::Store {
                        ptr: Operand::Value(TO),
                        value: Operand::Value(LOADED),
                    }),
                    inst(
                        IV_STEP,
                        Type::I64,
                        InstOp::Bin {
                            op: BinOp::Add,
                            lhs: Operand::Value(AT),
                            rhs: Operand::Const(Constant::Int(1)),
                        },
                    ),
                    void(InstOp::Store {
                        ptr: Operand::Value(IV),
                        value: Operand::Value(IV_STEP),
                    }),
                ],
                Terminator::Br {
                    target: BlockId(1),
                    args: Vec::new(),
                },
            ),
            block(
                BlockId(3),
                "copy_exit",
                vec![],
                Terminator::Br {
                    target: BlockId(4),
                    args: Vec::new(),
                },
            ),
            block(BlockId(4), "after", vec![], Terminator::Ret { value: None }),
        ],
        entry: BlockId(0),
        metadata: Vec::new(),
    }
}

const IV_STEP: ValueId = ValueId(12);

fn fixture() -> Function {
    let mut func = tile_mma_func();
    func.blocks[0].insts.push(inst(
        IV,
        Type::I64,
        InstOp::Alloca {
            alloc_ty: Type::I64,
            addr_space: AddressSpace::Private,
        },
    ));
    func
}

fn strip_policy() -> MappingPolicy {
    MappingPolicy {
        launch: LaunchShape::flat(128),
        reduce: ReducePolicy::WorkgroupTree,
        mma: MmaPolicy::ThreadStrips,
        vector: VectorPolicy::None,
        shared_cap: 16384,
    }
}

#[test]
fn the_blocked_spelling_fixes_the_column_and_drops_the_copy() {
    let mapped = mapping::run(fixture(), &strip_policy()).expect("the strip mma maps");
    let text = sile_lsir::format_function(&mapped);
    // The slot array holds one slot per strip element.
    assert!(
        text.contains("alloca [4 x float]"),
        "the blocked strip keeps four private slots:\n{text}"
    );
    // The contraction loop is the only loop the stage spells; the seed and
    // the copy-out are straight lines over constant slot indices.
    assert!(
        !text.contains("_j_header") && !text.contains("_s_header") && !text.contains("_o_header"),
        "the strip is spelled straight out:\n{text}"
    );
    // The result tile's copy loop is gone and the copy-out wrote the
    // accumulator directly, so nothing reads the tile again: the stage's
    // dead-inst elimination takes the alloca itself.
    assert_eq!(
        text.matches("%tile0x1").count(),
        0,
        "the result tile does not survive the stage:\n{text}"
    );
    // Two barriers, both ordering the panels.
    assert_eq!(
        text.matches("call void @llvm.nvvm.barrier0()").count(),
        2,
        "two barriers order the panels:\n{text}"
    );
}

#[test]
fn a_row_length_the_width_cannot_fix_keeps_the_per_lane_shape() {
    let mut func = fixture();
    // Retile the multiply so the row length no longer divides the width.
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            if let InstOp::Intrinsic {
                intrinsic: Intrinsic::TileMma { n, rhs_cols, .. },
                ..
            } = &mut inst.op
            {
                *n = 12;
                *rhs_cols = 12;
            }
        }
    }
    let mapped = mapping::run(func, &strip_policy()).expect("the strip mma maps");
    let text = sile_lsir::format_function(&mapped);
    assert!(
        text.contains("mm0_body"),
        "the per-lane strip keeps its lane loop:\n{text}"
    );
}
