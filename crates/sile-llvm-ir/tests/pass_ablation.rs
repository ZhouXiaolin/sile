//! Does each LLVM IR pass actually do anything on the shape it claims to handle?
//!
//! Each test builds the smallest IR that should trigger the pass and asserts
//! the pass changes it. A pass that cannot move its own canonical input has no
//! reason to sit in the pipeline.
//!
//! This is the micro-ablation: it proves the mechanism works in isolation. The
//! macro-ablation, which decides what actually runs on real kernels, is
//! `crates/sile/tests/pass_ablation_e2e.rs`. The two disagree often enough that
//! the macro-ablation wins.

use sile_llvm_ir::passes::{cse, canonicalize, loop_simplify, phi_alias, verify};
use sile_llvm_ir::Function as Fn;
use sile_llvm_ir::{
    BasicBlock, BinOp, BlockId, BlockParam, CmpPred, Constant, Function, Inst, InstOp,
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
/// The pass only touches a block that receives constant block arguments from
/// its single predecessor, so the fixture must build that shape.
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
    // The original `iv + 1` plus the clone should collapse to one definition.
    assert_ne!(
        after, original,
        "CSE did not change an IR that contains an obvious duplicate"
    );
    let live = after
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| {
            matches!(
                &i.op,
                InstOp::Bin { op: BinOp::Add, lhs: Operand::Value(v), rhs: Operand::Const(Constant::Int(1)) }
                    if *v == ValueId(1)
            )
        });
    assert!(live, "CSE removed a live definition");
}

/// The verifier must accept an IR every pass already left alone; if it rejects
/// valid IR the pipeline cannot run at all.
#[test]
fn verify_accepts_ir_the_passes_left_alone() {
    for pass in [
        canonicalize::run as fn(Fn) -> Fn,
        loop_simplify::run as fn(Fn) -> Fn,
        phi_alias::run as fn(Fn) -> Fn,
        cse::run as fn(Fn) -> Fn,
    ] {
        let after = pass(counted_loop_with_invariant());
        assert!(
            verify::verify_function(&after).is_ok(),
            "verify rejected IR that a pipeline pass produced"
        );
    }
}
