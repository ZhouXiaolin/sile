use sile_backend::{CodegenTarget, passes::verify};
use sile_llir::{
    BasicBlock, BlockId, Constant, Function, Inst, InstOp, Intrinsic, Operand, Terminator, Type,
    ValueId,
};

fn simple_function(insts: Vec<Inst>, terminator: Terminator) -> Function {
    Function {
        name: "verify_test".into(),
        params: vec![],
        blocks: vec![BasicBlock {
            id: BlockId(0),
            name: "entry".into(),
            params: vec![],
            insts,
            terminator,
        }],
        entry: BlockId(0),
        metadata: vec![],
    }
}

#[test]
fn shared_verify_rejects_switch_terminator() {
    let func = Function {
        name: "switch_test".into(),
        params: vec![],
        blocks: vec![
            BasicBlock {
                id: BlockId(0),
                name: "entry".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Switch {
                    value: Operand::Const(Constant::Int(0)),
                    default: BlockId(1),
                    cases: vec![(1, BlockId(1))],
                },
            },
            BasicBlock {
                id: BlockId(1),
                name: "exit".into(),
                params: vec![],
                insts: vec![],
                terminator: Terminator::Ret { value: None },
            },
        ],
        entry: BlockId(0),
        metadata: vec![],
    };

    let err = verify::run(&func, "test").expect_err("switch should be rejected");
    assert!(
        err.to_string()
            .contains("does not support switch terminators")
    );
}

#[test]
fn metal_verify_rejects_thread_id_intrinsic() {
    let func = simple_function(
        vec![Inst {
            result: Some(ValueId(0)),
            result_name: Some("tid".into()),
            ty: Type::I64,
            op: InstOp::Intrinsic {
                intrinsic: Intrinsic::ThreadId { dim: 0 },
                args: vec![],
            },
            metadata: vec![],
        }],
        Terminator::Ret { value: None },
    );

    let err = verify::run_for_target(&func, CodegenTarget::Metal, "test")
        .expect_err("thread_id should be rejected for metal");
    assert!(
        err.to_string()
            .contains("does not support thread_id intrinsic")
    );
}

#[test]
fn metal_verify_rejects_unknown_helper_call() {
    let func = simple_function(
        vec![Inst {
            result: None,
            result_name: None,
            ty: Type::Void,
            op: InstOp::Call {
                func: "unknown_helper".into(),
                args: vec![],
            },
            metadata: vec![],
        }],
        Terminator::Ret { value: None },
    );

    let err = verify::run_for_target(&func, CodegenTarget::Metal, "test")
        .expect_err("unknown helper should be rejected for metal");
    assert!(err.to_string().contains("only supports helper calls"));
}

#[test]
fn c_verify_allows_thread_id_intrinsic() {
    let func = simple_function(
        vec![Inst {
            result: Some(ValueId(0)),
            result_name: Some("tid".into()),
            ty: Type::I64,
            op: InstOp::Intrinsic {
                intrinsic: Intrinsic::ThreadId { dim: 0 },
                args: vec![],
            },
            metadata: vec![],
        }],
        Terminator::Ret { value: None },
    );

    verify::run_for_target(&func, CodegenTarget::C, "test")
        .expect("C target should allow thread_id intrinsic");
}
