use sile_backend::{BackendArtifact, CodegenTarget, compile};
use sile_llvm_ir::{
    BasicBlock, BlockId, Constant, Function, Inst, InstOp, Intrinsic, Operand, Terminator, Type,
    ValueId,
};

fn simple_function(insts: Vec<Inst>, terminator: Terminator) -> Function {
    Function {
        name: "compile_test".into(),
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
fn compile_rejects_switch_terminator() {
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

    let err = compile(&func, CodegenTarget::C).expect_err("switch should be rejected");
    assert!(
        err.to_string()
            .contains("does not support switch terminators")
    );
}

#[test]
fn compile_rejects_metal_thread_id_intrinsic() {
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

    let err =
        compile(&func, CodegenTarget::Metal).expect_err("thread_id should be rejected for metal");
    assert!(
        err.to_string()
            .contains("does not support thread_id intrinsic")
    );
}

#[test]
fn compile_rejects_metal_unknown_helper_call() {
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

    let err = compile(&func, CodegenTarget::Metal).expect_err("call should be rejected");
    assert!(
        err.to_string()
            .contains("does not support LLVM IR call instructions")
    );
}

#[test]
fn c_compile_allows_thread_id_intrinsic() {
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

    let artifact =
        compile(&func, CodegenTarget::C).expect("C target should allow thread_id intrinsic");
    match artifact {
        BackendArtifact::CSource(source) => {
            assert!(source.contains("#include <omp.h>"));
            assert!(source.contains("#define llir_thread_id_0() (0)"));
        }
        other => panic!("expected C source, got {other:?}"),
    }
}
