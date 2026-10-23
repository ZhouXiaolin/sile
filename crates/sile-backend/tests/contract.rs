//! The input contract holds for every target: the SIMD-group intrinsics are
//! the mapping's plants, so one in the input means the pipeline changed
//! under the mapping. A kernel carrying one compiles nowhere.

use sile_backend::{compile, CodegenTarget};
use sile_lsir::{
    AddressSpace, BasicBlock, BlockId, Function, Inst, InstOp, Intrinsic, Operand,
    TileReduceOp, Type, ValueId,
};

fn simd_input() -> Function {
    let tile = ValueId(1);
    let red = ValueId(2);
    Function {
        name: "simd_plant".into(),
        params: Vec::new(),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            name: "entry".into(),
            params: Vec::new(),
            insts: vec![
                Inst {
                    result: Some(tile),
                    result_name: Some("t".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::array(4, Type::F32)),
                    op: InstOp::Alloca {
                        alloc_ty: Type::array(4, Type::F32),
                        addr_space: AddressSpace::Private,
                    },
                    metadata: Vec::new(),
                },
                Inst {
                    result: Some(red),
                    result_name: Some("red".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::array(1, Type::F32)),
                    op: InstOp::Intrinsic {
                        intrinsic: Intrinsic::SimdReduce {
                            reduce: TileReduceOp::Sum,
                        },
                        args: vec![Operand::Value(tile)],
                    },
                    metadata: Vec::new(),
                },
            ],
            terminator: sile_lsir::Terminator::Ret {
                value: Some(Operand::Value(red)),
            },
        }],
        entry: BlockId(0),
        metadata: Vec::new(),
    }
}

#[test]
fn the_input_contract_rejects_the_simd_plant_on_every_target() {
    for target in [
        CodegenTarget::C,
        CodegenTarget::Metal,
        CodegenTarget::Wgsl,
        CodegenTarget::Ptx,
    ] {
        let err = compile(&simd_input(), target)
            .expect_err("the lowering never produces the SIMD-group intrinsics");
        assert!(
            err.to_string().contains("SIMD-group"),
            "{target:?}: {err}"
        );
    }
}
