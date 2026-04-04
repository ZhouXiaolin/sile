use sile_hir::ParamKind;
use sile_lir::{
    ElemType, ExecutableKernel, FloatType, Function, KernelAbi, KernelParamAbi, LaunchSemantics,
    Param, ParamPassing, ShapeLayout, Type, ValueInfo, ValueInfoTable,
    print::format_executable_kernel,
};

#[test]
fn printer_includes_abi_and_function_sections() {
    let kernel = ExecutableKernel {
        name: "softmax".into(),
        abi: KernelAbi {
            params: vec![KernelParamAbi {
                index: 0,
                name: "x".into(),
                kind: ParamKind::Input,
                elem: ElemType::F32,
                rank: 2,
                passing: ParamPassing::Buffer,
            }],
            shape_layout: ShapeLayout {
                total_dims: 2,
                offsets: vec![0],
            },
            launch: LaunchSemantics { program_id_dims: 2 },
        },
        func: Function::new(
            "softmax",
            vec![Param {
                name: "x".into(),
                ty: Type::ptr(Type::Float(FloatType::F32)),
            }],
            Type::Void,
        ),
        value_info: ValueInfoTable {
            params: vec![ValueInfo::Buffer {
                elem: ElemType::F32,
                rank: 2,
            }],
            instructions: vec![ValueInfo::Index],
        },
    };

    let text = format_executable_kernel(&kernel);
    assert!(text.contains("kernel @softmax"));
    assert!(text.contains("abi:"));
    assert!(text.contains("arg0 input buffer<f32, rank=2>"));
    assert!(text.contains("launch program_id_dims=2"));
    assert!(text.contains("func:"));
}
