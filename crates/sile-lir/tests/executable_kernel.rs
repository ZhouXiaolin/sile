use sile_hir::ParamKind;
use sile_lir::{
    ElemType, ExecutableKernel, FloatType, Function, KernelAbi, KernelParamAbi, LaunchSemantics,
    Param, ParamPassing, ShapeLayout, Type, ValueInfo, ValueInfoTable,
};

#[test]
fn executable_kernel_keeps_abi_and_value_info_together() {
    let kernel = ExecutableKernel {
        name: "vec_add".into(),
        abi: KernelAbi {
            params: vec![
                KernelParamAbi {
                    index: 0,
                    name: "a".into(),
                    kind: ParamKind::Input,
                    elem: ElemType::F32,
                    rank: 1,
                    passing: ParamPassing::Buffer,
                },
                KernelParamAbi {
                    index: 1,
                    name: "c".into(),
                    kind: ParamKind::Output,
                    elem: ElemType::F32,
                    rank: 1,
                    passing: ParamPassing::Buffer,
                },
            ],
            shape_layout: ShapeLayout {
                total_dims: 2,
                offsets: vec![0, 1],
            },
            launch: LaunchSemantics { program_id_dims: 1 },
        },
        func: Function::new(
            "vec_add",
            vec![Param {
                name: "a".into(),
                ty: Type::ptr(Type::Float(FloatType::F32)),
            }],
            Type::Void,
        ),
        value_info: ValueInfoTable {
            params: vec![
                ValueInfo::Buffer {
                    elem: ElemType::F32,
                    rank: 1,
                },
                ValueInfo::Buffer {
                    elem: ElemType::F32,
                    rank: 1,
                },
            ],
            instructions: vec![ValueInfo::Tile {
                elem: ElemType::F32,
                rows: 1,
                cols: 4,
            }],
        },
    };

    assert_eq!(kernel.abi.shape_layout.total_dims, 2);
    assert_eq!(kernel.abi.launch.program_id_dims, 1);
    assert!(matches!(
        kernel.value_info.instructions[0],
        ValueInfo::Tile {
            elem: ElemType::F32,
            rows: 1,
            cols: 4
        }
    ));
}

#[test]
fn shape_layout_offsets_match_param_order() {
    let layout = ShapeLayout {
        total_dims: 5,
        offsets: vec![0, 2, 4],
    };

    assert_eq!(layout.offsets, vec![0, 2, 4]);
    assert_eq!(layout.total_dims, 5);
}
