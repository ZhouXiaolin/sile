use sile_compiler::compile;
use sile_hir::typeck::check_kernel;
use sile_hir::{ElemType, Kernel, Param, ParamKind, ShapeExpr, Type};
use sile_lir::ValueInfo;

fn empty_vec_add_kernel() -> Kernel {
    Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::Tensor {
                    elem: ElemType::F32,
                    shape: ShapeExpr::tuple([ShapeExpr::symbol("N")]),
                },
            ),
        ],
        vec![],
    )
}

#[test]
fn compile_returns_executable_kernel_with_abi() {
    let typed = check_kernel(&empty_vec_add_kernel()).expect("typed kernel");
    let executable = compile(&typed);

    assert_eq!(executable.name, "vec_add");
    assert_eq!(executable.abi.params.len(), 3);
    assert_eq!(executable.abi.shape_layout.offsets, vec![0, 1, 2]);
    assert_eq!(executable.abi.launch.program_id_dims, 1);
    assert!(matches!(
        executable.value_info.params[2],
        ValueInfo::Buffer { rank: 1, .. }
    ));
}
