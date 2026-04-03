use sile_hir::{
    BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Type, ValueCategory,
};

#[test]
fn shape_expr_tracks_dynamic_symbolic_and_const_dims() {
    let shape = ShapeExpr::tuple([
        ShapeExpr::dynamic(),
        ShapeExpr::symbol("BN"),
        ShapeExpr::constant(1),
    ]);
    assert_eq!(shape.rank(), 3);
    assert!(shape.contains_dynamic());
    assert_eq!(shape.to_string(), "[-1, BN, 1]");
}

#[test]
fn kernel_model_distinguishes_tensor_and_tile_values() {
    let input = Param::new(
        "x",
        ParamKind::Input,
        Type::tensor(ElemType::F32, ShapeExpr::tuple([ShapeExpr::dynamic()])),
    );
    let output = Param::new(
        "y",
        ParamKind::Output,
        Type::tensor(
            ElemType::F32,
            ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
        ),
    );

    let kernel = Kernel::new(
        "demo",
        vec![("BM".into(), 0), ("BN".into(), 0)],
        vec![input, output],
        vec![],
    );

    assert_eq!(kernel.name, "demo");
    assert_eq!(
        kernel.const_params,
        vec![("BM".to_string(), 0), ("BN".to_string(), 0)]
    );
    assert_eq!(kernel.params[0].ty.category(), ValueCategory::Tensor);
    assert_eq!(
        Type::tile(ElemType::F32, ShapeExpr::tuple([ShapeExpr::symbol("BM")])).category(),
        ValueCategory::Tile
    );
    assert_eq!(Expr::builtin(BuiltinOp::ProgramId).kind_name(), "builtin");
}
