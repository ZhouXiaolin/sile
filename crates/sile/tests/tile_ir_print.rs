use sile::{
    compiler,
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    tileir, typeck,
};

#[test]
fn tile_ir_print_uses_cuda_tile_style_ops() {
    let kernel = Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
        ],
        vec![
            Stmt::Let {
                name: "tid".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ShapeDim,
                    args: vec![Expr::builtin(BuiltinOp::ProgramId), Expr::ScalarI64(0)],
                },
            },
            Stmt::Let {
                name: "x".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("a".into()),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::constant(4)])),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("tid")])),
                    ],
                },
            },
            Stmt::Let {
                name: "y".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("b".into()),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::constant(4)])),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("tid")])),
                    ],
                },
            },
            Stmt::Store {
                target: "c".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Add,
                    args: vec![Expr::Var("x".into()), Expr::Var("y".into())],
                },
            },
        ],
    );

    let typed = typeck::check_kernel(&kernel).unwrap();
    let tile_ir =
        compiler::run_hir_to_tile_ir_pipeline(&typed, compiler::ACTIVE_HIR_TO_TILE_IR_PIPELINE)
            .unwrap();
    let printed = tileir::format_tile_ir(&tile_ir);

    assert!(printed.contains("module {"));
    assert!(printed.contains("entry @vec_add("));
    assert!(printed.contains("kind = \"launch.index\", dim = 0"));
    assert!(printed.contains("load_ptr_tko"));
    assert!(printed.contains("addf"));
    assert!(printed.contains("store_ptr_tko"));
    assert!(printed.contains(", [%5, %3] {shape = [1, 4], stride_dim = 0}"));
    assert!(printed.contains("return"));
    assert!(printed.contains("tile<1x4xf32>"));
    assert!(printed.contains("tile<ptr<f32>>"));
    assert!(!printed.contains("kind = \"sile.program_id\""));
    assert!(!printed.contains("sile.get_tile_block_id_dim"));
    assert!(!printed.contains("dim = 1"));
}

#[test]
fn tile_ir_pointwise_fusion_pass_forms_explicit_map_op() {
    let kernel = Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::tensor(ElemType::F32, ShapeExpr::dynamic()),
            ),
        ],
        vec![
            Stmt::Let {
                name: "tid".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ShapeDim,
                    args: vec![Expr::builtin(BuiltinOp::ProgramId), Expr::ScalarI64(0)],
                },
            },
            Stmt::Let {
                name: "x".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("a".into()),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::constant(2)])),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("tid")])),
                    ],
                },
            },
            Stmt::Let {
                name: "y".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("b".into()),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::constant(2)])),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("tid")])),
                    ],
                },
            },
            Stmt::Store {
                target: "c".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Add,
                    args: vec![Expr::Var("x".into()), Expr::Var("y".into())],
                },
            },
        ],
    );

    let typed = typeck::check_kernel(&kernel).unwrap();
    let tile_ir =
        compiler::run_hir_to_tile_ir_pipeline(&typed, compiler::ACTIVE_HIR_TO_TILE_IR_PIPELINE)
            .unwrap();
    let tile_ir = compiler::run_tile_ir_passes(tile_ir).unwrap();
    let printed = tileir::format_tile_ir(&tile_ir);

    assert!(printed.contains("map addf(load_ptr_tko("));
    assert!(!printed.contains("= addf "));
    assert!(!printed.contains("= load_ptr_tko %0"));
    assert!(!printed.contains("= load_ptr_tko %1"));
}
