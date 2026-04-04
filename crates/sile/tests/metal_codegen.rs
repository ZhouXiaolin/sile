use sile::{
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    typeck,
};

#[test]
fn metal_codegen_emits_structured_loop_for_dynamic_k_matmul() {
    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let executable = sile_mir::lower_to_mir(&typed);
    let executable = sile_mir::dce::run(executable);
    let executable = sile_mir::lower_mir_to_lir(&executable, &typed);

    let metal = sile_backend_metal::codegen_metal::generate(
        &executable.func,
        &executable.abi,
        &executable.value_info,
    )
    .unwrap();

    assert!(metal.contains("while (true)"));
    assert!(metal.contains("break;"));
    assert!(!metal.contains("goto "));
}

fn build_dynamic_k_matmul_kernel() -> Kernel {
    Kernel::new(
        "matmul",
        vec![("BM".into(), 2), ("BN".into(), 2), ("BK".into(), 2)],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::tensor(
                    ElemType::F32,
                    ShapeExpr::tuple([ShapeExpr::dynamic(), ShapeExpr::dynamic()]),
                ),
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::tensor(
                    ElemType::F32,
                    ShapeExpr::tuple([ShapeExpr::dynamic(), ShapeExpr::dynamic()]),
                ),
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::tensor(
                    ElemType::F32,
                    ShapeExpr::tuple([ShapeExpr::symbol("BM"), ShapeExpr::symbol("BN")]),
                ),
            ),
        ],
        vec![
            Stmt::Let {
                name: "m_idx".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ShapeDim,
                    args: vec![Expr::builtin(BuiltinOp::ProgramId), Expr::ScalarI64(0)],
                },
            },
            Stmt::Let {
                name: "n_idx".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ShapeDim,
                    args: vec![Expr::builtin(BuiltinOp::ProgramId), Expr::ScalarI64(1)],
                },
            },
            Stmt::Let {
                name: "acc".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::Constant,
                    args: vec![
                        Expr::ScalarF32(0.0),
                        Expr::Shape(ShapeExpr::tuple([
                            ShapeExpr::symbol("BM"),
                            ShapeExpr::symbol("BN"),
                        ])),
                    ],
                },
            },
            Stmt::ForLoop {
                var: "k_idx".into(),
                start: Expr::ScalarI64(0),
                end: Expr::Builtin {
                    op: BuiltinOp::Div,
                    args: vec![
                        Expr::Builtin {
                            op: BuiltinOp::ShapeDim,
                            args: vec![Expr::Var("a".into()), Expr::ScalarI64(1)],
                        },
                        Expr::ScalarI64(2),
                    ],
                },
                body: vec![
                    Stmt::Let {
                        name: "a_tile".into(),
                        ty: None,
                        expr: Expr::Builtin {
                            op: BuiltinOp::LoadTile,
                            args: vec![
                                Expr::Var("a".into()),
                                Expr::Shape(ShapeExpr::tuple([
                                    ShapeExpr::symbol("BM"),
                                    ShapeExpr::symbol("BK"),
                                ])),
                                Expr::Shape(ShapeExpr::tuple([
                                    ShapeExpr::symbol("m_idx"),
                                    ShapeExpr::symbol("k_idx"),
                                ])),
                            ],
                        },
                    },
                    Stmt::Let {
                        name: "b_tile".into(),
                        ty: None,
                        expr: Expr::Builtin {
                            op: BuiltinOp::LoadTile,
                            args: vec![
                                Expr::Var("b".into()),
                                Expr::Shape(ShapeExpr::tuple([
                                    ShapeExpr::symbol("BK"),
                                    ShapeExpr::symbol("BN"),
                                ])),
                                Expr::Shape(ShapeExpr::tuple([
                                    ShapeExpr::symbol("k_idx"),
                                    ShapeExpr::symbol("n_idx"),
                                ])),
                            ],
                        },
                    },
                    Stmt::Assign {
                        name: "acc".into(),
                        expr: Expr::Builtin {
                            op: BuiltinOp::Mma,
                            args: vec![
                                Expr::Var("a_tile".into()),
                                Expr::Var("b_tile".into()),
                                Expr::Var("acc".into()),
                            ],
                        },
                    },
                ],
            },
            Stmt::Store {
                target: "c".into(),
                value: Expr::Var("acc".into()),
            },
        ],
    )
}
