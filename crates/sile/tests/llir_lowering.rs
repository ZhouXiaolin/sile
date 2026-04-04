use sile::{
    compiler,
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    llir, typeck,
};

#[test]
fn dynamic_k_matmul_lowers_to_llir_with_cfg_and_private_tiles() {
    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let mir = compiler::lower_to_mir(&typed);
    let mir = compiler::dce::run(mir);
    let llir_func = compiler::lower_mir_to_llir(&mir, &typed);
    let printed = llir::format_function(&llir_func);

    assert!(printed.contains("define void @matmul("));
    assert!(printed.contains("%a [rank=2, shape_offset=0]"));
    assert!(printed.contains("%b [rank=2, shape_offset=2]"));
    assert!(printed.contains("%c [rank=2, shape_offset=4]"));
    assert!(printed.contains("bb1("));
    assert!(printed.contains("condbr"));
    assert!(printed.contains("ptr<private, [2 x [2 x f32]]>"));
    assert!(printed.contains("shape.dim %a, 1"));
    assert!(printed.contains("mul %v"));
    assert!(printed.contains("add %v"));
    assert!(printed.contains("gep %a, ["));
    assert!(printed.contains("gep %b, ["));
    assert!(printed.contains("gep %c, ["));
    assert!(printed.contains("load %v"));
    assert!(printed.contains("store %v"));
    assert!(!printed.contains("intrinsic matmul_fragment"));
    assert!(!printed.contains("call @tile_splat_f32"));
    assert!(!printed.contains("call @tile_load_2d_f32"));
    assert!(!printed.contains("call @tile_store_2d_f32"));
}

#[test]
fn vec_add_lowers_to_llir_without_tile_binary_helpers() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let mir = compiler::lower_to_mir(&typed);
    let mir = compiler::dce::run(mir);
    let llir_func = compiler::lower_mir_to_llir(&mir, &typed);
    let printed = llir::format_function(&llir_func);

    assert!(printed.contains("define void @vec_add("));
    assert!(printed.contains("gep %a, ["));
    assert!(printed.contains("gep %b, ["));
    assert!(printed.contains("gep %c, ["));
    assert!(printed.contains("load %v"));
    assert!(printed.contains("add %v"));
    assert!(printed.contains("store %v"));
    assert!(!printed.contains("call @tile_add_f32"));
    assert!(!printed.contains("call @tile_sub_f32"));
    assert!(!printed.contains("call @tile_mul_f32"));
    assert!(!printed.contains("call @tile_div_f32"));
}

#[test]
fn softmax_lowers_to_llir_without_reduce_or_broadcast_helpers() {
    let kernel = build_softmax_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let mir = compiler::lower_to_mir(&typed);
    let mir = compiler::dce::run(mir);
    let llir_func = compiler::lower_mir_to_llir(&mir, &typed);
    let printed = llir::format_function(&llir_func);

    assert!(printed.contains("define void @softmax("));
    assert!(printed.contains("intrinsic exp("));
    assert!(printed.contains("select %v"));
    assert!(!printed.contains("intrinsic reduce_add"));
    assert!(!printed.contains("intrinsic reduce_max"));
    assert!(!printed.contains("call @tile_broadcast_f32"));
    assert!(!printed.contains("call @tile_splat_f32"));
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

fn build_vec_add_kernel() -> Kernel {
    Kernel::new(
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
    )
}

fn build_softmax_kernel() -> Kernel {
    Kernel::new(
        "softmax",
        vec![("BM".into(), 2), ("BN".into(), 8)],
        vec![
            Param::new(
                "x",
                ParamKind::Input,
                Type::tensor(
                    ElemType::F32,
                    ShapeExpr::tuple([ShapeExpr::dynamic(), ShapeExpr::dynamic()]),
                ),
            ),
            Param::new(
                "y",
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
                name: "tile_x".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("x".into()),
                        Expr::Shape(ShapeExpr::tuple([
                            ShapeExpr::symbol("BM"),
                            ShapeExpr::symbol("BN"),
                        ])),
                        Expr::Shape(ShapeExpr::tuple([
                            ShapeExpr::symbol("m_idx"),
                            ShapeExpr::symbol("n_idx"),
                        ])),
                    ],
                },
            },
            Stmt::Let {
                name: "tile_x_max".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ReduceMax,
                    args: vec![Expr::Var("tile_x".into())],
                },
            },
            Stmt::Let {
                name: "tile_x_max_bcast".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::Broadcast,
                    args: vec![
                        Expr::Var("tile_x_max".into()),
                        Expr::Shape(ShapeExpr::tuple([
                            ShapeExpr::symbol("BM"),
                            ShapeExpr::symbol("BN"),
                        ])),
                    ],
                },
            },
            Stmt::Let {
                name: "num".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::Exp,
                    args: vec![Expr::Builtin {
                        op: BuiltinOp::Sub,
                        args: vec![
                            Expr::Var("tile_x".into()),
                            Expr::Var("tile_x_max_bcast".into()),
                        ],
                    }],
                },
            },
            Stmt::Let {
                name: "denom".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::ReduceSum,
                    args: vec![Expr::Var("num".into())],
                },
            },
            Stmt::Let {
                name: "denom_bcast".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::Broadcast,
                    args: vec![
                        Expr::Var("denom".into()),
                        Expr::Shape(ShapeExpr::tuple([
                            ShapeExpr::symbol("BM"),
                            ShapeExpr::symbol("BN"),
                        ])),
                    ],
                },
            },
            Stmt::Store {
                target: "y".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Div,
                    args: vec![Expr::Var("num".into()), Expr::Var("denom_bcast".into())],
                },
            },
        ],
    )
}
