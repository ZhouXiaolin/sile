use sile::{
    codegen::llvmir_c,
    compiler,
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    typeck,
};

#[test]
fn dynamic_k_matmul_llir_codegen_emits_structured_c_without_helper_calls() {
    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_matmul("));
    assert!(c.contains("int64_t* __sile_shapes"));
    assert!(c.contains("sile_llvm_ir_matmul(a, b, c, shapes);"));
    assert!(c.contains("for ("));
    assert!(!c.contains("goto "));
    assert!(!c.contains("while (true)"));
    assert_eq!(c.matches("tile_fill_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_mma_accumulate_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(!c.contains("llir_matmul_fragment("));
    assert!(!c.contains("tile_splat_f32("));
    assert!(c.contains("*(v"));
    assert!(c.contains("&(a["));
    assert!(c.contains("&(b["));
    assert!(c.contains("&(c["));
    assert!(c.contains("float v"));
    assert!(!c.contains("while ("));
    assert!(c.matches("for (").count() >= 3);
    assert!(!c.contains("sile_shape_dim("));
    assert!(!c.contains("static const int64_t* sile_shapes"));
    assert!(!c.contains("_storage"));
    assert!(!c.contains("if (v"));
}

#[test]
fn vec_add_llir_codegen_emits_explicit_c_loops() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_vec_add("));
    assert!(c.contains("for ("));
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_add_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(c.contains(" + "));
    assert!(c.contains("&(a["));
    assert!(c.contains("&(b["));
    assert!(c.contains("&(c["));
    assert!(!c.contains("while ("));
    assert!(!c.contains("__launch_idx1"));
    assert!(!c.contains(" ? "));
    assert!(!c.contains("_storage"));
}

#[test]
fn softmax_llir_codegen_emits_explicit_reduce_and_broadcast_loops() {
    let kernel = build_softmax_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_softmax("));
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_reduce_max_axis1_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_broadcast_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_sub_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_exp_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_reduce_sum_axis1_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_div_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(c.contains("expf("));
    assert!(c.contains(" ? "));
    assert!(c.contains("for ("));
    assert!(!c.contains("while ("));
    assert!(!c.contains("llir_reduce_add("));
    assert!(!c.contains("llir_reduce_max("));
    assert!(!c.contains("tile_splat_f32("));

    // Verify alloca count for softmax — the fusion optimizations should keep this low.
    // Currently: tile_x(2x8), tile_x_max(2x1), num(2x8), denom(2x1), div_result(2x8) = 5
    // TODO: reduce to ≤ 3 with multi-use fusion
    let alloca_count = llir_func
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i.op, sile_llvm_ir::InstOp::Alloca { .. }))
        .count();
    assert!(
        alloca_count <= 5,
        "softmax alloca count should be ≤ 5, got {alloca_count}"
    );
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
