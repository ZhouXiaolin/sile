use sile::{
    compiler,
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    llvmir, typeck,
};

#[test]
fn dynamic_k_matmul_lowers_to_llvm_like_llvm_ir_with_explicit_memory_ops() {
    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llvm_ir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let printed = llvmir::format_llvm_ir(&llvm_ir_func);

    assert!(printed.contains("define void @matmul("));
    assert!(printed.contains("ptr addrspace(1) %a"));
    assert!(printed.contains("ptr addrspace(1) %b"));
    assert!(printed.contains("ptr addrspace(1) %c"));
    assert!(printed.contains("ptr addrspace(4) %__sile_shapes"));
    assert!(printed.contains("; sile.param %a [rank=2, shape_offset=0, elem=f32]"));
    assert!(printed.contains("; sile.param %b [rank=2, shape_offset=2, elem=f32]"));
    assert!(printed.contains("; sile.param %c [rank=2, shape_offset=4, elem=f32]"));
    assert!(printed.contains("; sile.param %__sile_shapes [elem=i64]"));
    assert!(printed.contains("bb0:"));
    assert!(printed.contains("matmul_row_header_"));
    assert!(printed.contains("matmul_k_header_"));
    assert!(printed.contains("; args("));
    assert!(printed.contains("br i1"));
    assert!(
        printed.contains("%__shape_a = getelementptr i64, ptr addrspace(4) %__sile_shapes, i64 0")
    );
    assert!(
        printed.contains("%__shape_b = getelementptr i64, ptr addrspace(4) %__sile_shapes, i64 2")
    );
    assert!(
        printed.contains("%__shape_c = getelementptr i64, ptr addrspace(4) %__sile_shapes, i64 4")
    );
    assert!(printed.contains("getelementptr i64, ptr addrspace(4) %__shape_a, i64 1"));
    assert!(printed.contains("load i64, ptr addrspace(4)"));
    assert!(!printed.contains("alloca [2 x [2 x f32]], addrspace(5)"));
    assert!(!printed.contains("ptr addrspace(5)"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %a,"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %b,"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %c,"));
    assert!(!printed.contains("tile_fill_"));
    assert!(!printed.contains("icmp eq"));
    assert!(printed.contains("load f32, ptr"));
    assert!(printed.contains("store f32"));
    assert!(!printed.contains("store f64"));
    assert!(!printed.contains("call @tile_"));
    assert!(!printed.contains("@llvm.sile.shape.dim"));
    assert!(printed.contains("@llvm.nvvm.ctaid.x"));
}

#[test]
fn vec_add_lowers_to_llvm_like_llvm_ir() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llvm_ir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let printed = llvmir::format_llvm_ir(&llvm_ir_func);

    assert!(printed.contains("define void @vec_add("));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %a,"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %b,"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %c,"));
    assert!(printed.contains("load f32, ptr"));
    assert!(printed.contains("store f32"));
    assert!(printed.contains("fadd %"));
    assert!(!printed.contains("@llvm.nvvm.ctaid.y"));
    assert!(!printed.contains("select i1 "));
    assert!(!printed.contains("alloca [1 x [2 x f32]]"));
    assert!(!printed.contains("call @tile_"));
}

#[test]
fn softmax_lowers_to_llvm_like_llvm_ir() {
    let kernel = build_softmax_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llvm_ir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let printed = llvmir::format_llvm_ir(&llvm_ir_func);

    assert!(printed.contains("define void @softmax("));
    assert!(printed.contains("call f32 @llvm.exp.f32("));
    assert!(printed.contains("fcmp ogt"));
    assert!(printed.contains("select i1 "));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %x,"));
    assert!(printed.contains("getelementptr f32, ptr addrspace(1) %y,"));
    assert!(printed.contains("load f32, ptr"));
    assert!(printed.contains("store f32"));
    assert!(!printed.contains("call @tile_"));
    assert!(!printed.contains("intrinsic exp("));
}

#[test]
fn relu_max_tile_lowers_to_fcmp_and_select() {
    let kernel = build_relu_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llvm_ir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let printed = llvmir::format_llvm_ir(&llvm_ir_func);

    assert!(printed.contains("define void @relu("));
    assert!(printed.contains("fcmp oge"));
    assert!(printed.contains("select i1 "));
    assert!(printed.contains("store f32"));
    assert!(!printed.contains("call @tile_"));
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

fn build_relu_kernel() -> Kernel {
    Kernel::new(
        "relu",
        vec![("D".into(), 8)],
        vec![Param::new(
            "x",
            ParamKind::Output,
            Type::tensor(ElemType::F32, ShapeExpr::tuple([ShapeExpr::symbol("D")])),
        )],
        vec![
            Stmt::Let {
                name: "zero".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::Constant,
                    args: vec![
                        Expr::ScalarF32(0.0),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("D")])),
                    ],
                },
            },
            Stmt::Let {
                name: "data".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![
                        Expr::Var("x".into()),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::symbol("D")])),
                        Expr::Shape(ShapeExpr::tuple([ShapeExpr::constant(0)])),
                    ],
                },
            },
            Stmt::Store {
                target: "x".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Max,
                    args: vec![Expr::Var("zero".into()), Expr::Var("data".into())],
                },
            },
        ],
    )
}
