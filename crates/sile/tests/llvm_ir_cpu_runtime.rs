use sile::{
    Device, LaunchConfig, Tensor, compiler,
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    typeck,
};
use sile_backend::cpu::CpuBackend;

#[test]
fn cpu_backend_executes_vec_add_through_llir() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();

    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();
    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();

    let a = Tensor::from_vec(vec![1.0; 16], [16], &device).unwrap();
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device).unwrap();
    let mut c = Tensor::zeros([16], &device).unwrap();

    let backend = CpuBackend::new();
    let args = vec![a.as_kernel_arg(), b.as_kernel_arg(), c.as_kernel_arg_mut()];
    backend
        .execute_llir(
            &llir_func,
            &args,
            &LaunchConfig { grid: [4, 1, 1] },
            &stream,
        )
        .unwrap();
    drop(args);

    assert_eq!(c.to_vec(&stream).unwrap(), vec![3.0; 16]);
}

#[test]
fn cpu_backend_executes_dynamic_k_matmul_through_llir() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();

    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();
    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();

    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
        [4, 4],
        &device,
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![
            1.0, 0.0, 2.0, 1.0, //
            0.0, 1.0, 3.0, 2.0, //
            1.0, 0.0, 4.0, 3.0, //
            0.0, 1.0, 5.0, 4.0,
        ],
        [4, 4],
        &device,
    )
    .unwrap();
    let mut c = Tensor::zeros([4, 4], &device).unwrap();

    let backend = CpuBackend::new();
    let args = vec![a.as_kernel_arg(), b.as_kernel_arg(), c.as_kernel_arg_mut()];
    backend
        .execute_llir(
            &llir_func,
            &args,
            &LaunchConfig { grid: [4, 1, 1] },
            &stream,
        )
        .unwrap();
    drop(args);

    let actual = c.to_vec(&stream).unwrap();
    let expected = vec![
        4.0, 6.0, 40.0, 30.0, //
        12.0, 14.0, 96.0, 70.0, //
        20.0, 22.0, 152.0, 110.0, //
        28.0, 30.0, 208.0, 150.0,
    ];

    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-4,
            "mismatch at {idx}: got {lhs}, expected {rhs}"
        );
    }
}

#[test]
fn cpu_backend_executes_relu_max_tile_through_llir() {
    let device = Device::cpu();
    let stream = device.create_stream().unwrap();

    let kernel = build_relu_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();
    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();

    let mut x = Tensor::from_vec(
        vec![-3.0, -1.0, 0.0, 2.5, 4.0, -7.0, 8.0, -0.5],
        [8],
        &device,
    )
    .unwrap();

    let backend = CpuBackend::new();
    let args = vec![x.as_kernel_arg_mut()];
    backend
        .execute_llir(
            &llir_func,
            &args,
            &LaunchConfig { grid: [1, 1, 1] },
            &stream,
        )
        .unwrap();
    drop(args);

    assert_eq!(
        x.to_vec(&stream).unwrap(),
        vec![0.0, 0.0, 0.0, 2.5, 4.0, 0.0, 8.0, 0.0]
    );
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
