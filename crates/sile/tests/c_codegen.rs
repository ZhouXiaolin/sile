use sile::{
    codegen::c::{BufferKind, KernelGenInfo},
    hir::{BuiltinOp, ElemType, Expr, Kernel, Param, ParamKind, ShapeExpr, Stmt, Type},
    lir, passes, scheduling, ssa, typeck,
};

#[test]
fn c_codegen_emits_vec_add_with_openmp() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();
    let ssa = passes::canonicalize::run(ssa::lower_typed_kernel_to_ssa(&typed));
    let ssa = passes::dce::run(ssa);
    let lir_func = lir::lower_ssa_to_lir(&ssa, &typed);
    let annotations = scheduling::annotate(&lir_func);

    let info = KernelGenInfo {
        name: kernel.name.clone(),
        num_buffers: kernel.params.len(),
        buffer_kinds: kernel
            .params
            .iter()
            .map(|p| match p.kind {
                ParamKind::Input => BufferKind::Input,
                ParamKind::Output => BufferKind::Output,
            })
            .collect(),
        num_shapes: 1,
        param_ranks: vec![1, 1, 1],
        shape_offsets: vec![0, 1, 2],
    };

    let c = sile::codegen::c::generate(&lir_func, &info).unwrap();

    assert!(c.contains("void sile_kernel_vec_add"));
    assert!(c.contains("#include <omp.h>"));
    assert!(c.contains("#pragma omp parallel for"));
}

fn build_vec_add_kernel() -> Kernel {
    Kernel::new(
        "vec_add",
        vec![],
        vec![
            Param::new(
                "a",
                ParamKind::Input,
                Type::tile(ElemType::F32, ShapeExpr::symbol("S")),
            ),
            Param::new(
                "b",
                ParamKind::Input,
                Type::tile(ElemType::F32, ShapeExpr::symbol("S")),
            ),
            Param::new(
                "c",
                ParamKind::Output,
                Type::tile(ElemType::F32, ShapeExpr::symbol("S")),
            ),
        ],
        vec![
            Stmt::Let {
                name: "tid".into(),
                ty: None,
                expr: Expr::builtin(BuiltinOp::ProgramId),
            },
            Stmt::Let {
                name: "tile_a".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![Expr::Var("a".into())],
                },
            },
            Stmt::Let {
                name: "tile_b".into(),
                ty: None,
                expr: Expr::Builtin {
                    op: BuiltinOp::LoadTile,
                    args: vec![Expr::Var("b".into())],
                },
            },
            Stmt::Store {
                target: "c".into(),
                value: Expr::Builtin {
                    op: BuiltinOp::Add,
                    args: vec![Expr::Var("tile_a".into()), Expr::Var("tile_b".into())],
                },
            },
        ],
    )
}
