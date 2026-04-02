use sile::{codegen, BinaryOp, KernelSpec, Node, Param, ParamKind, ScalarType, Shape, Store, TileExpr};

#[test]
fn c_codegen_emits_mvp_abi_and_add_loop() {
    let spec = KernelSpec {
        name: "vec_add".to_string(),
        params: vec![
            Param { index: 0, kind: ParamKind::Input, ty: ScalarType::F32, shape: Shape::new([16]) },
            Param { index: 1, kind: ParamKind::Input, ty: ScalarType::F32, shape: Shape::new([16]) },
            Param { index: 2, kind: ParamKind::Output, ty: ScalarType::F32, shape: Shape::new([16]) },
        ],
        nodes: vec![
            Node::LoadTile { param: 0, tile: TileExpr::grid_x(), shape: Shape::new([4]) },
            Node::LoadTile { param: 1, tile: TileExpr::grid_x(), shape: Shape::new([4]) },
            Node::Binary { op: BinaryOp::Add, lhs: 0, rhs: 1, shape: Shape::new([4]) },
        ],
        stores: vec![Store { param: 2, tile: TileExpr::grid_x(), value: 2 }],
    };

    let c = codegen::c::generate(&spec).unwrap();
    assert!(c.contains("typedef struct"));
    assert!(c.contains("void sile_kernel_vec_add"));
    assert!(c.contains("tile_id[0]"));
    assert!(c.contains("tmp_2[i] = tmp_0[i] + tmp_1[i];"));
}
