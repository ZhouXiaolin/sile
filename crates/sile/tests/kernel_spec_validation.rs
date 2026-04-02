use sile::{
    Device, KernelSpec, LaunchConfig, Node, Param, ParamKind, ScalarType, Shape, Store, Tensor,
    TileExpr,
};

fn vec_add_spec() -> KernelSpec {
    KernelSpec {
        name: "vec_add".to_string(),
        params: vec![
            Param {
                index: 0,
                kind: ParamKind::Input,
                ty: ScalarType::F32,
                shape: Shape::new([16]),
            },
            Param {
                index: 1,
                kind: ParamKind::Input,
                ty: ScalarType::F32,
                shape: Shape::new([16]),
            },
            Param {
                index: 2,
                kind: ParamKind::Output,
                ty: ScalarType::F32,
                shape: Shape::new([16]),
            },
        ],
        nodes: vec![
            Node::LoadTile {
                param: 0,
                tile: TileExpr::grid_x(),
                shape: Shape::new([4]),
            },
            Node::LoadTile {
                param: 1,
                tile: TileExpr::grid_x(),
                shape: Shape::new([4]),
            },
            Node::Binary {
                op: sile::BinaryOp::Add,
                lhs: 0,
                rhs: 1,
                shape: Shape::new([4]),
            },
        ],
        stores: vec![Store {
            param: 2,
            tile: TileExpr::grid_x(),
            value: 2,
        }],
    }
}

#[test]
fn launch_validation_accepts_matching_grid_and_shape() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let c = Tensor::zeros([16], &device).unwrap();

    let spec = vec_add_spec();
    spec.validate_launch(
        &[a.as_kernel_arg(), b.as_kernel_arg(), c.as_kernel_arg()],
        &LaunchConfig { grid: [4, 1, 1] },
    )
    .unwrap();
}

#[test]
fn launch_validation_rejects_non_divisible_grid() {
    let device = Device::cpu();
    let a = Tensor::ones([16], &device).unwrap();
    let b = Tensor::ones([16], &device).unwrap();
    let c = Tensor::zeros([16], &device).unwrap();

    let err = vec_add_spec()
        .validate_launch(
            &[a.as_kernel_arg(), b.as_kernel_arg(), c.as_kernel_arg()],
            &LaunchConfig { grid: [3, 1, 1] },
        )
        .unwrap_err();

    assert!(err.to_string().contains("grid"));
}
