use sile_hir::Kernel;

use sile_lir::ElemType;
use sile_lir::executable::{
    ExecutableKernel, KernelAbi, KernelParamAbi, LaunchSemantics, ParamPassing, ShapeLayout, ValueInfo,
    ValueInfoTable,
};

#[test]
fn executable_kernel_keeps_abi_and_value_info_together() {
    let params = vec![KernelParamAbi {
        index: 0,
        name: "input".to_owned(),
        kind: ElemType::F32,
        elem: ElemType::F32,
        rank: 2,
        passing: ParamPassing::Buffer,
    }];

    let layout = ShapeLayout {
        total_dims: 4,
        offsets: vec![0, 8],
    };

    let abi = KernelAbi {
        params: params.clone(),
        shape_layout: layout.clone(),
        launch: LaunchSemantics {
            program_id_dims: [1, 1, 1],
        },
    };

    let value_info = ValueInfoTable {
        params: params.clone(),
        instructions: vec![ValueInfo::Tile {
            elem: ElemType::F32,
            rank: 2,
            layout: layout.clone(),
            param_index: 0,
        }],
    };

    let kernel = ExecutableKernel {
        name: "test".to_owned(),
        abi,
        func: zilch_kernel(),
        value_info,
    };

    assert_eq!(kernel.abi.params.len(), kernel.value_info.params.len());

    if let Some(ValueInfo::Tile { param_index, .. }) = kernel.value_info.instructions.first() {
        assert_eq!(*param_index, 0);
    } else {
        panic!("expected a tile entry");
    }
}

#[test]
fn shape_layout_offsets_match_param_order() {
    let offsets = vec![0, 16, 32];
    let layout = ShapeLayout {
        total_dims: 6,
        offsets: offsets.clone(),
    };

    let params = offsets
        .iter()
        .enumerate()
        .map(|(idx, _)| KernelParamAbi {
            index: idx,
            name: format!("arg{}", idx),
            kind: ElemType::F32,
            elem: ElemType::F32,
            rank: 2,
            passing: ParamPassing::Buffer,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        layout.offsets,
        params
            .iter()
            .map(|param| param.index * 16)
            .collect::<Vec<_>>()
    );
}

fn zilch_kernel() -> Kernel {
    Kernel::new("zilch", vec![], vec![], vec![])
}
