//! The simd-group tiers of the mapping: the reduce plants one hardware fold
//! per site — no scratch and no barrier when the workgroup fits one
//! SIMD-group — and the mma stays whole as a `SimdGroupMma` with its
//! partition named. The fixtures are the minimal ones each stage accepts,
//! the same shape `mapping_identity.rs` builds its reduce from.

use sile_lsir::{
    AddressSpace, BasicBlock, BlockId, Function, Inst, InstOp, Intrinsic, MmaPolicy,
    MappingPolicy, Operand, ReducePolicy, TileReduceOp, Type, ValueId, VectorPolicy,
};
use sile_lsir::{LaunchShape, mapping};

fn thread_id(result: ValueId) -> Inst {
    Inst {
        result: Some(result),
        result_name: Some("tid".into()),
        ty: Type::I64,
        op: InstOp::Intrinsic {
            intrinsic: Intrinsic::ThreadId { dim: 0 },
            args: Vec::new(),
        },
        metadata: Vec::new(),
    }
}

fn tile_alloca(result: ValueId, name: &str, len: usize) -> Inst {
    Inst {
        result: Some(result),
        result_name: Some(name.into()),
        ty: Type::ptr(AddressSpace::Private, Type::array(len, Type::F32)),
        op: InstOp::Alloca {
            alloc_ty: Type::array(len, Type::F32),
            addr_space: AddressSpace::Private,
        },
        metadata: Vec::new(),
    }
}

fn tile_reduce_func(reduce: TileReduceOp, extent: u64) -> Function {
    let tile = ValueId(1);
    let red = ValueId(2);
    let out = ValueId(3);
    let red_slot = ValueId(4);
    let loaded = ValueId(5);
    Function {
        name: "reduce_simd".into(),
        params: Vec::new(),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            name: "entry".into(),
            params: Vec::new(),
            insts: vec![
                thread_id(ValueId(9)),
                tile_alloca(tile, "t", extent as usize),
                Inst {
                    result: Some(red),
                    result_name: Some("red".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::array(1, Type::F32)),
                    op: InstOp::Intrinsic {
                        intrinsic: Intrinsic::TileReduce {
                            reduce,
                            extent,
                            inner: 1,
                            outer: 1,
                        },
                        args: vec![Operand::Value(tile)],
                    },
                    metadata: Vec::new(),
                },
                // The result is consumed the way a real kernel consumes it:
                // one element load, so the fold's chain stays live.
                Inst {
                    result: Some(red_slot),
                    result_name: Some("red_slot".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::F32),
                    op: InstOp::Gep {
                        base: Operand::Value(red),
                        indices: vec![Operand::Const(sile_lsir::Constant::Int(0))],
                    },
                    metadata: Vec::new(),
                },
                Inst {
                    result: Some(loaded),
                    result_name: Some("out".into()),
                    ty: Type::F32,
                    op: InstOp::Load {
                        ptr: Operand::Value(red_slot),
                    },
                    metadata: Vec::new(),
                },
            ],
            terminator: sile_lsir::Terminator::Ret {
                value: Some(Operand::Value(loaded)),
            },
        }],
        entry: BlockId(0),
        metadata: Vec::new(),
    }
}

fn simd_policy(threads: u32) -> MappingPolicy {
    MappingPolicy {
        launch: LaunchShape::flat(threads),
        reduce: ReducePolicy::SimdGroupTree,
        mma: MmaPolicy::Serial,
        vector: VectorPolicy::None,
        shared_cap: 16384,
    }
}

fn intrinsics(func: &Function) -> Vec<&Intrinsic> {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.op {
            InstOp::Intrinsic { intrinsic, .. } => Some(intrinsic),
            _ => None,
        })
        .collect()
}

fn shared_allocas(func: &Function) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| {
            matches!(
                &inst.op,
                InstOp::Alloca {
                    addr_space: AddressSpace::Shared,
                    ..
                }
            )
        })
        .count()
}

#[test]
fn simd_tree_folds_the_partial_in_hardware() {
    let mapped =
        mapping::run(tile_reduce_func(TileReduceOp::Sum, 8), &simd_policy(8))
            .expect("the simd tree maps");
    assert!(
        intrinsics(&mapped)
            .iter()
            .any(|intrinsic| matches!(intrinsic, Intrinsic::SimdReduce { reduce: TileReduceOp::Sum })),
        "the strip partial must fold through a SimdReduce:\n{:?}",
        mapped
    );
}

#[test]
fn one_simd_group_needs_no_scratch_and_no_barrier() {
    let mapped =
        mapping::run(tile_reduce_func(TileReduceOp::Sum, 8), &simd_policy(8))
            .expect("the simd tree maps");
    assert!(
        !mapped
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| inst
                .result_name
                .as_deref()
                .map(|name| name.starts_with("sile_reduce_scratch"))
                .unwrap_or(false)),
        "a one-SIMD-group fold writes no scratch:\n{:?}",
        mapped
    );
    assert!(
        !intrinsics(&mapped).iter().any(|intrinsic| matches!(
            intrinsic,
            Intrinsic::SimdLaneId | Intrinsic::SimdGroupId
        )),
        "a one-SIMD-group fold needs no lane or group coordinate:\n{:?}",
        mapped
    );
}

#[test]
fn several_simd_groups_relay_through_group_slots() {
    let mapped =
        mapping::run(tile_reduce_func(TileReduceOp::Sum, 64), &simd_policy(64))
            .expect("the simd tree maps");
    // 64 threads are two SIMD-groups: one shared relay of two slots, named
    // lane and group through the builtins.
    assert!(
        mapped
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| inst
                .result_name
                .as_deref()
                .map(|name| name.starts_with("sile_reduce_scratch"))
                .unwrap_or(false)),
        "the group partials relay through scratch:\n{:?}",
        mapped
    );
    assert!(
        intrinsics(&mapped).iter().any(|intrinsic| matches!(
            intrinsic,
            Intrinsic::SimdLaneId | Intrinsic::SimdGroupId
        )),
        "the relay names the lane and the group through the builtins:\n{:?}",
        mapped
    );
    // Head, one tree step for the two slots, and the tail. The threads stage
    // may add its own barrier when it shares the source tile; only this
    // tier's three are counted from the bottom.
    let barriers = intrinsics(&mapped)
        .iter()
        .filter(|intrinsic| matches!(intrinsic, Intrinsic::Barrier { .. }))
        .count();
    assert!(
        barriers >= 3,
        "head + tree step + tail, got {barriers}:\n{:?}",
        mapped
    );
}

/// One multiply site: `dst = lhs @ rhs + acc` over 32x16x8, the shape the
/// gemm example lowers to.
fn tile_mma_func(m: usize, n: usize, k: usize) -> Function {
    let lhs = ValueId(1);
    let rhs = ValueId(2);
    let acc = ValueId(3);
    let dst = ValueId(4);
    Function {
        name: "mma_simd".into(),
        params: Vec::new(),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            name: "entry".into(),
            params: Vec::new(),
            insts: vec![
                thread_id(ValueId(9)),
                tile_alloca(lhs, "lhs", m * k),
                tile_alloca(rhs, "rhs", k * n),
                tile_alloca(acc, "acc", m * n),
                Inst {
                    result: Some(dst),
                    result_name: Some("dst".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::array(m * n, Type::F32)),
                    op: InstOp::Intrinsic {
                        intrinsic: Intrinsic::TileMma {
                            m: m as u64,
                            n: n as u64,
                            k: k as u64,
                            rhs_cols: n as u64,
                        },
                        args: vec![
                            Operand::Value(lhs),
                            Operand::Value(rhs),
                            Operand::Value(acc),
                        ],
                    },
                    metadata: Vec::new(),
                },
            ],
            terminator: sile_lsir::Terminator::Ret {
                value: Some(Operand::Value(dst)),
            },
        }],
        entry: BlockId(0),
        metadata: Vec::new(),
    }
}

fn simd_tiles_policy(threads: u32) -> MappingPolicy {
    MappingPolicy {
        launch: LaunchShape::flat(threads),
        reduce: ReducePolicy::Serial,
        mma: MmaPolicy::SimdGroupTiles,
        vector: VectorPolicy::None,
        shared_cap: 16384,
    }
}

#[test]
fn simd_tiles_leaves_the_multiply_whole_with_its_partition() {
    // 128 threads are four SIMD-groups; the tile's 4x2 fragment grid splits
    // 2x2, every group holding two fragments.
    let mapped = mapping::run(tile_mma_func(32, 16, 8), &simd_tiles_policy(128))
        .expect("the simd tiles map");
    let splits: Vec<_> = intrinsics(&mapped)
        .iter()
        .filter_map(|intrinsic| match intrinsic {
            Intrinsic::SimdGroupMma {
                row_groups,
                col_groups,
                ..
            } => Some((*row_groups, *col_groups)),
            _ => None,
        })
        .collect();
    assert_eq!(
        splits,
        vec![(2, 2)],
        "the fragment grid splits 2x2 across four SIMD-groups:\n{:?}",
        mapped
    );
    assert!(
        !intrinsics(&mapped)
            .iter()
            .any(|intrinsic| matches!(intrinsic, Intrinsic::TileMma { .. })),
        "no serial multiply survives:\n{:?}",
        mapped
    );
    // The staged handoff, the fragment stores, and the copy-out: three.
    let barriers = intrinsics(&mapped)
        .iter()
        .filter(|intrinsic| matches!(intrinsic, Intrinsic::Barrier { .. }))
        .count();
    assert_eq!(barriers, 3, "{:?}", mapped);
    // The site's consumers keep reading a tile alloca under the site's own
    // id — where cleanup keeps it (private today, shared when the share pass
    // judges otherwise) is that pass's decision, not this tier's.
    let result_allocas: Vec<_> = mapped
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| matches!(&inst.op, InstOp::Alloca { .. }))
        .collect();
    assert!(
        result_allocas
            .iter()
            .any(|inst| inst.result == Some(ValueId(4))),
        "the result alloca keeps its id:\n{:?}",
        mapped
    );
}

#[test]
fn simd_tiles_feasibility_mirrors_the_stage_checks() {
    assert!(
        mapping::simd_tiles_feasible(&tile_mma_func(32, 16, 8), 128, 16384),
        "the gemm shape tiles"
    );
    assert!(
        !mapping::simd_tiles_feasible(&tile_mma_func(12, 16, 8), 128, 16384),
        "an m that misses the multiple of eight does not tile"
    );
    assert!(
        !mapping::simd_tiles_feasible(&tile_mma_func(32, 16, 8), 8, 16384),
        "a partial SIMD-group has no collective spelling"
    );
    assert!(
        !mapping::simd_tiles_feasible(&tile_mma_func(32, 16, 8), 128, 1024),
        "staging past the budget does not tile"
    );
}
