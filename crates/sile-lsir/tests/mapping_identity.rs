//! The workgroup-tree reduce seeds its per-thread partial with the fold's
//! identity. Max/Min must seed ±inf — the unit the TileIR fold itself
//! carries — or a fold over data containing infinities folds wrong
//! (max(-f32::MAX, -inf) == -f32::MAX != -inf).
//!
//! The fixture is the minimal one the reduce stage accepts: a tile alloca of
//! one 4-wide f32 row, one TileReduce over it, threads=2 (extent % threads ==
//! 0 holds), and the ThreadId the policy stage inserts.

use sile_lsir::{
    AddressSpace, BasicBlock, BlockId, Function, Inst, InstOp, Intrinsic, MmaPolicy,
    MappingPolicy, Operand, ReducePolicy, TileReduceOp, Type, ValueId, VectorPolicy,
};
use sile_lsir::{LaunchShape, mapping};

fn tile_reduce_func(reduce: TileReduceOp) -> Function {
    let tile = ValueId(1);
    let red = ValueId(2);
    Function {
        name: "reduce_inf".into(),
        params: Vec::new(),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            name: "entry".into(),
            params: Vec::new(),
            insts: vec![
                // The policy stage inserts ThreadId at index 0; pre-insert it
                // so `thread_id_of` finds one without relying on stage order.
                Inst {
                    result: Some(ValueId(9)),
                    result_name: Some("tid".into()),
                    ty: Type::I64,
                    op: InstOp::Intrinsic {
                        intrinsic: Intrinsic::ThreadId { dim: 0 },
                        args: Vec::new(),
                    },
                    metadata: Vec::new(),
                },
                Inst {
                    result: Some(tile),
                    result_name: Some("t".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::F32),
                    op: InstOp::Alloca {
                        alloc_ty: Type::array(4, Type::F32),
                        addr_space: AddressSpace::Private,
                    },
                    metadata: Vec::new(),
                },
                Inst {
                    result: Some(red),
                    result_name: Some("red".into()),
                    ty: Type::ptr(AddressSpace::Private, Type::array(1, Type::F32)),
                    op: InstOp::Intrinsic {
                        intrinsic: Intrinsic::TileReduce {
                            reduce,
                            extent: 4,
                            inner: 1,
                            outer: 1,
                        },
                        args: vec![Operand::Value(tile)],
                    },
                    metadata: Vec::new(),
                },
            ],
            terminator: sile_lsir::Terminator::Ret {
                value: Some(Operand::Value(red)),
            },
        }],
        entry: BlockId(0),
        metadata: Vec::new(),
    }
}

fn tree_policy() -> MappingPolicy {
    MappingPolicy {
        launch: LaunchShape::flat(2),
        reduce: ReducePolicy::WorkgroupTree,
        mma: MmaPolicy::Serial,
        vector: VectorPolicy::None,
        shared_cap: 16384,
    }
}

fn seed_literal(reduce: TileReduceOp) -> String {
    let mapped =
        mapping::run(tile_reduce_func(reduce), &tree_policy()).expect("the tree reduce maps");
    sile_lsir::format_function(&mapped)
}

#[test]
fn max_reduce_partial_seeds_negative_infinity() {
    let text = seed_literal(TileReduceOp::Max);
    assert!(text.contains("-inf"), "the max partial seed is not -inf:\n{text}");
}

#[test]
fn min_reduce_partial_seeds_positive_infinity() {
    let text = seed_literal(TileReduceOp::Min);
    assert!(
        text.contains("inf") && !text.contains("-inf"),
        "the min partial seed is not +inf:\n{text}"
    );
}

#[test]
fn sum_reduce_partial_seeds_zero() {
    let text = seed_literal(TileReduceOp::Sum);
    assert!(
        text.contains("0x0p+0") || text.contains("0.0"),
        "the sum partial seed is not 0.0:\n{text}"
    );
}

#[test]
fn product_reduce_partial_seeds_one() {
    let text = seed_literal(TileReduceOp::Product);
    assert!(
        text.contains("0x1p+0") || text.contains("1.0"),
        "the product partial seed is not 1.0:\n{text}"
    );
}
