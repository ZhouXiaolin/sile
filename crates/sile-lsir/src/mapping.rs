//! The mapping pass family: how a tile lands on a target's execution units.
//!
//! A tile is a logical unit: the kernel names the elements it computes, not
//! the thread that computes each one. That freedom is spent here, once, under
//! an explicit policy, so the backends stay spellers. The family runs after
//! the scalar pipeline — optimization sees the clean tile-level form, and the
//! barriers materialization introduces are never visible to a pass that could
//! hoist across them — and before the per-target contract verify.
//!
//! # The stages
//!
//! 1. `policy` — the thread count becomes real: one `thread.id` value enters
//!    the entry block, and every later stage indexes with it. Triton's
//!    counterpart is the launch grid: a block's coordinates exist before any
//!    layout is chosen, and every layout indexes with them.
//! 2. `threads` — the lowering's element loops are stripped across the
//!    threads: iteration `j` moves to `tid, tid + W, …`. A `vec4` relay copy
//!    is scalarized first, because a vector relay's owner map (`chunk % W`)
//!    would disagree with a scalar strip's (`element % W`), and the
//!    disagreement is a race on private storage, not a slowdown. This is
//!    Triton's blocked layout: `sizePerThread` splits one tile's elements
//!    across the threads of a block, and the split is the same `element % W`
//!    ownership the strip spells, except that Triton derives it from the
//!    layout while this stage rewrites the loop that walks it.
//! 3. `reduce` — a `TileReduce` becomes per-thread strip partials in shared
//!    scratch and a tree reduction over them. The tree takes `log2(W)` steps
//!    for a compile-time `W`, so the steps are straight-line: one guarded
//!    combine and one barrier each. Triton's counterpart is its reduction
//!    layout and its `log2`-step shuffle/tree combine: same partials, same
//!    tree, the difference is that Triton selects a warp or shared-memory
//!    combine per size while this stage always goes through shared scratch.
//! 4. `mma` — a `TileMma`'s operands move to shared storage, a barrier orders
//!    the panel writes against the reads, and the multiply becomes a
//!    per-thread strip of the accumulator. Triton's counterpart is the
//!    dot-operand layout and the shared-memory staging a `tl.dot` lowers to:
//!    the same operands to shared, the same barrier between the panel write
//!    and the read, the same per-thread accumulator fragment.
//! 5. `cleanup` — the placement the earlier stages earned: the owner-strip
//!    plumbing loops unroll, the tiles they touch shrink to each thread's
//!    fragment, the remaining owner-ordered tiles flip into workgroup
//!    storage, and the relay copies the strip form made dead die by the same
//!    storage pass the scalar pipeline ends with. Triton's counterparts are
//!    the mapped-loop unrolling and the `sizePerThread` fragment a register
//!    tile keeps: a tile that is one element per thread becomes a register,
//!    a tile every thread must see becomes shared storage.
//!
//! # The words the stages pass each other
//!
//! The stages communicate through the IR they leave, and the words they write
//! on an instruction are the part that has to survive: `OwnerOrdered`
//! (`ir.rs`, `Metadata`) says a slot is the reader's own by construction, and
//! `ReduceRelay` says a slot is a broadcast. Both are claims the builder — the
//! reduce stage, or the lowering where it emitted the access — has already
//! checked; a later stage that re-derived them from the pointer chain would be
//! guessing at the same fact a second time.
//!
//! So the survival rule is explicit. A pass that keeps an instruction keeps its
//! metadata: `loop_unroll` clones the body (`inst.clone()`), `canonicalize`
//! copies it when it rewrites an instruction, and `cse` refuses to merge an
//! instruction that carries any metadata at all, so a stamped slot is never
//! confused with an unstamped one. A pass that mints a *new* instruction decides
//! for itself: `storage`'s relay forwarding answers the same slot the load did
//! and carries the load's stamps over, everything else spells an empty set,
//! because a new slot is a new claim. A stage that would break the rule — drop
//! metadata, or move a stamped access to another slot — has to say why in the
//! stage, not leave the next stage to notice.
//!
//! # The invariant
//!
//! Mapping only changes placement, never semantics — cutile's rule that a
//! transform never changes whether a kernel traps. Each stage's output
//! computes the same elements with the same per-element operation order; what
//! changes is which thread runs it and which storage holds it. The one
//! contract the stages owe each other is ownership: every stripped loop hands
//! element `e` to thread `e % W`, so a private value one loop writes is
//! exactly the value the next loop reads.
//!
//! Floating-point folds are the one place the result moves, not by placement
//! but by association: a tree reduction combines partials in a different
//! order than a serial fold. Every target already reassociated these folds
//! (OpenMP reduction clauses spell the same freedom), so the numeric contract
//! stays the tolerance-based one the backends test with.

use crate::ir::{
    AddressSpace, BasicBlock, BinOp, BlockId, CmpPred, Constant, Function, Inst, InstOp, Intrinsic,
    Metadata, Operand, SyncScope, Terminator, TileReduceOp, Type, ValueId,
};
use crate::launch::LaunchShape;

/// How a `TileReduce` materializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducePolicy {
    /// The intrinsic stays; the target spells it as a serial loop.
    Serial,
    /// Per-thread partials in shared scratch and a tree reduction over them,
    /// one barrier per tree step.
    WorkgroupTree,
}

/// How a `TileMma` materializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaPolicy {
    /// The intrinsic stays; the target spells the whole multiply.
    Serial,
    /// Operands in shared storage, one barrier per side, and each thread
    /// computes the accumulator strip it owns.
    ThreadStrips,
}

/// Whether the target wants a loop vectorization hint.
///
/// The hint is a mapping judgment — it says how the target wants a flat
/// element run executed — so the decision lives with the rest of the policy
/// instead of in a dialect. The dialect still spells it; only the yes/no
/// moved here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorPolicy {
    None,
    /// C's `#pragma omp simd` over a simple loop body.
    OmpSimd,
}

/// The mapping decisions one target made, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappingPolicy {
    /// Invocations that cooperate on one tile.
    pub launch: LaunchShape,
    pub reduce: ReducePolicy,
    pub mma: MmaPolicy,
    pub vector: VectorPolicy,
    /// The workgroup-storage budget a tile may claim when the mapping moves it
    /// out of per-thread storage. Zero keeps every tile where it is.
    pub shared_cap: i64,
}

impl MappingPolicy {
    /// One thread to a tile: every stage is identity, and the intrinsic
    /// surface stays whole for the dialect to spell.
    pub const fn serial() -> Self {
        Self {
            launch: LaunchShape::SERIAL,
            reduce: ReducePolicy::Serial,
            mma: MmaPolicy::Serial,
            vector: VectorPolicy::None,
            shared_cap: 0,
        }
    }

    /// True when any stage has work to do.
    pub fn is_parallel(&self) -> bool {
        self.launch.is_parallel()
    }
}

/// The mapping stages, in run order. One dump segment each.
pub const MAPPING_STAGES: [&str; 5] = ["policy", "threads", "reduce", "mma", "cleanup"];

/// Run the whole family a policy asks for.
///
/// A serial policy is identity: the function comes back untouched, which is
/// what keeps the C, Metal and PTX outputs byte-identical to the pre-mapping
/// pipeline.
pub fn run(func: Function, policy: &MappingPolicy) -> Result<Function, String> {
    (0..MAPPING_STAGES.len()).try_fold(func, |func, stage| run_stage(func, policy, stage))
}

/// Run one stage. `stage` indexes [`MAPPING_STAGES`]; the dump walks these.
pub fn run_stage(func: Function, policy: &MappingPolicy, stage: usize) -> Result<Function, String> {
    if !policy.is_parallel() {
        return Ok(func);
    }
    match stage {
        0 => Ok(insert_thread_id(func)),
        1 => strip_element_loops(func, policy),
        2 => match policy.reduce {
            ReducePolicy::Serial => Ok(func),
            ReducePolicy::WorkgroupTree => materialize_reduces(func, policy),
        },
        3 => match policy.mma {
            MmaPolicy::Serial => Ok(func),
            MmaPolicy::ThreadStrips => materialize_mmas(func, policy),
        },
        4 => {
            let mut func = crate::passes::loop_unroll::run(func);
            expand_owner_strips(&mut func, policy)?;
            privatize_owner_tiles(&mut func, policy)?;
            share_owner_tiles(&mut func, policy)?;
            crate::passes::storage::run(&mut func);
            crate::passes::canonicalize::eliminate_dead_insts(&mut func);
            Ok(func)
        }
        _ => Err(format!("no mapping stage {stage}")),
    }
}

/// The value as `a * tid + b` over the thread index, through the indirections
/// the mapping introduced. A stripped loop's index — a header parameter fed
/// the thread id on its entry edge, or a slot the entry stored the thread id
/// into — moves with the thread one step per iteration, so it spells as
/// `(1, 0)`; anything else stays outside the map and reads as unknown.
fn affine_over_tid(
    defs: &Defs<'_>,
    tid_feeds: &std::collections::HashMap<ValueId, ()>,
    value: &Operand,
) -> Option<(i64, i64)> {
    match value {
        Operand::Const(Constant::Int(c)) => Some((0, *c)),
        Operand::Value(id) if tid_feeds.contains_key(id) => Some((1, 0)),
        Operand::Value(id) => {
            let inst = defs.get(id)?;
            match &inst.op {
                InstOp::Bin { op: BinOp::Add, lhs, rhs } => {
                    let (a, b) = affine_over_tid(defs, tid_feeds, lhs)?;
                    let (c, d) = affine_over_tid(defs, tid_feeds, rhs)?;
                    Some((a + c, b + d))
                }
                InstOp::Bin { op: BinOp::Mul, lhs, rhs } => {
                    let (a, b) = affine_over_tid(defs, tid_feeds, lhs)?;
                    let (c, d) = affine_over_tid(defs, tid_feeds, rhs)?;
                    Some((a * d + b * c, b * d))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Move the tiles the threads strip in owner order into workgroup storage.
///
/// After the threads stage, a stripped tile's every access lands on the
/// accessing thread's own slots — that is what `check_strip_ownership`
/// demanded before the strip. But the tile's storage stayed per thread: a
/// 4096-element tile spelled as a private array is 16 KB of scratch in every
/// one of the workgroup's threads, megabytes of spilled globals a launch
/// pays for and never shares. One workgroup copy carries the same values.
///
/// The flip is safe exactly where every load and store on the tile spells
/// `tid + b` with `b` a multiple of the thread count — then no slot ever has
/// two threads' hands on it, and no barrier orders anything. That is the
/// same shape the ownership check admits, solved over the thread index
/// instead of a loop index, so a tile that fails it here stays private:
/// correct and slow, never wrong. Tiles over the policy's storage budget
/// stay private too — the adapter that must hold them may not have it.
///
/// The entry barrier exists for the contract's shape, shared paired with a
/// barrier, and for nothing else: no access here crosses threads.
/// The result-id to instruction map the mapping's analyses read.
fn defs_of(func: &Function) -> Defs<'_> {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect()
}

/// The values that move with the thread: the thread index itself, the header
/// parameters the entry edges feed it, and the loaded slot of an index alloca
/// the entry stored it into.
fn tid_feeds_of(func: &Function, tid: ValueId) -> std::collections::HashMap<ValueId, ()> {
    let mut tid_feeds: std::collections::HashMap<ValueId, ()> =
        std::collections::HashMap::new();
    tid_feeds.insert(tid, ());
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Store { ptr, value } = &inst.op
                && matches!(value, Operand::Value(v) if tid_feeds.contains_key(v))
                && let Some(loaded) = loads_of_alloca(func, ptr.clone())
            {
                tid_feeds.insert(loaded, ());
            }
        }
    }
    let dominators = crate::passes::cfg::dominators(func);
    for block in &func.blocks {
        let Some(param) = block.params.first() else {
            continue;
        };
        // The entry edge's first argument names what the index starts at; a
        // loop whose entry edge starts at the thread index moves with it.
        let fed = func.blocks.iter().any(|source| {
            source.id != block.id
                && !dominators[&source.id].contains(&block.id)
                && terminator_arg(&source.terminator, block.id, 0)
                    .is_some_and(|arg| matches!(arg, Operand::Value(v) if *v == param.id || *v == tid))
        });
        if fed {
            tid_feeds.insert(param.id, ());
        }
    }
    tid_feeds
}

/// A plumbing loop the threads stage produced: one induction starting at the
/// thread index, stepping by the thread count to a constant extent.
struct StripLoop {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    /// The false edge's arguments: what the exit block reads after the loop.
    exit_args: Vec<Operand>,
    /// The induction value: a header parameter (phi form) or a load of the
    /// index slot (alloca form).
    induction: ValueId,
    /// The alloca-form induction's slot, whose step store the copies drop.
    slot: Option<ValueId>,
    /// The loop's start value: the thread index or a value that moves with it.
    start: Operand,
    step: i64,
    extent: i64,
}

/// Find one unexpanded owner-strip plumbing loop, or `None` when none are
/// left. The shape is narrow on purpose — one block of body, no barrier in
/// it, the induction stepping by exactly the thread count — because the
/// expansion copies the body once per iteration and anything wider would
/// need a remapper this pass does not carry.
fn find_strip_loop(
    func: &Function,
    tid: ValueId,
    tid_feeds: &std::collections::HashMap<ValueId, ()>,
    threads: i64,
    declined: &std::collections::HashSet<BlockId>,
) -> Option<StripLoop> {
    let moving = |operand: &Operand| {
        matches!(operand, Operand::Value(v) if *v == tid || tid_feeds.contains_key(v))
    };
    for header in &func.blocks {
        if declined.contains(&header.id) {
            continue;
        }
        // The phi form: the induction is a header parameter tested against a
        // constant, fed by the preheader and stepped by the body.
        if let Some(plan) = (|| {
            let param = header.params.first()?.id;
            let [cmp_inst] = header.insts.as_slice() else {
                return None;
            };
            let InstOp::Cmp { pred: CmpPred::Slt, lhs, rhs } = &cmp_inst.op else {
                return None;
            };
            if !matches!(lhs, Operand::Value(v) if *v == param)
            {
                return None;
            }
            let Operand::Const(Constant::Int(extent)) = rhs else {
                return None;
            };
            let Terminator::CondBr {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } = &header.terminator
            else {
                return None;
            };
            let body = *true_target;
            // An edge may hand the body the induction, or the body may read
            // the header's parameter directly.
            let induction = match true_args.as_slice() {
                [] => Operand::Value(param),
                [arg] if moving(arg) => arg.clone(),
                _ => return None,
            };
            let Operand::Value(induction_value) = &induction else {
                return None;
            };
            // The body steps by the thread count and jumps straight back.
            let body_block = func.blocks.iter().find(|block| block.id == body)?;
            if !matches!(&body_block.terminator, Terminator::Br { target, args }
                if *target == header.id && args.len() == 1)
            {
                return None;
            }
            let step_value = match &body_block.terminator {
                Terminator::Br { args, .. } => match &args[0] {
                    Operand::Value(v) => *v,
                    _ => return None,
                },
                _ => return None,
            };
            let step = body_block.insts.iter().find_map(|inst| {
                if inst.result != Some(step_value) {
                    return None;
                }
                match &inst.op {
                    InstOp::Bin {
                        op: BinOp::Add,
                        lhs,
                        rhs,
                    } => match (lhs, rhs) {
                        (Operand::Value(v), Operand::Const(Constant::Int(amount)))
                            if *v == param && *amount == threads =>
                        {
                            Some(*amount)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            })?;
            let preheader = func.blocks.iter().find(|block| {
                matches!(&block.terminator, Terminator::Br { target, args }
                    if *target == header.id && args.len() == 1 && moving(&args[0]))
            })?;
            let start = match &preheader.terminator {
                Terminator::Br { args, .. } => args[0].clone(),
                _ => return None,
            };
            Some(StripLoop {
                preheader: preheader.id,
                header: header.id,
                body,
                exit: *false_target,
                exit_args: false_args.clone(),
                induction: *induction_value,
                slot: None,
                start,
                step,
                extent: *extent,
            })
        })() {
            return Some(plan);
        }
        // The alloca form: the induction is a private slot the preheader
        // stores the start into and the body loads, steps, and stores back.
        if let Some(plan) = (|| {
            let [load_inst, cmp_inst] = header.insts.as_slice() else {
                return None;
            };
            let InstOp::Load { ptr } = &load_inst.op else {
                return None;
            };
            let Operand::Value(slot) = ptr else {
                return None;
            };
            let at = load_inst.result?;
            let InstOp::Cmp { pred: CmpPred::Slt, lhs, rhs } = &cmp_inst.op else {
                return None;
            };
            if !matches!(lhs, Operand::Value(v) if *v == at) {
                return None;
            }
            let Operand::Const(Constant::Int(extent)) = rhs else {
                return None;
            };
            let Terminator::CondBr {
                true_target,
                false_target,
                false_args,
                ..
            } = &header.terminator
            else {
                return None;
            };
            let body = *true_target;
            let body_block = func.blocks.iter().find(|block| block.id == body)?;
            if !matches!(&body_block.terminator, Terminator::Br { target, .. } if *target == header.id)
            {
                return None;
            }
            // The preheader stores the start into the slot and jumps here.
            let store_is_start = |inst: &Inst| match &inst.op {
                InstOp::Store { ptr, value } => {
                    matches!(ptr, Operand::Value(v) if *v == *slot) && moving(value)
                }
                _ => false,
            };
            let preheader = func.blocks.iter().find(|block| {
                matches!(&block.terminator, Terminator::Br { target, .. } if *target == header.id)
                    && block.insts.iter().any(store_is_start)
            })?;
            let start = preheader
                .insts
                .iter()
                .find(|inst| store_is_start(inst))
                .map(|inst| match &inst.op {
                    InstOp::Store { value, .. } => value.clone(),
                    _ => unreachable!("found by store_is_start"),
                })?;
            // The body loads the slot, steps it by the thread count, and
            // stores the step back into it.
            let step_value = body_block.insts.iter().find_map(|inst| {
                let result = inst.result?;
                match &inst.op {
                    InstOp::Bin {
                        op: BinOp::Add,
                        lhs,
                        rhs,
                    } => match (lhs, rhs) {
                        (Operand::Value(v), Operand::Const(Constant::Int(amount)))
                            if *v == at && *amount == threads =>
                        {
                            Some(result)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            })?;
            if !body_block.insts.iter().any(|inst| {
                matches!(&inst.op, InstOp::Store { ptr, value }
                    if matches!(ptr, Operand::Value(v) if *v == *slot)
                        && matches!(value, Operand::Value(v) if *v == step_value))
            }) {
                return None;
            }
            Some(StripLoop {
                preheader: preheader.id,
                header: header.id,
                body,
                exit: *false_target,
                exit_args: false_args.clone(),
                induction: at,
                slot: Some(*slot),
                start,
                step: threads,
                extent: *extent,
            })
        })() {
            return Some(plan);
        }
    }
    None
}

/// Fully unroll the owner-strip plumbing loops.
///
/// The threads stage rewrites a 0..extent element loop into `tid, tid + W,
/// …`, and the reduce stage spells its strip partials the same way. Their
/// trip count is `extent / W`, known at compile time, and unrolling turns
/// every induction use into `tid + i * W` with `i` a constant. That constant
/// is what [`privatize_owner_tiles`] divides a tile slot by to get the
/// thread's fragment index, and a fragment indexed by constants is what the
/// backends can keep in registers instead of shared memory. Triton's mapped
/// loops unroll for the same reason: the per-thread loop exists only to be
/// straight-line code.
///
/// The matcher is narrow — one block of body, a single induction starting at
/// the thread index, no barrier in the body — so a loop it does not
/// recognize stays a loop: a missed unroll, never a wrong one.
fn expand_owner_strips(func: &mut Function, policy: &MappingPolicy) -> Result<(), String> {
    let threads = policy.launch.count() as i64;
    if threads < 2 {
        return Ok(());
    }
    let Some(tid) = thread_id_of(func) else {
        return Ok(());
    };
    let mut declined: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
    loop {
        let tid_feeds = tid_feeds_of(func, tid);
        let Some(plan) = find_strip_loop(func, tid, &tid_feeds, threads, &declined) else {
            return Ok(());
        };
        let body_block = func
            .blocks
            .iter()
            .find(|block| block.id == plan.body)
            .expect("the plan's body block");
        let barrier_in_body = body_block.insts.iter().any(|inst| {
            matches!(
                &inst.op,
                InstOp::Intrinsic {
                    intrinsic: Intrinsic::Barrier { .. },
                    ..
                }
            ) || matches!(&inst.op, InstOp::Call { func: callee, .. } if callee.contains("barrier"))
        });
        let trip = plan.extent / plan.step;
        if barrier_in_body
            || plan.extent % plan.step != 0
            || trip < 1
            || trip > 16
            || body_block.insts.len() * trip as usize > 400
            || plan.exit_args.iter().any(|operand| {
                matches!(operand, Operand::Value(v) if *v == plan.induction)
            })
        {
            // A loop this pass declines keeps its form: record it so the
            // search moves on instead of offering it again.
            declined.insert(plan.header);
            continue;
        }
        expand_strip_once(func, &plan, trip);
    }
}

/// Copy one strip loop's body once per iteration, straight-line. The
/// induction's uses in iteration `i` read `start + i * step`, spelled as a
/// fresh add (or the start itself for the first copy); every other value the
/// body defines gets a fresh id per copy, so the copies are independent.
/// The alloca form's step store dies with the loop — the copies spell the
/// sequence the stores would have produced.
fn expand_strip_once(func: &mut Function, plan: &StripLoop, trip: i64) {
    let body_source = func
        .blocks
        .iter()
        .find(|block| block.id == plan.body)
        .expect("the plan's body block")
        .clone();
    let mut next_value = next_value_id_of(func);
    let mut next_block = func
        .blocks
        .iter()
        .map(|block| block.id.0)
        .max()
        .unwrap_or(1)
        + 1;
    // The value the induction reads in each copy, with any add the spelling
    // needs ahead of the copy's body.
    let mut starts: Vec<Operand> = Vec::new();
    let mut prologues: Vec<Vec<Inst>> = Vec::new();
    for iteration in 0..trip {
        let offset = iteration * plan.step;
        match &plan.start {
            Operand::Value(start_value) if offset == 0 => {
                starts.push(Operand::Value(*start_value));
                prologues.push(Vec::new());
            }
            Operand::Value(start_value) => {
                let id = ValueId(next_value);
                next_value += 1;
                prologues.push(vec![Inst {
                    result: Some(id),
                    result_name: Some(format!("strip_at{}", id.0)),
                    ty: Type::I64,
                    op: InstOp::Bin {
                        op: BinOp::Add,
                        lhs: Operand::Value(*start_value),
                        rhs: Operand::Const(Constant::Int(offset)),
                    },
                    metadata: Vec::new(),
                }]);
                starts.push(Operand::Value(id));
            }
            Operand::Const(Constant::Int(base)) => {
                starts.push(Operand::Const(Constant::Int(base + offset)));
                prologues.push(Vec::new());
            }
            Operand::Const(_) => return,
        }
    }
    let mut copies: Vec<BasicBlock> = Vec::new();
    // The last copy's value table, for the exit edge: a value the exit reads
    // through the header's false arguments names the last copy's version.
    let mut last_map: std::collections::HashMap<ValueId, Operand> =
        std::collections::HashMap::new();
    for iteration in 0..trip {
        // Per-copy value remapping: the induction reads its start, and a
        // body-defined value reads this copy's fresh version.
        let mut value_map: std::collections::HashMap<ValueId, Operand> =
            std::collections::HashMap::new();
        value_map.insert(plan.induction, starts[iteration as usize].clone());
        let mut insts = prologues[iteration as usize].clone();
        for source_inst in &body_source.insts {
            if let Some(result) = source_inst.result {
                let fresh = ValueId(next_value);
                next_value += 1;
                value_map.insert(result, Operand::Value(fresh));
            }
        }
        // Now every body-defined result has its fresh id, so a second pass
        // over the instructions can remap operands against the full table.
        for source_inst in &body_source.insts {
            // The alloca form's step store dies with the loop: the copies
            // spell the sequence the store would have produced.
            if let InstOp::Store { ptr, value } = &source_inst.op
                && plan.slot.is_some()
                && matches!(ptr, Operand::Value(v) if Some(*v) == plan.slot)
                && matches!(value, Operand::Value(_))
            {
                continue;
            }
            let mut cloned = source_inst.clone();
            cloned.op = remap_inst_op(&source_inst.op, &value_map);
            if let Some(result) = source_inst.result {
                cloned.result = match value_map.get(&result) {
                    Some(Operand::Value(fresh)) => Some(*fresh),
                    _ => None,
                };
                cloned.result_name = source_inst
                    .result_name
                    .clone()
                    .map(|name| format!("{name}_u{}", cloned.result.map(|id| id.0).unwrap_or(0)));
            }
            insts.push(cloned);
        }
        copies.push(BasicBlock {
            id: BlockId(next_block),
            name: format!("{}_u{}", body_source.name, iteration),
            params: Vec::new(),
            insts,
            terminator: Terminator::Br {
                target: BlockId(next_block + 1),
                args: Vec::new(),
            },
        });
        if iteration + 1 == trip {
            last_map = value_map;
        }
        next_block += 1;
    }
    // The last copy falls through to the loop's exit with the values the
    // header's false edge carried; every earlier one into the next copy.
    let exit_args = plan
        .exit_args
        .iter()
        .map(|operand| match operand {
            Operand::Value(value) => last_map
                .get(value)
                .cloned()
                .unwrap_or_else(|| Operand::Value(*value)),
            Operand::Const(_) => operand.clone(),
        })
        .collect();
    if let Some(last_copy) = copies.last_mut() {
        last_copy.terminator = Terminator::Br {
            target: plan.exit,
            args: exit_args,
        };
    }
    // The preheader enters the first copy; the loop's own blocks drop out.
    if let Some(preheader) = func
        .blocks
        .iter_mut()
        .find(|block| block.id == plan.preheader)
    {
        preheader.terminator = Terminator::Br {
            target: copies[0].id,
            args: Vec::new(),
        };
    }
    let loop_blocks: std::collections::HashSet<BlockId> = [plan.header, plan.body].into();
    let insert_at = func
        .blocks
        .iter()
        .position(|block| block.id == plan.header)
        .unwrap_or(func.blocks.len());
    func.blocks.retain(|block| !loop_blocks.contains(&block.id));
    let insert_at = insert_at.min(func.blocks.len());
    for (offset, block) in copies.into_iter().enumerate() {
        func.blocks.insert(insert_at + offset, block);
    }
}

fn remap_inst_op(
    op: &InstOp,
    value_map: &std::collections::HashMap<ValueId, Operand>,
) -> InstOp {
    let operand = |operand: &Operand| match operand {
        Operand::Value(value) => value_map
            .get(value)
            .cloned()
            .unwrap_or_else(|| Operand::Value(*value)),
        Operand::Const(_) => operand.clone(),
    };
    match op {
        InstOp::Alloca { alloc_ty, addr_space } => InstOp::Alloca {
            alloc_ty: alloc_ty.clone(),
            addr_space: addr_space.clone(),
        },
        InstOp::Gep { base, indices } => InstOp::Gep {
            base: operand(base),
            indices: indices.iter().map(|index| operand(index)).collect(),
        },
        InstOp::Load { ptr } => InstOp::Load {
            ptr: operand(ptr),
        },
        InstOp::Store { ptr, value } => InstOp::Store {
            ptr: operand(ptr),
            value: operand(value),
        },
        InstOp::AtomicAdd { ptr, value } => InstOp::AtomicAdd {
            ptr: operand(ptr),
            value: operand(value),
        },
        InstOp::Memcpy { dst, src, size } => InstOp::Memcpy {
            dst: operand(dst),
            src: operand(src),
            size: operand(size),
        },
        InstOp::Bin { op, lhs, rhs } => InstOp::Bin {
            op: *op,
            lhs: operand(lhs),
            rhs: operand(rhs),
        },
        InstOp::Cmp { pred, lhs, rhs } => InstOp::Cmp {
            pred: *pred,
            lhs: operand(lhs),
            rhs: operand(rhs),
        },
        InstOp::Cast { op, value, to } => InstOp::Cast {
            op: *op,
            value: operand(value),
            to: to.clone(),
        },
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => InstOp::Select {
            cond: operand(cond),
            on_true: operand(on_true),
            on_false: operand(on_false),
        },
        InstOp::Call { func, args } => InstOp::Call {
            func: func.clone(),
            args: args.iter().map(|arg| operand(arg)).collect(),
        },
        InstOp::Intrinsic { intrinsic, args } => InstOp::Intrinsic {
            intrinsic: intrinsic.clone(),
            args: args.iter().map(|arg| operand(arg)).collect(),
        },
    }
}

fn next_value_id_of(func: &Function) -> u32 {
    let mut max_value = 0u32;
    for param in &func.params {
        max_value = max_value.max(param.id.0);
    }
    for block in &func.blocks {
        for param in &block.params {
            max_value = max_value.max(param.id.0);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                max_value = max_value.max(result.0);
            }
        }
    }
    max_value + 1
}

/// Shrink every owner-ordered tile to the thread's own fragment.
///
/// A stripped tile is `[len]` of private storage in every thread, though each
/// thread touches only the `len / W` slots it owns: slot `tid + k * W` for
/// the constant `k` the unrolled strip spells. Shrinking it to `[len / W]`
/// and rewriting slot `k * W` to index `k` makes the access indexed by a
/// constant, which is what lets a backend hold the fragment in registers
/// instead of an array in a spill of per-thread storage. That is Triton's
/// `sizePerThread` fragment: the layout says which elements each thread owns,
/// and the ones it does not own need no storage.
///
/// The rewrite is safe exactly where the owner map is `tid + k * W` with the
/// same `k` for every access on the tile, since then the slot each thread
/// writes is the slot it later reads, at the same index. Fail closed: a tile
/// with one access outside that shape stays whole, private, and slow.
///
/// The tiles this pass takes are exactly the ones [`share_owner_tiles`] would
/// otherwise flip into workgroup storage, so it must run first: the fragment
/// is the better placement, and the flip is the floor it leaves behind.
fn privatize_owner_tiles(func: &mut Function, policy: &MappingPolicy) -> Result<(), String> {
    let threads = policy.launch.count() as i64;
    if threads < 2 || policy.shared_cap <= 0 {
        return Ok(());
    }
    let Some(tid) = thread_id_of(func) else {
        return Ok(());
    };
    let defs: Defs<'_> = defs_of(func);
    let tid_feeds = tid_constant_feeds(func, tid);

    // The private f32 tiles within the budget, and a whole number of threads'
    // worth of elements each.
    let candidates: Vec<ValueId> = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| {
            matches!(
                &inst.op,
                InstOp::Alloca { alloc_ty: Type::Array { elem, len }, addr_space: AddressSpace::Private }
                    if **elem == Type::F32
                        && (*len * 4) as i64 <= policy.shared_cap
                        && *len as i64 % threads == 0
            )
        })
        .filter_map(|inst| inst.result)
        .collect();
    if candidates.is_empty() {
        return Ok(());
    }

    // Every access on a candidate must index the owner's own slot: the slot
    // then reads as a constant fragment index, and the tile shrinks to one
    // slot per thread.
    let mut fragment: std::collections::HashMap<ValueId, i64> = std::collections::HashMap::new();
    // Each access spells its own fragment index; the tile keeps the highest so
    // the storage covers them all.
    let mut access: std::collections::HashMap<ValueId, (ValueId, i64)> =
        std::collections::HashMap::new();
    let mut failed: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            let InstOp::Gep { base, indices } = &inst.op else {
                continue;
            };
            let Operand::Value(base_id) = base else {
                continue;
            };
            if !candidates.contains(base_id) || failed.contains(base_id) {
                continue;
            }
            if indices.len() != 1 {
                failed.insert(*base_id);
                continue;
            }
            // A pointer that escapes would address slots the thread does not
            // own; only a load or store may consume it, and only as its
            // pointer.
            let consumed = |op: &InstOp| {
                matches!(op, InstOp::Load { ptr: Operand::Value(v) } | InstOp::Store { ptr: Operand::Value(v), .. } if *v == result)
            };
            let escapes = func.blocks.iter().any(|block| {
                block
                    .insts
                    .iter()
                    .any(|other| other.result != Some(result) && inst_mentions(&other.op, result) && !consumed(&other.op))
                    || terminator_mentions(&block.terminator, result)
            });
            if escapes {
                failed.insert(*base_id);
                continue;
            }
            let Some((a, b)) = affine_over_tid(&defs, &tid_feeds, &indices[0]) else {
                failed.insert(*base_id);
                continue;
            };
            if a != 1 || b % threads != 0 {
                failed.insert(*base_id);
                continue;
            }
            let k = b / threads;
            // The slot reads as the thread's fragment `k`. The map is the same
            // for every access — slot `tid + k * W` is fragment `k` — so one
            // thread's write and read of a slot still meet. A slot past the
            // shrunken tile would go out of bounds, and the tile keeps its
            // size for that.
            let slots = match func
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .find(|other| other.result == Some(*base_id))
                .map(|other| &other.op)
            {
                Some(InstOp::Alloca { alloc_ty: Type::Array { len, .. }, .. }) => *len as i64,
                _ => {
                    failed.insert(*base_id);
                    continue;
                }
            };
            if k >= slots / threads {
                failed.insert(*base_id);
                continue;
            }
            access.insert(result, (*base_id, k));
            if let Some(highest) = fragment.get(base_id) {
                if k > *highest {
                    fragment.insert(*base_id, k);
                }
            } else {
                fragment.insert(*base_id, k);
            }
        }
    }
    // An alloca pointer used as anything but a gep base cannot be shrunk.
    for block in &func.blocks {
        for inst in &block.insts {
            for candidate in &candidates {
                let is_gep_base = matches!(&inst.op, InstOp::Gep { base: Operand::Value(v), .. } if v == candidate);
                if inst_mentions(&inst.op, *candidate) && !is_gep_base {
                    failed.insert(*candidate);
                }
            }
        }
        for candidate in &candidates {
            if terminator_mentions(&block.terminator, *candidate) {
                failed.insert(*candidate);
            }
        }
    }

    let taken: std::collections::HashMap<ValueId, i64> = fragment
        .into_iter()
        .filter(|(id, _)| !failed.contains(id))
        .collect();
    if taken.is_empty() {
        return Ok(());
    }
    let threads = threads as usize;
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            if let Some(result) = inst.result
                && taken.contains_key(&result)
            {
                let InstOp::Alloca { alloc_ty, addr_space } = &mut inst.op else {
                    continue;
                };
                if let Type::Array { len, .. } = alloc_ty {
                    *len /= threads;
                    inst.ty = Type::ptr(addr_space.clone(), alloc_ty.clone());
                }
            }
        }
    }
    // The fragment index is a constant, so each access is a constant index
    // into the shrunken tile.
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            let Some((base, k)) = access.get(&result) else {
                continue;
            };
            if !taken.contains_key(base) {
                continue;
            }
            let InstOp::Gep { indices, .. } = &mut inst.op else {
                continue;
            };
            if indices.len() == 1 {
                indices[0] = Operand::Const(Constant::Int(*k));
            }
        }
    }
    Ok(())
}

fn inst_mentions(op: &InstOp, value: ValueId) -> bool {
    let mut operands = Vec::new();
    crate::passes::sroa::collect_op_operands(op, &mut operands);
    operands
        .iter()
        .any(|operand| matches!(operand, Operand::Value(used) if *used == value))
}

fn terminator_mentions(terminator: &Terminator, value: ValueId) -> bool {
    let mut operands = Vec::new();
    crate::passes::sroa::collect_terminator_operands(terminator, &mut operands);
    operands
        .iter()
        .any(|operand| matches!(operand, Operand::Value(used) if *used == value))
}

/// The values that hold the thread index and never change: the thread index
/// itself and the loaded slot of an index alloca one store seeded with it.
///
/// A loop induction that merely starts at the thread index is left out. It
/// walks every fragment, not one, so an access through it is no constant
/// fragment index.
fn tid_constant_feeds(func: &Function, tid: ValueId) -> std::collections::HashMap<ValueId, ()> {
    let mut stores: std::collections::HashMap<ValueId, usize> = std::collections::HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Store { ptr: Operand::Value(p), .. } = &inst.op {
                *stores.entry(*p).or_default() += 1;
            }
        }
    }
    let mut feeds: std::collections::HashMap<ValueId, ()> = std::collections::HashMap::new();
    feeds.insert(tid, ());
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Store { ptr: Operand::Value(p), value } = &inst.op
                && matches!(value, Operand::Value(v) if *v == tid)
                && !stores.get(p).is_some_and(|count| *count > 1)
                && let Some(loaded) = loads_of_alloca(func, Operand::Value(*p))
            {
                feeds.insert(loaded, ());
            }
        }
    }
    feeds
}

fn share_owner_tiles(func: &mut Function, policy: &MappingPolicy) -> Result<(), String> {
    let threads = policy.launch.count() as i64;
    if threads < 2 || policy.shared_cap <= 0 {
        return Ok(());
    }
    let tid = thread_id_of(func).ok_or_else(|| "the threads stage never ran".to_string())?;
    let defs: Defs<'_> = defs_of(func);
    let tid_feeds = tid_feeds_of(func, tid);

    // The private f32 tiles within the budget.
    let candidates: Vec<ValueId> = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| {
            matches!(
                &inst.op,
                InstOp::Alloca { alloc_ty: Type::Array { elem, len }, addr_space: AddressSpace::Private }
                    if **elem == Type::F32 && (*len * 4) as i64 <= policy.shared_cap
            )
        })
        .filter_map(|inst| inst.result)
        .collect();

    // Every access on a candidate must be the owner's own slot.
    let mut conformant: std::collections::HashMap<ValueId, bool> =
        candidates.iter().map(|id| (*id, true)).collect();
    for block in &func.blocks {
        for inst in &block.insts {
            let (ptr, _) = match &inst.op {
                InstOp::Store { ptr, .. } => (ptr, "stores"),
                InstOp::Load { ptr } => (ptr, "loads"),
                _ => continue,
            };
            let Operand::Value(g) = ptr else {
                continue;
            };
            let Some(gep) = defs.get(g) else {
                continue;
            };
            // The reduce tree's fold reads are owner ordered by construction —
            // it checked the extent against the thread count before spelling
            // `tid + k * threads` — so its stamp stands in for the arithmetic
            // the row loop hides.
            if gep.metadata.contains(&Metadata::OwnerOrdered) {
                continue;
            }
            let Some(InstOp::Gep { base, indices, .. }) = Some(&gep.op) else {
                continue;
            };
            let Operand::Value(base_id) = base else {
                continue;
            };
            let Some(true) = conformant.get_mut(base_id) else {
                continue;
            };
            let Some(slot) = indices.last() else {
                continue;
            };
            let Some((a, b)) = affine_over_tid(&defs, &tid_feeds, slot) else {
                *conformant.get_mut(base_id).expect("checked above") = false;
                continue;
            };
            if a != 1 || b % threads != 0 {
                *conformant.get_mut(base_id).expect("checked above") = false;
            }
        }
    }

    let flipped: Vec<ValueId> = candidates
        .into_iter()
        .filter(|id| conformant[id])
        .collect();
    if flipped.is_empty() {
        return Ok(());
    }
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            if let Some(result) = inst.result
                && flipped.contains(&result)
            {
                let InstOp::Alloca {
                    alloc_ty,
                    addr_space,
                } = &mut inst.op
                else {
                    continue;
                };
                *addr_space = AddressSpace::Shared;
                inst.ty = Type::ptr(AddressSpace::Shared, alloc_ty.clone());
            }
        }
    }
    // No access here crosses threads, so no barrier orders anything — but
    // the contract pairs shared storage with a barrier, and a kernel the
    // reduce or mma stages already ordered carries one anyway.
    let has_barrier = func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| {
            matches!(
                &inst.op,
                InstOp::Intrinsic { intrinsic: Intrinsic::Barrier { .. }, .. }
            )
        })
    });
    if !has_barrier {
        let entry = func
            .blocks
            .iter_mut()
            .find(|block| block.id == func.entry)
            .expect("a function has an entry block");
        let index = entry
            .insts
            .iter()
            .position(|inst| inst.result == Some(tid))
            .map(|at| at + 1)
            .unwrap_or(0);
        entry.insts.insert(
            index,
            Inst {
                result: None,
                result_name: None,
                ty: Type::Void,
                op: InstOp::Intrinsic {
                    intrinsic: Intrinsic::Barrier {
                        scope: SyncScope::Workgroup,
                    },
                    args: Vec::new(),
                },
                metadata: Vec::new(),
            },
        );
    }
    Ok(())
}

/// The value a load of this pointer's alloca produces, when the pointer is
/// the index slot the `open_loop` shape reads directly.
fn loads_of_alloca(func: &Function, ptr: Operand) -> Option<ValueId> {
    let Operand::Value(p) = ptr else {
        return None;
    };
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| matches!(&inst.op, InstOp::Load { ptr: Operand::Value(v) } if *v == p))
        .filter_map(|inst| inst.result)
        .next()
}

/// The argument an edge to `target` hands its position `at`, when this
/// terminator branches there.
fn terminator_arg(terminator: &Terminator, target: BlockId, at: usize) -> Option<&Operand> {
    match terminator {
        Terminator::Br { target: t, args } if *t == target => args.get(at),
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == target {
                true_args.get(at)
            } else if *false_target == target {
                false_args.get(at)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The output buffer's tile shape, which the workgroup capacity comes from.
///
/// The last kernel buffer whose ABI names tile dimensions is the output, the
/// same rule the backends' `find_output_buffer` applies.
pub fn output_tile(func: &Function) -> Option<(usize, usize)> {
    func.params
        .iter()
        .rev()
        .filter(|param| !is_descriptor_param(param))
        .find_map(|param| param.abi.as_ref().and_then(|abi| abi.tile))
}

fn is_descriptor_param(param: &crate::ir::Param) -> bool {
    param.name == crate::ir::SHAPES_PARAM_NAME
        || param.name == crate::ir::SCALARS_PARAM_NAME
        || param.name == crate::ir::SCALARS_I64_PARAM_NAME
}

/// The storage and element type of a tile pointer, when `ty` names one.
fn tile_elem(ty: &Type) -> Option<(AddressSpace, Type)> {
    match ty {
        Type::Ptr { addr_space, pointee } => match pointee.as_ref() {
            Type::Array { elem, .. } => Some((addr_space.clone(), elem.as_ref().clone())),
            _ => None,
        },
        _ => None,
    }
}

/// The element type at the bottom of a (possibly nested) array.
fn base_elem(ty: &Type) -> Type {
    let mut current = ty;
    while let Type::Array { elem, .. } = current {
        current = elem;
    }
    current.clone()
}

/// The address space and element type the tile storage `id` names.
///
/// A tile alloca spells its element one-wide in the instruction's own pointer
/// type, with the full array only in its `alloc_ty`; a buffer parameter spells
/// it in a (possibly nested) array. Both flatten to one element type under
/// the gep indices the mapping builds.
/// One resolved slot pointer: the space and element type it names.
fn tile_element_of(func: &Function, defs: &Defs<'_>, id: ValueId) -> Option<(AddressSpace, Type)> {
    // A slot pointer is a gep into a tile, so the space and element live one
    // link up the chain — on the alloca (or the launch's buffer parameter),
    // not on the gep, whose own type already points at the scalar element.
    let mut current = id;
    for _ in 0..4 {
        let inst = defs.get(&current).copied();
        let Some(inst) = inst else {
            return func
                .params
                .iter()
                .find(|param| param.id == current)
                .and_then(|param| tile_elem(&param.ty))
                .map(|(space, elem)| (space, base_elem(&elem)));
        };
        match &inst.op {
            InstOp::Alloca { alloc_ty, addr_space } => {
                return Some((addr_space.clone(), base_elem(alloc_ty)));
            }
            InstOp::Gep { base, .. } => match base {
                Operand::Value(prev) => current = *prev,
                _ => return None,
            },
            _ => return tile_elem(&inst.ty).map(|(space, elem)| (space, base_elem(&elem))),
        }
    }
    None
}

/// The type of a value, following parameters and instruction results.
fn value_type(func: &Function, id: ValueId) -> Option<&Type> {
    if let Some(param) = func.params.iter().find(|param| param.id == id) {
        return Some(&param.ty);
    }
    func.blocks.iter().find_map(|block| {
        block
            .params
            .iter()
            .find(|param| param.id == id)
            .map(|param| &param.ty)
            .or_else(|| {
                block
                    .insts
                    .iter()
                    .find(|inst| inst.result == Some(id))
                    .map(|inst| &inst.ty)
            })
    })
}

fn max_value_id(func: &Function) -> u32 {
    func.blocks
        .iter()
        .flat_map(|block| {
            block
                .params
                .iter()
                .map(|param| param.id)
                .chain(block.insts.iter().filter_map(|inst| inst.result))
        })
        .chain(func.params.iter().map(|param| param.id))
        .map(|id| id.0)
        .max()
        .unwrap_or(0)
        + 1
}

fn max_block_id(func: &Function) -> u32 {
    func.blocks
        .iter()
        .map(|block| block.id.0)
        .max()
        .unwrap_or(0)
        + 1
}

fn thread_id_of(func: &Function) -> Option<ValueId> {
    func.blocks.iter().find_map(|block| {
        block
            .insts
            .iter()
            .find(|inst| {
                matches!(
                    &inst.op,
                    InstOp::Intrinsic {
                        intrinsic: Intrinsic::ThreadId { dim: 0 },
                        ..
                    }
                )
            })
            .and_then(|inst| inst.result)
    })
}

fn find_block_with_inst(func: &Function, id: ValueId) -> Option<BlockId> {
    func.blocks
        .iter()
        .find(|block| block.insts.iter().any(|inst| inst.result == Some(id)))
        .map(|block| block.id)
}

// ── stage 0: policy ──────────────────────────────────────────────────────

/// The flat thread index, in the entry block where every loop can reach it.
fn insert_thread_id(mut func: Function) -> Function {
    let tid = ValueId(max_value_id(&func));
    let entry = func
        .blocks
        .iter_mut()
        .find(|block| block.id == func.entry)
        .expect("a function has an entry block");
    entry.insts.insert(
        0,
        Inst {
            result: Some(tid),
            result_name: Some("sile_tid".to_string()),
            ty: Type::I64,
            op: InstOp::Intrinsic {
                intrinsic: Intrinsic::ThreadId { dim: 0 },
                args: Vec::new(),
            },
            metadata: Vec::new(),
        },
    );
    func
}

// ── stage 1: threads ─────────────────────────────────────────────────────

/// Loop name prefixes the lowering emits for per-element tile work.
///
/// The instruction that defines each value, indexed once per pass: the
/// ownership checks resolve operands to definitions, and a flat scan per
/// access turned the causal fmha's multi-thousand-instruction IR into a crawl.
type Defs<'a> = std::collections::HashMap<ValueId, &'a Inst>;

/// Each of these loops writes every element of a tile exactly once, from
/// values that do not depend on the iteration order — that is what makes the
/// strip legal. A user `for` region is absent on purpose: its body may carry
/// a dependency between iterations (a gemm's contraction step), and the
/// lowering spells it with a `for` prefix.
const PLUMBING_PREFIXES: [&str; 11] = [
    "mt", "mtr", "mi", "vc", "sv", "rd", "cp", "cc", "rs", "iota", "ys",
];

fn is_plumbing_header(name: &str) -> bool {
    // The lowering suffixes every block name with its block id (`mt_header_2`),
    // so the match is a prefix one. The prefixes themselves never shadow one
    // another: `mtr` differs from `mt` at the letter the `_header` tail lands
    // on.
    PLUMBING_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(&format!("{prefix}_header")))
}

/// The smallest constant trip count any plumbing loop runs, which no thread
/// count should overshoot: a thread past a loop's extent idles through it
/// while still paying for its barrier rounds, and a slot a loop bases off a
/// row start (`index + row`) stays in bounds only while the count divides the
/// row pitch. Single-iteration loops take no part in the count: they have no
/// parallelism to divide, and their presence would pin every kernel with a
/// one-tile relay to the serial mapping.
pub fn plumbing_extent(func: &Function) -> Option<i64> {
    let mut limits = Vec::new();
    for header in func
        .blocks
        .iter()
        .filter(|block| is_plumbing_header(&block.name))
    {
        let Terminator::CondBr {
            cond: Operand::Value(cond_id),
            ..
        } = &header.terminator
        else {
            continue;
        };
        for inst in &header.insts {
            if inst.result == Some(*cond_id)
                && let InstOp::Cmp {
                    rhs: Operand::Const(Constant::Int(limit)),
                    ..
                } = &inst.op
                && *limit > 1
            {
                limits.push(*limit);
            }
        }
    }
    limits.into_iter().min()
}

/// Strip the lowering's element loops across the threads.
fn strip_element_loops(mut func: Function, policy: &MappingPolicy) -> Result<Function, String> {
    let step = policy.launch.count() as i64;
    scalarize_vec_relays(&mut func)?;
    check_strip_ownership(&func, step)?;
    let headers: Vec<BlockId> = func
        .blocks
        .iter()
        .filter(|block| is_plumbing_header(&block.name))
        .map(|block| block.id)
        .collect();
    for header in headers {
        strip_loop(&mut func, header, step)?;
    }
    let leftovers = func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| {
            matches!(
                &inst.op,
                InstOp::Intrinsic {
                    intrinsic: Intrinsic::VecLoad { .. } | Intrinsic::VecStore { .. },
                    ..
                }
            )
        })
    });
    if leftovers {
        return Err(
            "a vector load or store survived relay scalarization, so its owner map would \
             disagree with the strips"
                .into(),
        );
    }
    Ok(func)
}

/// A `vec4` relay loop becomes the scalar copy it always was on this target.
///
/// The WGSL spelling of a vector load is one read per lane, so the vector
/// form buys nothing here — and its owner map would disagree with the scalar
/// strips'. The loop folds to one element per iteration and strips like any
/// other plumbing loop.
fn scalarize_vec_relays(func: &mut Function) -> Result<(), String> {
    let header_ids: Vec<BlockId> = func
        .blocks
        .iter()
        .filter(|block| block.name.starts_with("vc_header"))
        .map(|block| block.id)
        .collect();
    let mut fresh = max_value_id(func);
    for header_id in header_ids {
        // One index per relay, dropped before the rewrite below borrows the
        // function mutably: the analysis only reads, the rewrite only writes.
        let defs: Defs<'_> = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| inst.result.map(|id| (id, inst)))
            .collect();
        let Some(header) = func.blocks.iter().find(|block| block.id == header_id) else {
            continue;
        };
        let header_name = header.name.clone();
        let Terminator::CondBr {
            cond: Operand::Value(cond_id),
            true_target,
            ..
        } = &header.terminator
        else {
            return Err(format!("vec relay loop `{header_name}` has no counted header"));
        };
        let (cond_id, body_id) = (*cond_id, *true_target);
        let cmp = header
            .insts
            .iter()
            .find(|inst| inst.result == Some(cond_id))
            .ok_or_else(|| format!("vec relay loop `{header_name}` has no compare"))?;
        let InstOp::Cmp {
            lhs: Operand::Value(iv),
            rhs: Operand::Const(Constant::Int(limit)),
            ..
        } = &cmp.op
        else {
            return Err(format!("vec relay loop `{header_name}` has no constant limit"));
        };
        let (iv, limit) = (*iv, *limit);
        let Some(body) = func.blocks.iter().find(|block| block.id == body_id) else {
            return Err("a vec relay loop has no body".into());
        };
        let Some(relay) = match_vec_relay(body) else {
            return Err("a vec relay body is not the load/store pair the lowering emits".into());
        };
        let (off, len, src, dst) = relay;
        // The offset must be the loop index times the vector length: that is
        // the only shape whose widening back to elements is the identity.
        let off_def = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find(|inst| inst.result == Some(off))
            .ok_or_else(|| "a vec relay offset has no definition".to_string())?;
        let InstOp::Bin {
            op: BinOp::Mul,
            lhs: Operand::Value(multiplicand),
            rhs: Operand::Const(Constant::Int(width)),
        } = &off_def.op
        else {
            return Err("a vec relay offset is not index times a constant".into());
        };
        if *multiplicand != iv || *width != len as i64 {
            return Err("a vec relay offset does not read the loop index".into());
        }
        let src_id = operand_value(src, "a vec relay reads through a literal")?;
        let dst_id = operand_value(dst, "a vec relay writes through a literal")?;
        let (src_space, src_elem) = tile_element_of(func, &defs, src_id)
            .ok_or_else(|| "a vec relay source is not tile storage".to_string())?;
        let (dst_space, dst_elem) = tile_element_of(func, &defs, dst_id)
            .ok_or_else(|| "a vec relay target is not tile storage".to_string())?;
        if src_elem != dst_elem {
            return Err("a vec relay moves between different element types".into());
        }
        drop(defs);
        // Widen the compare: the scalar run has `len` times the iterations.
        let header = func
            .blocks
            .iter_mut()
            .find(|block| block.id == header_id)
            .expect("checked above");
        for inst in &mut header.insts {
            if inst.result == Some(cond_id)
                && let InstOp::Cmp {
                    rhs: rhs @ Operand::Const(Constant::Int(_)),
                    ..
                } = &mut inst.op
            {
                *rhs = Operand::Const(Constant::Int(limit * len as i64));
            }
        }
        // Rewrite the body: one element moves per iteration, addressed by the
        // loop index itself. The widened run covers the same slots the vector
        // relay did, and one element per iteration keeps every thread's slot
        // its own (`index % W == tid`) — a `index + lane` window would have a
        // thread writing the three slots past its own, off the tile at the
        // loop's tail.
        let body = func
            .blocks
            .iter_mut()
            .find(|block| block.id == body_id)
            .expect("checked above");
        body.insts.retain(|inst| {
            !matches!(&inst.op, InstOp::Intrinsic {
                intrinsic: Intrinsic::VecLoad { .. } | Intrinsic::VecStore { .. },
                ..
            })
        });
        let off_index = body
            .insts
            .iter()
            .position(|inst| inst.result == Some(off))
            .ok_or_else(|| "a vec relay body lost its offset".to_string())?;
        body.insts.remove(off_index);
        let mut scalar = Vec::new();
        {
            let index = Operand::Value(iv);
            let read_ptr = ValueId(fresh);
            fresh += 1;
            scalar.push(Inst {
                result: Some(read_ptr),
                result_name: Some(format!("sile_from{fresh}")),
                ty: Type::ptr(src_space.clone(), src_elem.clone()),
                op: InstOp::Gep {
                    base: Operand::Value(src_id),
                    indices: vec![index.clone()],
                },
                metadata: Vec::new(),
            });
            let read = ValueId(fresh);
            fresh += 1;
            scalar.push(Inst {
                result: Some(read),
                result_name: Some(format!("sile_elt{fresh}")),
                ty: src_elem.clone(),
                op: InstOp::Load {
                    ptr: Operand::Value(read_ptr),
                },
                metadata: Vec::new(),
            });
            let write_ptr = ValueId(fresh);
            fresh += 1;
            scalar.push(Inst {
                result: Some(write_ptr),
                result_name: Some(format!("sile_to{fresh}")),
                ty: Type::ptr(dst_space.clone(), dst_elem.clone()),
                op: InstOp::Gep {
                    base: Operand::Value(dst_id),
                    indices: vec![index],
                },
                metadata: Vec::new(),
            });
            scalar.push(Inst {
                result: None,
                result_name: None,
                ty: Type::Void,
                op: InstOp::Store {
                    ptr: Operand::Value(write_ptr),
                    value: Operand::Value(read),
                },
                metadata: Vec::new(),
            });
        }
        body.insts.splice(off_index..off_index, scalar);
    }
    Ok(())
}

fn operand_value(operand: Operand, message: &str) -> Result<ValueId, String> {
    match operand {
        Operand::Value(id) => Ok(id),
        Operand::Const(_) => Err(message.to_string()),
    }
}

/// The vector relay a `vc` body must hold: the offset definition, the vector
/// length, and both tile operands.
type VecRelay = (ValueId, usize, Operand, Operand);

fn match_vec_relay(body: &BasicBlock) -> Option<VecRelay> {
    let mut load = None;
    let mut store = None;
    for inst in &body.insts {
        if let InstOp::Intrinsic { intrinsic, args } = &inst.op {
            match intrinsic {
                Intrinsic::VecLoad { len } => {
                    load = Some((
                        inst.result?,
                        *len,
                        args.first()?.clone(),
                        args.get(1)?.clone(),
                    ));
                }
                Intrinsic::VecStore { len } => {
                    store = Some((
                        *len,
                        args.first()?.clone(),
                        args.get(1)?.clone(),
                        args.get(2)?.clone(),
                    ));
                }
                _ => {}
            }
        }
    }
    let (load_id, load_len, src, load_off) = load?;
    let (store_len, dst, store_off, stored) = store?;
    if load_len != store_len || load_off != store_off {
        return None;
    }
    if stored != Operand::Value(load_id) {
        return None;
    }
    let off = operand_value(load_off, "").ok()?;
    Some((off, load_len, src, dst))
}

/// A value as `a * iv + b` over the loop induction, or `None` when the chain
/// steps outside addition and multiplication by constants.
fn affine_of(defs: &Defs<'_>, value: &Operand, iv: ValueId) -> Option<(i64, i64)> {
    match value {
        Operand::Const(Constant::Int(c)) => Some((0, *c)),
        Operand::Value(id) if *id == iv => Some((1, 0)),
        Operand::Value(id) => {
            let inst = defs.get(id)?;
            match &inst.op {
                InstOp::Bin { op: BinOp::Add, lhs, rhs } => {
                    let (a, b) = affine_of(defs, lhs, iv)?;
                    let (c, d) = affine_of(defs, rhs, iv)?;
                    Some((a + c, b + d))
                }
                InstOp::Bin { op: BinOp::Mul, lhs, rhs } => {
                    let (a, b) = affine_of(defs, lhs, iv)?;
                    let (c, d) = affine_of(defs, rhs, iv)?;
                    Some((a * d + b * c, b * d))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether the access reads a `TileReduce`'s relay rather than tile storage the
/// strip partitions. The lowering stamps the relay when it emits the reduce, so
/// this follows the pointer chain back to a word the builder wrote — it does
/// not guess at the shape of the chain.
fn is_reduce_relay(defs: &Defs<'_>, mut id: ValueId) -> bool {
    for _ in 0..4 {
        let Some(inst) = defs.get(&id) else {
            return false;
        };
        if inst.metadata.contains(&Metadata::ReduceRelay) {
            return true;
        }
        match &inst.op {
            InstOp::Gep { base, .. } => match base {
                Operand::Value(prev) => id = *prev,
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}

/// Every tile slot a stripped loop hands out must belong to the thread that
/// strips it: slot `iv + b` with `b` a multiple of the thread count. A slot
/// past that shape would have one thread write what another reads through its
/// own private copy, and the value would never arrive. Fail closed: such a
/// kernel falls back to the serial mapping one level up.
fn check_strip_ownership(func: &Function, step: i64) -> Result<(), String> {
    // The checks resolve operands to definitions; the flat index keeps each
    // resolution a lookup. The causal fmha's IR runs thousands of instructions
    // and a scan per access made the check crawl.
    let defs: Defs<'_> = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect();
    for header in func
        .blocks
        .iter()
        .filter(|block| is_plumbing_header(&block.name))
    {
        let Some(param) = header.params.first() else {
            continue;
        };
        let iv = param.id;
        let Terminator::CondBr { true_target, .. } = &header.terminator else {
            continue;
        };
        let Some(body) = func
            .blocks
            .iter()
            .find(|block| block.id == *true_target)
        else {
            continue;
        };
        for inst in &body.insts {
            // The owner rule guards the private tiles, on reads as much as on
            // writes: a thread may only touch slots it wrote itself, and a
            // slot another thread wrote is garbage until a barrier hands it
            // over — which the private space has no way to do. A shared slot's
            // handover is the barrier's business, and a global buffer belongs
            // to no thread, so both stay outside this check.
            let (ptr, access) = match &inst.op {
                InstOp::Store { ptr, .. } => (ptr, "stores"),
                InstOp::Load { ptr } => (ptr, "loads"),
                _ => continue,
            };
            let Operand::Value(base) = ptr else {
                continue;
            };
            let Some((space, _)) = tile_element_of(func, &defs, *base) else {
                continue;
            };
            if space != AddressSpace::Private {
                continue;
            }
            // A reduce's relay is a broadcast, not a partitioned tile: the
            // reduce stage hands every thread the same copied values, so a
            // constant-slot read of it neither splits nor races. The lowering
            // stamps the relay when it emits the reduce; the strip looks past
            // it instead of partitioning slots every thread must see.
            if is_reduce_relay(&defs, *base) {
                continue;
            }
            let slot = defs
                .get(base)
                .and_then(|inst| match &inst.op {
                    InstOp::Gep { indices, .. } => indices.last().cloned(),
                    _ => None,
                })
                .ok_or_else(|| {
                    format!("plumbing loop `{}` touches a tile without a slot", header.name)
                })?;
            let (a, b) = affine_of(&defs, &slot, iv)
                .ok_or_else(|| {
                    format!(
                        "plumbing loop `{}` {access} a tile at a slot that is not `index * a + b`",
                        header.name
                    )
                })?;
            if a != 1 {
                return Err(format!(
                    "plumbing loop `{}` {access} a tile at `{a} * index + {b}`: the strip would \
                     split one owner's slots across threads",
                    header.name
                ));
            }
            if b % step != 0 {
                return Err(format!(
                    "plumbing loop `{}` {access} a tile at `index + {b}`, which is not a multiple \
                     of the {step}-thread strip: a thread would own slots past the tile's end",
                    header.name
                ));
            }
        }
    }
    Ok(())
}

/// One plumbing loop, stripped: the entry edge hands the thread index instead
/// of zero, and the latch steps by the thread count instead of one.
fn strip_loop(func: &mut Function, header_id: BlockId, step: i64) -> Result<(), String> {
    let Some(header) = func.blocks.iter().find(|block| block.id == header_id) else {
        return Ok(());
    };
    let header_name = header.name.clone();
    let Terminator::CondBr {
        cond: Operand::Value(cond_id),
        ..
    } = &header.terminator
    else {
        return Err(format!("plumbing loop `{header_name}` has no counted header"));
    };
    let cond_id = *cond_id;
    let cmp = header
        .insts
        .iter()
        .find(|inst| inst.result == Some(cond_id))
        .ok_or_else(|| format!("plumbing loop `{header_name}` has no compare"))?;
    let InstOp::Cmp {
        rhs: Operand::Const(Constant::Int(_)),
        ..
    } = &cmp.op
    else {
        return Err(format!("plumbing loop `{header_name}` has no constant limit"));
    };
    // The two edges into the header: the entry edge carries the init, the
    // latch carries the step. A plumbing loop has exactly these two. The back
    // edge is the one whose source the header dominates — a nested loop's
    // entry also *reaches* the header from inside an outer body, so plain
    // reachability would misread it as a latch.
    let dominators = crate::passes::cfg::dominators(func);
    let mut init_edges = Vec::new();
    let mut latch_edges = Vec::new();
    for block in &func.blocks {
        let mut visit = |target: BlockId, args: &[Operand]| {
            if target != header_id {
                return;
            }
            if dominators[&block.id].contains(&header_id) {
                latch_edges.push((block.id, args.to_vec()));
            } else {
                init_edges.push((block.id, args.to_vec()));
            }
        };
        match &block.terminator {
            Terminator::Br { target, args } => visit(*target, args),
            Terminator::CondBr {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                visit(*true_target, true_args);
                visit(*false_target, false_args);
            }
            _ => {}
        }
    }
    if init_edges.len() != 1 || latch_edges.len() != 1 {
        return Err(format!(
            "plumbing loop `{header_name}` has {} init edges and {} latch edges, expected one of each",
            init_edges.len(),
            latch_edges.len()
        ));
    }
    let (init_source, init_args) = &init_edges[0];
    let (latch_source, latch_args) = &latch_edges[0];
    if init_args.first() != Some(&Operand::Const(Constant::Int(0))) {
        return Err(format!(
            "plumbing loop `{header_name}` does not start its index at zero"
        ));
    }
    let Some(Operand::Value(step_value)) = latch_args.first() else {
        return Err(format!(
            "plumbing loop `{header_name}` has no step value on its backedge"
        ));
    };
    let step_value = *step_value;
    let tid = thread_id_of(func)
        .ok_or_else(|| "the policy stage did not insert a thread id".to_string())?;
    for block in &mut func.blocks {
        if block.id == *init_source {
            patch_edge_arg(&mut block.terminator, header_id, tid);
        }
        if block.id == *latch_source {
            for inst in &mut block.insts {
                if inst.result == Some(step_value)
                    && let InstOp::Bin {
                        rhs: rhs @ Operand::Const(Constant::Int(1)),
                        ..
                    } = &mut inst.op
                {
                    *rhs = Operand::Const(Constant::Int(step));
                }
            }
        }
    }
    Ok(())
}

fn patch_edge_arg(terminator: &mut Terminator, header: BlockId, value: ValueId) {
    let patch = |args: &mut Vec<Operand>| {
        if !args.is_empty() {
            args[0] = Operand::Value(value);
        }
    };
    match terminator {
        Terminator::Br { target, args } if *target == header => patch(args),
        Terminator::CondBr {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == header {
                patch(true_args);
            }
            if *false_target == header {
                patch(false_args);
            }
        }
        _ => {}
    }
}

// ── stage 2: reduce ──────────────────────────────────────────────────────

/// One `TileReduce` site, found by its result id.
struct ReduceSite {
    result: ValueId,
    reduce: TileReduceOp,
    extent: u64,
    inner: u64,
    outer: u64,
    source: ValueId,
}

fn find_reduce_sites(func: &Function) -> Vec<ReduceSite> {
    let mut sites = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Intrinsic {
                intrinsic:
                    Intrinsic::TileReduce {
                        reduce,
                        extent,
                        inner,
                        outer,
                    },
                args,
            } = &inst.op
                && let Some(result) = inst.result
                && let Some(&Operand::Value(source)) = args.first()
            {
                sites.push(ReduceSite {
                    result,
                    reduce: *reduce,
                    extent: *extent,
                    inner: *inner,
                    outer: *outer,
                    source,
                });
            }
        }
    }
    sites
}

/// Replace every `TileReduce` with the shared-scratch tree reduction.
fn materialize_reduces(func: Function, policy: &MappingPolicy) -> Result<Function, String> {
    let sites = find_reduce_sites(&func);
    if sites.is_empty() {
        return Ok(func);
    }
    let threads = policy.launch.count();
    // The tree reads the tile's columns across the strip: thread `tid` reads
    // slot `base + tid`, so the column count has to be a whole number of
    // strips or the last ones read past the tile. Fail closed; the caller
    // falls back to the serial mapping.
    if let Some(site) = sites.iter().find(|site| site.extent % threads as u64 != 0) {
        return Err(format!(
            "a tile reduce over {} columns does not divide across {} threads",
            site.extent, threads
        ));
    }
    let scratch_len = sites
        .iter()
        .map(|site| site.outer as usize)
        .max()
        .expect("checked non-empty")
        * threads as usize;
    let mut ctx = Ctx::new(func);
    let scratch = ctx.alloca(
        "sile_reduce_scratch",
        Type::array(scratch_len, Type::F32),
        AddressSpace::Shared,
    );
    let scratch_id = scratch.result.expect("an alloca has a result");
    ctx.append_to_entry(scratch);
    for (order, site) in sites.into_iter().enumerate() {
        build_tree_reduce(&mut ctx, scratch_id, site, threads, order)?;
    }
    Ok(ctx.finish())
}

/// The edit context: fresh ids from a running counter, and the small
/// constructors the materializations are written with.
struct Ctx {
    func: Function,
    next_value: u32,
    next_block: u32,
    names: u32,
}

/// An alloca-form counted loop under construction, in the canonical shape the
/// emitter spells as `for (iv = init; iv < limit; iv += step)`.
struct CountedLoop {
    header: BlockId,
    /// The index alloca, which the entry init and the latch step store.
    iv: ValueId,
    /// The index load inside the body.
    index: ValueId,
    body: BlockId,
    exit: BlockId,
}

impl Ctx {
    fn new(func: Function) -> Self {
        let next_value = max_value_id(&func);
        let next_block = max_block_id(&func);
        Self {
            func,
            next_value,
            next_block,
            names: 0,
        }
    }

    fn value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    /// A name no other value in the function carries.
    fn name(&mut self, stem: &str) -> String {
        self.names += 1;
        format!("{stem}{:#x}", self.names)
    }

    fn finish(self) -> Function {
        self.func
    }

    fn block_ref(&self, id: BlockId) -> &BasicBlock {
        self.func
            .blocks
            .iter()
            .find(|block| block.id == id)
            .expect("a block the context holds")
    }

    fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        self.func
            .blocks
            .iter_mut()
            .find(|block| block.id == id)
            .expect("a block the context holds")
    }

    fn named_inst(&mut self, stem: &str, ty: Type, op: InstOp) -> Inst {
        let id = self.value();
        Inst {
            result: Some(id),
            result_name: Some(self.name(stem)),
            ty,
            op,
            metadata: Vec::new(),
        }
    }

    fn void_inst(&mut self, op: InstOp) -> Inst {
        Inst {
            result: None,
            result_name: None,
            ty: Type::Void,
            op,
            metadata: Vec::new(),
        }
    }

    fn alloca(&mut self, stem: &str, alloc_ty: Type, addr_space: AddressSpace) -> Inst {
        let id = self.value();
        Inst {
            result: Some(id),
            result_name: Some(self.name(stem)),
            ty: Type::ptr(addr_space.clone(), alloc_ty.clone()),
            op: InstOp::Alloca {
                alloc_ty,
                addr_space,
            },
            metadata: Vec::new(),
        }
    }

    fn append_to_entry(&mut self, inst: Inst) {
        let entry = self.func.entry;
        self.block_mut(entry).insts.insert(0, inst);
    }

    fn block(&mut self, stem: &str) -> BlockId {
        let id = self.block_id();
        let name = self.name(stem);
        self.func.blocks.push(BasicBlock {
            id,
            name,
            params: Vec::new(),
            insts: Vec::new(),
            terminator: Terminator::Ret { value: None },
        });
        id
    }

    fn block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    fn set_terminator(&mut self, id: BlockId, terminator: Terminator) {
        self.block_mut(id).terminator = terminator;
    }

    fn push(&mut self, id: BlockId, inst: Inst) {
        self.block_mut(id).insts.push(inst);
    }

    /// Split `at` before instruction `index`: the tail moves to a fresh
    /// block, `at` keeps the head and branches to the tail.
    fn split_before(&mut self, at: BlockId, index: usize) -> BlockId {
        let stem = self.block_ref(at).name.clone();
        let tail = self.block(&format!("{stem}_cont"));
        let head = self.block_mut(at);
        let moved: Vec<Inst> = head.insts.split_off(index);
        let terminator = std::mem::replace(
            &mut head.terminator,
            Terminator::Br {
                target: tail,
                args: Vec::new(),
            },
        );
        let tail_block = self.block_mut(tail);
        tail_block.insts = moved;
        tail_block.terminator = terminator;
        tail
    }

    fn barrier(&mut self, at: BlockId) {
        let inst = self.void_inst(InstOp::Intrinsic {
            intrinsic: Intrinsic::Barrier {
                scope: SyncScope::Workgroup,
            },
            args: Vec::new(),
        });
        self.push(at, inst);
    }

    fn load(&mut self, at: BlockId, stem: &str, ptr: Operand, ty: Type) -> ValueId {
        let inst = self.named_inst(stem, ty, InstOp::Load { ptr });
        let id = inst.result.expect("a load has a result");
        self.push(at, inst);
        id
    }

    fn store(&mut self, at: BlockId, ptr: Operand, value: Operand) {
        let inst = self.void_inst(InstOp::Store { ptr, value });
        self.push(at, inst);
    }

    /// One element address off a tile's base pointer.
    fn gep(
        &mut self,
        at: BlockId,
        stem: &str,
        base: ValueId,
        index: Operand,
        elem: Type,
    ) -> ValueId {
        let space = match value_type(&self.func, base) {
            Some(Type::Ptr { addr_space, .. }) => addr_space.clone(),
            other => panic!("a gep base names storage, not {other:?}"),
        };
        let inst = self.named_inst(
            stem,
            Type::ptr(space, elem),
            InstOp::Gep {
                base: Operand::Value(base),
                indices: vec![index],
            },
        );
        let id = inst.result.expect("a gep has a result");
        self.push(at, inst);
        id
    }

    /// Stamp a just-pushed instruction with metadata.
    fn stamp(&mut self, id: ValueId, meta: Metadata) {
        for block in &mut self.func.blocks {
            for inst in &mut block.insts {
                if inst.result == Some(id) {
                    inst.metadata.push(meta);
                    return;
                }
            }
        }
    }

    fn int_bin(
        &mut self,
        at: BlockId,
        stem: &str,
        op: BinOp,
        lhs: Operand,
        rhs: Operand,
    ) -> ValueId {
        let inst = self.named_inst(stem, Type::I64, InstOp::Bin { op, lhs, rhs });
        let id = inst.result.expect("a bin has a result");
        self.push(at, inst);
        id
    }

    fn float_bin(
        &mut self,
        at: BlockId,
        stem: &str,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let inst = self.named_inst(stem, Type::F32, InstOp::Bin {
            op,
            lhs: Operand::Value(lhs),
            rhs: Operand::Value(rhs),
        });
        let id = inst.result.expect("a bin has a result");
        self.push(at, inst);
        id
    }

    fn icmp(
        &mut self,
        at: BlockId,
        stem: &str,
        pred: CmpPred,
        lhs: Operand,
        rhs: Operand,
    ) -> ValueId {
        let inst = self.named_inst(stem, Type::I1, InstOp::Cmp { pred, lhs, rhs });
        let id = inst.result.expect("a compare has a result");
        self.push(at, inst);
        id
    }

    /// An alloca-form counted loop that starts where `preheader`'s text ends.
    ///
    /// The preheader gains the index alloca, its init store and the branch;
    /// the caller fills the body and ends it with [`Ctx::close_loop`].
    fn open_loop(
        &mut self,
        stem: &str,
        preheader: BlockId,
        init: Operand,
        limit: i64,
        step: i64,
    ) -> CountedLoop {
        let iv = self.alloca(&format!("{stem}_iv"), Type::I64, AddressSpace::Private);
        let iv_id = iv.result.expect("an alloca has a result");
        self.push(preheader, iv);
        // The index alloca keeps the lowering's shape — direct load and store
        // on the pointer — because the emitters' counted-loop recognition
        // aggregates traffic on the alloca's own id.
        self.store(preheader, Operand::Value(iv_id), init);
        let header = self.block(&format!("{stem}_header"));
        let body = self.block(&format!("{stem}_body"));
        let exit = self.block(&format!("{stem}_exit"));
        self.set_terminator(
            preheader,
            Terminator::Br {
                target: header,
                args: Vec::new(),
            },
        );
        let load = self.named_inst(
            &format!("{stem}_at"),
            Type::I64,
            InstOp::Load {
                ptr: Operand::Value(iv_id),
            },
        );
        let load_id = load.result.expect("a load has a result");
        let cmp = self.named_inst(
            &format!("{stem}_in_limits"),
            Type::I1,
            InstOp::Cmp {
                pred: if step > 0 {
                    CmpPred::Slt
                } else {
                    CmpPred::Sgt
                },
                lhs: Operand::Value(load_id),
                rhs: Operand::Const(Constant::Int(limit)),
            },
        );
        let cmp_id = cmp.result.expect("a compare has a result");
        self.push(header, load);
        self.push(header, cmp);
        self.set_terminator(
            header,
            Terminator::CondBr {
                cond: Operand::Value(cmp_id),
                true_target: body,
                true_args: Vec::new(),
                false_target: exit,
                false_args: Vec::new(),
            },
        );
        CountedLoop {
            header,
            iv: iv_id,
            index: load_id,
            body,
            exit,
        }
    }

    /// End the loop body's text: the index step and the back edge.
    /// End the loop body's text: the index step and the back edge.
    ///
    /// `tail` is the block a full trip of the body ends on — the body itself,
    /// or the exit of the innermost loop nested in it. The step stores land
    /// there, and the back edge replaces a placeholder terminator only: a
    /// body that already ends in a nested loop keeps that loop's entry, and
    /// the nested loop's exit is what carries the edge back.
    fn close_loop(&mut self, loop_: &CountedLoop, step: i64, tail: BlockId) {
        let stepped = self.named_inst(
            "next",
            Type::I64,
            InstOp::Bin {
                op: if step > 0 { BinOp::Add } else { BinOp::Sub },
                lhs: Operand::Value(loop_.index),
                rhs: Operand::Const(Constant::Int(step.abs())),
            },
        );
        let stepped_id = stepped.result.expect("a bin has a result");
        self.push(tail, stepped);
        self.store(tail, Operand::Value(loop_.iv), Operand::Value(stepped_id));
        if matches!(self.block_ref(tail).terminator, Terminator::Ret { value: None }) {
            self.set_terminator(
                tail,
                Terminator::Br {
                    target: loop_.header,
                    args: Vec::new(),
                },
            );
        }
    }

    /// The scalar fold of `reduce` over two operands, as instructions on `at`.
    fn fold(
        &mut self,
        at: BlockId,
        stem: &str,
        reduce: TileReduceOp,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        match reduce {
            TileReduceOp::Sum => self.float_bin(at, stem, BinOp::Add, a, b),
            TileReduceOp::Product => self.float_bin(at, stem, BinOp::Mul, a, b),
            TileReduceOp::Max | TileReduceOp::Min => {
                let pred = if reduce == TileReduceOp::Max {
                    CmpPred::Ogt
                } else {
                    CmpPred::Olt
                };
                let cond = self.icmp(
                    at,
                    &format!("{stem}_picks"),
                    pred,
                    Operand::Value(a),
                    Operand::Value(b),
                );
                let inst = self.named_inst(stem, Type::F32, InstOp::Select {
                    cond: Operand::Value(cond),
                    on_true: Operand::Value(a),
                    on_false: Operand::Value(b),
                });
                let id = inst.result.expect("a select has a result");
                self.push(at, inst);
                id
            }
        }
    }
}

/// The tree reduction one site becomes.
///
/// The site's call is replaced by the result alloca it promised, and the
/// blocks between compute it: per-thread partials per output row, a barrier,
/// `log2(W)` straight-line tree steps, the result copy, and a barrier that
/// lets the scratch serve the next fold.
fn build_tree_reduce(
    ctx: &mut Ctx,
    scratch_id: ValueId,
    site: ReduceSite,
    threads: u32,
    order: usize,
) -> Result<(), String> {
    let span = (site.extent * site.inner) as i64;
    let outer = site.outer as i64;
    let w = threads as i64;
    let ReduceSite {
        result,
        reduce,
        source,
        ..
    } = site;
    // The fold reads tile storage, and this policy only spells f32 folds.
    let defs: Defs<'_> = ctx
        .func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect();
    let elem = tile_element_of(&ctx.func, &defs, source)
        .map(|(_, elem)| elem)
        .ok_or_else(|| "a reduce source is not tile storage".to_string())?;
    if elem != Type::F32 {
        return Err("the workgroup-tree reduce policy only spells f32 folds".into());
    }
    // Swap the call for the result alloca it promised, and split the block
    // so the tree lands between the site and its consumers.
    let at = find_block_with_inst(&ctx.func, result)
        .ok_or_else(|| "a reduce site vanished before materialization".to_string())?;
    let index = ctx.block_ref(at)
        .insts
        .iter()
        .position(|inst| inst.result == Some(result))
        .expect("the block holds the call");
    let dst = Inst {
        result: Some(result),
        result_name: Some(ctx.name("red")),
        ty: Type::ptr(AddressSpace::Private, Type::array(outer as usize, Type::F32)),
        op: InstOp::Alloca {
            alloc_ty: Type::array(outer as usize, Type::F32),
            addr_space: AddressSpace::Private,
        },
        metadata: Vec::new(),
    };
    ctx.block_mut(at).insts[index] = dst;
    let cont = ctx.split_before(at, index + 1);

    let tid = thread_id_of(&ctx.func).expect("the policy stage ran");
    let tag = format!("mr{order}");

    // Per-thread partials, one strip per output row.
    let rows = ctx.open_loop(&tag, at, Operand::Const(Constant::Int(0)), outer, 1);
    let acc = ctx.alloca(
        &format!("{tag}_acc"),
        Type::array(1, Type::F32),
        AddressSpace::Private,
    );
    let acc_id = acc.result.expect("an alloca has a result");
    ctx.push(rows.body, acc);
    let acc_slot = ctx.gep(
        rows.body,
        &format!("{tag}_acc_slot"),
        acc_id,
        Operand::Const(Constant::Int(0)),
        Type::F32,
    );
    ctx.store(
        rows.body,
        Operand::Value(acc_slot),
        Operand::Const(Constant::Float(identity(reduce))),
    );
    let cols = ctx.open_loop(
        &format!("{tag}j"),
        rows.body,
        Operand::Value(tid),
        span,
        w,
    );
    // partial = combine(partial, source[r * span + j])
    let row = ctx.load(rows.body, &format!("{tag}_r"), Operand::Value(rows.iv), Type::I64);
    let base = ctx.int_bin(
        cols.body,
        &format!("{tag}_base"),
        BinOp::Mul,
        Operand::Value(row),
        Operand::Const(Constant::Int(span)),
    );
    let addr = ctx.int_bin(
        cols.body,
        &format!("{tag}_addr"),
        BinOp::Add,
        Operand::Value(base),
        Operand::Value(cols.index),
    );
    let src_ptr = ctx.gep(
        cols.body,
        &format!("{tag}_src"),
        source,
        Operand::Value(addr),
        Type::F32,
    );
    // The fold reads `source[row * span + tid + k * threads]`: the entry check
    // `extent % threads` makes `row * span` a whole strip count, so every slot
    // is the reader's own. The placement check reads the stamp instead of
    // re-deriving that from the row loop's index.
    ctx.stamp(src_ptr, Metadata::OwnerOrdered);
    let v = ctx.load(cols.body, &format!("{tag}_v"), Operand::Value(src_ptr), Type::F32);
    let p_slot = ctx.gep(
        cols.body,
        &format!("{tag}_p_slot"),
        acc_id,
        Operand::Const(Constant::Int(0)),
        Type::F32,
    );
    let p = ctx.load(cols.body, &format!("{tag}_p"), Operand::Value(p_slot), Type::F32);
    let combined = ctx.fold(cols.body, &format!("{tag}_fold"), reduce, p, v);
    ctx.store(cols.body, Operand::Value(p_slot), Operand::Value(combined));
    ctx.close_loop(&cols, w, cols.body);
    // scratch[r * W + tid] = partial; the continuation of the rows body
    // lives on the cols loop's exit, which carries the rows back edge.
    let row_after = ctx.load(cols.exit, &format!("{tag}_row"), Operand::Value(rows.iv), Type::I64);
    let lane = ctx.int_bin(
        cols.exit,
        &format!("{tag}_lane"),
        BinOp::Mul,
        Operand::Value(row_after),
        Operand::Const(Constant::Int(w)),
    );
    let slot = ctx.int_bin(
        cols.exit,
        &format!("{tag}_slot"),
        BinOp::Add,
        Operand::Value(lane),
        Operand::Value(tid),
    );
    let out = ctx.gep(
        cols.exit,
        &format!("{tag}_out"),
        scratch_id,
        Operand::Value(slot),
        Type::F32,
    );
    let partial = ctx.load(cols.exit, &format!("{tag}_partial"), Operand::Value(acc_slot), Type::F32);
    ctx.store(cols.exit, Operand::Value(out), Operand::Value(partial));
    ctx.close_loop(&rows, 1, cols.exit);

    // The tree: `log2(W)` straight-line steps, each a guarded combine over
    // the rows and one barrier.
    let mut guard = rows.exit;
    ctx.barrier(guard);
    let mut stride = w / 2;
    while stride > 0 {
        let picks = ctx.icmp(
            guard,
            &format!("{tag}_step{stride}"),
            CmpPred::Slt,
            Operand::Value(tid),
            Operand::Const(Constant::Int(stride)),
        );
        let step_body = ctx.block(&format!("{tag}_step{stride}_body"));
        let step_join = ctx.block(&format!("{tag}_step{stride}_join"));
        ctx.set_terminator(
            guard,
            Terminator::CondBr {
                cond: Operand::Value(picks),
                true_target: step_body,
                true_args: Vec::new(),
                false_target: step_join,
                false_args: Vec::new(),
            },
        );
        let walks = ctx.open_loop(
            &format!("{tag}t{stride}"),
            step_body,
            Operand::Const(Constant::Int(0)),
            outer,
            1,
        );
        let walk_row = ctx.load(walks.body, &format!("{tag}_tr"), Operand::Value(walks.iv), Type::I64);
        let walk_base = ctx.int_bin(
            walks.body,
            &format!("{tag}_tb"),
            BinOp::Mul,
            Operand::Value(walk_row),
            Operand::Const(Constant::Int(w)),
        );
        let low_index = ctx.int_bin(
            walks.body,
            &format!("{tag}_tl"),
            BinOp::Add,
            Operand::Value(walk_base),
            Operand::Value(tid),
        );
        let high_index = ctx.int_bin(
            walks.body,
            &format!("{tag}_th"),
            BinOp::Add,
            Operand::Value(low_index),
            Operand::Const(Constant::Int(stride)),
        );
        let low = ctx.gep(
            walks.body,
            &format!("{tag}_low"),
            scratch_id,
            Operand::Value(low_index),
            Type::F32,
        );
        let high = ctx.gep(
            walks.body,
            &format!("{tag}_high"),
            scratch_id,
            Operand::Value(high_index),
            Type::F32,
        );
        let a = ctx.load(walks.body, &format!("{tag}_ta"), Operand::Value(low), Type::F32);
        let b = ctx.load(walks.body, &format!("{tag}_tbv"), Operand::Value(high), Type::F32);
        let merged = ctx.fold(walks.body, &format!("{tag}_tm"), reduce, a, b);
        ctx.store(walks.body, Operand::Value(low), Operand::Value(merged));
        ctx.close_loop(&walks, 1, walks.body);
        ctx.set_terminator(
            walks.exit,
            Terminator::Br {
                target: step_join,
                args: Vec::new(),
            },
        );
        ctx.barrier(step_join);
        guard = step_join;
        stride /= 2;
    }

    // Every thread copies the results out, then one more barrier lets the
    // scratch serve the next fold.
    let copy = ctx.open_loop(
        &format!("{tag}c"),
        guard,
        Operand::Const(Constant::Int(0)),
        outer,
        1,
    );
    let copy_base = ctx.int_bin(
        copy.body,
        &format!("{tag}_cb"),
        BinOp::Mul,
        Operand::Value(copy.index),
        Operand::Const(Constant::Int(w)),
    );
    let from = ctx.gep(
        copy.body,
        &format!("{tag}_cf"),
        scratch_id,
        Operand::Value(copy_base),
        Type::F32,
    );
    let value = ctx.load(copy.body, &format!("{tag}_cv"), Operand::Value(from), Type::F32);
    let to = ctx.gep(
        copy.body,
        &format!("{tag}_ct"),
        result,
        Operand::Value(copy.index),
        Type::F32,
    );
    ctx.store(copy.body, Operand::Value(to), Operand::Value(value));
    ctx.close_loop(&copy, 1, copy.body);
    ctx.barrier(copy.exit);
    ctx.set_terminator(
        copy.exit,
        Terminator::Br {
            target: cont,
            args: Vec::new(),
        },
    );
    Ok(())
}

fn identity(reduce: TileReduceOp) -> f64 {
    match reduce {
        TileReduceOp::Sum => 0.0,
        TileReduceOp::Product => 1.0,
        TileReduceOp::Max => -f64::from(f32::MAX),
        TileReduceOp::Min => f64::from(f32::MAX),
    }
}

// ── stage 3: mma ─────────────────────────────────────────────────────────

/// One `TileMma` site, found by its result id.
struct MmaSite {
    result: ValueId,
    m: i64,
    n: i64,
    k: i64,
    rhs_cols: i64,
    lhs: ValueId,
    rhs: ValueId,
    acc: Option<ValueId>,
}

fn find_mma_sites(func: &Function) -> Vec<MmaSite> {
    let mut sites = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Intrinsic {
                intrinsic:
                    Intrinsic::TileMma {
                        m,
                        n,
                        k,
                        rhs_cols,
                    },
                args,
            } = &inst.op
                && let Some(result) = inst.result
                && let (Some(&Operand::Value(lhs)), Some(&Operand::Value(rhs))) =
                    (args.first(), args.get(1))
            {
                sites.push(MmaSite {
                    result,
                    m: *m as i64,
                    n: *n as i64,
                    k: *k as i64,
                    rhs_cols: *rhs_cols as i64,
                    lhs,
                    rhs,
                    acc: args.get(2).and_then(|operand| match operand {
                        Operand::Value(id) => Some(*id),
                        _ => None,
                    }),
                });
            }
        }
    }
    sites
}

/// Replace every `TileMma` with its per-thread accumulator strip.
///
/// The operand panels move to shared storage: a strip reads arbitrary
/// elements of them, which private storage cannot serve. One barrier orders
/// the panel writes against the strip's reads, one orders the reads against
/// the next iteration's writes.
fn materialize_mmas(func: Function, policy: &MappingPolicy) -> Result<Function, String> {
    let sites = find_mma_sites(&func);
    if sites.is_empty() {
        return Ok(func);
    }
    let threads = policy.launch.count() as i64;
    let mut ctx = Ctx::new(func);
    for (order, site) in sites.into_iter().enumerate() {
        build_mma_strip(&mut ctx, site, threads, order)?;
    }
    Ok(ctx.finish())
}

/// Flip one tile's storage to shared.
fn flip_to_shared(ctx: &mut Ctx, tile: ValueId) -> Result<(), String> {
    let at = find_block_with_inst(&ctx.func, tile)
        .ok_or_else(|| "an mma operand vanished".to_string())?;
    let index = ctx.block_ref(at)
        .insts
        .iter()
        .position(|inst| inst.result == Some(tile))
        .expect("the block holds the alloca");
    let block = ctx.block_mut(at);
    let inst = &mut block.insts[index];
    let InstOp::Alloca {
        alloc_ty,
        addr_space,
    } = &mut inst.op
    else {
        return Err("an mma operand is not tile storage this policy can share".into());
    };
    if !matches!(alloc_ty, Type::Array { elem, .. } if **elem == Type::F32) {
        return Err("the thread-strip mma policy only spells f32 panels".into());
    }
    let alloc_ty = alloc_ty.clone();
    *addr_space = AddressSpace::Shared;
    inst.ty = Type::ptr(AddressSpace::Shared, alloc_ty);
    Ok(())
}

fn build_mma_strip(ctx: &mut Ctx, site: MmaSite, threads: i64, order: usize) -> Result<(), String> {
    let MmaSite {
        result,
        m,
        n,
        k,
        rhs_cols,
        lhs,
        rhs,
        acc,
    } = site;
    let defs: Defs<'_> = ctx
        .func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect();
    let carry_elem = acc.map(|carry| {
        tile_element_of(&ctx.func, &defs, carry)
            .map(|(_, elem)| elem)
            .ok_or_else(|| "an mma accumulator is not tile storage".to_string())
    }).transpose()?;
    if let Some(elem) = &carry_elem {
        if *elem != Type::F32 {
            return Err("the thread-strip mma policy only spells f32 accumulators".into());
        }
    }
    let lhs_elem = tile_element_of(&ctx.func, &defs, lhs)
        .map(|(_, elem)| elem)
        .ok_or_else(|| "an mma operand is not tile storage".to_string())?;
    let rhs_elem = tile_element_of(&ctx.func, &defs, rhs)
        .map(|(_, elem)| elem)
        .ok_or_else(|| "an mma operand is not tile storage".to_string())?;
    drop(defs);
    flip_to_shared(ctx, lhs)?;
    // The rhs panel is gathered across threads just the same: lane `e` reads
    // `rhs[k * rhs_cols + e % n]`, slots other lanes loaded. Private rhs would
    // make every lane multiply mostly zeros.
    flip_to_shared(ctx, rhs)?;
    if lhs_elem != Type::F32 || rhs_elem != Type::F32 {
        return Err("the thread-strip mma policy only spells f32 panels".into());
    }

    let at = find_block_with_inst(&ctx.func, result)
        .ok_or_else(|| "an mma site vanished before materialization".to_string())?;
    let index = ctx.block_ref(at)
        .insts
        .iter()
        .position(|inst| inst.result == Some(result))
        .expect("the block holds the call");
    let dst = Inst {
        result: Some(result),
        result_name: Some(ctx.name("tile")),
        ty: Type::ptr(
            AddressSpace::Private,
            Type::array((m * n) as usize, Type::F32),
        ),
        op: InstOp::Alloca {
            alloc_ty: Type::array((m * n) as usize, Type::F32),
            addr_space: AddressSpace::Private,
        },
        metadata: Vec::new(),
    };
    ctx.block_mut(at).insts[index] = dst;
    let cont = ctx.split_before(at, index + 1);

    let tid = thread_id_of(&ctx.func).expect("the policy stage ran");
    let tag = format!("mm{order}");

    // The panels are shared now: a barrier separates the writes every thread
    // made from the reads every thread is about to make.
    ctx.barrier(at);
    let lane = ctx.open_loop(&tag, at, Operand::Value(tid), m * n, threads);
    // row = e / n; col = e % n
    let row = ctx.int_bin(
        lane.body,
        &format!("{tag}_row"),
        BinOp::Div,
        Operand::Value(lane.index),
        Operand::Const(Constant::Int(n)),
    );
    let col = ctx.int_bin(
        lane.body,
        &format!("{tag}_col"),
        BinOp::Rem,
        Operand::Value(lane.index),
        Operand::Const(Constant::Int(n)),
    );
    // The accumulator: the carry's element where one exists, else zero.
    let acc_alloca = ctx.alloca(
        &format!("{tag}_acc"),
        Type::array(1, Type::F32),
        AddressSpace::Private,
    );
    let acc_id = acc_alloca.result.expect("an alloca has a result");
    ctx.push(lane.body, acc_alloca);
    let acc_slot = ctx.gep(
        lane.body,
        &format!("{tag}_acc_slot"),
        acc_id,
        Operand::Const(Constant::Int(0)),
        Type::F32,
    );
    let seed = match acc {
        Some(carry) => {
            let from = ctx.gep(
                lane.body,
                &format!("{tag}_seed"),
                carry,
                Operand::Value(lane.index),
                Type::F32,
            );
            ctx.load(lane.body, &format!("{tag}_a0"), Operand::Value(from), Type::F32)
        }
        None => {
            let zero = ctx.named_inst(&format!("{tag}_a0"), Type::F32, InstOp::Select {
                cond: Operand::Const(Constant::Bool(false)),
                on_true: Operand::Const(Constant::Float(0.0)),
                on_false: Operand::Const(Constant::Float(0.0)),
            });
            let zero_id = zero.result.expect("a select has a result");
            ctx.push(lane.body, zero);
            zero_id
        }
    };
    ctx.store(lane.body, Operand::Value(acc_slot), Operand::Value(seed));
    let steps = ctx.open_loop(
        &format!("{tag}k"),
        lane.body,
        Operand::Const(Constant::Int(0)),
        k,
        1,
    );
    let a = ctx.load(steps.body, &format!("{tag}_a"), Operand::Value(acc_slot), Type::F32);
    let x_base = ctx.int_bin(
        steps.body,
        &format!("{tag}_xb"),
        BinOp::Mul,
        Operand::Value(row),
        Operand::Const(Constant::Int(k)),
    );
    let x_addr = ctx.int_bin(
        steps.body,
        &format!("{tag}_xa"),
        BinOp::Add,
        Operand::Value(x_base),
        Operand::Value(steps.index),
    );
    let x_ptr = ctx.gep(
        steps.body,
        &format!("{tag}_x"),
        lhs,
        Operand::Value(x_addr),
        Type::F32,
    );
    let x = ctx.load(steps.body, &format!("{tag}_xv"), Operand::Value(x_ptr), Type::F32);
    let y_base = ctx.int_bin(
        steps.body,
        &format!("{tag}_yb"),
        BinOp::Mul,
        Operand::Value(steps.index),
        Operand::Const(Constant::Int(rhs_cols)),
    );
    let y_addr = ctx.int_bin(
        steps.body,
        &format!("{tag}_ya"),
        BinOp::Add,
        Operand::Value(y_base),
        Operand::Value(col),
    );
    let y_ptr = ctx.gep(
        steps.body,
        &format!("{tag}_y"),
        rhs,
        Operand::Value(y_addr),
        Type::F32,
    );
    let y = ctx.load(steps.body, &format!("{tag}_yv"), Operand::Value(y_ptr), Type::F32);
    let product = ctx.float_bin(steps.body, &format!("{tag}_mul"), BinOp::Mul, x, y);
    let sum = ctx.float_bin(steps.body, &format!("{tag}_add"), BinOp::Add, a, product);
    ctx.store(steps.body, Operand::Value(acc_slot), Operand::Value(sum));
    ctx.close_loop(&steps, 1, steps.body);
    // dst[e] = accumulator; the lane body's continuation lives on the k
    // loop's exit, which carries the lane back edge.
    let total = ctx.load(steps.exit, &format!("{tag}_tot"), Operand::Value(acc_slot), Type::F32);
    let to = ctx.gep(
        steps.exit,
        &format!("{tag}_out"),
        result,
        Operand::Value(lane.index),
        Type::F32,
    );
    ctx.store(steps.exit, Operand::Value(to), Operand::Value(total));
    ctx.close_loop(&lane, threads, steps.exit);
    // The reads are done: the next iteration's panel writes may proceed.
    ctx.barrier(lane.exit);
    ctx.set_terminator(
        lane.exit,
        Terminator::Br {
            target: cont,
            args: Vec::new(),
        },
    );
    Ok(())
}
