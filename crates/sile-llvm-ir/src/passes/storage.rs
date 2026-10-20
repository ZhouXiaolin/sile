//! Storage cleanup: tile relays that carry no information.
//!
//! The lowering gives every tile value storage of its own, so a value that
//! reads another tile's storage is born as a full-extent copy loop, and a
//! value nobody reads is born as a store loop into the void. Both cost real
//! loop iterations on every target. Two facts clean them up without touching
//! the loop shapes the backends spell:
//!
//! - A private allocation nothing loads carries no information: its stores,
//!   their address chains and the allocation itself go away.
//! - A private allocation whose only writes are one full-extent copy from
//!   another allocation equals that source from the copy on. Its loads
//!   re-address the source and the copy dies as a write-only storage.
//!
//! The relays here are full-extent by construction: the lowering copies a
//! tile's whole extent, and a kernel never reads a slot a relay did not
//! write. A copy whose source is written again after the copy is left alone —
//! the destination keeps the older values, and forwarding would lose them.

use std::collections::{HashMap, HashSet};

use super::cfg;
use super::sroa::array_dims;
use crate::{AddressSpace, Function, Inst, InstOp, Operand, Terminator, Type, ValueId};

pub fn run(func: &mut Function) {
    loop {
        let mut changed = forward_copy_loads(func);
        changed |= drop_write_only(func);
        if !changed {
            break;
        }
    }
}

/// One private allocation and everything derived from its address.
struct Storage {
    /// The array extents the allocation reserves, outermost first. Empty for
    /// a scalar such as an induction variable.
    dims: Vec<usize>,
    /// The allocation and every `gep` rooted at it.
    derived: HashSet<ValueId>,
    /// The blocks that load from this storage.
    load_blocks: Vec<crate::BlockId>,
    /// The stores into this storage.
    stores: Vec<Store>,
    /// The blocks whose instructions hand a derived address to something this
    /// pass cannot see through: a call argument, an intrinsic operand, an
    /// atomic or a memcpy.
    escape_blocks: Vec<crate::BlockId>,
}

struct Store {
    block: crate::BlockId,
    ptr: ValueId,
    value: Operand,
}

/// Map every private allocation to its address chain and its uses.
///
/// A tile intrinsic's result is storage too: `sile.tile.reduce.*` and
/// `sile.tile.mma.*` return a pointer to private storage they write, so a
/// relay can read from a call result just as from an `alloca`. The call site
/// counts as that storage's one write.
fn private_storages(func: &Function) -> HashMap<ValueId, Storage> {
    let mut geps: HashMap<ValueId, &Operand> = HashMap::new();
    let mut gep_users: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    let mut roots: Vec<ValueId> = Vec::new();
    let mut call_writes: Vec<(ValueId, crate::BlockId)> = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else { continue };
            match &inst.op {
                InstOp::Alloca {
                    alloc_ty,
                    addr_space,
                } if *addr_space == AddressSpace::Private => roots.push(result),
                InstOp::Call { .. } | InstOp::Intrinsic { .. }
                    if matches!(
                        inst.ty,
                        Type::Ptr {
                            addr_space: AddressSpace::Private,
                            ..
                        }
                    ) =>
                {
                    roots.push(result);
                    call_writes.push((result, block.id));
                }
                InstOp::Gep { base, .. } => {
                    if let Operand::Value(base_id) = base {
                        geps.insert(result, base);
                        gep_users.entry(*base_id).or_default().push(result);
                    }
                }
                _ => {}
            }
        }
    }

    // An `alloca` knows its extent; a call result's extent is read off the
    // geps that address it, so it starts unknown.
    let mut alloc_dims: HashMap<ValueId, Vec<usize>> = HashMap::new();
    let mut storages: HashMap<ValueId, Storage> = HashMap::new();
    for root in &roots {
        storages.insert(
            *root,
            Storage {
                dims: Vec::new(),
                derived: HashSet::from([*root]),
                load_blocks: Vec::new(),
                stores: Vec::new(),
                escape_blocks: Vec::new(),
            },
        );
    }
    for block in &func.blocks {
        for inst in &block.insts {
            if let InstOp::Alloca {
                alloc_ty,
                addr_space,
            } = &inst.op
            {
                if *addr_space == AddressSpace::Private {
                    if let Some(result) = inst.result {
                        alloc_dims.insert(result, array_dims(alloc_ty));
                    }
                }
            }
        }
    }
    for (root, dims) in &alloc_dims {
        storages.get_mut(root).unwrap().dims = dims.clone();
    }

    // A `gep` of a `gep` joins the same storage, wherever the block order
    // puts it: walk the address chains from each root.
    for root in roots.iter().copied() {
        let mut worklist = vec![root];
        while let Some(current) = worklist.pop() {
            for derived in gep_users.get(&current).into_iter().flatten() {
                if storages.get_mut(&root).unwrap().derived.insert(*derived) {
                    worklist.push(*derived);
                }
            }
        }
    }
    let root_of: HashMap<ValueId, ValueId> = storages
        .iter()
        .flat_map(|(root, storage)| storage.derived.iter().map(move |id| (*id, *root)))
        .collect();

    // A tile intrinsic writes its result storage at the call itself.
    for (result, block) in &call_writes {
        storages
            .get_mut(result)
            .unwrap()
            .escape_blocks
            .push(*block);
    }

    let mut pointer_root = |operand: &Operand| match operand {
        Operand::Value(id) => root_of.get(id).copied(),
        Operand::Const(_) => None,
    };

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                InstOp::Load { ptr } => {
                    if let Some(root) = pointer_root(ptr) {
                        storages.get_mut(&root).unwrap().load_blocks.push(block.id);
                    }
                }
                InstOp::Store { ptr, value } => {
                    if let Some(root) = pointer_root(ptr) {
                        storages.get_mut(&root).unwrap().stores.push(Store {
                            block: block.id,
                            ptr: match ptr {
                                Operand::Value(id) => *id,
                                Operand::Const(_) => unreachable!(),
                            },
                            value: value.clone(),
                        });
                    }
                    mark_escape(value, &mut pointer_root, &mut storages, block.id);
                }
                InstOp::AtomicAdd { ptr, value } => {
                    for operand in [ptr, value] {
                        if let Some(root) = pointer_root(operand) {
                            let storage = storages.get_mut(&root).unwrap();
                            storage.escape_blocks.push(block.id);
                        }
                    }
                }
                InstOp::Memcpy { dst, src, .. } => {
                    for operand in [dst, src] {
                        if let Some(root) = pointer_root(operand) {
                            let storage = storages.get_mut(&root).unwrap();
                            storage.escape_blocks.push(block.id);
                        }
                    }
                }
                InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => {
                    for arg in args {
                        mark_escape(arg, &mut pointer_root, &mut storages, block.id);
                    }
                }
                _ => {}
            }
        }
    }
    storages
}

fn mark_escape(
    operand: &Operand,
    pointer_root: &mut impl FnMut(&Operand) -> Option<ValueId>,
    storages: &mut HashMap<ValueId, Storage>,
    block: crate::BlockId,
) {
    if let Some(root) = pointer_root(operand) {
        storages
            .get_mut(&root)
            .unwrap()
            .escape_blocks
            .push(block);
    }
}

/// Delete every private allocation nothing loads, with its stores and its
/// address chain. True when something went away.
fn drop_write_only(func: &mut Function) -> bool {
    let storages = private_storages(func);
    let mut dead: HashSet<ValueId> = HashSet::new();
    for (root, storage) in &storages {
        if storage.escape_blocks.is_empty()
            && storage.load_blocks.is_empty()
            && !storage.stores.is_empty()
        {
            dead.insert(*root);
            dead.extend(&storage.derived);
        }
    }
    if dead.is_empty() {
        return false;
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| match &inst.op {
            InstOp::Store { ptr, .. } => match ptr {
                Operand::Value(id) => !dead.contains(id),
                Operand::Const(_) => true,
            },
            _ => inst.result.map_or(true, |result| !dead.contains(&result)),
        });
    }
    true
}

/// The re-addressing one relayed storage takes: every load reads the source
/// base with the type the source's own gep spells.
struct Plan {
    src_base: Operand,
    src_ty: Type,
}

/// Re-address the loads of a once-copied storage onto its source.
///
/// The pattern is the relay the lowering spells for a tile value that reads
/// another tile: one counted loop whose body is
/// `store(gep(dst, i), load(gep(src, i)))`, the same `i` on both sides, and
/// nothing else ever writes `dst`. Then `dst` equals `src` from the loop's
/// exit on, and every load of `dst` past the exit reads `src` instead. True
/// when some storage was re-addressed.
fn forward_copy_loads(func: &mut Function) -> bool {
    let storages = private_storages(func);
    let root_of: HashMap<ValueId, ValueId> = storages
        .iter()
        .flat_map(|(root, storage)| storage.derived.iter().map(move |id| (*id, *root)))
        .collect();
    let inst_by_result: HashMap<ValueId, &Inst> = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| inst.result.map(|result| (result, inst)))
        .collect();
    let mut uses: HashMap<ValueId, usize> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            for operand in inst_operands(&inst.op) {
                if let Operand::Value(id) = operand {
                    *uses.entry(*id).or_default() += 1;
                }
            }
        }
        for operand in terminator_operands(&block.terminator) {
            if let Operand::Value(id) = operand {
                *uses.entry(*id).or_default() += 1;
            }
        }
    }

    // One plan per storage: the source base operand to re-address with and
    // the source gep's type.
    let mut plans: HashMap<ValueId, Plan> = HashMap::new();
    for (root, storage) in &storages {
        let plan = copy_plan(
            *root,
            storage,
            &storages,
            &root_of,
            &inst_by_result,
            &uses,
            func,
        );
        if let Some(plan) = plan {
            plans.insert(*root, plan);
        }
    }
    if plans.is_empty() {
        return false;
    }

    // One rewrite per relayed load site, resolved while the function is
    // still shared: the source base to re-address with, the source gep's
    // type, and the index list the load already uses.
    let mut rewrites: HashMap<ValueId, (Operand, Type, Vec<Operand>)> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let InstOp::Load { ptr } = &inst.op else {
                continue;
            };
            let Operand::Value(gep_id) = ptr else {
                continue;
            };
            let Some(root) = root_of.get(gep_id) else {
                continue;
            };
            let Some(plan) = plans.get(root) else {
                continue;
            };
            let InstOp::Gep { indices, .. } = &inst_by_result[gep_id].op else {
                continue;
            };
            rewrites.insert(
                *gep_id,
                (plan.src_base.clone(), plan.src_ty.clone(), indices.clone()),
            );
        }
    }
    if rewrites.is_empty() {
        return false;
    }

    let mut next_id = next_value_id(func);
    for block in &mut func.blocks {
        let mut spelled = Vec::with_capacity(block.insts.len());
        for mut inst in std::mem::take(&mut block.insts) {
            if let InstOp::Load { ptr: Operand::Value(gep_id) } = &inst.op {
                if let Some((src_base, src_ty, indices)) = rewrites.get(gep_id) {
                    // The load keeps its index; the address becomes the
                    // source's.
                    let gep = Inst {
                        result: Some(ValueId(next_id)),
                        result_name: None,
                        ty: src_ty.clone(),
                        op: InstOp::Gep {
                            base: src_base.clone(),
                            indices: indices.clone(),
                        },
                        metadata: Vec::new(),
                    };
                    next_id += 1;
                    spelled.push(gep);
                    inst.op = InstOp::Load {
                        ptr: Operand::Value(ValueId(next_id - 1)),
                    };
                }
            }
            spelled.push(inst);
        }
        block.insts = spelled;
    }
    true
}

/// Decide whether one storage is a pure relay of another, and where its copy
/// loop ends. `None` leaves the storage alone.
fn copy_plan(
    root: ValueId,
    storage: &Storage,
    storages: &HashMap<ValueId, Storage>,
    root_of: &HashMap<ValueId, ValueId>,
    inst_by_result: &HashMap<ValueId, &Inst>,
    uses: &HashMap<ValueId, usize>,
    func: &Function,
) -> Option<Plan> {
    if storage.dims.is_empty()
        || storage.load_blocks.is_empty()
        || storage.stores.is_empty()
        || !storage.escape_blocks.is_empty()
    {
                return None;
    }
    // The relay is one loop, so every store lives in one block.
    let store_block = storage.stores[0].block;
    if storage
        .stores
        .iter()
        .any(|store| store.block != store_block)
    {
                return None;
    }

    let mut source: Option<(Operand, Type, ValueId)> = None;
    for store in &storage.stores {
        // store(gep(dst, i), v) where v = load(gep(src, i)), the same `i`.
        let Operand::Value(value_id) = store.value else {
                        return None;
        };
        let InstOp::Load { ptr } = &inst_by_result.get(&value_id)?.op else {
                        return None;
        };
        let Operand::Value(src_gep_id) = ptr else {
                        return None;
        };
        let InstOp::Gep {
            base: src_base_operand,
            indices: src_indices,
        } = &inst_by_result.get(src_gep_id)?.op
        else {
                        return None;
        };
        let Some(store_src_root) = root_of.get(src_gep_id) else {
                        return None;
        };
        if *store_src_root == root {
                        return None;
        }
        let InstOp::Gep {
            indices: dst_indices,
            ..
        } = &inst_by_result.get(&store.ptr)?.op
        else {
                        return None;
        };
        if dst_indices != src_indices {
                        return None;
        }
        let this = (
            src_base_operand.clone(),
            inst_by_result.get(src_gep_id)?.ty.clone(),
            *store_src_root,
        );
        // The relayed value feeds this store alone; other readers would break
        // once the relay dies.
        if uses.get(&value_id) != Some(&1) {
                        return None;
        }
        // Both sides address one flat element run, so a relayed index means
        // the same slot on either storage. The read's own element type is
        // what the re-addressed gep keeps.
        match &source {
            None => source = Some(this),
            Some(found) if *found == this => {}
            Some(_) => return None,
        }
    }
    let (src_base, src_ty, src_root) = source?;
    let source = &storages[&src_root];
    
    // The copy loop is the innermost one around the store; past its exit the
    // relay holds, before it or inside it the loads would read mid-copy.
    let loops = cfg::natural_loops(func);
    let copy_loop = loops
        .iter()
        .filter(|loop_| loop_.blocks.contains(&store_block))
        .min_by_key(|loop_| loop_.blocks.len())?;
    let header = func
        .blocks
        .iter()
        .find(|block| block.id == copy_loop.header)?;
    let exit = match &header.terminator {
        Terminator::CondBr {
            true_target,
            false_target,
            ..
        } => {
            let outside = |target: &crate::BlockId| !copy_loop.blocks.contains(target);
            match (outside(true_target), outside(false_target)) {
                (true, false) => *true_target,
                (false, true) => *false_target,
                _ => return None,
            }
        }
        _ => return None,
    };

    let dominators = cfg::dominators(func);
    let dominates_header = |block: crate::BlockId| dominators[&copy_loop.header].contains(&block);
    // Every read of the relay happens after the copy.
    for load_block in &storage.load_blocks {
        if !dominators[load_block].contains(&exit) {
                        return None;
        }
    }
    // Every write to the source happens before the copy, so re-addressed
    // reads see the values the copy saw.
    for site in source
        .stores
        .iter()
        .map(|store| store.block)
        .chain(source.escape_blocks.iter().copied())
    {
        if !dominates_header(site) {
                        return None;
        }
    }
    
    Some(Plan { src_base, src_ty })
}

fn inst_operands(op: &InstOp) -> Vec<&Operand> {
    match op {
        InstOp::Alloca { .. } => Vec::new(),
        InstOp::Gep { base, indices } => {
            let mut operands = vec![base];
            operands.extend(indices.iter());
            operands
        }
        InstOp::Load { ptr } => vec![ptr],
        InstOp::Store { ptr, value } => vec![ptr, value],
        InstOp::AtomicAdd { ptr, value } => vec![ptr, value],
        InstOp::Memcpy { dst, src, .. } => vec![dst, src],
        InstOp::Bin { lhs, rhs, .. } | InstOp::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
        InstOp::Cast { value, .. } => vec![value],
        InstOp::Select {
            cond,
            on_true,
            on_false,
        } => vec![cond, on_true, on_false],
        InstOp::Call { args, .. } | InstOp::Intrinsic { args, .. } => args.iter().collect(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::Br { .. } | Terminator::Ret { .. } => Vec::new(),
        Terminator::CondBr { cond, .. } => vec![cond],
        Terminator::Switch { value, .. } => vec![value],
    }
}

fn next_value_id(func: &Function) -> u32 {
    let mut max_id = 0u32;
    for param in &func.params {
        max_id = max_id.max(param.id.0);
    }
    for block in &func.blocks {
        for param in &block.params {
            max_id = max_id.max(param.id.0);
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                max_id = max_id.max(result.0);
            }
        }
    }
    max_id + 1
}
