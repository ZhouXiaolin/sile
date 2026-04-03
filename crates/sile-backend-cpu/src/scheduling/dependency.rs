use sile_lir::ir::{Function, Instruction, Terminator, Value};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub base: String,
    pub indices: Vec<Value>,
    pub is_write: bool,
    pub block_label: String,
}

#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub header_block: String,
    pub body_blocks: Vec<String>,
    pub exit_block: String,
    pub induction_var: Option<Value>,
    pub bound: Option<Value>,
}

pub fn find_natural_loops(func: &Function) -> Vec<LoopInfo> {
    let mut loops = Vec::new();

    for block in &func.blocks {
        if let Some(target) = get_branch_target(&block.terminator) {
            if func.blocks.iter().any(|b| b.label == *target) {
                let body = collect_loop_body(func, &target, &block.label);
                if !body.is_empty() {
                    loops.push(LoopInfo {
                        header_block: target.clone(),
                        body_blocks: body,
                        exit_block: find_loop_exit(func, &block.label).unwrap_or_default(),
                        induction_var: extract_induction_var(func, &target),
                        bound: extract_loop_bound(func, &target),
                    });
                }
            }
        }
    }

    loops
}

pub fn analyze_memory_accesses(func: &Function, loop_info: &LoopInfo) -> Vec<MemoryAccess> {
    let mut accesses = Vec::new();

    for label in &loop_info.body_blocks {
        if let Some(block) = func.get_block(label) {
            for inst in &block.instructions {
                if let Some(access) = extract_memory_access(inst, label) {
                    accesses.push(access);
                }
            }
        }
    }

    accesses
}

pub fn has_loop_carried_dependency(accesses: &[MemoryAccess]) -> bool {
    let writes: Vec<_> = accesses.iter().filter(|a| a.is_write).collect();
    let reads: Vec<_> = accesses.iter().filter(|a| !a.is_write).collect();

    for write in &writes {
        for read in &reads {
            if write.base == read.base && indices_overlap(&write.indices, &read.indices) {
                return true;
            }
        }
    }

    false
}

pub fn is_reduction_pattern(accesses: &[MemoryAccess]) -> Option<ReductionType> {
    let writes: Vec<_> = accesses.iter().filter(|a| a.is_write).collect();
    if writes.len() <= 1 {
        return None;
    }

    let targets: HashSet<_> = writes.iter().map(|w| &w.base).collect();
    if targets.len() == 1 {
        return Some(ReductionType::Sum);
    }

    None
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Sum,
    Max,
    Min,
    Product,
}

fn get_branch_target(terminator: &Terminator) -> Option<&String> {
    match terminator {
        Terminator::Br { target } => Some(target),
        Terminator::CondBr { true_target, .. } => Some(true_target),
        _ => None,
    }
}

fn collect_loop_body(func: &Function, header: &str, back_edge_from: &str) -> Vec<String> {
    let mut body = Vec::new();
    let mut found_header = false;
    for block in &func.blocks {
        if block.label == header {
            found_header = true;
            continue;
        }
        if found_header && block.label != back_edge_from {
            body.push(block.label.clone());
        }
        if block.label == back_edge_from {
            body.push(block.label.clone());
            break;
        }
    }
    body
}

fn find_loop_exit(func: &Function, back_edge_from: &str) -> Option<String> {
    if let Some(block) = func.get_block(back_edge_from) {
        if let Terminator::CondBr { false_target, .. } = &block.terminator {
            return Some(false_target.clone());
        }
    }
    None
}

fn extract_induction_var(func: &Function, header: &str) -> Option<Value> {
    if let Some(block) = func.get_block(header) {
        for inst in &block.instructions {
            if let Instruction::Load { ptr, .. } = inst {
                return Some(ptr.clone());
            }
        }
    }
    None
}

fn extract_loop_bound(func: &Function, header: &str) -> Option<Value> {
    if let Some(block) = func.get_block(header) {
        for inst in &block.instructions {
            match inst {
                Instruction::Icmp(_, _, rhs) | Instruction::Fcmp(_, _, rhs) => {
                    return Some(rhs.clone());
                }
                _ => {}
            }
        }
    }
    None
}

fn extract_memory_access(inst: &Instruction, block_label: &str) -> Option<MemoryAccess> {
    match inst {
        Instruction::Load { ptr, .. } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: vec![],
            is_write: false,
            block_label: block_label.to_string(),
        }),
        Instruction::Store { ptr, .. } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: vec![],
            is_write: true,
            block_label: block_label.to_string(),
        }),
        Instruction::Gep { ptr, indices } => Some(MemoryAccess {
            base: format!("{:?}", ptr),
            indices: indices.clone(),
            is_write: false,
            block_label: block_label.to_string(),
        }),
        _ => None,
    }
}

fn indices_overlap(a: &[Value], b: &[Value]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}
