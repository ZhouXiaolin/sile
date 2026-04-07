use std::collections::HashSet;

use crate::{Constant, Function, Metadata, Operand, Terminator};

/// Conservative loop-oriented canonicalization.
///
/// Current behavior:
/// - Fold degenerate conditional branches (`condbr` with identical targets or
///   compile-time boolean condition) into plain branches.
/// - Remove duplicate loop metadata (`Parallel`, `Reduction`, `Unroll`, etc.)
///   so later passes/codegen can reason on a normalized annotation set.
pub fn run(mut func: Function) -> Function {
    for block in &mut func.blocks {
        block.terminator = simplify_terminator(block.terminator.clone());
        dedup_metadata(&mut block.insts);
    }
    dedup_function_metadata(&mut func.metadata);
    func
}

fn simplify_terminator(terminator: Terminator) -> Terminator {
    match terminator {
        Terminator::CondBr {
            cond,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            if true_target == false_target {
                return Terminator::Br {
                    target: true_target,
                    args: true_args,
                };
            }

            if let Operand::Const(Constant::Bool(flag)) = cond {
                if flag {
                    return Terminator::Br {
                        target: true_target,
                        args: true_args,
                    };
                }
                return Terminator::Br {
                    target: false_target,
                    args: false_args,
                };
            }

            Terminator::CondBr {
                cond,
                true_target,
                true_args,
                false_target,
                false_args,
            }
        }
        other => other,
    }
}

fn dedup_metadata(insts: &mut [crate::Inst]) {
    for inst in insts {
        let mut seen = HashSet::<MetadataKey>::new();
        inst.metadata
            .retain(|meta| seen.insert(MetadataKey::from(meta)));
    }
}

fn dedup_function_metadata(metadata: &mut Vec<Metadata>) {
    let mut seen = HashSet::<MetadataKey>::new();
    metadata.retain(|meta| seen.insert(MetadataKey::from(meta)));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MetadataKey {
    Parallel,
    Reduction,
    VectorizeWidth(u32),
    Unroll(u32),
    Alignment(u32),
    NoAlias,
    ReadOnly,
    WriteOnly,
}

impl From<&Metadata> for MetadataKey {
    fn from(value: &Metadata) -> Self {
        match value {
            Metadata::Parallel => Self::Parallel,
            Metadata::Reduction => Self::Reduction,
            Metadata::VectorizeWidth(width) => Self::VectorizeWidth(*width),
            Metadata::Unroll(width) => Self::Unroll(*width),
            Metadata::Alignment(align) => Self::Alignment(*align),
            Metadata::NoAlias => Self::NoAlias,
            Metadata::ReadOnly => Self::ReadOnly,
            Metadata::WriteOnly => Self::WriteOnly,
        }
    }
}
