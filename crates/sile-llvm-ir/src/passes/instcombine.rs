//! Constant folding and instruction combining.
//!
//! The lowering emits literal index arithmetic and constant comparisons that a
//! later stage can evaluate, and the other LLVM IR passes leave them in place.
//! This pass folds an instruction whose operands are all constants, and it
//! applies the algebraic identities that are exact for the target semantics.
//! A folded instruction becomes an operand, every use is rewritten, and the
//! instruction is removed. Folding runs to a fixed point, so a fold that
//! exposes the next one is taken in the same visit.

use std::collections::HashMap;

use crate::passes::canonicalize::{
    eliminate_dead_insts, rewrite_inst_operands, rewrite_operand, rewrite_terminator_operands,
};
use crate::{
    BinOp, CastOp, CmpPred, Constant, Function, InstOp, Operand, Type, ValueId,
};

pub fn run(mut func: Function) -> Function {
    loop {
        let types = value_types(&func);
        let mut replacements: HashMap<ValueId, Operand> = HashMap::new();

        for block in &func.blocks {
            for inst in &block.insts {
                let Some(result) = inst.result else { continue };
                if replacements.contains_key(&result) {
                    continue;
                }
                let folded = match &inst.op {
                    InstOp::Bin { op, lhs, rhs } => {
                        let lhs = rewrite_operand(lhs.clone(), &replacements);
                        let rhs = rewrite_operand(rhs.clone(), &replacements);
                        fold_bin(&inst.ty, *op, &lhs, &rhs)
                    }
                    InstOp::Cmp { pred, lhs, rhs } => {
                        let lhs = rewrite_operand(lhs.clone(), &replacements);
                        let rhs = rewrite_operand(rhs.clone(), &replacements);
                        fold_cmp(*pred, &lhs, &rhs)
                    }
                    InstOp::Cast { op, value, to } => {
                        let from = operand_type(value, &types);
                        from.and_then(|from| fold_cast(*op, &from, to, value))
                    }
                    InstOp::Select {
                        cond,
                        on_true,
                        on_false,
                    } => {
                        let cond = rewrite_operand(cond.clone(), &replacements);
                        match cond {
                            Operand::Const(Constant::Bool(value)) => Some(if value {
                                on_true.clone()
                            } else {
                                on_false.clone()
                            }),
                            Operand::Const(Constant::Int(value)) => Some(if value != 0 {
                                on_true.clone()
                            } else {
                                on_false.clone()
                            }),
                            Operand::Const(Constant::Float(_)) | Operand::Value(_) => None,
                        }
                    }
                    _ => None,
                };
                if let Some(replacement) = folded {
                    replacements.insert(result, replacement);
                }
            }
        }

        if replacements.is_empty() {
            break;
        }
        apply_replacements(&mut func, &replacements);
    }

    eliminate_dead_insts(&mut func);
    func
}

/// The type of every block parameter and instruction result.
fn value_types(func: &Function) -> HashMap<ValueId, Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                types.insert(result, inst.ty.clone());
            }
        }
    }
    types
}

/// The type of an operand, when it is a value the function defines.
fn operand_type(operand: &Operand, types: &HashMap<ValueId, Type>) -> Option<Type> {
    match operand {
        Operand::Value(id) => types.get(id).cloned(),
        Operand::Const(Constant::Int(_)) => Some(Type::I64),
        Operand::Const(Constant::Float(_)) => Some(Type::F64),
        Operand::Const(Constant::Bool(_)) => Some(Type::I1),
    }
}

fn apply_replacements(func: &mut Function, replacements: &HashMap<ValueId, Operand>) {
    for block in &mut func.blocks {
        block.insts = std::mem::take(&mut block.insts)
            .into_iter()
            .map(|inst| rewrite_inst_operands(inst, replacements))
            .filter(|inst| inst.result.map_or(true, |result| !replacements.contains_key(&result)))
            .collect();
        block.terminator =
            rewrite_terminator_operands(block.terminator.clone(), replacements);
    }
}

// ── Binary operations ──────────────────────────────────────────────

fn fold_bin(ty: &Type, op: BinOp, lhs: &Operand, rhs: &Operand) -> Option<Operand> {
    if let (Operand::Const(left), Operand::Const(right)) = (lhs, rhs) {
        return match (left, right) {
            (Constant::Int(a), Constant::Int(b)) => fold_int_bin(ty, op, *a, *b),
            (Constant::Float(a), Constant::Float(b)) => fold_float_bin(op, *a, *b),
            _ => None,
        };
    }

    // Algebraic identities. Only the exact ones are here: integer
    // wrap-around makes them exact for every value, and no fast-math flag
    // widens them. Each rule keeps the operand on the side that survives.
    match (op, lhs, rhs) {
        (BinOp::Mul, _, Operand::Const(Constant::Int(1))) => Some(lhs.clone()),
        (BinOp::Mul, Operand::Const(Constant::Int(1)), _) => Some(rhs.clone()),
        (BinOp::Mul, _, Operand::Const(Constant::Int(0))) => {
            Some(Operand::Const(Constant::Int(0)))
        }
        (BinOp::Mul, Operand::Const(Constant::Int(0)), _) => {
            Some(Operand::Const(Constant::Int(0)))
        }
        (BinOp::Add, _, Operand::Const(Constant::Int(0))) => Some(lhs.clone()),
        (BinOp::Add, Operand::Const(Constant::Int(0)), _) => Some(rhs.clone()),
        (BinOp::Sub, _, Operand::Const(Constant::Int(0))) => Some(lhs.clone()),
        (BinOp::Div, _, Operand::Const(Constant::Int(1))) => Some(lhs.clone()),
        (BinOp::Rem, _, Operand::Const(Constant::Int(1))) => {
            Some(Operand::Const(Constant::Int(0)))
        }
        (BinOp::And, _, Operand::Const(Constant::Int(0))) => {
            Some(Operand::Const(Constant::Int(0)))
        }
        (BinOp::And, Operand::Const(Constant::Int(0)), _) => {
            Some(Operand::Const(Constant::Int(0)))
        }
        (BinOp::Or, _, Operand::Const(Constant::Int(0))) => Some(lhs.clone()),
        (BinOp::Or, Operand::Const(Constant::Int(0)), _) => Some(rhs.clone()),
        (BinOp::Xor, _, Operand::Const(Constant::Int(0))) => Some(lhs.clone()),
        (BinOp::Xor, Operand::Const(Constant::Int(0)), _) => Some(rhs.clone()),
        (BinOp::Shl, _, Operand::Const(Constant::Int(0)))
        | (BinOp::Shr, _, Operand::Const(Constant::Int(0))) => Some(lhs.clone()),
        (BinOp::Sub, Operand::Value(_), Operand::Value(right))
            if matches!(lhs, Operand::Value(left) if left == right) =>
        {
            Some(Operand::Const(Constant::Int(0)))
        }
        _ => None,
    }
}

fn fold_int_bin(ty: &Type, op: BinOp, a: i64, b: i64) -> Option<Operand> {
    let value = match &ty {
        Type::I32 => {
            let (a, b) = (a as i32, b as i32);
            let value = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Div => a.checked_div(b)?,
                BinOp::Rem => a.checked_rem(b)?,
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::Shl | BinOp::Shr if !(0..32).contains(&b) => return None,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::Shr => a.wrapping_shr(b as u32),
            };
            value as i64
        }
        Type::I64 => match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a.checked_div(b)?,
            BinOp::Rem => a.checked_rem(b)?,
            BinOp::And => a & b,
            BinOp::Or => a | b,
            BinOp::Xor => a ^ b,
            BinOp::Shl | BinOp::Shr if !(0..64).contains(&b) => return None,
            BinOp::Shl => a.wrapping_shl(b as u32),
            BinOp::Shr => a.wrapping_shr(b as u32),
        },
        _ => match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a.checked_div(b)?,
            BinOp::Rem => a.checked_rem(b)?,
            BinOp::And => a & b,
            BinOp::Or => a | b,
            BinOp::Xor => a ^ b,
            BinOp::Shl => a.wrapping_shl(b as u32),
            BinOp::Shr => a.wrapping_shr(b as u32),
        },
    };
    Some(Operand::Const(Constant::Int(value)))
}

fn fold_float_bin(op: BinOp, a: f64, b: f64) -> Option<Operand> {
    // The IR computes f32. Fold in f32 and store the widened result, so a
    // literal round-trips through the same precision the kernel runs at.
    let (a, b) = (a as f32, b as f32);
    let value = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        // `frem` has no C spelling and the lowering does not emit it, so a
        // float remainder is left for the target.
        BinOp::Rem
        | BinOp::And
        | BinOp::Or
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Shr => return None,
    };
    Some(Operand::Const(Constant::Float(value as f64)))
}

// ── Comparisons ────────────────────────────────────────────────────

fn fold_cmp(pred: CmpPred, lhs: &Operand, rhs: &Operand) -> Option<Operand> {
    let value = match (lhs, rhs) {
        (Operand::Const(Constant::Int(a)), Operand::Const(Constant::Int(b))) => {
            fold_int_cmp(pred, *a, *b)?
        }
        (Operand::Const(Constant::Float(a)), Operand::Const(Constant::Float(b))) => {
            fold_float_cmp(pred, *a as f32, *b as f32)?
        }
        (Operand::Const(Constant::Bool(a)), Operand::Const(Constant::Bool(b))) => match pred {
            CmpPred::Eq => a == b,
            CmpPred::Ne => a != b,
            _ => return None,
        },
        _ => return None,
    };
    Some(Operand::Const(Constant::Bool(value)))
}

fn fold_int_cmp(pred: CmpPred, a: i64, b: i64) -> Option<bool> {
    Some(match pred {
        CmpPred::Eq => a == b,
        CmpPred::Ne => a != b,
        CmpPred::Slt | CmpPred::Olt => a < b,
        CmpPred::Sle | CmpPred::Ole => a <= b,
        CmpPred::Sgt | CmpPred::Ogt => a > b,
        CmpPred::Sge | CmpPred::Oge => a >= b,
    })
}

fn fold_float_cmp(pred: CmpPred, a: f32, b: f32) -> Option<bool> {
    // The ordered predicates are false when either operand is NaN; the equal
    // and not-equal ones are the only ones the lowering emits on floats that
    // are not ordered, and they follow IEEE-754.
    Some(match pred {
        CmpPred::Eq => a == b,
        CmpPred::Ne => a != b,
        CmpPred::Olt => a < b,
        CmpPred::Ole => a <= b,
        CmpPred::Ogt => a > b,
        CmpPred::Oge => a >= b,
        CmpPred::Slt | CmpPred::Sle | CmpPred::Sgt | CmpPred::Sge => return None,
    })
}

// ── Conversions ────────────────────────────────────────────────────

fn fold_cast(op: CastOp, from: &Type, to: &Type, value: &Operand) -> Option<Operand> {
    let constant = match value {
        Operand::Const(constant) => constant,
        Operand::Value(_) => return None,
    };
    match (op, constant) {
        (CastOp::Sext, Constant::Int(value)) => {
            Some(Operand::Const(Constant::Int(sign_extend_to(*value, from, to)?)))
        }
        (CastOp::Zext, Constant::Int(value)) => {
            Some(Operand::Const(Constant::Int(zero_extend_to(*value, from, to)?)))
        }
        (CastOp::Trunc, Constant::Int(value)) => {
            Some(Operand::Const(Constant::Int(truncate_to(*value, to)?)))
        }
        (CastOp::Sitofp, Constant::Int(value)) => Some(Operand::Const(Constant::Float(
            (sign_extend_to(*value, from, to)? as f64) as f32 as f64,
        ))),
        (CastOp::Fptosi, Constant::Float(value)) => {
            Some(Operand::Const(Constant::Int(truncate_to(
                *value as i64,
                to,
            )?)))
        }
        (CastOp::Bitcast, Constant::Int(value)) => match to {
            Type::F32 => Some(Operand::Const(Constant::Float(
                f32::from_bits(*value as u32) as f64,
            ))),
            Type::F64 => Some(Operand::Const(Constant::Float(f64::from_bits(
                *value as u64,
            )))),
            _ => None,
        },
        (CastOp::Bitcast, Constant::Float(value)) => match to {
            Type::I32 => Some(Operand::Const(Constant::Int(
                (*value as f32).to_bits() as i64,
            ))),
            Type::I64 => Some(Operand::Const(Constant::Int(
                value.to_bits() as i64,
            ))),
            _ => None,
        },
        // A half constant has no representation in `Constant`, so the two
        // half conversions stay for the target.
        (CastOp::Fext | CastOp::Ftrunc, _) => None,
        _ => None,
    }
}

fn sign_extend_to(value: i64, from: &Type, to: &Type) -> Option<i64> {
    let signed = match from {
        Type::I1 => value & 1,
        Type::I32 => (value as u32 as i32) as i64,
        Type::I64 => value,
        _ => return None,
    };
    Some(fit_width(signed, to))
}

fn zero_extend_to(value: i64, from: &Type, to: &Type) -> Option<i64> {
    let unsigned = match from {
        Type::I1 => (value & 1) as u64,
        Type::I32 => (value as u32) as u64,
        Type::I64 => value as u64,
        _ => return None,
    };
    Some(fit_width(unsigned as i64, to))
}

fn truncate_to(value: i64, to: &Type) -> Option<i64> {
    Some(match to {
        Type::I1 => value & 1,
        Type::I32 => (value as u32 as i32) as i64,
        Type::I64 => value,
        _ => return None,
    })
}

fn fit_width(value: i64, ty: &Type) -> i64 {
    match ty {
        Type::I1 => value & 1,
        Type::I32 => (value as u32 as i32) as i64,
        _ => value,
    }
}
