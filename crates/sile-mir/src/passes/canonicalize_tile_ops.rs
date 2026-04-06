use crate::{BinOp, CmpOp, MirFunction, MirOp};

pub fn run(mut func: MirFunction) -> MirFunction {
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.op {
                MirOp::TileBinary { op, lhs, rhs, .. } if is_commutative(*op) => {
                    if lhs.0 > rhs.0 {
                        std::mem::swap(lhs, rhs);
                    }
                }
                MirOp::IBinary { op, lhs, rhs } if is_commutative(*op) => {
                    if lhs.0 > rhs.0 {
                        std::mem::swap(lhs, rhs);
                    }
                }
                MirOp::ICmp { op, lhs, rhs } => match op {
                    CmpOp::Gt => {
                        *op = CmpOp::Lt;
                        std::mem::swap(lhs, rhs);
                    }
                    CmpOp::Ge => {
                        *op = CmpOp::Le;
                        std::mem::swap(lhs, rhs);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    func
}

fn is_commutative(op: BinOp) -> bool {
    matches!(op, BinOp::Add | BinOp::Mul)
}
