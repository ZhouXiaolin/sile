//! Integer interval arithmetic, as cutile's `bounds.rs` defines it.
//!
//! cutile folds an integer expression to a `constant` when it can prove the
//! value is a single number (`compile_binary_op.rs`), and it does that *after*
//! allocating the id for the operation it then throws away. Those discarded ids
//! are why a printed program has gaps in its numbering, so reproducing the
//! gaps means reproducing the reasoning that caused them.
//!
//! Only the operations the fixtures use are here. Each one takes the extremes
//! of both operands, which is sound for a monotone operation and is what
//! cutile does for `add`, `sub` and `mul`.

use sile_hir::BuiltinOp;

/// An inclusive interval `[start, end]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub start: i64,
    pub end: i64,
}

impl Bounds {
    pub fn exact(value: i64) -> Self {
        Self {
            start: value,
            end: value,
        }
    }

    /// Whether the interval names a single value — the condition cutile folds
    /// on.
    pub fn is_exact(&self) -> bool {
        self.start == self.end
    }

    /// The four corner samples of `f` over both intervals, as an interval.
    fn corners(self, other: Self, f: impl Fn(i64, i64) -> i64) -> Self {
        if self.is_exact() && other.is_exact() {
            return Self::exact(f(self.start, other.start));
        }
        let samples = [
            f(self.start, other.start),
            f(self.start, other.end),
            f(self.end, other.start),
            f(self.end, other.end),
        ];
        Self {
            start: *samples.iter().min().expect("four samples"),
            end: *samples.iter().max().expect("four samples"),
        }
    }
}

/// The interval of `op(lhs, rhs)`, or `None` where this IR does not reason
/// about the operation.
///
/// `Div` and `Rem` need a divisor that excludes zero; cutile rejects such a
/// divisor before it gets here, so a zero divisor makes the result unknown
/// rather than a panic.
pub fn transfer(op: BuiltinOp, lhs: Bounds, rhs: Bounds) -> Option<Bounds> {
    match op {
        BuiltinOp::Add => Some(lhs.corners(rhs, i64::saturating_add)),
        BuiltinOp::Sub => Some(lhs.corners(rhs, i64::saturating_sub)),
        BuiltinOp::Mul => Some(lhs.corners(rhs, i64::saturating_mul)),
        BuiltinOp::Div => {
            if rhs.start <= 0 && 0 <= rhs.end {
                return None;
            }
            Some(lhs.corners(rhs, i64::saturating_div))
        }
        BuiltinOp::Rem => {
            if rhs.start <= 0 && 0 <= rhs.end {
                return None;
            }
            if lhs.is_exact() && rhs.is_exact() {
                return Some(Bounds::exact(lhs.start.checked_rem(rhs.start).unwrap_or(0)));
            }
            // |a % b| <= |b| - 1, and the sign follows the dividend's.
            let limit = (rhs.start.unsigned_abs().max(rhs.end.unsigned_abs()) - 1) as i64;
            Some(Bounds {
                start: if lhs.start >= 0 { 0 } else { lhs.start.max(-limit) },
                end: if lhs.end <= 0 { 0 } else { lhs.end.min(limit) },
            })
        }
        BuiltinOp::Min => Some(lhs.corners(rhs, i64::min)),
        BuiltinOp::Max => Some(lhs.corners(rhs, i64::max)),
        // `ceil_div(a, b)` is `(a + b - 1) / b` for a positive divisor.
        BuiltinOp::CeilDiv => {
            if rhs.start <= 0 || !rhs.is_exact() {
                return None;
            }
            let divisor = rhs.start;
            Some(lhs.corners(rhs, |a, _| {
                let quotient = a.saturating_div(divisor);
                if a.checked_rem(divisor).unwrap_or(0) == 0 {
                    quotient
                } else {
                    quotient.saturating_add(1)
                }
            }))
        }
        _ => None,
    }
}
