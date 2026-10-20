//! Token ordering: what a memory operation's completion is a value of.
//!
//! CuTile orders memory work by data, not by position. A load yields a token;
//! a store consumes one. The rule is two lines:
//!
//! 1. Every memory op takes an optional token and yields a token.
//! 2. The yielded token means: the operation that produced the consumed token
//!    has finished.
//!
//! So two independent loads produce two unrelated tokens and the compiler is
//! free to issue them together. A `join_tokens` of the two produces one token
//! that means both finished, and a store that consumes it is the only place
//! the two meet:
//!
//! ```text
//!   t1 ──┐
//!        ├──> joined ──> store
//!   t2 ──┘
//! ```
//!
//! A token orders; it does not make visible. Completion and visibility are two
//! answers to two questions, so a memory op states both: the token says what
//! it waited for, and a [`MemoryScope`] says who can see the result. How
//! strongly it orders against other operations is a third: the
//! [`MemoryOrdering`] of the atomics vocabulary.

use crate::function_ir::{Opcode, TileIrOp, TileIrType, ValueId};

/// The token a store waits for, or nothing when it waits for nothing.
///
/// The absent case is not "unordered": it means this operation's predecessors
/// are established elsewhere, so it adds no edge of its own.
pub type TokenOperand = Option<ValueId>;

/// One memory operation's ordering state, as it appears on an op.
///
/// These three fields are the whole of what a `_tko` op says about ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TkoSemantics {
    /// The token this op waits for.
    pub token: TokenOperand,
    /// How strongly this op orders against other memory operations.
    pub ordering: crate::function_ir::MemoryOrdering,
    /// How widely this op's effect is visible.
    pub scope: crate::function_ir::MemoryScope,
}

impl Default for TkoSemantics {
    fn default() -> Self {
        // A plain load or store in a tile block is the weak default: it orders
        // only through the token it exchanges, and it is visible within the
        // block. Widening either is a decision an op states.
        Self {
            token: None,
            ordering: crate::function_ir::MemoryOrdering::Weak,
            scope: crate::function_ir::MemoryScope::TileBlock,
        }
    }
}

impl TkoSemantics {
    /// The semantics of an op that only waits: no token in, weak, block-scoped.
    pub fn waiting(token: TokenOperand) -> Self {
        Self {
            token,
            ..Self::default()
        }
    }

    /// The same semantics made visible device-wide.
    pub fn with_device_scope(self) -> Self {
        Self {
            scope: crate::function_ir::MemoryScope::Device,
            ..self
        }
    }

    /// The same semantics at the given ordering.
    pub fn with_ordering(self, ordering: crate::function_ir::MemoryOrdering) -> Self {
        Self { ordering, ..self }
    }
}

/// `join_tokens %a, %b`: one token that means both operands finished.
///
/// This is the only way two ordering chains meet. Nothing about where the op
/// sits in a block moves a token: the joined token is a value, and an op waits
/// for it by naming it.
pub fn join_tokens(lhs: ValueId, rhs: ValueId) -> TileIrOp {
    TileIrOp::Opaque {
        opcode: Opcode::JoinTokens,
        operands: vec![lhs, rhs],
        results: vec![TileIrType::Token],
        attributes: Vec::new(),
    }
}

/// `make_token`: a token with nothing behind it.
///
/// A chain has to start somewhere. A `make_token` states "this point has no
/// predecessor", which is what a kernel entry needs before its first store.
pub fn make_token() -> TileIrOp {
    TileIrOp::Opaque {
        opcode: Opcode::MakeToken,
        operands: Vec::new(),
        results: vec![TileIrType::Token],
        attributes: Vec::new(),
    }
}

/// Whether an opcode is one of the token-ordering family.
///
/// These are the ops whose signature is `(..., Option<Token>) -> (..., Token)`,
/// so a pipeline that reasons about ordering can find them by code.
pub fn is_token_op(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::MakeToken
            | Opcode::JoinTokens
            | Opcode::LoadPtrTko
            | Opcode::LoadViewTko
            | Opcode::StorePtrTko
            | Opcode::StoreViewTko
            | Opcode::AtomicRMW
            | Opcode::AtomicCAS
            | Opcode::AtomicRedViewTko
            | Opcode::MemoryFenceAliasTko
            | Opcode::GdcLaunchDependentsTko
            | Opcode::GdcWaitTko
    )
}

/// The token a loop accumulates, and how to fold the next one in.
///
/// A loop's stores are independent of each other: each reads the loop-invariant
/// view token, so they fork instead of chaining. What the loop must still say is
/// that *all* of them finished by the time it exits. The accumulator is how:
/// entry token `acc`, and each iteration folds in the token that iteration's
/// store produced. On exit the accumulator stands for every write in the loop,
/// so one consumer after the loop waits for all of them at once.
///
/// ```text
///   acc0 = make_token
///   for i:
///     acc_{i+1} = join_tokens(acc_i, store_i)
///   after: consumer(acc_n)      // waited for every store
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopToken {
    /// The token that stands for every operation so far, including this
    /// iteration's when `fold` has run.
    pub accumulator: ValueId,
}

impl LoopToken {
    /// Start a loop's accumulator from a token with nothing behind it.
    pub fn new(entry: ValueId) -> Self {
        Self { accumulator: entry }
    }

    /// Fold this iteration's store token into the accumulator.
    ///
    /// The op is returned rather than placed: the caller owns the value
    /// counter, and an ordering edge that no block defines is no edge at all.
    pub fn fold(&mut self, iteration_token: ValueId) -> TileIrOp {
        // The join is built from the token the loop has accumulated so far, and
        // the result of that join is what the next iteration accumulates. The
        // caller owns the joined value's id, so it passes it back in.
        let op = join_tokens(self.accumulator, iteration_token);
        op
    }

    /// Record the value the caller gave the [`LoopToken::fold`] join.
    ///
    /// A fold and the value it defines are two steps because only the caller
    /// can allocate an id.
    pub fn advance(&mut self, joined: ValueId) {
        self.accumulator = joined;
    }

    /// The token a consumer after the loop waits for.
    pub fn exit_token(self) -> ValueId {
        self.accumulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_ir::{MemoryOrdering, MemoryScope};

    #[test]
    fn make_token_defines_a_token() {
        match make_token() {
            TileIrOp::Opaque {
                opcode, results, ..
            } => {
                assert_eq!(opcode, Opcode::MakeToken);
                assert_eq!(results, vec![TileIrType::Token]);
            }
            other => panic!("make_token built {other:?}"),
        }
    }

    #[test]
    fn join_tokens_names_both_operands() {
        // Two unrelated loads meet nowhere but the join; the joined token is the
        // value a store names to wait for both.
        let (t1, t2) = (ValueId(7), ValueId(9));
        match join_tokens(t1, t2) {
            TileIrOp::Opaque {
                opcode, operands, ..
            } => {
                assert_eq!(opcode, Opcode::JoinTokens);
                assert_eq!(operands, vec![t1, t2]);
            }
            other => panic!("join_tokens built {other:?}"),
        }
    }

    #[test]
    fn loop_accumulator_advances_one_iteration_at_a_time() {
        // The accumulator after n folds must be the nth iteration's token: that
        // is what makes one consumer after the loop wait for all of them.
        let mut acc = LoopToken::new(ValueId(0));
        let stores = [ValueId(10), ValueId(11), ValueId(12)];
        for (i, store) in stores.iter().enumerate() {
            // What the fold waits on is whatever the previous iteration left in
            // the accumulator: the entry token on the first pass, the join
            // value after it.
            let expected_acc = if i == 0 { ValueId(0) } else { ValueId(20 + i as u32 - 1) };
            match acc.fold(*store) {
                TileIrOp::Opaque { operands, .. } => {
                    assert_eq!(operands[0], expected_acc, "fold {i} waited on the wrong token");
                    assert_eq!(operands[1], *store);
                }
                other => panic!("fold built {other:?}"),
            }
            // The caller defines the joined value; its id becomes the next
            // iteration's accumulator.
            acc.advance(ValueId(20 + i as u32));
        }
        assert_eq!(acc.exit_token(), ValueId(22));
    }

    #[test]
    fn default_semantics_are_weak_and_block_scoped() {
        let s = TkoSemantics::default();
        assert_eq!(s.token, None);
        assert_eq!(s.ordering, MemoryOrdering::Weak);
        assert_eq!(s.scope, MemoryScope::TileBlock);
    }

    #[test]
    fn device_scope_and_ordering_are_separate_answers() {
        // A store can be complete (token taken) and still not visible
        // device-wide; the two fields say so independently.
        let s = TkoSemantics::waiting(Some(ValueId(3)))
            .with_device_scope()
            .with_ordering(MemoryOrdering::Release);
        assert_eq!(s.token, Some(ValueId(3)));
        assert_eq!(s.scope, MemoryScope::Device);
        assert_eq!(s.ordering, MemoryOrdering::Release);
    }

    #[test]
    fn the_tko_family_is_recognizable_by_opcode() {
        // Every op whose signature carries a token in and out.
        for opcode in [
            Opcode::MakeToken,
            Opcode::JoinTokens,
            Opcode::LoadPtrTko,
            Opcode::LoadViewTko,
            Opcode::StorePtrTko,
            Opcode::StoreViewTko,
            Opcode::AtomicRMW,
            Opcode::AtomicCAS,
            Opcode::AtomicRedViewTko,
            Opcode::MemoryFenceAliasTko,
            Opcode::GdcLaunchDependentsTko,
            Opcode::GdcWaitTko,
        ] {
            assert!(is_token_op(opcode), "{} must be a token op", opcode.name());
        }
        assert!(!is_token_op(Opcode::AddF));
    }
}
