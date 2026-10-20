//! The predicate algebra: a canonical form for the integer goals an access
//! check states, and the stage at which each one can be decided.
//!
//! An access check is a pair of linear goals per axis, `0 <= i` and
//! `i * tile < extent`. Both are relations over integer expressions, and the
//! expression may mention quantities the compiler does not know: a tensor's
//! extent is fixed by the host before the launch, and a loop counter is only a
//! value once the kernel runs. The same goal is therefore decided at one of
//! three stages, and the earliest stage that can decide it is the cheapest one:
//!
//! - [`Stage::Jit`] — the goal is a constant fact.
//! - [`Stage::Launch`] — the host knows every operand, so the check moves out of
//!   the kernel and runs once per launch.
//! - [`Stage::Device`] — nothing else decided it, so the check becomes an
//!   `assert` inside the kernel.
//!
//! A goal's stage is the maximum over its atoms, so "can this move to the host"
//! is computed rather than judged case by case.
//!
//! ## Normal form
//!
//! A [`Term`] is a sparse coefficient map rather than an expression tree, so
//! operand order, constant folding and like-term combination are automatic: two
//! terms are semantically equal exactly when they are structurally equal, which
//! makes `PartialEq` the equality test. This follows LLVM's ScalarEvolution and
//! MLIR's `AffineExpr`. Anything non-linear (a residue, an opaque product)
//! enters as a leaf [`Atom`], the way ScalarEvolution treats `SCEVUnknown`.
//!
//! Overflow follows MLIR: a construction that would overflow `i64` is not
//! canonicalizable and returns `None`, so the caller falls back to a later
//! stage. That keeps the algebra sound — a term that cannot be formed fails to
//! discharge a goal early, it never states a wrong equality.

use std::collections::BTreeMap;

/// When a value becomes known: `Jit ⊑ Launch ⊑ Device`.
///
/// Earlier is cheaper — a fact decided at JIT time costs nothing at run time,
/// and one decided at launch costs no device registers. The ordering is what
/// makes [`Term::stage`] a maximum rather than a lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Jit,
    Launch,
    Device,
}

/// An indivisible operand, tagged with the stage at which its value is known.
///
/// Atoms are identified by *resolved* symbols, never by source names: a tensor
/// axis by parameter position, an induction variable by its HIR value id. That
/// makes the stage of a term structural, and it keeps a name from standing for
/// two different quantities after inlining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// `extent(param, axis)` — a tensor axis extent, by kernel parameter
    /// position. The host has every extent before it launches.
    ///
    /// This is the **root** frame: the whole tensor, which is what a declared
    /// precondition is stated over.
    Dim { param: usize, axis: usize },
    /// An induction variable, by HIR value id.
    Iv(u32),
    /// The tile-block id on grid axis `k`. A per-block runtime value.
    TileBlockId(usize),
    /// The number of tile blocks on grid axis `k`. The host fixes the grid
    /// before the launch.
    ///
    /// This is the CTA **slab** count, `div_ceil(root_shape, slab_shape)` — not
    /// the tile count inside the kernel-visible view. The two are different
    /// quantities for a `&mut` parameter, and treating one as the other admits
    /// out-of-bounds stores.
    NumTileBlocks(usize),
    /// `view_extent(param, axis)` — the extent of the parameter's
    /// kernel-visible view.
    ///
    /// This is the **view** frame, and it is a separate atom from [`Atom::Dim`]
    /// on purpose: for an immutable parameter the view is the whole tensor and
    /// the two agree, but for a slabbed `&mut` parameter it is one block's
    /// piece. Keeping the frames as different variants makes a root-frame fact
    /// unable to entail a view-frame goal by structural equality, so the frame
    /// confusion is unrepresentable rather than merely avoided.
    ViewExtent { param: usize, axis: usize },
    /// `ceil(extent(param, axis) / tile)` — how many `tile`-sized tiles fit
    /// along a tensor axis. The tile size is part of the atom's identity,
    /// because the same axis at two tile sizes is two different quantities.
    ///
    /// Root frame, like [`Atom::Dim`]. This is a name, not a fact: no
    /// relationship between it and `Dim` is known to the solver, so a goal over
    /// tile counts is discharged only by an explicit equality.
    TileCount {
        param: usize,
        axis: usize,
        tile: i32,
    },
}

impl Atom {
    /// The stage at which this atom's value is known.
    pub fn stage(&self) -> Stage {
        match self {
            Atom::Dim { .. }
            | Atom::NumTileBlocks(_)
            | Atom::ViewExtent { .. }
            | Atom::TileCount { .. } => Stage::Launch,
            Atom::Iv(_) | Atom::TileBlockId(_) => Stage::Device,
        }
    }
}

/// A canonical linear integer term: `sum(coeff_i * atom_i) + constant`.
///
/// `coeffs` holds no zero coefficients and is ordered by [`Atom`], which is what
/// makes structural equality semantic equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Term {
    coeffs: BTreeMap<Atom, i64>,
    constant: i64,
}

impl Term {
    /// The constant term `c`.
    pub fn constant(c: i64) -> Self {
        Term {
            coeffs: BTreeMap::new(),
            constant: c,
        }
    }

    /// The term `atom`.
    pub fn atom(a: Atom) -> Self {
        Term::affine(a, 1, 0)
    }

    /// `scale * atom + offset`. A zero scale folds to the constant `offset`.
    pub fn affine(a: Atom, scale: i64, offset: i64) -> Self {
        if scale == 0 {
            return Term::constant(offset);
        }
        let mut coeffs = BTreeMap::new();
        coeffs.insert(a, scale);
        Term {
            coeffs,
            constant: offset,
        }
    }

    /// The whole-term constant, when the term has no atoms.
    pub fn as_constant(&self) -> Option<i64> {
        self.coeffs.is_empty().then_some(self.constant)
    }

    /// The coefficient map, sorted and free of zeros.
    pub fn coeffs(&self) -> &BTreeMap<Atom, i64> {
        &self.coeffs
    }

    /// The constant part, which may be nonzero even when atoms are present.
    pub fn constant_part(&self) -> i64 {
        self.constant
    }

    /// When the whole term becomes evaluable: the maximum over its atoms, or
    /// [`Stage::Jit`] for a constant.
    pub fn stage(&self) -> Stage {
        self.coeffs
            .keys()
            .map(Atom::stage)
            .max()
            .unwrap_or(Stage::Jit)
    }

    /// `self + other`, or `None` on `i64` overflow.
    pub fn add(&self, other: &Term) -> Option<Term> {
        let mut coeffs = self.coeffs.clone();
        for (atom, &c) in &other.coeffs {
            let entry = coeffs.entry(*atom).or_insert(0);
            *entry = entry.checked_add(c)?;
            if *entry == 0 {
                coeffs.remove(atom);
            }
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_add(other.constant)?,
        })
    }

    /// `-self`, or `None` on overflow.
    pub fn neg(&self) -> Option<Term> {
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &self.coeffs {
            coeffs.insert(*atom, c.checked_neg()?);
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_neg()?,
        })
    }

    /// `self - other`, or `None` on overflow.
    pub fn sub(&self, other: &Term) -> Option<Term> {
        self.add(&other.neg()?)
    }

    /// `k * self`, or `None` on overflow.
    pub fn mul_const(&self, k: i64) -> Option<Term> {
        if k == 0 {
            return Some(Term::constant(0));
        }
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &self.coeffs {
            coeffs.insert(*atom, c.checked_mul(k)?);
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_mul(k)?,
        })
    }

    /// A representative that is the same for `t` and `-t`: negate iff the
    /// leading coefficient (or the constant, when there are no atoms) is
    /// negative.
    ///
    /// This gives the sign-invariant predicates (`== 0`, `!= 0`, `% d == 0`)
    /// one representative each, so `a == b` and `b == a` unify.
    fn sign_canonical(&self) -> Term {
        let leading_negative = match self.coeffs.iter().next() {
            Some((_, &c)) => c < 0,
            None => self.constant < 0,
        };
        if leading_negative {
            // A canonical term's negation does not overflow in practice. If it
            // somehow would, keep the original: still sound, since a missed
            // equality only sinks the goal to a later stage.
            self.neg().unwrap_or_else(|| self.clone())
        } else {
            self.clone()
        }
    }

    /// `(atom, scale, offset)` when the term is `scale * atom + offset`, i.e.
    /// exactly one atom.
    ///
    /// This is the single-variable fragment a loop hoister can substitute a
    /// strongest instance into. A richer term returns `None`, so the hoister
    /// declines it rather than guessing.
    pub fn as_single_affine(&self) -> Option<(Atom, i64, i64)> {
        if self.coeffs.len() != 1 {
            return None;
        }
        let (&atom, &scale) = self.coeffs.iter().next().expect("one atom");
        Some((atom, scale, self.constant))
    }

    /// Evaluate against an atom resolver. `None` if an atom is unresolved or the
    /// arithmetic overflows.
    pub fn eval(&self, resolve_atom: &impl Fn(&Atom) -> Option<i64>) -> Option<i64> {
        let mut acc = self.constant;
        for (atom, &c) in &self.coeffs {
            let v = resolve_atom(atom)?;
            acc = acc.checked_add(c.checked_mul(v)?)?;
        }
        Some(acc)
    }
}

/// A relation over canonical terms.
///
/// Every constructor puts its predicate into a canonical representative, so two
/// predicates that state the same goal compare equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// `term == 0`. An equality `a == b` is stored as `Zero(a - b)`.
    Zero(Term),
    /// `term != 0`.
    Nonzero(Term),
    /// `term > 0`. The canonical form of `a < b` is `Positive(b - a)`.
    ///
    /// Unlike `Zero` and `Nonzero` this is not sign-invariant — `a < b` is not
    /// `b < a` — so no sign canonicalization applies.
    Positive(Term),
    /// `term % divisor == 0`, with `divisor >= 1`.
    DivisibleBy { term: Term, divisor: i64 },
}

impl Predicate {
    /// `a == b`, as `Zero(a - b)`. `None` on overflow.
    pub fn eq(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Zero(a.sub(b)?.sign_canonical()))
    }

    /// `term != 0`.
    pub fn nonzero(term: Term) -> Predicate {
        Predicate::Nonzero(term.sign_canonical())
    }

    /// `a < b`, as `Positive(b - a)`. `None` on overflow.
    pub fn lt(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Positive(b.sub(a)?))
    }

    /// `a <= b`, as `Positive(b - a + 1)`. `None` on overflow.
    ///
    /// This is the form an access-safety goal takes: an index drawn from
    /// `[0, ceil(a / t))` stays inside an axis of extent `b` exactly when
    /// `a <= b` bounds it above. Stating equality there instead would reject
    /// every `b` strictly larger than `a`, which is safe.
    pub fn le(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Positive(b.add(&Term::constant(1))?.sub(a)?))
    }

    /// `term % divisor == 0`.
    ///
    /// Coefficients and constant are reduced into `[0, divisor)`, which is valid
    /// because `c * a ≡ (c mod d) * a (mod d)` for every integer `a`. `None` if
    /// `divisor < 1`.
    pub fn divisible_by(term: Term, divisor: i64) -> Option<Predicate> {
        if divisor < 1 {
            return None;
        }
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &term.coeffs {
            let r = c.rem_euclid(divisor);
            if r != 0 {
                coeffs.insert(*atom, r);
            }
        }
        let reduced = Term {
            coeffs,
            constant: term.constant.rem_euclid(divisor),
        };
        Some(Predicate::DivisibleBy {
            term: reduced.sign_canonical(),
            divisor,
        })
    }

    /// The earliest stage at which this predicate can be decided.
    pub fn stage(&self) -> Stage {
        match self {
            Predicate::Zero(t) | Predicate::Nonzero(t) | Predicate::Positive(t) => t.stage(),
            Predicate::DivisibleBy { term, .. } => term.stage(),
        }
    }

    /// Evaluate against an atom resolver. `None` if an atom is unresolved.
    ///
    /// The same predicate is folded at JIT time and checked at launch, because
    /// the resolver is the only thing that differs.
    pub fn eval(&self, resolve_atom: &impl Fn(&Atom) -> Option<i64>) -> Option<bool> {
        match self {
            Predicate::Zero(t) => Some(t.eval(resolve_atom)? == 0),
            Predicate::Nonzero(t) => Some(t.eval(resolve_atom)? != 0),
            Predicate::Positive(t) => Some(t.eval(resolve_atom)? > 0),
            Predicate::DivisibleBy { term, divisor } => {
                Some(term.eval(resolve_atom)?.rem_euclid(*divisor) == 0)
            }
        }
    }
}

/// A predicate the compiler could not decide but the host can, carried to the
/// launcher with the message to print if it fails.
///
/// It is the same [`Predicate`] the compiler worked on, evaluated at launch
/// instead of emitted as an in-kernel `assert`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCheck {
    pub predicate: Predicate,
    pub cause: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(param: usize, axis: usize) -> Atom {
        Atom::Dim { param, axis }
    }

    #[test]
    fn like_terms_combine_and_order_is_irrelevant() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        // (a + b) and (b + a) reach the same normal form.
        assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
    }

    #[test]
    fn like_terms_cancel_to_drop_zero_coefficients() {
        let a = Term::atom(dim(0, 0));
        let sum = a.add(&a.neg().unwrap()).unwrap();
        assert_eq!(sum, Term::constant(0));
        assert!(sum.coeffs().is_empty(), "a cancelled coefficient is removed");
    }

    #[test]
    fn affine_consolidates_scale_atom_offset() {
        let t = Term::affine(dim(0, 1), 3, 5);
        assert_eq!(t.as_single_affine(), Some((dim(0, 1), 3, 5)));
        assert_eq!(Term::affine(dim(0, 1), 0, 7), Term::constant(7));
    }

    #[test]
    fn as_single_affine_declines_a_two_atom_term() {
        let two = Term::atom(dim(0, 0))
            .add(&Term::atom(dim(1, 0)))
            .unwrap();
        assert_eq!(two.as_single_affine(), None);
    }

    #[test]
    fn stage_is_max_over_atoms() {
        assert_eq!(Term::constant(3).stage(), Stage::Jit);
        assert_eq!(Term::atom(dim(0, 0)).stage(), Stage::Launch);
        assert_eq!(Term::atom(Atom::Iv(7)).stage(), Stage::Device);
        // A launch atom plus a device atom is a device term.
        let mixed = Term::atom(dim(0, 0))
            .add(&Term::atom(Atom::Iv(7)))
            .unwrap();
        assert_eq!(mixed.stage(), Stage::Device);
    }

    #[test]
    fn overflow_bails_to_none() {
        let big = Term::constant(i64::MAX);
        assert_eq!(big.add(&Term::constant(1)), None);
        assert_eq!(Term::constant(i64::MIN).neg(), None);
        assert_eq!(big.mul_const(2), None);
    }

    #[test]
    fn equality_is_symmetric_after_canonicalization() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        assert_eq!(Predicate::eq(&a, &b), Predicate::eq(&b, &a));
    }

    #[test]
    fn nonzero_is_sign_invariant() {
        let a = Term::atom(dim(0, 0));
        assert_eq!(
            Predicate::nonzero(a.clone()),
            Predicate::nonzero(a.neg().unwrap())
        );
    }

    #[test]
    fn divisibility_reduces_coefficients_mod_divisor() {
        // 4 * a is divisible by 2 exactly when 0 * a is.
        let four = Term::affine(dim(0, 0), 4, 0);
        let reduced = Predicate::divisible_by(four, 2).unwrap();
        assert_eq!(reduced, Predicate::divisible_by(Term::constant(0), 2).unwrap());
        assert_eq!(Predicate::divisible_by(Term::constant(0), 0), None);
    }

    #[test]
    fn eval_resolves_atoms_and_decides() {
        let t = Term::affine(dim(0, 0), 2, 1);
        assert_eq!(t.eval(&|_| Some(3)), Some(7));
        assert_eq!(t.eval(&|_| None), None);
        let goal = Predicate::lt(&t, &Term::constant(10)).unwrap();
        assert_eq!(goal.eval(&|_| Some(3)), Some(true));
        assert_eq!(goal.eval(&|_| Some(5)), Some(false));
    }

    #[test]
    fn lt_is_directional() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        assert_ne!(Predicate::lt(&a, &b), Predicate::lt(&b, &a));
    }

    #[test]
    fn le_admits_equality_and_rejects_a_larger_left() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        // a <= b holds when a == b, unlike a < b.
        let le = Predicate::le(&a, &b).unwrap();
        let resolve = |atom: &Atom| match atom {
            Atom::Dim { param, .. } if *param == 0 => Some(5),
            Atom::Dim { .. } => Some(5),
            _ => None,
        };
        assert_eq!(le.eval(&resolve), Some(true));
        let too_big = |atom: &Atom| match atom {
            Atom::Dim { param, .. } if *param == 0 => Some(6),
            Atom::Dim { .. } => Some(5),
            _ => None,
        };
        assert_eq!(le.eval(&too_big), Some(false));
    }

    #[test]
    fn eval_divisibility() {
        let t = Term::affine(dim(0, 0), 1, 0);
        let goal = Predicate::divisible_by(t, 4).unwrap();
        assert_eq!(goal.eval(&|_| Some(8)), Some(true));
        assert_eq!(goal.eval(&|_| Some(9)), Some(false));
    }

    #[test]
    fn special_register_stages() {
        assert_eq!(Atom::TileBlockId(0).stage(), Stage::Device);
        assert_eq!(Atom::NumTileBlocks(0).stage(), Stage::Launch);
        assert_eq!(
            Atom::TileCount {
                param: 0,
                axis: 0,
                tile: 16
            }
            .stage(),
            Stage::Launch
        );
    }

    #[test]
    fn the_two_extent_frames_never_unify() {
        // A root-frame fact must not entail a view-frame goal by structural
        // equality: that confusion once admitted out-of-bounds stores.
        assert_ne!(Atom::Dim { param: 0, axis: 0 }, Atom::ViewExtent { param: 0, axis: 0 });
    }
}
