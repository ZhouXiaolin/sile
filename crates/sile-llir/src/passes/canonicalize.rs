use crate::Function;

/// Canonical LLIR rewrites live here.
///
/// The active implementation is intentionally minimal while the pass pipeline
/// is being established. Profitability-driven rewrites should grow here rather
/// than in MIR lowering.
pub fn run(func: Function) -> Function {
    func
}
