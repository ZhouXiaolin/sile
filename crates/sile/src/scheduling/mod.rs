mod annotate;
mod dependency;

pub use annotate::{annotate, ParallelRegion, ReductionOp, ScheduleAnnotation, SimdRegion};
pub use dependency::{find_natural_loops, has_loop_carried_dependency, LoopInfo};
