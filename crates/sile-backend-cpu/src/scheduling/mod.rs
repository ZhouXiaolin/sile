mod annotate;
mod dependency;

pub use annotate::{ParallelRegion, ReductionOp, ScheduleAnnotation, SimdRegion, annotate};
pub use dependency::{LoopInfo, find_natural_loops, has_loop_carried_dependency};
