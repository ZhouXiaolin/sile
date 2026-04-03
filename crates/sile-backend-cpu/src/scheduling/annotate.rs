use super::dependency::{
    analyze_memory_accesses, find_natural_loops, has_loop_carried_dependency, is_reduction_pattern,
    ReductionType,
};
use sile_lir::ir::{Constant, Function, Value};

#[derive(Debug, Clone)]
pub struct ScheduleAnnotation {
    pub regions: Vec<ParallelRegion>,
}

#[derive(Debug, Clone)]
pub enum ParallelRegion {
    ParallelFor {
        loop_var: Value,
        bounds: (Value, Value),
        body_blocks: Vec<String>,
        simd_regions: Vec<SimdRegion>,
    },
    ParallelReduction {
        loop_var: Value,
        bounds: (Value, Value),
        body_blocks: Vec<String>,
        reduction_op: ReductionOp,
        accumulator: Value,
    },
}

#[derive(Debug, Clone)]
pub struct SimdRegion {
    pub loop_var: Value,
    pub bounds: (Value, Value),
    pub body_blocks: Vec<String>,
    pub vector_width: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Max,
    Sum,
    Min,
    Product,
}

pub fn annotate(func: &Function) -> ScheduleAnnotation {
    let mut annotation = ScheduleAnnotation { regions: vec![] };

    let loops = find_natural_loops(func);

    for loop_info in &loops {
        let accesses = analyze_memory_accesses(func, loop_info);

        if let Some(reduction_type) = is_reduction_pattern(&accesses) {
            let reduction_op = match reduction_type {
                ReductionType::Sum => ReductionOp::Sum,
                ReductionType::Max => ReductionOp::Max,
                ReductionType::Min => ReductionOp::Min,
                ReductionType::Product => ReductionOp::Product,
            };
            annotation.regions.push(ParallelRegion::ParallelReduction {
                loop_var: loop_info
                    .induction_var
                    .clone()
                    .unwrap_or(Value::Const(Constant::Int(0))),
                bounds: (
                    Value::Const(Constant::Int(0)),
                    loop_info
                        .bound
                        .clone()
                        .unwrap_or(Value::Const(Constant::Int(0))),
                ),
                body_blocks: loop_info.body_blocks.clone(),
                reduction_op,
                accumulator: Value::Const(Constant::Int(0)),
            });
        } else if !has_loop_carried_dependency(&accesses) {
            annotation.regions.push(ParallelRegion::ParallelFor {
                loop_var: loop_info
                    .induction_var
                    .clone()
                    .unwrap_or(Value::Const(Constant::Int(0))),
                bounds: (
                    Value::Const(Constant::Int(0)),
                    loop_info
                        .bound
                        .clone()
                        .unwrap_or(Value::Const(Constant::Int(0))),
                ),
                body_blocks: loop_info.body_blocks.clone(),
                simd_regions: vec![],
            });
        }
    }

    annotation
}
