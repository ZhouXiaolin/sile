use sile_backend::{
    BackendPassKind, CodegenTarget, SHARED_BACKEND_PIPELINE, compose_backend_pipeline,
};

#[test]
fn backend_pipeline_uses_shared_front_half_before_emit() {
    let c_pipeline = compose_backend_pipeline(CodegenTarget::C);
    let metal_pipeline = compose_backend_pipeline(CodegenTarget::Metal);

    assert_eq!(
        &c_pipeline[..SHARED_BACKEND_PIPELINE.len()],
        SHARED_BACKEND_PIPELINE
    );
    assert_eq!(
        &metal_pipeline[..SHARED_BACKEND_PIPELINE.len()],
        SHARED_BACKEND_PIPELINE
    );
    assert_eq!(
        c_pipeline.last().copied(),
        Some(BackendPassKind::Emit(CodegenTarget::C))
    );
    assert_eq!(
        metal_pipeline.last().copied(),
        Some(BackendPassKind::Emit(CodegenTarget::Metal))
    );
}
