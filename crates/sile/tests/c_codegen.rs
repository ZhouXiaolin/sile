use sile::{
    backend_ir::ir::{BackendInstruction, BackendKernel, BackendOp},
    codegen,
};

#[test]
fn c_codegen_emits_vec_add_loop() {
    let kernel = BackendKernel {
        op: BackendOp::VecAdd1D,
        tile_rank: 1,
        tile_shape_symbols: vec!["S".to_string()],
        instructions: vec![BackendInstruction::Compute {
            dest: "c".to_string(),
            op: "add".to_string(),
            args: vec!["a".to_string(), "b".to_string()],
        }],
    };

    let c = codegen::c::generate(&kernel).unwrap();
    assert!(c.contains("void sile_kernel_vec_add"));
    assert!(c.contains("c[base + i] = a[base + i] + b[base + i];"));
}
