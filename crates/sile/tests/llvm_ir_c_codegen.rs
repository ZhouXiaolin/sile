use sile::{
    codegen::llvmir_c,
    compiler,
    typeck,
};
use sile::fixtures::{build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel};

#[test]
fn dynamic_k_matmul_llir_codegen_emits_structured_c_without_helper_calls() {
    let kernel = build_dynamic_k_matmul_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_matmul("));
    assert!(c.contains("int64_t* __sile_shapes"));
    assert!(c.contains("sile_llvm_ir_matmul(a, b, c, shapes);"));
    assert!(c.contains("for ("));
    assert!(!c.contains("goto "));
    assert!(!c.contains("while (true)"));
    assert_eq!(c.matches("tile_fill_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_mma_accumulate_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(!c.contains("llir_matmul_fragment("));
    assert!(!c.contains("tile_splat_f32("));
    assert!(c.contains("*(v"));
    assert!(c.contains("&(a["));
    assert!(c.contains("&(b["));
    assert!(c.contains("&(c["));
    assert!(c.contains("float v"));
    assert!(!c.contains("while ("));
    assert!(c.matches("for (").count() >= 3);
    assert!(!c.contains("sile_shape_dim("));
    assert!(!c.contains("static const int64_t* sile_shapes"));
    assert!(!c.contains("_storage"));
    assert!(!c.contains("if (v"));
}

#[test]
fn vec_add_llir_codegen_emits_explicit_c_loops() {
    let kernel = build_vec_add_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_vec_add("));
    assert!(c.contains("for ("));
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_add_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(c.contains(" + "));
    assert!(c.contains("&(a["));
    assert!(c.contains("&(b["));
    assert!(c.contains("&(c["));
    assert!(!c.contains("while ("));
    assert!(!c.contains("__launch_idx1"));
    assert!(!c.contains(" ? "));
    assert!(!c.contains("_storage"));
}

#[test]
fn softmax_llir_codegen_emits_explicit_reduce_and_broadcast_loops() {
    let kernel = build_softmax_kernel();
    let typed = typeck::check_kernel(&kernel).unwrap();

    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::dce::run(tile_ir);
    let llir_func = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir_func =
        compiler::run_llvm_ir_pipeline(llir_func, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    let c = llvmir_c::generate(&llir_func).unwrap();

    assert!(c.contains("void sile_llvm_ir_softmax("));
    assert_eq!(c.matches("tile_load_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_reduce_max_axis1_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_broadcast_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_sub_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_exp_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_reduce_sum_axis1_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_div_2d_f32(").count(), 0);
    assert_eq!(c.matches("tile_store_2d_f32(").count(), 0);
    assert!(c.contains("expf("));
    assert!(c.contains(" ? "));
    assert!(c.contains("for ("));
    assert!(!c.contains("while ("));
    assert!(!c.contains("llir_reduce_add("));
    assert!(!c.contains("llir_reduce_max("));
    assert!(!c.contains("tile_splat_f32("));

    // Verify alloca count for softmax — the fusion optimizations should keep this low.
    // Currently: tile_x(2x8), tile_x_max(2x1), num(2x8), denom(2x1), div_result(2x8) = 5
    // TODO: reduce to ≤ 3 with multi-use fusion
    let alloca_count = llir_func
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i.op, sile_llvm_ir::InstOp::Alloca { .. }))
        .count();
    assert!(
        alloca_count <= 5,
        "softmax alloca count should be ≤ 5, got {alloca_count}"
    );
}






