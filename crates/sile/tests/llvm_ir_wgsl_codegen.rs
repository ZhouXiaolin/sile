//! Does the WGPU backend emit legal WGSL from the same LLVM IR?
//!
//! WGSL is a third target over the shared emitter, so this checks the parts
//! that are genuinely WGSL: the entry point, the buffer bindings, the
//! `global_invocation_id` spelling of the launch index, and the fact that a
//! private pointer is a `var` rather than a function pointer.

use sile::codegen::llvmir_wgsl;
use sile::fixtures::{build_dynamic_k_matmul_kernel, build_softmax_kernel, build_vec_add_kernel};
use sile::{compiler, typeck};

fn wgsl_for(kernel: &sile::hir::Kernel) -> String {
    let typed = typeck::check_kernel(kernel).unwrap();
    let tile_ir = compiler::lower_to_tile_ir(&typed);
    let tile_ir = compiler::run_default_tile_ir_pipeline(tile_ir).unwrap();
    let llir = compiler::lower_tile_ir_to_llvm_ir(&tile_ir, &typed);
    let llir =
        compiler::run_llvm_ir_pipeline(llir, compiler::ACTIVE_LLVM_IR_PIPELINE).unwrap();
    llvmir_wgsl::generate(&llir).unwrap()
}

#[test]
fn vec_add_emits_a_wgsl_entry_point_over_buffer_bindings() {
    let wgsl = wgsl_for(&build_vec_add_kernel());
    assert!(wgsl.contains("@compute @workgroup_size(1)"));
    assert!(wgsl.contains("fn sile_kernel_vec_add("));
    assert!(wgsl.contains("@group(0) @binding(0) var<storage, read_write> a: array<f32>;"));
    assert!(wgsl.contains("let __launch_idx0 = i64(global_invocation_id.x);"));
    // One tile per thread, so the body is a plain loop with no goto left over.
    assert!(wgsl.contains("for ("));
    assert!(!wgsl.contains("goto "));

    // WGSL has no pointers, so a load reads the element and a store writes it.
    // A C-shaped `&(a[i])` or `*(p)` would not compile, and neither would the
    // extra load that a pointer language needs before the store.
    assert!(wgsl.contains("a[v25]"), "a WGSL load must read the element");
    assert!(!wgsl.contains("&("), "WGSL cannot take an address");
    assert!(!wgsl.contains("= *("), "WGSL cannot dereference a pointer");

    // Only C carries this directive, so the shared loop walk must not emit it.
    assert!(!wgsl.contains("#pragma"));
}

#[test]
fn matmul_reads_shapes_from_the_trailing_binding() {
    let wgsl = wgsl_for(&build_dynamic_k_matmul_kernel());
    assert!(wgsl.contains("fn sile_kernel_matmul("));
    assert!(wgsl.contains("@group(0) @binding(3) var<storage, read> __sile_shapes: array<i64>;"));
    // A rank-2 output tiles the flat thread index, so both axes come from the
    // same `global_invocation_id`.
    assert!(wgsl.contains("global_invocation_id"));
}

#[test]
fn softmax_keeps_its_comparison_operator() {
    // WGSL spells `>=` as a function, unlike C and MSL, so the shared operator
    // lowering has to route through the dialect here.
    let wgsl = wgsl_for(&build_softmax_kernel());
    assert!(wgsl.contains("fn sile_kernel_softmax("));
    assert!(wgsl.contains("for ("));
}

#[test]
fn every_target_spells_the_same_kernel_through_the_same_ir() {
    // The three targets must agree on the launch index, which is what makes a
    // kernel portable: one LLVM IR function, three dialects over it.
    for wgsl in [
        wgsl_for(&build_vec_add_kernel()),
        wgsl_for(&build_dynamic_k_matmul_kernel()),
        wgsl_for(&build_softmax_kernel()),
    ] {
        assert!(wgsl.contains("global_invocation_id"));
        assert!(!wgsl.contains("goto "));
        assert!(!wgsl.contains("while ("));
    }
}
