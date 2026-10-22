//! The Tile IR fixture, under `cargo test`.
//!
//! `scripts/check_tileir_alignment.sh` compares all 17 examples against the
//! fixtures byte for byte, but only when someone runs it by hand. This test
//! compiles the saxpy example's kernel through the same macro and the same
//! emitter the dump uses, and compares the text against the fixture, so a
//! change to the frontend, the HIR lowering or the printer fails the default
//! test run. `TILEIR_ALIGNMENT.md` holds the contract: diff = 0, byte for byte.

use sile::{Device, Tensor};

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn saxpy<const BM: i64, const BN: i64>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        a: f32,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

/// The kernel source, the launch shapes and the strides mirror
/// `examples/saxpy.rs` — the fixture is the printed Tile IR of exactly that
/// instantiation (`BM = BN = 2`, buffers 4x8, row-major strides `[8, 1]`).
#[test]
fn saxpy_tileir_matches_fixture() {
    let device = Device::cpu();
    let (m, n) = (4i64, 8i64);
    let values: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(values.clone(), [m, n], &device).unwrap();
    let mut y = Tensor::from_vec(values.clone(), [m, n], &device).unwrap();
    let mut y_part = y.partition_mut([2, 2]);

    let launcher = kernels::saxpy::<2, 2>(&mut y_part, 2.0f32, &x);
    let kernel = launcher.kernel();

    let typed = sile::hir::run_hir_pipeline(kernel, sile::hir::ACTIVE_HIR_PIPELINE)
        .map_err(|err| err.to_string())
        .unwrap();
    let text = sile::tileir_text::emit(&typed).unwrap();

    let want = include_str!("fixtures/tileir/saxpy.tileir");
    assert_eq!(
        want, text,
        "saxpy: Tile IR no longer matches crates/sile/tests/fixtures/tileir/saxpy.tileir"
    );
}
