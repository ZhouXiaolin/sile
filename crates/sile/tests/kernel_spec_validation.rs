use sile::{Device, Tensor, schedule};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = sile::tile::id().0;
    let tile_a = a.load_tile([4], [tid]);
    let tile_b = b.load_tile([4], [tid]);
    c.store(tile_a + tile_b);
}

#[test]
fn launch_validation_accepts_divisible_grid() {
    let result = schedule::require_divisible(16, 4);
    assert!(result.is_ok());
}

#[test]
fn launch_validation_rejects_non_divisible_grid() {
    let err = schedule::require_divisible(16, 3).unwrap_err();
    assert!(err.to_string().contains("not divisible"));
}
