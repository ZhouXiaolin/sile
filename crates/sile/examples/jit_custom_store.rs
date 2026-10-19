//! `jit_custom_store`, ported from
//! cutile-examples/examples/jit_custom_store.rs.
//!
//! The original is about implementing a `JitStore` backend: a byte-oriented
//! `get`/`put`/`delete`/`clear` interface a crate can implement over an object
//! store, a database, or a read-through chain. sile compiles per process and
//! keeps no disk cache, so there is no store to plug into, and this port keeps
//! only what the Tile IR decides: the kernel.
//!
//! The kernel is the same elementwise add as [`async_add`](../async_add.rs),
//! and so is the tile: the fixture pins `S = 2`, which is the shape the
//! alignment kernel is compiled at. The host side matches the original's
//! inputs: `ones(1024)` for both inputs, `zeros(1024)` partitioned the same
//! way, so every output element is 2. What differs is only where the reference
//! comes from: the original asserts against the constant 2.0, this one against
//! `numpy_ref.py vec_add`.

use sile::{Device, Tensor};
use kernels::add;

#[path = "common/mod.rs"]
mod common;

#[sile::module]
mod kernels {
    use sile::Tensor;

    #[sile::entry]
    pub fn add<const S: i64>(
        z: &mut Tensor<f32, { [S] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    const S: i64 = 2;
    let len = 1024i64;
    let x_data = vec![1.0f32; len as usize];
    let y_data = vec![1.0f32; len as usize];
    let x = Tensor::from_vec(x_data.clone(), [len], &device)?;
    let y = Tensor::from_vec(y_data.clone(), [len], &device)?;
    let mut z = Tensor::zeros([len], &device)?;
    let mut z_part = z.partition_mut([S]);

    add::<S>(&mut z_part, &x, &y).apply(&stream)?;

    let out = z_part.unpartition().to_vec(&stream)?;
    let expected = common::numpy("vec_add", &[], &[&x_data, &y_data]);
    for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
        assert!((got - want).abs() < 1e-5, "element {i}: {got}, want {want}");
    }
    println!("jit_custom_store {len} ok on {device:?}");
    Ok(())
}
