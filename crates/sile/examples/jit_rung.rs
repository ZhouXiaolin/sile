//! `jit_rung`: a loop whose bound is `num_tiles`, so the access it indexes is
//! proven at JIT time from that bound alone.

use sile::{Device, Tensor};
use kernels::jit_rung;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn jit_rung<const S: i64>(
        z: &mut Tensor<f32, { [S] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part = x.partition(shape![S]);
        for i in 0i64..num_tiles(&part, 0) {
            let tile_x = part.load([i]);
            z.store(tile_x);
        }
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    const S: i64 = 2;
    let x = Tensor::from_vec(vec![1.0f32; 8], [8], &device)?;
    let mut z = Tensor::zeros([8], &device)?;
    let mut z_part = z.partition_mut([S]);
    jit_rung::<S>(&mut z_part, &x).apply(&stream)?;
    println!("jit_rung ok on {device:?}");
    Ok(())
}
