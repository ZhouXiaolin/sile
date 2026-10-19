//! `hoist`: an access whose index is a compile-time constant while the extent
//! is dynamic, so the check is decided from launch-time facts.

use sile::{Device, Tensor};
use kernels::hoist;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn hoist<const S: i64>(
        z: &mut Tensor<f32, { [S] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part = x.partition(shape![S]);
        let tile_x = part.load([0i64]);
        z.store(tile_x);
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    const S: i64 = 2;
    let x = Tensor::from_vec(vec![1.0f32; 8], [8], &device)?;
    let mut z = Tensor::zeros([8], &device)?;
    let mut z_part = z.partition_mut([S]);
    hoist::<S>(&mut z_part, &x).apply(&stream)?;
    println!("hoist ok on {device:?}");
    Ok(())
}
