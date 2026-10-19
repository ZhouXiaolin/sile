//! `static_extent`: the input declares a static shape, so the compiler knows
//! the extent without an `assume`.

use sile::{Device, Tensor};
use kernels::static_extent;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::Tensor;

    #[sile::entry]
    pub fn static_extent<const S: i64>(
        z: &mut Tensor<f32, { [S] }>,
        x: &Tensor<f32, { [64] }>,
    ) {
        let part = x.partition(shape![S]);
        for i in 0i64..4i64 {
            let tile_x = part.load([i + 1]);
            z.store(tile_x);
        }
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    const S: i64 = 2;
    let x = Tensor::from_vec(vec![1.0f32; 64], [64], &device)?;
    let mut z = Tensor::zeros([64], &device)?;
    let mut z_part = z.partition_mut([S]);
    static_extent::<S>(&mut z_part, &x).apply(&stream)?;
    println!("static_extent ok on {device:?}");
    Ok(())
}
