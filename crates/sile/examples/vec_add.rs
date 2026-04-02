use sile::{tile, Device, Tensor};

#[sile::kernel]
fn vec_add(a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>) {
    let tid = tile::id().0;
    let x = a.load_tile([4], [tid]);
    let y = b.load_tile([4], [tid]);
    c.store(x + y);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    let a = Tensor::from_vec(vec![1.0; 16], [16], &device)?;
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device)?;
    let mut c = Tensor::zeros([16], &device)?;

    vec_add(&a, &b, &mut c).grid((4, 1, 1)).apply(&stream)?;

    println!("{:?}", c.to_vec(&stream)?);
    Ok(())
}
