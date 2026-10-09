use sile::{Device, Tensor, tile};

#[sile::kernel]
fn add_via_program_id<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, { [-1] }>,
) {
    let pid = tile::program_id(0).0;
    let x = a.load_tile([TILE], [pid]);
    let y = b.load_tile([TILE], [pid]);
    c.store(x + y);
}

fn main() -> sile::Result<()> {
    const TILE: i64 = 1;
    let len = 8i64;
    let device = Device::cpu();
    let stream = device.create_stream()?;
    let a = Tensor::from_vec((1..=len).map(|i| i as f32).collect(), [len], &device)?;
    let b = Tensor::from_vec(vec![10.0; len as usize], [len], &device)?;
    let mut c = Tensor::zeros([len], &device)?;
    add_via_program_id::<TILE>(&a, &b, &mut c)
        .grid(((len / TILE) as u32, 1, 1))
        .apply(&stream)?;
    println!("{:?}", c.to_vec(&stream)?);
    Ok(())
}
