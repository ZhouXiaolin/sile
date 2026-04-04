use sile::{Device, Tensor, tile};

#[sile::kernel]
fn vec_add<const TILE: i64>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, { [TILE] }>,
) {
    let pid = tile::id().0;
    let tile_a = a.load_tile([TILE], [pid]);
    let tile_b = b.load_tile([TILE], [pid]);
    c.store(tile_a + tile_b);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    const TILE: i64 = 2;
    let len = 16i64;
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        [len],
        &device,
    )?;
    let b = Tensor::from_vec(vec![2.0; len as usize], [len], &device)?;
    let mut c = Tensor::zeros([len], &device)?;

    vec_add::<TILE>(&a, &b, &mut c)
        .grid(((len / TILE) as u32, 1, 1))
        .apply(&stream)?;

    println!("{:?}", c.to_vec(&stream)?);
    Ok(())
}
