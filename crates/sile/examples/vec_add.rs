use sile::{Device, Tensor, tile};

#[sile::kernel]
fn vec_add<const S: [i32; 1]>(
    a: &Tensor<f32, { [-1] }>,
    b: &Tensor<f32, { [-1] }>,
    c: &mut Tensor<f32, S>,
) {
    let pid = get_tile_block_id().0;
    let tile_a = a.load_tile(const_shape!(S), [pid]);
    let tile_b = b.load_tile(const_shape!(S), [pid]);
    c.store(tile_a + tile_b);
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        [16],
        &device,
    )?;
    let b = Tensor::from_vec(vec![2.0; 16], [16], &device)?;
    let mut c = Tensor::zeros([16], &device)?;
    let c_partitioned = c.partition([4]);
    vec_add(&a, &b, &mut c_partitioned)
        .grid((4, 1, 1))
        .apply(&stream)?;

    let c_host_vec = c_partitioned.unpartition();
    println!("{:?}", c_host_vec.to_vec(&stream)?);
    Ok(())
}
