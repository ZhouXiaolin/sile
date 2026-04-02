use sile::{Device, Tensor};

#[sile::kernel]
fn softmax(x: &Tensor<f32>, y: &mut Tensor<f32>) {
    let _tid = sile::tile::id().0;
    // Placeholder — will be replaced with real softmax kernel DSL in later tasks
}

fn main() -> Result<(), sile::Error> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    let (m, n) = (4i64, 8i64);
    let data: Vec<f32> = (0..(m * n) as i32).map(|v| v as f32).collect();
    let x = Tensor::from_vec(data, [m, n], &device)?;
    let mut y = Tensor::zeros([m, n], &device)?;

    softmax(&x, &mut y)
        .grid((2, 1, 1))
        .apply(&stream)?;

    let y_host = y.to_vec(&stream)?;
    println!("{:?}", y_host);
    Ok(())
}
