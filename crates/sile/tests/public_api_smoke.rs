use sile::{Device, Stream, Tensor};

#[test]
fn cpu_device_and_tensor_api_exist() {
    let device = Device::cpu();
    let stream = device.create_stream().expect("stream should be created");
    let tensor = Tensor::zeros([4], &device).expect("tensor should be created");

    let _: Stream = stream;
    assert_eq!(tensor.shape(), &[4]);
}
