//! The reference results, computed by numpy instead of a Rust twin.
//!
//! Each example hands its inputs to [`numpy_ref.py`] over stdin as raw f32le
//! and reads the expected values back the same way, so one Python script
//! carries every op and the examples keep no reference math of their own.

/// The expected values for `op`, with `inputs` in order and `scalars` as text.
pub fn numpy(op: &str, scalars: &[String], inputs: &[&[f32]]) -> Vec<f32> {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};

    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/numpy_ref.py");
    let mut command = Command::new("python3");
    command.arg(script).arg(op);
    command.arg(inputs.len().to_string());
    command.args(inputs.iter().map(|input| input.len().to_string()));
    command.args(scalars);

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn python3 for the reference result");
    let mut stdin = child.stdin.take().expect("python3 stdin");
    let mut bytes = Vec::new();
    for input in inputs {
        for value in *input {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    stdin
        .write_all(&bytes)
        .expect("write the inputs to python3");
    drop(stdin);

    let mut expected_bytes = Vec::new();
    child
        .stdout
        .take()
        .expect("python3 stdout")
        .read_to_end(&mut expected_bytes)
        .expect("read the reference result");
    let status = child.wait().expect("wait for python3");
    assert!(status.success(), "numpy reference for {op} failed");

    expected_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
