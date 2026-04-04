use sile::Device;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn default_device_is_cpu_when_env_is_missing() {
    let _guard = env_lock().lock().unwrap();
    unsafe { std::env::remove_var("SILE_DEVICE") };

    let device = Device::default().expect("default device should resolve");
    assert!(matches!(device, Device::Cpu(_)));
}

#[test]
fn default_device_uses_requested_backend_name() {
    let _guard = env_lock().lock().unwrap();
    unsafe { std::env::set_var("SILE_DEVICE", "METAL") };

    let device = Device::default().expect("default device should resolve");
    assert!(matches!(device, Device::Metal(_)));
}

#[test]
fn invalid_device_name_is_rejected() {
    let _guard = env_lock().lock().unwrap();
    unsafe { std::env::set_var("SILE_DEVICE", "BAD") };

    let err = Device::default().expect_err("invalid backend should fail");
    assert!(err.to_string().contains("SILE_DEVICE"));
}
