//! The DeviceOp layer: device work as a value, composed before it runs.
//!
//! A `DeviceOp` is the cutile shape of an async kernel call: the launch is a
//! value that owns its tensors, `.then` continues the chain on the stream
//! that ran the previous stage, and `zip` joins two chains. Nothing runs
//! until the op meets a stream — `schedule` hands it to the queue and takes
//! a future, `sync_on` schedules and waits.
//!
//! The ops own their tensors because a job outlives the call site; the sync
//! path keeps borrowing, and a kernel written for it is called inside
//! `DeviceOp::new` by a builder that moves the tensors through.

use sile_core::{Result, Stream};

/// One device computation, built and composed on the host, run on a stream.
pub struct DeviceOp<T> {
    run: Box<dyn FnOnce(&Stream) -> Result<T> + Send>,
}

impl<T: Send + 'static> DeviceOp<T> {
    /// An op from the work it does. The closure runs on the stream's worker,
    /// so plain sync launches inside it cost nothing but the launch.
    pub fn new<F>(run: F) -> Self
    where
        F: FnOnce(&Stream) -> Result<T> + Send + 'static,
    {
        Self {
            run: Box::new(run),
        }
    }

    /// Continue the chain with the previous stage's output.
    ///
    /// The continuation builds the next op and runs it on the same stream,
    /// inline: chaining never re-enqueues behind itself, so a chain of any
    /// length stays one queue turn and can never deadlock on its own stream.
    pub fn then<U, N>(self, next: N) -> DeviceOp<U>
    where
        U: Send + 'static,
        N: FnOnce(T) -> DeviceOp<U> + Send + 'static,
    {
        DeviceOp::new(move |stream| {
            let value = (self.run)(stream)?;
            next(value).run_on(stream)
        })
    }

    /// Hand the op to the stream's queue and take the future it completes.
    pub fn schedule(self, stream: &Stream) -> sile_core::DeviceFuture<T> {
        stream.dispatch(self.run)
    }

    /// Schedule and wait. The whole chain runs before this returns.
    pub fn sync_on(self, stream: &Stream) -> Result<T> {
        self.schedule(stream).block_on()
    }

    fn run_on(self, stream: &Stream) -> Result<T> {
        (self.run)(stream)
    }
}

/// `to_host_vec(tensor)`: the copy back as an op, so a chain ends the way
/// cutile's does — `.then(to_host_vec)` hands a `Vec<f32>` to the host.
pub fn to_host_vec(tensor: crate::Tensor<f32>) -> DeviceOp<Vec<f32>> {
    DeviceOp::new(move |_| Ok(tensor.into_vec()))
}
