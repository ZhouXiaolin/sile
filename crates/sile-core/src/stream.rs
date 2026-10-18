//! The stream: one device's work queue, and the futures its jobs complete.

use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use crate::{Error, Result, device::Device};

/// One device's work queue.
///
/// A stream is a FIFO of jobs with one worker thread. Scheduling hands a job
/// to the worker and returns a future at once; the jobs of one stream run in
/// order, and the jobs of two streams run beside each other. That is the
/// whole async story every target tells today: on the CPU target the worker
/// runs the kernel itself, and on Metal and WGPU it performs the copies, the
/// submit and the wait that one launch costs, so two streams overlap even
/// while one launch still waits inside. Kernels of one stream do not overlap
/// until buffers live on the device, which is the next milestone, not this
/// one.
///
/// A job that dispatches onto another stream may wait on it. A job that
/// dispatches onto its own stream and waits would deadlock — the worker
/// cannot run the next job while it waits inside the current one — so a
/// chain continues inline instead, the way [`crate::stream`] combinators do.
#[derive(Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

struct StreamInner {
    device: Device,
    /// `Some` until the last handle drops. It must go before the join in
    /// `Drop`: a field outlives the drop body, so a sender left in place
    /// would hold the queue open while the join below waits for the worker
    /// to see it close.
    jobs: Option<flume::Sender<Job>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// One unit of stream work. The handle it runs on travels inside the job, so
/// a kernel launched within it joins the same queue behind it.
type Job = Box<dyn FnOnce() + Send>;

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream").field("device", &self.inner.device).finish()
    }
}

impl Stream {
    pub(crate) fn new(device: Device) -> Self {
        let (jobs, receiver) = flume::unbounded::<Job>();
        // The worker holds only the receiving end. Every queued job carries a
        // sender of its own inside its stream handle, so the queue closes when
        // the last job has run, and this loop — and the join in `Drop` — ends
        // with it. A sender kept here instead would hold the queue open
        // forever, and every program would hang on its way out.
        let worker = std::thread::Builder::new()
            .name("sile-stream".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .expect("spawn the stream worker");
        Self {
            inner: Arc::new(StreamInner {
                device,
                jobs: Some(jobs),
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Hand one job to the worker and take the future it completes.
    pub fn dispatch<T, F>(&self, job: F) -> DeviceFuture<T>
    where
        T: Send + 'static,
        F: FnOnce(&Stream) -> Result<T> + Send + 'static,
    {
        let (complete, future) = oneshot();
        // The job carries the handle it runs on, which is also what keeps the
        // queue open: as long as work is queued a sender exists, and the
        // moment the last job has run the queue closes.
        let handle = self.clone();
        self.inner
            .jobs
            .as_ref()
            .expect("the stream has not been dropped")
            .send(Box::new(move || {
                // A backend that panics would otherwise take the worker down with
                // it, and the future would never be filled: the caller waits
                // forever on work that has already failed. The panic becomes an
                // error the future carries instead.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    job(&handle)
                }))
                .unwrap_or_else(|payload| Err(Error::Compile(panic_message(payload))));
                complete.fill(result);
            }))
            .expect("the stream worker outlives its queue");
        future
    }

    /// Enqueue a barrier and wait for the queue to drain.
    pub fn synchronize(&self) -> Result<()> {
        self.dispatch(|_| Ok::<(), Error>(())).block_on()
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        // Close the queue first: the worker holds only the receiving end, so
        // with this sender gone — and every queued job's handle gone with its
        // job — recv errors and the loop ends. The join then only waits out
        // whatever job is still running.
        drop(self.jobs.take());
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
    }
}

/// The text a panicking job left behind, so the error names the real cause.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "stream job panicked".to_string()
    }
}

/// The one-shot handoff a scheduled job completes.
///
/// The worker fills the slot and wakes whoever polled; the future reads the
/// slot and parks otherwise. One mutex, because there is one value and two
/// ends.
struct Shared<T> {
    state: Mutex<OneshotState<T>>,
}

enum OneshotState<T> {
    Empty,
    Polled(Waker),
    Filled(Result<T>),
}

pub(crate) fn oneshot<T>() -> (Filler<T>, DeviceFuture<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(OneshotState::Empty),
    });
    (
        Filler {
            shared: Arc::clone(&shared),
        },
        DeviceFuture { shared },
    )
}

pub(crate) struct Filler<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Filler<T> {
    fn fill(self, value: Result<T>) {
        let waker = {
            let mut state = self.shared.state.lock().unwrap();
            let previous = std::mem::replace(&mut *state, OneshotState::Filled(value));
            match previous {
                OneshotState::Polled(waker) => Some(waker),
                _ => None,
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The future a scheduled job completes, and the unit every combinator
/// carries.
pub struct DeviceFuture<T> {
    shared: Arc<Shared<T>>,
}

impl<T> std::future::Future for DeviceFuture<T> {
    type Output = Result<T>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let mut state = self.shared.state.lock().unwrap();
        match &mut *state {
            OneshotState::Filled(_) => {
                let previous = std::mem::replace(&mut *state, OneshotState::Empty);
                match previous {
                    OneshotState::Filled(value) => Poll::Ready(value),
                    _ => unreachable!("the state was read as filled"),
                }
            }
            OneshotState::Empty => {
                *state = OneshotState::Polled(cx.waker().clone());
                Poll::Pending
            }
            OneshotState::Polled(waker) => {
                if !waker.will_wake(cx.waker()) {
                    *waker = cx.waker().clone();
                }
                Poll::Pending
            }
        }
    }
}

impl<T> DeviceFuture<T> {
    /// Run to completion on this thread. The host blocks; a stream worker
    /// that waits for the stage before the one it is building blocks the
    /// same way, which is what keeps a chain ordered.
    pub fn block_on(self) -> Result<T> {
        pollster::block_on(self)
    }

    /// Both futures as one: the pair completes when each side has. Two sides
    /// scheduled on two streams run beside each other.
    pub fn join<B>(self, other: DeviceFuture<B>) -> JoinFuture<T, B> {
        JoinFuture {
            left: Some(self),
            right: Some(other),
            left_value: None,
        }
    }
}

/// The future [`DeviceFuture::join`] returns.
pub struct JoinFuture<A, B> {
    left: Option<DeviceFuture<A>>,
    right: Option<DeviceFuture<B>>,
    left_value: Option<Result<A>>,
}

impl<A: Unpin, B: Unpin> std::future::Future for JoinFuture<A, B> {
    type Output = Result<(A, B)>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        // A DeviceFuture is Unpin — an Arc behind it — so the pinned
        // projection a join needs is a plain `Pin::new`.
        if let Some(left) = this.left.as_mut() {
            let value = std::task::ready!(std::pin::Pin::new(left).poll(cx));
            this.left = None;
            this.left_value = Some(value);
        }
        if let Some(right) = this.right.as_mut() {
            let value = std::task::ready!(std::pin::Pin::new(right).poll(cx));
            this.right = None;
            let left = this.left_value.take().expect("the left side completed first");
            return Poll::Ready(match (left, value) {
                (Ok(a), Ok(b)) => Ok((a, b)),
                (Err(error), _) | (_, Err(error)) => Err(error),
            });
        }
        Poll::Pending
    }
}
