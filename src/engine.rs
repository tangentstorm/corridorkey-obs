//! Async inference worker.
//!
//! GreenFormer costs tens of milliseconds a frame. Running it inline in
//! `video_render` would peg OBS's render thread to the model's framerate and
//! stutter every other source in the scene.
//!
//! So the render thread never blocks on the model. It drops the newest 512x512
//! frame into a one-slot mailbox and picks up whatever result is ready. The
//! keyer's output lags the picture by a frame or two, which is invisible for a
//! person in front of a green screen, and the full-resolution compositing still
//! happens every frame in the shader.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::hint::{bgra_to_chw, chroma_hint, HintParams};
use crate::log;

/// Inference resolutions we ship models for, largest (best) first.
///
/// Each is a separate ONNX export: Hiera's windowed attention reshapes on spatial
/// dimensions, so one graph cannot serve several resolutions. The size is read
/// back off the loaded model rather than assumed, so a mismatched file is caught
/// at load instead of producing a garbled matte.
pub const INFER_SIZES: [usize; 3] = [512, 384, 256];

/// Used before a model is loaded, to size the initial graphics resources.
pub const DEFAULT_INFER_SIZE: usize = 512;

/// A frame handed to the worker: raw BGRA from the staged surface, plus the hint
/// settings live at the time it was captured.
struct Job {
    bgra: Vec<u8>,
    stride: usize,
    params: HintParams,
}

/// Model output, packed as one RGBA8 image so the render thread does a single
/// texture upload: RGB = straight (un-premultiplied) foreground in sRGB,
/// A = linear alpha.
pub struct KeyResult {
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Loading,
    Ready,
    Failed,
}

pub struct Engine {
    tx: SyncSender<Job>,
    latest: Arc<Mutex<Option<Arc<KeyResult>>>>,
    status: Arc<AtomicU32>,
    error: Arc<Mutex<Option<String>>>,
    /// Rolling inference time in microseconds, for the properties readout.
    last_us: Arc<AtomicU64>,
    /// Breakdown of the most recent inference, for the properties readout.
    last_timings: Arc<Mutex<Option<Timings>>>,
    /// Resolution of the loaded graph, published once the worker is ready. Zero
    /// until then, so the render thread knows not to size anything off it yet.
    infer_size: Arc<AtomicU32>,
    /// True between handing a frame over and the worker finishing with it. The
    /// render thread uses this to skip the whole capture/downscale/readback pass
    /// on frames the worker could not accept anyway.
    busy: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

const ST_LOADING: u32 = 0;
const ST_READY: u32 = 1;
const ST_FAILED: u32 = 2;

impl Engine {
    /// Spawns the worker. Returns immediately — loading a 275MB ONNX graph and
    /// building an execution provider takes seconds, and must not happen on the
    /// thread OBS calls `create` from.
    pub fn spawn(model_path: &Path, device_id: i32) -> Engine {
        // Depth 1 with try_send: if the worker is busy we discard the new frame
        // rather than queue it. A backlog would only add latency to a result
        // that is about to be superseded anyway.
        let (tx, rx) = sync_channel::<Job>(1);

        let latest = Arc::new(Mutex::new(None));
        let status = Arc::new(AtomicU32::new(ST_LOADING));
        let error = Arc::new(Mutex::new(None));
        let last_us = Arc::new(AtomicU64::new(0));
        let last_timings = Arc::new(Mutex::new(None));
        let infer_size = Arc::new(AtomicU32::new(0));
        let busy = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker = {
            let path = model_path.to_path_buf();
            let latest = Arc::clone(&latest);
            let status = Arc::clone(&status);
            let error = Arc::clone(&error);
            let last_us = Arc::clone(&last_us);
            let last_timings = Arc::clone(&last_timings);
            let infer_size = Arc::clone(&infer_size);
            let busy = Arc::clone(&busy);
            let shutdown = Arc::clone(&shutdown);
            std::thread::Builder::new()
                .name("corridorkey-infer".into())
                .spawn(move || {
                    worker_main(
                        &path, device_id, rx, latest, status, error, last_us, last_timings,
                        infer_size, busy,
                        shutdown,
                    );
                })
                .expect("failed to spawn CorridorKey inference thread")
        };

        Engine {
            tx,
            latest,
            status,
            error,
            last_us,
            last_timings,
            infer_size,
            busy,
            shutdown,
            worker: Some(worker),
        }
    }

    /// Hands a frame to the worker. Returns false if it was dropped because the
    /// worker is still busy, which is the normal steady state.
    pub fn submit(&self, bgra: Vec<u8>, stride: usize, params: HintParams) -> bool {
        match self.tx.try_send(Job { bgra, stride, params }) {
            Ok(()) => {
                self.busy.store(true, Ordering::Release);
                true
            }
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Whether the worker has capacity for another frame. Checked before doing
    /// any capture work, so frames the worker would drop cost nothing at all.
    pub fn accepting(&self) -> bool {
        !self.busy.load(Ordering::Acquire)
    }

    /// Most recent completed result, if any. Cheap to call every frame; the same
    /// result is returned repeatedly until a newer one lands.
    pub fn latest(&self) -> Option<Arc<KeyResult>> {
        self.latest.lock().clone()
    }

    pub fn status(&self) -> Status {
        match self.status.load(Ordering::Acquire) {
            ST_READY => Status::Ready,
            ST_FAILED => Status::Failed,
            _ => Status::Loading,
        }
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().clone()
    }

    /// Resolution of the loaded graph, or None until the worker is ready.
    pub fn infer_size(&self) -> Option<usize> {
        match self.infer_size.load(Ordering::Acquire) {
            0 => None,
            n => Some(n as usize),
        }
    }

    /// Breakdown of the most recent inference, if one has completed.
    pub fn last_timings(&self) -> Option<Timings> {
        *self.last_timings.lock()
    }

    pub fn last_inference_ms(&self) -> f32 {
        self.last_us.load(Ordering::Relaxed) as f32 / 1000.0
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Dropping the sender wakes the worker out of recv().
        let (dead_tx, _) = sync_channel::<Job>(0);
        let _ = std::mem::replace(&mut self.tx, dead_tx);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

fn worker_main(
    model_path: &Path,
    device_id: i32,
    rx: Receiver<Job>,
    latest: Arc<Mutex<Option<Arc<KeyResult>>>>,
    status: Arc<AtomicU32>,
    error: Arc<Mutex<Option<String>>>,
    last_us: Arc<AtomicU64>,
    last_timings: Arc<Mutex<Option<Timings>>>,
    infer_size: Arc<AtomicU32>,
    busy: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
) {
    let (mut session, size) = match build_session(model_path, device_id) {
        Ok(v) => v,
        Err(e) => {
            log::error(&format!("failed to load model: {e}"));
            *error.lock() = Some(e);
            status.store(ST_FAILED, Ordering::Release);
            // Keep draining so submit() doesn't wedge on a full channel.
            while rx.recv().is_ok() {
                busy.store(false, Ordering::Release);
                if shutdown.load(Ordering::Acquire) {
                    return;
                }
            }
            return;
        }
    };
    infer_size.store(size as u32, Ordering::Release);
    status.store(ST_READY, Ordering::Release);
    log::info("inference worker ready");

    // Reused across frames so a steady state does no allocation.
    let mut scratch = Scratch::new(size);

    while let Ok(job) = rx.recv() {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let started = std::time::Instant::now();

        match key_bgra(&mut session, &job.bgra, job.stride, &job.params, &mut scratch) {
            Ok((result, t)) => {
                last_us.store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                last_timings.lock().replace(t);
                *latest.lock() = Some(Arc::new(result));
            }
            Err(e) => {
                // One bad frame shouldn't kill the filter; log it and keep the
                // previous result on screen.
                log::error(&format!("inference failed: {e}"));
            }
        }
        // Cleared whether or not the frame succeeded — a failing model must not
        // wedge the render thread into never offering another frame.
        busy.store(false, Ordering::Release);
    }
}

/// Preprocessing buffers, hoisted out so the steady state allocates nothing.
pub struct Scratch {
    size: usize,
    rgb: Vec<f32>,
    hint: Vec<f32>,
}

impl Scratch {
    pub fn new(size: usize) -> Self {
        let px = size * size;
        Self { size, rgb: vec![0.0; px * 3], hint: vec![0.0; px] }
    }
}

/// Hint + inference for one staged BGRA frame. Public so `examples/key_image.rs`
/// can exercise the identical path offline, without OBS.
/// Where the time went in one call, in microseconds.
///
/// Worth keeping rather than guessing: the split between CPU-side conversion and
/// the network itself decides whether it is worth moving tensors onto the GPU.
#[derive(Clone, Copy, Default, Debug)]
pub struct Timings {
    /// BGRA -> planar float, plus the chroma hint.
    pub preprocess_us: u64,
    /// Tensor construction and `Session::run`, which includes ORT's own host
    /// copies in and out.
    pub infer_us: u64,
    /// Planar float outputs -> interleaved RGBA8 for upload.
    pub pack_us: u64,
}

impl Timings {
    pub fn total_us(&self) -> u64 {
        self.preprocess_us + self.infer_us + self.pack_us
    }
}

pub fn key_bgra(
    session: &mut ort::session::Session,
    bgra: &[u8],
    stride: usize,
    params: &HintParams,
    scratch: &mut Scratch,
) -> Result<(KeyResult, Timings), String> {
    let size = scratch.size;
    let mut t = Timings::default();

    let start = std::time::Instant::now();
    bgra_to_chw(bgra, size, size, stride, &mut scratch.rgb);
    chroma_hint(bgra, size, size, stride, params, &mut scratch.hint);
    t.preprocess_us = start.elapsed().as_micros() as u64;

    let result = run(session, size, &scratch.rgb, &scratch.hint, &mut t)?;
    Ok((result, t))
}

/// `device_id` is a DXGI adapter index for DirectML (or a CUDA device ordinal);
/// negative means "let ONNX Runtime choose", which is the default and normally
/// right.
///
/// It matters on laptops. Measured here on a machine with an RTX 4050 and an
/// Intel iGPU, this network runs in 165ms on adapter 1 and 1251ms on adapter 0 —
/// so an explicit wrong index is far worse than no index at all, which is why
/// this is opt-in rather than defaulting to 0.
/// Returns the session and the resolution its graph expects.
pub fn build_session(
    model_path: &Path,
    device_id: i32,
) -> Result<(ort::session::Session, usize), String> {
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;

    if !model_path.is_file() {
        return Err(format!(
            "model not found at {}. Run tools/export/export_onnx.py to produce it.",
            model_path.display()
        ));
    }

    let mut builder = Session::builder().map_err(|e| e.to_string())?;

    // Execution providers are tried in order and ort silently falls through to
    // the next when one isn't available, so a CPU-only machine still works.
    #[cfg(feature = "cuda")]
    {
        builder = builder
                        .with_execution_providers([if device_id >= 0 {
                ort::ep::CUDA::default().with_device_id(device_id).build()
            } else {
                ort::ep::CUDA::default().build()
            }])
            .map_err(|e| e.to_string())?;
    }
    #[cfg(feature = "directml")]
    {
        // DirectML manages its own allocations and does not support ORT's memory
        // pattern planner or the parallel executor. Leaving either on makes
        // session init fail outright with a bare E_INVALIDARG out of
        // AbiCustomRegistry, which is not a hint anyone would enjoy chasing.
        builder = builder
            .with_memory_pattern(false)
            .map_err(|e| e.to_string())?
            .with_parallel_execution(false)
            .map_err(|e| e.to_string())?
                        .with_execution_providers([if device_id >= 0 {
                ort::ep::DirectML::default().with_device_id(device_id).build()
            } else {
                ort::ep::DirectML::default().build()
            }])
            .map_err(|e| e.to_string())?;
    }

    let session = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_intra_threads(2)
        .map_err(|e| e.to_string())?
        .commit_from_file(model_path)
        .map_err(|e| e.to_string())?;

    let size = model_input_size(&session)?;

    let dev = if device_id >= 0 { device_id.to_string() } else { "auto".to_string() };
    log::info(&format!(
        "loaded model {} ({size}x{size}, device {dev})",
        model_path.display()
    ));
    Ok((session, size))
}

/// Reads the square input resolution off the loaded graph.
///
/// Taken from the model rather than assumed, so the staging surfaces and the
/// preprocessing buffers are always sized to what the network actually wants.
/// A dynamic or non-square input is rejected here rather than becoming a garbled
/// matte later.
fn model_input_size(session: &ort::session::Session) -> Result<usize, String> {
    for input in session.inputs() {
        let Some(dims) = input.dtype().tensor_shape() else { continue };
        if dims.len() != 4 {
            continue;
        }
        let (h, w) = (dims[2], dims[3]);
        if h < 1 || w < 1 {
            return Err(format!(
                "model input '{}' has a dynamic size ({w}x{h}); a fixed square export is required",
                input.name()
            ));
        }
        if h != w {
            return Err(format!(
                "model input '{}' is {w}x{h}; only square inputs are supported",
                input.name()
            ));
        }
        return Ok(w as usize);
    }
    Err("model has no 4-D input to take a resolution from".to_string())
}

fn run(
    session: &mut ort::session::Session,
    size: usize,
    rgb: &[f32],
    hint: &[f32],
    t: &mut Timings,
) -> Result<KeyResult, String> {
    let pixels = size * size;
    use ort::value::Tensor;
    let infer_start = std::time::Instant::now();

    let rgb_t = Tensor::from_array(([1usize, 3, size, size], rgb.to_vec()))
        .map_err(|e| e.to_string())?;
    let hint_t = Tensor::from_array(([1usize, 1, size, size], hint.to_vec()))
        .map_err(|e| e.to_string())?;

    let outputs = session
        .run(ort::inputs!["rgb" => rgb_t, "hint" => hint_t])
        .map_err(|e| e.to_string())?;

    let (_, alpha) = outputs["alpha"]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let (_, fg) = outputs["fg"]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;

    if alpha.len() < pixels || fg.len() < pixels * 3 {
        return Err(format!(
            "model returned {} alpha and {} fg values, expected {} and {}",
            alpha.len(),
            fg.len(),
            pixels,
            pixels * 3
        ));
    }

    t.infer_us = infer_start.elapsed().as_micros() as u64;

    // Pack planar float -> interleaved RGBA8 for a single texture upload.
    let pack_start = std::time::Instant::now();
    let mut rgba = vec![0u8; pixels * 4];
    for i in 0..pixels {
        rgba[i * 4] = to_u8(fg[i]);
        rgba[i * 4 + 1] = to_u8(fg[pixels + i]);
        rgba[i * 4 + 2] = to_u8(fg[2 * pixels + i]);
        rgba[i * 4 + 3] = to_u8(alpha[i]);
    }
    t.pack_us = pack_start.elapsed().as_micros() as u64;
    Ok(KeyResult { rgba })
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
