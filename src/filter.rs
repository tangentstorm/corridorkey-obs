//! The `corridorkey_keyer` OBS video filter.
//!
//! Per frame, on OBS's render thread:
//!
//!   1. Render the filter's target into a full-resolution texrender.
//!   2. Box-downscale that to 512x512 and stage it for CPU readback.
//!   3. Map *last* frame's staged surface (never this frame's — mapping a
//!      surface you just wrote stalls the GPU) and hand the pixels to the
//!      inference worker, which drops them if it's still busy.
//!   4. Upload whatever matte the worker last finished into a texture.
//!   5. Composite at full resolution in the shader.
//!
//! Only step 5 scales with output resolution, and it's a handful of taps.

use std::ffi::{c_char, c_float, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;

use crate::engine::{Engine, Status, DEFAULT_INFER_SIZE, INFER_SIZES};
use crate::hint::{HintParams, ScreenColor};
use crate::obs::*;
use crate::{log, module_file};

pub const FILTER_ID: &CStr = c"corridorkey_keyer";
const FILTER_NAME: &CStr = c"CorridorKey (Neural Green Screen)";

// Setting keys.
const S_SCREEN: &CStr = c"screen_color";
const S_SIMILARITY: &CStr = c"similarity";
const S_SMOOTHNESS: &CStr = c"smoothness";
const S_DESPILL: &CStr = c"despill";
const S_FG_DETAIL: &CStr = c"fg_detail";
const S_MATTE_BLACK: &CStr = c"matte_black";
const S_MATTE_WHITE: &CStr = c"matte_white";
const S_BYPASS: &CStr = c"bypass";
const S_DEVICE: &CStr = c"gpu_device";
const S_QUALITY: &CStr = c"quality";

#[derive(Clone, Copy)]
struct Settings {
    screen: ScreenColor,
    similarity: f32,
    smoothness: f32,
    despill: f32,
    fg_detail: f32,
    matte_black: f32,
    matte_white: f32,
    bypass: bool,
    device_id: i32,
    /// Inference resolution. Larger is a better matte; smaller is less lag,
    /// which on a slower GPU is the difference between clean edges and visible
    /// outlines trailing a moving arm.
    infer_size: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            screen: ScreenColor::Green,
            similarity: 0.30,
            smoothness: 0.12,
            despill: 1.0,
            fg_detail: 1.0,
            matte_black: 0.0,
            matte_white: 1.0,
            bypass: false,
            device_id: -1, // -1 = let ONNX Runtime pick the adapter
            infer_size: DEFAULT_INFER_SIZE,
        }
    }
}

impl Settings {
    fn hint(&self) -> HintParams {
        HintParams {
            screen: self.screen,
            similarity: self.similarity,
            smoothness: self.smoothness,
        }
    }
}

/// Cached effect parameter handles. Looking these up by name every frame is
/// wasteful and they're stable for the life of the effect.
struct EffectParams {
    // "image" is deliberately absent: OBS binds the filter's input texture to it
    // in obs_source_process_filter_*_end, for both the analysis and visible passes.
    key_tex: *mut gs_eparam_t,
    tap_step: *mut gs_eparam_t,
    despill_strength: *mut gs_eparam_t,
    screen_channel: *mut gs_eparam_t,
    fg_detail: *mut gs_eparam_t,
    matte_black: *mut gs_eparam_t,
    matte_white: *mut gs_eparam_t,
    bypass: *mut gs_eparam_t,
}

impl EffectParams {
    unsafe fn lookup(effect: *mut gs_effect_t) -> Self {
        let p = |n: &CStr| gs_effect_get_param_by_name(effect, n.as_ptr());
        Self {
            key_tex: p(c"key_tex"),
            tap_step: p(c"tap_step"),
            despill_strength: p(c"despill_strength"),
            screen_channel: p(c"screen_channel"),
            fg_detail: p(c"fg_detail"),
            matte_black: p(c"matte_black"),
            matte_white: p(c"matte_white"),
            bypass: p(c"bypass"),
        }
    }
}

pub struct Filter {
    source: *mut obs_source_t,
    settings: Settings,

    engine: Option<Engine>,
    /// What the live engine was built for, so `update` knows when a settings
    /// change (screen color, GPU) requires reloading the session.
    engine_screen: ScreenColor,
    engine_device: i32,
    engine_size: usize,

    effect: *mut gs_effect_t,
    params: Option<EffectParams>,

    small: *mut gs_texrender_t,
    /// Staged on one frame, mapped on the next. Mapping a surface in the same
    /// frame it was staged forces a CPU/GPU sync.
    stage: *mut gs_stagesurf_t,
    stage_pending: bool,
    /// Resolution `stage` and `key_tex` were created at.
    buffer_size: usize,

    key_tex: *mut gs_texture_t,
    /// Scratch buffer for the mapped surface, reused every frame.
    readback: Vec<u8>,
    /// Debug: set CORRIDORKEY_DUMP to a path to write the first frame the model
    /// actually receives, as raw BGRA. Invaluable when the matte looks wrong and
    /// the question is whether the fault is the capture or the network.
    dumped: bool,
}

impl Filter {
    /// Prefers the half-precision graph when one is installed, falling back to
    /// the float32 export. Measured on an RTX 4050 via DirectML the gain is
    /// modest — roughly 200ms to 150ms — but it is free, and it halves both the
    /// file and the VRAM footprint.
    fn model_path(screen: ScreenColor, size: usize) -> Option<PathBuf> {
        let name = |suffix: &str| {
            format!("models/corridorkey_{}_{}{}.onnx", screen.as_str(), size, suffix)
        };
        module_file(&name("_fp16")).or_else(|| module_file(&name("")))
    }

    fn start_engine(&mut self, screen: ScreenColor, device_id: i32, size: usize) {
        self.engine = None; // drop the old worker (and its model) before loading another
        match Self::model_path(screen, size) {
            Some(path) => {
                log::info(&format!(
                    "starting {} engine at {size}x{size} from {}",
                    screen.as_str(),
                    path.display()
                ));
                self.engine = Some(Engine::spawn(&path, device_id));
                self.engine_screen = screen;
                self.engine_device = device_id;
                self.engine_size = size;
                // Safe to size the graphics resources from the setting rather than
                // waiting for the worker: build_session rejects a model whose input
                // doesn't match, so a mismatch never reaches the render path.
                unsafe { self.resize_buffers(size) };
            }
            None => {
                self.engine_size = 0;
                log::error(&format!(
                    "no {}-screen model at {size}x{size}. Expected \
                     <obs-plugins>/corridorkey-obs/models/corridorkey_{}_{size}_fp16.onnx — \
                     generate it with: export_onnx.py --size {size} --color {} --fp16",
                    screen.as_str(),
                    screen.as_str(),
                    screen.as_str()
                ));
            }
        }
    }

    /// Recreates the staging surface and matte texture at a new resolution.
    ///
    /// Both are sized to the model's input, so switching quality means rebuilding
    /// them. Cheap and rare — it only happens when the user changes a setting.
    unsafe fn resize_buffers(&mut self, size: usize) {
        if self.buffer_size == size {
            return;
        }
        obs_enter_graphics();
        if !self.stage.is_null() {
            gs_stagesurface_destroy(self.stage);
        }
        if !self.key_tex.is_null() {
            gs_texture_destroy(self.key_tex);
        }
        self.stage = gs_stagesurface_create(size as u32, size as u32, GS_BGRA);
        self.key_tex =
            gs_texture_create(size as u32, size as u32, GS_RGBA, 1, ptr::null(), GS_DYNAMIC);
        obs_leave_graphics();

        // Anything staged against the old surface is gone.
        self.stage_pending = false;
        self.readback.clear();
        self.buffer_size = size;
    }

    /// Everything that must happen inside an OBS graphics context.
    unsafe fn init_graphics(&mut self) {
        obs_enter_graphics();

        self.small = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
        // `stage` and `key_tex` are created by resize_buffers once the resolution
        // is known, which happens when apply_settings starts the engine.

        if let Some(path) = module_file("effects/corridorkey.effect") {
            if let Ok(c) = CString::new(path.to_string_lossy().as_bytes()) {
                let mut err: *mut c_char = ptr::null_mut();
                self.effect = gs_effect_create_from_file(c.as_ptr(), &mut err);
                if self.effect.is_null() {
                    let msg = if err.is_null() {
                        "unknown error".to_string()
                    } else {
                        CStr::from_ptr(err).to_string_lossy().into_owned()
                    };
                    log::error(&format!("failed to compile corridorkey.effect: {msg}"));
                }
                if !err.is_null() {
                    bfree(err.cast());
                }
            }
        } else {
            log::error("effects/corridorkey.effect not found in the plugin data directory");
        }

        if !self.effect.is_null() {
            self.params = Some(EffectParams::lookup(self.effect));
        }

        obs_leave_graphics();
    }

    unsafe fn free_graphics(&mut self) {
        obs_enter_graphics();
        if !self.small.is_null() {
            gs_texrender_destroy(self.small);
        }
        if !self.stage.is_null() {
            gs_stagesurface_destroy(self.stage);
        }
        if !self.key_tex.is_null() {
            gs_texture_destroy(self.key_tex);
        }
        if !self.effect.is_null() {
            gs_effect_destroy(self.effect);
        }
        obs_leave_graphics();
    }

    fn apply_settings(&mut self, data: *mut obs_data_t) {
        unsafe {
            let screen = {
                let s = obs_data_get_string(data, S_SCREEN.as_ptr());
                if s.is_null() {
                    ScreenColor::Green
                } else {
                    ScreenColor::from_str(&CStr::from_ptr(s).to_string_lossy())
                }
            };
            self.settings = Settings {
                screen,
                similarity: obs_data_get_double(data, S_SIMILARITY.as_ptr()) as f32,
                smoothness: obs_data_get_double(data, S_SMOOTHNESS.as_ptr()) as f32,
                despill: obs_data_get_double(data, S_DESPILL.as_ptr()) as f32,
                fg_detail: obs_data_get_double(data, S_FG_DETAIL.as_ptr()) as f32,
                matte_black: obs_data_get_double(data, S_MATTE_BLACK.as_ptr()) as f32,
                matte_white: obs_data_get_double(data, S_MATTE_WHITE.as_ptr()) as f32,
                bypass: obs_data_get_bool(data, S_BYPASS.as_ptr()),
                device_id: obs_data_get_int(data, S_DEVICE.as_ptr()) as i32,
                infer_size: {
                    // Guard against a settings file naming a size we no longer ship.
                    let want = obs_data_get_int(data, S_QUALITY.as_ptr()) as usize;
                    if INFER_SIZES.contains(&want) { want } else { DEFAULT_INFER_SIZE }
                },
            };
        }

        // Green and blue are different checkpoints, and the device is baked into
        // the session, so either change means building a new one.
        if self.engine.is_none()
            || self.engine_screen != self.settings.screen
            || self.engine_device != self.settings.device_id
            || self.engine_size != self.settings.infer_size
        {
            let s = self.settings;
            self.start_engine(s.screen, s.device_id, s.infer_size);
        }
    }

    /// Render the filter's input straight into the 512x512 texrender and stage it
    /// for CPU readback.
    ///
    /// The input comes from `obs_source_process_filter_begin`, not from calling
    /// `obs_source_video_render` on the filter target. Re-rendering the target by
    /// hand from inside a filter's `video_render` produces an empty texture for
    /// some source types — an image source yields a frame of pure zeros, which
    /// the chroma hint then reads as "all foreground" and the network dutifully
    /// returns a fully opaque matte. Letting OBS supply its own input texture is
    /// both correct and cheaper: the downscale runs as the filter's output pass,
    /// so there is no separate full-resolution capture at all.
    unsafe fn run_analysis_pass(&mut self, width: u32, height: u32) {
        let Some(params) = self.params.as_ref() else { return };

        if !obs_source_process_filter_begin(self.source, GS_BGRA, OBS_NO_DIRECT_RENDERING) {
            return;
        }

        // Spread the 3x3 taps over the source footprint of one destination texel.
        // The plate is squashed to a square rather than letterboxed: upstream
        // CorridorKey resizes to img_size x img_size the same way, and the shader
        // samples the matte with the same normalized coordinates, so the aspect
        // ratio comes back out correctly.
        let tap = Vec2 {
            x: 1.0 / (self.buffer_size as f32 * 3.0),
            y: 1.0 / (self.buffer_size as f32 * 3.0),
        };
        gs_effect_set_vec2(params.tap_step, &tap);

        gs_texrender_reset(self.small);
        if !gs_texrender_begin(self.small, self.buffer_size as u32, self.buffer_size as u32) {
            // Every begin needs its end, or OBS's filter state is left dangling.
            obs_source_process_filter_tech_end(
                self.source,
                self.effect,
                width,
                height,
                c"Downscale".as_ptr(),
            );
            return;
        }

        let clear = Vec4::default();
        gs_clear(GS_CLEAR_COLOR, &clear, 0.0, 0);
        gs_ortho(0.0, self.buffer_size as f32, 0.0, self.buffer_size as f32, -100.0, 100.0);
        gs_blend_state_push();
        gs_enable_blending(false);

        // Draws OBS's input texture through our Downscale technique, straight into
        // the 512x512 render target.
        obs_source_process_filter_tech_end(
            self.source,
            self.effect,
            self.buffer_size as u32,
            self.buffer_size as u32,
            c"Downscale".as_ptr(),
        );

        gs_blend_state_pop();
        gs_texrender_end(self.small);

        let small_tex = gs_texrender_get_texture(self.small);
        if !small_tex.is_null() {
            gs_stage_texture(self.stage, small_tex);
            self.stage_pending = true;
        }
    }

    /// Map the surface staged on an earlier frame and hand the pixels to the
    /// worker. Never maps a surface staged this same frame — that forces a
    /// CPU/GPU sync and stalls the render thread.
    unsafe fn drain_staged_frame(&mut self) {
        self.stage_pending = false;

        let mut data: *mut u8 = ptr::null_mut();
        let mut linesize: u32 = 0;
        if !gs_stagesurface_map(self.stage, &mut data, &mut linesize) {
            return;
        }
        // Cleared unconditionally: otherwise a frame whose map yields an unusable
        // surface would resubmit the previous frame's pixels with this frame's stride.
        self.readback.clear();
        if !data.is_null() && linesize as usize >= self.buffer_size * 4 {
            let len = linesize as usize * self.buffer_size;
            self.readback
                .extend_from_slice(std::slice::from_raw_parts(data, len));
        }
        gs_stagesurface_unmap(self.stage);

        if self.readback.is_empty() {
            return;
        }
        {
            if let Ok(path) = std::env::var("CORRIDORKEY_DUMP") {
                let first = !self.dumped;
                self.dumped = true;
                let mean = self.readback.iter().map(|&b| b as u64).sum::<u64>()
                    / self.readback.len().max(1) as u64;
                match std::fs::write(&path, &self.readback) {
                    Ok(()) if first => log::info(&format!(
                        "dumped {} bytes of staged BGRA (stride {}, mean byte {}) to {}",
                        self.readback.len(),
                        linesize,
                        mean,
                        path
                    )),
                    Ok(()) => {}
                    Err(e) => log::error(&format!("dump to {path} failed: {e}")),
                }
            }
        }
        if let Some(engine) = self.engine.as_ref() {
            let buf = std::mem::take(&mut self.readback);
            engine.submit(buf, linesize as usize, self.settings.hint());
        }
    }

    unsafe fn render(&mut self) {
        let source = self.source;

        let target = obs_filter_get_target(source);
        if target.is_null() || self.effect.is_null() || self.params.is_none() {
            obs_source_skip_video_filter(source);
            return;
        }
        let width = obs_source_get_base_width(target);
        let height = obs_source_get_base_height(target);
        if width == 0 || height == 0 {
            obs_source_skip_video_filter(source);
            return;
        }

        // While the model is loading (or failed to load) the filter is a no-op
        // rather than a black frame. Bypass takes the same path, so it costs
        // nothing rather than running the network and discarding the result.
        let ready = matches!(self.engine.as_ref().map(|e| e.status()), Some(Status::Ready));
        if !ready || self.settings.bypass {
            obs_source_skip_video_filter(source);
            return;
        }

        // Feed the model at the rate it can actually consume. Inference takes far
        // longer than one frame, so capturing and reading back every frame would
        // burn GPU time producing pixels that get dropped. On the frames in
        // between, this costs nothing.
        if self.stage_pending {
            self.drain_staged_frame();
        } else if self.engine.as_ref().is_some_and(|e| e.accepting()) {
            // Note this consumes a full begin/end cycle of its own; the visible
            // pass below starts a second one. Rendering the input twice on these
            // frames is far cheaper than the inference it feeds.
            self.run_analysis_pass(width, height);
        }

        // Upload the newest finished matte.
        if let Some(result) = self.engine.as_ref().and_then(|e| e.latest()) {
            gs_texture_set_image(
                self.key_tex,
                result.rgba.as_ptr(),
                (self.buffer_size * 4) as u32,
                false,
            );
        }

        // Output through OBS's own filter plumbing rather than drawing the
        // captured texture ourselves: it sets up the view and the "image"
        // parameter, and keeps this filter composing correctly with whatever else
        // is in the chain.
        if !obs_source_process_filter_begin(source, GS_BGRA, OBS_NO_DIRECT_RENDERING) {
            return;
        }

        let params = self.params.as_ref().unwrap();
        let s = self.settings;
        gs_effect_set_texture(params.key_tex, self.key_tex);
        gs_effect_set_float(params.despill_strength, s.despill as c_float);
        gs_effect_set_int(params.screen_channel, s.screen.channel() as i32);
        gs_effect_set_float(params.fg_detail, s.fg_detail as c_float);
        gs_effect_set_float(params.matte_black, s.matte_black as c_float);
        gs_effect_set_float(params.matte_white, s.matte_white as c_float);
        gs_effect_set_float(params.bypass, 0.0);

        obs_source_process_filter_end(source, self.effect, width, height);
    }
}

// --- OBS callbacks ----------------------------------------------------------
//
// Every one of these is called from C, so each catches unwinding: a panic
// crossing the FFI boundary is undefined behaviour, and taking down OBS
// mid-stream over a bug in a filter is not acceptable.

macro_rules! guard {
    ($default:expr, $body:block) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => {
                crate::log::error("panic in filter callback (recovered)");
                $default
            }
        }
    };
}

unsafe extern "C" fn get_name(_type_data: *mut c_void) -> *const c_char {
    FILTER_NAME.as_ptr()
}

unsafe extern "C" fn create(settings: *mut obs_data_t, source: *mut obs_source_t) -> *mut c_void {
    guard!(ptr::null_mut(), {
        let mut filter = Box::new(Filter {
            source,
            settings: Settings::default(),
            engine: None,
            engine_screen: ScreenColor::Green,
            engine_device: -1,
            engine_size: 0,
            effect: ptr::null_mut(),
            params: None,
            small: ptr::null_mut(),
            stage: ptr::null_mut(),
            stage_pending: false,
            buffer_size: 0,
            key_tex: ptr::null_mut(),
            readback: Vec::new(),
            dumped: false,
        });
        filter.init_graphics();
        filter.apply_settings(settings);
        Box::into_raw(filter).cast()
    })
}

unsafe extern "C" fn destroy(data: *mut c_void) {
    guard!((), {
        if data.is_null() {
            return;
        }
        let mut filter: Box<Filter> = Box::from_raw(data.cast());
        // Stop the worker before tearing down graphics resources.
        filter.engine = None;
        filter.free_graphics();
    })
}

unsafe extern "C" fn update(data: *mut c_void, settings: *mut obs_data_t) {
    guard!((), {
        if data.is_null() {
            return;
        }
        let filter = &mut *(data as *mut Filter);
        filter.apply_settings(settings);
    })
}

unsafe extern "C" fn video_render(data: *mut c_void, _effect: *mut gs_effect_t) {
    guard!((), {
        if data.is_null() {
            return;
        }
        let filter = &mut *(data as *mut Filter);
        filter.render();
    })
}

unsafe extern "C" fn get_defaults(settings: *mut obs_data_t) {
    guard!((), {
        let d = Settings::default();
        obs_data_set_default_string(settings, S_SCREEN.as_ptr(), c"green".as_ptr());
        obs_data_set_default_double(settings, S_SIMILARITY.as_ptr(), d.similarity as f64);
        obs_data_set_default_double(settings, S_SMOOTHNESS.as_ptr(), d.smoothness as f64);
        obs_data_set_default_double(settings, S_DESPILL.as_ptr(), d.despill as f64);
        obs_data_set_default_double(settings, S_FG_DETAIL.as_ptr(), d.fg_detail as f64);
        obs_data_set_default_double(settings, S_MATTE_BLACK.as_ptr(), d.matte_black as f64);
        obs_data_set_default_double(settings, S_MATTE_WHITE.as_ptr(), d.matte_white as f64);
        obs_data_set_default_bool(settings, S_BYPASS.as_ptr(), d.bypass);
        obs_data_set_default_int(settings, S_DEVICE.as_ptr(), d.device_id as i64);
        obs_data_set_default_int(settings, S_QUALITY.as_ptr(), d.infer_size as i64);
    })
}

unsafe extern "C" fn get_properties(data: *mut c_void) -> *mut obs_properties_t {
    guard!(ptr::null_mut(), {
        let props = obs_properties_create();

        let screen = obs_properties_add_list(
            props,
            S_SCREEN.as_ptr(),
            c"Screen color".as_ptr(),
            OBS_COMBO_TYPE_LIST,
            OBS_COMBO_FORMAT_STRING,
        );
        obs_property_list_add_string(screen, c"Green".as_ptr(), c"green".as_ptr());
        obs_property_list_add_string(screen, c"Blue".as_ptr(), c"blue".as_ptr());
        obs_property_set_long_description(
            screen,
            c"Selects the CorridorKey checkpoint and the despill channel. Switching reloads the model.".as_ptr(),
        );

        let quality = obs_properties_add_list(
            props,
            S_QUALITY.as_ptr(),
            c"Quality / speed".as_ptr(),
            OBS_COMBO_TYPE_LIST,
            OBS_COMBO_FORMAT_INT,
        );
        obs_property_list_add_int(quality, c"Best (512px)".as_ptr(), 512);
        obs_property_list_add_int(quality, c"Balanced (384px)".as_ptr(), 384);
        obs_property_list_add_int(quality, c"Fastest (256px)".as_ptr(), 256);
        obs_property_set_long_description(
            quality,
            c"Resolution the network runs at. Lower is faster, which cuts how far the matte               lags the picture - that lag is what leaves an outline trailing a moving arm.               Raise it if edges look soft, lower it if movement ghosts. Changing this loads               a different model."
                .as_ptr(),
        );

        let sim = obs_properties_add_float_slider(
            props,
            S_SIMILARITY.as_ptr(),
            c"Hint: similarity".as_ptr(),
            0.0,
            1.0,
            0.01,
        );
        obs_property_set_long_description(
            sim,
            c"How close a pixel's chroma must be to the screen color to seed the hint as background. \
              Raise it if parts of your subject are being cut away; lower it if the screen isn't being detected."
                .as_ptr(),
        );

        obs_properties_add_float_slider(
            props,
            S_SMOOTHNESS.as_ptr(),
            c"Hint: smoothness".as_ptr(),
            0.0,
            0.5,
            0.005,
        );

        let despill = obs_properties_add_float_slider(
            props,
            S_DESPILL.as_ptr(),
            c"Despill strength".as_ptr(),
            0.0,
            1.0,
            0.01,
        );
        obs_property_set_long_description(
            despill,
            c"Removes screen color bounced onto the subject, preserving luminance.".as_ptr(),
        );

        let detail = obs_properties_add_float_slider(
            props,
            S_FG_DETAIL.as_ptr(),
            c"Full-res detail in solid areas".as_ptr(),
            0.0,
            1.0,
            0.01,
        );
        obs_property_set_long_description(
            detail,
            c"The model runs at 512x512. At 1.0 the original full-resolution image is used wherever the \
              matte is solid, keeping the model's unmixed color only for soft edges. Lower it to see the \
              raw model output."
                .as_ptr(),
        );

        obs_properties_add_float_slider(
            props,
            S_MATTE_BLACK.as_ptr(),
            c"Matte black point".as_ptr(),
            0.0,
            1.0,
            0.005,
        );
        obs_properties_add_float_slider(
            props,
            S_MATTE_WHITE.as_ptr(),
            c"Matte white point".as_ptr(),
            0.0,
            1.0,
            0.005,
        );

        obs_properties_add_bool(props, S_BYPASS.as_ptr(), c"Bypass (show original)".as_ptr());

        let device = obs_properties_add_int(props, S_DEVICE.as_ptr(), c"GPU device index (-1 = auto)".as_ptr(), -1, 7, 1);
        obs_property_set_long_description(
            device,
            c"Which graphics adapter runs the network. Leave at -1 unless inference looks               far slower than it should: on laptops adapter 0 is often the integrated GPU,               and this network is roughly 8x slower there than on the discrete one.               Changing this reloads the model."
                .as_ptr(),
        );

        // Live status line: which backend loaded, how fast it's running, or why
        // it isn't. Without this a failed model load is a silent no-op filter.
        if !data.is_null() {
            let filter = &*(data as *const Filter);
            let text = match filter.engine.as_ref().map(|e| (e.status(), e)) {
                Some((Status::Ready, e)) => {
                    let ms = e.last_inference_ms();
                    let size = e.infer_size().unwrap_or(0);
                    format!(
                        "Running at {size}x{size}. Inference: {ms:.0} ms/frame                          (matte lags the picture by about that much)"
                    )
                }
                Some((Status::Loading, _)) => "Loading model...".to_string(),
                Some((Status::Failed, e)) => format!(
                    "Model failed to load: {}",
                    e.error().unwrap_or_else(|| "unknown error".into())
                ),
                None => "No model found. See the plugin README.".to_string(),
            };
            if let Ok(c) = CString::new(text) {
                obs_properties_add_text(props, c"status".as_ptr(), c.as_ptr(), OBS_TEXT_INFO);
            }
        }

        props
    })
}

pub fn info() -> ObsSourceInfo {
    let mut info = ObsSourceInfo::zeroed();
    info.id = FILTER_ID.as_ptr();
    info.type_ = OBS_SOURCE_TYPE_FILTER;
    // CUSTOM_DRAW because we render the target into our own texrender in order
    // to read it back; OBS must not set up its default pass for us.
    info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
    info.get_name = Some(get_name);
    info.create = Some(create);
    info.destroy = Some(destroy);
    info.update = Some(update);
    info.video_render = Some(video_render);
    info.get_defaults = Some(get_defaults);
    info.get_properties = Some(get_properties);
    info
}
