//! CorridorKey for OBS Studio — realtime neural green screen keying.
//!
//! Upstream CorridorKey (https://github.com/nikopueringer/CorridorKey) is an
//! offline VFX tool: PyTorch, 2048x2048, EXR output, an alpha hint produced by a
//! separate heavyweight model. This plugin keeps its keyer — the GreenFormer
//! network that unmixes foreground color from screen color — and rebuilds
//! everything around it for live use:
//!
//!   * the network is exported to ONNX at 512x512 and run via ONNX Runtime,
//!   * the alpha hint comes from a cheap chroma key instead of GVM/VideoMaMa,
//!   * inference runs off the render thread, and compositing happens at full
//!     resolution in a shader.
//!
//! See README.md for the quality and licensing implications of that trade.

pub mod composite;
pub mod engine;
mod filter;
pub mod hint;
mod obs;

use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use obs::*;

/// Set by OBS immediately after loading the DLL, via `obs_module_set_pointer`.
/// Needed to resolve files in the plugin's data directory.
static MODULE: AtomicPtr<obs_module_t> = AtomicPtr::new(ptr::null_mut());

/// Resolves a path inside the plugin's `data/` directory, e.g.
/// `"effects/corridorkey.effect"`. Returns None if OBS can't find it.
pub fn module_file(relative: &str) -> Option<PathBuf> {
    let module = MODULE.load(Ordering::Acquire);
    if module.is_null() {
        return None;
    }
    let rel = CString::new(relative).ok()?;
    unsafe {
        let raw = obs_find_module_file(module, rel.as_ptr());
        if raw.is_null() {
            return None;
        }
        let path = PathBuf::from(CStr::from_ptr(raw).to_string_lossy().into_owned());
        bfree(raw.cast());
        Some(path)
    }
}

pub mod log {
    use super::*;

    fn emit(level: std::ffi::c_int, msg: &str) {
        // blog is variadic; always go through "%s" so a '%' in a model path or
        // an ONNX Runtime error can't be interpreted as a format specifier.
        if let Ok(c) = CString::new(format!("[corridorkey] {msg}")) {
            unsafe { blog(level, c"%s".as_ptr(), c.as_ptr()) }
        }
    }

    pub fn info(msg: &str) {
        emit(LOG_INFO, msg);
    }

    pub fn error(msg: &str) {
        emit(LOG_ERROR, msg);
    }
}

// --- OBS module entry points ------------------------------------------------
//
// Normally provided by the OBS_DECLARE_MODULE macro; spelled out here because
// this is a Rust cdylib.

#[no_mangle]
pub extern "C" fn obs_module_set_pointer(module: *mut obs_module_t) {
    MODULE.store(module, Ordering::Release);
}

#[no_mangle]
pub extern "C" fn obs_module_ver() -> u32 {
    LIBOBS_API_VER
}

#[no_mangle]
pub extern "C" fn obs_module_name() -> *const c_char {
    c"CorridorKey".as_ptr()
}

#[no_mangle]
pub extern "C" fn obs_module_description() -> *const c_char {
    c"Realtime neural green screen keying using the CorridorKey model".as_ptr()
}

#[no_mangle]
pub extern "C" fn obs_module_load() -> bool {
    let result = std::panic::catch_unwind(|| {
        let info = filter::info();
        unsafe {
            obs_register_source_s(&info, std::mem::size_of::<ObsSourceInfo>());
        }
        log::info(concat!("v", env!("CARGO_PKG_VERSION"), " loaded"));
        true
    });
    match result {
        Ok(v) => v,
        Err(_) => {
            log::error("panic during module load");
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn obs_module_unload() {}
