//! Hand-written bindings to the parts of libobs this plugin uses.
//!
//! Transcribed from the obs-studio 32.2.2 headers (`libobs/obs-source.h`,
//! `libobs/graphics/graphics.h`). Keeping this hand-written rather than
//! bindgen-generated means building the plugin needs nothing but an OBS install
//! (see `build.rs`) — no obs-studio source tree, no libclang.
//!
//! `ObsSourceInfo` must stay byte-identical to `struct obs_source_info`; its
//! `size_of` is handed to `obs_register_source_s`, and OBS uses that to decide
//! which trailing callbacks exist.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_float, c_int, c_void};

/// `LIBOBS_API_VER` for the OBS release these bindings were transcribed from.
///
/// OBS refuses to load a module whose major version differs from its own, which
/// is the behaviour we want: if OBS 33 changes `obs_source_info`, this plugin
/// should be rejected rather than corrupt memory.
pub const LIBOBS_API_VER: u32 = (32 << 24) | (2 << 16) | 2;

// --- opaque handles ---------------------------------------------------------

pub enum obs_module_t {}
pub enum obs_source_t {}
pub enum obs_data_t {}
pub enum obs_properties_t {}
pub enum obs_property_t {}
pub enum gs_texture_t {}
pub enum gs_texrender_t {}
pub enum gs_stagesurf_t {}
pub enum gs_effect_t {}
pub enum gs_eparam_t {}
pub enum gs_technique_t {}

// --- enums / flags ----------------------------------------------------------

pub const OBS_SOURCE_TYPE_FILTER: c_int = 1;

pub const OBS_SOURCE_VIDEO: u32 = 1 << 0;
pub const OBS_SOURCE_ASYNC: u32 = 1 << 2;
pub const OBS_SOURCE_CUSTOM_DRAW: u32 = 1 << 3;
pub const OBS_SOURCE_SRGB: u32 = 1 << 15;

// enum gs_color_format
pub const GS_UNKNOWN: c_int = 0;
pub const GS_A8: c_int = 1;
pub const GS_R8: c_int = 2;
pub const GS_RGBA: c_int = 3;
pub const GS_BGRX: c_int = 4;
pub const GS_BGRA: c_int = 5;
pub const GS_RGBA16F: c_int = 9;
pub const GS_RGBA32F: c_int = 10;
pub const GS_R32F: c_int = 14;

// enum gs_zstencil_format
pub const GS_ZS_NONE: c_int = 0;

pub const GS_BUILD_MIPMAPS: u32 = 1 << 0;
pub const GS_DYNAMIC: u32 = 1 << 1;
pub const GS_RENDER_TARGET: u32 = 1 << 2;

// enum obs_allow_direct_render
pub const OBS_NO_DIRECT_RENDERING: i32 = 0;
pub const OBS_ALLOW_DIRECT_RENDERING: i32 = 1;

// enum obs_text_type. INFO is 3, not 2 - 2 is MULTILINE, which renders the
// status line as a large editable text box instead of a label.
pub const OBS_TEXT_DEFAULT: c_int = 0;
pub const OBS_TEXT_PASSWORD: c_int = 1;
pub const OBS_TEXT_MULTILINE: c_int = 2;
pub const OBS_TEXT_INFO: c_int = 3;

// enum obs_combo_type. Spelled out in full because getting LIST wrong is silent:
// OBS rejects an EDITABLE list with a non-string format by returning NULL, and
// the property simply never appears in the UI.
pub const OBS_COMBO_TYPE_INVALID: c_int = 0;
pub const OBS_COMBO_TYPE_EDITABLE: c_int = 1;
pub const OBS_COMBO_TYPE_LIST: c_int = 2;
pub const OBS_COMBO_TYPE_RADIO: c_int = 3;

// enum obs_combo_format
pub const OBS_COMBO_FORMAT_INVALID: c_int = 0;
pub const OBS_COMBO_FORMAT_INT: c_int = 1;
pub const OBS_COMBO_FORMAT_FLOAT: c_int = 2;
pub const OBS_COMBO_FORMAT_STRING: c_int = 3;
pub const OBS_COMBO_FORMAT_BOOL: c_int = 4;

pub const LOG_ERROR: c_int = 100;
pub const LOG_WARNING: c_int = 200;
pub const LOG_INFO: c_int = 300;
pub const LOG_DEBUG: c_int = 400;

// --- struct obs_source_info -------------------------------------------------

pub type obs_source_enum_proc_t =
    Option<unsafe extern "C" fn(*mut obs_source_t, *mut obs_source_t, *mut c_void)>;

/// Mirrors `struct obs_source_info` from obs-source.h exactly, in declaration
/// order. Unused callbacks are typed as opaque pointers because we always leave
/// them null; only the ones we implement carry real signatures.
#[repr(C)]
pub struct ObsSourceInfo {
    pub id: *const c_char,
    pub type_: c_int,
    pub output_flags: u32,

    pub get_name: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
    pub create: Option<unsafe extern "C" fn(*mut obs_data_t, *mut obs_source_t) -> *mut c_void>,
    pub destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    pub get_width: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    pub get_height: Option<unsafe extern "C" fn(*mut c_void) -> u32>,

    pub get_defaults: Option<unsafe extern "C" fn(*mut obs_data_t)>,
    pub get_properties: Option<unsafe extern "C" fn(*mut c_void) -> *mut obs_properties_t>,
    pub update: Option<unsafe extern "C" fn(*mut c_void, *mut obs_data_t)>,
    pub activate: Option<unsafe extern "C" fn(*mut c_void)>,
    pub deactivate: Option<unsafe extern "C" fn(*mut c_void)>,
    pub show: Option<unsafe extern "C" fn(*mut c_void)>,
    pub hide: Option<unsafe extern "C" fn(*mut c_void)>,
    pub video_tick: Option<unsafe extern "C" fn(*mut c_void, c_float)>,
    pub video_render: Option<unsafe extern "C" fn(*mut c_void, *mut gs_effect_t)>,
    pub filter_video: *mut c_void,
    pub filter_audio: *mut c_void,
    pub enum_active_sources: *mut c_void,
    pub save: *mut c_void,
    pub load: *mut c_void,
    pub mouse_click: *mut c_void,
    pub mouse_move: *mut c_void,
    pub mouse_wheel: *mut c_void,
    pub focus: *mut c_void,
    pub key_click: *mut c_void,
    pub filter_remove: *mut c_void,

    pub type_data: *mut c_void,
    pub free_type_data: *mut c_void,
    pub audio_render: *mut c_void,
    pub enum_all_sources: *mut c_void,
    pub transition_start: *mut c_void,
    pub transition_stop: *mut c_void,
    pub get_defaults2: *mut c_void,
    pub get_properties2: *mut c_void,
    pub audio_mix: *mut c_void,

    pub icon_type: c_int,

    pub media_play_pause: *mut c_void,
    pub media_restart: *mut c_void,
    pub media_stop: *mut c_void,
    pub media_next: *mut c_void,
    pub media_previous: *mut c_void,
    pub media_get_duration: *mut c_void,
    pub media_get_time: *mut c_void,
    pub media_set_time: *mut c_void,
    pub media_get_state: *mut c_void,

    pub version: u32,
    pub unversioned_id: *const c_char,

    pub missing_files: *mut c_void,
    pub video_get_color_space: *mut c_void,
    pub filter_add: *mut c_void,
    pub get_dark_icon: *mut c_void,
    pub get_light_icon: *mut c_void,
}

// Safe to share: we only ever build one, as a static, containing fn pointers
// and 'static C strings.
unsafe impl Sync for ObsSourceInfo {}

impl ObsSourceInfo {
    /// All-null skeleton; set the fields you implement and leave the rest.
    pub const fn zeroed() -> Self {
        // SAFETY: every field is a pointer, an integer, or Option<fn ptr>, all
        // of which have a valid all-zero representation.
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

// --- imported functions -----------------------------------------------------

extern "C" {
    pub fn obs_register_source_s(info: *const ObsSourceInfo, size: usize);

    pub fn blog(log_level: c_int, format: *const c_char, ...);
    pub fn bfree(ptr: *mut c_void);

    pub fn obs_find_module_file(module: *mut obs_module_t, file: *const c_char) -> *mut c_char;

    pub fn obs_enter_graphics();
    pub fn obs_leave_graphics();

    // filter plumbing
    pub fn obs_filter_get_target(filter: *const obs_source_t) -> *mut obs_source_t;
    pub fn obs_filter_get_parent(filter: *const obs_source_t) -> *mut obs_source_t;
    pub fn obs_source_get_base_width(source: *mut obs_source_t) -> u32;
    pub fn obs_source_get_base_height(source: *mut obs_source_t) -> u32;
    pub fn obs_source_skip_video_filter(filter: *mut obs_source_t);
    pub fn obs_source_process_filter_begin(
        filter: *mut obs_source_t,
        format: c_int,
        allow_direct: c_int,
    ) -> bool;
    pub fn obs_source_process_filter_end(
        filter: *mut obs_source_t,
        effect: *mut gs_effect_t,
        width: u32,
        height: u32,
    );
    pub fn obs_source_process_filter_tech_end(
        filter: *mut obs_source_t,
        effect: *mut gs_effect_t,
        width: u32,
        height: u32,
        tech_name: *const c_char,
    );
    pub fn obs_source_video_render(source: *mut obs_source_t);

    // settings
    pub fn obs_data_get_double(data: *mut obs_data_t, name: *const c_char) -> f64;
    pub fn obs_data_get_int(data: *mut obs_data_t, name: *const c_char) -> i64;
    pub fn obs_data_get_bool(data: *mut obs_data_t, name: *const c_char) -> bool;
    pub fn obs_data_get_string(data: *mut obs_data_t, name: *const c_char) -> *const c_char;
    pub fn obs_data_set_default_double(data: *mut obs_data_t, name: *const c_char, val: f64);
    pub fn obs_data_set_default_int(data: *mut obs_data_t, name: *const c_char, val: i64);
    pub fn obs_data_set_default_bool(data: *mut obs_data_t, name: *const c_char, val: bool);
    pub fn obs_data_set_default_string(data: *mut obs_data_t, name: *const c_char, val: *const c_char);

    // properties
    pub fn obs_properties_create() -> *mut obs_properties_t;
    pub fn obs_properties_add_bool(
        props: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_float_slider(
        props: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        min: f64,
        max: f64,
        step: f64,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_int(
        props: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        min: c_int,
        max: c_int,
        step: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_list(
        props: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        list_type: c_int,
        format: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_text(
        props: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        text_type: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_property_list_add_int(
        prop: *mut obs_property_t,
        name: *const c_char,
        val: i64,
    ) -> usize;
    pub fn obs_property_list_add_string(
        prop: *mut obs_property_t,
        name: *const c_char,
        val: *const c_char,
    ) -> usize;
    pub fn obs_property_set_long_description(prop: *mut obs_property_t, long_desc: *const c_char);

    // graphics
    pub fn gs_texrender_create(format: c_int, zsformat: c_int) -> *mut gs_texrender_t;
    pub fn gs_texrender_destroy(texrender: *mut gs_texrender_t);
    pub fn gs_texrender_begin(texrender: *mut gs_texrender_t, cx: u32, cy: u32) -> bool;
    pub fn gs_texrender_end(texrender: *mut gs_texrender_t);
    pub fn gs_texrender_reset(texrender: *mut gs_texrender_t);
    pub fn gs_texrender_get_texture(texrender: *const gs_texrender_t) -> *mut gs_texture_t;

    pub fn gs_stagesurface_create(width: u32, height: u32, format: c_int) -> *mut gs_stagesurf_t;
    pub fn gs_stagesurface_destroy(surf: *mut gs_stagesurf_t);
    pub fn gs_stagesurface_map(surf: *mut gs_stagesurf_t, data: *mut *mut u8, linesize: *mut u32) -> bool;
    pub fn gs_stagesurface_unmap(surf: *mut gs_stagesurf_t);
    pub fn gs_stage_texture(dst: *mut gs_stagesurf_t, src: *mut gs_texture_t);

    pub fn gs_texture_create(
        width: u32,
        height: u32,
        format: c_int,
        levels: u32,
        data: *const *const u8,
        flags: u32,
    ) -> *mut gs_texture_t;
    pub fn gs_texture_destroy(tex: *mut gs_texture_t);
    pub fn gs_texture_set_image(tex: *mut gs_texture_t, data: *const u8, linesize: u32, invert: bool);

    pub fn gs_effect_create_from_file(file: *const c_char, error_string: *mut *mut c_char) -> *mut gs_effect_t;
    pub fn gs_effect_destroy(effect: *mut gs_effect_t);
    pub fn gs_effect_get_param_by_name(effect: *const gs_effect_t, name: *const c_char) -> *mut gs_eparam_t;
    pub fn gs_effect_set_texture(param: *mut gs_eparam_t, val: *mut gs_texture_t);
    pub fn gs_effect_set_texture_srgb(param: *mut gs_eparam_t, val: *mut gs_texture_t);
    pub fn gs_effect_set_float(param: *mut gs_eparam_t, val: c_float);
    pub fn gs_effect_set_int(param: *mut gs_eparam_t, val: c_int);
    pub fn gs_effect_set_bool(param: *mut gs_eparam_t, val: bool);
    pub fn gs_effect_set_vec2(param: *mut gs_eparam_t, val: *const Vec2);

    pub fn gs_ortho(left: c_float, right: c_float, top: c_float, bottom: c_float, znear: c_float, zfar: c_float);
    pub fn gs_clear(clear_flags: u32, color: *const Vec4, depth: c_float, stencil: u8);
    pub fn gs_set_viewport(x: c_int, y: c_int, cx: c_int, cy: c_int);
    pub fn gs_blend_state_push();
    pub fn gs_blend_state_pop();
    pub fn gs_enable_blending(enable: bool);
    pub fn gs_draw_sprite(tex: *mut gs_texture_t, flip: u32, width: u32, height: u32);

    pub fn gs_effect_loop(effect: *mut gs_effect_t, name: *const c_char) -> bool;
}

pub const GS_CLEAR_COLOR: u32 = 1 << 0;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Vec2 {
    pub x: c_float,
    pub y: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Vec4 {
    pub x: c_float,
    pub y: c_float,
    pub z: c_float,
    pub w: c_float,
}
