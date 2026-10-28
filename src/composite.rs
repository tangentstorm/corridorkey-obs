//! Despill and final compositing math.
//!
//! This is the reference implementation. `data/effects/corridorkey.effect`
//! mirrors it in HLSL because it has to run per-pixel on the GPU; the two must
//! stay in step, and the tests here are what pin the behaviour down. The offline
//! harness (`examples/key_image.rs`) runs these functions, so a change that
//! breaks the math shows up without needing OBS.

use crate::hint::{rgb_to_uv, ScreenColor, BLACK_FLOOR};

/// Which same-frame GPU corrections to apply to the (possibly stale) neural
/// matte. Veto attacks false white (matte opaque over what is now screen);
/// fill attacks false black (matte transparent over what is now subject).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CorrectionMode {
    Off,
    Veto,
    Fill,
    Both,
}

impl CorrectionMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "veto" => CorrectionMode::Veto,
            "fill" => CorrectionMode::Fill,
            "both" => CorrectionMode::Both,
            _ => CorrectionMode::Off,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CorrectionMode::Off => "off",
            CorrectionMode::Veto => "veto",
            CorrectionMode::Fill => "fill",
            CorrectionMode::Both => "both",
        }
    }

    pub fn veto(self) -> bool {
        matches!(self, CorrectionMode::Veto | CorrectionMode::Both)
    }

    pub fn fill(self) -> bool {
        matches!(self, CorrectionMode::Fill | CorrectionMode::Both)
    }
}

/// How (and where) despill is applied in the composite.
///
/// `Legacy` despills the final mix with energy redistribution into the other
/// two channels. That redistribution turns removed green spill into a red+blue
/// (magenta) boost, and applying it after the mix also re-despills the model's
/// already-unmixed foreground — together, a purple fringe on dark backgrounds.
/// The other modes exist to test avoiding one or both halves of that:
/// `Plate` despills only the plate contribution (model FG untouched),
/// `Clamp` caps the screen channel at the other channels' mean with no
/// redistribution (edges go neutral/darker instead of magenta), and
/// `PlateClamp` is both changes at once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DespillMode {
    Legacy,
    Plate,
    Clamp,
    PlateClamp,
}

impl DespillMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "plate" => DespillMode::Plate,
            "clamp" => DespillMode::Clamp,
            "plate_clamp" => DespillMode::PlateClamp,
            _ => DespillMode::Legacy,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DespillMode::Legacy => "legacy",
            DespillMode::Plate => "plate",
            DespillMode::Clamp => "clamp",
            DespillMode::PlateClamp => "plate_clamp",
        }
    }

    /// Despill the plate before mixing, leaving model FG untouched.
    pub fn plate_only(self) -> bool {
        matches!(self, DespillMode::Plate | DespillMode::PlateClamp)
    }

    /// Cap the screen channel instead of redistributing the removed energy.
    pub fn clamp_style(self) -> bool {
        matches!(self, DespillMode::Clamp | DespillMode::PlateClamp)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompositeParams {
    pub screen: ScreenColor,
    pub despill: f32,
    pub despill_mode: DespillMode,
    pub fg_detail: f32,
    pub matte_black: f32,
    pub matte_white: f32,
    // Same-frame GPU matte correction (see correct_alpha).
    pub correct_mode: CorrectionMode,
    pub veto_strength: f32,
    pub fill_strength: f32,
    /// Chroma evidence thresholds, same meaning as the hint's, but separate so
    /// correction can be tuned without moving the network's hint.
    pub correct_similarity: f32,
    pub correct_smoothness: f32,
    /// 0 = ignore whether the pixel changed since the matte was computed,
    /// 1 = only correct pixels that changed.
    pub change_gate: f32,
    /// Per-channel absolute difference that counts as "changed".
    pub change_threshold: f32,
    /// 0 = correct anywhere, 1 = only where matte neighborhood context allows
    /// (near matte edges / enclosed holes; see correct_alpha).
    pub spatial_gate: f32,
    /// Neighborhood radius, in matte texels, for the spatial context taps.
    pub correct_reach: f32,
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self {
            screen: ScreenColor::Green,
            despill: 1.0,
            despill_mode: DespillMode::Legacy,
            fg_detail: 1.0,
            matte_black: 0.0,
            matte_white: 1.0,
            correct_mode: CorrectionMode::Off,
            veto_strength: 1.0,
            fill_strength: 1.0,
            correct_similarity: 0.30,
            correct_smoothness: 0.12,
            change_gate: 1.0,
            change_threshold: 0.08,
            spatial_gate: 1.0,
            correct_reach: 8.0,
        }
    }
}

/// Per-pixel context the correction needs beyond the plate and matte samples.
#[derive(Clone, Copy, Debug)]
pub struct CorrectionInputs {
    /// The plate pixel the current matte was computed from (low-res, upsampled
    /// by the sampler). Used by the change gate.
    pub matte_plate: [f32; 3],
    /// Min/max shaped alpha within `correct_reach` of this pixel in the matte.
    /// Used by the spatial gate.
    pub matte_min: f32,
    pub matte_max: f32,
}

impl Default for CorrectionInputs {
    fn default() -> Self {
        Self { matte_plate: [0.0; 3], matte_min: 0.0, matte_max: 1.0 }
    }
}

/// Luminance-preserving despill, ported from CorridorKey's
/// `core/color_utils.py::despill_torch`.
///
/// Subtracts whatever the screen channel has in excess of the mean of the other
/// two, then hands half the removed energy to each of them, so the pixel doesn't
/// simply go darker where spill is removed.
pub fn despill(c: [f32; 3], screen: ScreenColor, strength: f32) -> [f32; 3] {
    if strength <= 0.0 {
        return c;
    }
    let si = screen.channel();
    let (ai, bi) = match si {
        1 => (0, 2),
        _ => (0, 1),
    };

    let limit = (c[ai] + c[bi]) * 0.5;
    let spill = (c[si] - limit).max(0.0) * strength;

    let mut out = c;
    out[si] = c[si] - spill;
    out[ai] = c[ai] + spill * 0.5;
    out[bi] = c[bi] + spill * 0.5;
    out
}

/// Clamp-style despill: pull the screen channel down toward the mean of the
/// other two, without handing the removed energy to them. The pixel gets
/// darker/more neutral where spill is removed instead of magenta-shifted,
/// which on a dark background reads as nearly invisible.
pub fn despill_clamp(c: [f32; 3], screen: ScreenColor, strength: f32) -> [f32; 3] {
    if strength <= 0.0 {
        return c;
    }
    let si = screen.channel();
    let (ai, bi) = match si {
        1 => (0, 2),
        _ => (0, 1),
    };

    let limit = (c[ai] + c[bi]) * 0.5;
    let spill = (c[si] - limit).max(0.0) * strength;

    let mut out = c;
    out[si] = c[si] - spill;
    out
}

/// Applies the despill variant selected by `mode` at the given strength.
fn despill_as(c: [f32; 3], p: &CompositeParams) -> [f32; 3] {
    if p.despill_mode.clamp_style() {
        despill_clamp(c, p.screen, p.despill)
    } else {
        despill(c, p.screen, p.despill)
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Applies the matte black/white points.
pub fn shape_alpha(alpha: f32, p: &CompositeParams) -> f32 {
    let span = (p.matte_white - p.matte_black).max(1e-4);
    ((alpha - p.matte_black) / span).clamp(0.0, 1.0)
}

/// Same-frame chroma evidence that a plate pixel is screen / not screen.
///
/// Returns (bg_conf, fg_conf), each in [0, 1]. The underlying hint value `g`
/// ramps 0 (screen chroma) to 1 (far from screen) across
/// similarity..similarity+smoothness, exactly as in hint::chroma_hint. The
/// confidences add a dead zone in the middle of that ramp: semi-transparent
/// edges, spill, and ambiguous colors produce *no* evidence either way, so the
/// neural matte keeps sole ownership of precisely the pixels a chroma test
/// cannot judge. Near-black pixels carry no hue and also produce no evidence.
pub fn chroma_evidence(plate: [f32; 3], p: &CompositeParams) -> (f32, f32) {
    if plate[0].max(plate[1]).max(plate[2]) < BLACK_FLOOR {
        return (0.0, 0.0);
    }
    let (ku, kv) = p.screen.key_uv();
    let (u, v) = rgb_to_uv(plate[0], plate[1], plate[2]);
    let dist = ((u - ku) * (u - ku) + (v - kv) * (v - kv)).sqrt();
    let lo = p.correct_similarity;
    let hi = p.correct_similarity + p.correct_smoothness.max(1e-4);
    let g = smoothstep(lo, hi, dist);
    let bg_conf = 1.0 - smoothstep(0.15, 0.45, g);
    let fg_conf = smoothstep(0.55, 0.85, g);
    (bg_conf, fg_conf)
}

/// Corrects a shaped neural alpha with same-frame chroma evidence.
///
/// Both directions of matte error are handled, and both directions of chroma
/// evidence error are guarded against:
///
/// - Veto (false white): definite-screen evidence pulls alpha toward 0.
///   Guards: green wardrobe deep inside the subject is protected by the
///   spatial gate (veto needs background-level matte within reach) and by the
///   change gate (a pixel unchanged since the matte was computed is trusted).
/// - Fill (false black): definite-not-screen evidence pulls alpha toward 1.
///   Guards: non-green set pieces deep in the background are protected by the
///   spatial gate (fill needs solid matte within reach, i.e. a leading edge or
///   an enclosed hole) and by the change gate.
///
/// The pull is proportional (not a hard clamp), so partial weights degrade
/// gracefully and a correct alpha already at the bound is unmoved.
pub fn correct_alpha(
    alpha: f32,
    plate: [f32; 3],
    ci: &CorrectionInputs,
    p: &CompositeParams,
) -> f32 {
    if p.correct_mode == CorrectionMode::Off {
        return alpha;
    }
    let (bg_conf, fg_conf) = chroma_evidence(plate, p);
    if bg_conf <= 0.0 && fg_conf <= 0.0 {
        return alpha;
    }

    // Change gate: a pixel that matches what the network saw cannot have a
    // stale matte, whatever the chroma test thinks of it.
    let diff = (plate[0] - ci.matte_plate[0])
        .abs()
        .max((plate[1] - ci.matte_plate[1]).abs())
        .max((plate[2] - ci.matte_plate[2]).abs());
    let t = p.change_threshold.max(1e-4);
    let changed = smoothstep(t, t * 2.0 + 0.02, diff);
    let gate = 1.0 - p.change_gate.clamp(0.0, 1.0) * (1.0 - changed);
    if gate <= 0.0 {
        return alpha;
    }

    // Spatial gate from matte neighborhood within reach.
    let fill_ctx = smoothstep(0.5, 0.9, ci.matte_max);
    let veto_ctx = 1.0 - smoothstep(0.1, 0.5, ci.matte_min);
    let sg = p.spatial_gate.clamp(0.0, 1.0);
    let fill_ctx = 1.0 - sg * (1.0 - fill_ctx);
    let veto_ctx = 1.0 - sg * (1.0 - veto_ctx);

    let mut a = alpha;
    if p.correct_mode.veto() {
        let w = (p.veto_strength * bg_conf * gate * veto_ctx).clamp(0.0, 1.0);
        a *= 1.0 - w;
    }
    if p.correct_mode.fill() {
        let w = (p.fill_strength * fg_conf * gate * fill_ctx).clamp(0.0, 1.0);
        a += (1.0 - a) * w;
    }
    a.clamp(0.0, 1.0)
}

/// One output pixel.
///
/// `plate` is the full-resolution source pixel; `model_fg` and `model_alpha` come
/// from the (lower-resolution) network output, already upsampled by the sampler.
///
/// Where the matte is solid, the network's prediction carries no detail the plate
/// doesn't already have, so we fade back to the plate there and keep the model's
/// unmixed color for the soft edges — the only place the unmixing matters.
pub fn composite(
    plate: [f32; 3],
    model_fg: [f32; 3],
    model_alpha: f32,
    ci: &CorrectionInputs,
    p: &CompositeParams,
) -> ([f32; 3], f32) {
    let alpha = correct_alpha(shape_alpha(model_alpha, p), plate, ci, p);

    // Plate-only modes despill the plate contribution before the mix, so the
    // model's already-unmixed foreground is never despilled a second time.
    let plate = if p.despill_mode.plate_only() { despill_as(plate, p) } else { plate };

    let w = p.fg_detail * smoothstep(0.85, 1.0, alpha);
    let mixed = [
        model_fg[0] + (plate[0] - model_fg[0]) * w,
        model_fg[1] + (plate[1] - model_fg[1]) * w,
        model_fg[2] + (plate[2] - model_fg[2]) * w,
    ];

    let fg = if p.despill_mode.plate_only() { mixed } else { despill_as(mixed, p) };
    (fg, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn despill_removes_green_excess() {
        let out = despill([0.4, 0.9, 0.4], ScreenColor::Green, 1.0);
        // Green was 0.5 above the 0.4 mean of red and blue; it should end up level
        // with them once the excess is redistributed.
        assert!((out[1] - 0.4).abs() < 1e-5, "{out:?}");
        assert!((out[0] - 0.65).abs() < 1e-5, "{out:?}");
        assert!((out[2] - 0.65).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn despill_preserves_total_energy() {
        let c = [0.3, 0.85, 0.45];
        let out = despill(c, ScreenColor::Green, 1.0);
        let before: f32 = c.iter().sum();
        let after: f32 = out.iter().sum();
        assert!((before - after).abs() < 1e-5, "{before} vs {after}");
    }

    #[test]
    fn despill_leaves_non_spill_colors_alone() {
        // Red-dominant pixel: nothing to remove from the green channel.
        let c = [0.8, 0.3, 0.25];
        assert_eq!(despill(c, ScreenColor::Green, 1.0), c);
    }

    #[test]
    fn despill_strength_scales_continuously() {
        let c = [0.4, 0.9, 0.4];
        let full = despill(c, ScreenColor::Green, 1.0);
        let half = despill(c, ScreenColor::Green, 0.5);
        assert_eq!(despill(c, ScreenColor::Green, 0.0), c);
        assert!((half[1] - (c[1] + full[1]) * 0.5).abs() < 1e-5, "{half:?}");
    }

    #[test]
    fn blue_screen_despills_the_blue_channel() {
        let out = despill([0.4, 0.4, 0.9], ScreenColor::Blue, 1.0);
        assert!((out[2] - 0.4).abs() < 1e-5, "{out:?}");
        assert!((out[0] - 0.65).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn matte_points_clamp_and_stretch() {
        let p = CompositeParams { matte_black: 0.1, matte_white: 0.9, ..Default::default() };
        assert_eq!(shape_alpha(0.05, &p), 0.0);
        assert_eq!(shape_alpha(0.95, &p), 1.0);
        assert!((shape_alpha(0.5, &p) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn solid_interior_uses_the_full_res_plate() {
        let plate = [0.9, 0.2, 0.2];
        let model = [0.1, 0.1, 0.1];
        let (rgb, a) =
            composite(plate, model, 1.0, &CorrectionInputs::default(), &CompositeParams::default());
        assert_eq!(a, 1.0);
        assert!((rgb[0] - plate[0]).abs() < 1e-5, "{rgb:?}");
    }

    #[test]
    fn soft_edges_use_the_model_prediction() {
        // At alpha 0.5 the plate is still contaminated with screen color, so the
        // model's unmixed foreground must win outright.
        let plate = [0.3, 0.7, 0.3];
        let model = [0.8, 0.25, 0.2];
        let (rgb, _) =
            composite(plate, model, 0.5, &CorrectionInputs::default(), &CompositeParams::default());
        let expected = despill(model, ScreenColor::Green, 1.0);
        assert!((rgb[0] - expected[0]).abs() < 1e-5, "{rgb:?} vs {expected:?}");
    }

    #[test]
    fn fg_detail_zero_never_reaches_for_the_plate() {
        let p = CompositeParams { fg_detail: 0.0, ..Default::default() };
        let plate = [0.9, 0.2, 0.2];
        let model = [0.1, 0.1, 0.1];
        let (rgb, _) = composite(plate, model, 1.0, &CorrectionInputs::default(), &p);
        assert!((rgb[0] - 0.1).abs() < 1e-5, "{rgb:?}");
    }

    // --- Same-frame correction ---------------------------------------------
    //
    // Colors: pure screen green, and a skin-ish tone far from it. "Old plate"
    // is what the network saw when it produced the matte being corrected.

    const GREEN: [f32; 3] = [0.05, 0.9, 0.05];
    const SKIN: [f32; 3] = [0.85, 0.6, 0.45];

    fn both() -> CompositeParams {
        CompositeParams { correct_mode: CorrectionMode::Both, ..Default::default() }
    }

    fn ctx(matte_plate: [f32; 3], matte_min: f32, matte_max: f32) -> CorrectionInputs {
        CorrectionInputs { matte_plate, matte_min, matte_max }
    }

    #[test]
    fn evidence_is_silent_near_black_and_opinionated_on_pure_colors() {
        let p = both();
        assert_eq!(chroma_evidence([0.001, 0.002, 0.001], &p), (0.0, 0.0));
        let (bg, fg) = chroma_evidence(GREEN, &p);
        assert!(bg > 0.95 && fg == 0.0, "green: bg={bg} fg={fg}");
        let (bg, fg) = chroma_evidence(SKIN, &p);
        assert!(fg > 0.95 && bg == 0.0, "skin: bg={bg} fg={fg}");
    }

    #[test]
    fn veto_kills_false_white_where_subject_moved_away() {
        // Matte says opaque; pixel is green now but was subject when keyed.
        // Background-level matte is within reach (this is the vacated edge).
        let a = correct_alpha(1.0, GREEN, &ctx(SKIN, 0.0, 1.0), &both());
        assert!(a < 0.05, "false white should be vetoed, got {a}");
    }

    #[test]
    fn fill_repairs_false_black_where_subject_moved_in() {
        // Matte says transparent; pixel is subject now but was green when keyed.
        let a = correct_alpha(0.0, SKIN, &ctx(GREEN, 0.0, 1.0), &both());
        assert!(a > 0.95, "false black should be filled, got {a}");
    }

    #[test]
    fn change_gate_trusts_matte_on_unchanged_pixels() {
        // Green wardrobe, correctly opaque, pixel identical to what the model
        // saw: chroma evidence must not eat it.
        let a = correct_alpha(1.0, GREEN, &ctx(GREEN, 0.0, 1.0), &both());
        assert_eq!(a, 1.0);
        // Non-green set piece, correctly transparent, also unchanged.
        let a = correct_alpha(0.0, SKIN, &ctx(SKIN, 0.0, 1.0), &both());
        assert_eq!(a, 0.0);
        // With the gate off, the same pixels *are* overridden — that contrast
        // is what the gate knob exists for.
        let p = CompositeParams { change_gate: 0.0, ..both() };
        assert!(correct_alpha(1.0, GREEN, &ctx(GREEN, 0.0, 1.0), &p) < 0.05);
        assert!(correct_alpha(0.0, SKIN, &ctx(SKIN, 0.0, 1.0), &p) > 0.95);
    }

    #[test]
    fn spatial_gate_blocks_correction_away_from_matte_edges() {
        // Deep interior (min matte within reach is solid): no veto, even on a
        // changed green pixel.
        let a = correct_alpha(1.0, GREEN, &ctx(SKIN, 1.0, 1.0), &both());
        assert_eq!(a, 1.0);
        // Deep background (max matte within reach is empty): no fill, even on
        // a changed non-green pixel.
        let a = correct_alpha(0.0, SKIN, &ctx(GREEN, 0.0, 0.0), &both());
        assert_eq!(a, 0.0);
        // Gate off removes that protection.
        let p = CompositeParams { spatial_gate: 0.0, ..both() };
        assert!(correct_alpha(1.0, GREEN, &ctx(SKIN, 1.0, 1.0), &p) < 0.05);
        assert!(correct_alpha(0.0, SKIN, &ctx(GREEN, 0.0, 0.0), &p) > 0.95);
    }

    #[test]
    fn modes_are_independent_and_off_is_identity() {
        let ci_vacated = ctx(SKIN, 0.0, 1.0);
        let ci_entered = ctx(GREEN, 0.0, 1.0);
        let veto_only =
            CompositeParams { correct_mode: CorrectionMode::Veto, ..Default::default() };
        assert!(correct_alpha(1.0, GREEN, &ci_vacated, &veto_only) < 0.05);
        assert_eq!(correct_alpha(0.0, SKIN, &ci_entered, &veto_only), 0.0);

        let fill_only =
            CompositeParams { correct_mode: CorrectionMode::Fill, ..Default::default() };
        assert_eq!(correct_alpha(1.0, GREEN, &ci_vacated, &fill_only), 1.0);
        assert!(correct_alpha(0.0, SKIN, &ci_entered, &fill_only) > 0.95);

        let off = CompositeParams::default();
        assert_eq!(correct_alpha(0.7, GREEN, &ci_vacated, &off), 0.7);
    }

    #[test]
    fn correction_leaves_correct_alpha_at_the_bound_alone() {
        // Already-transparent green and already-opaque subject pixels must not
        // drift just because correction is enabled.
        let p = both();
        assert_eq!(correct_alpha(0.0, GREEN, &ctx(SKIN, 0.0, 1.0), &p), 0.0);
        assert_eq!(correct_alpha(1.0, SKIN, &ctx(GREEN, 0.0, 1.0), &p), 1.0);
    }

    #[test]
    fn strengths_scale_the_pull() {
        let p = CompositeParams { veto_strength: 0.5, fill_strength: 0.5, ..both() };
        let a = correct_alpha(1.0, GREEN, &ctx(SKIN, 0.0, 1.0), &p);
        assert!((a - 0.5).abs() < 0.06, "half-strength veto, got {a}");
        let a = correct_alpha(0.0, SKIN, &ctx(GREEN, 0.0, 1.0), &p);
        assert!((a - 0.5).abs() < 0.06, "half-strength fill, got {a}");
    }

    // --- Despill modes -------------------------------------------------------

    /// (R+B)/2 - G: positive reads magenta/purple, negative reads green.
    fn magenta_excess(c: [f32; 3]) -> f32 {
        (c[0] + c[2]) * 0.5 - c[1]
    }

    #[test]
    fn clamp_despill_never_boosts_the_other_channels() {
        let c = [0.4, 0.9, 0.4];
        let out = despill_clamp(c, ScreenColor::Green, 1.0);
        // Green is capped at the 0.4 mean; red and blue are untouched, so no
        // magenta is manufactured (the legacy despill pushes them to 0.65).
        assert!((out[1] - 0.4).abs() < 1e-5, "{out:?}");
        assert_eq!(out[0], c[0]);
        assert_eq!(out[2], c[2]);
        assert!(magenta_excess(out).abs() < 1e-5, "{out:?}");
        assert!(magenta_excess(despill(c, ScreenColor::Green, 1.0)) > 0.2);
    }

    #[test]
    fn clamp_despill_scales_with_strength_and_respects_blue_screen() {
        let c = [0.4, 0.9, 0.4];
        assert_eq!(despill_clamp(c, ScreenColor::Green, 0.0), c);
        let half = despill_clamp(c, ScreenColor::Green, 0.5);
        assert!((half[1] - 0.65).abs() < 1e-5, "{half:?}");

        let out = despill_clamp([0.4, 0.4, 0.9], ScreenColor::Blue, 1.0);
        assert!((out[2] - 0.4).abs() < 1e-5, "{out:?}");
        assert_eq!(out[0], 0.4);
        assert_eq!(out[1], 0.4);
    }

    #[test]
    fn plate_modes_leave_model_fg_untouched_at_soft_edges() {
        // At alpha 0.5 the plate is not used, so plate-only modes must return
        // the model prediction verbatim — no second despill pass over already
        // unmixed color. Legacy mode measurably shifts it magenta.
        let plate = [0.05, 0.95, 0.05];
        let model = [0.55, 0.62, 0.5]; // slightly green-leaning pale tone
        let ci = CorrectionInputs::default();

        for mode in [DespillMode::Plate, DespillMode::PlateClamp] {
            let p = CompositeParams { despill_mode: mode, ..Default::default() };
            let (rgb, _) = composite(plate, model, 0.5, &ci, &p);
            assert_eq!(rgb, model, "mode {mode:?} should not touch model FG");
        }

        let legacy = CompositeParams::default();
        let (rgb, _) = composite(plate, model, 0.5, &ci, &legacy);
        assert!(
            magenta_excess(rgb) > magenta_excess(model) + 0.02,
            "legacy re-despill should push magenta: {rgb:?} vs model {model:?}"
        );
    }

    #[test]
    fn plate_clamp_despills_solid_interior_without_magenta() {
        // Solid pixel (alpha 1) comes from the plate under fg_detail, so the
        // plate-side despill must still clean it up — neutrally.
        let p = CompositeParams { despill_mode: DespillMode::PlateClamp, ..Default::default() };
        let plate = [0.4, 0.9, 0.4]; // heavy green spill on the subject
        let model = [0.1, 0.1, 0.1];
        let (rgb, a) = composite(plate, model, 1.0, &CorrectionInputs::default(), &p);
        assert_eq!(a, 1.0);
        assert!((rgb[1] - 0.4).abs() < 1e-5, "spill not capped: {rgb:?}");
        assert!(magenta_excess(rgb).abs() < 1e-5, "magenta crept in: {rgb:?}");
    }
}
