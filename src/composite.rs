//! Despill and final compositing math.
//!
//! This is the reference implementation. `data/effects/corridorkey.effect`
//! mirrors it in HLSL because it has to run per-pixel on the GPU; the two must
//! stay in step, and the tests here are what pin the behaviour down. The offline
//! harness (`examples/key_image.rs`) runs these functions, so a change that
//! breaks the math shows up without needing OBS.

use crate::hint::ScreenColor;

#[derive(Clone, Copy, Debug)]
pub struct CompositeParams {
    pub screen: ScreenColor,
    pub despill: f32,
    pub fg_detail: f32,
    pub matte_black: f32,
    pub matte_white: f32,
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self {
            screen: ScreenColor::Green,
            despill: 1.0,
            fg_detail: 1.0,
            matte_black: 0.0,
            matte_white: 1.0,
        }
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

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Applies the matte black/white points.
pub fn shape_alpha(alpha: f32, p: &CompositeParams) -> f32 {
    let span = (p.matte_white - p.matte_black).max(1e-4);
    ((alpha - p.matte_black) / span).clamp(0.0, 1.0)
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
    p: &CompositeParams,
) -> ([f32; 3], f32) {
    let alpha = shape_alpha(model_alpha, p);

    let w = p.fg_detail * smoothstep(0.85, 1.0, alpha);
    let mixed = [
        model_fg[0] + (plate[0] - model_fg[0]) * w,
        model_fg[1] + (plate[1] - model_fg[1]) * w,
        model_fg[2] + (plate[2] - model_fg[2]) * w,
    ];

    (despill(mixed, p.screen, p.despill), alpha)
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
        let (rgb, a) = composite(plate, model, 1.0, &CompositeParams::default());
        assert_eq!(a, 1.0);
        assert!((rgb[0] - plate[0]).abs() < 1e-5, "{rgb:?}");
    }

    #[test]
    fn soft_edges_use_the_model_prediction() {
        // At alpha 0.5 the plate is still contaminated with screen color, so the
        // model's unmixed foreground must win outright.
        let plate = [0.3, 0.7, 0.3];
        let model = [0.8, 0.25, 0.2];
        let (rgb, _) = composite(plate, model, 0.5, &CompositeParams::default());
        let expected = despill(model, ScreenColor::Green, 1.0);
        assert!((rgb[0] - expected[0]).abs() < 1e-5, "{rgb:?} vs {expected:?}");
    }

    #[test]
    fn fg_detail_zero_never_reaches_for_the_plate() {
        let p = CompositeParams { fg_detail: 0.0, ..Default::default() };
        let plate = [0.9, 0.2, 0.2];
        let model = [0.1, 0.1, 0.1];
        let (rgb, _) = composite(plate, model, 1.0, &p);
        assert!((rgb[0] - 0.1).abs() < 1e-5, "{rgb:?}");
    }
}
