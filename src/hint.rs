//! Coarse alpha-hint generation.
//!
//! GreenFormer takes 4 channels: RGB plus a rough alpha hint telling it which
//! blob is the subject. Upstream CorridorKey gets that hint from GVM, VideoMaMa
//! or BiRefNet — all far too heavy to run per-frame alongside the keyer itself.
//!
//! For a live green screen the hint is nearly free: a classic chroma key is
//! exactly the "rough black-and-white mask" the model was trained to refine. It
//! costs well under a millisecond at 512x512 and adds no second network.

/// Which screen we're keying. Also selects the despill channel and the model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScreenColor {
    Green,
    Blue,
}

impl ScreenColor {
    /// Index into RGB of the screen's own channel.
    pub fn channel(self) -> usize {
        match self {
            ScreenColor::Green => 1,
            ScreenColor::Blue => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ScreenColor::Green => "green",
            ScreenColor::Blue => "blue",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "blue" => ScreenColor::Blue,
            _ => ScreenColor::Green,
        }
    }

    /// Reference chroma (U, V) of a fully saturated screen, BT.709.
    fn key_uv(self) -> (f32, f32) {
        match self {
            ScreenColor::Green => rgb_to_uv(0.0, 1.0, 0.0),
            ScreenColor::Blue => rgb_to_uv(0.0, 0.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HintParams {
    pub screen: ScreenColor,
    /// Chroma distance below which a pixel is considered pure screen.
    pub similarity: f32,
    /// Width of the ramp from screen to subject.
    pub smoothness: f32,
}

impl Default for HintParams {
    fn default() -> Self {
        Self { screen: ScreenColor::Green, similarity: 0.30, smoothness: 0.12 }
    }
}

/// Brightness-normalized BT.709 chroma.
///
/// Dividing by the max channel first is the important part. Raw (U, V) scales
/// with brightness, so a shadowed corner of the screen and a hot spot under the
/// key light land at very different distances from the reference chroma, and no
/// single similarity threshold covers both — which is exactly why classic
/// chroma keys fall apart on unevenly lit screens. Normalizing first makes the
/// hint depend on hue and saturation only.
///
/// Near-black pixels have no meaningful hue; callers handle those separately.
fn rgb_to_uv(r: f32, g: f32, b: f32) -> (f32, f32) {
    let m = r.max(g).max(b).max(1e-6);
    let (r, g, b) = (r / m, g / m, b / m);
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    ((b - y) / 1.8556, (r - y) / 1.5748)
}

/// Below this max-channel level a pixel carries no usable hue. Treated as
/// foreground: near-black is far more often deep shadow on the subject (hair,
/// dark clothing) than it is screen, and the model can erode a too-generous
/// hint far more easily than it can invent detail a too-tight one removed.
const BLACK_FLOOR: f32 = 5.0 / 255.0;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fills `hint` (len w*h) from BGRA8 pixels.
///
/// `bgra` is what OBS hands back from a staged surface: 4 bytes per pixel,
/// `stride` bytes per row (which is *not* always `w * 4`).
pub fn chroma_hint(bgra: &[u8], w: usize, h: usize, stride: usize, p: &HintParams, hint: &mut [f32]) {
    debug_assert_eq!(hint.len(), w * h);
    let (ku, kv) = p.screen.key_uv();
    let lo = p.similarity;
    let hi = p.similarity + p.smoothness.max(1e-4);

    for y in 0..h {
        let row = &bgra[y * stride..y * stride + w * 4];
        let out = &mut hint[y * w..(y + 1) * w];
        for x in 0..w {
            let px = &row[x * 4..x * 4 + 4];
            let b = px[0] as f32 * (1.0 / 255.0);
            let g = px[1] as f32 * (1.0 / 255.0);
            let r = px[2] as f32 * (1.0 / 255.0);

            if r.max(g).max(b) < BLACK_FLOOR {
                out[x] = 1.0;
                continue;
            }
            let (u, v) = rgb_to_uv(r, g, b);
            let dist = ((u - ku) * (u - ku) + (v - kv) * (v - kv)).sqrt();
            // Far from the screen chroma => foreground => hint 1.
            out[x] = smoothstep(lo, hi, dist);
        }
    }
}

/// Unpacks BGRA8 into the planar RGB f32 [0,1] tensor layout the model wants.
pub fn bgra_to_chw(bgra: &[u8], w: usize, h: usize, stride: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), w * h * 3);
    let plane = w * h;
    for y in 0..h {
        let row = &bgra[y * stride..y * stride + w * 4];
        for x in 0..w {
            let px = &row[x * 4..x * 4 + 4];
            let i = y * w + x;
            out[i] = px[2] as f32 * (1.0 / 255.0);
            out[plane + i] = px[1] as f32 * (1.0 / 255.0);
            out[2 * plane + i] = px[0] as f32 * (1.0 / 255.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(b: u8, g: u8, r: u8) -> [u8; 4] {
        [b, g, r, 255]
    }

    #[test]
    fn pure_green_is_background_and_skin_is_foreground() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&px(0, 255, 0)); // screen
        buf.extend_from_slice(&px(120, 160, 220)); // skin-ish
        let mut hint = [0.0f32; 2];
        chroma_hint(&buf, 2, 1, 8, &HintParams::default(), &mut hint);
        assert!(hint[0] < 0.05, "green should read as background, got {}", hint[0]);
        assert!(hint[1] > 0.95, "subject should read as foreground, got {}", hint[1]);
    }

    #[test]
    fn hint_is_chroma_not_luma() {
        // A dark and a bright patch of the same green must give the same hint;
        // that is the whole point of keying on chroma.
        let mut buf = Vec::new();
        buf.extend_from_slice(&px(0, 80, 0));
        buf.extend_from_slice(&px(0, 240, 0));
        let mut hint = [0.0f32; 2];
        chroma_hint(&buf, 2, 1, 8, &HintParams::default(), &mut hint);
        assert!((hint[0] - hint[1]).abs() < 1e-3, "{:?}", hint);
        assert!(hint[0] < 0.05, "both should read as background, got {:?}", hint);
    }

    #[test]
    fn unevenly_lit_screen_keys_uniformly() {
        // A screen lit from one side: same hue, brightness falling off. All of
        // it must read as background under one threshold.
        let mut buf = Vec::new();
        for level in [60u8, 110, 170, 230] {
            buf.extend_from_slice(&px(level / 8, level, level / 8));
        }
        let mut hint = [0.0f32; 4];
        chroma_hint(&buf, 4, 1, 16, &HintParams::default(), &mut hint);
        assert!(hint.iter().all(|&h| h < 0.05), "{:?}", hint);
    }

    #[test]
    fn near_black_is_kept_as_foreground() {
        let buf = px(1, 2, 1);
        let mut hint = [0.0f32; 1];
        chroma_hint(&buf, 1, 1, 4, &HintParams::default(), &mut hint);
        assert_eq!(hint[0], 1.0);
    }

    #[test]
    fn blue_screen_flips_which_color_is_background() {
        let p = HintParams { screen: ScreenColor::Blue, ..Default::default() };
        let mut buf = Vec::new();
        buf.extend_from_slice(&px(255, 0, 0)); // pure blue
        buf.extend_from_slice(&px(0, 255, 0)); // pure green
        let mut hint = [0.0f32; 2];
        chroma_hint(&buf, 2, 1, 8, &p, &mut hint);
        assert!(hint[0] < 0.05);
        assert!(hint[1] > 0.95);
    }

    #[test]
    fn chw_unpack_respects_stride_and_channel_order() {
        // 1x1 image in a buffer with a padded stride.
        let buf = [10u8, 20, 30, 255, 0, 0, 0, 0];
        let mut out = [0.0f32; 3];
        bgra_to_chw(&buf, 1, 1, 8, &mut out);
        assert!((out[0] - 30.0 / 255.0).abs() < 1e-6, "R");
        assert!((out[1] - 20.0 / 255.0).abs() < 1e-6, "G");
        assert!((out[2] - 10.0 / 255.0).abs() < 1e-6, "B");
    }
}
