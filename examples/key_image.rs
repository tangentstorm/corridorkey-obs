//! Offline harness: run the exact keying path the OBS filter uses, on a PNG.
//!
//! This exercises everything except the OBS graphics calls — hint generation,
//! ONNX inference, and the composite math the shader mirrors. It's how you check
//! a model export or a threshold change without restarting OBS.
//!
//! Usage:
//!     cargo run --release --example key_image -- <input.png> [truth_alpha.png]
//!
//! Writes <input>_alpha.png, <input>_fg.png and <input>_comp.png next to the
//! input, and — if a ground-truth matte is given — prints error statistics.

use std::path::{Path, PathBuf};

use corridorkey_obs::composite::{
    composite, shape_alpha, CompositeParams, CorrectionInputs, CorrectionMode, DespillMode,
};
use corridorkey_obs::engine::{build_session, key_bgra, Scratch, DEFAULT_INFER_SIZE};
use corridorkey_obs::hint::{HintParams, ScreenColor};
use image::{GrayImage, ImageReader, RgbImage, RgbaImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tests/data/test_plate.png".to_string()),
    );
    let truth = args.next().map(PathBuf::from);

    // Mirrors the plugin's lookup: prefer the fp16 graph, fall back to fp32.
    // CORRIDORKEY_MODEL overrides both, for comparing exports side by side.
    let model = match std::env::var("CORRIDORKEY_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(format!(
            "models/corridorkey_green_{DEFAULT_INFER_SIZE}_fp16.onnx"
        )),
    };
    println!("model: {}", model.display());
    // -1: let ONNX Runtime pick the adapter, as the plugin does by default.
    let device: i32 = std::env::var("CORRIDORKEY_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(-1);
    // The graph's resolution comes back with the session; the input is resized to match.
    let (mut session, size) = build_session(&model, device)?;
    println!("model resolution: {size}x{size}");

    let img = ImageReader::open(&input)?.decode()?;
    let img = image::imageops::resize(
        &img.to_rgb8(),
        size as u32,
        size as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Pack to BGRA exactly as OBS's staged surface would hand it over.
    let stride = size * 4;
    let mut bgra = vec![0u8; stride * size];
    for (i, px) in img.pixels().enumerate() {
        bgra[i * 4] = px[2];
        bgra[i * 4 + 1] = px[1];
        bgra[i * 4 + 2] = px[0];
        bgra[i * 4 + 3] = 255;
    }

    let hint_params = HintParams { screen: ScreenColor::Green, ..Default::default() };
    let mut scratch = Scratch::new(size);

    // Time a few runs; the first includes lazy kernel setup in the EP.
    let mut result = None;
    for i in 0..4 {
        let t = std::time::Instant::now();
        let (r, timing) = key_bgra(&mut session, &bgra, stride, &hint_params, &mut scratch)?;
        let wall = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "run {i}: {wall:6.1} ms total  |  preprocess {:5.2}  infer {:6.1}  pack {:5.2}",
            timing.preprocess_us as f64 / 1000.0,
            timing.infer_us as f64 / 1000.0,
            timing.pack_us as f64 / 1000.0,
        );
        result = Some(r);
    }
    let result = result.unwrap();

    // Composite, matching what the shader does per pixel.
    //
    // Stale-matte simulation for testing the same-frame correction:
    // CORRIDORKEY_SHIFT=N displays the matte (and the plate it was computed
    // from) shifted N px right of the live plate, which manufactures pure
    // false-white error on one side of the subject and false-black on the
    // other. CORRIDORKEY_CORRECT=off|veto|fill|both selects the correction,
    // with CORRIDORKEY_VETO / _FILL / _CHANGE_GATE / _SPATIAL_GATE (0..1),
    // CORRIDORKEY_REACH (matte px) and CORRIDORKEY_CSIM overriding knobs.
    let ef = |name: &str, dflt: f32| -> f32 {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
    };
    let shift: i32 = std::env::var("CORRIDORKEY_SHIFT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let dflt = CompositeParams::default();
    let cp = CompositeParams {
        despill: ef("CORRIDORKEY_DESPILL", dflt.despill),
        despill_mode: DespillMode::from_str(
            std::env::var("CORRIDORKEY_DESPILL_MODE").as_deref().unwrap_or("legacy"),
        ),
        correct_mode: CorrectionMode::from_str(
            std::env::var("CORRIDORKEY_CORRECT").as_deref().unwrap_or("off"),
        ),
        veto_strength: ef("CORRIDORKEY_VETO", dflt.veto_strength),
        fill_strength: ef("CORRIDORKEY_FILL", dflt.fill_strength),
        correct_similarity: ef("CORRIDORKEY_CSIM", dflt.correct_similarity),
        change_gate: ef("CORRIDORKEY_CHANGE_GATE", dflt.change_gate),
        spatial_gate: ef("CORRIDORKEY_SPATIAL_GATE", dflt.spatial_gate),
        correct_reach: ef("CORRIDORKEY_REACH", dflt.correct_reach),
        ..dflt
    };
    if shift != 0 || cp.correct_mode != CorrectionMode::Off {
        println!(
            "correction test: shift={shift}px mode={} veto={:.2} fill={:.2} \
             change_gate={:.2} spatial_gate={:.2} reach={:.0} csim={:.2} \
             despill={:.2}/{}",
            cp.correct_mode.as_str(),
            cp.veto_strength,
            cp.fill_strength,
            cp.change_gate,
            cp.spatial_gate,
            cp.correct_reach,
            cp.correct_similarity,
            cp.despill,
            cp.despill_mode.as_str(),
        );
    }

    // Plate, stale (shifted) model outputs, and the matte-era plate.
    let plate_at = |x: usize, y: usize| -> [f32; 3] {
        let px = img.get_pixel(x as u32, y as u32);
        [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0]
    };
    let src_x = |x: usize| -> usize {
        (x as i32 - shift).clamp(0, size as i32 - 1) as usize
    };
    let mut stale_alpha = vec![0.0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            stale_alpha[y * size + x] =
                result.rgba[(y * size + src_x(x)) * 4 + 3] as f32 / 255.0;
        }
    }
    // Shaped neighborhood min/max over the stale matte, as the shader's taps
    // approximate (here: a full box, which errs on the generous side).
    let reach = cp.correct_reach.max(0.0) as i32;
    let shaped: Vec<f32> = stale_alpha.iter().map(|&a| shape_alpha(a, &cp)).collect();
    let mut matte_min = vec![0.0f32; size * size];
    let mut matte_max = vec![0.0f32; size * size];
    for y in 0..size as i32 {
        for x in 0..size as i32 {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let nx = (x + dx).clamp(0, size as i32 - 1) as usize;
                    let ny = (y + dy).clamp(0, size as i32 - 1) as usize;
                    let v = shaped[ny * size + nx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            matte_min[y as usize * size + x as usize] = lo;
            matte_max[y as usize * size + x as usize] = hi;
        }
    }

    let mut alpha_img = GrayImage::new(size as u32, size as u32);
    let mut fg_img = RgbImage::new(size as u32, size as u32);
    let mut comp_img = RgbaImage::new(size as u32, size as u32);
    let mut alpha_out = vec![0.0f32; size * size];
    let mut edge_magenta_sum = 0.0f64;
    let mut edge_magenta_n = 0usize;

    for i in 0..size * size {
        let (x, y) = (i % size, i / size);
        let plate = plate_at(x, y);
        let si = y * size + src_x(x);
        let model_fg = [
            result.rgba[si * 4] as f32 / 255.0,
            result.rgba[si * 4 + 1] as f32 / 255.0,
            result.rgba[si * 4 + 2] as f32 / 255.0,
        ];
        let model_a = stale_alpha[i];
        let ci = CorrectionInputs {
            matte_plate: plate_at(src_x(x), y),
            matte_min: matte_min[i],
            matte_max: matte_max[i],
        };

        let (rgb, a) = composite(plate, model_fg, model_a, &ci, &cp);
        alpha_out[i] = a;
        // Edge color bias: signed (R+B)/2 - G of the straight foreground over
        // semi-transparent pixels. Positive = magenta/purple fringe (the
        // legacy despill redistribution), negative = green fringe.
        if a > 0.02 && a < 0.98 {
            edge_magenta_sum += ((rgb[0] + rgb[2]) * 0.5 - rgb[1]) as f64;
            edge_magenta_n += 1;
        }

        let (x, y) = ((i % size) as u32, (i / size) as u32);
        alpha_img.put_pixel(x, y, image::Luma([(a * 255.0 + 0.5) as u8]));
        fg_img.put_pixel(x, y, image::Rgb([b(rgb[0]), b(rgb[1]), b(rgb[2])]));

        // Over a checkerboard, so semi-transparency is visible at a glance.
        let checker = if ((x / 32) + (y / 32)) % 2 == 0 { 0.25 } else { 0.6 };
        comp_img.put_pixel(
            x,
            y,
            image::Rgba([
                b(rgb[0] * a + checker * (1.0 - a)),
                b(rgb[1] * a + checker * (1.0 - a)),
                b(rgb[2] * a + checker * (1.0 - a)),
                255,
            ]),
        );
    }

    // ".keyed_" rather than "_": the ground-truth matte for tests/data/test_plate.png
    // is test_plate_alpha.png, and a "_alpha" suffix would silently overwrite the
    // very file the scoring below compares against.
    let stem = input.with_extension("");
    let out = |suffix: &str| PathBuf::from(format!("{}.keyed_{suffix}.png", stem.display()));
    for (path, saved) in [
        (out("alpha"), alpha_img.save(out("alpha"))),
        (out("fg"), fg_img.save(out("fg"))),
        (out("comp"), comp_img.save(out("comp"))),
    ] {
        saved?;
        println!("wrote {}", path.display());
    }

    if edge_magenta_n > 0 {
        println!(
            "edge color bias (semi-transparent px, +magenta/-green): {:+.4} ({} px)",
            edge_magenta_sum / edge_magenta_n as f64,
            edge_magenta_n
        );
    }

    if let Some(truth_path) = truth {
        score(&truth_path, &alpha_out, size)?;
    }
    Ok(())
}

fn b(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Reports where the matte disagrees with the truth, split by region — a mean
/// error over the whole frame is dominated by the flat interior and hides
/// exactly the edge behaviour that matters here.
fn score(truth_path: &Path, got: &[f32], size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Ground truth is authored at 512; compare at whatever the model ran at.
    let truth = image::imageops::resize(
        &ImageReader::open(truth_path)?.decode()?.to_luma8(),
        size as u32,
        size as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let mut solid = (0.0f64, 0usize);
    let mut empty = (0.0f64, 0usize);
    let mut edge = (0.0f64, 0usize);
    // Hard classification errors at a 0.5 threshold: false white is clear
    // backing shown as foreground; false black is solid subject cut away.
    let mut false_white = 0usize;
    let mut false_black = 0usize;

    for (i, t) in truth.pixels().enumerate() {
        let t = t[0] as f32 / 255.0;
        let d = (t - got[i]).abs() as f64;
        let bucket = if t > 0.98 {
            if got[i] < 0.5 {
                false_black += 1;
            }
            &mut solid
        } else if t < 0.02 {
            if got[i] > 0.5 {
                false_white += 1;
            }
            &mut empty
        } else {
            &mut edge
        };
        bucket.0 += d;
        bucket.1 += 1;
    }

    let mean = |(sum, n): (f64, usize)| if n == 0 { 0.0 } else { sum / n as f64 };
    println!("\nmean absolute alpha error vs ground truth:");
    println!("  solid interior : {:.4}  ({} px)", mean(solid), solid.1);
    println!("  clear backing  : {:.4}  ({} px)", mean(empty), empty.1);
    println!("  soft edges     : {:.4}  ({} px)", mean(edge), edge.1);
    println!("hard errors at 0.5 threshold:");
    println!(
        "  false white    : {} px ({:.2}% of clear backing)",
        false_white,
        100.0 * false_white as f64 / empty.1.max(1) as f64
    );
    println!(
        "  false black    : {} px ({:.2}% of solid interior)",
        false_black,
        100.0 * false_black as f64 / solid.1.max(1) as f64
    );
    Ok(())
}
