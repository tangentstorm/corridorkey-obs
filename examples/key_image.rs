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

use corridorkey_obs::composite::{composite, CompositeParams};
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
    let cp = CompositeParams::default();
    let mut alpha_img = GrayImage::new(size as u32, size as u32);
    let mut fg_img = RgbImage::new(size as u32, size as u32);
    let mut comp_img = RgbaImage::new(size as u32, size as u32);
    let mut alpha_out = vec![0.0f32; size * size];

    for i in 0..size * size {
        let px = img.get_pixel((i % size) as u32, (i / size) as u32);
        let plate = [
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        ];
        let model_fg = [
            result.rgba[i * 4] as f32 / 255.0,
            result.rgba[i * 4 + 1] as f32 / 255.0,
            result.rgba[i * 4 + 2] as f32 / 255.0,
        ];
        let model_a = result.rgba[i * 4 + 3] as f32 / 255.0;

        let (rgb, a) = composite(plate, model_fg, model_a, &cp);
        alpha_out[i] = a;

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

    for (i, t) in truth.pixels().enumerate() {
        let t = t[0] as f32 / 255.0;
        let d = (t - got[i]).abs() as f64;
        let bucket = if t > 0.98 {
            &mut solid
        } else if t < 0.02 {
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
    Ok(())
}
