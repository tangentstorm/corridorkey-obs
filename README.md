# CorridorKey for OBS

A realtime OBS Studio filter that keys a green (or blue) screen with
[CorridorKey](https://github.com/nikopueringer/CorridorKey)'s neural unmixing
model, written in Rust.

> **No support.** This is shared as-is, and I don't provide help with it — please
> don't open issues asking for setup help. Windows users can use the installer
> from the [latest release](../../releases/tag/latest). For anything else (other platforms, other OBS
> versions, building it yourself), use the source code, or point an AI coding
> agent at this repository and let it figure it out.

Traditional chroma keys decide how opaque each pixel is. CorridorKey's model does
something harder: for every pixel — including motion blur, out-of-focus edges and
single strands of hair — it predicts both a linear alpha *and* the true
un-multiplied color of the foreground, as if the screen had never been there.
That's what makes edges composite cleanly instead of looking cut out.

## What this is, and what it isn't

Upstream CorridorKey is an offline VFX tool. It runs at 2048×2048 in PyTorch,
writes linear EXRs for Nuke/Fusion/Resolve, and gets its required alpha hint from
a separate heavyweight model (GVM wants roughly 80GB of VRAM).

None of that fits in a filter that has ~16ms per frame. This project keeps the
part worth keeping — the GreenFormer network — and rebuilds everything around it:

| | Upstream CorridorKey | This plugin |
|---|---|---|
| Runtime | PyTorch + `uv` | ONNX Runtime, statically linked |
| Resolution | 2048×2048 | 512×512, composited back at full res |
| Alpha hint | GVM / VideoMaMa / BiRefNet | built-in chroma key |
| Output | 16/32-bit linear EXR | 8-bit straight RGBA in the OBS pipeline |
| Latency | seconds per frame | matte ~100ms behind live |

**The output is not as good as running CorridorKey properly offline.** At 512×512
the model sees roughly a seventh of the detail it was trained to use at 2048, and
the matte is upsampled back to your output resolution. For finishing work, use the
real thing. For a live stream, this gets you edges that a chroma key cannot.

## Performance and the quality/speed setting

The **Quality / speed** setting picks the resolution the network runs at. Each is
a separate ONNX export; the plugin reads the resolution off whichever model it
loads and resizes its buffers to match, so switching is a runtime change.

Measured on an RTX 4050 Laptop (6GB) via DirectML with `examples/key_image`,
fp16, GPU otherwise idle:

| Setting | ms per frame | matte updates/sec |
|---|---|---|
| Best (512px) | ~100 | ~10 |
| Balanced (384px) | ~55 | ~18 |
| Fastest (256px) | ~20 | ~50 |

**Why this setting is the one that matters.** Your video always runs at full
framerate — the network is on a worker thread and the composite runs every frame
in the shader. What the resolution changes is how far the *matte* lags the
picture. At 512px on this GPU the matte is ~0.1s behind, and when you move your
arm quickly the old outline is still there for a moment. At 256px that drops to
~0.02s, which is close to imperceptible.

So: raise it if edges look soft, lower it if movement leaves outlines. On a
faster GPU, 512px lag shrinks enough that the trade mostly disappears.

Two other things worth knowing:

- **fp16 is free.** It scores the same as fp32 against ground truth (0.1049 vs
  0.1050 mean alpha error in the solid interior) and is ~1.4x faster, so the
  plugin prefers it and the export defaults to producing it.
- **On a laptop, adapter 0 is often the integrated GPU**, where this network is
  roughly 8x slower (~1250ms vs ~150ms measured here). ONNX Runtime picks
  sensibly by default, which is why **GPU device index** is -1 (auto) — but it is
  the first thing to check if inference looks absurd.

## Requirements

- Windows, OBS Studio 32.x (the plugin declares the 32.x module ABI and will be
  refused by a different major version — deliberately, since `obs_source_info`
  can change between them)
- A DX12 GPU. Tested on an RTX 4050 Laptop (6GB).
- Rust 1.82+ and Visual Studio Build Tools (for `lib.exe`), to build.
- Python with [uv](https://docs.astral.sh/uv/), to export the model once.

You do **not** need an obs-studio source checkout. `build.rs` reads the export
table out of your installed `obs.dll` and generates the import library itself.

## Install (prebuilt)

Download [`corridorkey-obs-setup.exe`](../../releases/download/latest/corridorkey-obs-setup.exe)
from the [latest release](../../releases/tag/latest) and run it with OBS closed.
It installs the plugin into `%ProgramData%\obs-studio\plugins\corridorkey-obs`.

**The installer does not include the models.** They derive from the upstream
CorridorKey checkpoints, which carry a non-commercial licence, so they aren't
redistributed here. Generate them with steps 1 and 2 of
[Build and install](#build-and-install) below, and copy the `.onnx` files into
`%ProgramData%\obs-studio\plugins\corridorkey-obs\data\models`. Until at least one
is there, the filter loads but passes video through untouched, and says so in its
properties panel.

## Build and install

```powershell
# 1. Get the upstream model definition (checkpoints download on first export)
git clone https://github.com/nikopueringer/CorridorKey.git ref/CorridorKey

# 2. Export the network to ONNX
cd tools/export
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe timm safetensors onnx onnxruntime huggingface_hub numpy pillow
# One export per resolution you want offered in the Quality setting (~136MB each)
foreach ($sz in 512, 384, 256) {
    .venv\Scripts\python.exe export_onnx.py --size $sz --color green --fp16
}
# Blue screen support is optional: same loop with --color blue
cd ../..

# 3. Build and install
.\install.ps1
```

Then restart OBS and add **CorridorKey (Neural Green Screen)** as a filter on your
camera source.

`install.ps1` installs to `%ProgramData%\obs-studio\plugins`, which is where OBS
for Windows actually looks. (Many guides say `%APPDATA%\obs-studio\plugins`; OBS's
`AddExtraModulePaths` calls `GetProgramDataPath` on Windows, so that path is
silently ignored.) Use `-IntoObsDir` to install into the OBS program folder
instead, from an elevated shell.

## Settings

| Setting | What it does |
|---|---|
| **Quality / speed** | 512 / 384 / 256px. The main lever on how far the matte lags the picture. See Performance. |
| **Screen color** | Green or blue. Selects the checkpoint *and* the despill channel; switching reloads the model. |
| **Hint: similarity** | How close a pixel's chroma must be to the screen color to seed the hint as background. Raise if your subject is being eaten, lower if the screen isn't detected. |
| **Hint: smoothness** | Width of the ramp between the two. |
| **Despill strength** | Removes screen color bounced onto the subject, preserving luminance. |
| **Full-res detail in solid areas** | At 1.0, the original full-resolution image is used wherever the matte is solid, keeping the model's prediction only for soft edges. Drop to 0 to see raw model output. |
| **Matte black / white point** | The usual matte tightening controls. |
| **Bypass** | Passes the source through untouched, skipping the network entirely rather than computing a matte and discarding it. |
| **GPU device index** | -1 (auto) is right on almost all machines. See Performance. |

The properties panel also shows a live status line: which model loaded, how many
milliseconds per frame it's taking, or why it isn't running.

## How it works

Every frame, on OBS's render thread, the filter uploads the newest finished matte
and composites at full resolution through `obs_source_process_filter_begin`/`_end`.
That is the only per-frame cost that scales with your output resolution, and it's
a handful of texture taps.

Separately, and only when the inference worker is actually free, it runs an
*analysis pass*: a second `process_filter_begin`/`_tech_end` cycle that draws
OBS's input texture through the effect's `Downscale` technique straight into a
512x512 render target, then stages that for CPU readback. The staged surface is
mapped on the **following** frame — mapping one you just wrote forces a CPU/GPU
sync and stalls the render thread — and the pixels go to the worker.

Two details are load-bearing:

- **The input comes from OBS, not from re-rendering the target.** Calling
  `obs_source_video_render` on the filter target from inside `video_render` looks
  reasonable and works for some sources, but yields a frame of pure zeros for
  others (an image source, for one). The chroma hint reads an all-black frame as
  "all foreground", so the network returns a fully opaque matte and the filter
  silently degrades into a plain despill. Letting OBS supply its own input
  texture is both correct and cheaper.
- **The analysis pass is gated on the worker being idle.** Inference takes ~10
  frames, so capturing and reading back every frame would burn GPU time producing
  pixels that get thrown away.

If a matte ever looks wrong, set `CORRIDORKEY_DUMP` to a file path before starting
OBS: the filter writes the raw BGRA frame it is actually feeding the model, which
settles in one step whether the fault is the capture or the network.

The alpha hint is the interesting substitution. The model requires a coarse mask
telling it which blob is the subject, and upstream spends an enormous model on
producing one. But for a green screen specifically, a classic chroma key *is* that
rough mask — it's precisely the "rough black-and-white mask" the network was
trained to refine. It costs well under a millisecond and needs no second network.
The hint keys on brightness-normalized chroma rather than raw chroma, so an
unevenly lit screen still reads uniformly (see `src/hint.rs`).

## Testing without OBS

```powershell
cargo run --release --example key_image -- tests/data/test_plate.png tests/data/test_plate_alpha.png
```

This runs the exact hint → inference → composite path the filter uses, writes
`*.keyed_alpha.png`, `*.keyed_fg.png` and `*.keyed_comp.png`, and scores the matte
against ground truth split by region (solid interior, clear backing, soft edges) —
a whole-frame mean would be dominated by the flat interior and hide exactly the
edge behaviour that matters.

`tools/make_test_plate.py` regenerates the synthetic plate: a subject with hair
strands, a motion-blurred wedge, an unevenly lit screen and green spill, composited
forward from a known matte so there *is* a ground truth to score against.

`cargo test` covers the hint generator and the despill/composite math (15 tests).
`data/effects/corridorkey.effect` mirrors `src/composite.rs` in HLSL; the tests in
that module are what pin the behaviour the shader has to match.

## Notes on the ONNX export

Two things in `tools/export/export_onnx.py` are load-bearing:

- **Static shapes.** Hiera's windowed attention reshapes on spatial dimensions, so
  a dynamic-axis export produces a graph that's either wrong or unoptimizable.
- **Rank-4 attention.** Hiera's mask-unit attention works on rank-5 tensors, and
  DirectML's GEMM only handles four dimensions — the graph loads fine and then
  dies on the first attention block with a bare `E_INVALIDARG`. The export folds
  the leading dimensions into a single batch axis, which is algebraically
  identical and free at runtime. `assert_no_high_rank_matmul` fails the export if
  this ever stops applying, so the problem surfaces at export rather than as a
  mystery at inference.

The export verifies its own output against PyTorch and refuses to write a model
that doesn't match.

## Licensing

Upstream CorridorKey is released under a variation of **CC BY-NC-SA 4.0 with
commercial restrictions**. That covers the model and the checkpoints this plugin
loads, so it applies to you as a user of this plugin:

- You may process images commercially — including streaming.
- You may **not** resell the tool or offer paid inference as a service without a
  written agreement with the author.

No checkpoint is redistributed here. `export_onnx.py` downloads it from the
author's HuggingFace repos on first run, and the derived ONNX file stays local.
If you redistribute an exported model, the upstream license travels with it.
Check the upstream repository for the current terms before doing anything
commercial with it.
