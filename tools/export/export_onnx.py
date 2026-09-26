"""Export CorridorKey's GreenFormer to ONNX for realtime inference in the OBS plugin.

The upstream engine runs at 2048x2048 and writes EXRs from Python. For a live OBS
filter we need a single static-shape graph that a Rust process can feed at 30-60fps,
so this script:

  1. Builds GreenFormer at a realtime-sized `--size` (Hiera pos-embeds are resized
     from the checkpoint, exactly as inference_engine._load_model does).
  2. Folds the ImageNet normalization into the graph, so the plugin can hand over
     plain sRGB [0,1] RGB + hint and stay out of the preprocessing business.
  3. Exports static shapes. Hiera's windowed attention reshapes on spatial dims, so
     dynamic axes produce a graph that is either wrong or unoptimizable; a fixed
     size is also what lets DirectML/TensorRT pick fast kernels.

Usage:
    uv run python export_onnx.py --size 512 --color green
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The upstream repo is vendored under ref/CorridorKey; import its model definition
# rather than restating a 300-line architecture we would then have to keep in sync.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORRIDORKEY_DIR = REPO_ROOT / "ref" / "CorridorKey"
if not CORRIDORKEY_DIR.is_dir():
    sys.exit(f"CorridorKey checkout not found at {CORRIDORKEY_DIR}.\n"
             f"Run: git clone https://github.com/nikopueringer/CorridorKey.git \"{CORRIDORKEY_DIR}\"")
sys.path.insert(0, str(CORRIDORKEY_DIR))


def _load_greenformer():
    """Import model_transformer.py by path.

    `import CorridorKeyModule...` would execute the package __init__, which pulls in
    the whole inference engine (cv2, OpenEXR, ffmpeg helpers) that we don't need and
    don't want to install just to trace a graph. The model file itself only needs
    torch + timm.
    """
    mod_path = CORRIDORKEY_DIR / "CorridorKeyModule" / "core" / "model_transformer.py"
    spec = importlib.util.spec_from_file_location("ck_model_transformer", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GreenFormer


GreenFormer = _load_greenformer()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_REPOS = {
    "green": ("nikopueringer/CorridorKey_v1.0", "CorridorKey_v1.0.safetensors"),
    "blue": ("nikopueringer/CorridorKeyBlue_1.0", "CorridorKeyBlue_1.0.safetensors"),
}


def ensure_checkpoint(color: str) -> Path:
    """Fetch the ~300MB safetensors checkpoint from HuggingFace if we don't have it."""
    repo_id, filename = HF_REPOS[color]
    dest = CORRIDORKEY_DIR / "CorridorKeyModule" / "checkpoints" / filename
    if dest.is_file():
        print(f"[ckpt] using existing {dest}")
        return dest

    from huggingface_hub import hf_hub_download

    print(f"[ckpt] downloading {filename} from {repo_id} (~300MB)...")
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    print(f"[ckpt] saved to {dest}")
    return dest


def load_state_dict(path: Path, model: nn.Module) -> dict:
    """Load weights, resizing Hiera position embeddings to our target resolution.

    Mirrors CorridorKeyModule.inference_engine.IntelligentEngine._load_model.
    """
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    model_state = model.state_dict()
    out: dict[str, torch.Tensor] = {}

    for k, v in raw.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]

        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            n_src, c = v.shape[1], v.shape[2]
            n_dst = model_state[k].shape[1]
            grid_src, grid_dst = int(math.sqrt(n_src)), int(math.sqrt(n_dst))
            print(f"[ckpt] resizing {k}: {grid_src}x{grid_src} -> {grid_dst}x{grid_dst}")
            v_img = v.permute(0, 2, 1).view(1, c, grid_src, grid_src)
            v_img = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
            v = v_img.flatten(2).transpose(1, 2)

        out[k] = v

    missing, unexpected = model.load_state_dict(out, strict=False)
    if missing:
        print(f"[ckpt] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[ckpt] WARNING unexpected keys: {unexpected}")
    return out


class _Rank4Matmul:
    """Context manager that flattens >4D attention into batched 3D matmuls.

    Hiera's mask-unit attention works on rank-5 tensors,
    [batch, heads, windows, tokens, head_dim]. ONNX export decomposes that into
    MatMuls that keep all five dimensions, and DirectML's GEMM only handles four:
    the graph loads fine and then dies on the first attention block with a bare
    E_INVALIDARG out of MLOperatorAuthorImpl.

    Folding the leading dimensions into a single batch axis is algebraically
    identical — a batched matmul doesn't care how its batch is shaped — and free
    at runtime, since the reshapes are metadata on contiguous tensors.

    Both attention paths are patched: `F.scaled_dot_product_attention`, which is
    what timm actually calls when `fused_attn` is on, and `Tensor.__matmul__` for
    the explicit fallback path.
    """

    def __enter__(self):
        self._sdpa = F.scaled_dot_product_attention
        self._matmul = torch.Tensor.__matmul__
        orig_sdpa, orig_matmul = self._sdpa, self._matmul

        def flat(t):
            return t.reshape(-1, t.shape[-2], t.shape[-1])

        def sdpa(q, k, v, *args, **kwargs):
            if q.dim() > 4:
                lead = q.shape[:-2]
                out = orig_sdpa(flat(q), flat(k), flat(v), *args, **kwargs)
                return out.reshape(*lead, out.shape[-2], out.shape[-1])
            return orig_sdpa(q, k, v, *args, **kwargs)

        def matmul(a, b):
            if isinstance(b, torch.Tensor) and a.dim() == b.dim() and a.dim() > 4:
                lead = a.shape[:-2]
                out = orig_matmul(flat(a), flat(b))
                return out.reshape(*lead, out.shape[-2], out.shape[-1])
            return orig_matmul(a, b)

        F.scaled_dot_product_attention = sdpa
        torch.Tensor.__matmul__ = matmul
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._sdpa
        torch.Tensor.__matmul__ = self._matmul
        return False


def assert_no_high_rank_matmul(path: Path) -> None:
    """Fail the export if any MatMul still takes rank>4 inputs.

    Without this the model exports and loads fine, and only falls over at the
    first inference on a DirectML device — a long way from the cause.
    """
    import onnx
    from onnx import shape_inference

    model = shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=False)
    known = {
        v.name: v
        for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    offenders = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for name in node.input:
            vi = known.get(name)
            if vi is not None and len(vi.type.tensor_type.shape.dim) > 4:
                offenders.append(node.name)
                break
    if offenders:
        sys.exit(
            f"{len(offenders)} MatMul node(s) still take rank>4 inputs, which DirectML "
            f"cannot execute (first: {offenders[0]}). The _Rank4Matmul patch did not apply."
        )
    print("[verify] no rank>4 MatMul inputs  [OK]")


class ExportWrapper(nn.Module):
    """Normalization + GreenFormer + tuple output, so the plugin feeds raw sRGB.

    Inputs:
        rgb  [1, 3, S, S] sRGB in [0, 1]
        hint [1, 1, S, S] coarse alpha in [0, 1] (linear)
    Outputs:
        alpha [1, 1, S, S] linear alpha
        fg    [1, 3, S, S] straight (un-premultiplied) foreground, sRGB
    """

    def __init__(self, model: GreenFormer) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor, hint: torch.Tensor):
        x = torch.cat([(rgb - self.mean) / self.std, hint], dim=1)
        out = self.model(x)
        return out["alpha"], out["fg"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=512,
                    help="Square inference resolution. Must be a multiple of 64 for Hiera. Default 512.")
    ap.add_argument("--color", choices=sorted(HF_REPOS), default="green")
    ap.add_argument("--no-refiner", action="store_true",
                    help="Drop the full-res CNN refiner. ~35%% faster, softer edges.")
    ap.add_argument("--fp16", action="store_true",
                    help="Export half-precision weights. Roughly 2-3x faster on DirectML and "
                         "CUDA (tensor cores); inputs and outputs stay float32.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.size % 64 != 0:
        sys.exit(f"--size must be a multiple of 64 (Hiera downsamples by 32 and the "
                 f"decoder works at /4); got {args.size}")

    ckpt = ensure_checkpoint(args.color)

    print(f"[model] building GreenFormer at {args.size}x{args.size} (refiner={not args.no_refiner})")
    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=args.size,
        use_refiner=not args.no_refiner,
    )
    load_state_dict(ckpt, model)
    model.eval()

    wrapper = ExportWrapper(model).eval()

    rgb = torch.rand(1, 3, args.size, args.size)
    hint = torch.rand(1, 1, args.size, args.size)

    # Reference values come from the patched model too, so the ONNX check below
    # verifies the export, not the reshape rewrite; the rewrite is verified by
    # construction (a batched matmul is invariant to how the batch is shaped).
    with _Rank4Matmul(), torch.inference_mode():
        ref_alpha, ref_fg = wrapper(rgb, hint)
    print(f"[model] torch output OK: alpha{tuple(ref_alpha.shape)} fg{tuple(ref_fg.shape)}")

    suffix = "_fp16" if args.fp16 else ""
    out_path = args.out or (
        REPO_ROOT / "models" / f"corridorkey_{args.color}_{args.size}{suffix}.onnx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[onnx] exporting opset {args.opset} -> {out_path}")
    with _Rank4Matmul():
        torch.onnx.export(
            wrapper,
            (rgb, hint),
            str(out_path),
            input_names=["rgb", "hint"],
            output_names=["alpha", "fg"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    if args.fp16:
        to_fp16(out_path)

    assert_no_high_rank_matmul(out_path)
    # fp16 accumulates visible rounding; the bar is "same picture", not bit-equal.
    verify(out_path, rgb, hint, ref_alpha, ref_fg, tol=3e-2 if args.fp16 else 2e-3)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[done] {out_path}  ({size_mb:.1f} MB)")


def to_fp16(path: Path) -> None:
    """Convert the saved graph's weights and arithmetic to float16, in place.

    Done on the ONNX graph rather than by calling `.half()` on the torch model:
    exporting is normally run on a CPU-only install, and torch's CPU half kernels
    are missing or glacially slow for several ops in this network.

    `keep_io_types` leaves the graph's inputs and outputs float32, so the plugin
    doesn't need to know which precision it loaded.
    """
    from onnxconverter_common import float16
    import onnx

    print("[fp16] converting graph to half precision")
    model = onnx.load(str(path))
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        # Normalization constants and the refiner's 10x output scaling stay in
        # fp32 range comfortably, but the softmax denominators in attention do
        # not; disabling op blocking wholesale invites NaNs, so keep the default
        # blocked list.
        disable_shape_infer=False,
    )
    _retarget_internal_casts(model)
    onnx.save(model, str(path), save_as_external_data=False)
    print(f"[fp16] saved ({path.stat().st_size / (1024 * 1024):.1f} MB)")


def _retarget_internal_casts(model) -> None:
    """Point the graph's own `Cast`-to-float nodes at float16 instead.

    The ONNX export of scaled-dot-product attention emits explicit `Cast` nodes
    with `to=FLOAT`. onnxconverter_common rewrites tensor types around them but
    leaves those attributes alone, so the converted model fails to load with
    "Type (tensor(float16)) ... does not match expected type (tensor(float))".

    Only the torch-emitted casts are touched. onnxconverter_common inserts its own
    casts to bridge float32 into and out of blocked ops (Resize, and the graph's
    own inputs and outputs), and those are already correct — flipping them breaks
    the model just as thoroughly, in the other direction. The two are told apart
    by name: torch names its nodes ".../Cast[_N]", the converter appends
    "_input_cast0" / "_output_cast0" / "_cast_to_...".
    """
    import onnx

    def is_torch_cast(name: str) -> bool:
        leaf = name.rsplit("/", 1)[-1]
        return leaf == "Cast" or leaf.startswith("Cast_")

    retargeted = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not is_torch_cast(node.name):
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                attr.i = onnx.TensorProto.FLOAT16
                retargeted += 1
    print(f"[fp16] retargeted {retargeted} internal Cast node(s) to float16")


def verify(path: Path, rgb, hint, ref_alpha, ref_fg, tol: float = 2e-3) -> None:
    """Check the exported graph against PyTorch. A silently-wrong export is the
    main risk with Hiera's reshape-heavy attention, so this is not optional."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got_alpha, got_fg = sess.run(None, {"rgb": rgb.numpy(), "hint": hint.numpy()})

    for name, ref, got in (("alpha", ref_alpha, got_alpha), ("fg", ref_fg, got_fg)):
        err = float(np.abs(ref.numpy() - got).max())
        status = "OK" if err < tol else "MISMATCH"
        print(f"[verify] {name}: max abs diff {err:.3e} (tol {tol:.0e})  [{status}]")
        if err >= tol:
            sys.exit(f"ONNX export does not match PyTorch for '{name}'. Refusing to ship this model.")


if __name__ == "__main__":
    main()
