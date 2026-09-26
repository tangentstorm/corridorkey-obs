"""Synthesizes a 512x512 green screen test plate with a known ground-truth matte.

Real footage would be a better test, but it comes without a reference matte. Here
the compositing math is run forward — a subject over green, with soft edges, motion
blur and green spill — so the keyer's output can be scored against the alpha that
produced the image.

Writes test_plate.png (the green screen frame) and test_plate_alpha.png (truth).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
OUT = Path(__file__).resolve().parents[1] / "tests" / "data"

SCREEN = np.array([0.05, 0.62, 0.10])  # a realistic, not-fully-saturated green


def box_blur(img: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Separable box blur, repeated to approximate a Gaussian.

    Only used to shape the spill falloff, so exactness doesn't matter — and it
    keeps this script to numpy + Pillow.
    """
    out = img.astype(np.float32)
    k = 2 * radius + 1
    for _ in range(passes):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius, radius)
            padded = np.pad(out, pad, mode="edge")
            cs = np.cumsum(padded, axis=axis)
            cs = np.concatenate(
                [np.zeros_like(np.take(cs, [0], axis=axis)), cs], axis=axis
            )
            hi = np.take(cs, range(k, cs.shape[axis]), axis=axis)
            lo = np.take(cs, range(0, cs.shape[axis] - k), axis=axis)
            out = (hi - lo) / k
    return out


def build() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)

    # --- ground truth alpha ---
    alpha = np.zeros((SIZE, SIZE), np.float32)

    # Torso: a hard-edged rounded shape, antialiased over ~1.5px.
    body = np.sqrt(((xx - 256) / 130) ** 2 + ((yy - 340) / 190) ** 2)
    alpha = np.maximum(alpha, np.clip((1.04 - body) / 0.03, 0, 1))

    # Head.
    head = np.sqrt(((xx - 256) / 78) ** 2 + ((yy - 165) / 88) ** 2)
    alpha = np.maximum(alpha, np.clip((1.03 - head) / 0.03, 0, 1))

    # Hair strands: thin, partially transparent, the case that separates a real
    # keyer from a threshold.
    for i in range(44):
        t = np.linspace(0, 1, 260)
        ang = -math.pi / 2 + (i / 43 - 0.5) * 2.5
        sx = 256 + 72 * math.cos(ang)
        sy = 165 + 82 * math.sin(ang)
        px = sx + t * 95 * math.cos(ang) + 20 * np.sin(t * 7 + i)
        py = sy + t * 95 * math.sin(ang) - 12 * t * t
        width = 1.1 + 0.9 * (i % 3)
        opacity = 0.35 + 0.5 * ((i * 7) % 5) / 4
        for x0, y0, fade in zip(px, py, np.linspace(1.0, 0.25, len(t))):
            x0i, y0i = int(round(x0)), int(round(y0))
            r = int(math.ceil(width)) + 1
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = x0i + dx, y0i + dy
                    if 0 <= x < SIZE and 0 <= y < SIZE:
                        d = math.hypot(x - x0, y - y0)
                        a = opacity * fade * max(0.0, 1.0 - d / width)
                        alpha[y, x] = max(alpha[y, x], a)

    # Motion-blurred arm: a wide, uniformly semi-transparent wedge.
    arm = (np.abs((xx - 150) * 0.6 + (yy - 380) * 0.3) < 46) & (yy > 300) & (yy < 470)
    alpha[arm] = np.maximum(alpha[arm], 0.45)

    # --- foreground color ---
    fg = np.zeros((SIZE, SIZE, 3), np.float32)
    fg[..., 0] = 0.55 + 0.25 * np.sin(xx / 40) * np.cos(yy / 55)
    fg[..., 1] = 0.36 + 0.12 * np.cos(yy / 30)
    fg[..., 2] = 0.30 + 0.10 * np.sin(xx / 25)
    fg = np.clip(fg, 0, 1)

    # --- screen, unevenly lit: the thing a naive absolute-chroma key trips on ---
    fall = 0.55 + 0.45 * np.exp(-(((xx - 180) / 300) ** 2 + ((yy - 150) / 320) ** 2))
    bg = SCREEN[None, None, :] * fall[..., None]

    # Composite, then add spill: screen light bouncing onto the subject near its
    # edges, which is exactly what CorridorKey is meant to unmix.
    a = alpha[..., None]
    comp = fg * a + bg * (1 - a)

    rim = box_blur(alpha, 9) * alpha
    comp += (rim * 0.22)[..., None] * np.array([0.0, 1.0, 0.15])
    comp = np.clip(comp, 0, 1)

    return comp, alpha


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comp, alpha = build()
    Image.fromarray((comp * 255 + 0.5).astype(np.uint8)).save(OUT / "test_plate.png")
    Image.fromarray((alpha * 255 + 0.5).astype(np.uint8), "L").save(OUT / "test_plate_alpha.png")
    print(f"wrote {OUT / 'test_plate.png'} and {OUT / 'test_plate_alpha.png'}")


if __name__ == "__main__":
    main()
