"""Verify the enum constants in src/obs.rs against the real OBS headers.

src/obs.rs is hand-written FFI, which means every `pub const NAME: c_int = N` is
a value someone typed from a header. Getting one wrong is not a compile error and
usually is not a crash either - it is a silent behavioural bug:

  OBS_COMBO_TYPE_LIST as 1 instead of 2 means EDITABLE, and OBS then rejects the
  int-format list by returning NULL, so the property never appears in the UI.

  OBS_TEXT_INFO as 2 instead of 3 means MULTILINE, and a one-line status label
  renders as a large editable text box.

Both of those shipped. This script exists so the next one does not.

Usage:
    python tools/check_obs_constants.py [--obs-version 32.2.2]

Exits non-zero on any mismatch, so it can gate CI.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/obsproject/obs-studio/{ver}/libobs/{path}"

# Which headers to read, and which enums to take from each. Constants in obs.rs
# whose names appear in none of these are skipped - #defines and bit flags are
# checked separately below.
ENUM_SOURCES = {
    "obs-properties.h": ["obs_combo_type", "obs_combo_format", "obs_text_type", "obs_path_type"],
    "obs-source.h": ["obs_source_type", "obs_icon_type", "obs_media_state"],
    "graphics/graphics.h": ["gs_color_format", "gs_zstencil_format", "gs_cull_mode"],
    "obs.h": ["obs_allow_direct_render"],
}

# The log levels live in an anonymous enum, which the enum parser cannot name.
ANON_ENUM_SOURCES = {
    "util/base.h": ["LOG_ERROR", "LOG_WARNING", "LOG_INFO", "LOG_DEBUG"],
}

# Plain #defines worth pinning too, since they are just as easy to mistype.
DEFINE_SOURCES = {
    "obs-source.h": [
        "OBS_SOURCE_VIDEO", "OBS_SOURCE_AUDIO", "OBS_SOURCE_ASYNC",
        "OBS_SOURCE_CUSTOM_DRAW", "OBS_SOURCE_SRGB",
    ],
    "graphics/graphics.h": [
        "GS_BUILD_MIPMAPS", "GS_DYNAMIC", "GS_RENDER_TARGET", "GS_CLEAR_COLOR",
    ],
}


def fetch(version: str, path: str) -> str:
    url = RAW.format(ver=version, path=path)
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", "replace")


def parse_enums(src: str, wanted: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for enum in wanted:
        m = re.search(r"enum\s+%s\s*\{(.*?)\}" % re.escape(enum), src, re.S)
        if not m:
            continue
        value = 0
        for part in m.group(1).split(","):
            part = re.sub(r"/\*.*?\*/", "", part, flags=re.S)
            part = re.sub(r"//.*", "", part).strip()
            if not part:
                continue
            if "=" in part:
                name, raw = part.split("=", 1)
                name = name.strip()
                try:
                    value = int(raw.strip(), 0)
                except ValueError:
                    continue  # computed initialiser; skip rather than guess
            else:
                name = part.split()[0]
            out[name] = value
            value += 1
    return out


def parse_anon_enum(src: str, wanted: list[str]) -> dict[str, int]:
    """Pull `NAME = 123` members out of the file regardless of enum name.

    The log levels are declared in an anonymous enum, so they cannot be found by
    enum name; they are explicitly valued, so a direct match is enough.
    """
    out: dict[str, int] = {}
    for name in wanted:
        m = re.search(r"%s\s*=\s*(\d+)" % re.escape(name), src)
        if m:
            out[name] = int(m.group(1))
    return out


def parse_defines(src: str, wanted: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in wanted:
        m = re.search(r"#define\s+%s\s+(.+)" % re.escape(name), src)
        if not m:
            continue
        expr = m.group(1).split("/*")[0].split("//")[0].strip()
        shift = re.fullmatch(r"\(?\s*1\s*<<\s*(\d+)\s*\)?", expr)
        if shift:
            out[name] = 1 << int(shift.group(1))
            continue
        try:
            out[name] = int(expr.strip("()"), 0)
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-version", default="32.2.2")
    args = ap.parse_args()

    print(f"checking src/obs.rs against obs-studio {args.obs_version} headers")

    expected: dict[str, int] = {}
    for path, enums in ENUM_SOURCES.items():
        expected.update(parse_enums(fetch(args.obs_version, path), enums))
    for path, defines in DEFINE_SOURCES.items():
        expected.update(parse_defines(fetch(args.obs_version, path), defines))
    for path, names in ANON_ENUM_SOURCES.items():
        expected.update(parse_anon_enum(fetch(args.obs_version, path), names))

    rust = (REPO / "src" / "obs.rs").read_text(encoding="utf-8")
    declared = {
        name: int(val)
        for name, val in re.findall(r"pub const (\w+):\s*(?:c_int|u32|i32)\s*=\s*(\d+)\s*;", rust)
    }
    # Bit flags are written as shifts in obs.rs; evaluate those too.
    for name, shift in re.findall(r"pub const (\w+):\s*u32\s*=\s*1\s*<<\s*(\d+)\s*;", rust):
        declared[name] = 1 << int(shift)

    checked = mismatched = 0
    for name, value in sorted(declared.items()):
        if name not in expected:
            continue
        checked += 1
        if expected[name] != value:
            mismatched += 1
            print(f"  MISMATCH {name}: obs.rs has {value}, header says {expected[name]}")

    unchecked = sorted(n for n in declared if n not in expected)
    print(f"  {checked} checked, {mismatched} mismatched, {len(unchecked)} not found in headers")
    if unchecked:
        print("  (not found, verify by hand if you add more: " + ", ".join(unchecked) + ")")

    if mismatched:
        print("\nFAILED: a wrong enum value here is a silent UI or rendering bug, not a crash.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
