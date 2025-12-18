#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Cell Painting TIFF images to normalized PNGs.

This script is extracted/derived from the conversion logic in cp_pipeline_v3:
- robust per-channel percentile normalization (p1–p99)
- TIFF reading via OpenCV (fallback to tifffile if available)
- output folder structure is configurable

Typical use cases:
- preparing a lightweight PNG dataset for downstream tiling / CNN training
- reproducible preprocessing step for the thesis repository

Example:
  python preprocessing/convert_tiffs_to_pngs.py \
    --in_root  /scratch-shared/$USER/cellpainting/images \
    --out_root /scratch-shared/$USER/cellpainting/preprocessed_png \
    --pattern  "*.tif*" \
    --max_files 0
"""

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import cv2

try:
    import tifffile as tiff
except Exception:
    tiff = None


def robust_norm(img: np.ndarray) -> np.ndarray:
    """
    Normalize each channel to [0,1] using percentiles (1,99), then output 8-bit RGB.
    Mirrors the cp_pipeline_v3 behavior.
    """
    if img.ndim == 2:
        img = img[..., None]

    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):
        ch = img[:, :, c]
        finite = np.isfinite(ch)
        if finite.any():
            p1, p99 = np.percentile(ch[finite], [1, 99])
        else:
            p1, p99 = 0.0, 1.0

        if p99 > p1:
            out[:, :, c] = np.clip((ch - p1) / (p99 - p1), 0, 1)
        else:
            out[:, :, c] = 0.0

    C = out.shape[2]
    if C >= 3:
        rgb = out[:, :, :3]
    elif C == 2:
        rgb = np.concatenate([out, out[:, :, :1]], axis=2)
    else:
        rgb = np.repeat(out, 3, axis=2)

    return (rgb * 255).astype(np.uint8)


def imread_any(path: str):
    """
    Try OpenCV first; if it fails and tifffile is available, fallback to tifffile.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    if tiff is not None:
        try:
            arr = tiff.imread(path)
            # If TIFF is (C,H,W) with small C, move channels last.
            if arr.ndim == 3 and arr.shape[0] < 10:
                arr = np.moveaxis(arr, 0, -1)
            return arr
        except Exception:
            return None

    return None


def safe_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    """
    Preserve relative structure under out_root and change extension to .png.
    """
    rel = in_path.relative_to(in_root)
    rel_str = str(rel)

    # sanitize filename: whitespace -> underscore
    rel_str = re.sub(r"\s+", "_", rel_str)

    # change extension to .png
    rel_str = re.sub(r"\.tiff?$", ".png", rel_str, flags=re.IGNORECASE)
    rel_str = re.sub(r"\.tif$", ".png", rel_str, flags=re.IGNORECASE)

    return out_root / rel_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root folder containing TIFFs.")
    ap.add_argument("--out_root", required=True, help="Output root folder for PNGs.")
    ap.add_argument("--pattern", default="**/*.tif*", help="Glob pattern under in_root.")
    ap.add_argument("--max_files", type=int, default=0,
                    help="0 = no limit; otherwise stop after N files.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip conversion if output PNG already exists.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_root / args.pattern), recursive=True))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"[ERROR] No files found under {in_root} with pattern {args.pattern}")

    n_ok = 0
    n_fail = 0

    for i, fp in enumerate(files, 1):
        in_path = Path(fp)
        out_path = safe_out_path(in_path, in_root, out_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and out_path.exists():
            continue

        img = imread_any(str(in_path))
        if img is None:
            n_fail += 1
            print(f"[WARN] Could not read: {in_path}")
            continue

        rgb = robust_norm(img)
        ok = cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            n_fail += 1
            print(f"[WARN] Could not write: {out_path}")
            continue

        n_ok += 1
        if i % 200 == 0:
            print(f"[INFO] Converted {i}/{len(files)} files...")

    print(f"[DONE] Wrote PNGs: {n_ok} | Failed: {n_fail} | Out: {out_root}")


if __name__ == "__main__":
    main()
