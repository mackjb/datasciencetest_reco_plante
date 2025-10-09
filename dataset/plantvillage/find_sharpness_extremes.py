#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan image sharpness and detect extremes.

- Recursively scans a root directory for images
- Computes a sharpness score per image (variance of discrete Laplacian)
- Computes population statistics (mean, std) and z-scores
- Flags images whose absolute z-score >= threshold (default 5)
- Optionally computes robust z-scores using MAD and flags extremes with the same threshold
- Writes a CSV report and prints a short summary

Usage (standard z-score, threshold 5):
  python -u dataset/plantvillage/find_sharpness_extremes.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --z-threshold 5 \
    --report outputs/sharpness_extremes_color.csv

Usage (robust z-score via MAD):
  python -u dataset/plantvillage/find_sharpness_extremes.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --z-threshold 5 --robust \
    --report outputs/sharpness_extremes_color_robust.csv

Notes:
- Sharpness metric: variance of discrete Laplacian on grayscale image
- Does not require OpenCV or SciPy; uses Pillow and NumPy only
- Images are NOT modified by this script; it only reports
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


@dataclass
class Row:
    path: str
    width: int
    height: int
    bytes: int
    sharpness: float
    z: float
    z_robust: float
    extreme_flag: str  # none | z | robust | both
    direction: str     # low | high | none


def parse_args():
    parser = argparse.ArgumentParser(description="Detect sharpness extremes (variance of Laplacian) with z-score threshold")
    default_root = Path(__file__).resolve().parents[1] / "plantvillage" / "data" / "plantvillage dataset" / "color"
    parser.add_argument("--root", type=str, default=str(default_root), help="Root directory to scan recursively")
    parser.add_argument("--z-threshold", type=float, default=5.0, help="Absolute z-score threshold for extremes")
    parser.add_argument("--robust", action="store_true", help="Use robust z-score (MAD) in addition to standard z-score")
    parser.add_argument("--report", type=str, default=None, help="CSV report path (default: <root>/_sharpness_extremes.csv)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of images to process (for testing)")
    return parser.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def variance_of_laplacian(arr: np.ndarray) -> float:
    """Compute variance of a simple discrete Laplacian using edge padding.
    arr must be 2D grayscale float32/float64.
    """
    # Pad by replication to keep size
    pad = np.pad(arr, 1, mode="edge")
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    center = pad[1:-1, 1:-1]
    lap = (up + down + left + right) - 4.0 * center
    # Use float64 variance for numerical stability
    return float(np.var(lap, dtype=np.float64))


def compute_sharpness(path: Path) -> Tuple[int, int, int, float]:
    # Returns (width, height, bytes, sharpness)
    st = path.stat()
    with Image.open(path) as im0:
        im = ImageOps.exif_transpose(im0).convert("L")  # grayscale
        w, h = im.size
        arr = np.asarray(im, dtype=np.float32)
    score = variance_of_laplacian(arr)
    return w, h, st.st_size, score


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    report_path = Path(args.report) if args.report else (root / "_sharpness_extremes.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning: {root}")
    paths: List[Path] = []
    for p in iter_image_paths(root):
        paths.append(p)
        if args.limit and len(paths) >= args.limit:
            break
    print(f"[INFO] Found {len(paths)} images to process")

    rows: List[Row] = []
    scores: List[float] = []

    for idx, p in enumerate(paths, 1):
        try:
            w, h, nbytes, s = compute_sharpness(p)
        except Exception as e:
            # Skip unreadable files; log as empty row with sharpness NaN
            print(f"[WARN] Failed to process {p}: {e}")
            continue
        scores.append(s)
        rows.append(Row(
            path=str(p), width=w, height=h, bytes=nbytes,
            sharpness=s, z=0.0, z_robust=0.0, extreme_flag="none", direction="none"
        ))
        if idx % 1000 == 0:
            print(f"[INFO] Processed {idx}/{len(paths)} images…")

    if not rows:
        print("[WARN] No images processed; aborting")
        return

    # Stats for z-scores
    scores_np = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores_np))
    std = float(np.std(scores_np))

    med = float(np.median(scores_np))
    mad = float(np.median(np.abs(scores_np - med)))
    # Constant to approximate std from MAD under normality
    MAD_CONST = 0.6745

    # Compute z-scores and flags
    threshold = float(args.z_threshold)
    n_extreme = 0

    for r in rows:
        # Standard z
        if std > 0:
            r.z = (r.sharpness - mean) / std
        else:
            r.z = 0.0
        # Robust z
        if mad > 0:
            r.z_robust = (MAD_CONST * (r.sharpness - med)) / mad
        else:
            r.z_robust = 0.0

        z_flag = abs(r.z) >= threshold
        rz_flag = args.robust and abs(r.z_robust) >= threshold

        if z_flag and rz_flag:
            r.extreme_flag = "both"
        elif z_flag:
            r.extreme_flag = "z"
        elif rz_flag:
            r.extreme_flag = "robust"
        else:
            r.extreme_flag = "none"

        if r.extreme_flag != "none":
            n_extreme += 1
            r.direction = "low" if (r.z if not args.robust else r.z_robust) < 0 else "high"
        else:
            r.direction = "none"

    # Write CSV
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "width", "height", "bytes",
            "sharpness_lapvar", "zscore", "robust_zscore",
            "extreme_flag", "direction"
        ])
        for r in rows:
            w.writerow([r.path, r.width, r.height, r.bytes, f"{r.sharpness:.6f}", f"{r.z:.3f}", f"{r.z_robust:.3f}", r.extreme_flag, r.direction])

    # Print summary and top examples
    print("[INFO] Done.")
    print(f"[STATS] Images processed: {len(rows)}")
    print(f"[STATS] z-threshold: {threshold} (robust={args.robust})")
    print(f"[STATS] Mean sharpness: {mean:.6f}, Std: {std:.6f}, Median: {med:.6f}, MAD: {mad:.6f}")
    print(f"[STATS] Extremes detected: {n_extreme}")

    # Show top 5 lowest/highest by z-score (or robust)
    key = (lambda r: r.z_robust) if args.robust else (lambda r: r.z)
    lows = sorted(rows, key=key)[:5]
    highs = sorted(rows, key=key, reverse=True)[:5]

    print("\n[LOWEST 5]")
    for r in lows:
        print(f" {r.z_robust if args.robust else r.z:+.2f} | {r.sharpness:.3f} | {r.path}")
    print("\n[HIGHEST 5]")
    for r in highs:
        print(f" {r.z_robust if args.robust else r.z:+.2f} | {r.sharpness:.3f} | {r.path}")

    print(f"[INFO] Report written to: {report_path}")


if __name__ == "__main__":
    main()
