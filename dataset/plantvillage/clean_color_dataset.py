#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean PlantVillage color (non-segmented) dataset by:
- Keeping only one image for exact duplicate groups (by content, not filename) – keep the FIRST by lexicographic path
- Deleting images with too-low sharpness (variance of Laplacian) using a z-score threshold
- Writing a CSV listing ONLY the images to KEEP (not those to delete), and a console synthesis

SAFE BY DEFAULT: use --apply to actually delete files. Without --apply, it's a dry run.

Default CSV path: dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv

Usage (dry run):
  python -u dataset/plantvillage/clean_color_dataset.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --z-threshold 5 \
    --csv "dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv"

Apply deletions:
  python -u dataset/plantvillage/clean_color_dataset.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --z-threshold 5 \
    --csv "dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv" \
    --apply
"""

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ImgInfo:
    path: Path
    size: int
    sha256: Optional[str] = None
    sharpness: Optional[float] = None
    z: float = 0.0


def parse_args():
    here = Path(__file__).resolve()
    default_root = here.parent / "data" / "plantvillage dataset" / "color"
    default_csv = here.parent / "csv" / "deep_learning_clean_plantvillage_color.csv"

    p = argparse.ArgumentParser(description="Clean dataset: drop duplicates and low-sharpness images, write CSV of KEPT images")
    p.add_argument("--root", type=str, default=str(default_root), help="Root directory to clean recursively")
    p.add_argument("--z-threshold", type=float, default=5.0, help="Absolute z-score threshold for low sharpness (z <= -thresh → delete)")
    p.add_argument("--csv", type=str, default=str(default_csv), help="CSV output listing files to delete")
    p.add_argument("--apply", action="store_true", help="Actually delete files (otherwise dry-run)")
    p.add_argument("--progress-every", type=int, default=1000, help="Progress print frequency (files)")
    return p.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def variance_of_laplacian(arr: np.ndarray) -> float:
    # Discrete Laplacian with edge padding
    pad = np.pad(arr, 1, mode="edge")
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    center = pad[1:-1, 1:-1]
    lap = (up + down + left + right) - 4.0 * center
    return float(np.var(lap, dtype=np.float64))


def compute_sharpness(path: Path) -> float:
    with Image.open(path) as im0:
        im = ImageOps.exif_transpose(im0).convert("L")
        arr = np.asarray(im, dtype=np.float32)
    return variance_of_laplacian(arr)


def choose_keep_candidate_lex(infos: List[ImgInfo]) -> ImgInfo:
    # Keep FIRST by lexicographic path (deterministic)
    return sorted(infos, key=lambda x: str(x.path).lower())[0]


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    print(f"[INFO] Root: {root}")
    print(f"[INFO] CSV:  {csv_path}")
    print(f"[INFO] Mode: {'APPLY (delete files)' if args.apply else 'DRY-RUN (no deletion)'}")
    print(f"[INFO] Z-threshold (low sharpness removal): {args.z_threshold}")

    # Collect image infos
    infos: List[ImgInfo] = []
    for idx, p in enumerate(iter_image_paths(root), 1):
        try:
            size = p.stat().st_size
        except Exception:
            continue
        infos.append(ImgInfo(path=p, size=size))
        if idx % args.progress_every == 0:
            print(f"[INFO] Indexed {idx} files…")
    print(f"[INFO] Total files indexed: {len(infos)}")

    # Compute SHA-256 for duplicates (exact)
    for i, fi in enumerate(infos, 1):
        try:
            fi.sha256 = sha256_of_file(fi.path)
        except Exception as e:
            fi.sha256 = None
        if i % args.progress_every == 0:
            print(f"[INFO] Hashed {i}/{len(infos)} files…")

    # Compute sharpness for all
    scores: List[float] = []
    good_idx: List[int] = []
    for i, fi in enumerate(infos, 1):
        try:
            fi.sharpness = compute_sharpness(fi.path)
            scores.append(fi.sharpness)
            good_idx.append(i - 1)
        except Exception:
            fi.sharpness = None
        if i % args.progress_every == 0:
            print(f"[INFO] Sharpness {i}/{len(infos)} files…")

    # Compute z-scores
    z_thresh = float(args.z_threshold)
    if scores:
        s = np.asarray(scores, dtype=np.float64)
        mean = float(np.mean(s))
        std = float(np.std(s))
    else:
        mean = 0.0
        std = 1.0
    for i in good_idx:
        fi = infos[i]
        if std > 0:
            fi.z = (float(fi.sharpness) - mean) / std
        else:
            fi.z = 0.0

    # Low-sharpness deletions (z <= -z_thresh)
    low_sharp_set = set(
        fi.path for fi in infos if fi.sharpness is not None and fi.z <= -z_thresh
    )

    # Duplicate groups by (size, sha256)
    dup_groups: Dict[Tuple[int, str], List[ImgInfo]] = {}
    for fi in infos:
        if fi.sha256 is None:
            continue
        key = (fi.size, fi.sha256)
        dup_groups.setdefault(key, []).append(fi)

    # Determine KEPT images (exclude low-sharpness first, then keep first by path in each duplicate group)
    eligible: List[ImgInfo] = [fi for fi in infos if fi.path not in low_sharp_set]
    kept_paths = set()
    kept_rows: List[Tuple[Path, int, str, Optional[float], float, int, str]] = []
    # Columns: [path, size_bytes, sha256, sharpness, z, dup_group_size, keep_rule]

    # Helper: count group size among ALL files for info
    group_sizes_all: Dict[Tuple[int, str], int] = {k: len(v) for k, v in dup_groups.items()}

    # 1) Handle eligible files with missing sha256 as unique keeps
    for fi in eligible:
        if fi.sha256 is None:
            kept_paths.add(fi.path)
            kept_rows.append((fi.path, fi.size, "", fi.sharpness, fi.z, 1, "unique_nohash"))

    # 2) Group eligible files (with sha) by (size, sha256) and keep FIRST by path
    elig_by_key: Dict[Tuple[int, str], List[ImgInfo]] = {}
    for fi in eligible:
        if fi.sha256 is None:
            continue
        key = (fi.size, fi.sha256)
        elig_by_key.setdefault(key, []).append(fi)
    for key, group in elig_by_key.items():
        group_sorted = sorted(group, key=lambda x: str(x.path).lower())
        keep = choose_keep_candidate_lex(group_sorted)
        if keep.path not in kept_paths:
            kept_paths.add(keep.path)
            gsz_all = group_sizes_all.get(key, len(group))
            kept_rows.append((keep.path, keep.size, keep.sha256 or "", keep.sharpness, keep.z, gsz_all, "first_by_path" if gsz_all > 1 else "unique"))

    # 3) Also include eligible singleton files (with sha) that weren't added (group size 1)
    for fi in eligible:
        if fi.sha256 is None:
            continue
        key = (fi.size, fi.sha256)
        if group_sizes_all.get(key, 0) <= 1 and fi.path not in kept_paths:
            kept_paths.add(fi.path)
            kept_rows.append((fi.path, fi.size, fi.sha256 or "", fi.sharpness, fi.z, 1, "unique"))

    # Write CSV (images to KEEP only)
    kept_rows_sorted = sorted(kept_rows, key=lambda r: str(r[0]).lower())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "path", "size_bytes", "sha256", "sharpness_lapvar", "zscore", "dup_group_size", "keep_rule"])
        for path, size, sha, sharp, z, gsz, kr in kept_rows_sorted:
            w.writerow(["keep", str(path), size, sha, f"{(sharp or 0.0):.6f}", f"{z:.3f}", gsz, kr])

    # Apply deletions if requested
    deleted_cnt = 0
    errors_cnt = 0
    if args.apply:
        for reason, path, sha, sharp, z, gsz, kept in to_delete:
            try:
                os.remove(path)
                deleted_cnt += 1
            except Exception as e:
                print(f"[WARN] Failed to delete {path}: {e}")
                errors_cnt += 1

    # Synthesis
    total = len(infos)
    n_groups = sum(1 for g in dup_groups.values() if len(g) >= 2)
    n_low = len(low_sharp_set)
    kept_total = len(kept_rows_sorted)
    kept_dup = sum(1 for _, _, _, _, _, gsz, _ in kept_rows_sorted if gsz > 1)
    kept_unique = kept_total - kept_dup

    print("[INFO] Cleaning synthesis (KEEP list)")
    print(f" - Total files scanned: {total}")
    print(f" - Low-sharpness removed (z <= -{z_thresh}): {n_low}")
    print(f" - Duplicate groups (exact): {n_groups}")
    print(f" - Kept images (total): {kept_total}")
    print(f"   • from duplicate groups: {kept_dup}")
    print(f"   • unique: {kept_unique}")
    print(f" - CSV (kept list) written to: {csv_path}")
    if args.apply:
        print(f" - Deleted files (applied): {deleted_cnt}")
        print(f" - Deletion errors: {errors_cnt}")
    else:
        print(" - Dry-run: dataset unchanged. Re-run with --apply to delete non-kept files.")


if __name__ == "__main__":
    main()
