#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize PlantVillage color (non-segmented) images in-place.
- Traverse recursively a root directory (default: dataset/plantvillage/data/plantvillage dataset/color)
- Ensure images are 256x256, JPEG baseline, RGB (3 components)
- Overwrite modified files atomically
- Produce a CSV report listing modified files and changes

Usage:
  python -u dataset/plantvillage/normalize_color_images.py \
    --root "dataset/plantvillage/data/plantvillage dataset/color" \
    --size 256 256 \
    --quality 95 \
    --report "dataset/plantvillage/data/plantvillage dataset/color/_normalize_report.csv"

Notes:
- Only JPEG files are processed by default (extensions: .jpg/.jpeg/.JPG/.JPEG)
- If an image already matches (RGB, 256x256, JPEG), it is skipped unless --force-reencode is set
- Writes using progressive=False (baseline), subsampling=2 (4:2:0), quality configurable
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps

ALLOWED_JPEG_EXT = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize PlantVillage color images to 256x256 JPEG baseline RGB (in-place)")
    default_root = Path(__file__).resolve().parent / "data" / "plantvillage dataset" / "color"
    parser.add_argument("--root", type=str, default=str(default_root), help="Root directory to process")
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256), help="Target image size (W H)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--report", type=str, default=None, help="Path to write CSV report (default: <root>/_normalize_report.csv)")
    parser.add_argument("--force-reencode", action="store_true", help="Force re-encode even if already JPEG RGB 256x256")
    return parser.parse_args()


def is_jpeg_file(p: Path) -> bool:
    return p.suffix in ALLOWED_JPEG_EXT


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def process_image(path: Path, target_size: Tuple[int, int], quality: int, force_reencode: bool) -> Tuple[bool, List[str], dict]:
    """Process a single image; return (modified, reasons, info_dict).
    info_dict contains before/after metadata and sizes.
    """
    info = {
        "path": str(path),
        "before_w": None,
        "before_h": None,
        "before_mode": None,
        "before_format": None,
        "before_bytes": None,
        "after_w": None,
        "after_h": None,
        "after_mode": None,
        "after_format": None,
        "after_bytes": None,
        "error": None,
    }
    reasons: List[str] = []
    modified = False

    try:
        before_stat = path.stat()
        info["before_bytes"] = before_stat.st_size

        with Image.open(path) as im0:
            # Auto-apply EXIF orientation
            im = ImageOps.exif_transpose(im0)
            w, h = im.size
            info["before_w"], info["before_h"] = w, h
            info["before_mode"] = im.mode
            info["before_format"] = im0.format  # Use container format as detected

            # Normalize mode to RGB
            if im.mode not in ("RGB",):
                im = im.convert("RGB")
                modified = True
                reasons.append("convert_rgb")

            # Resize if needed
            if (w, h) != tuple(target_size):
                im = im.resize(tuple(target_size), Image.Resampling.LANCZOS)
                modified = True
                reasons.append("resize")

            # Re-encode decision
            if force_reencode or info["before_format"] != "JPEG":
                modified = True
                if info["before_format"] != "JPEG":
                    reasons.append("reencode_jpeg")

            if modified:
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                ensure_parent(tmp_path)
                # Save as JPEG baseline
                im.save(
                    tmp_path,
                    format="JPEG",
                    quality=int(quality),
                    optimize=False,
                    progressive=False,
                    subsampling=2,  # 4:2:0
                )
                # Atomic replace
                os.replace(tmp_path, path)

        # After stats
        with Image.open(path) as im_after:
            info["after_w"], info["after_h"] = im_after.size
            info["after_mode"] = im_after.mode
            info["after_format"] = im_after.format
        info["after_bytes"] = path.stat().st_size

    except Exception as e:
        info["error"] = str(e)
        modified = False  # Do not claim modified on failure
        reasons.append("error")

    return modified, reasons, info


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    target_size = tuple(args.size)
    report_path = Path(args.report) if args.report else root / "_normalize_report.csv"

    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Target size: {target_size}")
    print(f"[INFO] JPEG quality: {args.quality}")
    print(f"[INFO] Report: {report_path}")

    ensure_parent(report_path)

    processed = 0
    modified_cnt = 0
    skipped_cnt = 0
    error_cnt = 0

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "before_w","before_h","before_mode","before_format","before_bytes",
            "after_w","after_h","after_mode","after_format","after_bytes",
            "modified","reasons","error",
        ])

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if not is_jpeg_file(p):
                # Skip non-JPEG files silently (dataset is expected JPEG)
                continue

            processed += 1
            modified, reasons, info = process_image(p, target_size, args.quality, args.force_reencode)

            if info["error"]:
                error_cnt += 1
            elif modified:
                modified_cnt += 1
            else:
                skipped_cnt += 1

            writer.writerow([
                os.path.relpath(info["path"], start=str(root)),
                info["before_w"], info["before_h"], info["before_mode"], info["before_format"], info["before_bytes"],
                info["after_w"], info["after_h"], info["after_mode"], info["after_format"], info["after_bytes"],
                str(modified), ";".join(reasons), info["error"],
            ])

            if processed % 500 == 0:
                print(f"[INFO] Processed {processed} files… (modified={modified_cnt}, skipped={skipped_cnt}, errors={error_cnt})")

    print("[INFO] Done.")
    print(f"[STATS] Processed: {processed}")
    print(f"[STATS] Modified: {modified_cnt}")
    print(f"[STATS] Skipped:  {skipped_cnt}")
    print(f"[STATS] Errors:   {error_cnt}")
    print(f"[INFO] Report written to: {report_path}")


if __name__ == "__main__":
    main()
