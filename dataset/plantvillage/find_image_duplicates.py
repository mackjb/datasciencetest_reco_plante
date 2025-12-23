#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find duplicate (and near-duplicate) images by content, not by filename.

- Recursively scans a root directory for image files
- Exact duplicates: grouped by (size, sha256) of raw file bytes
- Optional near-duplicates: grouped by perceptual hash (dHash) with Hamming threshold
- Outputs groups with >= min_group_size files
- Optional CSV report

Usage (exact only):
  python -u dataset/plantvillage/find_image_duplicates.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --report outputs/duplicates_color_exact.csv

Usage (with near-duplicates):
  python -u dataset/plantvillage/find_image_duplicates.py \
    --root "/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/color" \
    --perceptual --phash-threshold 6 \
    --report outputs/duplicates_color_both.csv

Notes:
- Perceptual duplicate detection uses dHash (64-bit) and groups using union-find within prefix buckets
- Large folders: near-duplicate detection may take longer; adjust --bucket-bits to trade coverage vs speed
"""

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass
class FileInfo:
    path: Path
    size: int
    sha256: Optional[str] = None
    dhash64: Optional[int] = None


def iter_image_files(root: Path) -> Iterable[Path]:
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


def dhash_64(path: Path) -> Optional[int]:
    """Return 64-bit difference hash (dHash) as int. None if PIL not available or read error.
    Algorithm: grayscale, resize to 9x8, compare adjacent pixels horizontally to produce 64-bit hash.
    """
    if not PIL_AVAILABLE:
        return None
    try:
        with Image.open(path) as im0:
            im = ImageOps.exif_transpose(im0).convert("L")  # grayscale
            im = im.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(im.getdata())
            # build 8 rows of 9 columns
            rows = [pixels[i * 9:(i + 1) * 9] for i in range(8)]
            bits = 0
            bitpos = 0
            for r in rows:
                for c in range(8):
                    bits <<= 1
                    bitpos += 1
                    if r[c] > r[c + 1]:
                        bits |= 1
            # ensure 64 bits
            if bitpos < 64:
                bits <<= (64 - bitpos)
            return bits & ((1 << 64) - 1)
    except Exception:
        return None


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def group_exact(files: List[FileInfo], min_group_size: int) -> List[List[FileInfo]]:
    by_key: Dict[Tuple[int, str], List[FileInfo]] = {}
    for fi in files:
        if fi.sha256 is None:
            fi.sha256 = sha256_of_file(fi.path)
        key = (fi.size, fi.sha256)
        by_key.setdefault(key, []).append(fi)
    return [grp for grp in by_key.values() if len(grp) >= min_group_size]


def group_perceptual(files: List[FileInfo], min_group_size: int, phash_threshold: int, bucket_bits: int) -> List[List[FileInfo]]:
    if not PIL_AVAILABLE:
        print("[WARN] PIL not available; skipping perceptual duplicate detection")
        return []
    # Compute dHash
    for fi in files:
        if fi.dhash64 is None:
            fi.dhash64 = dhash_64(fi.path)
    # Bucket by prefix bits to avoid O(n^2) across all
    buckets: Dict[int, List[int]] = {}
    for idx, fi in enumerate(files):
        if fi.dhash64 is None:
            continue
        prefix = fi.dhash64 >> (64 - bucket_bits)
        buckets.setdefault(prefix, []).append(idx)
    # For each bucket, union pairs within threshold
    uf = UnionFind(len(files))
    for indices in buckets.values():
        k = len(indices)
        for i in range(k):
            a_idx = indices[i]
            ha = files[a_idx].dhash64
            if ha is None:
                continue
            for j in range(i + 1, k):
                b_idx = indices[j]
                hb = files[b_idx].dhash64
                if hb is None:
                    continue
                if hamming_distance64(ha, hb) <= phash_threshold:
                    uf.union(a_idx, b_idx)
    # Collect groups
    groups: Dict[int, List[FileInfo]] = {}
    for i, fi in enumerate(files):
        if fi.dhash64 is None:
            continue
        root = uf.find(i)
        groups.setdefault(root, []).append(fi)
    return [grp for grp in groups.values() if len(grp) >= min_group_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Detect duplicate images by content (sha256) and optional perceptual near-duplicates (dHash)")
    default_root = Path(__file__).resolve().parents[1] / "plantvillage" / "data" / "plantvillage dataset" / "color"
    parser.add_argument("--root", type=str, default=str(default_root), help="Root directory to scan recursively")
    parser.add_argument("--perceptual", action="store_true", help="Also detect near-duplicates using dHash")
    parser.add_argument("--phash-threshold", type=int, default=6, help="Hamming distance threshold for dHash grouping (smaller=closer)")
    parser.add_argument("--bucket-bits", type=int, default=16, help="Prefix bits for bucketing dHash to limit pairwise comparisons")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum group size to report")
    parser.add_argument("--report", type=str, default=None, help="Optional CSV report path")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    print(f"[INFO] Scanning: {root}")
    # Collect file info
    files: List[FileInfo] = []
    for p in iter_image_files(root):
        try:
            size = p.stat().st_size
        except Exception:
            continue
        files.append(FileInfo(path=p, size=size))

    print(f"[INFO] Found {len(files)} image files")

    # Exact duplicates
    exact_groups = group_exact(files, min_group_size=args.min_group_size)
    print(f"[INFO] Exact duplicate groups: {len(exact_groups)}")

    # Perceptual near-duplicates
    perceptual_groups: List[List[FileInfo]] = []
    if args.perceptual:
        if not PIL_AVAILABLE:
            print("[WARN] PIL not available; cannot run perceptual duplicate detection.")
        else:
            perceptual_groups = group_perceptual(files, args.min_group_size, args.phash_threshold, args.bucket_bits)
            print(f"[INFO] Perceptual near-duplicate groups: {len(perceptual_groups)} (threshold={args.phash_threshold})")

    # Print summary to stdout
    gid = 1
    for grp in exact_groups:
        key_size = grp[0].size
        key_sha = grp[0].sha256
        print(f"\n== Exact Group #{gid} | size={key_size} bytes | sha256={key_sha} | {len(grp)} files ==")
        for fi in grp:
            print(f" - {fi.path}")
        gid += 1

    for grp in perceptual_groups:
        print(f"\n== Perceptual Group #{gid} | size~varies | {len(grp)} files ==")
        for fi in grp:
            print(f" - {fi.path}")
        gid += 1

    # Optional CSV
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group_id", "method", "size_bytes", "sha256", "dhash64_hex", "path"])
            gid = 1
            for grp in exact_groups:
                for fi in grp:
                    w.writerow([gid, "exact", fi.size, fi.sha256, "", str(fi.path)])
                gid += 1
            for grp in perceptual_groups:
                for fi in grp:
                    dh_hex = f"{fi.dhash64:016x}" if fi.dhash64 is not None else ""
                    w.writerow([gid, "perceptual", fi.size, fi.sha256 or "", dh_hex, str(fi.path)])
                gid += 1
        print(f"[INFO] CSV report written to: {out}")


if __name__ == "__main__":
    main()
