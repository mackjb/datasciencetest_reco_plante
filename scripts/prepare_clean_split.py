#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a clean classification dataset split (train/val/test) from a CSV keep-list
by creating a directory structure with symlinks or copies.

Output layout (default): dataset/plantvillage/clean_split/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
  class_index.json
  splits.json
  README.md

Usage example:
  python -u scripts/prepare_clean_split.py \
    --keep_list_csv dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv \
    --out dataset/plantvillage/clean_split --seed 42 --train 0.7 --val 0.15 --test 0.15 --link symlink
"""
import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def species_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    return parent.split("___")[0] if "___" in parent else parent


def stratified_split(paths, labels, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    paths = np.array(paths)
    labels = np.array(labels)

    # remove classes with <2 samples (cannot stratify)
    label_counts = Counter(labels.tolist())
    keep_mask = np.array([label_counts[l] >= 2 for l in labels])
    if not np.all(keep_mask):
        paths = paths[keep_mask]
        labels = labels[keep_mask]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trval_idx, te_idx = next(sss_test.split(paths, labels))
    trval_paths, trval_labels = paths[trval_idx], labels[trval_idx]
    te_paths, te_labels = paths[te_idx], labels[te_idx]

    val_rel = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    tr_idx_rel, va_idx_rel = next(sss_val.split(trval_paths, trval_labels))
    tr_paths, tr_labels = trval_paths[tr_idx_rel], trval_labels[tr_idx_rel]
    va_paths, va_labels = trval_paths[va_idx_rel], trval_labels[va_idx_rel]
    return (tr_paths.tolist(), tr_labels.tolist(), va_paths.tolist(), va_labels.tolist(), te_paths.tolist(), te_labels.tolist())


def main():
    ap = argparse.ArgumentParser(description="Prepare symlinked/copy split from keep-list CSV")
    ap.add_argument("--keep_list_csv", type=str, required=True, help="CSV with a 'path' column (and optional 'action=keep')")
    ap.add_argument("--out", type=str, default="dataset/plantvillage/clean_split", help="Output root directory for split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--link", type=str, choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--clear", action="store_true", help="If set, delete output directory before creating")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.clear and os.path.isdir(args.out):
        print(f"[INFO] Clearing output directory: {args.out}")
        shutil.rmtree(args.out)
        os.makedirs(args.out, exist_ok=True)

    print(f"[INFO] Loading keep-list CSV: {args.keep_list_csv}")
    df = pd.read_csv(args.keep_list_csv)
    if "path" not in df.columns:
        raise ValueError("CSV must contain a 'path' column")
    if "action" in df.columns:
        df = df[df["action"].str.lower() == "keep"]

    # Filter valid image files that exist on disk
    paths = [p for p in df["path"].astype(str).tolist() if is_image_file(p) and os.path.isfile(p)]
    if not paths:
        raise RuntimeError("No valid image paths to process after filtering.")

    # Labels from path
    labels_species = [species_from_path(p) for p in paths]
    species = sorted(set(labels_species))
    s2i = {s: i for i, s in enumerate(species)}

    labels_idx = [s2i[s] for s in labels_species]
    tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_split(paths, labels_idx, seed=args.seed, train_ratio=args.train, val_ratio=args.val, test_ratio=args.test)

    # Create tree and populate
    for split_name, split_paths in [("train", tr_p), ("val", va_p), ("test", te_p)]:
        for p in split_paths:
            cls = species_from_path(p)
            dst_dir = os.path.join(args.out, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(p))
            if os.path.exists(dst):
                continue
            try:
                if args.link == "symlink":
                    os.symlink(p, dst)
                else:
                    shutil.copy2(p, dst)
            except FileExistsError:
                pass

    # Save class index and splits for reference
    with open(os.path.join(args.out, "class_index.json"), "w", encoding="utf-8") as f:
        json.dump({i: s for s, i in s2i.items()}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "train_paths": tr_p,
            "val_paths": va_p,
            "test_paths": te_p,
            "classes": species,
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# Clean Split\n\nRoot: {args.out}\n\n- seed: {args.seed}\n- ratios: train={args.train}, val={args.val}, test={args.test}\n- link: {args.link}\n")

    print("[INFO] Done.")
    print(f"[INFO] Output created under: {args.out}")


if __name__ == "__main__":
    main()
