#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a balanced CSV with real augmented images and features for the PlantVillage segmented dataset.

- Input CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv
- Output CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features_with_sampling.csv
- Augmented images saved to: dataset/plantvillage/data/clean_plantvillage_dataset/<Class_Name>/aug_*.jpg

Strategy:
- Define class_name = f"{nom_plante}___{nom_maladie}"
- Compute per-class target nt = int(0.7 * median + 0.3 * max)
- Undersample classes above nt by random sampling
- Oversample classes below nt by generating augmented images using Albumentations
- Compute features on augmented images using the same feature extractors as generate_clean_with_feature_csv.py

Notes:
- We avoid RandomResizedCrop: we keep the leaf shape by using Affine transforms and mild perspective.
- We compute md5 for augmented files.
- We preserve original feature values for kept rows; we compute features for augmented images.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

# skimage imports with fallback for different versions
try:
    from skimage.feature import graycomatrix, graycoprops, hog  # type: ignore
except Exception:  # pragma: no cover
    try:
        from skimage.feature.texture import graycomatrix, graycoprops  # type: ignore
        from skimage.feature import hog  # type: ignore
    except Exception as e:  # pragma: no cover
        raise e

try:
    from skimage.measure import label, regionprops
except Exception:  # pragma: no cover
    label = None
    regionprops = None


# ------------------------- Utilities -------------------------

def file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    m = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            m.update(chunk)
    return m.hexdigest()


def load_image_rgb(path: str, target_size: Optional[Tuple[int, int]] = (224, 224)) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size)
        return np.array(img)
    except Exception:
        return None


# ------------------------- Feature extractors (copied from generate_clean_with_feature_csv.py) -------------------------

def extract_shape_features(gray_img: np.ndarray, binary_thresh: Optional[int] = None) -> Dict[str, float]:
    if gray_img.ndim != 2:
        raise ValueError("extract_shape_features expects a grayscale image")
    if binary_thresh is None:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_img, binary_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "aire": np.nan,
            "périmètre": np.nan,
            "circularité": np.nan,
            "excentricité": np.nan,
            "aspect_ratio": np.nan,
        }
    cnt = max(contours, key=cv2.contourArea)
    aire = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularite = (4.0 * np.pi * aire) / (perim ** 2) if perim > 0 else 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    excentricite = 0.0
    if label is not None and regionprops is not None:
        lbl = label(binary > 0)
        props = regionprops(lbl)
        if props:
            largest_region = max(props, key=lambda p: p.area)
            excentricite = float(largest_region.eccentricity)
    return {
        "aire": aire,
        "périmètre": perim,
        "circularité": circularite,
        "excentricité": excentricite,
        "aspect_ratio": float(aspect_ratio),
    }


def extract_color_features(rgb_img: np.ndarray) -> Dict[str, float]:
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("extract_color_features expects an RGB image")
    R, G, B = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    return {
        "mean_R": float(np.mean(R)),
        "mean_G": float(np.mean(G)),
        "mean_B": float(np.mean(B)),
        "std_R": float(np.std(R)),
        "std_G": float(np.std(G)),
        "std_B": float(np.std(B)),
    }


def extract_hsv_features(rgb_img: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "mean_H": float(np.mean(h)),
        "mean_S": float(np.mean(s)),
        "mean_V": float(np.mean(v)),
    }


def extract_texture_features(gray_img: np.ndarray, levels: int = 256) -> Dict[str, float]:
    if gray_img.ndim != 2:
        raise ValueError("extract_texture_features expects a grayscale image")
    if levels != 256:
        gray = np.floor(gray_img.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    else:
        gray = gray_img
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    dissimilarite = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "dissimilarité": dissimilarite,
        "Correlation": correlation,
    }


def extract_contour_density(gray_img: np.ndarray) -> Dict[str, float]:
    edges = cv2.Canny(gray_img, 100, 200)
    density = float(np.sum(edges > 0) / edges.size)
    return {"contour_density": density}


def extract_hu_moments(gray_img: np.ndarray) -> Dict[str, float]:
    m = cv2.moments(gray_img)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f"hu_{i+1}": float(hu_log[i]) for i in range(7)}


def extract_sharpness(gray_img: np.ndarray) -> Dict[str, float]:
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    return {"netteté": float(np.var(lap))}


def extract_hog_features(gray_img: np.ndarray) -> Dict[str, float]:
    vec = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True,
    )
    vec = vec.astype(np.float64)
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    p = np.abs(vec)
    s = p.sum()
    if s <= 0:
        entropy = 0.0
    else:
        p = p / s
        entropy = float(-(p * (np.log2(p + 1e-12))).sum())
    return {"hog_mean": mean, "hog_std": std, "hog_entropy": entropy}


def extract_fft_features(gray_img: np.ndarray) -> Dict[str, float]:
    f = np.fft.fft2(gray_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2
    energy = float(power.sum())
    psum = power.sum()
    if psum <= 0:
        entropy = 0.0
    else:
        p = power / psum
        entropy = float(-(p * (np.log2(p + 1e-12))).sum())
    h, w = gray_img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = np.max(r)
    r_thresh = 0.25 * r_max
    low_mask = r <= r_thresh
    high_mask = ~low_mask
    low_power = float(power[low_mask].sum())
    high_power = float(power[high_mask].sum())
    return {
        "fft_energy": energy,
        "fft_entropy": entropy,
        "fft_low_freq_power": low_power,
        "fft_high_freq_power": high_power,
    }


def extract_all_features(rgb_img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    feats: Dict[str, float] = {}
    feats.update(extract_shape_features(gray))
    feats.update(extract_color_features(rgb_img))
    feats.update(extract_hsv_features(rgb_img))
    feats.update(extract_texture_features(gray))
    feats.update(extract_contour_density(gray))
    feats.update(extract_hu_moments(gray))
    feats.update(extract_sharpness(gray))
    feats.update(extract_hog_features(gray))
    feats.update(extract_fft_features(gray))
    return feats


# ------------------------- Augmentation -------------------------

def build_augmenter() -> A.Compose:
    """Albumentations pipeline: geometric + photometric, mild to preserve morphology."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.10), rotate=(-20, 20), shear=(-5, 5), p=0.9),
        A.Perspective(scale=(0.02, 0.05), p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2),
    ])


# ------------------------- Main processing -------------------------

def compute_target_count(counts: pd.Series) -> int:
    median = int(np.median(counts.values)) if len(counts) > 0 else 0
    maxc = int(np.max(counts.values)) if len(counts) > 0 else 0
    target = int(0.7 * median + 0.3 * maxc)
    return max(target, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate balanced CSV with augmented images and features (PlantVillage)")
    default_input = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv"))
    default_output = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features_with_sampling.csv"))
    default_img_base = str(Path("dataset/plantvillage/data/clean_plantvillage_dataset"))
    parser.add_argument("--input", dest="input_csv", default=default_input, help="Path to input CSV with features")
    parser.add_argument("--output", dest="output_csv", default=default_output, help="Path to output CSV")
    parser.add_argument("--image-base-dir", dest="image_base_dir", default=default_img_base, help="Base dir to save augmented images")
    parser.add_argument("--target-size", dest="target_size", default="224,224", help="Resize images to WxH (for features/aug)")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.target_size and isinstance(args.target_size, str) and args.target_size.lower() != "none":
        try:
            w, h = map(int, args.target_size.split(','))
            target_size = (w, h)
        except Exception:
            raise ValueError("--target-size must be 'none' or 'W,H'")
    else:
        target_size = None

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    image_base_dir = Path(args.image_base_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = ["nom_plante", "nom_maladie", "Image_Path"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in input CSV: {c}")

    # Build class name
    class_name = df["nom_plante"].astype(str) + "___" + df["nom_maladie"].astype(str)
    df = df.copy()
    df["__class_name__"] = class_name

    # Compute target per-class count
    counts = df["__class_name__"].value_counts()
    target = compute_target_count(counts)
    print("Class counts summary:")
    print(counts.describe())
    print(f"Target per-class count: {target}")

    rng = np.random.default_rng(args.random_state)
    augmenter = build_augmenter()

    kept_parts: List[pd.DataFrame] = []
    augmented_records: List[Dict[str, Any]] = []

    # Determine base and feature columns from input
    base_cols = [c for c in [
        "nom_plante", "nom_maladie", "Est_Saine", "Image_Path", "width_img", "height_img", "is_black", "md5"
    ] if c in df.columns]

    feature_cols = [
        "aire", "périmètre", "circularité", "excentricité", "aspect_ratio",
        "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
        "mean_H", "mean_S", "mean_V",
        "contrast", "energy", "homogeneity", "dissimilarité", "Correlation",
        "contour_density",
        "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
        "netteté",
        "hog_mean", "hog_std", "hog_entropy",
        "fft_energy", "fft_entropy", "fft_low_freq_power", "fft_high_freq_power",
    ]
    # Keep any additional non-feature columns to preserve
    other_cols = [c for c in df.columns if c not in base_cols + feature_cols + ["__class_name__"]]

    for cls, df_cls in df.groupby("__class_name__"):
        n = len(df_cls)
        class_dir = image_base_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        if n > target:
            kept = df_cls.sample(n=target, random_state=args.random_state)
            kept_parts.append(kept)
        elif n < target:
            kept_parts.append(df_cls)
            to_generate = target - n
            print(f"Class {cls}: {n} -> generating {to_generate} augmented samples")

            # Cycle through base rows for augmentation
            rows = list(df_cls.itertuples(index=False))
            cycle_iter = itertools.cycle(rows)

            gen = 0
            pbar = tqdm(total=to_generate, desc=f"Augmenting {cls}", leave=False)
            while gen < to_generate:
                base_row = next(cycle_iter)
                if hasattr(base_row, "_asdict"):
                    row_dict = base_row._asdict()
                else:
                    row_dict = dict(zip(df_cls.columns, base_row))

                img_path = row_dict.get("Image_Path")
                rgb = load_image_rgb(img_path, target_size=target_size)
                if rgb is None:
                    # skip this one
                    continue

                # Apply augmentation
                transformed = augmenter(image=rgb)
                aug_img = transformed["image"]

                # Save augmented image
                base_md5 = row_dict.get("md5")
                fname = f"aug_{(base_md5 or 'x')[:8]}_{gen}_{uuid4().hex[:8]}.jpg"
                save_path = class_dir / fname
                Image.fromarray(aug_img).save(str(save_path), quality=95)

                # Compute metadata
                h, w = aug_img.shape[:2]
                md5_hex = file_md5(str(save_path))

                # Compute features on augmented image
                feats = extract_all_features(aug_img)

                # Build new row
                new_row: Dict[str, Any] = {}
                # base columns copied then updated
                for c in base_cols:
                    if c in row_dict:
                        new_row[c] = row_dict[c]
                new_row["Image_Path"] = str(save_path.resolve())
                new_row["width_img"] = w
                new_row["height_img"] = h
                new_row["md5"] = md5_hex

                # preserve other columns if any
                for c in other_cols:
                    new_row[c] = row_dict.get(c, np.nan)

                # add features
                for c in feature_cols:
                    new_row[c] = feats.get(c, np.nan)

                augmented_records.append(new_row)
                gen += 1
                pbar.update(1)
            pbar.close()
        else:
            kept_parts.append(df_cls)

    df_kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame(columns=df.columns)
    df_aug = pd.DataFrame.from_records(augmented_records) if augmented_records else pd.DataFrame(columns=df.columns)

    # Ensure consistent columns and order
    for c in df.columns:
        if c not in df_aug.columns:
            df_aug[c] = np.nan
    df_aug = df_aug[df.columns]

    df_out = pd.concat([df_kept, df_aug], ignore_index=True)
    df_out = df_out.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    print(f"Writing output CSV: {output_csv}")
    df_out.drop(columns=["__class_name__"], errors="ignore").to_csv(output_csv, index=False, encoding='utf-8')
    print("Done.")


if __name__ == "__main__":
    main()
