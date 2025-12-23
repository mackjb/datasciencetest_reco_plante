#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a cleaned CSV with image features for PlantVillage segmented dataset.

Input CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv
Output CSV (default): dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv

Implements the following features per image:
- Shape: aire, périmètre, circularité, excentricité, aspect_ratio
- Color (RGB): mean_R, mean_G, mean_B, std_R, std_G, std_B
- HSV means: mean_H, mean_S, mean_V
- Texture (GLCM): contrast, energy, homogeneity, dissimilarité, Correlation
- Contour density: contour_density
- Hu moments: hu_1..hu_7 (log-scaled)
- Sharpness: netteté (variance of Laplacian)
- HOG: hog_mean, hog_std, hog_entropy
- FFT: fft_energy, fft_entropy, fft_low_freq_power, fft_high_freq_power

Separation of concerns: each feature family has its own function, orchestrated by extract_all_features.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

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


# ------------------------- Feature extractors -------------------------

def extract_shape_features(gray_img: np.ndarray, binary_thresh: Optional[int] = None) -> Dict[str, float]:
    """
    Shape features from the largest contour.
    Returns: aire, périmètre, circularité, excentricité, aspect_ratio
    """
    if gray_img.ndim != 2:
        raise ValueError("extract_shape_features expects a grayscale image")

    # Binarization: Otsu if threshold not provided
    if binary_thresh is None:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_img, binary_thresh, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Return NaNs to signal missing contour
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
    circularite = (4.0 * math.pi * aire) / (perim ** 2) if perim > 0 else 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    # Eccentricity via regionprops on labeled mask if available
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
    """
    Color features on RGB channels: mean and std per channel.
    Expects RGB uint8 image.
    """
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
    """
    Mean HSV components.
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "mean_H": float(np.mean(h)),
        "mean_S": float(np.mean(s)),
        "mean_V": float(np.mean(v)),
    }


def extract_texture_features(gray_img: np.ndarray, levels: int = 256) -> Dict[str, float]:
    """
    Texture features via GLCM for distance=1, angle=0.
    Returns: contrast, energy, homogeneity, dissimilarité, Correlation
    """
    if gray_img.ndim != 2:
        raise ValueError("extract_texture_features expects a grayscale image")
    # skimage expects values in [0, levels-1]
    if levels != 256:
        # Quantize if different levels requested
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
    """
    Canny edge pixel ratio to total pixels.
    """
    edges = cv2.Canny(gray_img, 100, 200)
    density = float(np.sum(edges > 0) / edges.size)
    return {"contour_density": density}


def extract_hu_moments(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Log-scaled Hu moments (7 values): hu_1..hu_7
    """
    m = cv2.moments(gray_img)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f"hu_{i+1}": float(hu_log[i]) for i in range(7)}


def extract_sharpness(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Variance of Laplacian as a sharpness measure.
    """
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    return {"netteté": float(np.var(lap))}


def extract_hog_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    HOG features reduced to summary stats: mean, std, entropy over descriptor.
    """
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
    """
    2D FFT magnitude power spectrum summary statistics.
    Returns: fft_energy, fft_entropy, fft_low_freq_power, fft_high_freq_power
    """
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

    # Radial split: low vs high frequency by radius threshold
    h, w = gray_img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = np.max(r)
    r_thresh = 0.25 * r_max  # 25% of max radius considered low-frequency
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
    """
    Orchestrates extraction over all feature families.
    Expects an RGB image (uint8). Will produce grayscale for relevant extractors.
    """
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


# ------------------------- I/O and pipeline -------------------------

def load_image_rgb(path: str, target_size: Optional[tuple[int, int]] = (224, 224)) -> Optional[np.ndarray]:
    """
    Load an image as RGB uint8. Optionally resize.
    Returns None if the image cannot be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size)
        return np.array(img)
    except Exception:
        return None


def process_dataframe(
    df: pd.DataFrame,
    image_col: str = "Image_Path",
    target_size: Optional[tuple[int, int]] = (224, 224),
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Iterate rows, compute features, and return a concatenated DataFrame with original columns + features.
    """
    records: List[Dict[str, Any]] = []

    it = df.itertuples(index=False)
    if limit is not None:
        it = list(it)[:limit]

    for row in tqdm(it, total=(limit if limit is not None else len(df)), desc="Extracting features"):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        img_path = row_dict.get(image_col)
        rgb = load_image_rgb(img_path, target_size=target_size)
        if rgb is None:
            # keep row but with NaN features
            feats = {k: np.nan for k in [
                "aire", "périmètre", "circularité", "excentricité", "aspect_ratio",
                "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
                "mean_H", "mean_S", "mean_V",
                "contrast", "energy", "homogeneity", "dissimilarité", "Correlation",
                "contour_density",
                "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
                "netteté",
                "hog_mean", "hog_std", "hog_entropy",
                "fft_energy", "fft_entropy", "fft_low_freq_power", "fft_high_freq_power",
            ]}
        else:
            feats = extract_all_features(rgb)
        records.append({**row_dict, **feats})

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description="Generate CSV with image features for PlantVillage segmented dataset")
    default_input = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv"))
    default_output = str(Path("dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv"))
    parser.add_argument("--input", "--input_csv", dest="input_csv", default=default_input, help="Path to input CSV")
    parser.add_argument("--output", "--output_csv", dest="output_csv", default=default_output, help="Path to output CSV")
    parser.add_argument("--image-col", dest="image_col", default="Image_Path", help="Column name with image paths")
    parser.add_argument("--target-size", dest="target_size", default="224,224", help="Resize images to WxH, or 'none'")
    parser.add_argument("--limit", dest="limit", type=int, default=None, help="Process only first N rows (debug)")

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
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    # Type normalization for expected columns if present
    for bcol in ["Est_Saine", "is_black"]:
        if bcol in df.columns:
            # Convert strings 'True'/'False' to int 1/0
            df[bcol] = df[bcol].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(df[bcol]).astype(str)
            df[bcol] = df[bcol].map({'1': 1, '0': 0}).fillna(0).astype(int)

    df_features = process_dataframe(df, image_col=args.image_col, target_size=target_size, limit=args.limit)

    print(f"Writing output CSV: {output_csv}")
    df_features.to_csv(output_csv, index=False, encoding='utf-8')
    print("Done.")


if __name__ == "__main__":
    main()
