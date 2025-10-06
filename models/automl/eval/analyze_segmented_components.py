#!/usr/bin/env python3
"""
Analyse les images segmentées (fond noir) pour compter le nombre de gros éléments (composants connectés)
par image et lister celles qui dépassent un seuil (ex: > 3 gros éléments).

Technique:
- Chargement en niveaux de gris
- Seuillage simple (pixels > gray_threshold) pour obtenir un masque binaire du foreground
- ConnectedComponentsWithStats pour récupérer l'aire de chaque composant (hors fond)
- "Gros" composant = aire >= max(min_area_pixels, min_area_ratio * (H*W))
- Comptage des gros composants et marquage des images au-delà d'un seuil

Entrées:
--src: dossier racine des images (par défaut: dataset/plantvillage/data/plantvillage dataset/segmented)
--out: CSV de sortie (par défaut: dataset/plantvillage/csv/segmentation_components_report.csv)
--gray-threshold: seuil de luminance pour binaire (par défaut: 5)
--min-area-ratio: ratio d'aire minimale d'un gros composant (par défaut: 0.01 => 1% des pixels)
--min-area-pixels: seuil absolu en pixels (prime sur ratio si > 0)
--flag-threshold: nombre de gros composants au-delà duquel on marque l'image (par défaut: 3)
--max-workers: parallélisation

Sortie CSV colonnes:
['classe','image_path','width','height','num_components_all','num_large_components',
 'large_areas','threshold_pixels','flagged']
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

# Import racine projet
import sys
THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from helpers.helpers import PROJECT_ROOT


def analyze_image_components(
    img_path: Path,
    gray_threshold: int,
    min_area_ratio: float,
    min_area_pixels: int | None,
) -> dict | None:
    try:
        # Lecture en niveaux de gris
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        h, w = gray.shape[:2]

        # Binarisation foreground: pixels > gray_threshold
        _, binary = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)

        # Composants connectés + stats
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # stats: [label, CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA]

        # Exclure background (label 0)
        areas = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            areas.append(area)

        total_pixels = h * w
        thr_pix = int(min_area_ratio * total_pixels)
        if min_area_pixels is not None and min_area_pixels > 0:
            thr_pix = max(thr_pix, int(min_area_pixels))

        large_areas = [a for a in areas if a >= thr_pix]
        num_large = len(large_areas)
        rec = {
            'image_path': str(img_path),
            'width': int(w),
            'height': int(h),
            'num_components_all': int(len(areas)),
            'num_large_components': int(num_large),
            'large_areas': large_areas,
            'threshold_pixels': int(thr_pix),
        }
        return rec
    except Exception:
        return None


def walk_images(src_root: Path) -> list[Path]:
    allowed = ('.jpg', '.jpeg', '.png')
    files: list[Path] = []
    for dirpath, _dirs, fnames in os.walk(src_root):
        for f in fnames:
            if f.lower().endswith(allowed):
                files.append(Path(dirpath) / f)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=str(PROJECT_ROOT / 'dataset' / 'plantvillage' / 'data' / 'plantvillage dataset' / 'segmented'))
    parser.add_argument('--out', type=str, default=str(PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'segmentation_components_report.csv'))
    parser.add_argument('--gray-threshold', type=int, default=5)
    parser.add_argument('--min-area-ratio', type=float, default=0.01)
    parser.add_argument('--min-area-pixels', type=int, default=0)
    parser.add_argument('--flag-threshold', type=int, default=3)
    parser.add_argument('--max-workers', type=int, default=0)
    args = parser.parse_args()

    src_root = Path(args.src)
    out_csv = Path(args.out)
    gray_threshold = args.gray_threshold
    min_area_ratio = float(args.min_area_ratio)
    min_area_pixels = int(args.min_area_pixels) if args.min_area_pixels and args.min_area_pixels > 0 else None
    flag_threshold = int(args.flag_threshold)

    if not src_root.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {src_root}")

    files = walk_images(src_root)
    total = len(files)
    print(f"Source: {src_root}")
    print(f"Fichiers trouvés: {total}")
    print(f"gray_threshold={gray_threshold}, min_area_ratio={min_area_ratio}, min_area_pixels={min_area_pixels}")

    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else min(32, (os.cpu_count() or 4) * 2)
    print(f"max_workers: {max_workers}")

    t0 = time.perf_counter()
    records: list[dict] = []

    def worker(p: Path):
        rec = analyze_image_components(p, gray_threshold, min_area_ratio, min_area_pixels)
        if rec is None:
            return None
        # classe = premier dossier sous racine
        try:
            rel = p.relative_to(src_root)
            classe = rel.parts[0] if len(rel.parts) > 1 else 'Unknown'
        except Exception:
            classe = 'Unknown'
        rec['classe'] = classe
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, p) for p in files]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df['flagged'] = df['num_large_components'] > flag_threshold
    else:
        df['flagged'] = []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    flagged_df = df[df['flagged']] if not df.empty else pd.DataFrame()
    t1 = time.perf_counter()

    print("--- Résumé ---")
    print(f"Total images analysées: {len(df)} / {total}")
    print(f"Images > {flag_threshold} gros éléments: {len(flagged_df)}")
    print(f"CSV écrit: {out_csv}")
    print(f"⏱️ Temps de traitement: {t1 - t0:.2f}s (~{(t1 - t0)/60:.2f} min)")

    # Aperçu des premières images problématiques
    if not flagged_df.empty:
        print("Exemples (5 premières):")
        for p in flagged_df['image_path'].head(5).tolist():
            print(" -", p)


if __name__ == '__main__':
    main()
