#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List

from ultralytics import YOLO
from PIL import Image


def list_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.suffix.lower() in exts]
    return []


def main():
    parser = argparse.ArgumentParser(description="Prédiction YOLOv8 (classification)")
    parser.add_argument("input", type=str, help="Chemin vers une image ou un dossier d'images")
    parser.add_argument("--weights", type=str, default="/workspaces/datasciencetest_reco_plante/results/yolov8_segmented_finetune/weights/best.pt", help="Chemin du fichier de poids YOLOv8 .pt")
    parser.add_argument("--topk", type=int, default=5, help="Afficher les top-K classes")
    parser.add_argument("--save", action="store_true", help="Sauvegarder les images annotées à côté des originaux")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights_path}")

    inp = Path(args.input)
    images = list_images(inp)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans: {inp}")

    model = YOLO(weights_path)

    for img_path in images:
        # Exécuter la prédiction
        results = model(img_path, verbose=False)
        res = results[0]

        # Afficher Top-K classes (si dispo)
        if res.probs is not None:
            probs = res.probs.topk(args.topk)
            print(f"\nImage: {img_path}")
            for score, cls_idx in zip(probs.values.tolist(), probs.indices.tolist()):
                cls_name = res.names.get(int(cls_idx), str(cls_idx))
                print(f" - {cls_name}: {score:.4f}")
        else:
            print(f"\nImage: {img_path} -> prédiction disponible, mais pas de probabilités détaillées.")

        # Sauvegarde d'une image annotée si demandé (classification = légende en overlay)
        if args.save:
            # Ultralytics save() gère surtout détection/segmentation. Ici, on sauvegarde une copie simple.
            try:
                im = Image.open(img_path).convert("RGB")
                out_dir = img_path.parent / "predictions"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / img_path.name
                im.save(out_path)
            except Exception as e:
                print(f" - Impossible de sauvegarder l'image annotée: {e}")


if __name__ == "__main__":
    main()
