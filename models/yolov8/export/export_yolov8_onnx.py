#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export du modèle YOLOv8 (classification) en ONNX")
    parser.add_argument("--weights", type=str, default="/workspaces/datasciencetest_reco_plante/results/yolov8_segmented_finetune/weights/best.pt", help="Chemin du .pt à exporter")
    parser.add_argument("--imgsz", type=int, default=224, help="Taille d'image pour l'export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument("--outdir", type=str, default="/workspaces/datasciencetest_reco_plante/results/export", help="Dossier de sortie")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    result = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, half=False, dynamic=False, simplify=True)
    onnx_path = Path(result) if isinstance(result, str) else outdir / "model.onnx"

    # Si Ultralytics écrit à côté des poids, déplace dans outdir
    if onnx_path.exists() and onnx_path.parent != outdir:
        target = outdir / onnx_path.name
        target.write_bytes(onnx_path.read_bytes())
        onnx_path = target

    print(f"✅ Export ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
