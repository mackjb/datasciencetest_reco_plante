#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper to train Ultralytics YOLOv8 classification on the clean split directory.
Writes all artifacts under outputs/yolov8_cls/<run_name> to preserve outputs/ logic.

Example:
  python -u scripts/train_yolov8_cls.py \
    --split_root dataset/plantvillage/clean_split \
    --model yolov8s-cls.pt --epochs 5 --batch 32 --imgsz 256 --name exp_s
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 classification (Ultralytics) on a prepared split")
    ap.add_argument("--split_root", type=str, required=True, help="Root directory with train/val/test subfolders")
    ap.add_argument("--model", type=str, default="yolov8s-cls.pt", help="Base YOLOv8 classification model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--name", type=str, default=None, help="Run name (subfolder under outputs/yolov8_cls)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is not installed. Please install 'ultralytics' in your environment.") from e

    project = os.path.join("outputs", "yolov8_cls")
    os.makedirs(project, exist_ok=True)
    run_name = args.name or f"exp_{os.path.basename(args.split_root)}_e{args.epochs}_bs{args.batch}_img{args.imgsz}"

    model = YOLO(args.model)
    model.train(task="classify", data=args.split_root, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                project=project, name=run_name)

    print("[INFO] YOLOv8 classification training complete.")
    print(f"[INFO] Artifacts under: {os.path.join(project, run_name)}")


if __name__ == "__main__":
    main()
