#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified evaluation script for Keras and YOLOv8 classification models.
Outputs per-class metrics CSV, confusion matrix image, and a short JSON summary.

Examples:
  # Evaluate Keras best model
  python -u scripts/evaluate_cls.py --framework keras \
    --weights outputs/cls_resnet50_e5_clean_bs32/best_model.keras \
    --splits outputs/cls_resnet50_e5_clean_bs32/splits.json \
    --out outputs/cls_resnet50_e5_clean_bs32/eval_test

  # Evaluate YOLOv8 best model
  python -u scripts/evaluate_cls.py --framework yolov8 \
    --weights outputs/yolov8_cls/exp_s/weights/best.pt \
    --split_root dataset/plantvillage/clean_split --split test \
    --out outputs/yolov8_cls/exp_s/eval_test
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.3), max(6, len(class_names)*0.3)))
    im = ax.imshow(cm, cmap='Blues'); ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names, ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    thr = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black', fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)


def eval_keras(weights, splits_json, split, out_dir):
    import tensorflow as tf
    from tensorflow import keras

    os.makedirs(out_dir, exist_ok=True)
    with open(splits_json, 'r', encoding='utf-8') as f:
        sp = json.load(f)
    classes = sp['classes']
    paths = sp[f'{split}_paths']
    labels = sp.get(f'{split}_labels', None)
    if labels is None:
        # If labels not stored for test in clean split, rebuild from class names
        def species_from_path(path: str) -> str:
            parent = os.path.basename(os.path.dirname(path))
            return parent.split("___")[0] if "___" in parent else parent
        s2i = {s: i for i, s in enumerate(classes)}
        labels = [s2i[species_from_path(p)] for p in paths]

    model = keras.models.load_model(weights)

    # Build TF dataset for inference (no augmentation)
    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (model.input_shape[1], model.input_shape[2]), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

    y_true = np.array(labels)
    y_prob = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    return finalize_metrics(y_true, y_pred, classes, out_dir)


def eval_yolov8(weights, split_root, split, out_dir):
    from ultralytics import YOLO
    os.makedirs(out_dir, exist_ok=True)

    # Build file list and labels from directory layout
    def walk_split(root):
        paths, labels = [], []
        classes = sorted(os.listdir(root))
        s2i = {s: i for i, s in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                p = os.path.join(cls_dir, fn)
                if os.path.isfile(p):
                    paths.append(p)
                    labels.append(s2i[cls])
        return classes, paths, labels

    classes, paths, labels = walk_split(os.path.join(split_root, split))

    model = YOLO(weights)
    # YOLO classification predict returns probabilities; use numpy output
    probs = model.predict(paths, imgsz=256, verbose=False)
    # probs is a list of Results with .probs.data (torch.Tensor)
    import torch
    y_prob = torch.stack([r.probs.data for r in probs]).cpu().numpy()
    y_true = np.array(labels)
    y_pred = np.argmax(y_prob, axis=1)

    return finalize_metrics(y_true, y_pred, classes, out_dir)


def finalize_metrics(y_true, y_pred, classes, out_dir):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    pr, rc, f1c, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)

    os.makedirs(out_dir, exist_ok=True)
    per_class_path = os.path.join(out_dir, 'per_class_metrics.csv')
    import csv
    with open(per_class_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["class_index","class_name","precision","recall","f1","support"])
        for i, name in enumerate(classes):
            w.writerow([i, name, f"{float(pr[i]):.6f}", f"{float(rc[i]):.6f}", f"{float(f1c[i]):.6f}", int(sup[i])])

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix.png'))

    summary = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'num_classes': len(classes),
        'num_samples': int(len(y_true)),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    print(f"[INFO] Wrote: {per_class_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Unified evaluation for Keras/YOLOv8 classification")
    ap.add_argument('--framework', type=str, choices=['keras','yolov8'], required=True)
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--splits', type=str, help='For Keras: path to splits.json saved by the train script')
    ap.add_argument('--split_root', type=str, help='For YOLOv8: root directory with train/val/test')
    ap.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    if args.framework == 'keras':
        if not args.splits:
            raise ValueError('--splits is required for framework=keras')
        eval_keras(args.weights, args.splits, args.split, args.out)
    else:
        if not args.split_root:
            raise ValueError('--split_root is required for framework=yolov8')
        eval_yolov8(args.weights, args.split_root, args.split, args.out)


if __name__ == '__main__':
    main()
