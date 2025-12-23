#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mono_tache_1_tete.py — Architecture 3: CNN mono-objectif - 1 tête unique (35 classes)

Objectif unique:
  - Prédire simultanément l'espèce ET l'état de santé/maladie (35 classes)
  - Classes = 14 espèces × (1 healthy + ~1-3 maladies par espèce) = 35 classes totales
  - Format: "Espèce_État" (ex: "Apple_healthy", "Apple_apple_scab", "Tomato_late_blight")

Différence avec multi-tâche:
- Pas de séparation Species/Disease: 1 seule sortie multi-classe
- Chaque combinaison (espèce, maladie) est une classe unique
- Plus simple architecturalement mais perd la structure hiérarchique

- tf-nightly + tf-keras-nightly via `tf_keras`
- Backbone: EfficientNetV2S (ImageNet), gelé au départ, avec fine-tune partiel
- tf.data pipeline depuis des chemins
- Prétraitement: images 0..255 (la couche 'rescaling' interne d'EfficientNetV2S applique 1/255)
- Rapport Markdown + courbes + matrice de confusion + classification report
"""

import argparse
import json
import math
import os
import random
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tf_keras as keras
from tf_keras.applications.efficientnet_v2 import EfficientNetV2S

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def set_mixed_precision_if_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        try:
            from tf_keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled.")
        except Exception:
            print("[WARN] Mixed precision unavailable; continuing in float32.")
    else:
        print("[INFO] No GPU detected; running in float32.")


# ----------------------
# Helpers dataset
# ----------------------

def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_image_files(data_root: str) -> List[str]:
    files = []
    for r, _, fns in os.walk(data_root):
        for fn in fns:
            fp = os.path.join(r, fn)
            if is_image_file(fp):
                files.append(fp)
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found under: {data_root}")
    return files


def species_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    return parent.split("___")[0] if "___" in parent else parent


def disease_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if "___" not in parent:
        return "healthy"
    dis = parent.split("___", 1)[1].strip().strip('_').lower()
    if dis in {"", "-", "none"}:
        return "healthy"
    return dis


def combined_label_from_path(path: str) -> str:
    """
    Crée une label combinée 'Espèce_État' pour chaque image.
    Exemples:
      - Apple___healthy → "Apple_healthy"
      - Apple___Apple_scab → "Apple_apple_scab"
      - Tomato___Late_blight → "Tomato_late_blight"
    """
    species = species_from_path(path)
    disease = disease_from_path(path)
    return f"{species}_{disease}"


def make_counts_table(names: List[str], labels: List[int]) -> pd.DataFrame:
    from collections import Counter
    cnt = Counter(labels)
    rows = [(names[i], cnt.get(i, 0)) for i in range(len(names))]
    return pd.DataFrame({"class": [r[0] for r in rows], "count": [r[1] for r in rows]})


# ----------------------
# tf.data pipeline
# ----------------------

def make_datasets(paths: List[str], labels: List[int], img_size=(256, 256), batch_size=64, num_classes=2, seed=42,
                  augment=True):
    AUTOTUNE = tf.data.AUTOTUNE

    rotator = keras.layers.RandomRotation(0.0416667, fill_mode="reflect")

    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def augment_fn(img):
        img = tf.image.random_flip_left_right(img)
        img = rotator(img, training=True)
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img

    def to_model_space(img):
        # Fournir 0..255 car EfficientNetV2S (tf_keras) inclut 'rescaling' interne (1/255)
        return img * 255.0

    def build(split_paths, split_labels, training):
        ds = tf.data.Dataset.from_tensor_slices((split_paths, split_labels))
        if training:
            ds = ds.shuffle(len(split_paths), seed=seed, reshuffle_each_iteration=True)
        def _map(p, y):
            x = decode_resize_preprocess(p)
            if training and augment:
                x = augment_fn(x)
            x = to_model_space(x)
            y = tf.one_hot(y, depth=num_classes)
            return x, y
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = build(paths[0], labels[0], True)
    val_ds   = build(paths[1], labels[1], False)
    test_ds  = build(paths[2], labels[2], False)

    # Datasets images‑seules pour sklearn
    def build_x(split_paths):
        ds = tf.data.Dataset.from_tensor_slices(split_paths)
        ds = ds.map(lambda p: to_model_space(decode_resize_preprocess(p)), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, build_x(paths[1]), build_x(paths[2])


# ----------------------
# Modèle et callbacks
# ----------------------

class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, val_images_ds, y_true_val):
        super().__init__()
        self.val_images_ds = val_images_ds
        self.y_true_val = np.array(y_true_val)
    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score, accuracy_score
        y_prob = self.model.predict(self.val_images_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(self.y_true_val, y_pred, average='macro')
        acc = accuracy_score(self.y_true_val, y_pred)
        logs = logs or {}
        logs['val_macro_f1'] = float(f1)
        logs['val_accuracy_sklearn'] = float(acc)
        print(f"\n[VAL] Epoch {epoch+1}: macro-F1={f1:.4f}, acc={acc:.4f}")


def build_model(num_classes, img_size=(256, 256), initial_lr=1e-3, weight_decay=1e-4, label_smoothing=0.1):
    """
    Construit le modèle avec 1 seule tête de sortie (num_classes = 35).
    """
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="image")
    base = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    base.trainable = False
    x = base.output
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='combined_output')(x)
    model = keras.Model(inputs, outputs)
    try:
        opt = keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=weight_decay)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=initial_lr)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model, base


def unfreeze_top_layers(base_model, fine_tune_at=50):
    total = len(base_model.layers)
    k = max(0, min(total, fine_tune_at))
    for layer in base_model.layers[-k:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    print(f"[INFO] Unfroze top {k}/{total} layers (BatchNorm frozen).")


# ----------------------
# Figures et rapport
# ----------------------

def plot_training_curves(history_df, out_path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history_df.get('loss', []));
    if 'val_loss' in history_df: plt.plot(history_df['val_loss']);
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(['train','val'])
    plt.subplot(1,2,2)
    if 'accuracy' in history_df: plt.plot(history_df['accuracy'])
    if 'val_accuracy' in history_df: plt.plot(history_df['val_accuracy'])
    if 'val_accuracy_sklearn' in history_df: plt.plot(history_df['val_accuracy_sklearn'], '--')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(['train','val','val_sklearn'])
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


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


def save_sanity_check_grid(train_paths, img_size, output_path, grid=(6,6)):
    rows, cols = grid; n = min(rows*cols, len(train_paths))
    sample_paths = random.sample(train_paths, n)
    rotator = keras.layers.RandomRotation(0.0416667, fill_mode="reflect")
    imgs = []
    for p in sample_paths:
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)/255.0
        img = tf.image.random_flip_left_right(img)
        img = rotator(img, training=True)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        imgs.append(tf.cast(img, tf.float32).numpy())
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(rows*cols):
        r, c = divmod(i, cols); ax = axes[r, c]
        if i < len(imgs): ax.imshow(imgs[i]); ax.axis('off')
        else: ax.axis('off')
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)


def write_report_md(output_dir, class_names, class_counts_df, best_metrics, fig_paths):
    """Génère le rapport Markdown."""
    p = os.path.join(output_dir, 'report.md')
    try:
        counts_block = class_counts_df.to_markdown(index=False)
    except Exception:
        counts_block = class_counts_df.to_string(index=False)
    lines = [
        f"# Architecture 3: CNN mono-objectif - 1 tête unique (35 classes)\n\n",
        f"**Date**: {datetime.now().isoformat()}\n\n",
        f"## Description\n\n",
        f"Modèle à 1 seule tête prédisant simultanément l'espèce ET l'état de santé/maladie.\n",
        f"Format des classes : `Espèce_État` (ex: `Apple_healthy`, `Tomato_late_blight`)\n\n",
        f"## Dataset\n\n",
        f"- **Nombre de classes**: {len(class_names)}\n",
        f"- **Format**: 14 espèces × (1 healthy + maladies) = 35 classes totales\n\n",
        f"### Comptes par classe\n\n",
        counts_block,
        f"\n\n## Meilleures métriques (Validation/Test)\n\n",
        f"- **Val Accuracy**: {best_metrics.get('val_accuracy','N/A')}\n",
        f"- **Val Macro-F1**: {best_metrics.get('val_macro_f1','N/A')}\n",
        f"- **Test Accuracy**: {best_metrics.get('test_accuracy','N/A')}\n",
        f"- **Test Macro-F1**: {best_metrics.get('test_macro_f1','N/A')}\n\n",
        f"## Figures\n\n",
        f"- Courbes: `{os.path.basename(fig_paths['training_curves'])}`\n",
        f"- Matrice de confusion: `{os.path.basename(fig_paths['confusion_matrix'])}`\n",
        f"- Classification report: `{os.path.basename(fig_paths['classification_report'])}`\n",
        f"- Sanity grid: `{os.path.basename(fig_paths['sanity_check'])}`\n\n",
        f"## Architecture\n\n",
        f"- **Backbone**: EfficientNetV2S (ImageNet pré-entraîné)\n",
        f"- **Stratégie**: Gel du backbone → Fine-tuning partiel\n",
        f"- **Sortie**: 1 Dense(35, softmax) - Classification multi-classe unique\n",
        f"- **Avantage**: Simplicité architecturale\n",
        f"- **Limite**: Perd la structure hiérarchique (espèce/maladie séparées)\n",
    ]
    with open(p, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"[INFO] Wrote report: {p}")


# ----------------------
# Construction des splits selon la tâche
# ----------------------

def stratified_splits(paths: List[str], y: List[int], seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    from sklearn.model_selection import StratifiedShuffleSplit
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    paths = np.array(paths)
    labels = np.array(y)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(paths, labels))
    tmp_paths, tmp_labels = paths[tmp_idx], labels[tmp_idx]
    tmp_test_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tmp_test_ratio, random_state=seed)
    va_rel, te_rel = next(sss2.split(tmp_paths, tmp_labels))
    return (
        paths[tr_idx].tolist(), labels[tr_idx].tolist(),
        tmp_paths[va_rel].tolist(), tmp_labels[va_rel].tolist(),
        tmp_paths[te_rel].tolist(), tmp_labels[te_rel].tolist(),
    )


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Architecture 3: CNN mono-objectif - 1 tête unique (35 classes: Espèce_État)")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_mono_1head_35classes')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, nargs=2, default=(256,256))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--fine_tune_at', type=int, default=50)
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping norm (0=disabled)')
    parser.add_argument('--report_only', action='store_true', help='Ne pas entraîner; charger best_model.keras et générer uniquement le rapport.')
    parser.add_argument('--no_sanity_grid', action='store_true', help='Désactive la génération de la grille de contrôle des images.')
    parser.add_argument('--sanity_grid', type=int, nargs=2, metavar=('ROWS','COLS'), default=(6,6),
                        help='Nombre de lignes et colonnes pour la grille de contrôle (défaut: 6 6).')
    parser.add_argument('--splits_file', type=str, default=None,
                        help='Chemin JSON pour charger des splits prédéfinis (keys: train,val,test).')
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    set_mixed_precision_if_gpu()

    # Scanner les images
    print(f"[INFO] Scanning images under: {args.data_root}")
    all_files = list_image_files(args.data_root)
    print(f"[INFO] Found {len(all_files)} images.")

    # Créer les labels combinées (Espèce_État)
    print("[INFO] Building combined labels (Species_State)...")
    combined_labels_all = [combined_label_from_path(fp) for fp in all_files]
    names = sorted(set(combined_labels_all))
    print(f"[INFO] Detected {len(names)} unique classes (Espèce_État combinations).")
    
    # Mapping
    name2idx = {n: i for i, n in enumerate(names)}
    idx2name = {i: n for n, i in name2idx.items()}
    labels_all = [name2idx[lbl] for lbl in combined_labels_all]

    # Splits
    if args.splits_file and os.path.isfile(args.splits_file):
        print(f"[INFO] Loading splits from: {args.splits_file}")
        with open(args.splits_file, 'r', encoding='utf-8') as f:
            sp = json.load(f)
        all_set = set(all_files)
        def _keep(seq):
            return [p for p in seq if p in all_set]
        tr_p = _keep(sp.get('train', []))
        va_p = _keep(sp.get('val', []))
        te_p = _keep(sp.get('test', []))
        if not (tr_p and va_p and te_p):
            print("[WARN] Loaded splits incomplete; falling back to stratified splits.")
            tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_splits(all_files, labels_all, seed=args.seed)
        else:
            tr_y = [name2idx[combined_label_from_path(p)] for p in tr_p]
            va_y = [name2idx[combined_label_from_path(p)] for p in va_p]
            te_y = [name2idx[combined_label_from_path(p)] for p in te_p]
    else:
        print("[INFO] No splits file provided; creating stratified splits...")
        tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_splits(all_files, labels_all, seed=args.seed)

    print(f"[INFO] Detected {len(names)} classes (Species_State combined).")
    print(f"[INFO] Split sizes: train={len(tr_p)}, val={len(va_p)}, test={len(te_p)}")

    # Sauver mapping classes et comptes
    with open(os.path.join(args.output_dir, 'class_index.json'), 'w', encoding='utf-8') as f:
        json.dump({i:idx2name[i] for i in range(len(names))}, f, indent=2, ensure_ascii=False)
    counts_df = make_counts_table(names, tr_y + va_y + te_y)
    counts_df.to_csv(os.path.join(args.output_dir, 'class_counts.csv'), index=False)

    # Sanity grid (sur le train)
    sanity_path = os.path.join(args.output_dir, 'sanity_check.png')
    if not args.no_sanity_grid:
        try:
            save_sanity_check_grid(tr_p, tuple(args.img_size), sanity_path, grid=tuple(args.sanity_grid))
            print(f"[INFO] Saved sanity check grid to: {sanity_path}")
        except Exception as e:
            print(f"[WARN] Sanity grid failed: {e}")
    else:
        print("[INFO] Sanity grid disabled (--no_sanity_grid).")

    # Datasets
    train_ds, val_ds, test_ds, val_x, test_x = make_datasets(
        (tr_p, va_p, te_p), (tr_y, va_y, te_y),
        img_size=tuple(args.img_size), batch_size=args.batch_size,
        num_classes=len(names), seed=args.seed,
        augment=True
    )

    # Modèle
    macro_cb = MacroF1Callback(val_images_ds=val_x, y_true_val=va_y)
    ckpt_path = os.path.join(args.output_dir, 'best_model.keras')

    if args.report_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"best_model.keras introuvable dans {args.output_dir}. Relancez un entraînement ou indiquez un dossier valide.")
        model = keras.models.load_model(ckpt_path)
        print(f"[INFO] Report-only: loaded best model from: {ckpt_path}")
        hist = None
        history_csv = os.path.join(args.output_dir, 'history.csv')
        if os.path.exists(history_csv):
            try:
                hist = pd.read_csv(history_csv)
                print(f"[INFO] Loaded history: {history_csv}")
            except Exception as e:
                print(f"[WARN] Could not read history.csv: {e}")
    else:
        model, base = build_model(len(names), img_size=tuple(args.img_size),
                                  initial_lr=args.initial_lr, weight_decay=args.weight_decay,
                                  label_smoothing=args.label_smoothing)
        callbacks = [
            macro_cb,
            keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1', mode='max', patience=5, factor=0.5, min_lr=5e-5, verbose=1),
            keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
        ]
        # Phase 1: tête seule
        print("[INFO] Stage 1: Train head (backbone frozen)")
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
        # Phase 2: fine‑tune top layers
        print("[INFO] Stage 2: Fine-tune top layers")
        unfreeze_top_layers(base, fine_tune_at=args.fine_tune_at)
        
        # Gradient clipping pour éviter les explosions
        clip_kwargs = {'clipnorm': args.gradient_clip} if args.gradient_clip > 0 else {}
        try:
            opt_ft = keras.optimizers.AdamW(learning_rate=args.ft_lr, weight_decay=args.weight_decay, **clip_kwargs)
            if args.gradient_clip > 0:
                print(f"[INFO] Gradient clipping enabled: clipnorm={args.gradient_clip}")
        except Exception:
            opt_ft = keras.optimizers.Adam(learning_rate=args.ft_lr, **clip_kwargs)
        
        model.compile(optimizer=opt_ft, loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing), metrics=['accuracy'])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

        hist = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
        history_csv = os.path.join(args.output_dir, 'history.csv')
        hist.to_csv(history_csv, index=False)

        try:
            model = keras.models.load_model(ckpt_path)
            print(f"[INFO] Loaded best model from: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Could not load best model ({e}); using current model.")

    # Évaluation
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    yv = np.array(va_y)
    yp_v = np.argmax(model.predict(val_x, verbose=0), axis=1)
    val_f1 = f1_score(yv, yp_v, average='macro')
    val_acc = accuracy_score(yv, yp_v)

    yt = np.array(te_y)
    yp_t = np.argmax(model.predict(test_x, verbose=0), axis=1)
    test_f1 = f1_score(yt, yp_t, average='macro')
    test_acc = accuracy_score(yt, yp_t)
    cm = confusion_matrix(yt, yp_t, labels=list(range(len(names))))
    
    # Classification report (test set)
    class_names_list = [idx2name[i] for i in range(len(names))]
    cr_text = classification_report(yt, yp_t, target_names=class_names_list, digits=4)
    cr_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(cr_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report - Test Set\n")
        f.write(f"Architecture: 1 tête unique (35 classes Species_State)\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(cr_text)
    print(f"[INFO] Saved classification report: {cr_path}")

    # Derive Species and Disease (global) reports from 35-class predictions
    class2species = {}
    class2disease = {}
    for p, y in zip(tr_p + va_p + te_p, tr_y + va_y + te_y):
        if y not in class2species:
            class2species[y] = species_from_path(p)
            class2disease[y] = disease_from_path(p)

    # Species metrics
    y_true_species = [species_from_path(p) for p in te_p]
    y_pred_species = [class2species[int(c)] for c in yp_t]
    species_names = sorted(list(set(y_true_species) | set(y_pred_species)))
    sp2i = {n: i for i, n in enumerate(species_names)}
    y_true_sp = np.array([sp2i[s] for s in y_true_species])
    y_pred_sp = np.array([sp2i[s] for s in y_pred_species])
    cr_species = classification_report(y_true_sp, y_pred_sp, target_names=species_names, digits=4)
    cr_species_path = os.path.join(args.output_dir, 'classification_report_species.txt')
    with open(cr_species_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report - Species (derived from 35 classes)\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(cr_species)
    sp_f1 = f1_score(y_true_sp, y_pred_sp, average='macro')
    sp_acc = accuracy_score(y_true_sp, y_pred_sp)

    # Disease metrics (global)
    y_true_disease = [disease_from_path(p) for p in te_p]
    y_pred_disease = [class2disease[int(c)] for c in yp_t]
    disease_names = sorted(list(set(y_true_disease) | set(y_pred_disease)))
    dz2i = {n: i for i, n in enumerate(disease_names)}
    y_true_dz = np.array([dz2i[d] for d in y_true_disease])
    y_pred_dz = np.array([dz2i[d] for d in y_pred_disease])
    cr_disease = classification_report(y_true_dz, y_pred_dz, target_names=disease_names, digits=4)
    cr_disease_path = os.path.join(args.output_dir, 'classification_report_disease_global.txt')
    with open(cr_disease_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report - Disease (global) from 35 classes\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(cr_disease)
    dz_f1 = f1_score(y_true_dz, y_pred_dz, average='macro')
    dz_acc = accuracy_score(y_true_dz, y_pred_dz)

    # Summary JSON
    summary_json = os.path.join(args.output_dir, 'summary_species_disease.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump({
            'species': {'macro_f1': float(sp_f1), 'accuracy': float(sp_acc)},
            'disease_global': {'macro_f1': float(dz_f1), 'accuracy': float(dz_acc)}
        }, f, indent=2, ensure_ascii=False)

    # Figures & rapport
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, [idx2name[i] for i in range(len(names))], cm_path)

    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    try:
        if 'hist' in locals() and hist is not None:
            plot_training_curves(hist, curves_path)
        else:
            # tenter de charger history.csv
            hist2 = None
            if os.path.exists(os.path.join(args.output_dir, 'history.csv')):
                hist2 = pd.read_csv(os.path.join(args.output_dir, 'history.csv'))
            if hist2 is not None:
                plot_training_curves(hist2, curves_path)
            else:
                raise RuntimeError("history is not available")
    except Exception as e:
        print(f"[WARN] Failed to plot curves: {e}")

    best_metrics = {
        'val_accuracy': f"{val_acc:.4f}",
        'val_macro_f1': f"{val_f1:.4f}",
        'test_accuracy': f"{test_acc:.4f}",
        'test_macro_f1': f"{test_f1:.4f}",
    }
    report_figs = {
        'training_curves': curves_path,
        'confusion_matrix': cm_path,
        'sanity_check': sanity_path,
        'classification_report': cr_path,
    }
    write_report_md(args.output_dir, [idx2name[i] for i in range(len(names))], counts_df, best_metrics, report_figs)
    print("[INFO] Done. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
