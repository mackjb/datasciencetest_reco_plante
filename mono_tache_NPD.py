#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mono_tache_NPD.py — Expérimentations mono‑tâche pour New Plant Diseases Dataset (Augmented)
Objectifs séparés:
  1) species  — Quelle est cette plante ?
  2) health   — La plante est‑elle malade ? (binaire: healthy vs diseased)
  3) disease  — Si malade: quelle maladie ? (multi‑classe hors "healthy")

- tf-nightly + tf-keras-nightly via `tf_keras`
- Backbone: EfficientNetV2S (ImageNet), gelé au départ, avec fine‑tune partiel optionnel
- tf.data pipeline depuis des chemins
- Prétraitement: images 0..255 (la couche 'rescaling' interne d'EfficientNetV2S applique 1/255)
- Rapport Markdown + courbes + matrice de confusion (si test labelisé)

Dataset attendu: pré‑splitté en train/valid/test
- Exemple: /workspaces/app/dataset/New_Plant_Disease/data
  - New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
  - New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid
  - test (parfois non labelisé)

Vous pouvez fournir directement --train_dir/--valid_dir/--test_dir pour éviter l'auto‑détection.
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Tuple

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


# ----------------------
# Utils & dataset helpers
# ----------------------

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


def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_image_files(root: str) -> List[str]:
    files = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(r, fn)
            if is_image_file(fp):
                files.append(fp)
    files.sort()
    return files


def contains_image_subdirs(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isdir(p):
                # au moins une image dans un sous‑dossier
                for r, _, fns in os.walk(p):
                    if any(is_image_file(os.path.join(r, fn)) for fn in fns):
                        return True
    except Exception:
        return False
    return False


def find_split_dir(data_root: str, target_name: str) -> str | None:
    cand = []
    for r, dirs, _ in os.walk(data_root):
        for d in dirs:
            if d.lower() == target_name.lower():
                full = os.path.join(r, d)
                if contains_image_subdirs(full) or target_name.lower() == 'test':
                    cand.append(full)
    # choisir le plus profond (probablement celui attendu)
    cand.sort(key=lambda p: (p.count(os.sep), len(p)))
    return cand[-1] if cand else None


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


def make_counts_table(names: List[str], labels: List[int]) -> pd.DataFrame:
    from collections import Counter
    cnt = Counter(labels)
    rows = [(names[i], cnt.get(i, 0)) for i in range(len(names))]
    return pd.DataFrame({"class": [r[0] for r in rows], "count": [r[1] for r in rows]})


# ----------------------
# tf.data pipeline
# ----------------------

def make_datasets(paths: Tuple[List[str], List[str], List[str]],
                  labels: Tuple[List[int], List[int], List[int] | None],
                  img_size=(256, 256), batch_size=64, num_classes=2, seed=42,
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

    def build_xy(split_paths, split_labels, training):
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

    def build_x(split_paths):
        ds = tf.data.Dataset.from_tensor_slices(split_paths)
        ds = ds.map(lambda p: to_model_space(decode_resize_preprocess(p)), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = build_xy(paths[0], labels[0], True)
    val_ds   = build_xy(paths[1], labels[1], False)

    test_ds = None
    test_x = None
    if labels[2] is not None:
        test_ds = build_xy(paths[2], labels[2], False)
        test_x = build_x(paths[2])
    elif paths[2]:
        # test non labelisé
        test_x = build_x(paths[2])

    val_x = build_x(paths[1])
    return train_ds, val_ds, test_ds, val_x, test_x


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
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="image")
    base = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = False
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
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


def write_report_md(output_dir, class_names, class_counts_df, best_metrics, fig_paths, task_name, test_labeled: bool):
    p = os.path.join(output_dir, 'report.md')
    try:
        counts_block = class_counts_df.to_markdown(index=False)
    except Exception:
        counts_block = class_counts_df.to_string(index=False)
    lines = [
        f"# Mono‑tâche (NPD): {task_name}\n",
        f"Date: {datetime.now().isoformat()}\n\n",
        f"## Dataset\n- Classes: {len(class_names)}\n- Liste (triée):\n"
    ] + [f"  - {s}\n" for s in class_names] + [
        "\n### Comptes par classe (train+valid+test labelisé)\n",
        counts_block,
        "\n\n## Meilleures métriques (Validation/Test)\n",
        f"- Val Accuracy: {best_metrics.get('val_accuracy','N/A')}\n",
        f"- Val Macro-F1: {best_metrics.get('val_macro_f1','N/A')}\n",
        f"- Test Accuracy: {best_metrics.get('test_accuracy','N/A' if test_labeled else 'N/A (unlabeled test)')}\n",
        f"- Test Macro-F1: {best_metrics.get('test_macro_f1','N/A' if test_labeled else 'N/A (unlabeled test)')}\n\n",
        "## Figures\n",
        f"- Courbes: {os.path.relpath(fig_paths['training_curves'], output_dir)}\n",
        f"- Matrice de confusion: {os.path.relpath(fig_paths['confusion_matrix'], output_dir) if fig_paths.get('confusion_matrix') else 'N/A'}\n",
        f"- Sanity grid: {os.path.relpath(fig_paths['sanity_check'], output_dir)}\n",
    ]
    with open(p, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"[INFO] Wrote report: {p}")


# ----------------------
# Construction à partir des splits existants
# ----------------------

def build_paths_and_labels(task: str, train_dir: str, valid_dir: str, test_dir: str | None):
    # Lire fichiers
    tr_files = list_image_files(train_dir)
    va_files = list_image_files(valid_dir)
    te_files = list_image_files(test_dir) if test_dir and os.path.isdir(test_dir) else []

    # Détecter si test est labelisé (présence de sous‑dossiers de classes)
    test_labeled = False
    if test_dir and os.path.isdir(test_dir):
        subs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        test_labeled = any(list_image_files(sd) for sd in subs)

    # Préparer labels selon la tâche
    if task == 'species':
        all_named = tr_files + va_files + (te_files if test_labeled else [])
        names = sorted({species_from_path(fp) for fp in all_named})
        n2i = {n:i for i,n in enumerate(names)}
        i2n = {i:n for n,i in n2i.items()}
        tr_y = [n2i[species_from_path(fp)] for fp in tr_files]
        va_y = [n2i[species_from_path(fp)] for fp in va_files]
        te_y = [n2i[species_from_path(fp)] for fp in te_files] if test_labeled else None
    elif task == 'health':
        names = ['healthy','diseased']
        n2i = {'healthy':0,'diseased':1}
        i2n = {0:'healthy', 1:'diseased'}
        tr_y = [1 if disease_from_path(fp) != 'healthy' else 0 for fp in tr_files]
        va_y = [1 if disease_from_path(fp) != 'healthy' else 0 for fp in va_files]
        te_y = [1 if disease_from_path(fp) != 'healthy' else 0 for fp in te_files] if test_labeled else None
    else:  # disease
        tr_files = [fp for fp in tr_files if disease_from_path(fp) != 'healthy']
        va_files = [fp for fp in va_files if disease_from_path(fp) != 'healthy']
        te_files = [fp for fp in te_files if disease_from_path(fp) != 'healthy'] if test_labeled else te_files
        all_named = tr_files + va_files + (te_files if test_labeled else [])
        names = sorted({disease_from_path(fp) for fp in all_named})
        if not names:
            raise RuntimeError("No diseased samples found for disease task.")
        n2i = {n:i for i,n in enumerate(names)}
        i2n = {i:n for n,i in n2i.items()}
        tr_y = [n2i[disease_from_path(fp)] for fp in tr_files]
        va_y = [n2i[disease_from_path(fp)] for fp in va_files]
        te_y = [n2i[disease_from_path(fp)] for fp in te_files] if test_labeled else None

    counts_df = make_counts_table(names, tr_y + va_y + (te_y if te_y is not None else []))
    return (tr_files, va_files, te_files), (tr_y, va_y, te_y), names, i2n, counts_df, test_labeled


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Expérimentations mono‑tâche NPD (species/health/disease)")
    parser.add_argument('--task', type=str, choices=['species','health','disease'], default='species')
    parser.add_argument('--data_root', type=str, default='/workspaces/app/dataset/New_Plant_Disease/data')
    parser.add_argument('--train_dir', type=str, default=None, help='Chemin explicite vers le dossier train')
    parser.add_argument('--valid_dir', type=str, default=None, help='Chemin explicite vers le dossier valid')
    parser.add_argument('--test_dir', type=str, default=None, help='Chemin explicite vers le dossier test (peut être non labelisé)')
    parser.add_argument('--output_dir', type=str, default='outputs_mono_npd')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, nargs=2, default=(256,256))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--fine_tune_at', type=int, default=50)
    parser.add_argument('--report_only', action='store_true', help='Ne pas entraîner; charger best_model.keras et générer uniquement le rapport.')
    parser.add_argument('--no_sanity_grid', action='store_true', help='Désactive la génération de la grille de contrôle des images.')
    parser.add_argument('--sanity_grid', type=int, nargs=2, metavar=('ROWS','COLS'), default=(6,6),
                        help='Nombre de lignes et colonnes pour la grille de contrôle (défaut: 6 6).')
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    set_mixed_precision_if_gpu()

    # Détection des répertoires de split si non fournis
    train_dir = args.train_dir or find_split_dir(args.data_root, 'train')
    valid_dir = args.valid_dir or find_split_dir(args.data_root, 'valid')
    test_dir  = args.test_dir  or find_split_dir(args.data_root, 'test')
    if not train_dir or not valid_dir:
        raise FileNotFoundError("Impossible de localiser les répertoires 'train' et 'valid'. Fournissez --train_dir/--valid_dir ou vérifiez --data_root.")
    print(f"[INFO] train_dir: {train_dir}")
    print(f"[INFO] valid_dir: {valid_dir}")
    print(f"[INFO] test_dir:  {test_dir if test_dir else 'None'}")

    # Construire listes de chemins + labels à partir des splits
    paths, labels, names, idx2name, counts_df, test_labeled = build_paths_and_labels(
        args.task, train_dir, valid_dir, test_dir)

    print(f"[INFO] Detected {len(names)} classes for task '{args.task}'.")
    print(f"[INFO] Split sizes: train={len(paths[0])}, val={len(paths[1])}, test={len(paths[2]) if paths[2] else 0} (labeled={test_labeled})")

    # Sauver mapping classes et comptes
    with open(os.path.join(args.output_dir, 'class_index.json'), 'w', encoding='utf-8') as f:
        json.dump({i:idx2name[i] for i in range(len(names))}, f, indent=2, ensure_ascii=False)
    counts_df.to_csv(os.path.join(args.output_dir, 'class_counts.csv'), index=False)

    # Sanity grid (sur le train)
    sanity_path = os.path.join(args.output_dir, 'sanity_check.png')
    if not args.no_sanity_grid:
        try:
            save_sanity_check_grid(paths[0], tuple(args.img_size), sanity_path, grid=tuple(args.sanity_grid))
            print(f"[INFO] Saved sanity check grid to: {sanity_path}")
        except Exception as e:
            print(f"[WARN] Sanity grid failed: {e}")
    else:
        print("[INFO] Sanity grid disabled (--no_sanity_grid).")

    # Datasets
    train_ds, val_ds, test_ds, val_x, test_x = make_datasets(
        (paths[0], paths[1], paths[2]), (labels[0], labels[1], labels[2]),
        img_size=tuple(args.img_size), batch_size=args.batch_size,
        num_classes=len(names), seed=args.seed,
        augment=True
    )

    # Modèle
    macro_cb = MacroF1Callback(val_images_ds=val_x, y_true_val=labels[1])
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
        try:
            opt_ft = keras.optimizers.AdamW(learning_rate=args.ft_lr, weight_decay=args.weight_decay)
        except Exception:
            opt_ft = keras.optimizers.Adam(learning_rate=args.ft_lr)
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
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    yv = np.array(labels[1])
    yp_v = np.argmax(model.predict(val_x, verbose=0), axis=1)
    val_f1 = f1_score(yv, yp_v, average='macro')
    val_acc = accuracy_score(yv, yp_v)

    test_f1 = None; test_acc = None; cm_path = None
    if test_x is not None and labels[2] is not None:
        yt = np.array(labels[2])
        yp_t = np.argmax(model.predict(test_x, verbose=0), axis=1)
        test_f1 = f1_score(yt, yp_t, average='macro')
        test_acc = accuracy_score(yt, yp_t)
        cm = confusion_matrix(yt, yp_t, labels=list(range(len(names))))
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, [idx2name[i] for i in range(len(names))], cm_path)
    else:
        print("[INFO] Test set non labelisé ou non fourni: métriques test ignorées.")

    # Figures & rapport
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
        'test_accuracy': f"{test_acc:.4f}" if test_acc is not None else 'N/A',
        'test_macro_f1': f"{test_f1:.4f}" if test_f1 is not None else 'N/A',
    }
    report_figs = {
        'training_curves': curves_path,
        'confusion_matrix': cm_path,
        'sanity_check': os.path.join(args.output_dir, 'sanity_check.png'),
    }
    write_report_md(args.output_dir, [idx2name[i] for i in range(len(names))], make_counts_table(names, labels[0]+labels[1]+(labels[2] if labels[2] is not None else [])), best_metrics, report_figs, args.task, test_labeled=(labels[2] is not None))
    print("[INFO] Done. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
