#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_tache_arch9_species_health_to_disease.py — Architecture 9: Species + Health → Disease
- 2 têtes de sortie: species (multi‑classe) et disease (multi‑classe hors "healthy")
- Architecture 9: La tête disease utilise:
  1) Les features du backbone
  2) Les probabilités complètes de la tête species (14 classes)
  3) La probabilité d'être malade de la tête health auxiliaire (1 valeur)
- La tête health est AUXILIAIRE (pas une sortie, utilisée uniquement comme feature)
- sample_weight=0 pour les échantillons healthy sur la tête disease
- Backbone: EfficientNetV2S (ImageNet), gelé au départ + fine‑tuning partiel
- tf.data pipeline depuis chemins; prétraitement 0..255 pour compatibilité avec preprocessing interne d'EfficientNetV2S
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
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


def class_name_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    return parent


def make_counts_table(names: List[str], labels: List[int]) -> pd.DataFrame:
    from collections import Counter
    cnt = Counter(labels)
    rows = [(names[i], cnt.get(i, 0)) for i in range(len(names))]
    return pd.DataFrame({"class": [r[0] for r in rows], "count": [r[1] for r in rows]})


# ----------------------
# tf.data pipeline (avec sample_weights par tête)
# ----------------------

def make_datasets_arch9(paths: Tuple[List[str], List[str], List[str]],
                        s2i: Dict[str,int], d2i: Dict[str,int],
                        img_size=(256, 256), batch_size=64, seed=42,
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

    def encode(fp: tf.Tensor):
        # tf.py_function wrapper to build labels + sample_weights
        def _py(fp_obj):
            # Convertir en str selon le type reçu (EagerTensor/bytes/numpy)
            try:
                if hasattr(fp_obj, 'numpy'):
                    p = fp_obj.numpy().decode('utf-8')
                elif isinstance(fp_obj, (bytes, bytearray)):
                    p = bytes(fp_obj).decode('utf-8')
                else:
                    p = fp_obj.tobytes().decode('utf-8') if hasattr(fp_obj, 'tobytes') else str(fp_obj)
            except Exception:
                p = str(fp_obj)
            sp = species_from_path(p)
            dis = disease_from_path(p)
            sp_idx = s2i[sp]
            y_sp = keras.utils.to_categorical(sp_idx, num_classes=len(s2i)).astype(np.float32)
            if dis != "healthy":
                dis_idx = d2i[dis]
                y_dis = keras.utils.to_categorical(dis_idx, num_classes=len(d2i)).astype(np.float32)
                w_dis = 1.0
            else:
                y_dis = np.zeros((len(d2i),), dtype=np.float32)
                w_dis = 0.0  # masque de perte
            w_sp = 1.0
            return y_sp, y_dis, np.float32(w_sp), np.float32(w_dis)
        y_sp, y_dis, w_sp, w_dis = tf.py_function(
            func=_py, inp=[fp], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
        )
        # set shapes for tf
        y_sp.set_shape((len(s2i),))
        y_dis.set_shape((len(d2i),))
        w_sp.set_shape(())
        w_dis.set_shape(())
        return y_sp, y_dis, w_sp, w_dis

    def build(split_paths, training):
        ds = tf.data.Dataset.from_tensor_slices(split_paths)
        if training:
            ds = ds.shuffle(len(split_paths), seed=seed, reshuffle_each_iteration=True)
        def _map(p):
            x = decode_resize_preprocess(p)
            if training and augment:
                x = augment_fn(x)
            x = to_model_space(x)
            y_sp, y_dis, w_sp, w_dis = encode(p)
            y = {"species": y_sp, "disease": y_dis}
            sw = {"species": w_sp, "disease": w_dis}
            return x, y, sw
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = build(paths[0], True)
    val_ds   = build(paths[1], False)
    test_ds  = build(paths[2], False)

    # Datasets images‑seules pour prédiction rapide
    def build_x(split_paths):
        ds = tf.data.Dataset.from_tensor_slices(split_paths)
        ds = ds.map(lambda p: to_model_space(decode_resize_preprocess(p)), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, build_x(paths[1]), build_x(paths[2])


# ----------------------
# Modèle et callbacks
# ----------------------

def build_model_arch9(n_species: int, n_disease: int, img_size=(256,256), initial_lr=1e-3, weight_decay=1e-4,
                      label_smoothing_species=0.1, label_smoothing_disease=0.1):
    """
    Architecture 9: Disease profite de Species + Health
    
    Structure:
    - Backbone → Features
    - Features → Species (Dense softmax 14)
    - Features → Health auxiliaire (Dense softmax 2, non exposé)
    - Features + Species probabilities + Health prob diseased → Disease (Dense softmax 20)
    """
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="image")
    base = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    base.trainable = False
    x = base.output
    x = layers.Dropout(0.2)(x)

    # Species head (sortie principale)
    sp_logits = layers.Dense(n_species, activation='softmax', dtype='float32', name='species')(x)

    # Internal auxiliary health head (pas une sortie, utilisée comme feature)
    hl_aux = layers.Dense(2, activation='softmax', dtype='float32', name='health_aux')(x)
    hl_prob_diseased = layers.Lambda(lambda t: t[..., 1:2], name='health_prob_diseased')(hl_aux)

    # Concatenate: Features + Species probabilities + Health prob diseased
    x_combined = layers.Concatenate(name='concat_species_health')([x, sp_logits, hl_prob_diseased])

    # Disease head uses base features + species + health
    dis_logits = layers.Dense(n_disease, activation='softmax', dtype='float32', name='disease')(x_combined)

    model = keras.Model(inputs, {'species': sp_logits, 'disease': dis_logits})

    # Optimizer
    try:
        opt = keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=weight_decay)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=initial_lr)

    # Losses/metrics
    loss = {
        'species': keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_species),
        'disease': keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_disease),
    }
    metrics = {
        'species': ['accuracy'],
        'disease': ['accuracy'],
    }
    weighted_metrics = {
        'species': ['accuracy'],
        'disease': ['accuracy'],
    }
    model.compile(optimizer=opt, loss=loss, metrics=metrics, weighted_metrics=weighted_metrics)
    return model, base, loss, metrics


def unfreeze_top_layers(base_model, fine_tune_at=50):
    total = len(base_model.layers)
    k = max(0, min(total, fine_tune_at))
    for layer in base_model.layers[-k:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    print(f"[INFO] Unfroze top {k}/{total} layers (BatchNorm frozen).")


class MacroF1Arch9Callback(keras.callbacks.Callback):
    """Architecture 9: Calcule macro‑F1 pour species et disease (disease: seulement sur malades)."""
    def __init__(self, val_images_ds, val_species_idx, val_disease_idx):
        super().__init__()
        self.val_images_ds = val_images_ds
        self.y_sp = np.array(val_species_idx)
        self.y_dis = np.array(val_disease_idx)  # -1 for healthy
    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score, accuracy_score
        preds = self.model.predict(self.val_images_ds, verbose=0)
        p_sp = np.argmax(preds['species'], axis=1)
        # disease: mask by diseased
        mask = self.y_dis >= 0
        logs = logs or {}
        f1_sp = f1_score(self.y_sp, p_sp, average='macro')
        acc_sp = accuracy_score(self.y_sp, p_sp)
        logs['val_species_macro_f1'] = float(f1_sp)
        logs['val_species_accuracy_sklearn'] = float(acc_sp)
        if np.any(mask):
            p_dis = np.argmax(preds['disease'][mask], axis=1)
            f1_dis = f1_score(self.y_dis[mask], p_dis, average='macro')
            acc_dis = accuracy_score(self.y_dis[mask], p_dis)
            logs['val_disease_macro_f1'] = float(f1_dis)
            logs['val_disease_accuracy_sklearn'] = float(acc_dis)
        print(f"\n[VAL] Epoch {epoch+1}: species F1={logs.get('val_species_macro_f1',float('nan')):.4f}, disease F1={logs.get('val_disease_macro_f1',float('nan')) if 'val_disease_macro_f1' in logs else float('nan'):.4f}")


# ----------------------
# Figures et rapport
# ----------------------

def plot_training_curves_arch9(history_df, out_path):
    plt.figure(figsize=(12,6))
    # Losses
    plt.subplot(2,2,1); plt.plot(history_df.get('loss', []));
    if 'val_loss' in history_df: plt.plot(history_df['val_loss']);
    plt.title('Loss (global)'); plt.xlabel('Epoch'); plt.legend(['train','val'])
    # Acc species
    plt.subplot(2,2,2)
    for k in ['species_accuracy','val_species_accuracy']:
        if k in history_df: plt.plot(history_df[k], label=k)
    plt.title('Species accuracy'); plt.xlabel('Epoch'); plt.legend()
    # Acc disease
    plt.subplot(2,2,3)
    for k in ['disease_accuracy','val_disease_accuracy']:
        if k in history_df: plt.plot(history_df[k], label=k)
    plt.title('Disease accuracy'); plt.xlabel('Epoch'); plt.legend()
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


def write_report_md(output_dir, species_names, disease_names, best_metrics, fig_paths, report_paths=None):
    p = os.path.join(output_dir, 'report.md')
    lines = [
        f"# Multi‑tâche 2 têtes (species, disease)\n",
        f"Date: {datetime.now().isoformat()}\n\n",
        f"## Label spaces\n",
        f"- #species: {len(species_names)}\n",
        f"- #diseases (excl. healthy): {len(disease_names)}\n\n",
        f"## Meilleures métriques (Validation/Test)\n",
        f"- Species: Val Acc={best_metrics.get('val_acc_species','N/A')}, Val F1={best_metrics.get('val_f1_species','N/A')}, Test Acc={best_metrics.get('test_acc_species','N/A')}, Test F1={best_metrics.get('test_f1_species','N/A')}\n",
        f"- Disease: Val Acc={best_metrics.get('val_acc_disease','N/A')}, Val F1={best_metrics.get('val_f1_disease','N/A')}, Test Acc={best_metrics.get('test_acc_disease','N/A')}, Test F1={best_metrics.get('test_f1_disease','N/A')}\n\n",
        "## Classification Reports\n",
    ]
    if report_paths:
        if report_paths.get('cr_species'):
            lines.append(f"- Species: {os.path.relpath(report_paths['cr_species'], output_dir)}\n")
        if report_paths.get('cr_disease'):
            lines.append(f"- Disease: {os.path.relpath(report_paths['cr_disease'], output_dir)}\n")
    lines.append("\n")
    lines.extend([
        "## Figures\n",
        f"- Courbes: {os.path.relpath(fig_paths['training_curves'], output_dir)}\n",
        f"- Confusion matrix (species): {os.path.relpath(fig_paths['cm_species'], output_dir)}\n",
        f"- Confusion matrix (disease): {os.path.relpath(fig_paths['cm_disease'], output_dir)}\n",
        f"- Sanity grid: {os.path.relpath(fig_paths['sanity_check'], output_dir)}\n",
    ])
    with open(p, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"[INFO] Wrote report: {p}")


# ----------------------
# Splits
# ----------------------

def stratified_split_by_class(paths: List[str], seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    from sklearn.model_selection import StratifiedShuffleSplit
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    cls_names = [class_name_from_path(fp) for fp in paths]
    # map class folder to integer
    uniq = sorted(set(cls_names))
    cls2i = {c:i for i,c in enumerate(uniq)}
    y = np.array([cls2i[c] for c in cls_names])
    paths = np.array(paths)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(paths, y))
    tmp_paths, tmp_labels = paths[tmp_idx], y[tmp_idx]
    tmp_test_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tmp_test_ratio, random_state=seed)
    va_rel, te_rel = next(sss2.split(tmp_paths, tmp_labels))
    return (
        paths[tr_idx].tolist(), tmp_paths[va_rel].tolist(), tmp_paths[te_rel].tolist()
    )


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Architecture 9: Species + Health → Disease")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_arch9_species_health_to_disease')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, nargs=2, default=(256,256))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing_species', type=float, default=0.1)
    parser.add_argument('--label_smoothing_disease', type=float, default=0.1)
    parser.add_argument('--fine_tune_at', type=int, default=50)
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping norm (0=disabled, recommended: 0.5-1.0)')
    parser.add_argument('--loss_w_species', type=float, default=1.0)
    parser.add_argument('--loss_w_disease', type=float, default=1.5)
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

    print(f"[INFO] Scanning images under: {args.data_root}")
    all_files = list_image_files(args.data_root)
    if not all_files:
        raise RuntimeError("No images found.")

    # Label spaces
    species_names = sorted({species_from_path(fp) for fp in all_files})
    disease_names = sorted({disease_from_path(fp) for fp in all_files if disease_from_path(fp) != 'healthy'})
    if not disease_names:
        raise RuntimeError("No diseased samples found for disease head.")
    s2i = {s:i for i,s in enumerate(species_names)}
    d2i = {d:i for i,d in enumerate(disease_names)}
    i2s = {i:s for s,i in s2i.items()}
    i2d = {i:d for d,i in d2i.items()}

    # Splits (optionnellement importés depuis --splits_file)
    if args.splits_file and os.path.isfile(args.splits_file):
        with open(args.splits_file, 'r', encoding='utf-8') as f:
            sp = json.load(f)
        print(f"[INFO] Loaded splits from: {args.splits_file}")
        all_set = set(all_files)
        def _keep(seq):
            return [p for p in seq if p in all_set]
        tr_p = _keep(sp.get('train', []))
        va_p = _keep(sp.get('val', []))
        te_p = _keep(sp.get('test', []))
        if not (tr_p and va_p and te_p):
            print("[WARN] Loaded splits incomplete or mismatched with data_root; falling back to stratified by class folder.")
            tr_p, va_p, te_p = stratified_split_by_class(all_files, seed=args.seed)
    else:
        # Stratifiés par dossier de classe (species___disease)
        tr_p, va_p, te_p = stratified_split_by_class(all_files, seed=args.seed)
    print(f"[INFO] Split sizes: train={len(tr_p)}, val={len(va_p)}, test={len(te_p)}")

    # Sauver labels
    with open(os.path.join(args.output_dir, 'species.json'), 'w', encoding='utf-8') as f:
        json.dump(species_names, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'diseases.json'), 'w', encoding='utf-8') as f:
        json.dump(disease_names, f, indent=2, ensure_ascii=False)

    # Sanity grid (train)
    sanity_path = os.path.join(args.output_dir, 'sanity_check.png')
    if not args.no_sanity_grid:
        try:
            save_sanity_check_grid(tr_p, tuple(args.img_size), sanity_path, grid=tuple(args.sanity_grid))
            print(f"[INFO] Saved sanity check grid to: {sanity_path}")
        except Exception as e:
            print(f"[WARN] Sanity grid failed: {e}")
    else:
        print("[INFO] Sanity grid disabled (--no_sanity_grid).")

    # Datasets (avec sample_weights)
    train_ds, val_ds, test_ds, val_x, test_x = make_datasets_arch9(
        (tr_p, va_p, te_p), s2i, d2i,
        img_size=tuple(args.img_size), batch_size=args.batch_size, seed=args.seed, augment=True
    )

    # Y val/test pour callback et évaluation
    def build_indices(paths: List[str]):
        sp_idx = [s2i[species_from_path(p)] for p in paths]
        dis_idx= [d2i[disease_from_path(p)] if disease_from_path(p) != 'healthy' else -1 for p in paths]
        return sp_idx, dis_idx
    va_sp, va_dis = build_indices(va_p)
    te_sp, te_dis = build_indices(te_p)

    # Modèle Architecture 9
    macro_cb = MacroF1Arch9Callback(val_images_ds=val_x, val_species_idx=va_sp, val_disease_idx=va_dis)
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
        model, base, loss_dict, metrics_dict = build_model_arch9(
            n_species=len(s2i), n_disease=len(d2i), img_size=tuple(args.img_size),
            initial_lr=args.initial_lr, weight_decay=args.weight_decay,
            label_smoothing_species=args.label_smoothing_species,
            label_smoothing_disease=args.label_smoothing_disease,
        )
        # Appliquer les poids de pertes via compile (fit ne supporte pas loss_weights)
        loss_w = {'species': args.loss_w_species, 'disease': args.loss_w_disease}
        weighted_metrics_dict = {'species': ['accuracy'], 'disease': ['accuracy']}
        model.compile(optimizer=model.optimizer, loss=loss_dict, metrics=metrics_dict, loss_weights=loss_w, weighted_metrics=weighted_metrics_dict)
        callbacks = [
            macro_cb,
            keras.callbacks.EarlyStopping(monitor='val_species_macro_f1', mode='max', patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_species_macro_f1', mode='max', patience=5, factor=0.5, min_lr=5e-5, verbose=1),
            keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_species_macro_f1', mode='max', save_best_only=True, verbose=1)
        ]
        # Phase 1: têtes (backbone frozen)
        print("[INFO] Stage 1: Train heads (backbone frozen)")
        h1 = model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=args.epochs,
                       callbacks=callbacks,
                       verbose=1)
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
        weighted_metrics_dict = {'species': ['accuracy'], 'disease': ['accuracy']}
        model.compile(optimizer=opt_ft, loss=loss_dict, metrics=metrics_dict, loss_weights=loss_w, weighted_metrics=weighted_metrics_dict)
        h2 = model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=args.epochs,
                       callbacks=callbacks,
                       verbose=1)
        hist = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
        history_csv = os.path.join(args.output_dir, 'history.csv')
        hist.to_csv(history_csv, index=False)

        try:
            model = keras.models.load_model(ckpt_path)
            print(f"[INFO] Loaded best model from: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Could not load best model ({e}); using current model.")

    # Évaluation (val/test)
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    # VAL
    pv = model.predict(val_x, verbose=0)
    p_sp_v = np.argmax(pv['species'], axis=1)
    mask_v = np.array(va_dis) >= 0
    f1_sp_v = f1_score(va_sp, p_sp_v, average='macro')
    acc_sp_v= accuracy_score(va_sp, p_sp_v)
    if np.any(mask_v):
        p_dis_v = np.argmax(pv['disease'][mask_v], axis=1)
        f1_dis_v= f1_score(np.array(va_dis)[mask_v], p_dis_v, average='macro')
        acc_dis_v=accuracy_score(np.array(va_dis)[mask_v], p_dis_v)
    else:
        f1_dis_v = float('nan'); acc_dis_v = float('nan')
    # TEST
    pt = model.predict(test_x, verbose=0)
    p_sp_t = np.argmax(pt['species'], axis=1)
    mask_t = np.array(te_dis) >= 0
    f1_sp_t = f1_score(te_sp, p_sp_t, average='macro')
    acc_sp_t= accuracy_score(te_sp, p_sp_t)
    if np.any(mask_t):
        p_dis_t = np.argmax(pt['disease'][mask_t], axis=1)
        f1_dis_t= f1_score(np.array(te_dis)[mask_t], p_dis_t, average='macro')
        acc_dis_t=accuracy_score(np.array(te_dis)[mask_t], p_dis_t)
    else:
        f1_dis_t = float('nan'); acc_dis_t = float('nan')

    # Confusion matrices
    cm_sp = confusion_matrix(te_sp, p_sp_t, labels=list(range(len(s2i))))
    present_labels = sorted(list(set(np.array(te_dis)[mask_t].tolist()))) if np.any(mask_t) else []
    cm_dis = confusion_matrix(np.array(te_dis)[mask_t], p_dis_t, labels=present_labels) if present_labels else np.zeros((0,0), dtype=np.int64)

    # Classification reports
    cr_sp = classification_report(te_sp, p_sp_t, target_names=[i2s[i] for i in range(len(s2i))], digits=4)
    cr_sp_path = os.path.join(args.output_dir, 'classification_report_species.txt')
    with open(cr_sp_path, 'w', encoding='utf-8') as f:
        f.write(cr_sp)
    print(f"[INFO] Saved species classification report: {cr_sp_path}")

    if present_labels:
        cr_dis = classification_report(np.array(te_dis)[mask_t], p_dis_t, target_names=[i2d[i] for i in present_labels], digits=4)
        cr_dis_path = os.path.join(args.output_dir, 'classification_report_disease.txt')
        with open(cr_dis_path, 'w', encoding='utf-8') as f:
            f.write(cr_dis)
        print(f"[INFO] Saved disease classification report: {cr_dis_path}")
    else:
        cr_dis_path = None

    # Sauvegardes
    cm_sp_path  = os.path.join(args.output_dir, 'cm_species.png')
    cm_dis_path = os.path.join(args.output_dir, 'cm_disease.png')
    plot_confusion_matrix(cm_sp, [i2s[i] for i in range(len(s2i))], cm_sp_path)
    if cm_dis.size:
        plot_confusion_matrix(cm_dis, [i2d[i] for i in present_labels], cm_dis_path)

    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    try:
        if 'hist' in locals() and hist is not None:
            plot_training_curves_arch9(hist, curves_path)
        else:
            hist2 = None
            if os.path.exists(os.path.join(args.output_dir, 'history.csv')):
                hist2 = pd.read_csv(os.path.join(args.output_dir, 'history.csv'))
            if hist2 is not None:
                plot_training_curves_arch9(hist2, curves_path)
            else:
                raise RuntimeError("history is not available")
    except Exception as e:
        print(f"[WARN] Failed to plot curves: {e}")

    best_metrics = {
        'val_acc_species': f"{acc_sp_v:.4f}", 'val_f1_species': f"{f1_sp_v:.4f}",
        'val_acc_disease': f"{acc_dis_v:.4f}", 'val_f1_disease': f"{f1_dis_v:.4f}",
        'test_acc_species': f"{acc_sp_t:.4f}", 'test_f1_species': f"{f1_sp_t:.4f}",
        'test_acc_disease': f"{acc_dis_t:.4f}", 'test_f1_disease': f"{f1_dis_t:.4f}",
    }
    fig_paths = {
        'training_curves': curves_path,
        'cm_species': cm_sp_path,
        'cm_disease': cm_dis_path,
        'sanity_check': os.path.join(args.output_dir, 'sanity_check.png'),
    }
    report_paths = {
        'cr_species': cr_sp_path,
        'cr_disease': cr_dis_path if present_labels else None,
    }
    write_report_md(args.output_dir, species_names, disease_names, best_metrics, fig_paths, report_paths)
    print("[INFO] Done. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
