#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input

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
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled.")
        except Exception:
            print("[WARN] Mixed precision unavailable; continuing in float32.")
    else:
        print("[INFO] No GPU detected; running in float32.")


def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_image_files(data_root: str):
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


def build_class_index(files):
    species = sorted({species_from_path(fp) for fp in files})
    s2i = {s: i for i, s in enumerate(species)}
    i2s = {i: s for s, i in s2i.items()}
    return species, s2i, i2s


def stratified_split(paths, labels, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    from sklearn.model_selection import StratifiedShuffleSplit
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    paths = np.array(paths)
    labels = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(paths, labels))
    tmp_paths, tmp_labels = paths[tmp_idx], labels[tmp_idx]
    tmp_test_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tmp_test_ratio, random_state=seed)
    va_rel, te_rel = next(sss2.split(tmp_paths, tmp_labels))
    return (paths[tr_idx].tolist(), labels[tr_idx].tolist(),
            tmp_paths[va_rel].tolist(), tmp_labels[va_rel].tolist(),
            tmp_paths[te_rel].tolist(), tmp_labels[te_rel].tolist())


def make_datasets(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
                  img_size=(256, 256), batch_size=64, num_classes=2, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE

    # Aug layers (rotation ±15° = 15/360 ≈ 0.0417)
    rotator = keras.layers.RandomRotation(0.0416667, fill_mode="reflect")

    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0  # 0..1 for tf.image.*
        return img

    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = rotator(img, training=True)
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img

    def to_model_space(img):
        # EfficientNetV2 preprocess expects 0..255
        return preprocess_input(img * 255.0)

    def build(paths, labels, training):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        def _map(p, y):
            x = decode_resize_preprocess(p)
            if training:
                x = augment(x)
            x = to_model_space(x)
            y = tf.one_hot(y, depth=num_classes)
            return x, y
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = build(train_paths, train_labels, True)
    val_ds = build(val_paths, val_labels, False)
    test_ds = build(test_paths, test_labels, False)

    # For sklearn eval: images-only ds
    def build_x(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: to_model_space(decode_resize_preprocess(p)), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, build_x(val_paths), build_x(test_paths)


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
    base = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
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


def plot_training_curves(history_df, out_path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history_df['loss']);
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
        imgs.append(img.numpy())
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(rows*cols):
        r, c = divmod(i, cols); ax = axes[r, c]
        if i < len(imgs): ax.imshow(imgs[i]); ax.axis('off')
        else: ax.axis('off')
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)


def write_report_md(output_dir, class_names, class_counts_df, best_metrics, fig_paths):
    p = os.path.join(output_dir, 'report.md')
    lines = [
        f"# Plant Species Classification Report\n",
        f"Date: {datetime.now().isoformat()}\n\n",
        f"## Dataset\n- Classes (species): {len(class_names)}\n- Species list (sorted):\n"
    ] + [f"  - {s}\n" for s in class_names] + [
        "\n### Image counts per species\n",
        class_counts_df.to_markdown(index=False),
        "\n\n## Best Validation Metrics\n",
        f"- Val Accuracy: {best_metrics.get('val_accuracy','N/A')}\n",
        f"- Val Macro-F1: {best_metrics.get('val_macro_f1','N/A')}\n",
        f"- Test Accuracy: {best_metrics.get('test_accuracy','N/A')}\n",
        f"- Test Macro-F1: {best_metrics.get('test_macro_f1','N/A')}\n\n",
        "## Figures\n",
        f"- Training curves: {os.path.relpath(fig_paths['training_curves'], output_dir)}\n",
        f"- Confusion matrix: {os.path.relpath(fig_paths['confusion_matrix'], output_dir)}\n",
        f"- Sanity check: {os.path.relpath(fig_paths['sanity_check'], output_dir)}\n",
    ]
    with open(p, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"[INFO] Wrote report: {p}")


def main():
    parser = argparse.ArgumentParser(description="Train species classifier on PlantVillage segmented dataset (Keras)")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, nargs=2, default=(256,256))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--fine_tune_at', type=int, default=50)
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    set_mixed_precision_if_gpu()

    print(f"[INFO] Scanning images under: {args.data_root}")
    all_files = list_image_files(args.data_root)
    species_names, s2i, i2s = build_class_index(all_files)
    print(f"[INFO] Detected {len(species_names)} species.")

    # Save index->species
    with open(os.path.join(args.output_dir, 'class_index.json'), 'w', encoding='utf-8') as f:
        json.dump(i2s, f, indent=2, ensure_ascii=False)

    # Counts per species
    labels_species = [species_from_path(fp) for fp in all_files]
    counts = Counter(labels_species)
    class_counts_df = pd.DataFrame({'species': list(counts.keys()), 'count': list(counts.values())}).sort_values('species')
    class_counts_df.to_csv(os.path.join(args.output_dir, 'class_counts.csv'), index=False)

    labels_idx = [s2i[s] for s in labels_species]
    tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_split(all_files, np.array(labels_idx), seed=args.seed)
    print(f"[INFO] Split sizes: train={len(tr_p)}, val={len(va_p)}, test={len(te_p)}")

    train_ds, val_ds, test_ds, val_x, test_x = make_datasets(
        tr_p, tr_y, va_p, va_y, te_p, te_y,
        img_size=tuple(args.img_size), batch_size=args.batch_size, num_classes=len(species_names), seed=args.seed
    )

    sanity_path = os.path.join(args.output_dir, 'sanity_check.png')
    try:
        save_sanity_check_grid(tr_p, tuple(args.img_size), sanity_path)
        print(f"[INFO] Saved sanity check grid to: {sanity_path}")
    except Exception as e:
        print(f"[WARN] Sanity grid failed: {e}")

    model, base = build_model(len(species_names), img_size=tuple(args.img_size),
                              initial_lr=args.initial_lr, weight_decay=args.weight_decay,
                              label_smoothing=args.label_smoothing)

    macro_cb = MacroF1Callback(val_images_ds=val_x, y_true_val=va_y)
    ckpt_path = os.path.join(args.output_dir, 'best_model.keras')
    callbacks = [
        macro_cb,
        keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1', mode='max', patience=5, factor=0.5, min_lr=5e-5, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
    ]

    print("[INFO] Stage 1: Train head (backbone frozen)")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

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
        best_model = keras.models.load_model(ckpt_path)
        print(f"[INFO] Loaded best model from: {ckpt_path}")
    except Exception as e:
        print(f"[WARN] Could not load best model ({e}); using current model.")
        best_model = model

    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    yv = np.array(va_y)
    yp_v = np.argmax(best_model.predict(val_x, verbose=0), axis=1)
    val_f1 = f1_score(yv, yp_v, average='macro')
    val_acc = accuracy_score(yv, yp_v)

    yt = np.array(te_y)
    yp_t = np.argmax(best_model.predict(test_x, verbose=0), axis=1)
    test_f1 = f1_score(yt, yp_t, average='macro')
    test_acc = accuracy_score(yt, yp_t)
    cm = confusion_matrix(yt, yp_t)

    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, species_names, cm_path)

    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    try:
        plot_training_curves(hist, curves_path)
    except Exception as e:
        print(f"[WARN] Failed to plot curves: {e}")

    report_figs = {
        'training_curves': curves_path,
        'confusion_matrix': cm_path,
        'sanity_check': sanity_path,
    }
    best_metrics = {
        'val_accuracy': f"{val_acc:.4f}",
        'val_macro_f1': f"{val_f1:.4f}",
        'test_accuracy': f"{test_acc:.4f}",
        'test_macro_f1': f"{test_f1:.4f}",
    }
    write_report_md(args.output_dir, species_names, class_counts_df, best_metrics, report_figs)

    print("[INFO] Done. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
