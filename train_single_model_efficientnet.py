#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approche DIRECTE avec UN SEUL modèle EfficientNetV2:
- Prédit directement les 38 classes finales (espèce___maladie)
- 'healthy' est une classe normale (pas séparée)
- Fine-tuning en 2 phases
- Génère heatmaps confusion + graphiques d'entraînement
"""
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] {len(gpus)} GPU(s) disponible(s)")
    except RuntimeError as e:
        print(f"[WARN] Erreur config GPU: {e}")

# Mixed precision pour accélérer
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
print(f"[INFO] Mixed precision enabled: {policy.name}")


def list_image_files(data_root):
    """Liste tous les fichiers image."""
    files = []
    for root, _, filenames in os.walk(data_root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, f))
    return sorted(files)


def parse_final_class_from_path(filepath):
    """
    Extrait la classe finale directement du chemin.
    Ex: .../Apple___Apple_scab/image.jpg -> "Apple___Apple_scab"
    """
    parts = filepath.split(os.sep)
    for part in reversed(parts):
        if '___' in part:
            return part
    return None


def stratified_split(paths, labels, test_ratio=0.15, val_ratio=0.15, seed=42):
    """Split stratifié train/val/test."""
    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(sss1.split(paths, labels))
        
        train_val_paths = np.array(paths)[train_val_idx]
        train_val_labels = np.array(labels)[train_val_idx]
        test_paths = np.array(paths)[test_idx]
        test_labels = np.array(labels)[test_idx]
        
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=seed)
        train_idx, val_idx = next(sss2.split(train_val_paths, train_val_labels))
        
        return (train_val_paths[train_idx].tolist(), train_val_labels[train_idx].tolist(),
                train_val_paths[val_idx].tolist(), train_val_labels[val_idx].tolist(),
                test_paths.tolist(), test_labels.tolist())
    except Exception as e:
        print(f"[WARN] Stratified split failed ({e}); using random split.")
        rng = np.random.default_rng(seed)
        idx = np.arange(len(paths))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(test_ratio * n))
        n_val = int(round(val_ratio * n))
        te_idx = idx[:n_test]
        va_idx = idx[n_test:n_test + n_val]
        tr_idx = idx[n_test + n_val:]
        return (np.array(paths)[tr_idx].tolist(), np.array(labels)[tr_idx].tolist(),
                np.array(paths)[va_idx].tolist(), np.array(labels)[va_idx].tolist(),
                np.array(paths)[te_idx].tolist(), np.array(labels)[te_idx].tolist())


def make_dataset(paths, labels, img_size=(256, 256), batch_size=32, 
                num_classes=38, training=False, seed=42):
    """Crée un tf.data.Dataset pour le modèle unique."""
    AUTOTUNE = tf.data.AUTOTUNE
    
    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return efficientnet_preprocess(img * 255.0)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    
    def _map(p, y):
        x = decode_resize_preprocess(p)
        y = tf.one_hot(y, depth=num_classes)
        return x, y
    
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_efficientnetv2_model(num_classes, img_size=(256, 256), 
                               initial_lr=1e-3, weight_decay=1e-4):
    """Construit un modèle EfficientNetV2B1 pour classification directe."""
    base = EfficientNetV2B1(include_top=False, weights='imagenet', 
                           input_shape=(*img_size, 3), pooling='avg')
    base.trainable = False  # Phase 1: gelé
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay),
                                 dtype='float32')(x)  # Force float32 pour output
    
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(optimizer=optimizer, loss=loss,
                 metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


class MacroF1Callback(keras.callbacks.Callback):
    """Callback pour calculer le F1 macro sur validation."""
    def __init__(self, val_ds, val_labels):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = val_labels
        self.best_f1 = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        f1 = f1_score(self.val_labels, pred_labels, average='macro', zero_division=0)
        acc = accuracy_score(self.val_labels, pred_labels)
        print(f"\n[Epoch {epoch+1}] Val Acc: {acc:.4f}, Val F1: {f1:.4f}")
        if f1 > self.best_f1:
            self.best_f1 = f1


def train_model(train_paths, train_labels, val_paths, val_labels,
               num_classes, class_names, output_dir, args):
    """Entraîne le modèle unique en 2 phases."""
    print("\n" + "="*70)
    print("MODÈLE UNIQUE - CLASSIFICATION DIRECTE (38 classes)")
    print(f"  EfficientNetV2B1 → {num_classes} classes finales (espèce___maladie)")
    print("="*70)
    
    train_ds = make_dataset(train_paths, train_labels, 
                           tuple(args.img_size), args.batch_size,
                           num_classes, True, SEED)
    
    val_ds = make_dataset(val_paths, val_labels,
                         tuple(args.img_size), args.batch_size,
                         num_classes, False, SEED)
    
    model = build_efficientnetv2_model(num_classes, tuple(args.img_size),
                                      args.initial_lr, args.weight_decay)
    
    model_path = os.path.join(output_dir, 'single_model_efficientnet.keras')
    callbacks = [
        MacroF1Callback(val_ds, val_labels),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                       save_best_only=True, mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience,
                                     restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                         patience=args.patience//2, min_lr=1e-7, verbose=1)
    ]
    
    # PHASE 1: Head only
    print(f"\n[PHASE 1] Head only ({args.epochs_phase1} époques)")
    history1 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_phase1, callbacks=callbacks, verbose=1)
    
    # PHASE 2: Fine-tuning
    print(f"\n[PHASE 2] Fine-tuning EfficientNetV2B1 ({args.epochs_phase2} époques)")
    base_model = model.layers[1]  # Le EfficientNetV2B1
    for layer in base_model.layers[-80:]:  # Dégeler les 80 dernières couches
        layer.trainable = True
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.initial_lr / 10),
                 loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                 metrics=['accuracy', keras.metrics.AUC(name='auc')])
    
    history2 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_phase2, callbacks=callbacks, verbose=1)
    
    # Sauvegarder historiques
    hist_df = pd.concat([pd.DataFrame(history1.history), pd.DataFrame(history2.history)], 
                        ignore_index=True)
    hist_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Générer graphiques d'entraînement
    plot_training_history(hist_df, output_dir)
    
    print(f"\n[INFO] Modèle sauvegardé: {model_path}")
    return model, history1, history2


def plot_training_history(hist_df, output_dir):
    """Génère les graphiques d'entraînement (loss, accuracy vs epochs)."""
    print("\n[INFO] Génération des graphiques d'entraînement...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss
    axes[0, 0].plot(hist_df['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(hist_df['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(hist_df['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(hist_df['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(hist_df['auc'], label='Train AUC', linewidth=2)
    axes[1, 0].plot(hist_df['val_auc'], label='Val AUC', linewidth=2)
    axes[1, 0].set_title('AUC vs Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('AUC', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(hist_df['learning_rate'], label='Learning Rate', linewidth=2, color='orange')
    axes[1, 1].set_title('Learning Rate vs Epochs', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Graphiques sauvegardés: training_curves.png")


def evaluate_model(model, test_paths, test_labels, class_names, output_dir, args):
    """Évalue le modèle et génère les heatmaps de confusion."""
    print("\n" + "="*70)
    print("ÉVALUATION")
    print("="*70)
    
    num_classes = len(class_names)
    test_ds = make_dataset(test_paths, test_labels,
                          tuple(args.img_size), args.batch_size,
                          num_classes, False, SEED)
    
    # Prédictions
    print("[INFO] Prédiction sur le test set...")
    preds = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    
    # Métriques
    acc = accuracy_score(test_labels, pred_labels)
    f1_macro = f1_score(test_labels, pred_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, pred_labels, average='weighted', zero_division=0)
    
    print(f"\n{'='*70}")
    print("RÉSULTATS")
    print(f"{'='*70}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1 Macro:        {f1_macro:.4f} (toutes classes égales)")
    print(f"F1 Weighted:     {f1_weighted:.4f} (pondéré par support)")
    
    # Sauvegarder résultats
    results = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'num_classes': num_classes,
        'num_test_samples': len(test_labels)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rapport de classification
    report = classification_report(test_labels, pred_labels, 
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Matrice de confusion
    print("\n[INFO] Génération de la matrice de confusion...")
    generate_confusion_matrices(test_labels, pred_labels, class_names, output_dir)
    
    print(f"\n[INFO] Résultats sauvegardés dans: {output_dir}")
    return results


def generate_confusion_matrices(true_labels, pred_labels, class_names, output_dir):
    """Génère les heatmaps de confusion."""
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    
    # Heatmap complète
    plt.figure(figsize=(28, 24))
    mask = cm == 0
    sns.heatmap(cm_df, annot=False, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Nombre de prédictions'},
                square=True, linewidths=0.5, linecolor='gray',
                mask=mask, vmin=0, vmax=cm.max())
    plt.title('Matrice de Confusion - Classification Directe (38 classes)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Classe Prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_full.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap normalisée
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    plt.figure(figsize=(28, 24))
    mask_norm = cm_normalized < 0.001
    sns.heatmap(cm_norm_df, annot=False, fmt='.2%', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Proportion (%)'},
                square=True, linewidths=0.5, linecolor='gray',
                mask=mask_norm, vmin=0, vmax=1)
    plt.title('Matrice de Confusion Normalisée (% par classe vraie)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Classe Prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap erreurs uniquement
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    if cm_errors.sum() > 0:
        cm_errors_df = pd.DataFrame(cm_errors, index=class_names, columns=class_names)
        
        plt.figure(figsize=(28, 24))
        mask_errors = cm_errors == 0
        sns.heatmap(cm_errors_df, annot=False, fmt='d', cmap='Reds', 
                    cbar_kws={'label': "Nombre d'erreurs"},
                    square=True, linewidths=0.5, linecolor='gray',
                    mask=mask_errors, vmin=0)
        plt.title('Matrice des Erreurs (Diagonale exclue)', 
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Classe Prédite (erreur)', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_errors.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Total erreurs: {cm_errors.sum()}")
    
    # Top confusions
    errors_list = []
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                errors_list.append({
                    'True': true_class,
                    'Predicted': pred_class,
                    'Count': cm[i, j],
                    'Percent_of_true': (cm[i, j] / cm[i].sum()) * 100
                })
    
    if errors_list:
        errors_df = pd.DataFrame(errors_list).sort_values('Count', ascending=False)
        errors_df.to_csv(os.path.join(output_dir, 'top_confusions.csv'), index=False)
        print("\n[INFO] Top 10 confusions:")
        print(errors_df.head(10).to_string(index=False))
    
    print(f"[INFO] Matrices de confusion sauvegardées")


def main():
    parser = argparse.ArgumentParser(description="Classification directe EfficientNetV2B1 (38 classes)")
    parser.add_argument('--data_root', type=str, 
                       default='/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented',
                       help='Chemin dataset PlantVillage')
    parser.add_argument('--output_dir', type=str, default='outputs/single_model_efficientnet',
                       help='Répertoire de sortie')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                       help='Taille des images')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs_phase1', type=int, default=15,
                       help='Époques phase 1 (head only)')
    parser.add_argument('--epochs_phase2', type=int, default=40,
                       help='Époques phase 2 (fine-tuning)')
    parser.add_argument('--initial_lr', type=float, default=1e-3,
                       help='Learning rate initial')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2)')
    parser.add_argument('--patience', type=int, default=7,
                       help='Patience pour early stopping')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Chargement données
    print(f"[INFO] Chargement des images depuis: {args.data_root}")
    files = list_image_files(args.data_root)
    print(f"[INFO] {len(files)} images trouvées")
    
    # Extraire les classes finales
    final_classes = []
    valid_files = []
    for f in files:
        final_class = parse_final_class_from_path(f)
        if final_class:
            final_classes.append(final_class)
            valid_files.append(f)
    
    # Mapping classe → index
    unique_classes = sorted(set(final_classes))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    labels = [class_to_idx[c] for c in final_classes]
    
    print(f"[INFO] {len(unique_classes)} classes finales")
    print(f"[INFO] Distribution: {Counter(final_classes).most_common(5)}")
    
    # Split stratifié (SEED=42)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
        valid_files, labels, seed=SEED
    )
    
    print(f"[INFO] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Sauvegarder les mappings
    with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
        json.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f, indent=2)
    
    # Entraînement
    model, hist1, hist2 = train_model(
        train_paths, train_labels,
        val_paths, val_labels,
        len(unique_classes), unique_classes,
        args.output_dir, args
    )
    
    # Évaluation
    results = evaluate_model(
        model, test_paths, test_labels,
        unique_classes, args.output_dir, args
    )
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)
    print(f"Résultats dans: {args.output_dir}/")
    print(f"  - single_model_efficientnet.keras (modèle)")
    print(f"  - results.json (métriques)")
    print(f"  - training_curves.png (graphiques d'entraînement)")
    print(f"  - confusion_matrix_*.png (heatmaps)")
    print(f"  - classification_report.csv (détails par classe)")


if __name__ == '__main__':
    main()
