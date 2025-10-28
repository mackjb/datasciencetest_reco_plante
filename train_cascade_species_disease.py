#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement en cascade pour PlantVillage avec DenseNet121:
  1) Modèle pour prédire l'espèce
  2) Modèle(s) pour prédire la maladie par espèce
  3) Évaluation cascade avec F1-score global (& logique)
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess

from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

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
            print("[WARN] Mixed precision unavailable.")
    else:
        print("[INFO] No GPU detected.")


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
        raise RuntimeError(f"No images found under: {data_root}")
    return files


def parse_labels_from_path(path: str):
    """Extrait (espèce, maladie) depuis le chemin."""
    parent = os.path.basename(os.path.dirname(path))
    if "___" in parent:
        species, disease = parent.split("___", 1)
    else:
        species, disease = parent, "unknown"
    return species, disease


def build_label_mappings(files):
    """Construit les mappings espèce et maladie."""
    all_species = set()
    all_diseases = set()
    species_diseases = defaultdict(set)
    
    for fp in files:
        species, disease = parse_labels_from_path(fp)
        all_species.add(species)
        all_diseases.add(disease)
        species_diseases[species].add(disease)
    
    species_list = sorted(all_species)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    
    disease_list = sorted(all_diseases)
    disease_to_idx = {d: i for i, d in enumerate(disease_list)}
    idx_to_disease = {i: d for d, i in disease_to_idx.items()}
    
    species_disease_mappings = {}
    for species in species_list:
        diseases_for_species = sorted(species_diseases[species])
        species_disease_mappings[species] = {
            d: i for i, d in enumerate(diseases_for_species)
        }
    
    return (species_to_idx, idx_to_species, 
            disease_to_idx, idx_to_disease,
            species_disease_mappings)


def stratified_split(paths, labels, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split stratifié sur les labels."""
    from sklearn.model_selection import StratifiedShuffleSplit
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    
    paths = np.array(paths)
    labels = np.array(labels)
    
    label_counts = Counter(labels.tolist())
    keep_mask = np.array([label_counts[l] >= 2 for l in labels])
    if not np.all(keep_mask):
        dropped = len(keep_mask) - int(keep_mask.sum())
        print(f"[WARN] Dropping {dropped} samples from rare classes.")
        paths = paths[keep_mask]
        labels = labels[keep_mask]
    
    try:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trval_idx, te_idx = next(sss_test.split(paths, labels))
        trval_paths, trval_labels = paths[trval_idx], labels[trval_idx]
        te_paths, te_labels = paths[te_idx], labels[te_idx]
        
        val_rel = val_ratio / (train_ratio + val_ratio)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
        tr_idx_rel, va_idx_rel = next(sss_val.split(trval_paths, trval_labels))
        tr_paths, tr_labels = trval_paths[tr_idx_rel], trval_labels[tr_idx_rel]
        va_paths, va_labels = trval_paths[va_idx_rel], trval_labels[va_idx_rel]
        
        return (tr_paths.tolist(), tr_labels.tolist(),
                va_paths.tolist(), va_labels.tolist(),
                te_paths.tolist(), te_labels.tolist())
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
        return (paths[tr_idx].tolist(), labels[tr_idx].tolist(),
                paths[va_idx].tolist(), labels[va_idx].tolist(),
                paths[te_idx].tolist(), labels[te_idx].tolist())


def make_dataset(paths, labels, img_size=(256, 256), batch_size=64, 
                 num_classes=2, training=False, seed=42):
    """Crée un tf.data.Dataset."""
    AUTOTUNE = tf.data.AUTOTUNE
    rotator = keras.layers.RandomRotation(0.0416667, fill_mode="reflect")
    
    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = rotator(img, training=True)
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
    
    def to_model_space(img):
        return densenet_preprocess(img * 255.0)
    
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


def build_densenet_classifier(num_classes, img_size=(256, 256), 
                               initial_lr=1e-3, weight_decay=1e-4, 
                               label_smoothing=0.1):
    """Construit un modèle DenseNet121."""
    base = DenseNet121(include_top=False, weights='imagenet', 
                       input_shape=(*img_size, 3), pooling='avg')
    base.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


class MacroF1Callback(keras.callbacks.Callback):
    """Callback pour F1 macro."""
    def __init__(self, val_ds, y_true_val):
        super().__init__()
        self.val_ds = val_ds
        self.y_true_val = np.array(y_true_val)
        self.best_f1 = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(self.y_true_val, y_pred, average='macro')
        acc = accuracy_score(self.y_true_val, y_pred)
        logs = logs or {}
        logs['val_macro_f1'] = f1
        print(f"\n[Epoch {epoch+1}] Val Acc: {acc:.4f}, Val F1: {f1:.4f}")
        if f1 > self.best_f1:
            self.best_f1 = f1


def train_species_model(train_paths, train_species_labels,
                        val_paths, val_species_labels,
                        num_species, output_dir, args):
    """Entraîne le modèle ESPÈCE."""
    print("\n" + "="*60)
    print("ÉTAPE 1: Entraînement modèle ESPÈCE")
    print("="*60)
    
    train_ds = make_dataset(train_paths, train_species_labels, 
                           img_size=tuple(args.img_size),
                           batch_size=args.batch_size,
                           num_classes=num_species, training=True, seed=SEED)
    
    val_ds = make_dataset(val_paths, val_species_labels,
                         img_size=tuple(args.img_size),
                         batch_size=args.batch_size,
                         num_classes=num_species, training=False, seed=SEED)
    
    model = build_densenet_classifier(num_species, 
                                      img_size=tuple(args.img_size),
                                      initial_lr=args.initial_lr,
                                      weight_decay=args.weight_decay,
                                      label_smoothing=args.label_smoothing)
    
    model_path = os.path.join(output_dir, 'species_model.keras')
    callbacks = [
        MacroF1Callback(val_ds, val_species_labels),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                       save_best_only=True, mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience,
                                     restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                         patience=args.patience//2, min_lr=1e-7, verbose=1)
    ]
    
    print(f"[INFO] Training species model for {args.epochs_stage1} epochs...")
    history = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.epochs_stage1, callbacks=callbacks, verbose=1)
    
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(output_dir, 'species_history.csv'), index=False)
    
    print(f"[INFO] Species model saved to: {model_path}")
    return model


def train_disease_models(train_paths, train_species_labels, train_disease_labels,
                        val_paths, val_species_labels, val_disease_labels,
                        species_to_idx, idx_to_species, species_disease_mappings,
                        output_dir, args):
    """Entraîne les modèles MALADIE par espèce."""
    print("\n" + "="*60)
    print("ÉTAPE 2: Entraînement modèles MALADIE par espèce")
    print("="*60)
    
    disease_models = {}
    
    for species_name, species_idx in species_to_idx.items():
        train_mask = np.array(train_species_labels) == species_idx
        val_mask = np.array(val_species_labels) == species_idx
        
        if train_mask.sum() < 10:
            print(f"[SKIP] {species_name}: pas assez de données")
            disease_models[species_name] = None
            continue
        
        train_paths_sp = [train_paths[i] for i in range(len(train_paths)) if train_mask[i]]
        train_disease_sp_global = [train_disease_labels[i] for i in range(len(train_disease_labels)) if train_mask[i]]
        
        val_paths_sp = [val_paths[i] for i in range(len(val_paths)) if val_mask[i]]
        val_disease_sp_global = [val_disease_labels[i] for i in range(len(val_disease_labels)) if val_mask[i]]
        
        disease_remap = species_disease_mappings[species_name]
        disease_to_local = {d: local_idx for d, local_idx in disease_remap.items()}
        
        train_disease_local = [disease_to_local.get(d, 0) for d in train_disease_sp_global]
        val_disease_local = [disease_to_local.get(d, 0) for d in val_disease_sp_global]
        
        num_diseases = len(disease_remap)
        
        if num_diseases < 2:
            print(f"[SKIP] {species_name}: une seule maladie")
            disease_models[species_name] = None
            continue
        
        print(f"\n[TRAIN] {species_name}: {num_diseases} maladies, {len(train_paths_sp)} samples")
        
        train_ds = make_dataset(train_paths_sp, train_disease_local,
                               img_size=tuple(args.img_size),
                               batch_size=min(args.batch_size, max(8, len(train_paths_sp))),
                               num_classes=num_diseases, training=True, seed=SEED)
        
        val_ds = make_dataset(val_paths_sp, val_disease_local,
                             img_size=tuple(args.img_size),
                             batch_size=min(args.batch_size, max(8, len(val_paths_sp))),
                             num_classes=num_diseases, training=False, seed=SEED)
        
        model = build_densenet_classifier(num_diseases,
                                          img_size=tuple(args.img_size),
                                          initial_lr=args.initial_lr,
                                          weight_decay=args.weight_decay,
                                          label_smoothing=args.label_smoothing)
        
        model_path = os.path.join(output_dir, f'disease_model_{species_name}.keras')
        callbacks = [
            MacroF1Callback(val_ds, val_disease_local),
            keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                           save_best_only=True, mode='max', verbose=0),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience,
                                         restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                             patience=args.patience//2, min_lr=1e-7, verbose=0)
        ]
        
        history = model.fit(train_ds, validation_data=val_ds,
                           epochs=args.epochs_stage2, callbacks=callbacks, verbose=0)
        
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(os.path.join(output_dir, f'disease_history_{species_name}.csv'), index=False)
        
        disease_models[species_name] = model
        print(f"[SAVED] {model_path}")
    
    return disease_models


def evaluate_cascade(species_model, disease_models, test_paths, test_species_labels, test_disease_labels,
                     species_to_idx, idx_to_species, disease_to_idx, idx_to_disease,
                     species_disease_mappings, output_dir, args):
    """Évalue la cascade et calcule le F1-score global (& logique)."""
    print("\n" + "="*60)
    print("ÉVALUATION CASCADE")
    print("="*60)
    
    # Prédictions espèce
    test_ds_species = make_dataset(test_paths, test_species_labels,
                                   img_size=tuple(args.img_size),
                                   batch_size=args.batch_size,
                                   num_classes=len(species_to_idx),
                                   training=False, seed=SEED)
    
    species_probs = species_model.predict(test_ds_species, verbose=1)
    species_pred = np.argmax(species_probs, axis=1)
    
    # Prédictions maladie conditionnelles
    disease_pred = []
    
    print("\n[INFO] Prédiction des maladies en cascade...")
    for i, (path, true_species, true_disease) in enumerate(zip(test_paths, test_species_labels, test_disease_labels)):
        pred_species_idx = species_pred[i]
        pred_species_name = idx_to_species[pred_species_idx]
        
        if pred_species_name in disease_models and disease_models[pred_species_name] is not None:
            disease_model = disease_models[pred_species_name]
            
            # Charger et prédire l'image
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, args.img_size, method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32) / 255.0
            img = densenet_preprocess(img * 255.0)
            img = tf.expand_dims(img, 0)
            
            disease_prob = disease_model.predict(img, verbose=0)
            disease_idx_in_species = np.argmax(disease_prob, axis=1)[0]
            
            # Remapper vers l'indice global
            disease_remap = species_disease_mappings[pred_species_name]
            disease_name = [k for k, v in disease_remap.items() if v == disease_idx_in_species]
            if disease_name:
                disease_pred_idx = disease_to_idx[disease_name[0]]
            else:
                disease_pred_idx = -1
        else:
            # Pas de modèle => première maladie de l'espèce prédite
            if pred_species_name in species_disease_mappings:
                disease_name = list(species_disease_mappings[pred_species_name].keys())[0]
                disease_pred_idx = disease_to_idx[disease_name]
            else:
                disease_pred_idx = -1
        
        disease_pred.append(disease_pred_idx)
    
    disease_pred = np.array(disease_pred)
    
    # Évaluation espèce seule
    species_acc = accuracy_score(test_species_labels, species_pred)
    species_f1 = f1_score(test_species_labels, species_pred, average='macro')
    
    # Évaluation maladie seule
    disease_acc = accuracy_score(test_disease_labels, disease_pred)
    disease_f1 = f1_score(test_disease_labels, disease_pred, average='macro')
    
    # Évaluation cascade (& logique)
    cascade_correct = (species_pred == np.array(test_species_labels)) & (disease_pred == np.array(test_disease_labels))
    cascade_acc = cascade_correct.mean()
    
    # F1 cascade: classe combinée
    true_combined = np.array(test_species_labels) * 1000 + np.array(test_disease_labels)
    pred_combined = species_pred * 1000 + disease_pred
    cascade_f1 = f1_score(true_combined, pred_combined, average='macro')
    
    print(f"\n{'='*60}")
    print("RÉSULTATS")
    print(f"{'='*60}")
    print(f"Espèce seule    - Accuracy: {species_acc:.4f}, Macro-F1: {species_f1:.4f}")
    print(f"Maladie seule   - Accuracy: {disease_acc:.4f}, Macro-F1: {disease_f1:.4f}")
    print(f"CASCADE (& log) - Accuracy: {cascade_acc:.4f}, Macro-F1: {cascade_f1:.4f}")
    
    # Sauvegarder
    results = {
        'species_accuracy': float(species_acc),
        'species_f1': float(species_f1),
        'disease_accuracy': float(disease_acc),
        'disease_f1': float(disease_f1),
        'cascade_accuracy': float(cascade_acc),
        'cascade_f1': float(cascade_f1),
    }
    
    with open(os.path.join(output_dir, 'cascade_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rapport espèce
    species_names = [idx_to_species[i] for i in range(len(species_to_idx))]
    species_report = classification_report(test_species_labels, species_pred, 
                                          target_names=species_names, output_dict=True, zero_division=0)
    pd.DataFrame(species_report).T.to_csv(os.path.join(output_dir, 'species_report.csv'))
    
    # Matrice confusion espèce
    cm_species = confusion_matrix(test_species_labels, species_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_species, annot=False, fmt='d', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names)
    plt.title('Confusion Matrix - Espèce')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_species.png'), dpi=150)
    plt.close()
    
    print(f"\n[INFO] Résultats sauvegardés dans: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Entraînement en cascade PlantVillage")
    parser.add_argument('--data_root', type=str, 
                       default='/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented',
                       help='Chemin dataset PlantVillage')
    parser.add_argument('--output_dir', type=str, default='outputs/cascade',
                       help='Répertoire de sortie')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                       help='Taille des images')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs_stage1', type=int, default=30,
                       help='Époques pour le modèle espèce')
    parser.add_argument('--epochs_stage2', type=int, default=20,
                       help='Époques pour les modèles maladie')
    parser.add_argument('--initial_lr', type=float, default=1e-3,
                       help='Learning rate initial')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay L2')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Setup
    set_mixed_precision_if_gpu()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n[INFO] Chargement des images depuis: {args.data_root}")
    files = list_image_files(args.data_root)
    print(f"[INFO] {len(files)} images trouvées")
    
    # Construire les mappings
    (species_to_idx, idx_to_species,
     disease_to_idx, idx_to_disease,
     species_disease_mappings) = build_label_mappings(files)
    
    print(f"[INFO] {len(species_to_idx)} espèces, {len(disease_to_idx)} maladies")
    
    # Extraire les labels
    species_labels = []
    disease_labels = []
    for fp in files:
        species, disease = parse_labels_from_path(fp)
        species_labels.append(species_to_idx[species])
        disease_labels.append(disease_to_idx[disease])
    
    # Stratifier sur combinaison espèce+maladie
    combined_labels = [f"{s}_{d}" for s, d in zip(species_labels, disease_labels)]
    
    (train_paths, train_combined,
     val_paths, val_combined,
     test_paths, test_combined) = stratified_split(files, combined_labels, seed=SEED)
    
    # Récupérer les labels séparés
    train_species_labels = [species_labels[files.index(p)] for p in train_paths]
    train_disease_labels = [disease_labels[files.index(p)] for p in train_paths]
    
    val_species_labels = [species_labels[files.index(p)] for p in val_paths]
    val_disease_labels = [disease_labels[files.index(p)] for p in val_paths]
    
    test_species_labels = [species_labels[files.index(p)] for p in test_paths]
    test_disease_labels = [disease_labels[files.index(p)] for p in test_paths]
    
    print(f"[INFO] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Sauvegarder les mappings
    with open(os.path.join(args.output_dir, 'species_index.json'), 'w') as f:
        json.dump(idx_to_species, f, indent=2)
    with open(os.path.join(args.output_dir, 'disease_index.json'), 'w') as f:
        json.dump(idx_to_disease, f, indent=2)
    with open(os.path.join(args.output_dir, 'species_disease_mappings.json'), 'w') as f:
        json.dump({k: {d: int(i) for d, i in v.items()} for k, v in species_disease_mappings.items()}, f, indent=2)
    
    # ÉTAPE 1: Entraîner modèle espèce
    species_model = train_species_model(
        train_paths, train_species_labels,
        val_paths, val_species_labels,
        len(species_to_idx), args.output_dir, args
    )
    
    # ÉTAPE 2: Entraîner modèles maladie
    disease_models = train_disease_models(
        train_paths, train_species_labels, train_disease_labels,
        val_paths, val_species_labels, val_disease_labels,
        species_to_idx, idx_to_species, species_disease_mappings,
        args.output_dir, args
    )
    
    # ÉTAPE 3: Évaluer en cascade
    results = evaluate_cascade(
        species_model, disease_models,
        test_paths, test_species_labels, test_disease_labels,
        species_to_idx, idx_to_species,
        disease_to_idx, idx_to_disease,
        species_disease_mappings,
        args.output_dir, args
    )
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT CASCADE TERMINÉ")
    print("="*60)
    print(f"Résultats dans: {args.output_dir}")
    print(f"  - species_model.keras")
    print(f"  - disease_model_<espèce>.keras")
    print(f"  - cascade_results.json")
    print(f"  - confusion_species.png")
    print("\n")


if __name__ == '__main__':
    main()