#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement en cascade amélioré pour PlantVillage (2 modèles):

ENTRAÎNEMENT:
  1) Modèle ESPÈCE: DenseNet121 (fine-tuning 2 phases)
     - Input: Image
     - Output: Espèce (14 classes)
  
  2) Modèle MALADIE GLOBAL: ResNet50V2 + Attention (fine-tuning 2 phases)
     - Input: (Image, Espèce_GT)  ← Espèce VRAIE pendant l'entraînement
     - Output: Maladie (21 classes)

ÉVALUATION:
  - ORACLE: Maladie avec espèce GT (performance théorique max)
  - CASCADE: Espèce_pred → Maladie avec espèce PRÉDITE (vraie performance)

Pas d'augmentation de données.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as resnet_preprocess

from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit

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


def make_dataset_species(paths, labels, img_size=(256, 256), batch_size=64, 
                         num_classes=2, training=False, seed=42):
    """Crée un tf.data.Dataset pour le modèle d'espèce (DenseNet121, pas d'augmentation)."""
    AUTOTUNE = tf.data.AUTOTUNE
    
    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return densenet_preprocess(img * 255.0)
    
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


def make_dataset_disease(paths, species_labels, disease_labels, img_size=(256, 256), batch_size=64,
                         num_species=14, num_diseases=2, training=False, seed=42):
    """Crée un tf.data.Dataset pour le modèle de maladie (ResNet50V2+Attention, 2 entrées: image + espèce)."""
    AUTOTUNE = tf.data.AUTOTUNE
    
    def decode_resize_preprocess(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return resnet_preprocess(img * 255.0)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, species_labels, disease_labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    
    def _map(p, sp, dis):
        img = decode_resize_preprocess(p)
        sp_onehot = tf.one_hot(sp, depth=num_species)
        dis_onehot = tf.one_hot(dis, depth=num_diseases)
        return (img, sp_onehot), dis_onehot
    
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_species_model_densenet(num_classes, img_size=(256, 256), 
                               initial_lr=1e-3, weight_decay=1e-4):
    """Construit un modèle DenseNet121 pour l'espèce."""
    base = DenseNet121(include_top=False, weights='imagenet', 
                       input_shape=(*img_size, 3), pooling='avg', name='densenet121_backbone')
    base.trainable = False  # Phase 1: gelé
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


def build_disease_model_resnet_attention(num_diseases, num_species, img_size=(256, 256), 
                                        initial_lr=1e-3, weight_decay=1e-4):
    """Construit ResNet50V2 + Attention pour la maladie, conditionné sur l'espèce."""
    # Entrée image
    img_input = keras.Input(shape=(*img_size, 3), name='image')
    # Entrée espèce (one-hot)
    species_input = keras.Input(shape=(num_species,), name='species')
    
    # Backbone ResNet50V2
    base = ResNet50V2(include_top=False, weights='imagenet', input_shape=(*img_size, 3), name='resnet50v2_backbone')
    base.trainable = False  # Phase 1: gelé
    
    # Features image
    img_features = base(img_input, training=False)  # (None, H, W, C)
    
    # Attention mechanism
    attention = keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid', 
                                    name='attention_map')(img_features)
    img_features_attended = keras.layers.Multiply()([img_features, attention])
    img_features_pooled = keras.layers.GlobalAveragePooling2D()(img_features_attended)
    
    # Embedder l'espèce
    species_embed = keras.layers.Dense(64, activation='relu', 
                                       name='species_embed')(species_input)
    
    # Fusion
    combined = keras.layers.Concatenate()([img_features_pooled, species_embed])
    x = keras.layers.Dropout(0.5)(combined)
    x = keras.layers.Dense(256, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_diseases, activation='softmax',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    
    model = keras.Model(inputs=[img_input, species_input], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
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
    """Entraîne le modèle ESPÈCE en 2 phases: head only puis fine-tuning."""
    print("\n" + "="*60)
    print("ÉTAPE 1: Modèle ESPÈCE (DenseNet121, 2 phases)")
    print("="*60)
    
    train_ds = make_dataset_species(train_paths, train_species_labels, 
                                    tuple(args.img_size), args.batch_size,
                                    num_species, True, SEED)
    
    val_ds = make_dataset_species(val_paths, val_species_labels,
                                  tuple(args.img_size), args.batch_size,
                                  num_species, False, SEED)
    
    model = build_species_model_densenet(num_species, tuple(args.img_size),
                                         args.initial_lr, args.weight_decay)
    
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
    
    # PHASE 1: Head only (backbone gelé)
    print(f"\n[PHASE 1] Entraînement tête seule ({args.epochs_phase1} époques)")
    history1 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_phase1, callbacks=callbacks, verbose=1)
    
    # PHASE 2: Fine-tuning (dégel des 50 dernières couches du backbone)
    print(f"\n[PHASE 2] Fine-tuning backbone ({args.epochs_phase2} époques)")
    densenet_backbone = model.get_layer('densenet121_backbone')
    for layer in densenet_backbone.layers[-50:]:
        layer.trainable = True
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.initial_lr / 10),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    
    history2 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_phase2, callbacks=callbacks, verbose=1)
    
    # Sauvegarder historiques combinés
    hist_df = pd.concat([pd.DataFrame(history1.history), pd.DataFrame(history2.history)], ignore_index=True)
    hist_df.to_csv(os.path.join(output_dir, 'species_history.csv'), index=False)
    
    print(f"[INFO] Species model saved: {model_path}")
    return model


def train_disease_model_global(train_paths, train_species_labels, train_disease_labels,
                               val_paths, val_species_labels, val_disease_labels,
                               num_species, num_diseases, output_dir, args):
    """Entraîne UN SEUL modèle MALADIE global (ResNet50V2+Attention) avec espèce en entrée."""
    print("\n" + "="*60)
    print("ÉTAPE 2: Modèle MALADIE GLOBAL (ResNet50V2+Attention)")
    print(f"  - {num_diseases} maladies")
    print(f"  - Espèce donnée en entrée (GT pendant entraînement)")
    print("="*60)
    
    # Créer datasets avec TOUTES les images + leur espèce GT
    train_ds = make_dataset_disease(train_paths, train_species_labels, train_disease_labels,
                                   tuple(args.img_size), args.batch_size,
                                   num_species, num_diseases, True, SEED)
    
    val_ds = make_dataset_disease(val_paths, val_species_labels, val_disease_labels,
                                 tuple(args.img_size), args.batch_size,
                                 num_species, num_diseases, False, SEED)
    
    # Construire modèle ResNet50V2 + Attention
    model = build_disease_model_resnet_attention(num_diseases, num_species,
                                                tuple(args.img_size),
                                                args.initial_lr, args.weight_decay)
    
    model_path = os.path.join(output_dir, 'disease_model_global.keras')
    callbacks = [
        MacroF1Callback(val_ds, val_disease_labels),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                       save_best_only=True, mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience,
                                     restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                         patience=args.patience//2, min_lr=1e-7, verbose=1)
    ]
    
    # PHASE 1: Head only (backbone gelé)
    print(f"\n[PHASE 1] Head only ({args.epochs_disease_p1} époques)")
    history1 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_disease_p1, callbacks=callbacks, verbose=1)
    
    # PHASE 2: Fine-tuning (dégel des 100 dernières couches du ResNet50V2)
    print(f"\n[PHASE 2] Fine-tuning ResNet50V2 ({args.epochs_disease_p2} époques)")
    resnet_backbone = model.get_layer('resnet50v2_backbone')
    for layer in resnet_backbone.layers[-100:]:
        layer.trainable = True
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.initial_lr / 10),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    
    history2 = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs_disease_p2, callbacks=callbacks, verbose=1)
    
    # Sauvegarder historiques combinés
    hist_df = pd.concat([pd.DataFrame(history1.history), pd.DataFrame(history2.history)], ignore_index=True)
    hist_df.to_csv(os.path.join(output_dir, 'disease_history_global.csv'), index=False)
    
    print(f"[INFO] Disease model saved: {model_path}")
    return model


def evaluate_cascade(species_model, disease_model, test_paths, test_species_labels, test_disease_labels,
                     num_species, num_diseases, idx_to_species, idx_to_disease, output_dir, args):
    """Évalue la VRAIE cascade: espèce prédite → maladie prédite."""
    print("\n" + "="*60)
    print("ÉVALUATION CASCADE (VRAIE)")
    print("  - Step 1: Prédire espèce")
    print("  - Step 2: Prédire maladie avec espèce PRÉDITE")
    print("="*60)
    
    # STEP 1: Prédictions espèce
    test_ds_species = make_dataset_species(test_paths, test_species_labels,
                                          tuple(args.img_size), args.batch_size,
                                          num_species, False, SEED)
    
    print("\n[Step 1] Prédiction espèce...")
    species_probs = species_model.predict(test_ds_species, verbose=1)
    species_pred = np.argmax(species_probs, axis=1)
    
    # STEP 2: Prédictions maladie avec espèce PRÉDITE (pas GT!)
    print("\n[Step 2] Prédiction maladie avec espèce PRÉDITE...")
    test_ds_disease_cascade = make_dataset_disease(test_paths, species_pred, test_disease_labels,
                                                   tuple(args.img_size), args.batch_size,
                                                   num_species, num_diseases, False, SEED)
    
    disease_probs = disease_model.predict(test_ds_disease_cascade, verbose=1)
    disease_pred = np.argmax(disease_probs, axis=1)
    
    # Évaluation espèce seule
    species_acc = accuracy_score(test_species_labels, species_pred)
    species_f1 = f1_score(test_species_labels, species_pred, average='macro')
    
    # Évaluation maladie seule
    disease_acc = accuracy_score(test_disease_labels, disease_pred)
    disease_f1 = f1_score(test_disease_labels, disease_pred, average='macro')
    
    # Évaluation cascade (& logique)
    cascade_correct = (species_pred == np.array(test_species_labels)) & (disease_pred == np.array(test_disease_labels))
    cascade_acc = cascade_correct.mean()
    
    # F1 cascade: sur les VRAIES classes finales (espèce___maladie)
    # Ceci pénalise correctement quand l'espèce est fausse (→ classe finale fausse)
    true_final_classes = [f"{idx_to_species[sp]}___{idx_to_disease[dis]}" 
                         for sp, dis in zip(test_species_labels, test_disease_labels)]
    pred_final_classes = [f"{idx_to_species[sp]}___{idx_to_disease[dis]}" 
                         for sp, dis in zip(species_pred, disease_pred)]
    
    cascade_f1_macro = f1_score(true_final_classes, pred_final_classes, average='macro', zero_division=0)
    cascade_f1_weighted = f1_score(true_final_classes, pred_final_classes, average='weighted', zero_division=0)
    
    # Rapport détaillé par classe finale
    cascade_report = classification_report(true_final_classes, pred_final_classes, 
                                          output_dict=True, zero_division=0)
    cascade_report_df = pd.DataFrame(cascade_report).T
    cascade_report_df.to_csv(os.path.join(output_dir, 'cascade_final_classes_report.csv'))
    
    print(f"\n{'='*60}")
    print("RÉSULTATS")
    print(f"{'='*60}")
    print(f"Espèce seule       - Accuracy: {species_acc:.4f}, Macro-F1: {species_f1:.4f}")
    print(f"Maladie seule      - Accuracy: {disease_acc:.4f}, Macro-F1: {disease_f1:.4f}")
    print(f"CASCADE (finale)   - Accuracy: {cascade_acc:.4f}")
    print(f"                   - Macro-F1: {cascade_f1_macro:.4f} (toutes classes égales)")
    print(f"                   - Weighted-F1: {cascade_f1_weighted:.4f} (pondéré par support)")
    print(f"\n[INFO] Rapport détaillé par classe: cascade_final_classes_report.csv")
    
    # Sauvegarder
    results = {
        'species_accuracy': float(species_acc),
        'species_f1': float(species_f1),
        'disease_accuracy': float(disease_acc),
        'disease_f1': float(disease_f1),
        'cascade_accuracy': float(cascade_acc),
        'cascade_f1_macro': float(cascade_f1_macro),
        'cascade_f1_weighted': float(cascade_f1_weighted),
    }
    
    with open(os.path.join(output_dir, 'cascade_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rapport espèce
    species_names = [idx_to_species[i] for i in range(num_species)]
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


def evaluate_oracle(disease_model, test_paths, test_species_labels, test_disease_labels,
                    num_species, num_diseases, output_dir, args):
    """Évalue le modèle de maladie avec espèce GT (performance théorique)."""
    print("\n" + "="*60)
    print("ÉVALUATION ORACLE (espèce GT)")
    print("  - Montre la performance si l'espèce était parfaite")
    print("="*60)
    
    # Prédictions maladie avec espèce GT
    test_ds_disease_oracle = make_dataset_disease(test_paths, test_species_labels, test_disease_labels,
                                                  tuple(args.img_size), args.batch_size,
                                                  num_species, num_diseases, False, SEED)
    
    disease_probs = disease_model.predict(test_ds_disease_oracle, verbose=1)
    disease_pred = np.argmax(disease_probs, axis=1)
    
    oracle_acc = accuracy_score(test_disease_labels, disease_pred)
    oracle_f1 = f1_score(test_disease_labels, disease_pred, average='macro')
    
    print(f"\n{'='*60}")
    print("RÉSULTATS ORACLE")
    print(f"{'='*60}")
    print(f"Maladie (espèce GT) - Accuracy: {oracle_acc:.4f}, Macro-F1: {oracle_f1:.4f}")
    
    results = {
        'oracle_disease_accuracy': float(oracle_acc),
        'oracle_disease_f1': float(oracle_f1),
    }
    
    with open(os.path.join(output_dir, 'oracle_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cascade amélioré: DenseNet121 (espèce) + ResNet50V2+Attention (maladie)")
    parser.add_argument('--data_root', type=str, 
                       default='/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented',
                       help='Chemin dataset PlantVillage')
    parser.add_argument('--output_dir', type=str, default='outputs/cascade_improved',
                       help='Répertoire de sortie')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                       help='Taille des images')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs_phase1', type=int, default=10,
                       help='Époques phase 1 espèce (head only)')
    parser.add_argument('--epochs_phase2', type=int, default=30,
                       help='Époques phase 2 espèce (fine-tuning)')
    parser.add_argument('--epochs_disease_p1', type=int, default=10,
                       help='Époques phase 1 maladie (head only)')
    parser.add_argument('--epochs_disease_p2', type=int, default=40,
                       help='Époques phase 2 maladie (fine-tuning)')
    parser.add_argument('--initial_lr', type=float, default=1e-3,
                       help='Learning rate initial')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay L2')
    parser.add_argument('--patience', type=int, default=7,
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
    
    # ÉTAPE 2: Entraîner UN modèle maladie global (avec espèce GT en entrée)
    disease_model = train_disease_model_global(
        train_paths, train_species_labels, train_disease_labels,
        val_paths, val_species_labels, val_disease_labels,
        len(species_to_idx), len(disease_to_idx),
        args.output_dir, args
    )
    
    # ÉTAPE 3: Évaluer ORACLE (espèce GT)
    oracle_results = evaluate_oracle(
        disease_model,
        test_paths, test_species_labels, test_disease_labels,
        len(species_to_idx), len(disease_to_idx),
        args.output_dir, args
    )
    
    # ÉTAPE 4: Évaluer CASCADE (espèce PRÉDITE - vraie performance)
    cascade_results = evaluate_cascade(
        species_model, disease_model,
        test_paths, test_species_labels, test_disease_labels,
        len(species_to_idx), len(disease_to_idx),
        idx_to_species, idx_to_disease,
        args.output_dir, args
    )
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT CASCADE TERMINÉ")
    print("="*60)
    print(f"Résultats dans: {args.output_dir}")
    print(f"  - species_model.keras")
    print(f"  - disease_model_global.keras")
    print(f"  - cascade_results.json (VRAIE performance)")
    print(f"  - oracle_results.json (performance théorique)")
    print(f"  - confusion_species.png")
    print("\n")


if __name__ == '__main__':
    main()