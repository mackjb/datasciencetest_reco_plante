#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'analyse de l'importance des caractéristiques pour la classification des espèces de plantes.
Utilise différentes stratégies de sélection de caractéristiques pour comparer leur performance.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torchvision import transforms
from PIL import Image

from joblib import Memory, Parallel, delayed
import time

# Import des fonctions existantes
from src.helpers.helpers import (
    PROJECT_ROOT, 
    compute_hu_features, 
    compute_fourier_energy, 
    compute_hog_features, 
    compute_pixel_ratio_and_segments
)
from src.data_loader.data_loader import FEATURE_COLS, HANDCRAFTED_FEATURE_COLS

# -----------------------------
# 1. Configuration & Constants
# -----------------------------
# Chemins
DATA_DIR = PROJECT_ROOT / "dataset" / "plantvillage" / "data"
CSV_FILE = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all.csv"
TARGET_COLUMN = "species"  # Cible: espèce de la plante
PATH_COLUMN = "filepath"   # Chemin de l'image

# Colonnes de caractéristiques (reprises du module existant)
FEATURE_COLUMNS = HANDCRAFTED_FEATURE_COLS  # Caractéristiques extraites des images

# Autres paramètres
RANDOM_STATE = 42
CACHE_DIR = PROJECT_ROOT / "cache"  # Dossier pour la mise en cache des calculs
CACHE_DIR.mkdir(exist_ok=True)

# Configuration du cache pour joblib.Memory
memory = Memory(CACHE_DIR, verbose=0)

# -----------------------------
# 2. Load DataFrame & Split
# -----------------------------
@memory.cache
def load_and_split_data(csv_path=CSV_FILE, target_col=TARGET_COLUMN, test_size=0.2, random_state=RANDOM_STATE):
    """
    Charge le DataFrame et effectue une division stratifiée.
    Mise en cache pour éviter de recharger les données à chaque exécution.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        target_col: Colonne cible pour la stratification
        test_size: Proportion du jeu de test
        random_state: État aléatoire pour la reproductibilité
        
    Returns:
        df_train, df_test: DataFrames d'entraînement et de test
    """
    print(f"Chargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Vérifier que la colonne cible existe
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame. "
                        f"Colonnes disponibles: {', '.join(df.columns)}")
    
    # Afficher la distribution des classes
    class_counts = df[target_col].value_counts()
    print(f"Distribution des classes ({len(class_counts)} classes):")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} exemples")
    
    # Division stratifiée
    print(f"Division stratifiée en train ({1-test_size:.0%}) / test ({test_size:.0%})...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(df, df[target_col]))
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)
    
    print(f"Train set: {len(df_train)} exemples")
    print(f"Test set: {len(df_test)} exemples")
    
    return df_train, df_test

# -----------------------------
# 3. Data Augmentation for Minority
# -----------------------------
@memory.cache
def extract_features_from_image(img_path):
    """
    Extrait toutes les caractéristiques d'une image en utilisant les fonctions existantes.
    
    Args:
        img_path: Chemin vers l'image
        
    Returns:
        dict: Dictionnaire des caractéristiques
    """
    try:
        # Ouvrir l'image
        img = Image.open(img_path)
        
        # Extraire les caractéristiques en utilisant les fonctions existantes
        features = {}
        features.update(compute_hu_features(img))
        features.update(compute_fourier_energy(img))
        features.update(compute_hog_features(img))
        features.update(compute_pixel_ratio_and_segments(img))
        
        return features
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {img_path}: {e}")
        return None

def augment_minority(df_train, data_dir=DATA_DIR, path_col=PATH_COLUMN, target_col=TARGET_COLUMN):
    """
    Augmente la classe minoritaire pour équilibrer le train set.
    Utilise les transformations de torchvision et recalcule les caractéristiques.
    
    Args:
        df_train: DataFrame d'entraînement
        data_dir: Répertoire racine des données
        path_col: Nom de la colonne contenant le chemin de l'image
        target_col: Nom de la colonne cible
        
    Returns:
        DataFrame étendu avec les mêmes colonnes + caractéristiques recalculées
    """
    print("Augmentation des classes minoritaires...")
    
    # Définir les transformations pour l'augmentation
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Compter les occurrences par classe
    counts = df_train[target_col].value_counts()
    max_count = counts.max()
    augmented_rows = []
    
    # Ajouter toutes les lignes originales
    augmented_rows.append(df_train)
    
    # Pour chaque classe minoritaire
    for cls, cnt in counts.items():
        subset = df_train[df_train[target_col] == cls]
        
        if cnt < max_count:
            to_generate = max_count - cnt
            print(f"Classe {cls}: génération de {to_generate} exemples supplémentaires...")
            
            # Nombre de répétitions nécessaires
            reps = int(np.ceil(to_generate / cnt))
            num_generated = 0
            
            for _ in range(reps):
                if num_generated >= to_generate:
                    break
                    
                for _, row in subset.iterrows():
                    if num_generated >= to_generate:
                        break
                        
                    img_path = row[path_col]
                    try:
                        img = Image.open(img_path)
                        img_aug = augment(img)
                        
                        # Extraire les caractéristiques de l'image augmentée
                        feats = extract_features_from_image(img)
                        if feats is None:
                            continue
                            
                        # Créer une nouvelle ligne
                        new_row = row.copy()
                        for k, v in feats.items():
                            new_row[k] = v
                            
                        # Ajouter au DataFrame
                        augmented_rows.append(pd.DataFrame([new_row]))
                        num_generated += 1
                        
                    except Exception as e:
                        print(f"Erreur lors de l'augmentation de {img_path}: {e}")
    
    # Concaténer toutes les lignes
    df_aug = pd.concat(augmented_rows, ignore_index=True)
    print(f"Dataset après augmentation: {len(df_aug)} exemples")
    
    # Vérifier la distribution des classes après augmentation
    new_counts = df_aug[target_col].value_counts()
    for cls, count in new_counts.items():
        print(f"  - {cls}: {count} exemples")
        
    return df_aug

# -----------------------------
# 4. Préparation X/y avec parallélisation
# -----------------------------
@memory.cache
def prepare_features(df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN, n_jobs=-1):
    """
    Prépare les matrices X et y à partir du DataFrame.
    Utilise parallélisation pour extraire/vérifier les caractéristiques manquantes.
    
    Args:
        df: DataFrame avec les métadonnées et caractéristiques
        feature_cols: Liste des colonnes de caractéristiques
        target_col: Nom de la colonne cible
        n_jobs: Nombre de jobs parallèles (-1 pour utiliser tous les cœurs)
        
    Returns:
        X, y: Matrices de caractéristiques et vecteur cible
    """
    print("Préparation des caractéristiques...")
    
    # Vérifier les colonnes manquantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
        print("Extraction des caractéristiques manquantes...")
        
        # Fonction pour extraire les caractéristiques d'une ligne
        def extract_features_for_row(row):
            img_path = row[PATH_COLUMN]
            features = extract_features_from_image(img_path)
            return features
        
        # Extraction parallèle des caractéristiques
        start_time = time.time()
        features_list = Parallel(n_jobs=n_jobs)(
            delayed(extract_features_for_row)(row) for _, row in df.iterrows()
        )
        end_time = time.time()
        print(f"Extraction terminée en {end_time - start_time:.2f} secondes")
        
        # Mettre à jour le DataFrame avec les caractéristiques extraites
        for i, features in enumerate(features_list):
            if features is not None:
                for col, val in features.items():
                    df.loc[i, col] = val
    
    # Vérifier à nouveau les colonnes manquantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Impossible d'extraire toutes les caractéristiques requises: {missing_cols}")
    
    # Extraire X et y
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y

# -----------------------------
# 5. Base Classifier & Scaler
# -----------------------------
def create_base_classifier(random_state=RANDOM_STATE):
    """Crée le classifieur de base"""
    return RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state
    )

# -----------------------------
# 6. Define Feature Selectors
# -----------------------------
def create_selectors(random_state=RANDOM_STATE):
    """Crée les différents sélecteurs de caractéristiques"""
    return {
        # Univariate: F-test, keep top 10 features
        'univariate_f': SelectKBest(score_func=f_classif, k=10),
        
        # RFE avec RandomForest
        'rfe_rf': RFE(
            estimator=RandomForestClassifier(n_estimators=100, random_state=random_state),
            n_features_to_select=10
        ),
        
        # Model-based: L1 LogReg
        'sfm_l1': SelectFromModel(
            estimator=LogisticRegression(
                penalty='l1', solver='liblinear', random_state=random_state
            ),
            threshold='median'
        ),
        
        # Model-based: Tree
        'sfm_tree': SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=random_state),
            threshold='median'
        )
    }

# -----------------------------
# 7. Evaluation Function
# -----------------------------
def evaluate_selectors(X, y, selectors, classifier, scaler=StandardScaler(), cv=5, random_state=RANDOM_STATE):
    """
    Compare plusieurs stratégies de sélection via CV.
    
    Args:
        X: Matrice de caractéristiques
        y: Vecteur cible
        selectors: Dictionnaire de sélecteurs de caractéristiques
        classifier: Classifieur de base
        scaler: Scaler pour normaliser les données
        cv: Nombre de folds pour la validation croisée
        random_state: État aléatoire pour la reproductibilité
        
    Returns:
        dict: Résultats des différentes stratégies
    """
    results = {}
    feature_importances = {}
    selected_features_idx = {}
    
    # Pour chaque sélecteur
    for name, selector in selectors.items():
        print(f"\nÉvaluation du sélecteur: {name}")
        
        # Créer le pipeline
        pipe = Pipeline([
            ('scaler', scaler),
            ('feat_sel', selector),
            ('clf', classifier)
        ])
        
        # Évaluation par validation croisée
        cv_results = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=True,
            return_estimator=True  # Pour récupérer les modèles entraînés
        )
        
        # Stocker les résultats moyens
        results[name] = {
            'accuracy': np.mean(cv_results['test_accuracy']),
            'f1_macro': np.mean(cv_results['test_f1_macro']),
            'precision_macro': np.mean(cv_results['test_precision_macro']),
            'recall_macro': np.mean(cv_results['test_recall_macro']),
            'train_accuracy': np.mean(cv_results['train_accuracy']),
            'train_f1_macro': np.mean(cv_results['train_f1_macro']),
        }
        
        # Récupérer les caractéristiques sélectionnées pour chaque fold
        selected_features = []
        importances = []
        
        for estimator in cv_results['estimator']:
            # Récupérer l'index des caractéristiques sélectionnées
            if name == 'univariate_f':
                mask = estimator.named_steps['feat_sel'].get_support()
            elif name == 'rfe_rf':
                mask = estimator.named_steps['feat_sel'].get_support()
            elif name in ['sfm_l1', 'sfm_tree']:
                mask = estimator.named_steps['feat_sel'].get_support()
            else:
                mask = np.ones(X.shape[1], dtype=bool)
                
            selected_features.append(mask)
            
            # Récupérer les importances des caractéristiques du classifieur final
            if hasattr(estimator.named_steps['clf'], 'feature_importances_'):
                # Pour les modèles basés sur les arbres (RandomForest)
                # Ces importances sont calculées sur les caractéristiques sélectionnées
                fold_importances = estimator.named_steps['clf'].feature_importances_
                
                # Créer un vecteur d'importances de la taille originale
                full_importances = np.zeros(X.shape[1])
                full_importances[mask] = fold_importances
                importances.append(full_importances)
        
        # Calculer la fréquence de sélection pour chaque caractéristique
        selection_frequency = np.mean(selected_features, axis=0)
        selected_features_idx[name] = selection_frequency
        
        # Calculer l'importance moyenne des caractéristiques si disponible
        if importances:
            feature_importances[name] = np.mean(importances, axis=0)
        
    return results, feature_importances, selected_features_idx

# -----------------------------
# 8. Affichage des résultats
# -----------------------------
def print_results(results, feature_importances, selected_features_idx, feature_cols):
    """
    Affiche les résultats de l'évaluation et l'importance des caractéristiques.
    
    Args:
        results: Résultats des différentes stratégies
        feature_importances: Importances des caractéristiques pour chaque stratégie
        selected_features_idx: Index des caractéristiques sélectionnées
        feature_cols: Noms des colonnes de caractéristiques
    """
    print("\n" + "="*50)
    print("RÉSULTATS DE LA SÉLECTION DE CARACTÉRISTIQUES")
    print("="*50)
    
    # Afficher les métriques de performance
    print("\nMétriques de performance:")
    for name, scores in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Test - Accuracy: {scores['accuracy']:.3f}")
        print(f"  Test - F1 macro: {scores['f1_macro']:.3f}")
        print(f"  Test - Precision macro: {scores['precision_macro']:.3f}")
        print(f"  Test - Recall macro: {scores['recall_macro']:.3f}")
        print(f"  Train - Accuracy: {scores['train_accuracy']:.3f}")
        print(f"  Train - F1 macro: {scores['train_f1_macro']:.3f}")
    
    # Afficher les caractéristiques les plus importantes pour chaque stratégie
    print("\nCaractéristiques importantes par stratégie:")
    for name, freq in selected_features_idx.items():
        print(f"\n{name.upper()} - Fréquence de sélection:")
        
        # Trier par fréquence de sélection
        indices = np.argsort(-freq)
        for i in indices[:10]:  # Top 10
            if freq[i] > 0:
                print(f"  {feature_cols[i]}: {freq[i]:.2f}")
    
    # Si les importances sont disponibles (pour les méthodes basées sur les arbres)
    print("\nImportance des caractéristiques (si disponible):")
    for name, importances in feature_importances.items():
        if importances is not None and np.any(importances):
            print(f"\n{name.upper()} - Feature Importance:")
            
            # Trier par importance
            indices = np.argsort(-importances)
            for i in indices[:10]:  # Top 10
                if importances[i] > 0:
                    print(f"  {feature_cols[i]}: {importances[i]:.4f}")

# -----------------------------
# 9. Main Function
# -----------------------------
def main():
    """
    Fonction principale qui exécute l'ensemble du processus.
    """
    print("\n" + "="*50)
    print("ANALYSE DE L'IMPORTANCE DES CARACTÉRISTIQUES")
    print("="*50)
    
    # 1. Charger les données et faire la division train/test
    df_train, df_test = load_and_split_data()
    
    # 2. Vérifier si on doit augmenter les données
    # Note: L'augmentation est commentée car elle peut être coûteuse en temps
    # df_train_aug = augment_minority(df_train)
    df_train_aug = df_train  # Pour le moment, on utilise le train set original
    
    # 3. Préparer X/y pour train et test
    X_train, y_train = prepare_features(df_train_aug)
    X_test, y_test = prepare_features(df_test)
    
    # Combiner pour l'évaluation complète
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    
    # 4. Créer les objets nécessaires
    base_clf = create_base_classifier()
    scaler = StandardScaler()
    selectors = create_selectors()
    
    # 5. Évaluer les sélecteurs
    results, feature_importances, selected_features_idx = evaluate_selectors(
        X_all, y_all, selectors, base_clf, scaler
    )
    
    # 6. Afficher les résultats
    print_results(results, feature_importances, selected_features_idx, FEATURE_COLUMNS)
    
    return results, feature_importances, selected_features_idx

# -----------------------------
# 10. Entry Point
# -----------------------------
if __name__ == '__main__':
    start_time = time.time()
    results, feature_importances, selected_features_idx = main()
    end_time = time.time()
    print(f"\nTemps total d'exécution: {end_time - start_time:.2f} secondes")
