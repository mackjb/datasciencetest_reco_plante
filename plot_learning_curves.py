#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualisation des courbes d'apprentissage pour XGBoost

Ce script permet de visualiser la fonction de perte pendant l'entraînement
pour détecter le surapprentissage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import json
from typing import Dict, List, Tuple
import os

# Configuration
SEED = 42
TEST_SIZE = 0.2
EVAL_SET_SIZE = 0.2  # Pour la validation pendant l'entraînement

def load_and_prepare_data(csv_path: Path, target_type: str) -> tuple:
    """Charge et prépare les données pour l'entraînement.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        target_type: Type de cible - 'espece' ou 'maladie'
    """
    print(f"Chargement des données depuis {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Déterminer les colonnes cibles et caractéristiques
    if target_type == 'espece':
        # Colonnes commençant par 'plant_'
        target_cols = [col for col in df.columns if col.startswith('plant_')]
        # Créer la colonne cible (une seule classe par échantillon)
        y = df[target_cols].idxmax(axis=1).str.replace('plant_', '')
    else:  # maladie
        # Colonnes commençant par 'disease_'
        target_cols = [col for col in df.columns if col.startswith('disease_')]
        # Créer la colonne cible (une seule maladie par échantillon)
        y = df[target_cols].idxmax(axis=1).str.replace('disease_', '')
    
    # Encodage des labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_.tolist()
    
    # Sélection des caractéristiques numériques
    # Exclure les colonnes de cibles et autres colonnes non pertinentes
    exclude_cols = target_cols + ['ID_Image', 'Est_Saine', 'Image_Path', 'is_black', 'dimensions']
    numeric_columns = [col for col in df.select_dtypes(include=np.number).columns 
                      if col not in exclude_cols]
    
    # Nettoyage des données
    X = df[numeric_columns].fillna(df[numeric_columns].median()).values
    
    return X, y_encoded, classes, numeric_columns

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
               X_val: np.ndarray, y_val: np.ndarray,
               n_estimators: int = 100) -> Tuple[XGBClassifier, Dict]:
    """
    Entraîne un modèle XGBoost et retourne le modèle et l'historique des pertes.
    """
    # Configuration du modèle
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        n_jobs=-1,
        random_state=SEED,
        use_label_encoder=False
    )
    
    # Entraînement simple
    model.fit(X_train, y_train)
    
    # Calcul des prédictions sur l'ensemble d'entraînement et de validation
    y_train_pred = model.predict_proba(X_train)
    y_val_pred = model.predict_proba(X_val)
    
    # Calcul de la perte (log loss)
    train_loss = log_loss(y_train, y_train_pred)
    val_loss = log_loss(y_val, y_val_pred)
    
    # Création d'un dictionnaire de résultats similaire à evals_result()
    results = {
        'validation_0': {'mlogloss': [train_loss]},
        'validation_1': {'mlogloss': [val_loss]}
    }
    
    print(f"\nRésultats d'entraînement:")
    print(f"- Perte d'entraînement: {train_loss:.4f}")
    print(f"- Perte de validation: {val_loss:.4f}")
    
    return model, results

def plot_learning_curves(results: Dict, title: str, save_path: Path) -> None:
    """Affiche les courbes d'apprentissage (train/validation loss)."""
    plt.figure(figsize=(12, 6))
    
    # Tracé des courbes
    plt.bar([0, 1], 
            [results['validation_0']['mlogloss'][0], results['validation_1']['mlogloss'][0]],
            tick_label=['Train', 'Validation'])
    
    # Configuration du graphique
    plt.title(f'Comparaison des pertes - {title}')
    plt.ylabel('Perte (logloss)')
    plt.grid(True, axis='y')
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate([results['validation_0']['mlogloss'][0], results['validation_1']['mlogloss'][0]]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # Sauvegarde
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique sauvegardé: {save_path}")

def main():
    # Configuration des chemins
    base_dir = Path("/workspaces/datasciencetest_reco_plante")
    csv_path = base_dir / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
    output_dir = base_dir / "results" / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Types de cibles à analyser
    target_types = ["espece", "maladie"]
    # Analyser les deux types de cibles séparément
    for target_type in target_types:
        print(f"\n{'='*50}")
        print(f"Analyse pour la cible: {target_type}")
        print(f"{'='*50}")
        
        # Chargement et préparation des données
        X, y, classes, feature_names = load_and_prepare_data(csv_path, target_type)
        
        # Afficher des informations sur les données
        unique_classes = len(classes)
        print(f"Nombre de classes uniques: {unique_classes}")
        print(f"Nombre de caractéristiques: {len(feature_names)}")
        print(f"Taille de l'ensemble de données: {X.shape[0]} échantillons")
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
    
        # Normalisation
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Vérification des dimensions
        print(f"\nDimensions des données pour {target_type}:")
        print(f"- Train: {X_train_scaled.shape}")
        print(f"- Test: {X_test_scaled.shape}")
    
        # Division supplémentaire pour la validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, 
            test_size=EVAL_SET_SIZE, 
            stratify=y_train,
            random_state=SEED
        )
        
        print(f"\nFormes des données après split de validation:")
        print(f"- Train final: {X_train_final.shape}")
        print(f"- Validation: {X_val.shape}")
        print(f"- Test: {X_test_scaled.shape}")
        
        # Entraînement du modèle
        print(f"\nEntraînement du modèle XGBoost pour {target_type}...")
        model, results = train_model(
            X_train_final, y_train_final,
            X_val, y_val,
            n_estimators=100
        )
        
        # Visualisation des courbes d'apprentissage
        plot_learning_curves(
            results,
            title=f"XGBoost - {target_type}",
            save_path=output_dir / f"learning_curves_{target_type}.png"
        )
        
        # Évaluation finale sur le test
        y_pred_proba = model.predict_proba(X_test_scaled)
        test_loss = log_loss(y_test, y_pred_proba)
        print(f"\nPerte finale sur l'ensemble de test ({target_type}): {test_loss:.4f}")
        
        # Sauvegarde du modèle
        model_save_path = output_dir / f"xgboost_model_{target_type}.json"
        model.save_model(model_save_path)
        print(f"Modèle sauvegardé: {model_save_path}")

if __name__ == "__main__":
    main()
