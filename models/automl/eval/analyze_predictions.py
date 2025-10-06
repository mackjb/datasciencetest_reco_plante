#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyse et affiche les prédictions du modèle optimisé
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
SEED = 42
TEST_SIZE = 0.2

# Chemins
base_dir = Path("/workspaces/datasciencetest_reco_plante")
data_path = base_dir / "dataset/plantvillage/csv/clean_with_features_data_plantvillage_segmented_all.csv"
output_dir = base_dir / "results/predictions"
output_dir.mkdir(parents=True, exist_ok=True)

def load_data_and_model(target_type):
    """Charge les données et le modèle pour un type de cible donné."""
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Préparer les cibles
    if target_type == 'espece':
        target_cols = [col for col in df.columns if col.startswith('plant_')]
        y = df[target_cols].idxmax(axis=1).str.replace('plant_', '')
    else:  # maladie
        target_cols = [col for col in df.columns if col.startswith('disease_')]
        y = df[target_cols].idxmax(axis=1).str.replace('disease_', '')
    
    # Sélectionner les caractéristiques
    exclude_cols = target_cols + ['ID_Image', 'Est_Saine', 'Image_Path', 'is_black', 'dimensions']
    X = df[[col for col in df.select_dtypes(include=np.number).columns 
            if col not in exclude_cols]]
    
    # Charger le modèle et le scaler
    model_path = base_dir / f"results/fast_optimization/best_model_{target_type}.pkl"
    scaler_path = base_dir / f"results/fast_optimization/scaler_{target_type}.pkl"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return X, y, model, scaler, target_cols

def analyze_predictions(X, y, model, scaler, target_cols, target_type):
    """Analyse les prédictions et génère des visualisations."""
    # Prétraitement
    X_scaled = scaler.transform(X)
    
    # Prédictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Créer un DataFrame avec les résultats
    # Convertir les étiquettes en chaînes de caractères
    y_str = y.astype(str)
    predicted_labels = [str(model.classes_[i]) for i in y_pred]
    
    results = pd.DataFrame({
        'true_label': y_str,
        'predicted_label': predicted_labels,
        'confidence': np.max(y_proba, axis=1)
    })
    
    # Ajouter les probabilités pour chaque classe
    for i, class_name in enumerate(model.classes_):
        results[f'prob_{class_name}'] = y_proba[:, i]
    
    # Sauvegarder les résultats
    results_path = output_dir / f"predictions_{target_type}.csv"
    results.to_csv(results_path, index=False)
    print(f"Prédictions sauvegardées dans: {results_path}")
    
    # Calculer la précision globale
    accuracy = (results['true_label'] == results['predicted_label']).mean()
    
    # Afficher les métriques de base
    print(f"\n=== PERFORMANCES DU MODÈLE ({target_type.upper()}) ===")
    print(f"Précision globale: {accuracy:.2%}")
    
    # Afficher la distribution des vraies étiquettes
    print("\nDistribution des vraies étiquettes:")
    true_dist = results['true_label'].value_counts().sort_values(ascending=False)
    print(true_dist.head(10))  # Afficher les 10 premières classes les plus fréquentes
    
    # Afficher la distribution des prédictions
    print("\nDistribution des prédictions:")
    pred_dist = results['predicted_label'].value_counts().sort_values(ascending=False)
    print(pred_dist.head(10))  # Afficher les 10 premières classes prédites les plus fréquentes
    
    # Afficher un échantillon des prédictions
    print("\nExemple de prédictions (5 premières lignes):")
    print(results[['true_label', 'predicted_label', 'confidence']].head())
    
    return results

def main():
    # Pour chaque type de cible
    for target_type in ['espece', 'maladie']:
        print(f"\n{'='*50}")
        print(f"ANALYSE DES PRÉDICTIONS: {target_type.upper()}")
        print(f"{'='*50}")
        
        # Charger les données et le modèle
        X, y, model, scaler, target_cols = load_data_and_model(target_type)
        
        # Analyser les prédictions
        results = analyze_predictions(X, y, model, scaler, target_cols, target_type)
        
        # Afficher un échantillon des prédictions
        print("\nExemple de prédictions:")
        print(results.sample(5, random_state=SEED)[['true_label', 'predicted_label', 'confidence']])

if __name__ == "__main__":
    main()
