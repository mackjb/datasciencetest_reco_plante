#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimisation XGBoost rapide avec recherche bayésienne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import log_loss, classification_report
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from tqdm import tqdm
import warnings
import joblib
warnings.filterwarnings("ignore")

# Configuration rapide
SEED = 42
TEST_SIZE = 0.2
N_ITER = 10  # Réduit de 30 à 10 itérations
CV_FOLDS = 2  # Réduit de 3 à 2 folds
SUBSAMPLE_SIZE = 0.5  # Utiliser 50% des données

# Créer le répertoire de sortie
output_dir = Path("results/fast_optimization")
output_dir.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data(csv_path, target_type, subsample=1.0):
    """Charge et prépare les données avec sous-échantillonnage optionnel."""
    print(f"\nChargement des données pour {target_type}...")
    df = pd.read_csv(csv_path)
    
    # Sous-échantillonnage aléatoire
    if subsample < 1.0:
        df = df.sample(frac=subsample, random_state=SEED)
    
    # Sélection des cibles
    if target_type == 'espece':
        target_cols = [col for col in df.columns if col.startswith('plant_')]
        y = df[target_cols].idxmax(axis=1).str.replace('plant_', '')
    else:  # maladie
        target_cols = [col for col in df.columns if col.startswith('disease_')]
        y = df[target_cols].idxmax(axis=1).str.replace('disease_', '')
    
    # Encodage des labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_.tolist()
    
    # Sélection des caractéristiques
    exclude_cols = target_cols + ['ID_Image', 'Est_Saine', 'Image_Path', 'is_black', 'dimensions']
    numeric_columns = [col for col in df.select_dtypes(include=np.number).columns 
                      if col not in exclude_cols]
    
    # Nettoyage des données
    X = df[numeric_columns].fillna(df[numeric_columns].median()).values
    
    print(f"- Classes: {len(classes)}")
    print(f"- Caractéristiques: {len(numeric_columns)}")
    print(f"- Échantillons: {X.shape[0]}")
    
    return X, y_encoded, classes, numeric_columns

def run_optimization(X, y, target_type):
    """Exécute l'optimisation bayésienne."""
    # Espace de recherche réduit
    search_spaces = {
        'learning_rate': Real(0.01, 0.2, 'log-uniform'),
        'max_depth': Integer(3, 6),
        'min_child_weight': Integer(1, 5),
        'subsample': Real(0.7, 1.0, 'uniform'),
        'colsample_bytree': Real(0.7, 1.0, 'uniform'),
        'gamma': Real(0, 0.3, 'uniform'),
        'reg_alpha': Real(1e-5, 1, 'log-uniform'),
        'reg_lambda': Real(1, 5, 'log-uniform')
    }
    
    # Modèle de base
    model = XGBClassifier(
        n_estimators=100,
        objective='multi:softprob',
        n_jobs=-1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Optimisation bayésienne
    opt = BayesSearchCV(
        model,
        search_spaces,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='neg_log_loss',
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
        return_train_score=True
    )
    
    print(f"\nDémarrage de l'optimisation pour {target_type}...")
    opt.fit(X, y)
    
    return opt

def main():
    # Chemin des données
    csv_path = Path("dataset/plantvillage/csv/clean_with_features_data_plantvillage_segmented_all.csv")
    
    # Pour chaque type de cible
    for target_type in ['espece', 'maladie']:
        print(f"\n{'='*50}")
        print(f"TRAITEMENT: {target_type.upper()}")
        print(f"{'='*50}")
        
        # Chargement des données
        X, y, classes, features = load_and_prepare_data(
            csv_path, target_type, subsample=SUBSAMPLE_SIZE
        )
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        
        # Normalisation
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Sauvegarde du scaler
        joblib.dump(scaler, output_dir / f'scaler_{target_type}.pkl')
        
        # Optimisation
        opt = run_optimization(X_train_scaled, y_train, target_type)
        
        # Sauvegarde du modèle
        model_path = output_dir / f'best_model_{target_type}.pkl'
        joblib.dump(opt.best_estimator_, model_path)
        
        # Évaluation
        y_pred = opt.predict(X_test_scaled)
        y_proba = opt.predict_proba(X_test_scaled)
        
        # Métriques
        test_loss = log_loss(y_test, y_proba)
        
        # Affichage des résultats
        print(f"\nRésultats pour {target_type}:")
        print(f"- Meilleur score (log loss): {-opt.best_score_:.4f}")
        print(f"- Perte sur le test: {test_loss:.4f}")
        
        # Sauvegarde des résultats
        results = {
            'best_params': opt.best_params_,
            'best_score': opt.best_score_,
            'test_loss': test_loss,
            'feature_importance': opt.best_estimator_.feature_importances_.tolist(),
            'features': features,
            'classes': classes
        }
        
        joblib.dump(results, output_dir / f'results_{target_type}.pkl')
        
        # Affichage des meilleurs paramètres
        print("\nMeilleurs paramètres:")
        for param, value in opt.best_params_.items():
            print(f"- {param}: {value}")
        
        print(f"\nModèle et résultats sauvegardés dans {output_dir}")

if __name__ == "__main__":
    main()
