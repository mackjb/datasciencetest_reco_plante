#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimisation bayésienne des hyperparamètres XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import log_loss, classification_report
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
import warnings
warnings.filterwarnings("ignore")

# Configuration
SEED = 42
TEST_SIZE = 0.2
EVAL_SET_SIZE = 0.2  # Taille de l'ensemble de validation
N_ITER = 30  # Nombre d'itérations de la recherche bayésienne
CV_FOLDS = 3  # Nombre de folds pour la validation croisée

def load_and_prepare_data(csv_path: Path, target_type: str) -> tuple:
    """Charge et prépare les données pour l'entraînement."""
    print(f"\nChargement des données depuis {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Déterminer les colonnes cibles
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
    
    # Sélection des caractéristiques numériques
    exclude_cols = target_cols + ['ID_Image', 'Est_Saine', 'Image_Path', 'is_black', 'dimensions']
    numeric_columns = [col for col in df.select_dtypes(include=np.number).columns 
                      if col not in exclude_cols]
    
    # Nettoyage des données
    X = df[numeric_columns].fillna(df[numeric_columns].median()).values
    
    print(f"\nAnalyse pour la cible: {target_type}")
    print(f"- Nombre de classes: {len(classes)}")
    print(f"- Nombre de caractéristiques: {len(numeric_columns)}")
    print(f"- Taille du dataset: {X.shape[0]} échantillons")
    
    return X, y_encoded, classes, numeric_columns

def run_bayesian_optimization(X_train, y_train, X_val, y_val, n_classes):
    """Exécute l'optimisation bayésienne des hyperparamètres."""
    # Définition de l'espace de recherche
    search_spaces = {
        'learning_rate': Real(0.01, 0.2, 'log-uniform'),
        'max_depth': Integer(3, 7),
        'min_child_weight': Integer(1, 10),
        'subsample': Real(0.6, 1.0, 'uniform'),
        'colsample_bytree': Real(0.6, 1.0, 'uniform'),
        'gamma': Real(0, 0.5, 'uniform'),
        'reg_alpha': Real(1e-5, 10, 'log-uniform'),  # Éviter 0 pour log-uniform
        'reg_lambda': Real(1, 10, 'log-uniform')
    }
    
    # Création du modèle de base
    xgb = XGBClassifier(
        n_estimators=200,
        objective='multi:softprob',
        n_jobs=-1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Configuration de la recherche bayésienne
    bayes_search = BayesSearchCV(
        estimator=xgb,
        search_spaces=search_spaces,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='neg_log_loss',
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
        return_train_score=True
    )
    
    print("\nDémarrage de l'optimisation bayésienne...")
    bayes_search.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val])
    )
    
    return bayes_search

def evaluate_model(model, X_train, y_train, X_test, y_test, classes, output_dir, target_type):
    """Évalue le modèle et enregistre les résultats."""
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilités pour le calcul de la perte
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    # Calcul des métriques
    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    
    # Affichage des résultats
    print("\n" + "="*50)
    print(f"Résultats pour {target_type}")
    print("="*50)
    print(f"Perte d'entraînement: {train_loss:.4f}")
    print(f"Perte de test: {test_loss:.4f}")
    
    # Rapport de classification
    print("\nRapport de classification sur l'ensemble de test:")
    print(classification_report(y_test, y_test_pred, target_names=classes))
    
    # Sauvegarde des résultats
    results = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'best_params': model.best_params_ if hasattr(model, 'best_params_') else {}
    }
    
    return results

def main():
    # Configuration des chemins
    base_dir = Path("/workspaces/datasciencetest_reco_plante")
    csv_path = base_dir / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
    output_dir = base_dir / "results" / "bayesian_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Types de cibles à analyser
    target_types = ["espece", "maladie"]
    
    # Pour stocker les résultats
    all_results = {}
    
    for target_type in target_types:
        # Chargement des données
        X, y, classes, feature_names = load_and_prepare_data(csv_path, target_type)
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        
        # Normalisation
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Division supplémentaire pour la validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, 
            test_size=EVAL_SET_SIZE, 
            stratify=y_train,
            random_state=SEED
        )
        
        print(f"\nFormes des données:")
        print(f"- Train final: {X_train_final.shape}")
        print(f"- Validation: {X_val.shape}")
        print(f"- Test: {X_test_scaled.shape}")
        
        # Optimisation bayésienne
        bayes_search = run_bayesian_optimization(
            X_train_final, y_train_final, 
            X_val, y_val,
            n_classes=len(classes)
        )
        
        # Évaluation du meilleur modèle
        results = evaluate_model(
            bayes_search,
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            classes, output_dir, target_type
        )
        
        # Sauvegarde du modèle
        model_save_path = output_dir / f"best_model_{target_type}.pkl"
        import joblib
        joblib.dump(bayes_search.best_estimator_, model_save_path)
        print(f"\nMeilleur modèle sauvegardé: {model_save_path}")
        
        # Courbe de convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(bayes_search.optimizer_results_[0])
        plt.title(f"Courbe de convergence - {target_type}")
        convergence_path = output_dir / f"convergence_{target_type}.png"
        plt.savefig(convergence_path)
        plt.close()
        print(f"Courbe de convergence sauvegardée: {convergence_path}")
        
        # Stockage des résultats
        all_results[target_type] = {
            'best_params': bayes_search.best_params_,
            'best_score': bayes_search.best_score_,
            'test_loss': results['test_loss']
        }
    
    # Affichage des résultats finaux
    print("\n" + "="*50)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*50)
    for target_type, res in all_results.items():
        print(f"\n{target_type.upper()}:")
        print(f"- Meilleur score (négatif log loss): {res['best_score']:.4f}")
        print(f"- Perte sur le test: {res['test_loss']:.4f}")
        print("\nMeilleurs paramètres:")
        for param, value in res['best_params'].items():
            print(f"  - {param}: {value}")

if __name__ == "__main__":
    main()
