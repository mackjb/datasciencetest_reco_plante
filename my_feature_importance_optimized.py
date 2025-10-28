#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'analyse de l'importance des caractéristiques pour la classification des espèces de plantes.
VERSION OPTIMISÉE : Utilise un CSV avec features pré-calculées pour éviter le recalcul.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Pour la visualisation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Pour l'explication du modèle
import shap

# Import des fonctions existantes
from src.helpers.helpers import PROJECT_ROOT

# -----------------------------
# 1. Configuration & Constants
# -----------------------------

# Chemins
CSV_FILE = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features_with_sampling.csv"

# Mapping des colonnes du CSV vers les noms standardisés
COLUMN_MAPPING = {
    'nom_plante': 'species',
    'nom_maladie': 'disease',
    'Est_Saine': 'is_healthy',
    'Image_Path': 'filepath',
    'aire': 'area',
    'périmètre': 'perimeter',
    'circularité': 'circularity',
    'excentricité': 'eccentricity',
    'aspect_ratio': 'aspect_ratio',
    'hu_1': 'phi1_distingue_large_vs_etroit',
    'hu_2': 'phi2_distinction_elongation_forme',
    'hu_3': 'phi3_asymetrie_maladie',
    'hu_4': 'phi4_symetrie_diagonale_forme',
    'hu_5': 'phi5_concavite_extremites',
    'hu_6': 'phi6_decalage_torsion_maladie',
    'hu_7': 'phi7_asymetrie_complexe',
    'fft_low_freq_power': 'energie_basse_forme_feuille',
    'fft_high_freq_power': 'energie_haute_details_maladie',
    'fft_entropy': 'fft_entropy',
    'fft_energy': 'fft_energy',
    'hog_mean': 'hog_moyenne_contours_forme',
    'hog_std': 'hog_ecarttype_texture',
    'hog_entropy': 'hog_entropy',
    'mean_R': 'mean_R',
    'mean_G': 'mean_G',
    'mean_B': 'mean_B',
    'std_R': 'std_R',
    'std_G': 'std_G',
    'std_B': 'std_B',
    'mean_H': 'mean_H',
    'mean_S': 'mean_S',
    'mean_V': 'mean_V',
    'contrast': 'contrast',
    'energy': 'energy',
    'homogeneity': 'homogeneity',
    'dissimilarité': 'dissimilarite',
    'Correlation': 'correlation',
    'netteté': 'nettete',
    'contour_density': 'contour_density',
}

# Features sélectionnées
FEATURE_COLS = [
    'phi1_distingue_large_vs_etroit', 'phi2_distinction_elongation_forme',
    'phi3_asymetrie_maladie', 'phi4_symetrie_diagonale_forme',
    'phi5_concavite_extremites', 'phi6_decalage_torsion_maladie',
    'phi7_asymetrie_complexe', 'energie_basse_forme_feuille',
    'energie_haute_details_maladie', 'fft_entropy', 'fft_energy',
    'hog_moyenne_contours_forme', 'hog_ecarttype_texture', 'hog_entropy',
    'mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B',
    'mean_H', 'mean_S', 'mean_V', 'contrast', 'energy', 'homogeneity',
    'dissimilarite', 'correlation', 'nettete', 'contour_density',
    'area', 'perimeter', 'circularity', 'eccentricity', 'aspect_ratio',
]

TARGET_COLUMN = "species"
RANDOM_STATE = 42

def load_and_prepare_data(csv_path=CSV_FILE, target_col=TARGET_COLUMN, test_size=0.2, 
                          random_state=RANDOM_STATE, sample_fraction=None):
    """Charge le CSV avec features pré-calculées"""
    print(f"\n{'='*60}")
    print(f"CHARGEMENT: {csv_path.name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ {len(df):,} exemples")
    
    df = df.rename(columns=COLUMN_MAPPING)
    
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"⚠ Features manquantes: {missing}")
        for f in missing:
            df[f] = 0
    
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    print(f"\n📊 Distribution:")
    for cls, cnt in df[target_col].value_counts().items():
        print(f"  • {cls}: {cnt:,}")
    
    if sample_fraction:
        df = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=random_state)
        ).reset_index(drop=True)
    
    X = df[FEATURE_COLS].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train: {len(y_train):,} | Test: {len(y_test):,}")
    return X_train, X_test, y_train, y_test, FEATURE_COLS


def create_selectors(random_state=RANDOM_STATE):
    return {
        'univariate_f': SelectKBest(score_func=f_classif, k=15),
        'rfe_rf': RFE(
            estimator=RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=random_state),
            n_features_to_select=15, step=5
        ),
        'sfm_l1': SelectFromModel(
            estimator=LogisticRegression(penalty='l1', solver='liblinear', max_iter=100, random_state=random_state),
            threshold='median'
        ),
        'sfm_tree': SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state),
            threshold='median'
        )
    }


def evaluate_selectors(X, y, selectors, cv=3, random_state=RANDOM_STATE):
    results, feature_importances, selected_features_idx = {}, {}, {}
    scaler = RobustScaler()
    base_clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state)
    
    print(f"\n{'='*60}")
    print(f"ÉVALUATION (CV={cv})")
    print(f"{'='*60}")
    
    for name, selector in selectors.items():
        print(f"\n🔍 {name.upper()}")
        start = time.time()
        
        # Pour SFM_L1, échantillonner pour accélérer
        if name == 'sfm_l1' and len(y) > 20000:
            print(f"  ⚡ Échantillonnage pour accélérer (20k exemples)")
            indices = np.random.RandomState(random_state).choice(len(y), 20000, replace=False)
            X_sample, y_sample = X[indices], y[indices]
        else:
            X_sample, y_sample = X, y
        
        try:
            pipe = Pipeline([('scaler', scaler), ('feat_sel', selector), ('clf', base_clf)])
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            
            cv_results = cross_validate(
                pipe, X_sample, y_sample, cv=cv_obj,
                scoring=['accuracy', 'f1_macro'],
                return_train_score=True,
                return_estimator=True,
                n_jobs=1
            )
            
            results[name] = {
                'accuracy': np.mean(cv_results['test_accuracy']),
                'accuracy_std': np.std(cv_results['test_accuracy']),
                'f1_macro': np.mean(cv_results['test_f1_macro']),
                'f1_macro_std': np.std(cv_results['test_f1_macro']),
            }
            
            selected_features, importances = [], []
            for estimator in cv_results['estimator']:
                mask = estimator.named_steps['feat_sel'].get_support()
                selected_features.append(mask)
                
                if hasattr(estimator.named_steps['clf'], 'feature_importances_'):
                    fold_imp = estimator.named_steps['clf'].feature_importances_
                    full_imp = np.zeros(X.shape[1])
                    full_imp[mask] = fold_imp
                    importances.append(full_imp)
            
            selected_features_idx[name] = np.mean(selected_features, axis=0)
            feature_importances[name] = np.mean(importances, axis=0) if importances else np.zeros(X.shape[1])
            
            print(f"  ✓ Acc: {results[name]['accuracy']:.4f}±{results[name]['accuracy_std']:.3f}")
            print(f"  ✓ F1: {results[name]['f1_macro']:.4f}±{results[name]['f1_macro_std']:.3f}")
            print(f"  ⏱ {time.time()-start:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            results[name] = {'accuracy': 0, 'f1_macro': 0, 'accuracy_std': 0, 'f1_macro_std': 0}
            feature_importances[name] = np.zeros(X.shape[1])
            selected_features_idx[name] = np.zeros(X.shape[1])
    
    return results, feature_importances, selected_features_idx


def plot_results(feature_importances, selected_features_idx, feature_cols, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VISUALISATIONS")
    print(f"{'='*60}")
    
    # Heatmap
    n_top = 20
    all_freq = np.array([freq for freq in selected_features_idx.values()])
    global_imp = all_freq.mean(axis=0)
    top_idx = np.argsort(-global_imp)[:n_top]
    
    heatmap_data = np.zeros((n_top, len(selected_features_idx)))
    for j, (name, freq) in enumerate(selected_features_idx.items()):
        for i, idx in enumerate(top_idx):
            heatmap_data[i, j] = freq[idx]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=list(selected_features_idx.keys()),
                yticklabels=[feature_cols[i] for i in top_idx],
                cbar_kws={'label': 'Fréquence'})
    plt.title('Heatmap - Top 20 Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ heatmap.png")


def explain_shap(X, y, feature_cols, output_dir, max_samples=1000, random_state=RANDOM_STATE):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SHAP")
    print(f"{'='*60}")
    
    try:
        if len(y) > max_samples:
            idx = np.random.RandomState(random_state).choice(len(y), max_samples, replace=False)
            X_sample, y_sample = X[idx], y[idx]
        else:
            X_sample, y_sample = X, y
        
        model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=random_state)
        model.fit(X_sample, y_sample)
        
        explainer = shap.TreeExplainer(model)
        X_small = X_sample[np.random.RandomState(random_state).choice(len(X_sample), min(500, len(X_sample)), replace=False)]
        shap_values = explainer.shap_values(X_small)
        
        if isinstance(shap_values, list):
            shap_imp = np.array([np.abs(sv).mean(axis=0) for sv in shap_values]).mean(axis=0)
        else:
            shap_imp = np.abs(shap_values).mean(axis=0)
        
        sorted_idx = np.argsort(shap_imp)[::-1][:20]
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), shap_imp[sorted_idx], color='coral')
        plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
        plt.xlabel('|SHAP| moyen')
        plt.title('Importance SHAP', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ shap_importance.png")
        
    except Exception as e:
        print(f"✗ Erreur SHAP: {e}")


def main():
    start = time.time()
    print(f"\n{'#'*60}")
    print(f"# ANALYSE FEATURES - VERSION OPTIMISÉE")
    print(f"{'#'*60}")
    
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(sample_fraction=None)
    
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    
    print(f"\n📊 Total: {X_all.shape[0]:,} × {X_all.shape[1]} features")
    
    selectors = create_selectors()
    results, feature_importances, selected_features_idx = evaluate_selectors(X_all, y_all, selectors, cv=3)
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS")
    print(f"{'='*60}")
    for name, scores in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name:<20} Acc: {scores['accuracy']:.4f}±{scores['accuracy_std']:.3f}  F1: {scores['f1_macro']:.4f}±{scores['f1_macro_std']:.3f}")
    
    plot_results(feature_importances, selected_features_idx, feature_names, figures_dir)
    explain_shap(X_all, y_all, feature_names, figures_dir)
    
    all_imp = np.array([imp for imp in feature_importances.values()])
    all_freq = np.array([freq for freq in selected_features_idx.values()])
    
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': all_imp.mean(axis=0),
        'mean_frequency': all_freq.mean(axis=0),
        'combined_score': (all_imp.mean(axis=0) + all_freq.mean(axis=0)) / 2
    }).sort_values('combined_score', ascending=False)
    
    ranking_df.to_csv(results_dir / 'feature_ranking.csv', index=False)
    print(f"\n✓ Sauvegardé: feature_ranking.csv")
    
    print(f"\n✓ TERMINÉ EN {time.time()-start:.1f}s")
    return results, feature_importances, selected_features_idx, ranking_df


if __name__ == '__main__':
    try:
        results, feature_importances, selected_features_idx, ranking_df = main()
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
