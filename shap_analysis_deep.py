#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyse SHAP approfondie pour la classification des espèces de plantes.
Utilise le maximum d'échantillons possibles pour une analyse fine.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import shap
from src.helpers.helpers import PROJECT_ROOT

# Configuration
CSV_FILE = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features_with_sampling.csv"
RANDOM_STATE = 42

# Mapping des colonnes
COLUMN_MAPPING = {
    'nom_plante': 'species', 'nom_maladie': 'disease', 'Est_Saine': 'is_healthy',
    'Image_Path': 'filepath', 'aire': 'area', 'périmètre': 'perimeter',
    'circularité': 'circularity', 'excentricité': 'eccentricity',
    'aspect_ratio': 'aspect_ratio', 'hu_1': 'phi1', 'hu_2': 'phi2',
    'hu_3': 'phi3', 'hu_4': 'phi4', 'hu_5': 'phi5', 'hu_6': 'phi6',
    'hu_7': 'phi7', 'fft_low_freq_power': 'fft_low', 'fft_high_freq_power': 'fft_high',
    'fft_entropy': 'fft_entropy', 'fft_energy': 'fft_energy',
    'hog_mean': 'hog_mean', 'hog_std': 'hog_std', 'hog_entropy': 'hog_entropy',
    'mean_R': 'mean_R', 'mean_G': 'mean_G', 'mean_B': 'mean_B',
    'std_R': 'std_R', 'std_G': 'std_G', 'std_B': 'std_B',
    'mean_H': 'mean_H', 'mean_S': 'mean_S', 'mean_V': 'mean_V',
    'contrast': 'contrast', 'energy': 'energy', 'homogeneity': 'homogeneity',
    'dissimilarité': 'dissimilarite', 'Correlation': 'correlation',
    'netteté': 'nettete', 'contour_density': 'contour_density',
}

FEATURE_COLS = [
    'phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7',
    'fft_low', 'fft_high', 'fft_entropy', 'fft_energy',
    'hog_mean', 'hog_std', 'hog_entropy',
    'mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B',
    'mean_H', 'mean_S', 'mean_V',
    'contrast', 'energy', 'homogeneity', 'dissimilarite', 'correlation',
    'nettete', 'contour_density',
    'area', 'perimeter', 'circularity', 'eccentricity', 'aspect_ratio',
]

def load_data(csv_path=CSV_FILE, test_size=0.2, random_state=RANDOM_STATE):
    """Charge les données"""
    print(f"\n{'='*70}")
    print(f"CHARGEMENT DES DONNÉES")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Lignes chargées: {len(df):,}")
    
    df = df.rename(columns=COLUMN_MAPPING)
    
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        elif df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    X = df[FEATURE_COLS].values
    y = df['species'].values
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Classes: {len(np.unique(y))}")
    
    class_counts = pd.Series(y).value_counts()
    print(f"\nDistribution des classes:")
    for cls, cnt in class_counts.items():
        print(f"  • {cls}: {cnt:,}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Train: {len(y_train):,} | Test: {len(y_test):,}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, max_depth=12):
    """Entraîne un RandomForest"""
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT DU MODÈLE")
    print(f"{'='*70}")
    print(f"Paramètres: n_estimators={n_estimators}, max_depth={max_depth}")
    
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"✓ Entraînement terminé en {elapsed:.1f}s")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle"""
    print(f"\n{'='*70}")
    print(f"ÉVALUATION DU MODÈLE")
    print(f"{'='*70}")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✓ Accuracy: {acc:.4f}")
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    
    return acc


def shap_analysis_full(model, X_train, y_train, X_test, feature_names, 
                       max_train_samples=5000, max_explain_samples=2000,
                       output_dir=None):
    """
    Analyse SHAP complète avec plusieurs visualisations.
    
    Args:
        model: Modèle entraîné
        X_train: Données d'entraînement (pour background)
        y_train: Labels d'entraînement
        X_test: Données à expliquer
        feature_names: Noms des features
        max_train_samples: Échantillons de background pour TreeExplainer
        max_explain_samples: Échantillons à expliquer
        output_dir: Dossier de sortie
    """
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures" / "shap"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"ANALYSE SHAP APPROFONDIE")
    print(f"{'='*70}")
    
    # Estimation du temps
    total_samples_to_explain = min(len(X_test), max_explain_samples)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Estimation: ~0.01s par sample pour TreeExplainer
    estimated_time = total_samples_to_explain * 0.01 * n_classes
    
    print(f"\nParamètres:")
    print(f"  • Échantillons de background: {min(len(X_train), max_train_samples):,}")
    print(f"  • Échantillons à expliquer: {total_samples_to_explain:,}")
    print(f"  • Features: {n_features}")
    print(f"  • Classes: {n_classes}")
    print(f"\n⏱ Temps estimé: {estimated_time/60:.1f} minutes (~{estimated_time:.0f}s)")
    
    # Échantillonner pour le background
    if len(X_train) > max_train_samples:
        print(f"\n📊 Échantillonnage du background...")
        bg_indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), max_train_samples, replace=False
        )
        X_background = X_train[bg_indices]
    else:
        X_background = X_train
    
    # Échantillonner les données à expliquer
    if len(X_test) > max_explain_samples:
        print(f"📊 Échantillonnage des données à expliquer...")
        explain_indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X_test), max_explain_samples, replace=False
        )
        X_explain = X_test[explain_indices]
    else:
        X_explain = X_test
    
    # Créer l'explainer
    print(f"\n🔧 Création du TreeExplainer...")
    start = time.time()
    explainer = shap.TreeExplainer(model, X_background)
    print(f"✓ Explainer créé en {time.time()-start:.1f}s")
    
    # Calculer les valeurs SHAP
    print(f"\n💡 Calcul des valeurs SHAP...")
    start = time.time()
    shap_values = explainer.shap_values(X_explain, check_additivity=False)
    elapsed = time.time() - start
    print(f"✓ Valeurs SHAP calculées en {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"   → Vitesse: {len(X_explain)/elapsed:.1f} samples/s")
    
    # Analyser la structure des valeurs SHAP
    if isinstance(shap_values, list):
        print(f"\n📐 Format: Liste de {len(shap_values)} classes")
        for i, sv in enumerate(shap_values):
            print(f"   Classe {i}: {sv.shape}")
        is_multiclass = True
        shap_format = "list"
    else:
        print(f"\n📐 Format: {shap_values.shape}")
        if len(shap_values.shape) == 3:
            is_multiclass = True
            shap_format = "3d"
        else:
            is_multiclass = False
            shap_format = "2d"
    
    # Calculer l'importance globale AVANT les visualisations
    if shap_format == "list":
        # Format liste: [classe1(n_samples, n_features), classe2(...), ...]
        mean_abs_shap = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
        global_importance = mean_abs_shap.mean(axis=0)
    elif shap_format == "3d":
        # Format 3D: (n_samples, n_features, n_classes)
        # Moyenne sur les samples et les classes
        global_importance = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        # Format 2D: (n_samples, n_features)
        global_importance = np.abs(shap_values).mean(axis=0)
    
    # === VISUALISATION 1: Summary Plot Global ===
    print(f"\n📊 Génération: Summary Plot Global...")
    try:
        plt.figure(figsize=(14, 10))
        if shap_format == "list":
            # Pour liste, on utilise la première classe
            shap.summary_plot(shap_values[0], X_explain, 
                            feature_names=feature_names, 
                            show=False, max_display=25)
        elif shap_format == "3d":
            # Pour 3D, on utilise la première classe (index 0 sur la 3e dimension)
            shap.summary_plot(shap_values[:, :, 0], X_explain,
                            feature_names=feature_names,
                            show=False, max_display=25)
        else:
            shap.summary_plot(shap_values, X_explain,
                            feature_names=feature_names,
                            show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary_global.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ shap_summary_global.png")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
    
    # === VISUALISATION 2: Bar Plot Global ===
    print(f"\n📊 Génération: Bar Plot Global...")
    try:
        
        sorted_idx = np.argsort(global_importance)[::-1][:25]
        
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(sorted_idx)), global_importance[sorted_idx], color='steelblue')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('|SHAP value| moyen', fontsize=12)
        plt.title('Importance Globale des Features (SHAP)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_global_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ shap_global_importance.png")
        
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
    
    # === VISUALISATION 3: Heatmap par Classe ===
    if is_multiclass:
        print(f"\n📊 Génération: Heatmap par Classe...")
        try:
            n_top_features = min(20, n_features)
            
            if shap_format == "list":
                n_classes_to_show = min(10, len(shap_values))
            elif shap_format == "3d":
                n_classes_to_show = min(10, shap_values.shape[2])
            else:
                n_classes_to_show = 1
            
            # Importance par classe
            class_importance = np.zeros((n_top_features, n_classes_to_show))
            top_idx = np.argsort(global_importance)[::-1][:n_top_features]
            
            for class_idx in range(n_classes_to_show):
                if shap_format == "list":
                    class_shap = np.abs(shap_values[class_idx]).mean(axis=0)
                elif shap_format == "3d":
                    class_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
                else:
                    class_shap = global_importance
                
                for i, feat_idx in enumerate(top_idx):
                    class_importance[i, class_idx] = class_shap[feat_idx]
            
            # Noms des classes
            unique_classes = np.unique(y_train)[:n_classes_to_show]
            top_feature_names = [feature_names[i] for i in top_idx]
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(class_importance, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=unique_classes,
                       yticklabels=top_feature_names,
                       cbar_kws={'label': '|SHAP| moyen'})
            plt.title('Importance des Features par Classe (SHAP)', fontsize=14, fontweight='bold')
            plt.xlabel('Classe', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / 'shap_heatmap_class.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ shap_heatmap_class.png")
            
        except Exception as e:
            print(f"   ✗ Erreur: {e}")
    
    # === VISUALISATION 4: Dependence Plots (top 5 features) ===
    print(f"\n📊 Génération: Dependence Plots (top 5)...")
    try:
        top_5_idx = np.argsort(global_importance)[::-1][:5]
        
        for idx in top_5_idx:
            feat_name = feature_names[idx]
            safe_name = feat_name.replace('/', '_').replace(' ', '_')
            
            plt.figure(figsize=(10, 6))
            
            if shap_format == "list":
                shap.dependence_plot(idx, shap_values[0], X_explain,
                                   feature_names=feature_names,
                                   show=False)
            elif shap_format == "3d":
                # Pour 3D, utiliser la première classe
                shap.dependence_plot(idx, shap_values[:, :, 0], X_explain,
                                   feature_names=feature_names,
                                   show=False)
            else:
                shap.dependence_plot(idx, shap_values, X_explain,
                                   feature_names=feature_names,
                                   show=False)
                                   
            plt.title(f'Dependence Plot: {feat_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'shap_dependence_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   ✓ 5 dependence plots créés")
        
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    # === SAUVEGARDE DES RÉSULTATS ===
    print(f"\n💾 Sauvegarde des résultats...")
    
    # DataFrame avec l'importance globale
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': global_importance
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance_df.to_csv(output_dir / 'shap_feature_importance.csv', index=False)
    print(f"   ✓ shap_feature_importance.csv")
    
    # Sauvegarder les valeurs SHAP brutes (optionnel, peut être volumineux)
    # np.save(output_dir / 'shap_values.npy', shap_values)
    
    print(f"\n{'='*70}")
    print(f"ANALYSE SHAP TERMINÉE")
    print(f"{'='*70}")
    print(f"Résultats sauvegardés dans: {output_dir}")
    
    return shap_values, global_importance


def main():
    """Fonction principale"""
    start_global = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# ANALYSE SHAP APPROFONDIE - MAXIMUM D'ÉCHANTILLONS")
    print(f"{'#'*70}")
    
    # 1. Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Entraîner le modèle
    model = train_model(X_train, y_train, n_estimators=100, max_depth=12)
    
    # 3. Évaluer
    acc = evaluate_model(model, X_test, y_test)
    
    # 4. Analyse SHAP avec MAXIMUM d'échantillons
    # Ajuster ces valeurs selon les ressources disponibles:
    # - max_train_samples: 10000 (background pour TreeExplainer)
    # - max_explain_samples: 5000 (échantillons à expliquer)
    
    shap_values, importance = shap_analysis_full(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        feature_names=FEATURE_COLS,
        max_train_samples=5000,   # Réduit pour test rapide
        max_explain_samples=2000,  # Réduit pour test rapide (~20-25 min)
        output_dir=PROJECT_ROOT / "figures" / "shap"
    )
    
    # Temps total
    elapsed_total = time.time() - start_global
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSE COMPLÈTE TERMINÉE")
    print(f"{'='*70}")
    print(f"Temps total: {elapsed_total:.1f}s ({elapsed_total/60:.2f} minutes)")
    
    # Top 10 features
    print(f"\n🏆 Top 10 Features par Importance SHAP:")
    top_10_idx = np.argsort(importance)[::-1][:10]
    for rank, idx in enumerate(top_10_idx, 1):
        print(f"   {rank:2d}. {FEATURE_COLS[idx]:<30} {importance[idx]:.4f}")
    
    return model, shap_values, importance


if __name__ == '__main__':
    try:
        model, shap_values, importance = main()
        print(f"\n✓ Succès!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
