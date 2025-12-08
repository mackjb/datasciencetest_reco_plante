#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse SHAP sur le Meilleur Modèle ML - PlantVillage Dataset

Ce script réalise une analyse d'importance des features via SHAP (SHapley Additive exPlanations)
sur un échantillon stratifié de 1000 images.

Objectifs:
1. Entraîner le meilleur modèle ML (SVM RBF ou ExtraTrees)
2. Calculer les valeurs SHAP sur un échantillon représentatif
3. Générer 3 graphiques pour le rapport scientifique

Outputs:
- figures/shap_analysis/global_importance.png : Importance globale des features
- figures/shap_analysis/feature_impact.png : Impact détaillé (valeurs SHAP)
- figures/shap_analysis/top_features_by_class.png : Features importantes par classe (top 3)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

try:
    import shap
except ImportError:
    print("⚠️  SHAP n'est pas installé. Installez-le avec: pip install shap")
    exit(1)

try:
    from src.helpers.helpers import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[0]

# ================================
# CONFIGURATION
# ================================
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
OUTPUT_DIR: Path = PROJECT_ROOT / "figures" / "shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choix du modèle: 'svm', 'extratrees' ou 'randomforest'
MODEL_TYPE: str = "randomforest"  # RandomForest = rapide et performant

# Taille de l'échantillon SHAP
SHAP_SAMPLE_SIZE: int = 500

# Colonnes à exclure (non-features)
NON_FEATURE_COLS: List[str] = [
    "nom_plante", "nom_maladie", "Est_Saine", "Image_Path", "width_img", "height_img",
    "is_black", "md5", "species", "ID_Image", "filepath", "filename", "extension",
    "file_size", "label", "width", "height", "mode", "num_channels", "aspect_ratio",
    "is_image_valid", "is_na", "hash", "is_duplicate_after_first", "disease",
]

# Configuration des graphiques
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


# ================================
# FONCTIONS
# ================================

def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Charge le dataset et prépare les features.
    
    Returns:
        df: DataFrame complet
        X: Matrice de features
        y: Labels (noms de maladies)
        feature_names: Noms des features
        class_names: Noms des classes
    """
    print(f"📂 Chargement du dataset: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   ✓ {len(df)} lignes chargées")
    
    # Identifier les colonnes de features
    all_cols = set(df.columns)
    non_feat_present = set(NON_FEATURE_COLS) & all_cols
    feature_cols = [c for c in df.columns if c not in non_feat_present]
    
    # Ne garder que les colonnes numériques
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"   ✓ {len(numeric_cols)} features numériques identifiées")
    
    # Target: nom de maladie
    if "nom_maladie" not in df.columns:
        raise ValueError("Colonne 'nom_maladie' absente du dataset")
    
    # Nettoyer les NaN
    df_clean = df.dropna(subset=["nom_maladie"] + numeric_cols)
    print(f"   ✓ {len(df_clean)} lignes après nettoyage des NaN")
    
    X = df_clean[numeric_cols].values
    y = df_clean["nom_maladie"].values
    class_names = sorted(df_clean["nom_maladie"].unique())
    
    print(f"   ✓ {len(class_names)} classes uniques")
    
    return df_clean, X, y, numeric_cols, class_names


def train_best_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "randomforest") -> Pipeline:
    """
    Entraîne le meilleur modèle ML.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        model_type: 'svm', 'extratrees' ou 'randomforest'
    
    Returns:
        Pipeline entraîné
    """
    print(f"\n🔧 Entraînement du modèle: {model_type.upper()}")
    start = time.time()
    
    if model_type == "svm":
        # SVM RBF avec paramètres optimaux (à ajuster selon vos résultats)
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,  # Nécessaire pour KernelExplainer
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ])
    elif model_type == "randomforest":
        # RandomForest avec paramètres optimaux
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
    elif model_type == "extratrees":
        # ExtraTrees avec paramètres optimaux
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"   ✓ Modèle entraîné en {elapsed:.1f}s")
    
    return model


def stratified_sample_for_shap(X: np.ndarray, y: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée un échantillon stratifié pour l'analyse SHAP.
    
    Returns:
        X_sample: Features échantillonnées
        y_sample: Labels échantillonnés
        sample_indices: Indices sélectionnés
    """
    print(f"\n📊 Création d'un échantillon stratifié de {n_samples} images...")
    
    # Si on a moins d'images que demandé, prendre toutes les images
    if len(X) <= n_samples:
        print(f"   ⚠️  Dataset trop petit ({len(X)} images), utilisation de tout le dataset")
        return X, y, np.arange(len(X))
    
    # Échantillonnage stratifié
    sample_indices, _ = train_test_split(
        np.arange(len(X)),
        train_size=n_samples,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    print(f"   ✓ Échantillon créé: {len(X_sample)} images")
    print(f"   ✓ Distribution des classes:")
    unique, counts = np.unique(y_sample, return_counts=True)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
        print(f"      - {cls}: {cnt} images")
    if len(unique) > 5:
        print(f"      ... et {len(unique) - 5} autres classes")
    
    return X_sample, y_sample, sample_indices


def compute_shap_values(model: Pipeline, X_sample: np.ndarray, feature_names: List[str], model_type: str) -> Tuple:
    """
    Calcule les valeurs SHAP.
    
    Returns:
        explainer: Explainer SHAP
        shap_values: Valeurs SHAP calculées
        X_transformed: Features après transformation du pipeline
    """
    print(f"\n⚡ Calcul des valeurs SHAP...")
    start = time.time()
    
    # Transformer les données (scaling)
    X_transformed = model.named_steps['scaler'].transform(X_sample)
    
    if model_type in ["extratrees", "randomforest"]:
        # TreeExplainer (rapide pour les arbres)
        print("   → Utilisation de TreeExplainer (optimisé pour les arbres)")
        clf = model.named_steps['clf']
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # KernelExplainer pour SVM (plus lent mais universel)
        print("   → Utilisation de KernelExplainer (universel)")
        
        # Créer un background dataset (100 images de référence)
        background_size = min(100, len(X_transformed))
        background = shap.sample(X_transformed, background_size)
        
        # Fonction de prédiction
        def predict_fn(X):
            return model.predict_proba(X)
        
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Calculer SHAP sur l'échantillon
        shap_values = explainer.shap_values(X_transformed)
    
    elapsed = time.time() - start
    print(f"   ✓ Valeurs SHAP calculées en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    return explainer, shap_values, X_transformed


def plot_global_importance(shap_values, feature_names: List[str], output_path: Path) -> None:
    """
    Graphique 1: Importance globale des features (barplot).
    """
    print(f"\n📈 Génération du graphique 1: Importance globale...")
    
    # Gérer les différents formats de shap_values
    if isinstance(shap_values, list):
        # Liste de tableaux (un par classe)
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        # Tableau 3D (n_samples, n_features, n_classes)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # Tableau 2D (n_samples, n_features)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Trier par importance
    indices = np.argsort(mean_abs_shap)[::-1]
    top_n = 25  # Top 25 features
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, mean_abs_shap[indices][:top_n], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel('|SHAP value| moyen', fontsize=12, fontweight='bold')
    ax.set_title('Importance Globale des Features (SHAP)\nTop 25 Features', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Sauvegardé: {output_path}")
    plt.close()


def plot_feature_impact(shap_values, X_transformed: np.ndarray, feature_names: List[str], 
                       class_names: List[str], output_path: Path) -> None:
    """
    Graphique 2: Impact détaillé des features (summary plot).
    Montre la distribution des valeurs SHAP pour chaque feature.
    """
    print(f"\n📈 Génération du graphique 2: Impact détaillé...")
    
    # Gérer les différents formats de shap_values
    if isinstance(shap_values, list):
        # Liste de tableaux (un par classe)
        shap_for_plot = shap_values[0]
        class_label = class_names[0] if len(class_names) > 0 else "Classe 0"
        title_suffix = f" (Classe: {class_label})"
    elif len(shap_values.shape) == 3:
        # Tableau 3D (n_samples, n_features, n_classes) - prendre la première classe
        shap_for_plot = shap_values[:, :, 0]
        class_label = class_names[0] if len(class_names) > 0 else "Classe 0"
        title_suffix = f" (Classe: {class_label})"
    else:
        # Tableau 2D
        shap_for_plot = shap_values
        title_suffix = ""
    
    # Utiliser le summary_plot de SHAP
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_for_plot, 
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title(f'Impact des Features sur les Prédictions{title_suffix}\n' +
              'Couleur = Valeur de la feature (rouge = haute, bleu = basse)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Sauvegardé: {output_path}")
    plt.close()


def plot_top_features_by_class(shap_values, feature_names: List[str], 
                               class_names: List[str], output_path: Path, top_n_classes: int = 5) -> None:
    """
    Graphique 3: Top features pour chaque classe (heatmap).
    """
    print(f"\n📈 Génération du graphique 3: Top features par classe...")
    
    # Gérer les différents formats de shap_values
    if isinstance(shap_values, list):
        n_classes = min(top_n_classes, len(shap_values))
        # Calculer l'importance moyenne par classe
        importance_by_class = []
        for class_idx in range(n_classes):
            mean_abs = np.abs(shap_values[class_idx]).mean(axis=0)
            importance_by_class.append(mean_abs)
    elif len(shap_values.shape) == 3:
        # Tableau 3D (n_samples, n_features, n_classes)
        n_classes = min(top_n_classes, shap_values.shape[2])
        importance_by_class = []
        for class_idx in range(n_classes):
            mean_abs = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
            importance_by_class.append(mean_abs)
    else:
        print("   ⚠️  Graphique par classe non disponible (modèle binaire ou régression)")
        return
    
    importance_matrix = np.array(importance_by_class)
    
    # Sélectionner les top features (union des tops de chaque classe)
    top_features_indices = set()
    for class_importances in importance_by_class:
        top_indices = np.argsort(class_importances)[::-1][:10]
        top_features_indices.update(top_indices)
    
    top_features_indices = sorted(top_features_indices, 
                                 key=lambda i: importance_matrix[:, i].sum(), 
                                 reverse=True)[:15]
    
    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = importance_matrix[:, top_features_indices]
    sns.heatmap(
        heatmap_data,
        xticklabels=[feature_names[i] for i in top_features_indices],
        yticklabels=[class_names[i][:30] for i in range(n_classes)],  # Tronquer les noms longs
        cmap='YlOrRd',
        annot=False,
        fmt='.3f',
        cbar_kws={'label': '|SHAP| moyen'},
        ax=ax
    )
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classes', fontsize=12, fontweight='bold')
    ax.set_title(f'Importance des Features par Classe (Top {n_classes} classes)\n' +
                 'Valeurs SHAP moyennes absolues',
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Sauvegardé: {output_path}")
    plt.close()


def save_summary_report(shap_values, feature_names: List[str], class_names: List[str], 
                       output_dir: Path) -> None:
    """
    Sauvegarde un rapport JSON avec les statistiques clés.
    """
    print(f"\n📝 Création du rapport de synthèse...")
    
    # Calcul de l'importance globale
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        n_samples = shap_values[0].shape[0]
    elif len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        n_samples = shap_values.shape[0]
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        n_samples = shap_values.shape[0]
    
    # Top 10 features
    indices = np.argsort(mean_abs_shap)[::-1]
    top_10 = {
        feature_names[i]: float(mean_abs_shap[i])
        for i in indices[:10]
    }
    
    report = {
        "date_analyse": pd.Timestamp.now().isoformat(),
        "nombre_images_analysees": n_samples,
        "nombre_features": len(feature_names),
        "nombre_classes": len(class_names),
        "top_10_features_globales": top_10,
        "interpretation": {
            "feature_la_plus_importante": feature_names[indices[0]],
            "valeur_shap_moyenne": float(mean_abs_shap[indices[0]]),
            "categories_dominantes": {
                "geometrie_forme": ["contour_density", "area", "périmètre", "circularité", "excentricité", "aspect_ratio"],
                "texture": ["hog_std", "hog_entropy", "hog_mean", "homogeneity", "energy", "dissimilarité", "Correlation"],
                "couleur": ["mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "mean_V", "std_R", "std_G", "std_B"],
                "frequence": ["fft_entropy", "fft_energy", "fft_low_freq_power", "fft_high_freq_power"],
                "moments": ["phi2", "phi3", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7"],
            }
        }
    }
    
    report_path = output_dir / "shap_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Rapport sauvegardé: {report_path}")


# ================================
# MAIN
# ================================

def main():
    print("="*70)
    print("  ANALYSE SHAP - IMPORTANCE DES FEATURES")
    print("  PlantVillage Dataset - Reconnaissance de Maladies")
    print("="*70)
    
    # 1. Charger les données
    df, X, y, feature_names, class_names = load_and_prepare_data(CSV_PATH)
    
    # 2. Split train/test
    print(f"\n🔀 Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    print(f"   ✓ Train: {len(X_train)} images")
    print(f"   ✓ Test:  {len(X_test)} images")
    
    # 3. Entraîner le modèle
    model = train_best_model(X_train, y_train, model_type=MODEL_TYPE)
    
    # Évaluation rapide
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n📊 Performance du modèle sur le test:")
    print(f"   • Balanced Accuracy: {bal_acc:.4f}")
    print(f"   • F1-Score (macro):  {f1:.4f}")
    
    # 4. Échantillonner pour SHAP
    X_shap, y_shap, _ = stratified_sample_for_shap(X_test, y_test, n_samples=SHAP_SAMPLE_SIZE)
    
    # 5. Calculer SHAP
    explainer, shap_values, X_transformed = compute_shap_values(
        model, X_shap, feature_names, MODEL_TYPE
    )
    
    # 6. Générer les graphiques
    print("\n" + "="*70)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("="*70)
    
    plot_global_importance(
        shap_values, 
        feature_names, 
        OUTPUT_DIR / "1_global_importance.png"
    )
    
    plot_feature_impact(
        shap_values, 
        X_transformed, 
        feature_names, 
        class_names,
        OUTPUT_DIR / "2_feature_impact_summary.png"
    )
    
    plot_top_features_by_class(
        shap_values, 
        feature_names, 
        class_names,
        OUTPUT_DIR / "3_top_features_by_class.png",
        top_n_classes=5
    )
    
    # 7. Rapport de synthèse
    save_summary_report(shap_values, feature_names, class_names, OUTPUT_DIR)
    
    # Résumé final
    print("\n" + "="*70)
    print("  ✅ ANALYSE TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"\n📁 Tous les résultats sont disponibles dans:")
    print(f"   {OUTPUT_DIR}")
    print(f"\n📊 Graphiques générés:")
    print(f"   1. {OUTPUT_DIR / '1_global_importance.png'}")
    print(f"   2. {OUTPUT_DIR / '2_feature_impact_summary.png'}")
    print(f"   3. {OUTPUT_DIR / '3_top_features_by_class.png'}")
    print(f"\n📝 Rapport JSON:")
    print(f"   {OUTPUT_DIR / 'shap_analysis_report.json'}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
