#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


Fonctionnalités:
- Chargement CSV propre avec features
- Encodage des labels
- Pipelines:
  * XGBoost (features brutes scalées par RobustScaler)
  * XGBoost + PCA
  * XGBoost + LDA
- SMOTE sur le train pendant la CV et le fit final
- CV 5-fold (stratifiée) sur le train pour chaque pipeline et config XGB
- Ré-entraînement sur tout le train et évaluation test (accuracy, F1 macro/weighted)
- Exports CSV des résultats globaux et par classe
- Sauvegarde de matrices de confusion et graphiques (PNG)

Notes:
-
- Ajout d'un flag --quick pour réduire la charge (moins d'arbres et CV à 3 folds) pour un premier run rapide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from src.helpers.helpers import PROJECT_ROOT


# Typage utilitaire pour clarifier les structures de données
from typing import List, Tuple, Dict, Any, TypeAlias

# Alias de types pour clarifier le code
FeatureMatrix: TypeAlias = np.ndarray
LabelArray: TypeAlias = np.ndarray
PipelineDatasets: TypeAlias = tuple[FeatureMatrix, FeatureMatrix]
XGBConfig: TypeAlias = dict[str, int | float]


# -----------------------------
# Config par défaut
# -----------------------------
SEED = 42
DEFAULT_CSV = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
TARGET_COL_NOTEBOOK = "nom_maladie"  # d'après le notebook

# Configs XGBoost (comme le notebook)
XGB_CONFIGS_FULL: Dict[str, XGBConfig] = {
    "Baseline": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
    "Deep Trees": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 10},
    "Shallow Trees": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 3},
}

XGB_CONFIGS_QUICK: Dict[str, XGBConfig] = {
    "Quick": {"n_estimators": 80, "learning_rate": 0.15, "max_depth": 6},
}


# -----------------------------
# Utils
# -----------------------------

def evaluate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcule un ensemble de métriques globales et par classe.

    Paramètres
    ----------
    y_true : np.ndarray
        Vecteur des vraies classes (après encodage LabelEncoder).
    y_pred : np.ndarray
        Vecteur des classes prédites par le modèle.

    Retour
    ------
    tuple
        (accuracy globale, F1 macro, F1 pondéré, précision par classe,
        rappel par classe, F1 par classe, support par classe)

    Notes
    -----
    - ``zero_division=0`` évite les warnings lorsqu'une classe est absente.
    - Le support correspond au nombre d'exemples par classe dans y_true.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.bincount(y_true)
    return acc, f1_macro, f1_weighted, precision, recall, f1_per_class, support


def save_confusion_and_worst_classes(cm: np.ndarray, classes: list[str], title_prefix: str, out_dir: Path) -> None:
    """Sauvegarde des matrices de confusion (brute et normalisée) et un barplot des classes les moins bien prédites.

    - confusion_matrix_<title>.png : valeurs brutes (comptes)
    - confusion_matrix_normalized_<title>.png : normalisée par lignes (rappels) avec annotations en %
    - worst_classes_<title>.png : tri des classes par accuracy (diagonale / somme ligne)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Matrice de confusion brute (comptes)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes,
                     cmap="Blues", cbar_kws={"label": "Nombre"})
    plt.title(f"Matrice de confusion (comptes) - {title_prefix}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{title_prefix.replace(' ', '_')}.png", dpi=150)
    plt.close()

    # 2) Matrice de confusion normalisée par lignes (rappel par classe)
    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = (cm / row_sums).astype(float)
        norm_cm = np.nan_to_num(norm_cm, nan=0.0)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(norm_cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes,
                     cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    # Annoter aussi en pourcentage pour la lisibilité
    for text in ax.texts:
        try:
            val = float(text.get_text())
            text.set_text(f"{val*100:.0f}%")
        except Exception:
            pass
    plt.title(f"Matrice de confusion normalisée (rappel %) - {title_prefix}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_normalized_{title_prefix.replace(' ', '_')}.png", dpi=150)
    plt.close()

    # 3) Classes les moins bien prédites (accuracy par classe = diagonale / somme ligne)
    class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).astype(float)
    class_acc_df = pd.DataFrame({"Classe": classes, "Accuracy": class_acc}).sort_values(by="Accuracy")
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Accuracy", y="Classe", data=class_acc_df, palette="Reds_r")
    plt.xlim(0, 1)
    plt.xlabel("Accuracy par classe")
    plt.title(f"Classes les moins bien prédites - {title_prefix}")
    plt.tight_layout()
    plt.savefig(out_dir / f"worst_classes_{title_prefix.replace(' ', '_')}.png", dpi=150)
    plt.close()


# -----------------------------
# Main logique
# -----------------------------

def run(csv_path: Path, target_col: str, quick: bool = False, test_size: float = 0.2, seed: int = SEED) -> None:
    """Pipeline complet d'entraînement/évaluation XGBoost sur PlantVillage.

    Étapes principales:
    1) Chargement et validation du CSV (+ fallback de la colonne cible).
    2) Encodage des labels, sélection et nettoyage des features numériques.
    3) Split train/test stratifié puis normalisation robuste (RobustScaler).
    4) Construction de représentations alternatives (PCA, LDA) sans fuite.
    5) Validation croisée stratifiée avec SMOTE sur le train à chaque fold.
    6) Ré-entraînement sur tout le train (avec SMOTE) et évaluation sur test.
    7) Export des résultats (globaux et par classe) et des figures.

    Paramètres
    ----------
    csv_path : Path
        Chemin du CSV contenant les features et la cible.
    target_col : str
        Nom de la colonne cible (ex: ``nom_maladie``). Fallback sur colonnes
        alternatives si absente.
    quick : bool
        Si True, exécution rapide (3-fold, moins d'arbres, PCA plus petite).
    test_size : float
        Proportion du test set pour ``train_test_split``.
    seed : int
        Graine aléatoire pour la reproductibilité.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    print(f"Chargement: {csv_path}")
    df: pd.DataFrame = pd.read_csv(csv_path)

    if target_col not in df.columns:
        # Fallback: s'il n'existe pas, tenter colonnes connues du projet
        alternatives = ["species", "disease", "label"]
        for alt in alternatives:
            if alt in df.columns:
                print(f"[WARN] Colonne cible '{target_col}' absente. Utilisation de '{alt}'.")
                target_col = alt
                break
        else:
            raise ValueError(f"Colonne cible '{target_col}' absente du CSV. Colonnes: {list(df.columns)[:30]} ...")

    # Créer un répertoire de résultats spécifique à la cible pour éviter les écrasements
    results_dir = PROJECT_ROOT / "results" / "models" / "xgboost" / str(target_col)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Encoder labels
    # On encode la cible en entiers [0..K-1] pour XGBoost et LDA.
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    classes: List[str] = le.classes_.tolist()
    n_classes = len(classes)
    print(f"Classes détectées ({n_classes}): {classes[:10]}{' ...' if n_classes>10 else ''}")

    # Colonnes numériques -> features
    # On ne garde que les colonnes numériques (features dérivées) et on
    # exclut explicitement des colonnes "métadonnées" qui n'apportent pas
    # d'information discriminante utile au modèle.
    numeric_columns: List[str] = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = [
        # colonnes non-features fréquentes
        'width', 'height', 'width_img', 'height_img', 'is_black', 'is_na', 'is_duplicate_after_first',
        'num_channels', 'file_size',
    ]
    numeric_columns = [c for c in numeric_columns if c not in exclude_cols]

    # Remplir NaN numériques par médiane (robuste aux outliers)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    X = df[numeric_columns].values

    print(f"X shape: {X.shape} | y shape: {y.shape} | #features: {len(numeric_columns)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Scaler global sur train (puis appliquer à test)
    # Important: fit sur train uniquement pour éviter toute fuite d'information
    # vers le jeu de test.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Pipelines
    if quick:
        n_components_pca = min(30, X_train_scaled.shape[1])
    else:
        n_components_pca = min(50, X_train_scaled.shape[1])

    # Fit PCA/LDA on training only, then transform both sets (avoid leakage)
    # PCA réduit la dimension en conservant la variance. Le nombre de
    # composantes est borné par le nombre de features et un plafond
    # (plus petit en mode quick pour accélérer).
    pca = PCA(n_components=n_components_pca, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # LDA projette dans un espace de dimension au plus (K-1), K = nb classes.
    lda = LDA(n_components=min(n_classes - 1, X_train_scaled.shape[1]))
    X_train_lda = lda.fit(X_train_scaled, y_train).transform(X_train_scaled)
    X_test_lda = lda.transform(X_test_scaled)

    pipelines: Dict[str, PipelineDatasets] = {
        "XGBoost": (X_train_scaled, X_test_scaled),
        "XGBoost + PCA": (X_train_pca, X_test_pca),
        "XGBoost + LDA": (X_train_lda, X_test_lda),
    }

    # Configs XGB et CV folds
    xgb_configs: Dict[str, XGBConfig] = XGB_CONFIGS_QUICK if quick else XGB_CONFIGS_FULL
    cv_splits = 3 if quick else 5

    results: List[Dict[str, Any]] = []
    class_results: List[Dict[str, Any]] = []

    for config_name, params in xgb_configs.items():
        print(f"\n===== Configuration : {config_name} =====")
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

        for pipe_name, (Xtr, Xte) in pipelines.items():
            print(f"\n🚀 Pipeline: {pipe_name}")

            # CV sur le train
            # À chaque fold: SMOTE est appliqué uniquement sur la portion
            # d'entraînement (X_tr, y_tr) pour équilibrer les classes.
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            f1_scores = []
            for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr, y_train), start=1):
                X_tr, X_val = Xtr[tr_idx], Xtr[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                smote = SMOTE(random_state=seed)
                X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

                # fit/predict
                # Entraînement sur les données rééchantillonnées puis
                # évaluation sur la validation du fold.
                model.fit(X_tr_bal, y_tr_bal)
                y_val_pred = model.predict(X_val)
                f1_fold = f1_score(y_val, y_val_pred, average="weighted")
                f1_scores.append(f1_fold)
                print(f"Fold {fold}/{cv_splits} - F1_weighted={f1_fold:.4f}")

            f1_mean, f1_std = float(np.mean(f1_scores)), float(np.std(f1_scores))

            # Réentraînement sur tout le train + SMOTE
            # On réapplique SMOTE sur l'ensemble du train de ce pipeline,
            # puis on réentraîne le modèle final avant de tester.
            smote = SMOTE(random_state=seed)
            X_train_bal, y_train_bal = smote.fit_resample(Xtr, y_train)
            model.fit(X_train_bal, y_train_bal)
            y_test_pred = model.predict(Xte)

            # Évaluations globales et par classe
            acc, f1_macro, f1_weighted, precision, recall, f1_per_class, support = evaluate_metrics(y_test, y_test_pred)

            # Stockage résultats globaux
            results.append({
                "Pipeline": pipe_name,
                "Config": config_name,
                "CV_F1_mean": f1_mean,
                "CV_F1_std": f1_std,
                "Test_Accuracy": float(acc),
                "Test_F1_macro": float(f1_macro),
                "Test_F1_weighted": float(f1_weighted),
                "Model": model,
                "y_test": y_test,
                "y_pred": y_test_pred,
            })

            # Résultats par classe
            # On assemble précision/rappel/F1 et le support pour chaque
            # classe afin d'identifier les catégories difficiles.
            for i, classe in enumerate(classes):
                class_results.append({
                    "Pipeline": pipe_name,
                    "Config": config_name,
                    "Classe": str(classe),
                    "Precision": float(precision[i]) if i < len(precision) else 0.0,
                    "Recall": float(recall[i]) if i < len(recall) else 0.0,
                    "F1_score": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
                    "Support": int(support[i]) if i < len(support) else 0,
                })

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_test_pred)
            save_confusion_and_worst_classes(cm, classes, f"{pipe_name} ({config_name})", results_dir)

    # Tableaux globaux
    # Tri des résultats par F1 pondéré sur test, puis export CSV.
    results_df = pd.DataFrame(results)
    # Trier par F1_weighted test
    results_df_sorted = results_df.sort_values(by="Test_F1_weighted", ascending=False)
    results_df_export = results_df_sorted.drop(columns=["Model", "y_test", "y_pred"], errors="ignore")
    results_df_export.to_csv(results_dir / "global_results.csv", index=False)

    class_results_df = pd.DataFrame(class_results)
    class_results_df.to_csv(results_dir / "class_results.csv", index=False)

    print("\n📊 Résultats globaux (top):")
    print(results_df_export.head(10))

    # Feature importance pour le meilleur modèle XGBoost simple
    # NB: uniquement si le meilleur pipeline est "XGBoost" (pas PCA/LDA),
    # et si le modèle expose ``feature_importances_`` aligné avec les
    # colonnes d'entrée d'origine.
    try:
        best_row = results_df_sorted.iloc[0]
        if best_row["Pipeline"] == "XGBoost":
            best_model: XGBClassifier = best_row["Model"]
            importances = getattr(best_model, "feature_importances_", None)
            if importances is not None and len(importances) == len(numeric_columns):
                feat_df = pd.DataFrame({"Feature": numeric_columns, "Importance": importances})
                feat_df = feat_df.sort_values("Importance", ascending=False).head(20)
                plt.figure(figsize=(10, 8))
                sns.barplot(x="Importance", y="Feature", data=feat_df)
                plt.title(f"Top 20 features - {best_row['Pipeline']} ({best_row['Config']})")
                plt.tight_layout()
                plt.savefig(results_dir / "top20_features_xgboost.png")
                plt.close()
    except Exception as e:
        print(f"[WARN] Plot feature importance ignoré: {e}")

    # Sauvegarde d'un petit récap JSON
    try:
        summary = results_df_export.head(10).to_dict(orient="records")
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass


# -----------------------------
# Entrée CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la CLI."""
    p = argparse.ArgumentParser(description="XGBoost PlantVillage (équivalent notebook)")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Chemin vers le CSV avec features")
    p.add_argument("--target", type=str, default=TARGET_COL_NOTEBOOK, help="Nom de la colonne cible")
    p.add_argument("--test_size", type=float, default=0.2, help="Taille du test set")
    p.add_argument("--seed", type=int, default=SEED, help="Seed aléatoire")
    p.add_argument("--quick", action="store_true", help="Exécution rapide (moins de folds/arbres)")
    return p.parse_args()


def main():
    """Point d'entrée: lit les arguments puis exécute ``run``."""
    args = parse_args()
    run(Path(args.csv), args.target, quick=args.quick, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
