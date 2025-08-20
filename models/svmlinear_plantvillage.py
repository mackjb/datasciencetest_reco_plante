#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear SVM (LinearSVC) sur PlantVillage.

Plan exécuté par ce script:
1) Extraction dataset PlantVillage (CSV propre existant) + traitement NA/doublons
2) Split stratifié train/test
3) Pipeline: Suppression NA (en amont) -> RobustScaler -> LinearSVC (l2)
4) CV 5-fold stratifiée avec métriques (balanced_accuracy, f1_macro)
5) Grid search hyperparamètres (C, loss, dual) + comparaison class_weight vs oversampling (ROS/SMOTE)
6) Évaluation finale sur test (balanced_accuracy, f1_macro)
7) Analyse et interprétation des coefficients (globale et par classe)

Notes:
- Le CSV propre utilisé est `dataset/plantvillage/csv/clean_with_features_data_plantvillage_segmented_all.csv`.
- LinearSVC ne fournit pas de probabilités; on utilise decision_function pour les scores si nécessaire.
- Contraintes scikit-learn: loss='hinge' n'est valide qu'avec dual=True. loss='squared_hinge' fonctionne avec dual=True/False.
- Penalty: on reste en 'l2' (plus stable en multi-classes et avec imbalance). 'l1' est possible (dual=False) mais plus coûteux; non inclus par défaut ici.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
import plotly.express as px
import plotly.io as pio
from src.helpers.helpers import PROJECT_ROOT

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "svmlinear_linearsvc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Colonnes non-features dans le CSV clean
NON_FEATURE_COLS: List[str] = [
    "species","filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_black","is_na","hash","is_duplicate_after_first","disease"
]


def print_plan() -> None:
    print("\nPlan d'exécution:")
    steps = [
        "1) Charger CSV propre + traiter NA/doublons",
        "2) Split stratifié train/test",
        "3) Pipeline: RobustScaler -> LinearSVC(l2)",
        "4) CV 5-fold stratifiée (balanced_accuracy, f1_macro)",
        "5) Grid search (C, loss, dual) + comparaison class_weight vs oversampling (ROS/SMOTE)",
        "6) Évaluation finale test (balanced_accuracy, f1_macro, rapport, matrice de confusion)",
        "7) Analyse des coefficients (global et top-k par classe)"
    ]
    for s in steps:
        print(" - ", s)


# -----------------------------
# Chargement & préparation des données
# -----------------------------

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Infère les colonnes de features: numériques et non présentes dans NON_FEATURE_COLS."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in NON_FEATURE_COLS and c != TARGET_COL]
    return features


def load_clean_dataset(csv_path: Path = CSV_PATH, drop_duplicates_flag: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Charge le CSV propre PlantVillage et applique:
    - suppression NA sur la cible
    - exclusion des doublons si colonne `is_duplicate_after_first` présente
    - sélection des colonnes features numériques
    Retourne df_clean, X (DataFrame), y (Series), feature_names (List[str]).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    n0 = len(df)

    # Cible non nulle
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET_COL}' absente du CSV. Colonnes: {list(df.columns)[:20]} ...")
    df = df.dropna(subset=[TARGET_COL])

    # Exclure doublons marqués
    if drop_duplicates_flag and ("is_duplicate_after_first" in df.columns):
        before = len(df)
        df = df[df["is_duplicate_after_first"] == False].copy()
        print(f"Doublons exclus via is_duplicate_after_first: {before - len(df)}")

    # Exclure lignes marquées NA si colonne is_na existe
    if "is_na" in df.columns:
        before = len(df)
        df = df[df["is_na"] == False].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Lignes exclues via is_na=True: {removed}")

    # Inférer features
    feature_names = get_feature_columns(df)
    if not feature_names:
        raise RuntimeError("Aucune colonne de feature détectée.")

    # Supprimer les lignes avec NA sur les features sélectionnées
    before = len(df)
    df = df.dropna(subset=feature_names)
    removed_feat_na = before - len(df)
    if removed_feat_na > 0:
        print(f"Lignes supprimées pour NA dans features: {removed_feat_na}")

    X = df[feature_names].copy()
    y = df[TARGET_COL].astype(str).copy()  # typage sûr

    print(f"Lignes chargées: {n0} -> après nettoyage: {len(df)}")
    print(f"#features: {len(feature_names)} | #classes: {y.nunique()}")
    return df, X, y, feature_names


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split: train={len(X_train)} | test={len(X_test)}")
    return X_train, X_test, y_train, y_test


# -----------------------------
# Modèle & évaluation
# -----------------------------

def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    svc = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        penalty="l2",
        dual=True,  # sera tuné
        class_weight="balanced",
        max_iter=5000,
        tol=1e-3,
        random_state=random_state,
    )
    # Imblearn Pipeline pour insérer un sampler dans la CV (passthrough/ROS/SMOTE)
    pipe = ImbPipeline(steps=[
        ("sampler", "passthrough"),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ("model", svc),
    ])
    return pipe


def cross_validate_baseline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> dict:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
    }
    print("\nCV 5-fold (baseline, sans tuning)...")
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1)
    # Sauvegarde
    pd.DataFrame(cv).to_csv(RESULTS_DIR / "cv_baseline_scores.csv", index=False)
    print({k: np.mean(v) for k, v in cv.items() if k.startswith("test_")})
    return cv


def tune_hyperparams(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> GridSearchCV:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
    }
    # Deux familles d'approches: class_weight vs oversampling (ROS/SMOTE), en évitant la double correction
    # Respect des contraintes: loss='hinge' -> dual=True uniquement. loss='squared_hinge' -> dual True/False.
    Cs = np.logspace(-3, 2, 6)
    param_grid = [
        # Passthrough (class_weight peut être activé)
        {
            "sampler": ["passthrough"],
            "model__class_weight": [None, "balanced"],
            "model__C": Cs,
            "model__penalty": ["l2"],
            # Deux sous-espaces compatibles
            "model__loss": ["squared_hinge"],
            "model__dual": [True, False],
        },
        {
            "sampler": ["passthrough"],
            "model__class_weight": [None, "balanced"],
            "model__C": Cs,
            "model__penalty": ["l2"],
            "model__loss": ["hinge"],
            "model__dual": [True],
        },
        # Oversampling (ne pas utiliser class_weight simultanément)
        {
            "sampler": [
                RandomOverSampler(random_state=random_state),
                SMOTE(random_state=random_state, k_neighbors=3),
            ],
            "model__class_weight": [None],
            "model__C": Cs,
            "model__penalty": ["l2"],
            "model__loss": ["squared_hinge"],
            "model__dual": [True, False],
        },
        {
            "sampler": [
                RandomOverSampler(random_state=random_state),
                SMOTE(random_state=random_state, k_neighbors=3),
            ],
            "model__class_weight": [None],
            "model__C": Cs,
            "model__penalty": ["l2"],
            "model__loss": ["hinge"],
            "model__dual": [True],
        },
    ]
    print("\nGridSearchCV (C, loss, dual) ...")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        refit="f1_macro",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )
    gs.fit(X, y)
    # Sauvegarder résultats bruts
    pd.DataFrame(gs.cv_results_).to_csv(RESULTS_DIR / "gridsearch_results.csv", index=False)
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({"best_params": gs.best_params_, "best_score_f1_macro": gs.best_score_}, f, indent=2)
    print(f"Best params: {gs.best_params_} | best f1_macro: {gs.best_score_:.4f}")
    return gs


def _label_approach_from_params(row: pd.Series) -> str:
    """Retourne un label d'approche (ClassWeight vs Oversampling) pour une ligne cv_results_."""
    sampler_val = row.get("param_sampler", None)
    cw_val = row.get("param_model__class_weight", None)
    s = str(sampler_val)
    if "RandomOverSampler" in s:
        return "Oversampling: ROS"
    if "SMOTE" in s:
        return "Oversampling: SMOTE"
    if str(cw_val) == "balanced":
        return "ClassWeight balanced"
    return "Baseline"


def summarize_and_plot(cv_results_df: pd.DataFrame) -> None:
    """Enrichit les résultats CV, sauvegarde des synthèses et génère des visualisations Plotly."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = cv_results_df.copy()
    # Normalisation des colonnes de paramètres
    if "param_model__C" in df.columns:
        df["C"] = pd.to_numeric(df["param_model__C"], errors="coerce")
    if "param_model__loss" in df.columns:
        df["loss"] = df["param_model__loss"].astype(str)
    if "param_model__dual" in df.columns:
        df["dual"] = df["param_model__dual"].astype(str)
    if "param_model__class_weight" in df.columns:
        df["class_weight"] = df["param_model__class_weight"].astype(str)
    else:
        df["class_weight"] = "None"

    # Label d'approche
    df["approach"] = df.apply(_label_approach_from_params, axis=1)
    # Nom sampler lisible
    def _sampler_name(v: object) -> str:
        s = str(v)
        if "RandomOverSampler" in s:
            return "ROS"
        if "SMOTE" in s:
            return "SMOTE"
        if s == "passthrough":
            return "passthrough"
        return s
    df["sampler_name"] = df.get("param_sampler", "passthrough").apply(_sampler_name)

    # Sauvegardes CSV
    df.to_csv(plots_dir / "cv_results_enriched.csv", index=False)
    best_per_approach = (
        df.sort_values("mean_test_f1_macro", ascending=False)
          .groupby("approach", as_index=False)
          .first()
    )
    best_per_approach.to_csv(plots_dir / "best_by_approach.csv", index=False)
    best_row = df.sort_values("mean_test_f1_macro", ascending=False).iloc[0]
    best_summary = {
        "approach": best_row.get("approach"),
        "sampler_name": best_row.get("sampler_name"),
        "class_weight": best_row.get("class_weight"),
        "C": float(best_row.get("C")) if pd.notnull(best_row.get("C")) else None,
        "loss": best_row.get("loss"),
        "dual": best_row.get("dual"),
        "mean_test_f1_macro": float(best_row.get("mean_test_f1_macro")),
        "mean_test_bal_acc": float(best_row.get("mean_test_bal_acc")) if "mean_test_bal_acc" in df.columns and pd.notnull(best_row.get("mean_test_bal_acc")) else None,
        "rank_test_f1_macro": int(best_row.get("rank_test_f1_macro")) if pd.notnull(best_row.get("rank_test_f1_macro")) else None,
    }
    with open(plots_dir / "best_overall.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    # Scatter Balanced Accuracy (log C)
    fig_scatter_bal = px.scatter(
        df,
        x="C",
        y="mean_test_bal_acc",
        color="approach",
        symbol="loss",
        hover_data=["mean_test_f1_macro", "loss", "dual", "class_weight"],
        title="Balanced Accuracy vs C (symbol=loss) par approche",
    )
    fig_scatter_bal.update_layout(xaxis_type="log")
    pio.write_html(fig_scatter_bal, plots_dir / "scatter_bal_acc.html", auto_open=False)

    # Scatter Macro F1 (log C)
    if "mean_test_f1_macro" in df.columns:
        fig_scatter_f1 = px.scatter(
            df,
            x="C",
            y="mean_test_f1_macro",
            color="approach",
            symbol="loss",
            hover_data=["mean_test_bal_acc", "loss", "dual", "class_weight"],
            title="Macro F1 vs C (symbol=loss) par approche",
        )
        fig_scatter_f1.update_layout(xaxis_type="log")
        pio.write_html(fig_scatter_f1, plots_dir / "scatter_f1_macro.html", auto_open=False)

    # Heatmap (moyenne bal_acc) par (C, loss) facettée par approche
    try:
        fig_heat = px.density_heatmap(
            df,
            x="C",
            y="loss",
            z="mean_test_bal_acc",
            facet_col="approach",
            histfunc="avg",
            color_continuous_scale="Viridis",
            title="Heatmap Balanced Accuracy par approche (C vs loss)",
        )
        fig_heat.update_layout(xaxis_type="log")
        pio.write_html(fig_heat, plots_dir / "heatmap_bal_acc.html", auto_open=False)
    except Exception as e:
        with open(plots_dir / "plot_errors.log", "a") as f:
            f.write(f"Heatmap error: {e}\n")

    # Heatmap (moyenne f1_macro) par (C, loss) facettée par approche
    try:
        fig_heat_f1 = px.density_heatmap(
            df,
            x="C",
            y="loss",
            z="mean_test_f1_macro",
            facet_col="approach",
            histfunc="avg",
            color_continuous_scale="Plasma",
            title="Heatmap Macro F1 par approche (C vs loss)",
        )
        fig_heat_f1.update_layout(xaxis_type="log")
        pio.write_html(fig_heat_f1, plots_dir / "heatmap_f1_macro.html", auto_open=False)
    except Exception as e:
        with open(plots_dir / "plot_errors.log", "a") as f:
            f.write(f"Heatmap F1 error: {e}\n")


def evaluate_on_test(best_estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print("\nÉvaluation finale sur test...")
    y_pred = best_estimator.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=best_estimator.named_steps["model"].classes_)

    # Sauvegarde
    (RESULTS_DIR / "evaluation").mkdir(exist_ok=True)
    with open(RESULTS_DIR / "evaluation" / "classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(RESULTS_DIR / "evaluation" / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    return {"balanced_accuracy": bal_acc, "f1_macro": f1m, "report": report, "confusion_matrix": cm}


def analyze_coefficients(best_estimator: Pipeline, feature_names: List[str], top_k: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (global_importance_df, per_class_topk_df) et sauvegarde en CSV."""
    svc: LinearSVC = best_estimator.named_steps["model"]
    classes = svc.classes_
    coefs = svc.coef_  # shape (n_classes, n_features)
    abs_coefs = np.abs(coefs)

    # Importance globale = moyenne des |coef| sur les classes
    global_importance = abs_coefs.mean(axis=0)
    global_df = pd.DataFrame({"feature": feature_names, "importance": global_importance})
    global_df = global_df.sort_values("importance", ascending=False)
    global_df.to_csv(RESULTS_DIR / "coefficients_global.csv", index=False)

    # Top-k par classe
    rows = []
    for ci, cls in enumerate(classes):
        order = np.argsort(-abs_coefs[ci])
        top_idx = order[:top_k]
        for rank, fi in enumerate(top_idx, start=1):
            rows.append({
                "class": cls,
                "rank": rank,
                "feature": feature_names[fi],
                "coef": coefs[ci, fi],
                "abs_coef": abs_coefs[ci, fi],
            })
    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(RESULTS_DIR / "coefficients_per_class_topk.csv", index=False)
    return global_df, per_class_df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = build_pipeline(RANDOM_STATE)

    # Baseline CV
    _ = cross_validate_baseline(pipe, X_train, y_train, cv_splits=5, random_state=RANDOM_STATE)

    # Hyperparameter tuning
    gs = tune_hyperparams(pipe, X_train, y_train, cv_splits=5, random_state=RANDOM_STATE)
    # Synthèse Plotly des résultats (approches + hyperparams)
    try:
        cv_df = pd.DataFrame(gs.cv_results_)
        summarize_and_plot(cv_df)
    except Exception as e:
        print(f"[WARN] summarize_and_plot a échoué: {e}")
    best_pipe: Pipeline = gs.best_estimator_

    # Fit final on full train
    best_pipe.fit(X_train, y_train)

    # Test evaluation
    _ = evaluate_on_test(best_pipe, X_test, y_test)

    # Coefficients analysis
    _ = analyze_coefficients(best_pipe, feature_names, top_k=15)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
