#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExtraTreesClassifier sur PlantVillage avec bonnes pratiques:
- Pas de scaler (arbres invariants au scale), mais imputation + winsorisation numeric.
- Encodage catégoriel OneHot (handle_unknown=ignore).
- Gestion du déséquilibre: comparaison class_weight vs over/under-sampling (pas de cumul).
- Split stratifié, CV 5-fold stratifiée, refit sur f1_macro.
- Sauvegardes complètes des résultats + interprétabilité (importances, SHAP optionnel).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

try:
    import shap  # optionnel
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from src.helpers.helpers import PROJECT_ROOT

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "extratrees_plantvillage"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Colonnes non-features (même logique que la logreg)
NON_FEATURE_COLS: List[str] = [
    "species","filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_black","is_na","hash","is_duplicate_after_first","disease"
]


def print_plan() -> None:
    print("\nPlan d'exécution (ExtraTrees):")
    steps = [
        "1) Charger CSV propre + traiter NA cible/drapeaux is_na & is_duplicate_after_first",
        "2) Détecter colonnes numériques/catégorielles (exclure NON_FEATURE_COLS)",
        "3) Pipeline: sampler -> preprocess(num: impute+winsorize, cat: impute+OHE) -> ExtraTrees",
        "4) CV 5-fold stratifiée (balanced_accuracy, f1_macro)",
        "5) GridSearch: n_estimators, max_depth, max_features, min_samples_*, class_weight vs ROS/SMOTE/RUS",
        "6) Évaluation test + rapport + matrice de confusion",
        "7) Interprétabilité: feature_importances_ (+ SHAP optionnel)",
    ]
    for s in steps:
        print(" -", s)


# -----------------------------
# Prétraitement utils
# -----------------------------

def make_one_hot_encoder():
    """Créer un OneHotEncoder compatible scikit-learn >=1.2 et versions antérieures."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip par quantiles pour atténuer valeurs extrêmes (outliers).
    Par défaut: 1e et 99e percentiles par feature.
    """
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.minimum(np.maximum(X, self.lower_bounds_), self.upper_bounds_)
        return X


# -----------------------------
# Chargement & préparation des données
# -----------------------------

def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Détecte colonnes numeric/categorical utilisables (exclut NON_FEATURE_COLS & TARGET_COL)."""
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS and c != TARGET_COL]
    num_cols = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in candidates if c not in num_cols and df[c].dtype.name in ("object", "category", "bool")]
    return num_cols, cat_cols


def load_clean_dataset(csv_path: Path = CSV_PATH, drop_duplicates_flag: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Charge le CSV propre PlantVillage et applique:
    - suppression NA sur la cible
    - exclusion des doublons si colonne `is_duplicate_after_first` présente
    - exclusion lignes is_na=True si colonne existe
    - sélection des colonnes features (numériques + catégorielles)
    ATTENTION: on n'élimine pas les NA des features (imputation dans le pipeline).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    n0 = len(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET_COL}' absente du CSV. Colonnes: {list(df.columns)[:20]} ...")
    df = df.dropna(subset=[TARGET_COL])

    if drop_duplicates_flag and ("is_duplicate_after_first" in df.columns):
        before = len(df)
        df = df[df["is_duplicate_after_first"] == False].copy()
        print(f"Doublons exclus via is_duplicate_after_first: {before - len(df)}")

    if "is_na" in df.columns:
        before = len(df)
        df = df[df["is_na"] == False].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Lignes exclues via is_na=True: {removed}")

    num_cols, cat_cols = infer_feature_columns(df)
    if not num_cols and not cat_cols:
        raise RuntimeError("Aucune colonne de feature détectée.")

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].astype(str).copy()

    print(f"Lignes chargées: {n0} -> après nettoyage: {len(df)}")
    print(f"#features: {X.shape[1]} (num={len(num_cols)}, cat={len(cat_cols)}) | #classes: {y.nunique()}")
    return df, X, y, num_cols, cat_cols


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split: train={len(X_train)} | test={len(X_test)}")
    return X_train, X_test, y_train, y_test


# -----------------------------
# Pipeline & évaluation
# -----------------------------

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(lower=0.01, upper=0.99)),
        # IMPORTANT: pas de scaler ici (arbres invariants aux échelles)
    ])
    cat_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_one_hot_encoder()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        n_jobs=None,  # laisser par défaut
    )
    return pre


def build_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int = RANDOM_STATE) -> ImbPipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        class_weight=None,  # comparé à "balanced" en Grid
        random_state=random_state,
        n_jobs=1,
    )
    pipe = ImbPipeline(steps=[
        ("sampler", "passthrough"),  # sera configuré par Grid (passthrough, ROS, SMOTE, RUS)
        ("preprocess", pre),
        ("model", model),
    ])
    return pipe


def cross_validate_baseline(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> dict:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
    }
    print("\nCV 5-fold (baseline, sans tuning)...")
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=1)
    pd.DataFrame(cv).to_csv(RESULTS_DIR / "cv_baseline_scores.csv", index=False)
    print({k: float(np.mean(v)) for k, v in cv.items() if k.startswith("test_")})
    return cv


def tune_hyperparams(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> GridSearchCV:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
    }

    # Grille réduite pour limiter la charge mémoire/CPU
    param_grid_common = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
        "model__bootstrap": [False],
        "model__criterion": ["gini"],
    }

    param_grid = [
        {
            "sampler": ["passthrough"],
            "model__class_weight": [None, "balanced"],
            **param_grid_common,
        },
        {
            # On ne garde que ROS pour réduire le coût; SMOTE/RUS retirés
            "sampler": [
                RandomOverSampler(random_state=random_state),
            ],
            "model__class_weight": [None],  # éviter sampler + balanced en même temps
            **param_grid_common,
        },
    ]

    print("\nGridSearchCV (ExtraTrees hyperparams + équilibrage) ...")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        refit="f1_macro",
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)
    pd.DataFrame(gs.cv_results_).to_csv(RESULTS_DIR / "gridsearch_results.csv", index=False)
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({"best_params": gs.best_params_, "best_score_f1_macro": float(gs.best_score_)}, f, indent=2)
    print(f"Best params: {gs.best_params_} | best f1_macro: {gs.best_score_:.4f}")
    return gs


def _label_approach_from_params(row: pd.Series) -> str:
    sampler_val = row.get("param_sampler", None)
    cw_val = row.get("param_model__class_weight", None)
    s = str(sampler_val)
    if "RandomUnderSampler" in s:
        return "Undersampling: RUS"
    if "RandomOverSampler" in s:
        return "Oversampling: ROS"
    if "SMOTE" in s:
        return "Oversampling: SMOTE"
    if str(cw_val) == "balanced":
        return "ClassWeight balanced"
    return "Baseline"


def summarize_and_plot(cv_results_df: pd.DataFrame) -> None:
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = cv_results_df.copy()
    for k in ["param_model__n_estimators","param_model__max_depth","param_model__min_samples_split","param_model__min_samples_leaf"]:
        if k in df.columns:
            df[k.replace("param_model__", "")] = pd.to_numeric(df[k], errors="coerce")
    if "param_model__max_features" in df.columns:
        df["max_features"] = df["param_model__max_features"].astype(str)

    df["approach"] = df.apply(_label_approach_from_params, axis=1)
    def _sampler_name(v: object) -> str:
        s = str(v)
        if "RandomUnderSampler" in s: return "RUS"
        if "RandomOverSampler" in s: return "ROS"
        if "SMOTE" in s: return "SMOTE"
        if s == "passthrough": return "passthrough"
        return s
    df["sampler_name"] = df.get("param_sampler", "passthrough").apply(_sampler_name)

    df.to_csv(plots_dir / "cv_results_enriched.csv", index=False)
    best_per_approach = (
        df.sort_values("mean_test_f1_macro", ascending=False)
          .groupby("approach", as_index=False)
          .first()
    )
    best_per_approach.to_csv(plots_dir / "best_by_approach.csv", index=False)

    # Scatter f1_macro vs n_estimators
    try:
        fig1 = px.scatter(
            df, x="n_estimators", y="mean_test_f1_macro",
            color="approach", symbol="max_features",
            hover_data=["max_depth","min_samples_split","min_samples_leaf"],
            title="ExtraTrees: Macro F1 vs n_estimators"
        )
        pio.write_html(fig1, plots_dir / "scatter_f1_vs_n_estimators.html", auto_open=False)
    except Exception as e:
        with open(plots_dir / "plot_errors.log", "a") as f:
            f.write(f"scatter_f1_vs_n_estimators error: {e}\n")


def get_transformed_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Essaie d'obtenir les noms des features transformées; fallback manuel si indisponible."""
    names: List[str] = []
    try:
        # sklearn >= 1.0
        names = preprocessor.get_feature_names_out().tolist()
        return names
    except Exception:
        pass

    try:
        # Fallback: concat num + OHE(cat)
        names = list(numeric_features)
        cat_pipe = preprocessor.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        cats = ohe.categories_
        for col, cats_col in zip(categorical_features, cats):
            names.extend([f"{col}={str(v)}" for v in cats_col])
        return names
    except Exception:
        # Dernier recours
        return list(numeric_features) + list(categorical_features)


def evaluate_on_test(best_estimator: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print("\nÉvaluation finale sur test...")
    y_pred = best_estimator.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=best_estimator.named_steps["model"].classes_)

    (RESULTS_DIR / "evaluation").mkdir(exist_ok=True)
    with open(RESULTS_DIR / "evaluation" / "classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(RESULTS_DIR / "evaluation" / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    return {"balanced_accuracy": bal_acc, "f1_macro": f1m, "report": report, "confusion_matrix": cm}


def analyze_feature_importances(best_estimator: ImbPipeline, feature_names_transformed: List[str], top_k: int = 40) -> pd.DataFrame:
    model: ExtraTreesClassifier = best_estimator.named_steps["model"]
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names_transformed,
        "importance": importances
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
    print("Top features:\n", imp_df.head(15))
    return imp_df.head(top_k)


def shap_analysis(best_estimator: ImbPipeline, X_train: pd.DataFrame, feature_names_transformed: List[str], sample_size: int = 500) -> None:
    if not HAS_SHAP:
        print("[INFO] SHAP non disponible - ignoré.")
        return
    try:
        preprocess = best_estimator.named_steps["preprocess"]
        model: ExtraTreesClassifier = best_estimator.named_steps["model"]
        Xtr = preprocess.transform(X_train)
        if sample_size and Xtr.shape[0] > sample_size:
            idx = np.random.RandomState(RANDOM_STATE).choice(Xtr.shape[0], size=sample_size, replace=False)
            Xtr_s = Xtr[idx]
        else:
            Xtr_s = Xtr

        explainer = shap.TreeExplainer(model)
        # Pour multiclasses, shap_values est une liste; on illustre la première classe
        shap_values = explainer.shap_values(Xtr_s)
        shap_dir = RESULTS_DIR / "shap"
        shap_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(shap_values, list) and len(shap_values) > 0:
            vals = shap_values[0]
        else:
            vals = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, Xtr_s, feature_names=feature_names_transformed, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary_class0.png", dpi=150)
        plt.close()
        print("SHAP summary sauvegardé.")
    except Exception as e:
        print(f"[WARN] SHAP a échoué: {e}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()
    t0 = time.time()

    # Charger données
    df, X, y, num_cols, cat_cols = load_clean_dataset(CSV_PATH)

    # Split stratifié
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Pipeline
    pipe = build_pipeline(num_cols, cat_cols, RANDOM_STATE)

    # Baseline CV
    _ = cross_validate_baseline(pipe, X_train, y_train, cv_splits=5, random_state=RANDOM_STATE)

    # Tuning
    gs = tune_hyperparams(pipe, X_train, y_train, cv_splits=5, random_state=RANDOM_STATE)

    # Synthèse plots
    try:
        cv_df = pd.DataFrame(gs.cv_results_)
        summarize_and_plot(cv_df)
    except Exception as e:
        print(f"[WARN] summarize_and_plot a échoué: {e}")

    best_pipe: ImbPipeline = gs.best_estimator_

    # Fit final sur train complet
    best_pipe.fit(X_train, y_train)

    # Évaluation test
    _ = evaluate_on_test(best_pipe, X_test, y_test)

    # Noms de features transformées (pour importances/SHAP)
    try:
        feature_names_transformed = get_transformed_feature_names(
            best_pipe.named_steps["preprocess"], num_cols, cat_cols
        )
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(best_pipe.named_steps["model"].n_features_in_)]

    # Importances
    _ = analyze_feature_importances(best_pipe, feature_names_transformed, top_k=40)

    # SHAP (optionnel)
    shap_analysis(best_pipe, X_train, feature_names_transformed, sample_size=500)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()