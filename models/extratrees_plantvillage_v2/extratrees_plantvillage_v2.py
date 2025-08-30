#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExtraTreesClassifier sur PlantVillage (v2) avec bonnes pratiques, commentaires et traçabilité complète.

Objectifs et justifications:
- Encodage / normalisation:
  * Par défaut: pas de standardisation/normalisation (arbres invariants aux échelles).
  * Numériques: imputation médiane + winsorisation désactivée par défaut (ablation 0.1%–99.9% en grille).
  * Conditionnel: RobustScaler activé uniquement quand un sur-échantillonnage synthétique (SMOTE/BorderlineSMOTE) est utilisé.
  * Catégorielles: OneHotEncoder(handle_unknown="ignore") pour robustesse aux nouvelles modalités.
- Déséquilibre de classes:
  * Comparer class_weight (None, "balanced") avec RandomOverSampler (ROS), sans les cumuler.
    Raison: éviter d'amplifier le rééquilibrage et d'induire un sur-apprentissage.
- Validation / métriques:
  * Split stratifié initial et CV stratifiée (5-fold).
  * Métrique principale: F1 macro (chaque classe compte autant, adapté au déséquilibre multi-classes).
  * Complément: balanced accuracy (moyenne des recalls par classe). Pour mémoire: micro-F1≈accuracy en multi-classes et est dominé par les classes majoritaires; weighted-F1 pèse par le support.
- Ressources:
  * Parallélisation contrôlée par variables d'env: N_JOBS_ESTIMATOR (threads ExtraTrees), N_JOBS_OUTER (process Grid/CV) pour ne pas surcharger la machine.
  * SHAP désactivable via ENABLE_SHAP.
- Traçabilité:
  * Sauvegarde baseline CV, résultats complets de GridSearch (CSV), meilleurs paramètres (JSON), évaluation test (rapport, matrice, prédictions), importances de variables.
  * Journal JSONL de TOUS les essais de la grille (paramètres + scores) pour audit/reproductibilité.

Exécution (exemples):
    N_JOBS_ESTIMATOR=2 N_JOBS_OUTER=1 ENABLE_SHAP=0 python -m models.extratrees_plantvillage_v2
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE

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
# CSV unique par défaut (override possible via CSV_PATH)
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
CSV_PATH_ENV = os.environ.get("CSV_PATH")
CSV_PATH: Path = Path(CSV_PATH_ENV) if CSV_PATH_ENV else DEFAULT_CSV_PATH
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "extratrees_plantvillage_v2"  # dossier séparé pour v2
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Contrôle ressources (par défaut conservateur)
CPU_COUNT = os.cpu_count() or 2
N_JOBS_ESTIMATOR = int(os.environ.get("N_JOBS_ESTIMATOR", max(1, CPU_COUNT // 2)))  # threads internes ExtraTrees
N_JOBS_OUTER = int(os.environ.get("N_JOBS_OUTER", 1))  # parallélisme externe Grid/CV
ENABLE_SHAP = os.environ.get("ENABLE_SHAP", "0") == "1"

# Configuration de tuning (pilotée par ENV)
TUNING_MODE = os.environ.get("TUNING_MODE", "lite").lower()  # 'lite', 'lite_plus', 'full'
TUNING_SEARCH = os.environ.get("TUNING_SEARCH", "grid").lower()  # 'grid' ou 'random'
PIPELINE_LITE = os.environ.get("PIPELINE_LITE", "1") == "1"  # si True, on restreint fortement samplers/selectors/winsor
N_ITER_RANDOM = int(os.environ.get("N_ITER_RANDOM", 120))
LITE_STRICT = os.environ.get("LITE_STRICT", "1") == "1"  # si True, réduit fortement la grille ExtraTrees (lite)

# Sous-échantillonnage des données pour le tuning (stabilité mémoire). 1.0 = pas de sous-échantillonnage.
try:
    TUNING_SUBSAMPLE_FRAC = float(os.environ.get("TUNING_SUBSAMPLE_FRAC", "1.0"))
except Exception:
    TUNING_SUBSAMPLE_FRAC = 1.0

# Sous-échantillonnage pour le refit final (évaluation test). 1.0 = pas de sous-échantillonnage.
try:
    REFIT_SUBSAMPLE_FRAC = float(os.environ.get("REFIT_SUBSAMPLE_FRAC", "1.0"))
except Exception:
    REFIT_SUBSAMPLE_FRAC = 1.0

# Override optionnel du nombre d'arbres pour le refit final
try:
    _refit_ne = int(os.environ.get("REFIT_N_ESTIMATORS", "0"))
    REFIT_N_ESTIMATORS: Optional[int] = _refit_ne if _refit_ne > 0 else None
except Exception:
    REFIT_N_ESTIMATORS = None

# Nombre de splits pour la CV (tuning et baseline)
try:
    CV_SPLITS = int(os.environ.get("CV_SPLITS", "5"))
except Exception:
    CV_SPLITS = 5

# Sécurité mémoire/CPU: sauter baseline/plots/permutation par défaut
SKIP_BASELINE = os.environ.get("SKIP_BASELINE", "1") == "1"
SKIP_PLOTS = os.environ.get("SKIP_PLOTS", "1") == "1"
SKIP_PERM_IMPORTANCE = os.environ.get("SKIP_PERM_IMPORTANCE", "1") == "1"

# Colonnes non-features (à exclure des entrées modèle)
NON_FEATURE_COLS: List[str] = [
    "species","filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_black","is_na","hash","is_duplicate_after_first","disease"
]


def print_plan() -> None:
    print("\nPlan d'exécution (ExtraTrees v2):")
    steps = [
        "1) Charger CSV propre + traiter NA cible/drapeaux is_na & is_duplicate_after_first",
        "2) Détecter colonnes numériques/catégorielles (exclure NON_FEATURE_COLS)",
        "3) Pipeline: preprocess(num: impute[+winsor optionnel]+[scaler cond. si SMOTE], cat: impute+OHE) -> sampler -> selector -> ExtraTrees",
        "4) CV 5-fold stratifiée (balanced_accuracy, f1_macro, f1_micro, f1_weighted)",
        "5) GridSearch: n_estimators, max_features, min_samples_*, class_weight vs ROS/SMOTE/BorderlineSMOTE, winsor ablation, scaler conditionnel pour SMOTE",
        "6) Évaluation test + rapport + matrice de confusion + prédictions",
        "7) Interprétabilité: feature_importances_ (+ SHAP optionnel)",
        "8) Registre JSONL: dump de TOUS les essais de grid (params + scores)",
        "9) Importance par permutation (scoring=f1_macro)",
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
        # Mapping automatique courant: 'nom_plante' -> 'species'
        if "nom_plante" in df.columns:
            df = df.rename(columns={"nom_plante": TARGET_COL})
            print("[INFO] Colonne cible renommée: 'nom_plante' -> 'species'")
        else:
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
    # Numériques: imputer médiane (winsor optionnel, désactivé par défaut)
    num_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", "passthrough"),  # ablation possible via grille si souhaité
        ("scaler", "passthrough"),  # RobustScaler activé conditionnellement si SMOTE/BorderlineSMOTE
    ])
    # Catégorielles: imputer mode + OHE robuste aux catégories inconnues
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
        n_jobs=None,
    )
    return pre


def build_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int = RANDOM_STATE) -> ImbPipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        class_weight=None,  # comparé à "balanced" en Grid
        random_state=random_state,
        n_jobs=N_JOBS_ESTIMATOR,  # parallélisation interne
    )
    # Ordre: preprocess -> sampler -> selector -> model (sampler agit sur les features déjà prétraitées)
    pipe = ImbPipeline(steps=[
        ("preprocess", pre),
        ("sampler", "passthrough"),  # configuré par Grid (passthrough, ROS)
        ("selector", "passthrough"),  # SelectKBest / SelectFromModel / passthrough
        ("model", model),
    ])
    return pipe


def cross_validate_baseline(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> dict:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # Métriques: F1 macro (refit plus tard) et balanced accuracy en complément
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
    }
    print("\nCV 5-fold (baseline, sans tuning)...")
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=N_JOBS_OUTER)
    pd.DataFrame(cv).to_csv(RESULTS_DIR / "cv_baseline_scores.csv", index=False)
    print({k: float(np.mean(v)) for k, v in cv.items() if k.startswith("test_")})
    return cv


def _param_grid_common() -> Dict[str, List[Any]]:
    # Grille v2-lite (rapide; winsorisation reste optionnelle ailleurs, désactivée par défaut)
    return {
        "model__n_estimators": [400, 800],
        "model__max_depth": [None, 40],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.8],
        "model__bootstrap": [False],
        "model__criterion": ["gini"],
        "model__ccp_alpha": [0.0],
    }

def _get_extratrees_param_grid(mode: str) -> Dict[str, List[Any]]:
    """Retourne la grille d'hyperparamètres ExtraTrees selon le mode.
    - lite: grille compacte (par défaut)
    - lite_plus: légère extension (ex: n_estimators plus large)
    - full: par défaut fallback sur lite si non défini
    """
    m = (mode or "lite").lower()
    if m == "lite":
        if LITE_STRICT:
            # Ultralite: 1 seule combinaison ExtraTrees pour limiter drastiquement le coût
            return {
                "model__n_estimators": [400],
                "model__max_depth": [40],
                "model__min_samples_split": [2],
                "model__min_samples_leaf": [1],
                "model__max_features": ["sqrt"],
                "model__bootstrap": [False],
                "model__criterion": ["gini"],
                "model__ccp_alpha": [0.0],
            }
        return _param_grid_common()
    if m in ("lite_plus", "plus"):
        return {
            "model__n_estimators": [400, 800, 1200],
            "model__max_depth": [None, 40],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.8],
            "model__bootstrap": [False],
            "model__criterion": ["gini"],
            "model__ccp_alpha": [0.0],
        }
    # Fallback (full non encore étendu): utiliser la lite par défaut
    return _param_grid_common()

def _build_param_grid(random_state: int, mode: str, pipeline_lite: bool) -> List[Dict[str, Any]]:
    """Construit la grille de paramètres complète (pipeline + ExtraTrees) selon le mode et la sévérité pipeline_lite.
    pipeline_lite=True réduit drastiquement les combinaisons pour stabilité mémoire/CPU.
    """
    grid_common = _get_extratrees_param_grid(mode)

    if pipeline_lite:
        # Version très compacte: pas de winsor ablation, pas de sélecteurs, pas de SMOTE/BorderlineSMOTE
        return [
            {
                "sampler": ["passthrough"],
                "selector": ["passthrough"],
                "preprocess__num__winsor": ["passthrough"],
                "preprocess__num__scaler": ["passthrough"],
                "model__class_weight": [None, "balanced"],
                **grid_common,
            },
            {
                "sampler": [RandomOverSampler(random_state=random_state)],
                "selector": ["passthrough"],
                "preprocess__num__winsor": ["passthrough"],
                "preprocess__num__scaler": ["passthrough"],
                "model__class_weight": [None],
                **grid_common,
            },
        ]

    # Mode complet: on reprend les 8 blocs initiaux
    selector_kbest = SelectKBest(score_func=mutual_info_classif)
    selector_sfm = SelectFromModel(
        estimator=ExtraTreesClassifier(
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=N_JOBS_ESTIMATOR,
        ),
        threshold="median",
    )

    return [
        {
            "sampler": ["passthrough"],
            "selector": ["passthrough"],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None, "balanced"],
            **grid_common,
        },
        {
            # Over-sampling simple (ROS)
            "sampler": [RandomOverSampler(random_state=random_state)],
            "selector": ["passthrough"],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # SMOTE + scaler robuste sur numériques
            "sampler": [SMOTE(random_state=random_state)],
            "selector": ["passthrough"],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": [RobustScaler()],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # BorderlineSMOTE + scaler robuste sur numériques
            "sampler": [BorderlineSMOTE(random_state=random_state)],
            "selector": ["passthrough"],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": [RobustScaler()],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # Sélection univariée mutual information
            "sampler": ["passthrough"],
            "selector": [selector_kbest],
            "selector__k": [30],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # Sélection univariée + ROS
            "sampler": [RandomOverSampler(random_state=random_state)],
            "selector": [selector_kbest],
            "selector__k": [30],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # SelectFromModel (ExtraTrees) seuil médian
            "sampler": ["passthrough"],
            "selector": [selector_sfm],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None],
            **grid_common,
        },
        {
            # SelectFromModel + ROS
            "sampler": [RandomOverSampler(random_state=random_state)],
            "selector": [selector_sfm],
            "preprocess__num__winsor": ["passthrough", Winsorizer(lower=0.001, upper=0.999)],
            "preprocess__num__scaler": ["passthrough"],
            "model__class_weight": [None],
            **grid_common,
        },
    ]


def tune_hyperparams(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> GridSearchCV:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
    }

    # Construire la grille selon ENV
    param_grid = _build_param_grid(random_state=random_state, mode=TUNING_MODE, pipeline_lite=PIPELINE_LITE)

    # Compter les combinaisons (diagnostic)
    if isinstance(param_grid, list):
        n_candidates = sum(len(ParameterGrid(g)) for g in param_grid)
    else:
        n_candidates = len(ParameterGrid(param_grid))

    # Sous-échantillonnage éventuel pour tuning
    X_tune, y_tune = X, y
    if 0.0 < TUNING_SUBSAMPLE_FRAC < 1.0:
        X_tune, _, y_tune, _ = train_test_split(
            X, y, train_size=TUNING_SUBSAMPLE_FRAC, random_state=random_state, stratify=y
        )
        print(f"[INFO] Sous-échantillonnage tuning: frac={TUNING_SUBSAMPLE_FRAC} -> {len(X_tune)} échantillons")

    print("\nTuning (ExtraTrees hyperparams + équilibrage) ...")
    print(f"[CFG] PIPELINE_LITE={PIPELINE_LITE} | LITE_STRICT={LITE_STRICT} | TUNING_MODE={TUNING_MODE} | TUNING_SEARCH={TUNING_SEARCH} | candidates={n_candidates}")

    if TUNING_SEARCH == "random":
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,  # listes autorisées
            n_iter=N_ITER_RANDOM,
            random_state=random_state,
            cv=skf,
            scoring=scoring,
            refit="f1_macro",
            n_jobs=N_JOBS_OUTER,
            verbose=1,
            return_train_score=True,
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=skf,
            scoring=scoring,
            refit="f1_macro",
            n_jobs=N_JOBS_OUTER,
            verbose=1,
            return_train_score=True,
        )

    search.fit(X_tune, y_tune)
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(RESULTS_DIR / "gridsearch_results.csv", index=False)

    # Best params summary
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "best_score_f1_macro": float(search.best_score_),
        }, f, indent=2)
    print(f"Best params: {search.best_params_} | best f1_macro: {search.best_score_:.4f}")

    # Dump JSONL registry of all trials
    grid_samplers = ["passthrough", "ROS"] if PIPELINE_LITE else ["passthrough", "ROS", "SMOTE", "BorderlineSMOTE"]
    selector_opts = ["passthrough"] if PIPELINE_LITE else ["passthrough", "SelectKBest", "SelectFromModel"]
    winsor_ablation = False if PIPELINE_LITE else True
    robust_scaler_conditional = False if PIPELINE_LITE else True
    dump_trials_jsonl(cv_df, RESULTS_DIR / "grid_trials.jsonl", meta={
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "csv_path": str(CSV_PATH),
        "target_col": TARGET_COL,
        "random_state": RANDOM_STATE,
        "cv_splits": cv_splits,
        "n_jobs_estimator": N_JOBS_ESTIMATOR,
        "n_jobs_outer": N_JOBS_OUTER,
        # Config tuning
        "tuning_mode": TUNING_MODE,
        "tuning_search": TUNING_SEARCH,
        "n_iter_random": N_ITER_RANDOM if TUNING_SEARCH == "random" else None,
        "pipeline_lite": PIPELINE_LITE,
        "n_param_settings": int(cv_df.shape[0]),
        # Métadonnées explicites de la grille
        "grid_samplers": grid_samplers,
        "winsor_ablation": winsor_ablation,
        "robust_scaler_conditional": robust_scaler_conditional,
        "selector_options": selector_opts,
        "refit_metric": "f1_macro",
        "tuning_subsample_frac": TUNING_SUBSAMPLE_FRAC,
    })

    return search


def _label_approach_from_params(row: pd.Series) -> str:
    sampler_val = row.get("param_sampler", None)
    cw_val = row.get("param_model__class_weight", None)
    s = str(sampler_val)
    if "RandomUnderSampler" in s:
        return "Undersampling: RUS"
    if "RandomOverSampler" in s:
        return "Oversampling: ROS"
    if "BorderlineSMOTE" in s:
        return "Oversampling: Borderline-SMOTE"
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
        if "BorderlineSMOTE" in s: return "BorderlineSMOTE"
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

    # Scatter f1_macro vs n_estimators avec lignes 200→400 et annotation du meilleur point
    try:
        # Créer un identifiant de combinaison pour relier 200→400
        combo_cols = ["approach","max_features","min_samples_split","min_samples_leaf","max_depth","sampler_name"]
        df["combo_id"] = df[combo_cols].astype(str).agg("|".join, axis=1)
        df_sorted = df.sort_values(by=["combo_id", "n_estimators"])  # assurer l'ordre des segments

        # Tracer lignes + marqueurs
        fig1 = px.line(
            df_sorted, x="n_estimators", y="mean_test_f1_macro",
            color="approach", line_group="combo_id", symbol="max_features",
            hover_data=["max_depth","min_samples_split","min_samples_leaf"],
            title="ExtraTrees: Macro F1 vs n_estimators",
            markers=True,
        )

        # Annoter les meilleurs points (un par approche)
        if not best_per_approach.empty:
            fig1.add_scatter(
                x=best_per_approach["n_estimators"],
                y=best_per_approach["mean_test_f1_macro"],
                mode="markers+text",
                text=[f"Best {a}" for a in best_per_approach["approach"]],
                textposition="top center",
                marker=dict(symbol="star", size=14, color="black", line=dict(width=2, color="white")),
                showlegend=False,
                hovertext=best_per_approach.apply(lambda r: f"{r['approach']} | n_estimators={int(r['n_estimators'])}", axis=1),
                hoverinfo="text",
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


def evaluate_on_test(best_estimator: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series, eval_dir: Path) -> dict:
    print("\nÉvaluation finale sur test...")
    y_pred = best_estimator.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=best_estimator.named_steps["model"].classes_)

    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / "classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(eval_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # Sauvegarde des prédictions pour analyse d'erreurs
    pred_df = pd.DataFrame({
        "y_true": y_test.astype(str).values,
        "y_pred": y_pred.astype(str),
    }, index=y_test.index)
    pred_df.to_csv(eval_dir / "predictions_test.csv")

    # Sauvegarde métriques test en JSON pour traçabilité
    with open(eval_dir / "test_metrics.json", "w") as f:
        json.dump({
            "balanced_accuracy": float(bal_acc),
            "f1_macro": float(f1m),
            "f1_micro": float(f1_micro),
            "f1_weighted": float(f1_weighted),
            "accuracy": float(acc),
        }, f, indent=2)

    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    print(f"f1_micro (test): {f1_micro:.4f}")
    print(f"f1_weighted (test): {f1_weighted:.4f}")
    print(f"accuracy (test): {acc:.4f}")
    return {
        "balanced_accuracy": bal_acc,
        "f1_macro": f1m,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }

    
def permutation_importance_analysis(
    best_estimator: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names_selected: List[str],
    eval_dir: Path,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Calcule et enregistre l'importance par permutation sur les features après preprocess/selector."""
    preprocess = best_estimator.named_steps["preprocess"]
    X_tr = preprocess.transform(X_test)
    selector = best_estimator.named_steps.get("selector", None)
    if selector is not None and hasattr(selector, "transform"):
        try:
            X_tr = selector.transform(X_tr)
        except Exception:
            pass
    model: ExtraTreesClassifier = best_estimator.named_steps["model"]
    r = permutation_importance(
        model, X_tr, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_OUTER,
    )
    pi_df = pd.DataFrame({
        "feature": list(feature_names_selected),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    eval_dir.mkdir(exist_ok=True, parents=True)
    pi_df.to_csv(eval_dir / "permutation_importances.csv", index=False)
    return pi_df


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


def shap_analysis(best_estimator: ImbPipeline, X_train: pd.DataFrame, feature_names_transformed: List[str], sample_size: int = 400) -> None:
    if not (HAS_SHAP and ENABLE_SHAP):
        print("[INFO] SHAP non activé ou indisponible - ignoré.")
        return
    try:
        preprocess = best_estimator.named_steps["preprocess"]
        model: ExtraTreesClassifier = best_estimator.named_steps["model"]
        Xtr = preprocess.transform(X_train)
        # Appliquer aussi le sélecteur s'il existe
        selector = best_estimator.named_steps.get("selector", None)
        if selector is not None and hasattr(selector, "transform"):
            try:
                Xtr = selector.transform(Xtr)
            except Exception:
                pass
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


def save_class_distribution(y_train: pd.Series, y_test: pd.Series, out_dir: Path) -> None:
    """Sauvegarde les distributions de classes pour train/test et un résumé d'imbalance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_counts = y_train.value_counts().sort_values(ascending=False)
    test_counts = y_test.value_counts().sort_values(ascending=False)
    train_counts.to_csv(out_dir / "class_distribution_train.csv", header=["count"])
    test_counts.to_csv(out_dir / "class_distribution_test.csv", header=["count"])

    def _stats(s: pd.Series) -> Dict[str, Any]:
        mn = int(s.min())
        mx = int(s.max())
        ratio = float(mn) / float(mx) if mx > 0 else None
        return {"min": mn, "max": mx, "ratio_min_max": ratio, "n_classes": int(s.shape[0])}

    with open(out_dir / "imbalance_summary.json", "w") as f:
        json.dump({"train": _stats(train_counts), "test": _stats(test_counts)}, f, indent=2)
    print("Distributions de classes sauvegardées dans:", out_dir)


# -----------------------------
# Registre JSONL pour tous les essais de grid
# -----------------------------

def dump_trials_jsonl(cv_results_df: pd.DataFrame, jsonl_path: Path, meta: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    records = cv_results_df.to_dict(orient="records")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in records:
            payload = {"meta": meta, "trial": r}
            f.write(json.dumps(payload, default=_json_default) + "\n")
    print(f"Trials JSONL append: {jsonl_path} ({len(records)} entrées)")


def _json_default(o: Any):
    try:
        return str(o)
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()
    t0 = time.time()

    # Charger données
    print(f"CSV utilisé: {CSV_PATH}")
    df, X, y, num_cols, cat_cols = load_clean_dataset(CSV_PATH)

    # Split stratifié
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Résumé du déséquilibre (train/test)
    save_class_distribution(y_train, y_test, RESULTS_DIR)

    # Pipeline
    pipe = build_pipeline(num_cols, cat_cols, RANDOM_STATE)

    # Baseline CV (pour référence sans tuning) — sauté par défaut pour stabilité mémoire
    if SKIP_BASELINE:
        print("[INFO] SKIP_BASELINE=1 -> CV baseline sauté.")
    else:
        _ = cross_validate_baseline(pipe, X_train, y_train, cv_splits=CV_SPLITS, random_state=RANDOM_STATE)

    # Tuning (GridSearchCV avec refit macro-F1)
    gs = tune_hyperparams(pipe, X_train, y_train, cv_splits=CV_SPLITS, random_state=RANDOM_STATE)

    # Synthèse plots
    if SKIP_PLOTS:
        print("[INFO] SKIP_PLOTS=1 -> synthèse plots sautée.")
    else:
        try:
            cv_df = pd.DataFrame(gs.cv_results_)
            summarize_and_plot(cv_df)
        except Exception as e:
            print(f"[WARN] summarize_and_plot a échoué: {e}")

    best_pipe: ImbPipeline = gs.best_estimator_

    # Optionnel: override n_estimators pour le refit final
    if REFIT_N_ESTIMATORS is not None:
        try:
            best_pipe.set_params(model__n_estimators=REFIT_N_ESTIMATORS)
            print(f"[INFO] REFIT_N_ESTIMATORS={REFIT_N_ESTIMATORS} -> override du nombre d'arbres pour le refit")
        except Exception as e:
            print(f"[WARN] Impossible d'appliquer REFIT_N_ESTIMATORS: {e}")

    # Optionnel: sous-échantillonnage pour le refit final (stabilité mémoire)
    X_refit, y_refit = X_train, y_train
    if REFIT_SUBSAMPLE_FRAC and REFIT_SUBSAMPLE_FRAC < 1.0:
        rs = np.random.RandomState(RANDOM_STATE)
        n_refit = max(1, int(len(X_train) * float(REFIT_SUBSAMPLE_FRAC)))
        try:
            idx = rs.choice(X_train.index, size=n_refit, replace=False)
            X_refit = X_train.loc[idx]
            y_refit = y_train.loc[idx]
        except Exception:
            # fallback si les index ne sont pas des Index Pandas standards
            idx = rs.choice(np.arange(len(X_train)), size=n_refit, replace=False)
            X_refit = X_train.iloc[idx]
            y_refit = y_train.iloc[idx]
        print(f"[INFO] Sous-échantillonnage refit: frac={REFIT_SUBSAMPLE_FRAC} -> {len(idx)} échantillons")

    # Fit final
    best_pipe.fit(X_refit, y_refit)

    # Évaluation test
    eval_dir = RESULTS_DIR / "evaluation"
    _ = evaluate_on_test(best_pipe, X_test, y_test, eval_dir)

    # Noms de features transformées (pour importances/SHAP)
    try:
        feature_names_transformed = get_transformed_feature_names(
            best_pipe.named_steps["preprocess"], num_cols, cat_cols
        )
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(best_pipe.named_steps["model"].n_features_in_)]

    # Si un sélecteur est utilisé, appliquer le masque pour garder les bons noms
    try:
        sel = best_pipe.named_steps.get("selector", None)
        if sel is not None and hasattr(sel, "get_support"):
            support = sel.get_support()
            if support is not None and len(support) == len(feature_names_transformed):
                feature_names_transformed = [n for n, keep in zip(feature_names_transformed, support) if keep]
    except Exception:
        pass

    # Importances
    _ = analyze_feature_importances(best_pipe, feature_names_transformed, top_k=40)

    # SHAP (optionnel)
    shap_analysis(best_pipe, X_train, feature_names_transformed, sample_size=400)

    # Permutation importance (sur test, scoring macro-F1)
    if SKIP_PERM_IMPORTANCE:
        print("[INFO] SKIP_PERM_IMPORTANCE=1 -> importance par permutation sautée.")
    else:
        _ = permutation_importance_analysis(best_pipe, X_test, y_test, feature_names_transformed, eval_dir, scoring="f1_macro", n_repeats=10)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
