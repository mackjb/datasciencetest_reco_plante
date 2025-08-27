#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparaison Baseline (SVM-RBF sur toutes les features) vs LDA -> SVM-RBF vs LDA(shrinkage) -> SVM-RBF vs PCA -> SVM-RBF sur PlantVillage.

- Données: CSV clean PlantVillage
- Nettoyage: même logique que les autres scripts (NA, doublons, indices problématiques)
- CV: StratifiedKFold(5), scoring={'bal_acc','f1_macro'}, refit='f1_macro'
- Grille SVM: C ∈ [75, 80, 85], class_weight ∈ {None, 'balanced'}, gamma='scale'
- LDA: n_components ∈ [None, 5, 10, 13] (borné par n_classes-1)
- LDA(shrinkage): solver='eigen', shrinkage='auto', n_components idem (transform supporté)
- PCA: n_components ∈ {10, 15, 20, 25, 30}
- Sorties: récapitulatif CV (moyenne±écart-type), test final, et graphes comparatifs
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.io as pio

try:
    from src.helpers.helpers import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "svm_rbf_lda_compare"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

IMAGE_PREFIX_OLD: str = r"C:\\Users\\bgasm\\Documents\\Bernadette\\IA\\projet\\notebook_rendu_1\\version_7_new_features\\plantvillage dataset\\segmented"
IMAGE_PREFIX_NEW: str = r"C:\\Users\\bgasm\\repository\\datasciencetest_reco_plante\\dataset\\plantvillage\\data\\plantvillage dataset\\segmented"

# -----------------------------
# Helpers
# -----------------------------

def to_file_uri_from_any(val: str) -> Optional[str]:
    if not isinstance(val, str) or not val:
        return None
    s = val
    if s.startswith("file://"):
        return s
    if IMAGE_PREFIX_NEW and IMAGE_PREFIX_OLD:
        s_norm = re.sub(r"[\\]+", "/", s).strip()
        old_norm = re.sub(r"[\\]+", "/", IMAGE_PREFIX_OLD).strip()
        if s_norm.lower().startswith(old_norm.lower()):
            rest = s_norm[len(old_norm):].lstrip("/")
            new_win = str(PureWindowsPath(IMAGE_PREFIX_NEW) / rest.replace("/", "\\"))
            try:
                return Path(new_win).as_uri()
            except Exception:
                new_posix = new_win.replace("\\", "/")
                return "file:///" + new_posix
    if re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith("\\\\"):
        s_posix = s.replace("\\", "/")
        return "file:///" + s_posix
    cand = s if os.path.isabs(s) else str((PROJECT_ROOT / s).resolve())
    try:
        return Path(cand).as_uri()
    except Exception:
        return "file:///" + cand.replace("\\", "/") if cand else None

NON_FEATURE_COLS: List[str] = [
    "species","ID_Image","Est_Saine","Image_Path","is_black",
    "plant_Apple","plant_Blueberry","plant_Cherry_(including_sour)","plant_Corn_(maize)","plant_Grape","plant_Orange","plant_Peach","plant_Pepper,_bell","plant_Potato","plant_Raspberry","plant_Soybean","plant_Squash","plant_Strawberry","plant_Tomato",
    "disease_Apple_scab","disease_Bacterial_spot","disease_Black_rot","disease_Cedar_apple_rust","disease_Cercospora_leaf_spot Gray_leaf_spot","disease_Common_rust_","disease_Early_blight","disease_Esca_(Black_Measles)","disease_Haunglongbing_(Citrus_greening)","disease_Late_blight","disease_Leaf_Mold","disease_Leaf_blight_(Isariopsis_Leaf_Spot)","disease_Leaf_scorch","disease_Northern_Leaf_Blight","disease_Powdery_mildew","disease_Septoria_leaf_spot","disease_Spider_mites Two-spotted_spider_mite","disease_Target_Spot","disease_Tomato_Yellow_Leaf_Curl_Virus","disease_Tomato_mosaic_virus","disease_healthy",
    "filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_na","hash","is_duplicate_after_first","disease",
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in NON_FEATURE_COLS and c != TARGET_COL]
    return features


def load_clean_dataset(csv_path: Path = CSV_PATH, drop_duplicates_flag: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    if "nom_plante" in df.columns and "species" not in df.columns:
        df = df.rename(columns={"nom_plante": "species"})
    if "species" not in df.columns:
        plant_cols = [c for c in df.columns if c.startswith("plant_")]
        if plant_cols:
            plant_mat = df[plant_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            winner_idx = plant_mat.argmax(axis=1)
            labels = []
            for j in winner_idx:
                col = plant_cols[int(j)] if len(plant_cols) else None
                if not col:
                    labels.append(None); continue
                name = col[len("plant_"):]
                name = re.split(r"[_,(]", name)[0]
                labels.append(name)
            df["species"] = labels
    if "filepath" not in df.columns and "Image_Path" in df.columns:
        df["filepath"] = df["Image_Path"]

    n0 = len(df)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET_COL}' absente du CSV.")
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

    if "ID_Image" in df.columns:
        df = df.drop(columns=["ID_Image"])  # éviter fuite

    feature_names = get_feature_columns(df)
    if not feature_names:
        raise RuntimeError("Aucune colonne de feature détectée.")

    before = len(df)
    df = df.dropna(subset=feature_names)
    removed_feat_na = before - len(df)
    if removed_feat_na > 0:
        print(f"Lignes supprimées pour NA dans features: {removed_feat_na}")

    bad_idx = [3004, 13775, 20539, 24002, 23742, 23923, 35368, 35192, 35901, 35507]
    if bad_idx:
        before = len(df)
        present = [i for i in bad_idx if i in df.index]
        if present:
            df = df.drop(index=present)
            removed = before - len(df)
            if removed > 0:
                print(f"Lignes exclues via liste idx (nettoyage manuel): {removed}")

    X = df[feature_names].copy()
    y = df[TARGET_COL].astype(str).copy()
    print(f"Lignes chargées: {n0} -> après nettoyage: {len(df)}")
    print(f"#features: {len(feature_names)} | #classes: {y.nunique()}")
    return df, X, y, feature_names


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# -----------------------------
# Pipelines
# -----------------------------

def build_baseline_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=random_state)),
    ])


def build_lda_shrinkage_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    # Utilise solver='eigen' pour supporter shrinkage='auto' et la transformation (transform)
    return Pipeline([
        ("scaler", RobustScaler()),
        ("lda", LDA(solver="eigen", shrinkage="auto")),
        ("svm", SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=random_state)),
    ])


def build_lda_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("lda", LDA(solver="svd")),  # n_components balayé par grille
        ("svm", SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=random_state)),
    ])


def build_pca_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(svd_solver="auto", random_state=random_state)),
        ("svm", SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=random_state)),
    ])


# -----------------------------
# Entraînement & évaluation
# -----------------------------

def run_grid_cv(pipe: Pipeline, X, y, param_grid: Dict, results_path: Path) -> Tuple[GridSearchCV, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring={"bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(results_path, index=False)
    return gs, cv_df


def cv_summary_str(cv_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    # moyenne ± std des métriques de test par combinaison (groupby keys)
    g = cv_df.groupby(group_keys, dropna=False).agg(
        mean_f1=("mean_test_f1_macro", "mean"), std_f1=("std_test_f1_macro", "mean"),
        mean_bal=("mean_test_bal_acc", "mean"), std_bal=("std_test_bal_acc", "mean"),
    ).reset_index()
    g["f1_macro"] = g.apply(lambda r: f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f}", axis=1)
    g["balanced_accuracy"] = g.apply(lambda r: f"{r['mean_bal']:.4f} ± {r['std_bal']:.4f}", axis=1)
    return g


def evaluate_test(model: Pipeline, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("\nPlan d'évaluation (Baseline vs LDA→SVM-RBF vs LDA(shrinkage)→SVM-RBF vs PCA→SVM-RBF):")
    for s in [
        "1) Charger CSV + nettoyer",
        "2) Split stratifié train/test",
        "3) Baseline: RobustScaler -> SVC(RBF): CV + test",
        "4) LDA: RobustScaler -> LDA -> SVC(RBF): CV + test",
        "5) LDA(shrinkage): RobustScaler -> LDA(eigen, shrinkage=auto) -> SVC(RBF): CV + test",
        "6) PCA: RobustScaler -> PCA -> SVC(RBF): CV + test",
        "7) Sauvegarde des résumés et graphes",
    ]:
        print(" - ", s)

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # ---------------- Baseline ----------------
    print("\n[Baseline] CV en cours...")
    base_pipe = build_baseline_pipeline(RANDOM_STATE)
    Cs = np.array([75, 80, 85], dtype=float)
    base_grid = {
        "svm__C": Cs,
        "svm__class_weight": [None, "balanced"],
    }
    base_gs, base_cv = run_grid_cv(base_pipe, X_train, y_train, base_grid, RESULTS_DIR / "baseline_grid.csv")
    base_best = base_gs.best_estimator_
    base_test = evaluate_test(base_best, X_test, y_test)
    base_summary = cv_summary_str(base_cv, ["param_svm__C", "param_svm__class_weight"]).sort_values("param_svm__C")

    with open(RESULTS_DIR / "baseline_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": base_gs.best_params_,
            "best_cv_f1_macro": float(base_gs.best_score_),
            "cv_rows": len(base_cv),
            "summary_preview": base_summary.head(10).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "baseline_test.json", "w", encoding="utf-8") as f:
        json.dump(base_test, f, ensure_ascii=False, indent=2)

    print(f"Baseline CV -> f1_macro: {base_gs.cv_results_['mean_test_f1_macro'][base_gs.best_index_]:.4f} ± {base_gs.cv_results_['std_test_f1_macro'][base_gs.best_index_]:.4f} | bal_acc: {base_gs.cv_results_['mean_test_bal_acc'][base_gs.best_index_]:.4f} ± {base_gs.cv_results_['std_test_bal_acc'][base_gs.best_index_]:.4f}")
    print(f"Baseline Test -> bal_acc: {base_test['balanced_accuracy']:.4f} | f1_macro: {base_test['f1_macro']:.4f}")

    # ---------------- LDA -> SVM-RBF ----------------
    print("\n[LDA→SVM-RBF] CV en cours...")
    lda_pipe = build_lda_pipeline(RANDOM_STATE)
    n_classes = y_train.nunique()
    max_comp = max(1, min(13, n_classes - 1))
    lda_grid = {
        "lda__n_components": [None, 5, 10, max_comp] if max_comp >= 10 else [None, max_comp],
        "svm__C": Cs,
        "svm__class_weight": [None, "balanced"],
    }
    lda_gs, lda_cv = run_grid_cv(lda_pipe, X_train, y_train, lda_grid, RESULTS_DIR / "lda_grid.csv")
    lda_best = lda_gs.best_estimator_
    lda_test = evaluate_test(lda_best, X_test, y_test)
    lda_summary = cv_summary_str(lda_cv, ["param_lda__n_components", "param_svm__C", "param_svm__class_weight"]).sort_values(["param_lda__n_components", "param_svm__C"]) 

    with open(RESULTS_DIR / "lda_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in lda_gs.best_params_.items()},
            "best_cv_f1_macro": float(lda_gs.best_score_),
            "cv_rows": len(lda_cv),
            "summary_preview": lda_summary.head(15).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "lda_test.json", "w", encoding="utf-8") as f:
        json.dump(lda_test, f, ensure_ascii=False, indent=2)

    print(f"LDA→SVM CV -> f1_macro: {lda_gs.cv_results_['mean_test_f1_macro'][lda_gs.best_index_]:.4f} ± {lda_gs.cv_results_['std_test_f1_macro'][lda_gs.best_index_]:.4f} | bal_acc: {lda_gs.cv_results_['mean_test_bal_acc'][lda_gs.best_index_]:.4f} ± {lda_gs.cv_results_['std_test_bal_acc'][lda_gs.best_index_]:.4f}")
    print(f"LDA→SVM Test -> bal_acc: {lda_test['balanced_accuracy']:.4f} | f1_macro: {lda_test['f1_macro']:.4f}")

    # ---------------- LDA(shrinkage) -> SVM-RBF ----------------
    print("\n[LDA(shrinkage)→SVM-RBF] CV en cours...")
    lda_sh_pipe = build_lda_shrinkage_pipeline(RANDOM_STATE)
    n_classes = y_train.nunique()
    max_comp = max(1, min(13, n_classes - 1, X_train.shape[1]))
    lda_sh_grid = {
        "lda__n_components": [None, 5, 10, max_comp] if max_comp >= 10 else [None, max_comp],
        "svm__C": Cs,
        "svm__class_weight": [None, "balanced"],
    }
    lda_sh_gs, lda_sh_cv = run_grid_cv(lda_sh_pipe, X_train, y_train, lda_sh_grid, RESULTS_DIR / "lda_shrinkage_grid.csv")
    lda_sh_best = lda_sh_gs.best_estimator_
    lda_sh_test = evaluate_test(lda_sh_best, X_test, y_test)
    lda_sh_summary = cv_summary_str(lda_sh_cv, ["param_lda__n_components", "param_svm__C", "param_svm__class_weight"]).sort_values(["param_lda__n_components", "param_svm__C"]) 

    with open(RESULTS_DIR / "lda_shrinkage_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in lda_sh_gs.best_params_.items()},
            "best_cv_f1_macro": float(lda_sh_gs.best_score_),
            "cv_rows": len(lda_sh_cv),
            "summary_preview": lda_sh_summary.head(15).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "lda_shrinkage_test.json", "w", encoding="utf-8") as f:
        json.dump(lda_sh_test, f, ensure_ascii=False, indent=2)

    print(f"LDA(shrinkage)→SVM CV -> f1_macro: {lda_sh_gs.cv_results_['mean_test_f1_macro'][lda_sh_gs.best_index_]:.4f} ± {lda_sh_gs.cv_results_['std_test_f1_macro'][lda_sh_gs.best_index_]:.4f} | bal_acc: {lda_sh_gs.cv_results_['mean_test_bal_acc'][lda_sh_gs.best_index_]:.4f} ± {lda_sh_gs.cv_results_['std_test_bal_acc'][lda_sh_gs.best_index_]:.4f}")
    print(f"LDA(shrinkage)→SVM Test -> bal_acc: {lda_sh_test['balanced_accuracy']:.4f} | f1_macro: {lda_sh_test['f1_macro']:.4f}")

    # ---------------- PCA -> SVM-RBF ----------------
    print("\n[PCA→SVM-RBF] CV en cours...")
    pca_pipe = build_pca_pipeline(RANDOM_STATE)
    # Limiter n_components à < n_features
    n_features = X_train.shape[1]
    pca_components = [k for k in [10, 15, 20, 25, 30] if k < n_features]
    if not pca_components:
        pca_components = [min(10, n_features-1)]
    pca_grid = {
        "pca__n_components": pca_components,
        "svm__C": Cs,
        "svm__class_weight": [None, "balanced"],
    }
    pca_gs, pca_cv = run_grid_cv(pca_pipe, X_train, y_train, pca_grid, RESULTS_DIR / "pca_grid.csv")
    pca_best = pca_gs.best_estimator_
    pca_test = evaluate_test(pca_best, X_test, y_test)
    pca_summary = cv_summary_str(pca_cv, ["param_pca__n_components", "param_svm__C", "param_svm__class_weight"]).sort_values(["param_pca__n_components", "param_svm__C"])

    with open(RESULTS_DIR / "pca_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": pca_gs.best_params_,
            "best_cv_f1_macro": float(pca_gs.best_score_),
            "cv_rows": len(pca_cv),
            "summary_preview": pca_summary.head(15).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "pca_test.json", "w", encoding="utf-8") as f:
        json.dump(pca_test, f, ensure_ascii=False, indent=2)

    print(f"PCA→SVM CV -> f1_macro: {pca_gs.cv_results_['mean_test_f1_macro'][pca_gs.best_index_]:.4f} ± {pca_gs.cv_results_['std_test_f1_macro'][pca_gs.best_index_]:.4f} | bal_acc: {pca_gs.cv_results_['mean_test_bal_acc'][pca_gs.best_index_]:.4f} ± {pca_gs.cv_results_['std_test_bal_acc'][pca_gs.best_index_]:.4f}")
    print(f"PCA→SVM Test -> bal_acc: {pca_test['balanced_accuracy']:.4f} | f1_macro: {pca_test['f1_macro']:.4f}")

    # ---------------- Récapitulatif global ----------------
    recap = {
        "baseline": {
            "best_params": base_gs.best_params_,
            "cv": {
                "f1_macro_mean": float(base_gs.cv_results_["mean_test_f1_macro"][base_gs.best_index_]),
                "f1_macro_std": float(base_gs.cv_results_["std_test_f1_macro"][base_gs.best_index_]),
                "bal_acc_mean": float(base_gs.cv_results_["mean_test_bal_acc"][base_gs.best_index_]),
                "bal_acc_std": float(base_gs.cv_results_["std_test_bal_acc"][base_gs.best_index_]),
            },
            "test": base_test,
        },
        "lda_svm": {
            "best_params": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in lda_gs.best_params_.items()},
            "cv": {
                "f1_macro_mean": float(lda_gs.cv_results_["mean_test_f1_macro"][lda_gs.best_index_]),
                "f1_macro_std": float(lda_gs.cv_results_["std_test_f1_macro"][lda_gs.best_index_]),
                "bal_acc_mean": float(lda_gs.cv_results_["mean_test_bal_acc"][lda_gs.best_index_]),
                "bal_acc_std": float(lda_gs.cv_results_["std_test_bal_acc"][lda_gs.best_index_]),
            },
            "test": lda_test,
        },
        "lda_shrinkage_svm": {
            "best_params": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in lda_sh_gs.best_params_.items()},
            "cv": {
                "f1_macro_mean": float(lda_sh_gs.cv_results_["mean_test_f1_macro"][lda_sh_gs.best_index_]),
                "f1_macro_std": float(lda_sh_gs.cv_results_["std_test_f1_macro"][lda_sh_gs.best_index_]),
                "bal_acc_mean": float(lda_sh_gs.cv_results_["mean_test_bal_acc"][lda_sh_gs.best_index_]),
                "bal_acc_std": float(lda_sh_gs.cv_results_["std_test_bal_acc"][lda_sh_gs.best_index_]),
            },
            "test": lda_sh_test,
        },
        "pca_svm": {
            "best_params": pca_gs.best_params_,
            "cv": {
                "f1_macro_mean": float(pca_gs.cv_results_["mean_test_f1_macro"][pca_gs.best_index_]),
                "f1_macro_std": float(pca_gs.cv_results_["std_test_f1_macro"][pca_gs.best_index_]),
                "bal_acc_mean": float(pca_gs.cv_results_["mean_test_bal_acc"][pca_gs.best_index_]),
                "bal_acc_std": float(pca_gs.cv_results_["std_test_bal_acc"][pca_gs.best_index_]),
            },
            "test": pca_test,
        },
    }
    with open(RESULTS_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    # ---------------- Graphes comparatifs (barres mean±std) ----------------
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame([
        {"method": "Baseline", "f1_mean": recap["baseline"]["cv"]["f1_macro_mean"], "f1_std": recap["baseline"]["cv"]["f1_macro_std"], "bal_mean": recap["baseline"]["cv"]["bal_acc_mean"], "bal_std": recap["baseline"]["cv"]["bal_acc_std"]},
        {"method": "LDA→SVM", "f1_mean": recap["lda_svm"]["cv"]["f1_macro_mean"], "f1_std": recap["lda_svm"]["cv"]["f1_macro_std"], "bal_mean": recap["lda_svm"]["cv"]["bal_acc_mean"], "bal_std": recap["lda_svm"]["cv"]["bal_acc_std"]},
        {"method": "LDA(shrink)→SVM", "f1_mean": recap["lda_shrinkage_svm"]["cv"]["f1_macro_mean"], "f1_std": recap["lda_shrinkage_svm"]["cv"]["f1_macro_std"], "bal_mean": recap["lda_shrinkage_svm"]["cv"]["bal_acc_mean"], "bal_std": recap["lda_shrinkage_svm"]["cv"]["bal_acc_std"]},
        {"method": "PCA→SVM", "f1_mean": recap["pca_svm"]["cv"]["f1_macro_mean"], "f1_std": recap["pca_svm"]["cv"]["f1_macro_std"], "bal_mean": recap["pca_svm"]["cv"]["bal_acc_mean"], "bal_std": recap["pca_svm"]["cv"]["bal_acc_std"]},
    ])

    fig1 = px.bar(df_plot, x="method", y="f1_mean", error_y="f1_std", title="CV f1_macro (moyenne ± écart-type)", labels={"method": "Méthode", "f1_mean": "f1_macro"})
    pio.write_html(fig1, str(plots_dir / "cv_f1_macro_bar.html"), auto_open=False)

    fig2 = px.bar(df_plot, x="method", y="bal_mean", error_y="bal_std", title="CV balanced_accuracy (moyenne ± écart-type)", labels={"method": "Méthode", "bal_mean": "balanced_accuracy"})
    pio.write_html(fig2, str(plots_dir / "cv_balanced_accuracy_bar.html"), auto_open=False)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
