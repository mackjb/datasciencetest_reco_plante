#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVM RBF + Sélection de features (évaluation réduite) sur PlantVillage.

Objectifs:
1) Baseline (toutes features): CV 5-fold -> moy ± std (f1_macro, balanced_accuracy) + test final
2) Sélection de features en pipeline (SelectKBest) durant la CV -> courbe score vs nombre de features
3) Choisir le plus petit k gardant le score dans une marge tolérée (ΔF1 ≤ tol, par défaut 0.003)

Les artefacts sont enregistrés sous results/models/svm_rbf_reduced.
"""
from __future__ import annotations

import os
import json
import time
import re
from urllib.parse import quote
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_selection import SelectKBest, f_classif

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
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "svm_rbf_reduced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Réécriture optionnelle des chemins d'images (pour galeries interactives, si utilisées)
IMAGE_PREFIX_OLD: str = r"C:\\Users\\bgasm\\Documents\\Bernadette\\IA\\projet\\notebook_rendu_1\\version_7_new_features\\plantvillage dataset\\segmented"
IMAGE_PREFIX_NEW: str = r"C:\\Users\\bgasm\\repository\\datasciencetest_reco_plante\\dataset\\plantvillage\\data\\plantvillage dataset\\segmented"

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
                return "file:///" + quote(new_posix, safe="/:")
    if re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith("\\\\"):
        s_posix = s.replace("\\", "/")
        return "file:///" + quote(s_posix, safe="/:")
    cand = s if os.path.isabs(s) else str((PROJECT_ROOT / s).resolve())
    try:
        return Path(cand).as_uri()
    except Exception:
        return "file:///" + quote(cand.replace("\\", "/"), safe="/:") if cand else None

# Colonnes non-features dans le CSV clean
NON_FEATURE_COLS: List[str] = [
    "species","ID_Image","Est_Saine","Image_Path","is_black",
    "plant_Apple","plant_Blueberry","plant_Cherry_(including_sour)","plant_Corn_(maize)","plant_Grape","plant_Orange","plant_Peach","plant_Pepper,_bell","plant_Potato","plant_Raspberry","plant_Soybean","plant_Squash","plant_Strawberry","plant_Tomato",
    "disease_Apple_scab","disease_Bacterial_spot","disease_Black_rot","disease_Cedar_apple_rust","disease_Cercospora_leaf_spot Gray_leaf_spot","disease_Common_rust_","disease_Early_blight","disease_Esca_(Black_Measles)","disease_Haunglongbing_(Citrus_greening)","disease_Late_blight","disease_Leaf_Mold","disease_Leaf_blight_(Isariopsis_Leaf_Spot)","disease_Leaf_scorch","disease_Northern_Leaf_Blight","disease_Powdery_mildew","disease_Septoria_leaf_spot","disease_Spider_mites Two-spotted_spider_mite","disease_Target_Spot","disease_Tomato_Yellow_Leaf_Curl_Virus","disease_Tomato_mosaic_virus","disease_healthy",
    "filepath","filename","extension","file_size","label","width","height","mode","num_channels","aspect_ratio","is_image_valid","is_na","hash","is_duplicate_after_first","disease",
]


def print_plan() -> None:
    print("\nPlan d'évaluation (réduit):")
    steps = [
        "1) Charger CSV propre + nettoyer",
        "2) Split stratifié train/test",
        "3) Baseline (toutes features): CV 5-fold (bal_acc, f1_macro) + test",
        "4) Feature selection (SelectKBest) en pipeline: CV vs k + courbe",
        "5) Choix du plus petit k dans tolérance puis test final",
    ]
    for s in steps:
        print(" - ", s)


# -----------------------------
# Chargement & préparation
# -----------------------------

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
                    labels.append(None)
                    continue
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
        df = df.drop(columns=["ID_Image"])  # éviter toute fuite

    feature_names = get_feature_columns(df)
    if not feature_names:
        raise RuntimeError("Aucune colonne de feature détectée.")

    # Supprimer lignes avec NA dans features
    before = len(df)
    df = df.dropna(subset=feature_names)
    removed_feat_na = before - len(df)
    if removed_feat_na > 0:
        print(f"Lignes supprimées pour NA dans features: {removed_feat_na}")

    # Exclusion manuelle d'indices (problématiques)
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
# Modèle & évaluation
# -----------------------------

def build_baseline_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("selector", "passthrough"),
        ("svm", SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=random_state)),
    ])


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    return {"balanced_accuracy": float(bal_acc), "f1_macro": float(f1m)}


def run_baseline_cv(pipe: Pipeline, X_train, y_train, Cs: np.ndarray, cv: StratifiedKFold) -> Tuple[pd.DataFrame, Dict[str, float]]:
    param_grid = {
        "selector": ["passthrough"],
        "svm__class_weight": [None, "balanced"],
        "svm__C": Cs,
    }
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
    gs.fit(X_train, y_train)
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(RESULTS_DIR / "baseline_grid.csv", index=False)

    # Calcul baseline: prendre les lignes selector=passthrough et agréger sur C/class_weight
    best_idx = cv_df["rank_test_f1_macro"].idxmin()
    best_params = gs.best_params_
    baseline_mean = {
        "f1_macro_mean": float(cv_df.loc[best_idx, "mean_test_f1_macro"]),
        "f1_macro_std": float(cv_df.loc[best_idx, "std_test_f1_macro"]),
        "bal_acc_mean": float(cv_df.loc[best_idx, "mean_test_bal_acc"]),
        "bal_acc_std": float(cv_df.loc[best_idx, "std_test_bal_acc"]),
    }
    with open(RESULTS_DIR / "baseline_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, **baseline_mean}, f, ensure_ascii=False, indent=2)
    return cv_df, baseline_mean


def run_feature_selection_cv(pipe: Pipeline, X_train, y_train, Cs: np.ndarray, ks: List[int], cv: StratifiedKFold) -> pd.DataFrame:
    param_grid = {
        "selector": [SelectKBest(score_func=f_classif)],
        "selector__k": ks,
        "svm__class_weight": [None, "balanced"],
        "svm__C": Cs,
    }
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
    gs.fit(X_train, y_train)
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(RESULTS_DIR / "featuresel_grid.csv", index=False)
    return cv_df


def summarize_scores_by_k(cv_df: pd.DataFrame) -> pd.DataFrame:
    # Choisir pour chaque k la meilleure combinaison (C, class_weight) selon mean_test_f1_macro
    df = cv_df.copy()
    if "param_selector__k" not in df.columns:
        raise ValueError("La colonne param_selector__k est absente des résultats de CV.")
    df["k"] = df["param_selector__k"].astype(float)
    grp = (
        df.sort_values("mean_test_f1_macro", ascending=False)
          .groupby("k", as_index=False)
          .first()
          [["k", "mean_test_f1_macro", "std_test_f1_macro", "mean_test_bal_acc", "std_test_bal_acc"]]
          .sort_values("k")
    )
    grp.to_csv(RESULTS_DIR / "featuresel_k_summary.csv", index=False)
    return grp


def plot_score_vs_k(k_summary: pd.DataFrame) -> Path:
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = px.line(
        k_summary,
        x="k",
        y=["mean_test_f1_macro", "mean_test_bal_acc"],
        title="Scores CV vs k (SelectKBest)",
        labels={"k": "#features", "value": "Score", "variable": "Métrique"},
        markers=True,
    )
    outfile = plots_dir / "featuresel_scores_vs_k.html"
    pio.write_html(fig, str(outfile), auto_open=False)
    return outfile


def choose_min_k_within_tolerance(k_summary: pd.DataFrame, baseline_f1: float, tol: float = 0.003) -> int:
    # Seuil minimal autorisé
    threshold = baseline_f1 - tol
    ok = k_summary[k_summary["mean_test_f1_macro"] >= threshold]
    if ok.empty:
        # si aucun ne respecte la tolérance, prendre k avec meilleur f1
        return int(k_summary.sort_values("mean_test_f1_macro", ascending=False).iloc[0]["k"])
    return int(ok.sort_values("k").iloc[0]["k"])


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)
    # Sauvegarder colonnes et features
    with open(RESULTS_DIR / "columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(list(df.columns)))
    with open(RESULTS_DIR / "columns.json", "w", encoding="utf-8") as f:
        json.dump(list(df.columns), f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    with open(RESULTS_DIR / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = build_baseline_pipeline(RANDOM_STATE)

    # Grille C autour du plateau
    Cs = np.array([75, 80, 85], dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Baseline sans sélection
    print("\n[Baseline] CV en cours...")
    base_cv, base_stats = run_baseline_cv(pipe, X_train, y_train, Cs, cv)
    print(f"Baseline CV -> f1_macro: {base_stats['f1_macro_mean']:.4f} ± {base_stats['f1_macro_std']:.4f} | "
          f"bal_acc: {base_stats['bal_acc_mean']:.4f} ± {base_stats['bal_acc_std']:.4f}")

    # Refit meilleur baseline et test
    base_best_idx = base_cv["rank_test_f1_macro"].idxmin()
    base_best_params = json.loads(json.dumps({
        "selector": "passthrough",
        "svm__C": float(base_cv.loc[base_best_idx, "param_svm__C"]),
        "svm__class_weight": base_cv.loc[base_best_idx, "param_svm__class_weight"],
    }))
    base_model = build_baseline_pipeline(RANDOM_STATE)
    base_model.set_params(**base_best_params)
    base_model.fit(X_train, y_train)
    base_test = evaluate_on_test(base_model, X_test, y_test)
    with open(RESULTS_DIR / "baseline_test.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": base_best_params, "test": base_test}, f, ensure_ascii=False, indent=2)

    # 2) Feature selection: définir ks
    max_k = len(feature_names)
    if max_k <= 10:
        ks = list(range(3, max_k + 1))
    else:
        step = max(1, max_k // 10)
        ks = sorted(set(list(range(5, max_k + 1, step)) + [max_k]))

    print("\n[Feature Selection] CV en cours...")
    fs_cv = run_feature_selection_cv(pipe, X_train, y_train, Cs, ks, cv)
    k_summary = summarize_scores_by_k(fs_cv)
    plot_path = plot_score_vs_k(k_summary)
    print(f"Courbe score vs k sauvegardée: {plot_path}")

    # Choix du k minimal dans tolérance
    tol = 0.003  # marge par défaut (peut être ajustée)
    chosen_k = choose_min_k_within_tolerance(k_summary, baseline_f1=base_stats["f1_macro_mean"], tol=tol)
    with open(RESULTS_DIR / "chosen_k.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_f1_macro": base_stats["f1_macro_mean"], "tolerance": tol, "chosen_k": chosen_k}, f, ensure_ascii=False, indent=2)
    print(f"k retenu (tol={tol}): {chosen_k}")

    # Refit avec k retenu: retrouver meilleurs C/class_weight à ce k
    sel_rows = fs_cv[fs_cv["param_selector__k"].astype(int) == int(chosen_k)]
    best_row = sel_rows.sort_values("mean_test_f1_macro", ascending=False).iloc[0]
    best_params_k = {
        "selector": SelectKBest(score_func=f_classif, k=int(chosen_k)),
        "svm__C": float(best_row["param_svm__C"]),
        "svm__class_weight": best_row["param_svm__class_weight"],
    }
    final_model = build_baseline_pipeline(RANDOM_STATE)
    final_model.set_params(**best_params_k)
    final_model.fit(X_train, y_train)
    final_test = evaluate_on_test(final_model, X_test, y_test)
    serializable_params = {
        "selector": {"type": "SelectKBest", "k": int(chosen_k)},
        "svm__C": float(best_row["param_svm__C"]),
        "svm__class_weight": (None if pd.isna(best_row["param_svm__class_weight"]) else best_row["param_svm__class_weight"]),
    }
    with open(RESULTS_DIR / "final_selected_k_test.json", "w", encoding="utf-8") as f:
        json.dump({"best_params_at_k": serializable_params, "test": final_test}, f, ensure_ascii=False, indent=2)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
