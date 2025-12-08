#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVM RBF "best reduced" sur PlantVillage avec sélection de features figée.

Hyperparamètres figés après étude:
- Scaler: RobustScaler
- Sélection: SelectKBest(f_classif, k=23)
- SVC: kernel=rbf, C=85.0, gamma='scale', class_weight=None, decision_function_shape='ovr'

Produit les mêmes artefacts que le script "the_best" + export des 23 features sélectionnées.
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    from src.helpers.helpers import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "svm_rbf_best_reduced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

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
    "species",
    "ID_Image",
    "Est_Saine",
    "Image_Path",
    "is_black",
    "plant_Apple","plant_Blueberry","plant_Cherry_(including_sour)","plant_Corn_(maize)","plant_Grape","plant_Orange","plant_Peach","plant_Pepper,_bell","plant_Potato","plant_Raspberry","plant_Soybean","plant_Squash","plant_Strawberry","plant_Tomato",
    "disease_Apple_scab","disease_Bacterial_spot","disease_Black_rot","disease_Cedar_apple_rust","disease_Cercospora_leaf_spot Gray_leaf_spot","disease_Common_rust_","disease_Early_blight","disease_Esca_(Black_Measles)","disease_Haunglongbing_(Citrus_greening)","disease_Late_blight","disease_Leaf_Mold","disease_Leaf_blight_(Isariopsis_Leaf_Spot)","disease_Leaf_scorch","disease_Northern_Leaf_Blight","disease_Powdery_mildew","disease_Septoria_leaf_spot","disease_Spider_mites Two-spotted_spider_mite","disease_Target_Spot","disease_Tomato_Yellow_Leaf_Curl_Virus","disease_Tomato_mosaic_virus","disease_healthy",
    "filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_na","hash","is_duplicate_after_first","disease",
]


def print_plan() -> None:
    print("\nPlan d'exécution (best reduced):")
    steps = [
        "1) Charger CSV propre + traiter NA/doublons",
        "2) Split stratifié train/test",
        "3) Pipeline: RobustScaler -> SelectKBest(k=23) -> SVC(RBF, C=85)",
        "4) Entraînement et évaluation test",
        "5) Export des features sélectionnées et des métriques",
    ]
    for s in steps:
        print(" - ", s)


# -----------------------------
# Chargement & préparation des données
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

def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=23)),
        ("svm", SVC(
            kernel="rbf",
            C=85.0,
            gamma="scale",
            class_weight=None,
            decision_function_shape="ovr",
            random_state=random_state,
        )),
    ])


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    # Rapport + matrice de confusion
    report = classification_report(y_test, y_pred, digits=4)
    print("\nRapport de classification (test):\n" + report)
    cm = confusion_matrix(y_test, y_pred)
    eval_dir = RESULTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm).to_csv(eval_dir / "confusion_matrix.csv", index=False, header=False)
    with open(eval_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return {"balanced_accuracy": float(bal_acc), "f1_macro": float(f1m)}


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)

    # Sauvegarder colonnes/features complètes
    with open(RESULTS_DIR / "columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(list(df.columns)))
    with open(RESULTS_DIR / "columns.json", "w", encoding="utf-8") as f:
        json.dump(list(df.columns), f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "features_all.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    with open(RESULTS_DIR / "features_all.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = build_pipeline(RANDOM_STATE)
    fitted = pipe.fit(X_train, y_train)

    # Sauvegarder les 23 features sélectionnées
    selector = fitted.named_steps["selector"]
    mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, mask) if m]
    with open(RESULTS_DIR / "features_selected.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))
    with open(RESULTS_DIR / "features_selected.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, ensure_ascii=False, indent=2)

    # Évaluation finale
    test_metrics = evaluate_on_test(fitted, X_test, y_test)

    # Export JSON des hyperparamètres et métriques
    export = {
        "model": "SVC",
        "scaler": "RobustScaler",
        "selector": {"type": "SelectKBest", "score_func": "f_classif", "k": 23},
        "params": {
            "kernel": "rbf",
            "C": 85.0,
            "gamma": "scale",
            "class_weight": None,
            "decision_function_shape": "ovr",
            "random_state": RANDOM_STATE,
        },
        "test_metrics": test_metrics,
        "timestamp": time.time(),
    }
    with open(RESULTS_DIR / "best_reduced_run.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
