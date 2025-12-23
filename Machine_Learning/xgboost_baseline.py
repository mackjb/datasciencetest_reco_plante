#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import base64
import mimetypes

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
    cross_validate,
    validation_curve,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import plotly.express as px
import plotly.io as pio

try:
    from src.helpers.helpers import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("xgboost n'est pas installé. Installez-le avec: pip install xgboost") from e

RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantVillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results_modifiés" / "models" / "xgb_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"
USE_GPU: bool = True

IMAGE_PREFIX_OLD: str = r"C:\\Users\\bgasm\\Documents\\Bernadette\\IA\\projet\\notebook_rendu_1\\version_7_new_features\\plantvillage dataset\\segmented"
IMAGE_PREFIX_NEW: str = r"/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"

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
        return ("file:///" + quote(cand.replace("\\", "/"), safe="/:")) if cand else None


def to_data_uri_if_readable(path_str: str) -> Optional[str]:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    s = path_str
    try:
        if IMAGE_PREFIX_NEW and IMAGE_PREFIX_OLD:
            s_norm = re.sub(r"[\\]+", "/", s).strip()
            old_norm = re.sub(r"[\\]+", "/", IMAGE_PREFIX_OLD).strip()
            if s_norm.lower().startswith(old_norm.lower()):
                rest = s_norm[len(old_norm):].lstrip("/")
                p = Path(IMAGE_PREFIX_NEW) / Path(rest)
            else:
                p = Path(s_norm if os.path.isabs(s_norm) else (PROJECT_ROOT / s_norm).resolve())
        else:
            p = Path(s if os.path.isabs(s) else (PROJECT_ROOT / s).resolve())
    except Exception:
        p = Path(s if os.path.isabs(s) else (PROJECT_ROOT / s).resolve())

    try:
        if not p.exists() or not p.is_file():
            return None
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime, _ = mimetypes.guess_type(str(p))
        if not mime:
            mime = "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

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


def build_xgb_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    if USE_GPU:
        try:
            clf = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",  # GPU via device='cuda' (XGBoost >= 2.0)
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
                device="cuda",
            )
        except TypeError:
            # Older XGBoost without 'device' param: fallback to CPU 'hist'
            clf = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
            )
    else:
        clf = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("xgb", clf),
    ])


def run_grid_cv(pipe: Pipeline, X, y, param_grid: Dict, results_path: Path) -> Tuple[GridSearchCV, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring={"bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        n_jobs=(1 if USE_GPU else -1),
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(results_path, index=False)
    return gs, cv_df


def cv_summary_str(cv_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
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


def repeated_cv_report(estimator: Pipeline, X, y, seeds: List[int], n_splits: int = 5) -> Dict[str, Dict[str, float]]:
    all_f1: List[float] = []
    all_bal: List[float] = []
    scoring = {"f1_macro": "f1_macro", "bal_acc": "balanced_accuracy"}
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_res = cross_validate(estimator, X, y, scoring=scoring, cv=cv, n_jobs=(1 if USE_GPU else -1))
        all_f1.extend(cv_res["test_f1_macro"].tolist())
        all_bal.extend(cv_res["test_bal_acc"].tolist())
    def stats(a: List[float]) -> Tuple[float, float, float]:
        arr = np.asarray(a, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        se = float(std / np.sqrt(arr.size))
        return mean, std, se
    f1_mean, f1_std, f1_se = stats(all_f1)
    bal_mean, bal_std, bal_se = stats(all_bal)
    return {
        "f1_macro": {"mean": f1_mean, "std": f1_std, "se": f1_se, "raw": list(map(float, all_f1))},
        "bal_acc": {"mean": bal_mean, "std": bal_std, "se": bal_se, "raw": list(map(float, all_bal))},
    }


def bootstrap_ci_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, B: int = 1000, random_state: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = y_true.shape[0]
    scores = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        scores[b] = metric_fn(y_true[idx], y_pred[idx])
    mean = float(np.mean(scores))
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return mean, low, high


def plot_validation_curves_xgb(method_name: str, pipe: Pipeline, X, y, best_params: Dict[str, float], results_dir: Path, n_splits: int = 3) -> Dict[str, object]:
    out: Dict[str, object] = {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_est_values = np.linspace(100, 600, num=5, dtype=int)
    train_scores, test_scores = validation_curve(
        pipe, X, y,
        param_name="xgb__n_estimators",
        param_range=n_est_values,
        scoring="f1_macro",
        cv=cv,
        n_jobs=(1 if USE_GPU else -1),
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    fig1 = px.line(x=n_est_values, y=test_mean, error_y=test_std, labels={"x": "n_estimators", "y": "F1_macro (CV)"}, title=f"Validation curve — n_estimators — {method_name}")
    fig1.add_scatter(x=n_est_values, y=train_mean, mode="lines", name="Train")
    p1 = results_dir / "plots" / method_name / "validation_curve_n_estimators.html"
    p1.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig1, str(p1), auto_open=False)
    out["validation_curve_n_estimators_path"] = p1

    md_values = np.array([3, 5, 7, 9, 11])
    pipe_md = Pipeline(pipe.steps)
    pipe_md.set_params(**{f"xgb__n_estimators": int(best_params.get("xgb__n_estimators", 300))})
    train_scores_g, test_scores_g = validation_curve(
        pipe_md, X, y,
        param_name="xgb__max_depth",
        param_range=md_values,
        scoring="f1_macro",
        cv=cv,
        n_jobs=(1 if USE_GPU else -1),
    )
    train_mean_g = train_scores_g.mean(axis=1)
    train_std_g = train_scores_g.std(axis=1)
    test_mean_g = test_scores_g.mean(axis=1)
    test_std_g = test_scores_g.std(axis=1)
    fig2 = px.line(x=md_values, y=test_mean_g, error_y=test_std_g, labels={"x": "max_depth", "y": "F1_macro (CV)"}, title=f"Validation curve — max_depth — {method_name}")
    fig2.add_scatter(x=md_values, y=train_mean_g, mode="lines", name="Train")
    p2 = results_dir / "plots" / method_name / "validation_curve_max_depth.html"
    p2.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig2, str(p2), auto_open=False)
    out["validation_curve_max_depth_path"] = p2

    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_confusion_heatmap_html(cm: np.ndarray, labels: List[str], out_html: Path, title: str) -> None:
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Vrai", color="Comptes"),
        title=title,
    )
    tickvals = list(range(len(labels)))
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=labels)
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=labels)
    pio.write_html(fig, str(out_html), auto_open=False)


def save_all_classif_artifacts(method_name: str, model: Pipeline, df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Path]:
    method_dir = RESULTS_DIR / "evaluation" / method_name
    plots_dir = RESULTS_DIR / "plots" / method_name
    _ensure_dir(method_dir)
    _ensure_dir(plots_dir)

    y_pred = model.predict(X_test)

    report_txt = method_dir / "classification_report.txt"
    report = classification_report(y_test, y_pred, digits=4)
    report_txt.write_text(report, encoding="utf-8")

    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_csv = method_dir / "confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_csv, header=False, index=False)
    cm_html = plots_dir / "confusion_matrix.html"
    save_confusion_heatmap_html(cm, list(map(str, labels)), cm_html, title=f"Matrice de confusion (test) — {method_name}")

    cm_interactive_html = plots_dir / "confusion_matrix_interactive.html"
    save_confusion_heatmap_html(cm, list(map(str, labels)), cm_interactive_html, title=f"Matrice de confusion interactive — {method_name}")

    return {
        "report_txt": report_txt,
        "cm_csv": cm_csv,
        "cm_html": cm_html,
        "cm_interactive_html": cm_interactive_html,
    }


def save_diagnostics(method_name: str, results_dir: Path, cv_rep: Dict[str, Dict[str, float]], bootstrap: Dict[str, Dict[str, float]], complexity: Dict[str, float], valcurves: Optional[Dict[str, object]]) -> None:
    method_dir = results_dir / "evaluation" / method_name
    plots_dir = results_dir / "plots" / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    diag = {
        "repeated_cv": cv_rep,
        "bootstrap": bootstrap,
        "complexity": complexity,
        "validation_curve_paths": {
            "n_estimators": str(valcurves.get("validation_curve_n_estimators_path")) if valcurves else None,
            "max_depth": str(valcurves.get("validation_curve_max_depth_path")) if valcurves else None,
        },
    }
    (method_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_xgb_complexity(fitted_pipe: Pipeline, y_train: pd.Series) -> Dict[str, float]:
    xgb = fitted_pipe.named_steps.get("xgb")
    if xgb is None:
        return {"n_estimators": float("nan"), "max_depth": float("nan")}
    n_classes = float(pd.Series(y_train).nunique())
    return {
        "n_estimators": float(getattr(xgb, "n_estimators", float("nan"))),
        "max_depth": float(getattr(xgb, "max_depth", float("nan"))),
        "n_classes": n_classes,
    }


def main() -> None:
    print("\nPlan d'évaluation (XGBoost Baseline):")
    for s in [
        "1) Charger CSV + nettoyer",
        "2) Split stratifié train/test",
        "3) RobustScaler -> XGBClassifier: CV + test",
        "4) Sauvegarde des résumés et graphes",
    ]:
        print(" - ", s)

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    (RESULTS_DIR / "label_mapping.json").write_text(json.dumps({int(i): str(c) for i, c in enumerate(le.classes_)}, ensure_ascii=False, indent=2), encoding="utf-8")
    X_train, X_test, y_train, y_test = stratified_split(X, y_enc, test_size=0.2, random_state=RANDOM_STATE)

    print("\n[XGBoost] CV en cours...")
    xgb_pipe = build_xgb_pipeline(RANDOM_STATE)
    xgb_grid = {
        "xgb__n_estimators": [300],
        "xgb__max_depth": [6],
        "xgb__learning_rate": [0.1, 0.05],
        "xgb__subsample": [0.8],
        "xgb__colsample_bytree": [0.8],
    }
    xgb_gs, xgb_cv = run_grid_cv(xgb_pipe, X_train, y_train, xgb_grid, RESULTS_DIR / "xgb_grid.csv")
    xgb_best = xgb_gs.best_estimator_
    xgb_test = evaluate_test(xgb_best, X_test, y_test)
    xgb_summary = cv_summary_str(xgb_cv, ["param_xgb__n_estimators", "param_xgb__max_depth", "param_xgb__learning_rate", "param_xgb__subsample", "param_xgb__colsample_bytree"]).sort_values(["param_xgb__n_estimators", "param_xgb__max_depth", "param_xgb__learning_rate"])  # type: ignore

    _ = save_all_classif_artifacts("xgboost", xgb_best, df, X_train, y_train, X_test, y_test)

    (RESULTS_DIR / "xgb_cv_summary.json").write_text(json.dumps({
        "best_params": xgb_gs.best_params_,
        "best_cv_f1_macro": float(xgb_gs.best_score_),
        "cv_rows": len(xgb_cv),
        "summary_preview": xgb_summary.head(10).to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "xgb_test.json").write_text(json.dumps(xgb_test, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"XGB CV -> f1_macro: {xgb_gs.cv_results_['mean_test_f1_macro'][xgb_gs.best_index_]:.4f} ± {xgb_gs.cv_results_['std_test_f1_macro'][xgb_gs.best_index_]:.4f} | "
          f"bal_acc: {xgb_gs.cv_results_['mean_test_bal_acc'][xgb_gs.best_index_]:.4f} ± {xgb_gs.cv_results_['std_test_bal_acc'][xgb_gs.best_index_]:.4f}")
    print(f"XGB Test -> bal_acc: {xgb_test['balanced_accuracy']:.4f} | f1_macro: {xgb_test['f1_macro']:.4f}")

    seeds = [1, 2, 3]
    xgb_cv_rep = repeated_cv_report(xgb_best, X_train, y_train, seeds=seeds, n_splits=5)
    mean_f1, std_f1, se_f1 = xgb_cv_rep["f1_macro"]["mean"], xgb_cv_rep["f1_macro"]["std"], xgb_cv_rep["f1_macro"]["se"]
    mean_bal, std_bal, se_bal = xgb_cv_rep["bal_acc"]["mean"], xgb_cv_rep["bal_acc"]["std"], xgb_cv_rep["bal_acc"]["se"]
    print(f"CV f1_macro = {mean_f1:.4f} ± {std_f1:.4f}  (SE ≈ {se_f1:.4f})")
    print(f"CV balanced_accuracy = {mean_bal:.4f} ± {std_bal:.4f}  (SE ≈ {se_bal:.4f})")

    y_pred_xgb = xgb_best.predict(X_test)
    f1_mean_b, f1_low_b, f1_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred_xgb, lambda yt, yp: f1_score(yt, yp, average="macro"), B=500, random_state=RANDOM_STATE)
    bal_mean_b, bal_low_b, bal_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred_xgb, lambda yt, yp: balanced_accuracy_score(yt, yp), B=500, random_state=RANDOM_STATE)
    print(f"F1_macro test = {f1_mean_b:.4f}  IC95% [{f1_low_b:.4f}, {f1_high_b:.4f}]")
    print(f"Balanced_accuracy test = {bal_mean_b:.4f}  IC95% [{bal_low_b:.4f}, {bal_high_b:.4f}]")

    complexity = compute_xgb_complexity(xgb_best, y_train)
    print(f"n_estimators = {complexity.get('n_estimators')} | max_depth = {complexity.get('max_depth')}")

    valcurves = plot_validation_curves_xgb("xgboost", build_xgb_pipeline(RANDOM_STATE), X_train, y_train, {
        "xgb__n_estimators": xgb_gs.best_params_.get("xgb__n_estimators", 300)
    }, RESULTS_DIR, n_splits=3)

    save_diagnostics(
        "xgboost",
        RESULTS_DIR,
        cv_rep=xgb_cv_rep,
        bootstrap={
            "f1_macro": {"mean": f1_mean_b, "low95": f1_low_b, "high95": f1_high_b},
            "balanced_accuracy": {"mean": bal_mean_b, "low95": bal_low_b, "high95": bal_high_b},
        },
        complexity=complexity,
        valcurves=valcurves,
    )

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
