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

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
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

RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantVillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results_modifiés" / "models" / "logreg_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

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
            new_posix = str(PureWindowsPath(IMAGE_PREFIX_NEW) / rest.replace("/", "\\")).replace("\\", "/")
            return "file:///" + quote(new_posix, safe="/:")
    if re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith("\\\\"):
        s_posix = s.replace("\\", "/")
        return "file:///" + quote(s_posix, safe="/:")
    cand = s if os.path.isabs(s) else str((PROJECT_ROOT / s).resolve())
    try:
        return Path(cand).as_uri()
    except Exception:
        return ("file:///" + quote(cand.replace("\\", "/"), safe="/:")) if cand else None


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


def build_logreg_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=2000,
        multi_class="multinomial",
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("logreg", clf),
    ])


def run_grid_cv(pipe: Pipeline, X, y, param_grid: Dict, results_path: Path) -> Tuple[GridSearchCV, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring={"bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        n_jobs=1,
        pre_dispatch='1*n_jobs',
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
        cv_res = cross_validate(estimator, X, y, scoring=scoring, cv=cv, n_jobs=1)
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


def bootstrap_ci_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, B: int = 500, random_state: int = 42) -> Tuple[float, float, float]:
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


def plot_validation_curves_logreg(method_name: str, pipe: Pipeline, X, y, best_C: float, results_dir: Path, n_splits: int = 3) -> Dict[str, object]:
    out: Dict[str, object] = {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    C_values = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    train_scores, test_scores = validation_curve(
        pipe, X, y,
        param_name="logreg__C",
        param_range=C_values,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    fig1 = px.line(x=C_values, y=test_mean, error_y=test_std, labels={"x": "C", "y": "F1_macro (CV)"}, title=f"Validation curve — C — {method_name}")
    fig1.add_scatter(x=C_values, y=train_mean, mode="lines", name="Train")
    p1 = results_dir / "plots" / method_name / "validation_curve_C.html"
    p1.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig1, str(p1), auto_open=False)
    out["validation_curve_C_path"] = p1

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
            "C": str(valcurves.get("validation_curve_C_path")) if valcurves else None,
        },
    }
    (method_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_logreg_complexity(fitted_pipe: Pipeline) -> Dict[str, float]:
    logreg = fitted_pipe.named_steps.get("logreg")
    if logreg is None:
        return {"n_classes": float("nan"), "n_features": float("nan"), "coef_l2": float("nan")}
    try:
        coef = getattr(logreg, "coef_", None)
        if coef is None:
            return {"n_classes": float("nan"), "n_features": float("nan"), "coef_l2": float("nan")}
        n_classes, n_features = coef.shape
        coef_l2 = float(np.linalg.norm(coef))
        return {"n_classes": float(n_classes), "n_features": float(n_features), "coef_l2": coef_l2}
    except Exception:
        return {"n_classes": float("nan"), "n_features": float("nan"), "coef_l2": float("nan")}


def main() -> None:
    print("\nPlan d'évaluation (Logistic Regression Baseline):")
    for s in [
        "1) Charger CSV + nettoyer",
        "2) Split stratifié train/test",
        "3) RobustScaler -> LogisticRegression: CV + test",
        "4) Sauvegarde des résumés et graphes",
    ]:
        print(" - ", s)

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("\n[LogReg] CV en cours...")
    pipe = build_logreg_pipeline(RANDOM_STATE)
    grid = {
        "logreg__C": [0.1, 1.0, 10.0],
        "logreg__class_weight": [None, "balanced"],
    }
    gs, cv_df = run_grid_cv(pipe, X_train, y_train, grid, RESULTS_DIR / "logreg_grid.csv")
    best = gs.best_estimator_
    test_scores = evaluate_test(best, X_test, y_test)
    summary = cv_summary_str(cv_df, ["param_logreg__C", "param_logreg__class_weight"]).sort_values("param_logreg__C")

    _ = save_all_classif_artifacts("logreg", best, df, X_train, y_train, X_test, y_test)

    (RESULTS_DIR / "logreg_cv_summary.json").write_text(json.dumps({
        "best_params": gs.best_params_,
        "best_cv_f1_macro": float(gs.best_score_),
        "cv_rows": len(cv_df),
        "summary_preview": summary.head(10).to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "logreg_test.json").write_text(json.dumps(test_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"LogReg CV -> f1_macro: {gs.cv_results_['mean_test_f1_macro'][gs.best_index_]:.4f} ± {gs.cv_results_['std_test_f1_macro'][gs.best_index_]:.4f} | "
          f"bal_acc: {gs.cv_results_['mean_test_bal_acc'][gs.best_index_]:.4f} ± {gs.cv_results_['std_test_bal_acc'][gs.best_index_]:.4f}")
    print(f"LogReg Test -> bal_acc: {test_scores['balanced_accuracy']:.4f} | f1_macro: {test_scores['f1_macro']:.4f}")

    seeds = [1, 2, 3]
    cv_rep = repeated_cv_report(best, X_train, y_train, seeds=seeds, n_splits=5)
    mean_f1, std_f1, se_f1 = cv_rep["f1_macro"]["mean"], cv_rep["f1_macro"]["std"], cv_rep["f1_macro"]["se"]
    mean_bal, std_bal, se_bal = cv_rep["bal_acc"]["mean"], cv_rep["bal_acc"]["std"], cv_rep["bal_acc"]["se"]
    print(f"CV f1_macro = {mean_f1:.4f} ± {std_f1:.4f}  (SE ≈ {se_f1:.4f})")
    print(f"CV balanced_accuracy = {mean_bal:.4f} ± {std_bal:.4f}  (SE ≈ {se_bal:.4f})")

    y_pred = best.predict(X_test)
    f1_mean_b, f1_low_b, f1_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro"), B=500, random_state=RANDOM_STATE)
    bal_mean_b, bal_low_b, bal_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred, lambda yt, yp: balanced_accuracy_score(yt, yp), B=500, random_state=RANDOM_STATE)
    print(f"F1_macro test = {f1_mean_b:.4f}  IC95% [{f1_low_b:.4f}, {f1_high_b:.4f}]")
    print(f"Balanced_accuracy test = {bal_mean_b:.4f}  IC95% [{bal_low_b:.4f}, {bal_high_b:.4f}]")

    complexity = compute_logreg_complexity(best)
    print(f"n_classes = {complexity.get('n_classes')} | n_features = {complexity.get('n_features')} | coef_l2 = {complexity.get('coef_l2'):.4f}")

    valcurves = plot_validation_curves_logreg("logreg", build_logreg_pipeline(RANDOM_STATE), X_train, y_train, float(gs.best_params_.get("logreg__C", 1.0)), RESULTS_DIR, n_splits=3)

    save_diagnostics(
        "logreg",
        RESULTS_DIR,
        cv_rep=cv_rep,
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
