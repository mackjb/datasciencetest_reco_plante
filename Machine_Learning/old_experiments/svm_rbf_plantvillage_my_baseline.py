#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline SVM-RBF sur PlantVillage, avec génération d'artefacts (rapports, matrices,
galeries, vue interactive images). Version allégée pour vérifier l'affichage images.
"""
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
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
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

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantVillage" / "csv" / "clean_with_features_data_plantVillage_segmented_all_modif.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results_modifiés" / "models" / "svm_rbf_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Réécriture optionnelle des chemins (Windows -> emplacement local)
IMAGE_PREFIX_OLD: str = r"C:\\Users\\bgasm\\Documents\\Bernadette\\IA\\projet\\notebook_rendu_1\\version_7_new_features\\plantvillage dataset\\segmented"
IMAGE_PREFIX_NEW: str = r"/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"

# -----------------------------
# Helpers
# -----------------------------

def to_file_uri_from_any(val: str) -> Optional[str]:
    if not isinstance(val, str) or not val:
        return None
    s = val
    # déjà une URI ?
    if s.startswith("file://"):
        return s
    # Réécriture Windows -> local
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
    # Chemin Windows absolu
    if re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith("\\\\"):
        s_posix = s.replace("\\", "/")
        return "file:///" + quote(s_posix, safe="/:")
    # Relatif/absolu local
    cand = s if os.path.isabs(s) else str((PROJECT_ROOT / s).resolve())
    try:
        return Path(cand).as_uri()
    except Exception:
        return ("file:///" + quote(cand.replace("\\", "/"), safe="/:")) if cand else None


def to_data_uri_if_readable(path_str: str) -> Optional[str]:
    """Lit un fichier image et renvoie une data URI base64. None si indisponible.
    Applique d'abord la réécriture IMAGE_PREFIX_OLD -> IMAGE_PREFIX_NEW.
    """
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

# Colonnes non-features
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
# Pipeline Baseline uniquement
# -----------------------------

def build_baseline_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
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
        cv_res = cross_validate(estimator, X, y, scoring=scoring, cv=cv, n_jobs=-1)
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


def compute_sv_ratio(fitted_pipe: Pipeline, n_train: int) -> float:
    svm = fitted_pipe.named_steps.get("svm")
    if svm is None or not hasattr(svm, "support_"):
        return float("nan")
    return float(len(svm.support_) / max(1, n_train))


def plot_validation_curves(method_name: str, pipe: Pipeline, X, y, best_C: float, results_dir: Path, n_splits: int = 5) -> Dict[str, object]:
    out: Dict[str, object] = {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Courbe pour C (gamma fixe)
    C_values = np.logspace(-1, 3, num=10)
    train_scores, test_scores = validation_curve(
        pipe, X, y,
        param_name="svm__C",
        param_range=C_values,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    figC = px.line(x=C_values, y=test_mean, error_y=test_std, labels={"x": "C", "y": "F1_macro (CV)"}, title=f"Validation curve — C — {method_name}")
    figC.add_scatter(x=C_values, y=train_mean, mode="lines", name="Train")
    figC.update_xaxes(type="log")
    pC = results_dir / "plots" / method_name / "validation_curve_C.html"
    pC.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(figC, str(pC), auto_open=False)
    out["validation_curve_C_path"] = pC
    out["validation_curve_C_data"] = {
        "C_values": C_values.tolist(),
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "cv_mean": test_mean.tolist(),
        "cv_std": test_std.tolist(),
    }

    # Courbe pour gamma (C fixé au best)
    gamma_values = np.logspace(-4, 1, num=10)
    pipe_gamma = Pipeline(pipe.steps)  # shallow copy
    pipe_gamma.set_params(**{"svm__C": best_C})
    train_scores_g, test_scores_g = validation_curve(
        pipe_gamma, X, y,
        param_name="svm__gamma",
        param_range=gamma_values,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    train_mean_g = train_scores_g.mean(axis=1)
    train_std_g = train_scores_g.std(axis=1)
    test_mean_g = test_scores_g.mean(axis=1)
    test_std_g = test_scores_g.std(axis=1)
    figG = px.line(x=gamma_values, y=test_mean_g, error_y=test_std_g, labels={"x": "gamma", "y": "F1_macro (CV)"}, title=f"Validation curve — gamma — {method_name}")
    figG.add_scatter(x=gamma_values, y=            train_mean_g, mode="lines", name="Train")
    figG.update_xaxes(type="log")
    pG = results_dir / "plots" / method_name / "validation_curve_gamma.html"
    pG.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(figG, str(pG), auto_open=False)
    out["validation_curve_gamma_path"] = pG
    out["validation_curve_gamma_data"] = {
        "gamma_values": gamma_values.tolist(),
        "train_mean": train_mean_g.tolist(),
        "train_std": train_std_g.tolist(),
        "cv_mean": test_mean_g.tolist(),
        "cv_std": test_std_g.tolist(),
    }

    return out


# -----------------------------
# Sauvegarde artefacts (rapports, matrices, galeries)
# -----------------------------

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


def save_simple_gallery(df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, out_html: Path, max_items: int = 40) -> None:
    test_idx = X_test.index
    df_test = df.loc[test_idx]
    rows = []
    for idx, yt, yp in zip(test_idx, y_test.values, y_pred):
        if yt != yp:
            fp = df_test.loc[idx].get("filepath", None)
            uri = None
            if isinstance(fp, (str, Path)):
                fp_str = str(fp)
                data_uri = to_data_uri_if_readable(fp_str)
                uri = data_uri if data_uri is not None else to_file_uri_from_any(fp_str)
            rows.append((str(idx), str(yt), str(yp), uri, str(fp) if isinstance(fp, str) else None))
    rows = rows[:max_items]

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Galerie des erreurs</title>",
        "<style>body{font-family:sans-serif} .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px} .card{border:1px solid #ccc;padding:8px} img{max-width:100%;max-height:200px;border:1px solid #ddd} .btn{margin-top:6px;padding:6px 10px;cursor:pointer}</style>",
        "<script>function copyText(str, btn){if(!str)return; if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(str).then(function(){if(btn){var t=btn.textContent; btn.textContent='Copié!'; setTimeout(function(){btn.textContent=t;}, 1200);}}).catch(function(){fallbackCopy(str, btn);});} else {fallbackCopy(str, btn);} } function fallbackCopy(str, btn){var ta=document.createElement('textarea'); ta.value=str; document.body.appendChild(ta); ta.select(); try{document.execCommand('copy'); if(btn){ var t=btn.textContent; btn.textContent='Copié!'; setTimeout(function(){btn.textContent=t;}, 1200);} } catch(e){} finally{document.body.removeChild(ta);} }</script>",
        "</head><body>",
        "<h1>Galerie des erreurs (quelques exemples)</h1>",
        f"<p>Nombre total d'exemples affichés: {len(rows)}</p>",
        "<div class='grid'>",
    ]
    for idx, yt, yp, uri, filepath_txt in rows:
        html.append("<div class='card'>")
        html.append(f"<div><b>idx</b>: {idx}</div>")
        html.append(f"<div><b>vrai</b>: {yt}</div>")
        html.append(f"<div><b>prédit</b>: {yp}</div>")
        if uri:
            html.append(f"<img src='{uri}' alt='img' />")
        else:
            html.append("<div>(image indisponible)</div>")
        if filepath_txt:
            filepath_attr = filepath_txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
            html.append(f"<div class='meta'><b>filepath</b>: {filepath_attr}</div>")
            html.append(f"<button class='btn' onclick=\"copyText('{filepath_attr}', this)\">Copier le chemin</button>")
        html.append("</div>")
    html.extend(["</div>", "</body></html>"])
    out_html.write_text("\n".join(html), encoding="utf-8")


def plot_confusion_matrix_interactive_with_images(
    df_full: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    method_name: str = "baseline",
    max_examples_search: int = 20,
) -> Path:
    plots_dir = RESULTS_DIR / "plots" / method_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Vrai", color="Comptes"),
        title=f"Matrice de confusion interactive (avec images) — {method_name}",
        x=labels_sorted,
        y=labels_sorted,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=900, margin=dict(l=80, r=20, t=60, b=80))

    train_by_class = {c: X_train.index[y_train.loc[X_train.index] == c] for c in pd.unique(y_train)}

    err_mask = (y_test.values != y_pred)
    df_err = pd.DataFrame({
        "idx": X_test.index,
        "true": y_test.values,
        "pred": y_pred,
    })
    df_err = df_err[err_mask]

    def pick_image_path(row_index: int) -> Optional[str]:
        for col in ["filepath", "filename", "Image_Path"]:
            if col in df_full.columns:
                val = df_full.loc[row_index, col]
                if isinstance(val, str) and val:
                    data_uri = to_data_uri_if_readable(val)
                    if data_uri is not None:
                        return data_uri
                    return to_file_uri_from_any(val)
        return None

    example_map = {}
    if not df_err.empty:
        grouped = df_err.groupby(["true", "pred"])  # type: ignore
        for (t, p), group in grouped:
            row = group.head(max_examples_search).head(1)
            test_path = None
            test_idx_one = None
            if not row.empty:
                test_idx_one = int(row.iloc[0]["idx"])  # index d'origine
                test_path = pick_image_path(test_idx_one)

            exemplar_path = None
            exemplar_idx = None
            cand_idx = train_by_class.get(p, [])
            if len(cand_idx) > 0:
                for ridx in list(cand_idx)[:max_examples_search]:
                    exemplar_path = pick_image_path(int(ridx))
                    if exemplar_path is not None:
                        exemplar_idx = int(ridx)
                        break

            example_map[(t, p)] = {
                "test": test_path,
                "test_idx": test_idx_one,
                "exemplar": exemplar_path,
                "exemplar_idx": exemplar_idx,
                "test_raw": df_full.loc[test_idx_one, "filepath"] if test_idx_one is not None else None,
                "exemplar_raw": df_full.loc[exemplar_idx, "filepath"] if exemplar_idx is not None else None,
            }

    base_div = fig.to_html(full_html=False, include_plotlyjs='inline')

    coord_map = {}
    for (t, p), paths in example_map.items():
        if t in labels_sorted and p in labels_sorted:
            i = labels_sorted.index(t)
            j = labels_sorted.index(p)
            coord_map[f"{i},{j}"] = paths

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        f'  <title>Matrice de confusion interactive (avec images) — {method_name}</title>\n'
        "  <style>\n"
        "    body { font-family: sans-serif; }\n"
        "    .container { display: grid; grid-template-columns: 1fr; gap: 16px; }\n"
        "    .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }\n"
        "    img { max-width: 100%; border: 1px solid #ddd; }\n"
        "    .muted { color: #777; font-size: 12px; }\n"
        "    .meta { font-size: 12px; color: #444; margin-top: 6px; }\n"
        "    .btn { margin-top: 6px; padding: 6px 10px; cursor: pointer; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>Matrice de confusion interactive (avec images) — {method_name}</h1>\n"
        "  <div class=\"container\">\n"
        f"    <div id=\"cm_div\">{base_div}</div>\n"
        "    <div id=\"info\" class=\"muted\">Cliquez une case hors-diagonale pour afficher des images (test vs exemplaire train). Les index sont affichés pour suppression ultérieure.</div>\n"
        "    <div id=\"pair\" class=\"pair\"></div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const coordMap = {json.dumps(coord_map)};\n"
        f"    const labels = {json.dumps(labels_sorted)};\n"
        "    function copyText(str, btn){\n"
        "      if(!str) return;\n"
        "      if(navigator.clipboard && navigator.clipboard.writeText){\n"
        "        navigator.clipboard.writeText(str).then(function(){ if(btn){ var t=btn.textContent; btn.textContent='Copié!'; setTimeout(function(){btn.textContent=t;}, 1200);} });\n"
        "      } else {\n"
        "        var ta=document.createElement('textarea'); ta.value=str; document.body.appendChild(ta); ta.select(); try{document.execCommand('copy'); if(btn){ var t=btn.textContent; btn.textContent='Copié!'; setTimeout(function(){btn.textContent=t;}, 1200);} } catch(e){} finally{document.body.removeChild(ta);}\n"
        "      }\n"
        "    }\n"
        "    function renderPair(obj){\n"
        "      const el = document.getElementById('pair');\n"
        "      el.innerHTML = '';\n"
        "      const left = document.createElement('div');\n"
        "      const right = document.createElement('div');\n"
        "      const testIdx = (obj && obj.test_idx != null) ? obj.test_idx : '—';\n"
        "      const exIdx = (obj && obj.exemplar_idx != null) ? obj.exemplar_idx : '—';\n"
        "      left.innerHTML = '<h3>Exemple test</h3>' + (obj && obj.test ? `<img src=\"${obj.test}\" />` : '<div class=\\'muted\\'>Aucun</div>') + `<div class=\"meta\">index: ${testIdx}</div>` + (obj && obj.test_raw ? `<div class=\"meta\"><b>filepath</b>: ${obj.test_raw}</div>` : '');\n"
        "      right.innerHTML = '<h3>Exemplaire train (classe prédite)</h3>' + (obj && obj.exemplar ? `<img src=\"${obj.exemplar}\" />` : '<div class=\\'muted\\'>Aucun</div>') + `<div class=\"meta\">index: ${exIdx}</div>` + (obj && obj.exemplar_raw ? `<div class=\"meta\"><b>filepath</b>: ${obj.exemplar_raw}</div>` : '');\n"
        "      if (obj && obj.test_raw){ var lb=document.createElement('button'); lb.className='btn'; lb.textContent='Copier le chemin'; lb.onclick=function(){copyText(obj.test_raw, lb);}; left.appendChild(lb);}\n"
        "      if (obj && obj.exemplar_raw){ var rb=document.createElement('button'); rb.className='btn'; rb.textContent='Copier le chemin'; rb.onclick=function(){copyText(obj.exemplar_raw, rb);}; right.appendChild(rb);}\n"
        "      el.appendChild(left); el.appendChild(right);\n"
        "    }\n"
        "    function bind(){\n"
        "      let cmDiv = document.querySelector('#cm_div .js-plotly-plot') || document.querySelector('#cm_div .plotly-graph-div');\n"
        "      if (!cmDiv || !cmDiv.on) return false;\n"
        "      cmDiv.on('plotly_click', function(data){\n"
        "        if (!data || !data.points || !data.points.length) return;\n"
        "        let i = data.points[0].y; let j = data.points[0].x;\n"
        "        if (typeof i === 'string') i = labels.indexOf(i);\n"
        "        if (typeof j === 'string') j = labels.indexOf(j);\n"
        "        const key = `${i},${j}`;\n"
        "        renderPair(coordMap[key] || null);\n"
        "      });\n"
        "      return true;\n"
        "    }\n"
        "    function tryBind(r){ if (bind()) return; if (r<=0) return; setTimeout(function(){tryBind(r-1);}, 200);}\n"
        "    window.addEventListener('load', function(){ tryBind(25); });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    outfile = plots_dir / "confusion_matrix_interactive_images.html"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile


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

    gallery_html = plots_dir / "confusion_matrix_gallery.html"
    save_simple_gallery(df, X_test, y_test, y_pred, gallery_html, max_items=48)

    return {
        "report_txt": report_txt,
        "cm_csv": cm_csv,
        "cm_html": cm_html,
        "cm_interactive_html": cm_interactive_html,
        "gallery_html": gallery_html,
    }


def save_diagnostics(method_name: str, results_dir: Path, cv_rep: Dict[str, Dict[str, float]], bootstrap: Dict[str, Dict[str, float]], sv_ratio: float, valcurves: Optional[Dict[str, object]]) -> None:
    method_dir = results_dir / "evaluation" / method_name
    plots_dir = results_dir / "plots" / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    diag = {
        "repeated_cv": cv_rep,
        "bootstrap": bootstrap,
        "sv_ratio": sv_ratio,
        "validation_curve_paths": {
            "C": str(valcurves.get("validation_curve_C_path")) if valcurves else None,
            "gamma": str(valcurves.get("validation_curve_gamma_path")) if valcurves else None,
        },
    }
    (method_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for metric in ("f1_macro", "bal_acc"):
        raw = cv_rep.get(metric, {}).get("raw", [])  # type: ignore
        for v in raw:
            rows.append({"metric": metric, "value": float(v)})
    if rows:
        pd.DataFrame(rows).to_csv(method_dir / "repeated_cv_scores.csv", index=False)


# -----------------------------
# Main (Baseline uniquement)
# -----------------------------

def main() -> None:
    print("\nPlan d'évaluation (Baseline uniquement):")
    for s in [
        "1) Charger CSV + nettoyer",
        "2) Split stratifié train/test",
        "3) Baseline: RobustScaler -> SVC(RBF): CV + test",
        "4) Sauvegarde des résumés et graphes",
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
        "svm__C": [76.0],
        "svm__class_weight": [None, "balanced"],
    }
    base_gs, base_cv = run_grid_cv(base_pipe, X_train, y_train, base_grid, RESULTS_DIR / "baseline_grid.csv")
    base_best = base_gs.best_estimator_
    base_test = evaluate_test(base_best, X_test, y_test)
    base_summary = cv_summary_str(base_cv, ["param_svm__C", "param_svm__class_weight"]).sort_values("param_svm__C")

    # Artefacts Baseline
    _ = save_all_classif_artifacts("baseline", base_best, df, X_train, y_train, X_test, y_test)
    try:
        y_pred_base_tmp = base_best.predict(X_test)
        plot_confusion_matrix_interactive_with_images(
            df_full=df,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred_base_tmp,
            method_name="baseline",
            max_examples_search=20,
        )
    except Exception as e:
        print(f"[WARN] Échec génération confusion_matrix_interactive_images (baseline): {e}")

    (RESULTS_DIR / "baseline_cv_summary.json").write_text(json.dumps({
        "best_params": base_gs.best_params_,
        "best_cv_f1_macro": float(base_gs.best_score_),
        "cv_rows": len(base_cv),
        "summary_preview": base_summary.head(10).to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "baseline_test.json").write_text(json.dumps(base_test, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Baseline CV -> f1_macro: {base_gs.cv_results_['mean_test_f1_macro'][base_gs.best_index_]:.4f} ± {base_gs.cv_results_['std_test_f1_macro'][base_gs.best_index_]:.4f} | "
          f"bal_acc: {base_gs.cv_results_['mean_test_bal_acc'][base_gs.best_index_]:.4f} ± {base_gs.cv_results_['std_test_bal_acc'][base_gs.best_index_]:.4f}")
    print(f"Baseline Test -> bal_acc: {base_test['balanced_accuracy']:.4f} | f1_macro: {base_test['f1_macro']:.4f}")

    # Diagnostics Baseline
    seeds = [1, 2, 3, 4, 5]
    base_cv_rep = repeated_cv_report(base_best, X_train, y_train, seeds=seeds, n_splits=5)
    mean_f1, std_f1, se_f1 = base_cv_rep["f1_macro"]["mean"], base_cv_rep["f1_macro"]["std"], base_cv_rep["f1_macro"]["se"]
    mean_bal, std_bal, se_bal = base_cv_rep["bal_acc"]["mean"], base_cv_rep["bal_acc"]["std"], base_cv_rep["bal_acc"]["se"]
    print(f"CV f1_macro = {mean_f1:.4f} ± {std_f1:.4f}  (SE ≈ {se_f1:.4f})")
    print(f"CV balanced_accuracy = {mean_bal:.4f} ± {std_bal:.4f}  (SE ≈ {se_bal:.4f})")

    # Bootstrap IC95% sur test
    y_pred_base = base_best.predict(X_test)
    f1_mean_b, f1_low_b, f1_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred_base, lambda yt, yp: f1_score(yt, yp, average="macro"), B=1000, random_state=RANDOM_STATE)
    bal_mean_b, bal_low_b, bal_high_b = bootstrap_ci_from_predictions(y_test.values, y_pred_base, lambda yt, yp: balanced_accuracy_score(yt, yp), B=1000, random_state=RANDOM_STATE)
    print(f"F1_macro test = {f1_mean_b:.4f}  IC95% [{f1_low_b:.4f}, {f1_high_b:.4f}]")
    print(f"Balanced_accuracy test = {bal_mean_b:.4f}  IC95% [{bal_low_b:.4f}, {bal_high_b:.4f}]")

    # Ratio SV et courbes de validation
    ratio_sv_base = compute_sv_ratio(base_best, n_train=X_train.shape[0])
    print(f"Ratio SV = {ratio_sv_base:.2f}")
    base_best_C = base_gs.best_params_.get("svm__C", None)
    base_valcurves = None
    if isinstance(base_best_C, (float, int)):
        base_valcurves = plot_validation_curves("baseline", build_baseline_pipeline(RANDOM_STATE), X_train, y_train, float(base_best_C), RESULTS_DIR, n_splits=5)

    # Sauvegarde diagnostics
    save_diagnostics(
        "baseline",
        RESULTS_DIR,
        cv_rep=base_cv_rep,
        bootstrap={
            "f1_macro": {"mean": f1_mean_b, "low95": f1_low_b, "high95": f1_high_b},
            "balanced_accuracy": {"mean": bal_mean_b, "low95": bal_low_b, "high95": bal_high_b},
        },
        sv_ratio=ratio_sv_base,
        valcurves=base_valcurves,
    )

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
