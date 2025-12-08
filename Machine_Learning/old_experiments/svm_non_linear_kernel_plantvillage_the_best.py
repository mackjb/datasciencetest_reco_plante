#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear SVM (LinearSVC) sur PlantVillage.

Plan exécuté par ce script:
1) Extraction dataset PlantVillage (CSV propre existant) + traitement NA/doublons
2) Split stratifié train/test
3) Pipeline: Suppression NA (en amont) -> RobustScaler -> LinearSVC (l2, squared_hinge)
4) CV 5-fold stratifiée avec métriques (balanced_accuracy, f1_macro)
5) Grid search simple uniquement sur l'hyperparamètre C
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
import re
from urllib.parse import quote
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
    # Fallback si le package src n'est pas importable
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Configuration & chemins
# -----------------------------
RANDOM_STATE: int = 42
CSV_PATH: Path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "models" / "svm_rbf_best"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL: str = "species"

# Réécriture optionnelle des chemins d'images (ex: dataset déplacé sous Windows)
# Si IMAGE_PREFIX_NEW est non vide, tout chemin commençant par IMAGE_PREFIX_OLD sera réécrit
# en remplaçant ce préfixe par IMAGE_PREFIX_NEW avant de produire une URI file://.
IMAGE_PREFIX_OLD: str = r"C:\\Users\\bgasm\\Documents\\Bernadette\\IA\\projet\\notebook_rendu_1\\version_7_new_features\\plantvillage dataset\\segmented"
IMAGE_PREFIX_NEW: str = r"C:\\Users\\bgasm\\repository\\datasciencetest_reco_plante\\dataset\\plantvillage\\data\\plantvillage dataset\\segmented"

# Helper de réécriture -> URI file:// utilisable dans les HTML
def to_file_uri_from_any(val: str) -> Optional[str]:
    if not isinstance(val, str) or not val:
        return None
    s = val
    # Déjà une URI file:// ?
    if s.startswith("file://"):
        return s
    # Réécriture avec préfixe fourni (Windows) — à faire AVANT de décider du traitement Windows absolu
    if IMAGE_PREFIX_NEW and IMAGE_PREFIX_OLD:
        s_norm = re.sub(r"[\\]+", "/", s).strip()
        old_norm = re.sub(r"[\\]+", "/", IMAGE_PREFIX_OLD).strip()
        if s_norm.lower().startswith(old_norm.lower()):
            rest = s_norm[len(old_norm):].lstrip("/")
            # Construire un chemin Windows propre
            new_win = str(PureWindowsPath(IMAGE_PREFIX_NEW) / rest.replace("/", "\\"))
            # Fabriquer une URI file:// sans valider l'existence (pour usage cross-OS)
            try:
                return Path(new_win).as_uri()
            except Exception:
                # Encode espaces et caractères spéciaux
                new_posix = new_win.replace("\\", "/")
                return "file:///" + quote(new_posix, safe="/:")
    # Chemin Windows absolu (ex: C:\... ou C:/... ou UNC \\server\share)
    if re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith("\\\\"):
        s_posix = s.replace("\\", "/")
        return "file:///" + quote(s_posix, safe="/:")
    # Fallback: relatif au projet ou absolu
    cand = s if os.path.isabs(s) else str((PROJECT_ROOT / s).resolve())
    try:
        return Path(cand).as_uri()
    except Exception:
        # Dernier recours: encoder en posix
        return "file:///" + quote(cand.replace("\\", "/"), safe="/:") if cand else None

# Colonnes non-features dans le CSV clean
NON_FEATURE_COLS: List[str] = [
    # Liste demandée (variables non-features)
    "species",
    "ID_Image",
    "Est_Saine",
    "Image_Path",
    "is_black",
    "plant_Apple",
    "plant_Blueberry",
    "plant_Cherry_(including_sour)",
    "plant_Corn_(maize)",
    "plant_Grape",
    "plant_Orange",
    "plant_Peach",
    "plant_Pepper,_bell",
    "plant_Potato",
    "plant_Raspberry",
    "plant_Soybean",
    "plant_Squash",
    "plant_Strawberry",
    "plant_Tomato",
    "disease_Apple_scab",
    "disease_Bacterial_spot",
    "disease_Black_rot",
    "disease_Cedar_apple_rust",
    "disease_Cercospora_leaf_spot Gray_leaf_spot",
    "disease_Common_rust_",
    "disease_Early_blight",
    "disease_Esca_(Black_Measles)",
    "disease_Haunglongbing_(Citrus_greening)",
    "disease_Late_blight",
    "disease_Leaf_Mold",
    "disease_Leaf_blight_(Isariopsis_Leaf_Spot)",
    "disease_Leaf_scorch",
    "disease_Northern_Leaf_Blight",
    "disease_Powdery_mildew",
    "disease_Septoria_leaf_spot",
    "disease_Spider_mites Two-spotted_spider_mite",
    "disease_Target_Spot",
    "disease_Tomato_Yellow_Leaf_Curl_Virus",
    "disease_Tomato_mosaic_virus",
    "disease_healthy",
    # Autres colonnes manifestement non-features à exclure si présentes
    "filepath","filename","extension","file_size","label","width","height","mode",
    "num_channels","aspect_ratio","is_image_valid","is_na","hash","is_duplicate_after_first","disease",
]


def print_plan() -> None:
    print("\nPlan d'exécution:")
    steps = [
        "1) Charger CSV propre + traiter NA/doublons",
        "2) Split stratifié train/test",
        "3) Pipeline: RobustScaler -> SVC(RBF) avec meilleurs hyperparamètres fixés",
        "4) Entraînement direct (pas de GridSearchCV)",
        "5) Évaluation test + sauvegarde des artefacts (rapports, matrices, galeries)",
        "6) Export JSON des hyperparamètres et métriques"
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
    # Harmoniser/ construire la cible species
    if "nom_plante" in df.columns and "species" not in df.columns:
        df = df.rename(columns={"nom_plante": "species"})
    if "species" not in df.columns:
        # Déduire species depuis les colonnes one-hot plant_*
        plant_cols = [c for c in df.columns if c.startswith("plant_")]
        if plant_cols:
            # argmax sur les colonnes plant_* (gérer bool/str)
            plant_mat = df[plant_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            winner_idx = plant_mat.argmax(axis=1)
            labels = []
            for j in winner_idx:
                col = plant_cols[int(j)] if len(plant_cols) else None
                if not col:
                    labels.append(None)
                    continue
                name = col[len("plant_"):]
                # Extraire le nom de la plante (avant '_', ',', '(')
                name = re.split(r"[_,(]", name)[0]
                labels.append(name)
            df["species"] = labels
    # Mapper Image_Path vers filepath si absent
    if "filepath" not in df.columns and "Image_Path" in df.columns:
        df["filepath"] = df["Image_Path"]
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

    # Supprimer la colonne ID_Image si présente (identifiant non pertinent pour l'apprentissage)
    if "ID_Image" in df.columns:
        df = df.drop(columns=["ID_Image"])  # évite toute fuite de cible/artefact

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

    # Exclure des lignes problématiques par index (observées dans la galerie d'erreurs)
    bad_idx = [3004, 13775, 20539, 24002, 23742, 23923, 35368, 35192, 35901, 35507]
    if len(bad_idx) > 0:
        before = len(df)
        present = [i for i in bad_idx if i in df.index]
        if present:
            df = df.drop(index=present)
            removed = before - len(df)
            if removed > 0:
                print(f"Lignes exclues via liste idx (nettoyage manuel): {removed}")

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
    """Pipeline fixé aux meilleurs hyperparamètres trouvés (RobustScaler + SVC RBF).

    Best params (fixed after cleaning): scaler=RobustScaler(), C=85.0, gamma='scale', class_weight=None, decision_function_shape='ovr'
    """
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=85.0,
                gamma="scale",
                class_weight=None,
                decision_function_shape="ovr",
                random_state=random_state,
            ),
        ),
    ])
    return pipe


def evaluate_on_test(best_estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print("\nÉvaluation finale sur test...")
    y_pred = best_estimator.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=best_estimator.named_steps["svm"].classes_)

    # Sauvegarde
    (RESULTS_DIR / "evaluation").mkdir(exist_ok=True)
    with open(RESULTS_DIR / "evaluation" / "classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(RESULTS_DIR / "evaluation" / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"balanced_accuracy (test): {bal_acc:.4f}")
    print(f"f1_macro (test): {f1m:.4f}")
    return {"balanced_accuracy": bal_acc, "f1_macro": f1m, "report": report, "confusion_matrix": cm}


def save_feature_ranking(feature_names: List[str], best_estimator: Pipeline) -> Optional[Path]:
    """Exporte un feature_ranking.csv si des coefficients linéaires sont disponibles.
    Pour les noyaux non-linéaires (ex: RBF SVC), retourne None.
    """
    svm = best_estimator.named_steps.get("svm")
    if svm is None or not hasattr(svm, "coef_") or getattr(svm, "coef_", None) is None:
        print("Avertissement: pas de coefficients disponibles (modèle non-linéaire). Saut du feature ranking.")
        return None

    coefs = svm.coef_  # (n_classes, n_features) en multi-classe; (1, n_features) en binaire
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    importance = np.mean(np.abs(coefs), axis=0)

    if len(feature_names) != importance.size:
        raise ValueError(f"Taille incohérente: {len(feature_names)} features vs {importance.size} coefficients")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    out_path = RESULTS_DIR / "feature_ranking.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_columns_list(columns: List[str]) -> Tuple[Path, Path]:
    """Enregistre la liste des colonnes du CSV clean dans results (TXT + JSON)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / "columns.txt"
    json_path = RESULTS_DIR / "columns.json"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(columns))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=False, indent=2)
    return txt_path, json_path


def save_feature_list(features: List[str]) -> Tuple[Path, Path]:
    """Enregistre la liste des features utilisées pour l'entraînement (TXT + JSON)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / "features.txt"
    json_path = RESULTS_DIR / "features.json"
    with open(txt_path, "w", encoding="utf-8") as f:
        for c in features:
            f.write(str(c) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    return txt_path, json_path


def save_misclassified_examples(df_full: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
                                max_per_pair: int = 3) -> Tuple[Path, Path]:
    """Enregistre un CSV des erreurs de classification et une galerie HTML d'exemples.
    - df_full: DataFrame complet (pour récupérer les colonnes comme 'filepath')
    - X_test: DataFrame test (pour ses index qui pointent vers df_full)
    - y_test: vraies étiquettes
    - y_pred: étiquettes prédites
    Retourne (csv_path, html_path).
    """
    eval_dir = RESULTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    idx = X_test.index
    # Conserver l'index d'origine pour le nettoyage ultérieur
    df_err = pd.DataFrame({
        "index": idx,  # identifiant de ligne d'origine dans df_full
        "true": y_test.values,
        "pred": y_pred,
    }, index=idx)

    # Ajout de colonnes de contexte si disponibles
    for col in ["filepath", "filename", "label", "species", "ID_Image"]:
        if col in df_full.columns:
            df_err[col] = df_full.loc[idx, col].values

    df_err = df_err[df_err["true"] != df_err["pred"]]
    csv_path = eval_dir / "misclassified.csv"
    df_err.to_csv(csv_path, index=False)

    # Création d'une petite galerie HTML (quelques exemples par (true,pred))
    html_path = eval_dir / "misclassified_examples.html"
    lines = [
        "<html><head><meta charset='utf-8'><title>Erreurs de classification</title>",
        "<style>body{font-family:sans-serif} .pair{margin:16px 0} .row{display:flex;flex-wrap:wrap;gap:8px} .card{width:200px} img{max-width:200px;border:1px solid #ccc}</style>",
        "</head><body>",
        "<h1>Exemples d'erreurs (true -> pred)</h1>",
    ]
    if not df_err.empty:
        for (t, p), group in df_err.groupby(["true", "pred"]):
            lines.append(f"<div class='pair'><h3>{t} → {p}</h3><div class='row'>")
            sample = group.head(max_per_pair)
            for _, row in sample.iterrows():
                # Construire une légende riche avec index + ID_Image si dispo
                idx_val = row.get("index", None)
                id_img = row.get("ID_Image", None)
                extra = []
                if pd.notna(idx_val):
                    try:
                        extra.append(f"idx={int(idx_val)}")
                    except Exception:
                        extra.append(f"idx={idx_val}")
                if pd.notna(id_img) if "ID_Image" in row.index else False:
                    try:
                        extra.append(f"ID_Image={int(id_img)}")
                    except Exception:
                        extra.append(f"ID_Image={id_img}")
                extra_txt = (" [" + ", ".join(extra) + "]") if extra else ""
                caption = f"true={row['true']} pred={row['pred']}{extra_txt}"
                img_src = None
                if "filepath" in row and pd.notna(row["filepath"]):
                    img_src = to_file_uri_from_any(str(row["filepath"]))
                elif "Image_Path" in row and pd.notna(row["Image_Path"]):
                    img_src = to_file_uri_from_any(str(row["Image_Path"]))
                elif "filename" in row and pd.notna(row["filename"]):
                    img_src = to_file_uri_from_any(str(row["filename"]))
                card = [
                    "<div class='card'>",
                    f"<div><img src='{img_src if img_src else ''}' alt='img'></div>",
                    f"<div style='font-size:12px;color:#555'>{caption}</div>",
                ]
                if "filename" in row and pd.notna(row["filename"]):
                    card.append(f"<div style='font-size:11px;color:#777'>file={row['filename']}</div>")
                lines.extend(card)
            lines.append("</div></div>")
    else:
        lines.append("<p>Aucune erreur dans le set de test.</p>")

    lines.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return csv_path, html_path


def plot_c_influence(cv_df: pd.DataFrame) -> Path:
    """Génère un graphique Plotly montrant l'influence de C sur bal_acc et f1_macro (moyennes CV)."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = cv_df.assign(C=cv_df["param_svm__C"].astype(float)).sort_values("C")

    fig = px.line(
        df,
        x="C",
        y=["mean_test_bal_acc", "mean_test_f1_macro"],
        markers=True,
        title="Influence de C sur les performances (CV 5-fold)",
        labels={
            "C": "C (log)",
            "value": "Score",
            "variable": "Métrique",
        },
    )
    fig.update_layout(xaxis_type="log", legend_title_text="")
    outfile = plots_dir / "c_influence.html"
    pio.write_html(fig, str(outfile), auto_open=False)
    return outfile


def build_error_table_with_margins(
    df_full: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    feature_names: List[str],
    topn: int = 5,
) -> Tuple[pd.DataFrame, Path]:
    """Construit un tableau des erreurs avec marge de décision et features contributives.
    - marge = score(pred) - score(true) d'après decision_function (One-vs-Rest)
    - contributions = z * w_pred (z = features standardisés, w_pred = coefficients classe prédite)
    Retourne le DataFrame et le chemin CSV sauvegardé.
    """
    # Scores de décision (via pipeline complet)
    scores: np.ndarray = pipe.decision_function(X_test)  # shape: [n, n_classes]
    # Features standardisées pour contributions (si applicable)
    if "scaler" in pipe.named_steps:
        z = pipe.named_steps["scaler"].transform(X_test)
    else:
        z = X_test.values
    svm = pipe.named_steps.get("svm", None)
    if svm is None:
        raise RuntimeError("Le pipeline ne contient pas de step 'svm'.")
    classes = svm.classes_.tolist()
    has_coef = hasattr(svm, "coef_") and getattr(svm, "coef_") is not None
    coef = getattr(svm, "coef_", None)

    # Construire lignes
    rows = []
    idxs = X_test.index.tolist()
    for r, idx in enumerate(idxs):
        true_lbl = y_test.iloc[r]
        pred_lbl = y_pred[r]
        if true_lbl == pred_lbl:
            continue
        # indices de classes
        try:
            ip = classes.index(pred_lbl)
            it = classes.index(true_lbl)
        except ValueError:
            # au cas où labels_sorted diffère
            continue
        score_pred = float(scores[r, ip]) if scores.ndim == 2 else float(scores[r])
        score_true = float(scores[r, it]) if scores.ndim == 2 else float(scores[r])
        # marge basée sur decision_function
        margin = score_pred - score_true

        # contributions topn vers la classe prédite (si coef_ disponible, e.g., LinearSVC)
        if has_coef and coef is not None:
            contrib = z[r, :] * coef[ip, :]
            top_idx = np.argsort(contrib)[::-1][:topn]
            top_feats = [(feature_names[i] if i < len(feature_names) else f"f{i}", float(contrib[i])) for i in top_idx]
        else:
            top_feats = []

        # chemin image
        img_path = None
        for col in ["filepath", "filename"]:
            if col in df_full.columns:
                val = df_full.loc[idx, col]
                if isinstance(val, str):
                    img_path = to_file_uri_from_any(val)
                    break

        rows.append({
            "index": idx,
            "true": true_lbl,
            "pred": pred_lbl,
            "score_pred": score_pred,
            "score_true": score_true,
            "margin": margin,
            "image_path": img_path,
            "top_features": json.dumps(top_feats, ensure_ascii=False),
        })

    df_errors = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "evaluation" / "errors_with_margins.csv"
    df_errors.to_csv(out_csv, index=False)
    return df_errors, out_csv


def plot_confusion_matrix_gallery(
    errors_df: pd.DataFrame,
    class_names: List[str],
    k: int = 6,
    title: str = "Matrice + Galerie d'erreurs",
) -> Path:
    """HTML interactif: clic sur une cellule (true, pred) => affiche jusqu'à k images d'erreurs
    avec leurs top features contributives."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Construire cm à partir d'errors_df si besoin
    pairs = errors_df[["true", "pred"]].value_counts().reset_index(name="count")
    # construire matrice ordonnée
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for _, row in pairs.iterrows():
        t, p, c = row["true"], row["pred"], int(row["count"])
        if t in name_to_idx and p in name_to_idx:
            cm[name_to_idx[t], name_to_idx[p]] = c

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Reds",
        labels=dict(x="Prédit", y="Vrai", color="Erreurs"),
        title=title,
        x=class_names,
        y=class_names,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=900)

    # Préparer mapping (i,j) -> liste d'objets {path, top_features}
    bucket: Dict[str, List[Dict[str, object]]] = {}
    for _, r in errors_df.iterrows():
        t, p = r["true"], r["pred"]
        if t not in name_to_idx or p not in name_to_idx:
            continue
        i, j = name_to_idx[t], name_to_idx[p]
        key = f"{i},{j}"
        entry = {
            "path": r.get("image_path"),
            "margin": float(r.get("margin", 0.0)) if pd.notna(r.get("margin")) else None,
            "top_features": json.loads(r.get("top_features", "[]")),
            "true": t,
            "pred": p,
        }
        bucket.setdefault(key, []).append(entry)

    base_div = fig.to_html(full_html=False, include_plotlyjs='inline')

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <title>Galerie erreurs</title>\n'
        "  <style>\n"
        "    body { font-family: sans-serif; }\n"
        "    .container { display: grid; grid-template-columns: 1fr; gap: 16px; }\n"
        "    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }\n"
        "    .card { border: 1px solid #ccc; padding: 8px; }\n"
        "    img { max-width: 100%; border: 1px solid #ddd; }\n"
        "    .feat { font-size: 12px; color: #333; margin-top: 6px; }\n"
        "    .muted { color: #777; font-size: 12px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{title}</h1>\n"
        "  <div class=\"container\">\n"
        f"    <div id=\"cm_div\">{base_div}</div>\n"
        "    <div id=\"info\" class=\"muted\">Cliquez une case hors-diagonale pour afficher jusqu'à k images.</div>\n"
        "    <div id=\"gallery\" class=\"gallery\"></div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const bucket = {json.dumps(bucket)};\n"
        f"    const labels = {json.dumps(class_names)};\n"
        f"    const K = {int(k)};\n"
        "    function renderCards(items){\n"
        "      const gal = document.getElementById('gallery');\n"
        "      gal.innerHTML = '';\n"
        "      items.slice(0, K).forEach(it => {\n"
        "        const card = document.createElement('div'); card.className='card';\n"
        "        const h = document.createElement('div'); h.textContent = `true=${it.true} | pred=${it.pred} | margin=${(it.margin??0).toFixed(3)}`; card.appendChild(h);\n"
        "        if (it.path) { const img = document.createElement('img'); img.src = it.path; card.appendChild(img); }\n"
        "        const ul = document.createElement('ul'); ul.className='feat';\n"
        "        (it.top_features||[]).forEach(([name, val]) => { const li=document.createElement('li'); li.textContent = `${name}: ${val.toFixed(4)}`; ul.appendChild(li); });\n"
        "        card.appendChild(ul);\n"
        "        gal.appendChild(card);\n"
        "      });\n"
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
        "        renderCards(bucket[key] || []);\n"
        "      });\n"
        "      return true;\n"
        "    }\n"
        "    function tryBind(r){ if (bind()) return; if (r<=0) return; setTimeout(function(){tryBind(r-1);}, 200);}\n"
        "    window.addEventListener('load', function(){ tryBind(25); });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    outfile = plots_dir / "confusion_matrix_gallery.html"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile

def plot_confusion_matrix_interactive(
    df_full: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    max_examples_search: int = 20,
) -> Path:
    """Génère une matrice de confusion interactive avec, sur clic d'une case (i,j),
    l'affichage d'un couple d'images:
      - à gauche: une image du test mal classée (true=i, pred=j)
      - à droite: une image d'entraînement correcte appartenant à la classe prédite (j)

    Note: l'affichage d'images locales dépend du navigateur (file://). Préférez des chemins absolus valides.
    """
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Prépare le Heatmap (mêmes valeurs que la confusion calculée)
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    labels_sorted = class_names if class_names is not None else sorted(pd.unique(y_test))
    cm = sk_confusion_matrix(y_test, y_pred, labels=labels_sorted)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Vrai", color="Comptes"),
        title="Matrice de confusion interactive (clic pour voir des exemples)",
        x=labels_sorted,
        y=labels_sorted,
    )
    fig.update_xaxes(side="top")
    # Taille explicite pour éviter un rendu vide (hauteur 600px)
    fig.update_layout(height=600, width=900, margin=dict(l=80, r=20, t=60, b=80))

    # Construire un mapping (true, pred) -> {test_example_path, train_exemplar_path}
    # Exemple test: une ligne de X_test où true!=pred, filtrée sur le couple (t,p)
    # Exemplar train: une ligne de X_train avec y_train==p
    test_idx = X_test.index
    train_idx = X_train.index

    # Indexer rapidement par classe
    train_by_class = {c: train_idx[y_train.loc[train_idx] == c] for c in pd.unique(y_train)}

    # DataFrame d'erreurs
    err_mask = (y_test.values != y_pred)
    df_err = pd.DataFrame({
        "idx": test_idx,
        "true": y_test.values,
        "pred": y_pred,
    })
    df_err = df_err[err_mask]

    # Helper pour piocher un chemin image depuis df_full
    def pick_image_path(row_index: int) -> Optional[str]:
        for col in ["filepath", "filename"]:
            if col in df_full.columns:
                val = df_full.loc[row_index, col]
                if isinstance(val, str):
                    return to_file_uri_from_any(val)
        return None

    # Pour chaque cellule, prélever au plus un exemple (limiter la taille des données embarquées)
    example_map = {}
    # Limiter la recherche pour performance
    grouped = df_err.groupby(["true", "pred"]) if not df_err.empty else []
    for (t, p), group in grouped:
        row = group.head(max_examples_search).head(1)
        test_path = None
        if not row.empty:
            test_idx_one = int(row.iloc[0]["idx"])
            test_path = pick_image_path(test_idx_one)

        # Exemplar côté prédiction (depuis le train)
        exemplar_path = None
        cand_idx = train_by_class.get(p, [])
        if len(cand_idx) > 0:
            # prendre le premier valable
            for ridx in list(cand_idx)[:max_examples_search]:
                exemplar_path = pick_image_path(int(ridx))
                if exemplar_path is not None:
                    break

        example_map[(t, p)] = {
            "test": test_path,
            "exemplar": exemplar_path,
        }

    # Exporter la figure en HTML et injecter un script Plotly pour gérer les clics
    base_div = fig.to_html(full_html=False, include_plotlyjs='inline')

    # Construire un dictionnaire indexé par coordonnées matricielles (i,j)
    # i -> index de t dans labels_sorted; j -> index de p dans labels_sorted
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
        '  <title>Matrice de confusion interactive</title>\n'
        ""
        "  <style>\n"
        "    body { font-family: sans-serif; }\n"
        "    .container { display: grid; grid-template-columns: 1fr 320px 320px; gap: 16px; align-items: start; }\n"
        "    .panel { border: 1px solid #ccc; padding: 8px; }\n"
        "    img { max-width: 300px; border: 1px solid #ddd; }\n"
        "    .caption { font-size: 12px; color: #555; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Matrice de confusion interactive</h1>\n"
        "  <p>Cliquez sur une case hors-diagonale pour voir: à gauche, un exemple test mal classé; à droite, un exemple d'entraînement de la classe prédite.</p>\n"
        "  <div class=\"container\">\n"
        f"    <div id=\"cm_div\" class=\"panel\" style=\"min-height: 620px;\">{base_div}</div>\n"
        "    <div class=\"panel\">\n"
        "      <h3>Exemple test (vrai)</h3>\n"
        '      <img id="img_test" src="" alt="exemple test" />\n'
        '      <div id="cap_test" class="caption"></div>\n'
        "    </div>\n"
        "    <div class=\"panel\">\n"
        "      <h3>Exemple classe prédite</h3>\n"
        '      <img id="img_pred" src="" alt="exemple prédit" />\n'
        '      <div id="cap_pred" class="caption"></div>\n'
        "    </div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const coordMap = {json.dumps(coord_map)};\n"
        f"    const labels = {json.dumps(labels_sorted)};\n"
        "    function handleEvent(data) {\n"
        "      if (!data || !data.points || !data.points.length) return;\n"
        "      const pt = data.points[0];\n"
        "      let i = pt.y; // may be label or index\n"
        "      let j = pt.x; // may be label or index\n"
        "      if (typeof i === 'string') { i = labels.indexOf(i); }\n"
        "      if (typeof j === 'string') { j = labels.indexOf(j); }\n"
        "      const key = `${i},${j}`;\n"
        "      const payload = coordMap[key];\n"
        "      const trueLabel = labels[i];\n"
        "      const predLabel = labels[j];\n"
        "\n"
        "      const imgTest = document.getElementById('img_test');\n"
        "      const imgPred = document.getElementById('img_pred');\n"
        "      const capTest = document.getElementById('cap_test');\n"
        "      const capPred = document.getElementById('cap_pred');\n"
        "\n"
        "      if (payload && payload.test) {\n"
        "        imgTest.src = payload.test;\n"
        "        capTest.textContent = `True = ${trueLabel}`;\n"
        "      } else {\n"
        "        imgTest.src = '';\n"
        "        capTest.textContent = 'Aucun exemple test disponible';\n"
        "      }\n"
        "      if (payload && payload.exemplar) {\n"
        "        imgPred.src = payload.exemplar;\n"
        "        capPred.textContent = `Exemplar (classe prédite) = ${predLabel}`;\n"
        "      } else {\n"
        "        imgPred.src = '';\n"
        "        capPred.textContent = 'Aucun exemple prédiction disponible';\n"
        "      }\n"
        "    }\n"
        "    function bindHandlers() {\n"
        "      let cmDiv = document.querySelector('#cm_div .js-plotly-plot') ||\n"
        "                  document.querySelector('#cm_div .plotly-graph-div') ||\n"
        "                  (document.getElementById('cm_div') ? document.getElementById('cm_div').children[0] : null);\n"
        "      if (!cmDiv || !cmDiv.on) return false;\n"
        "      cmDiv.on('plotly_click', handleEvent);\n"
        "      cmDiv.on('plotly_hover', handleEvent);\n"
        "      return true;\n"
        "    }\n"
        "    function tryBind(retries) {\n"
        "      if (bindHandlers()) return;\n"
        "      if (retries <= 0) return;\n"
        "      setTimeout(function(){ tryBind(retries-1); }, 200);\n"
        "    }\n"
        "    window.addEventListener('load', function(){ tryBind(25); });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    outfile = plots_dir / "confusion_matrix_interactive.html"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile

def plot_c_influence_from_csv(grid_csv_path: Path) -> Path:
    """Recharge gridsearch_results.csv et génère la même figure que plot_c_influence."""
    df = pd.read_csv(grid_csv_path)
    return plot_c_influence(df)


def plot_confusion_matrix_from_csv(cm_csv_path: Path, class_names: Optional[List[str]] = None) -> Path:
    """Affiche la matrice de confusion depuis CSV en heatmap Plotly."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = pd.read_csv(cm_csv_path, header=None).values
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Vrai", color="Comptes"),
        title="Matrice de confusion (test)",
    )
    if class_names is not None:
        tickvals = list(range(len(class_names)))
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=class_names)
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=class_names)
    outfile = plots_dir / "confusion_matrix.html"
    pio.write_html(fig, str(outfile), auto_open=False)
    return outfile


def plot_feature_ranking(feature_csv_path: Path) -> Path:
    """Bar chart des features à partir d'un feature_ranking.csv.
    Attend des colonnes ['feature', 'importance'] ou ['feature', 'score'/'weight'/'coef'].
    Lève des exceptions si le fichier est manquant ou mal formé.
    """
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"Feature ranking introuvable: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)
    if "feature" not in df.columns:
        raise ValueError(f"Colonne 'feature' absente dans {feature_csv_path}")
    y_col = None
    for cand in ["importance", "score", "weight", "coef"]:
        if cand in df.columns:
            y_col = cand
            break
    if y_col is None:
        raise ValueError(f"Aucune colonne d'importance trouvée dans {feature_csv_path}")

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    topk = df.sort_values(y_col, ascending=False).head(30)
    fig = px.bar(
        topk,
        x="feature",
        y=y_col,
        title=f"Top 30 features par {y_col}",
        labels={"feature": "Feature", y_col: y_col},
    )
    fig.update_layout(xaxis_tickangle=-45)
    outfile = plots_dir / "feature_ranking_top30.html"
    pio.write_html(fig, str(outfile), auto_open=False)
    return outfile


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print_plan()

    t0 = time.time()
    df, X, y, feature_names = load_clean_dataset(CSV_PATH)
    cols_txt, cols_json = save_columns_list(list(df.columns))
    print(f"Colonnes sauvegardées: {cols_txt} et {cols_json}")
    feat_txt, feat_json = save_feature_list(feature_names)
    print(f"Features sauvegardées: {feat_txt} et {feat_json}")

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Modèle fixé aux meilleurs hyperparamètres connus
    pipe = build_pipeline(RANDOM_STATE)
    best_pipe: Pipeline = pipe.fit(X_train, y_train)

    # Évaluation sur test
    test_metrics = evaluate_on_test(best_pipe, X_test, y_test)
    # Sauvegarde des erreurs de classification (CSV + galerie HTML)
    y_pred = best_pipe.predict(X_test)
    mis_csv, mis_html = save_misclassified_examples(df, X_test, y_test, y_pred, max_per_pair=3)
    print(f"Erreurs sauvegardées: {mis_csv} | Galerie: {mis_html}")
    # Tableau des erreurs avec marges + top features contributives
    errors_df, errors_csv = build_error_table_with_margins(df, X_test, y_test, y_pred, best_pipe, feature_names, topn=6)
    print(f"Tableau d'erreurs avec marges sauvegardé: {errors_csv}")
    # Visualisation matrice de confusion depuis CSV
    class_names = sorted(pd.unique(y))
    cm_html = plot_confusion_matrix_from_csv(RESULTS_DIR / "evaluation" / "confusion_matrix.csv", class_names)
    print(f"Matrice de confusion Plotly sauvegardée: {cm_html}")
    # Matrice de confusion interactive (hover/click -> exemples d'images)
    cm_inter_html = plot_confusion_matrix_interactive(df, X_train, y_train, X_test, y_test, y_pred, class_names)
    print(f"Matrice de confusion interactive sauvegardée: {cm_inter_html}")

    # Nouvelle galerie interactive: clic sur cellule => k images + top features
    gallery_html = plot_confusion_matrix_gallery(errors_df, class_names, k=8, title="Galerie erreurs (k=8)")
    print(f"Galerie d'erreurs interactive sauvegardée: {gallery_html}")

    # Génération et visualisation du feature ranking (si dispo pour modèles linéaires)
    fr_csv = save_feature_ranking(feature_names, best_pipe)
    fr_html = None
    if fr_csv is not None:
        fr_html = plot_feature_ranking(fr_csv)
        if fr_html is not None:
            print(f"Feature ranking Plotly sauvegardé: {fr_html}")

    # Export JSON des hyperparamètres et métriques
    export = {
        "model": "SVC",
        "scaler": "RobustScaler",
        "params": {
            "kernel": "rbf",
            "C": 85.0,
            "gamma": "scale",
            "class_weight": None,
            "decision_function_shape": "ovr",
            "random_state": RANDOM_STATE,
        },
        "test_metrics": {
            "balanced_accuracy": float(test_metrics.get("balanced_accuracy", np.nan)),
            "f1_macro": float(test_metrics.get("f1_macro", np.nan)),
        },
        "timestamp": time.time(),
    }
    with open(RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"\nTerminé en {time.time() - t0:.2f}s. Résultats: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
