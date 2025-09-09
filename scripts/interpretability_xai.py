#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interprétabilité des modèles XGBoost sur PlantVillage avec SHAP et LIME.

- Sélection automatique du meilleur pipeline/config depuis
  results/models/xgboost/<target>/global_results.csv (si présent),
  sinon valeurs par défaut (pipeline=xgb, config=Baseline).
- Recontruit et entraîne le pipeline choisi puis produit:
  - SHAP summary (global) + par classe (optionnel) + heatmap d'impact
  - LIME explications individuelles (HTML) pour quelques échantillons

Sorties par défaut:
  results/models/xgboost/<target>/interpretability/

Exemples:
  python scripts/interpretability_xai.py --target nom_maladie --n-shap 1000 --n-lime 10
  python scripts/interpretability_xai.py --target nom_plante --pipeline xgb --config Baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline as SkPipeline

from xgboost import XGBClassifier

# Imports facultatifs (SHAP/LIME). Si absents, le script l'indique proprement.
try:
    import shap  # type: ignore
except Exception as e:
    shap = None  # type: ignore
    _shap_import_error = str(e)

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception as e:
    LimeTabularExplainer = None  # type: ignore
    _lime_import_error = str(e)

from src.helpers.helpers import PROJECT_ROOT

SEED = 42
DEFAULT_CSV = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"
DEFAULT_TARGET = "nom_maladie"

# Mappage des configs utilisées dans le script d'entraînement
XGB_CONFIGS_FULL: Dict[str, Dict[str, int | float]] = {
    "Baseline": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
    "Deep Trees": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 10},
    "Shallow Trees": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 3},
}
XGB_CONFIGS_QUICK: Dict[str, Dict[str, int | float]] = {
    "Quick": {"n_estimators": 80, "learning_rate": 0.15, "max_depth": 6},
}


def infer_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric_columns: List[str] = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = [
        "width", "height", "width_img", "height_img", "is_black", "is_na",
        "is_duplicate_after_first", "num_channels", "file_size",
    ]
    return [c for c in numeric_columns if c not in exclude_cols]


def load_best_from_results(base_dir: Path) -> Optional[Tuple[str, str]]:
    """Lit global_results.csv pour extraire (Pipeline, Config) de la meilleure ligne.
    Retourne None si indisponible.
    """
    csv_path = base_dir / "global_results.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        best = df.sort_values("Test_F1_weighted", ascending=False).iloc[0]
        pipeline = str(best.get("Pipeline", "")).strip()
        config = str(best.get("Config", "")).strip()
        if pipeline and config:
            return pipeline, config
    except Exception:
        return None
    return None


def build_pipeline(pipeline_name: str, params: Dict[str, int | float]) -> Tuple[SkPipeline, List[str]]:
    """Construit un pipeline sklearn pour le nom donné.
    Retourne (pipeline, feature_names) où feature_names sont les noms d'entrées du modèle final.
    """
    scaler = RobustScaler()
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1,
        **params,
    )
    if pipeline_name == "XGBoost" or pipeline_name == "xgb":
        pipe = SkPipeline([("scaler", scaler), ("xgb", xgb)])
        feat_names = []  # assignés dynamiquement après fit
    elif pipeline_name == "XGBoost + PCA" or pipeline_name == "xgb_pca":
        pipe = SkPipeline([("scaler", scaler), ("pca", PCA(random_state=SEED)), ("xgb", xgb)])
        feat_names = []
    elif pipeline_name == "XGBoost + LDA" or pipeline_name == "xgb_lda":
        pipe = SkPipeline([("scaler", scaler), ("lda", LDA()), ("xgb", xgb)])
        feat_names = []
    else:
        raise ValueError(f"Pipeline inconnu: {pipeline_name}")
    return pipe, feat_names


def compute_shap_summary(pipe: SkPipeline, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], outdir: Path, class_names: List[str], n_samples: int = 1000) -> None:
    if shap is None:
        print(f"[INFO] SHAP indisponible: {_shap_import_error}")
        return

    try:
        # Echantillonnage pour accélérer
        bg_size = min(200, X_train.shape[0])
        test_size = min(n_samples, X_test.shape[0])
        rng = np.random.RandomState(SEED)
        bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
        test_idx = rng.choice(X_test.shape[0], size=test_size, replace=False)
        X_bg = X_train[bg_idx]
        X_te = X_test[test_idx]

        # Récupérer le classifieur sous-jacent
        model = pipe.named_steps["xgb"]
        # On applique les étapes amont sur les jeux (sauf le classifieur)
        def transform_only(X):
            Xt = X
            for name, step in pipe.named_steps.items():
                if name == "xgb":
                    break
                Xt = step.transform(Xt)
            return Xt

        X_bg_trans = transform_only(X_bg)
        X_te_trans = transform_only(X_te)

        # Noms de features en entrée du modèle (après éventuelle réduction)
        if "pca" in pipe.named_steps:
            n_comp = pipe.named_steps["pca"].n_components_
            feature_names_final = [f"PCA_{i+1}" for i in range(n_comp)]
        elif "lda" in pipe.named_steps:
            n_comp = pipe.named_steps["lda"].scalings_.shape[1]
            feature_names_final = [f"LDA_{i+1}" for i in range(n_comp)]
        else:
            feature_names_final = feature_names

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te_trans)

        outdir.mkdir(parents=True, exist_ok=True)
        # Summary plot (global)
        plt.figure()
        shap.summary_plot(shap_values, X_te_trans, feature_names=feature_names_final, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary_all_classes.png", dpi=150)
        plt.close()

        # Heatmap d'impact moyen absolu
        try:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            if isinstance(mean_abs, list):  # multiclasses -> moyenne sur classes
                mean_abs = np.mean(np.stack(mean_abs, axis=0), axis=0)
            order = np.argsort(mean_abs)[::-1]
            topk = min(30, len(feature_names_final))
            plt.figure(figsize=(10, max(6, 0.35 * topk)))
            sns.barplot(x=mean_abs[order][:topk], y=np.array(feature_names_final)[order][:topk], palette="mako")
            plt.xlabel("Impact moyen absolu (|SHAP|)")
            plt.ylabel("Feature")
            plt.title("Top features par |SHAP| (moyenne)")
            plt.tight_layout()
            plt.savefig(outdir / "shap_feature_impact_heatmap.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] Heatmap SHAP ignorée: {e}")

    except Exception as e:
        print(f"[WARN] SHAP summary échoué: {e}")


def compute_lime_examples(pipe: SkPipeline, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], class_names: List[str], outdir: Path, n_examples: int = 10) -> None:
    if LimeTabularExplainer is None:
        print(f"[INFO] LIME indisponible: {_lime_import_error}")
        return

    try:
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
            verbose=False,
            random_state=SEED,
        )

        # Fonction de prédiction proba sur le pipeline complet
        def predict_proba(X):
            return pipe.predict_proba(X)

        rng = np.random.RandomState(SEED)
        idxs = rng.choice(X_test.shape[0], size=min(n_examples, X_test.shape[0]), replace=False)
        saved = []
        for i, idx in enumerate(idxs, start=1):
            exp = explainer.explain_instance(
                data_row=X_test[idx],
                predict_fn=predict_proba,
                num_features=min(15, len(feature_names)),
                top_labels=1,
            )
            html_path = outdir / f"lime_example_{i}.html"
            exp.save_to_file(str(html_path))
            saved.append(str(html_path.name))
        (outdir / "lime_examples_index.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] LIME échoué: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interprétabilité (SHAP + LIME) PlantVillage")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Chemin CSV")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Colonne cible")
    p.add_argument("--pipeline", type=str, default=None, help="Nom pipeline (par défaut: meilleur des résultats)")
    p.add_argument("--config", type=str, default=None, help="Nom config XGB (par défaut: meilleur des résultats)")
    p.add_argument("--test-size", type=float, default=0.2, help="Taille test split")
    p.add_argument("--n-shap", type=int, default=800, help="Nombre d'échantillons pour SHAP")
    p.add_argument("--n-lime", type=int, default=10, help="Nombre d'exemples LIME")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Colonne cible '{args.target}' absente. Colonnes: {list(df.columns)[:30]} ...")

    base_dir = PROJECT_ROOT / "results" / "models" / "xgboost" / str(args.target)
    xai_dir = base_dir / "interpretability"
    xai_dir.mkdir(parents=True, exist_ok=True)

    # Déterminer pipeline/config
    picked = load_best_from_results(base_dir)
    pipeline_name, config_name = None, None
    if picked is not None:
        pipeline_name, config_name = picked
    # Override par les args si fournis
    if args.pipeline:
        pipeline_name = args.pipeline
    if args.config:
        config_name = args.config
    if not pipeline_name:
        pipeline_name = "XGBoost"
    if not config_name:
        config_name = "Baseline"

    # Récupérer params XGB
    params = XGB_CONFIGS_FULL.get(config_name, None)
    if params is None:
        params = XGB_CONFIGS_QUICK.get(config_name, None)
    if params is None:
        raise ValueError(f"Config XGB inconnue: {config_name}")

    # Préparer données
    y_raw = df[args.target].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    X_cols = infer_numeric_features(df)
    X = df[X_cols].fillna(df[X_cols].median()).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=SEED
    )

    # Construire pipeline
    pipe, _ = build_pipeline(pipeline_name, params)
    pipe.fit(X_train, y_train)

    # SHAP
    compute_shap_summary(
        pipe=pipe,
        X_train=X_train,
        X_test=X_test,
        feature_names=X_cols,
        outdir=xai_dir,
        class_names=class_names,
        n_samples=args.n_shap,
    )

    # LIME (sur les features originales; LIME passe par le pipeline complet)
    compute_lime_examples(
        pipe=pipe,
        X_train=X_train,
        X_test=X_test,
        feature_names=X_cols,
        class_names=class_names,
        outdir=xai_dir,
        n_examples=args.n_lime,
    )

    # Petit résumé
    summary = {
        "pipeline": pipeline_name,
        "config": config_name,
        "n_classes": len(class_names),
        "n_features": len(X_cols),
        "xai_dir": str(xai_dir),
        "notes": [
            "Si SHAP/LIME ne sont pas installés, les sections correspondantes seront ignorées avec un message explicite.",
            "Les plots SHAP sur PCA/LDA reflètent les composantes (PCA_i / LDA_i).",
        ],
    }
    (xai_dir / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Avertir si dépendances manquantes
    if shap is None:
        print("[INFO] SHAP non disponible. Installez 'shap' pour activer les graphiques.")
    if LimeTabularExplainer is None:
        print("[INFO] LIME non disponible. Installez 'lime' pour activer les explications HTML.")

    print(f"\nInterprétabilité exportée dans: {xai_dir}")


if __name__ == "__main__":
    main()
