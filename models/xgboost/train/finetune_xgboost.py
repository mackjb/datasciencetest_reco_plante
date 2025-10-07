#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetuning XGBoost sur PlantVillage avec sorties par cible et par pipeline.

Pipelines supportés:
- xgb: RobustScaler -> XGBClassifier
- xgb_pca: RobustScaler -> PCA -> XGBClassifier
- xgb_lda: RobustScaler -> LDA -> XGBClassifier

Par défaut, les résultats sont écrits dans:
  results/models/xgboost/<target>/tuning/<pipeline>/

Exemples:
  python scripts/finetune_xgboost.py --target nom_maladie --quick
  python scripts/finetune_xgboost.py --target nom_plante --pipelines xgb,xgb_pca --cv-folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier

from src.helpers.helpers import PROJECT_ROOT
from src.config import load_config


SEED = 42
# Central config (optional)
_CFG = load_config()
_paths = _CFG.get("paths", {}) if isinstance(_CFG, dict) else {}

# Prefer YAML paths if present; fallback to previous default
DEFAULT_CSV = Path(
    _paths.get(
        "csv_clean_features",
        str(PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all_with_features.csv"),
    )
)
DEFAULT_TARGET = "nom_maladie"


def build_param_grid(quick: bool, pipeline_name: str, n_classes: int) -> Dict[str, List]:
    """Retourne une grille de paramètres adaptée au pipeline.

    Note: pour LDA, le nombre de composantes est borné par (n_classes-1).
    """
    # Base XGB params across pipelines
    if quick:
        xgb_grid = {
            "xgb__n_estimators": [150, 250],
            "xgb__max_depth": [4, 6],
            "xgb__learning_rate": [0.1, 0.05],
            "xgb__subsample": [0.9, 1.0],
            "xgb__colsample_bytree": [0.9, 1.0],
        }
    else:
        xgb_grid = {
            "xgb__n_estimators": [200, 350, 500],
            "xgb__max_depth": [4, 6, 8],
            "xgb__learning_rate": [0.15, 0.1, 0.05],
            "xgb__subsample": [0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.8, 0.9, 1.0],
        }

    if pipeline_name == "xgb":
        return xgb_grid

    if pipeline_name == "xgb_pca":
        # Composantes PCA bornées pour rester raisonnable
        pca_grid = {"pca__n_components": [20, 35, 50] if not quick else [20, 35]}
        return {**pca_grid, **xgb_grid}

    if pipeline_name == "xgb_lda":
        # LDA n_components <= n_classes-1, on propose quelques valeurs
        max_comp = max(1, min(60, n_classes - 1))
        candidates = [min(k, max_comp) for k in [10, 20, 30, 50] if min(k, max_comp) > 0]
        if quick:
            candidates = candidates[:2] or [max_comp]
        return {"lda__n_components": sorted(set(candidates)), **xgb_grid}

    raise ValueError(f"Pipeline inconnu: {pipeline_name}")


def build_pipeline(pipeline_name: str, seed: int) -> SkPipeline:
    # utiliser tree_method='hist' pour accélérer, eval_metric mlogloss
    xgb = XGBClassifier(
        random_state=seed,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )

    steps: List[Tuple[str, object]] = [("scaler", RobustScaler())]
    if pipeline_name == "xgb":
        steps += [("xgb", xgb)]
    elif pipeline_name == "xgb_pca":
        steps += [("pca", PCA(random_state=seed)), ("xgb", xgb)]
    elif pipeline_name == "xgb_lda":
        steps += [("lda", LDA()), ("xgb", xgb)]
    else:
        raise ValueError(f"Pipeline inconnu: {pipeline_name}")

    return SkPipeline(steps)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric_columns: List[str] = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = [
        "width",
        "height",
        "width_img",
        "height_img",
        "is_black",
        "is_na",
        "is_duplicate_after_first",
        "num_channels",
        "file_size",
    ]
    return [c for c in numeric_columns if c not in exclude_cols]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetuning XGBoost PlantVillage")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Chemin CSV")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Colonne cible (nom_maladie/nom_plante)")
    p.add_argument("--pipelines", type=str, default="xgb,xgb_pca,xgb_lda", help="Liste de pipelines séparés par des virgules")
    p.add_argument("--cv-folds", type=int, default=5, help="Nombre de folds pour la CV")
    p.add_argument("--scoring", type=str, default="f1_weighted", help="Métrique de scoring")
    p.add_argument("--quick", action="store_true", help="Grille réduite pour accélérer")
    p.add_argument("--n-jobs", type=int, default=-1, help="n_jobs pour GridSearchCV")
    p.add_argument("--seed", type=int, default=SEED, help="Seed aléatoire")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    target = args.target
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' absente. Colonnes: {list(df.columns)[:30]} ...")

    # Résolution dossier résultats (YAML paths.results_dir if available)
    results_root = Path(_paths.get("results_dir", PROJECT_ROOT / "results"))
    base_dir = results_root / "models" / "xgboost" / str(target) / "tuning"
    ensure_dir(base_dir)

    # Préparation données
    y_raw = df[target].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_.tolist()
    n_classes = len(classes)

    X_cols = infer_numeric_features(df)
    X = df[X_cols].fillna(df[X_cols].median()).values

    print(f"X shape: {X.shape} | y shape: {y.shape} | #features: {len(X_cols)} | #classes: {n_classes}")

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    # Pour le résumé global
    summary_rows = []

    for pipe_name in pipelines:
        print(f"\n=== Pipeline: {pipe_name} ===")
        pipe = build_pipeline(pipe_name, seed=args.seed)
        param_grid = build_param_grid(args.quick, pipe_name, n_classes)
        pipe_outdir = base_dir / pipe_name
        ensure_dir(pipe_outdir)

        # CV stratifiée
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=args.scoring,
            cv=cv,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
            return_train_score=True,
        )

        gs.fit(X, y)

        # Export des résultats bruts
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df.sort_values("rank_test_score", inplace=True)
        cv_df.to_csv(pipe_outdir / "cv_results.csv", index=False)

        # Best params + score
        best_params = gs.best_params_
        best_score = float(gs.best_score_)
        with open(pipe_outdir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "best_score": best_score, "scoring": args.scoring}, f, indent=2)

        # Résumé lisible
        summary_txt = (
            f"Pipeline: {pipe_name}\n"
            f"Scoring: {args.scoring}\n"
            f"Best score (CV mean): {best_score:.4f}\n"
            f"Best params: {json.dumps(best_params)}\n"
        )
        (pipe_outdir / "best_summary.txt").write_text(summary_txt, encoding="utf-8")

        # Figure: top-N configs
        try:
            topn = min(15, len(cv_df))
            top_df = cv_df.head(topn).copy()
            top_df["label"] = top_df.index.astype(str)
            plt.figure(figsize=(10, max(6, 0.5 * topn)))
            sns.barplot(data=top_df, x="mean_test_score", y="label", palette="viridis")
            plt.xlabel(args.scoring)
            plt.ylabel("Config (rang)")
            plt.title(f"Top {topn} - {pipe_name}")
            plt.tight_layout()
            plt.savefig(pipe_outdir / "top_configs.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] Impossible de générer le barplot top configs: {e}")

        summary_rows.append({
            "Pipeline": pipe_name,
            "Best_CV_Score": best_score,
            "Scoring": args.scoring,
            "Best_Params": json.dumps(best_params),
        })

    # Export résumé global
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("Best_CV_Score", ascending=False)
        summary_df.to_csv(base_dir / "summary.csv", index=False)
        print("\nRésumé tuning:\n", summary_df.head(10))

    print(f"\nTuning terminé. Résultats: {base_dir}")


if __name__ == "__main__":
    main()
