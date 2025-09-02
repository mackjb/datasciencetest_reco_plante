#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualisation des résultats globaux (global_results.csv) produits par
models/xgboost_plantvillage.py

Génère plusieurs figures pour faciliter l'analyse:
- Top N configurations (barplot du Test_F1_weighted)
- Moyenne par pipeline (barplot)
- CV_F1_mean vs Test_F1_weighted (scatter + ligne y=x)
- Export d'un top résumé en HTML

Usage:
    python scripts/visualize_global_results.py \
        --csv results/models/xgboost/global_results.csv \
        --outdir results/models/xgboost/vis \
        --topn 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualisation de global_results.csv")
    p.add_argument(
        "--csv",
        type=str,
        default="results/models/xgboost/global_results.csv",
        help="Chemin du CSV des résultats globaux",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="results/models/xgboost/vis",
        help="Dossier de sortie pour les figures",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Nombre de meilleures lignes à afficher dans le barplot top N",
    )
    return p.parse_args()


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def barplot_topn(df: pd.DataFrame, outdir: Path, topn: int = 10) -> None:
    cols_required = {"Pipeline", "Config", "Test_F1_weighted"}
    if not cols_required.issubset(df.columns):
        print(f"[WARN] Colonnes manquantes pour barplot_topn: {cols_required - set(df.columns)}")
        return

    top_df = (
        df.sort_values("Test_F1_weighted", ascending=False)
          .head(topn)
          .copy()
    )
    # Ajoute une étiquette compacte pour l'axe Y
    top_df["Label"] = top_df["Pipeline"] + " | " + top_df["Config"]

    plt.figure(figsize=(10, max(6, 0.5 * len(top_df))))
    sns.barplot(
        data=top_df,
        x="Test_F1_weighted",
        y="Label",
        hue="Pipeline",
        dodge=False,
        palette="viridis",
    )
    plt.xlim(0, 1)
    plt.xlabel("Test F1 (weighted)")
    plt.ylabel("Pipeline | Config")
    plt.title(f"Top {len(top_df)} configurations par Test F1 (weighted)")
    plt.tight_layout()
    plt.savefig(outdir / "barplot_topN_test_f1_weighted.png", dpi=150)
    plt.close()


def barplot_mean_by_pipeline(df: pd.DataFrame, outdir: Path) -> None:
    if not {"Pipeline", "Test_F1_weighted"}.issubset(df.columns):
        print("[WARN] Colonnes manquantes pour barplot_mean_by_pipeline")
        return

    mean_df = (
        df.groupby("Pipeline", as_index=False)["Test_F1_weighted"].mean()
          .sort_values("Test_F1_weighted", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=mean_df, x="Test_F1_weighted", y="Pipeline", palette="mako")
    plt.xlim(0, 1)
    plt.xlabel("Moyenne Test F1 (weighted)")
    plt.ylabel("Pipeline")
    plt.title("Performance moyenne par pipeline")
    plt.tight_layout()
    plt.savefig(outdir / "barplot_mean_by_pipeline.png", dpi=150)
    plt.close()


def scatter_cv_vs_test(df: pd.DataFrame, outdir: Path) -> None:
    cols_required = {"Pipeline", "Config", "CV_F1_mean", "CV_F1_std", "Test_F1_weighted"}
    if not cols_required.issubset(df.columns):
        print(f"[WARN] Colonnes manquantes pour scatter_cv_vs_test: {cols_required - set(df.columns)}")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="CV_F1_mean",
        y="Test_F1_weighted",
        hue="Pipeline",
        style="Config",
        s=120,
        palette="tab10",
    )
    # Ligne de référence y = x
    lim_min = 0.0
    lim_max = max(df[["CV_F1_mean", "Test_F1_weighted"]].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4)
    plt.xlim(lim_min, 1.0)
    plt.ylim(lim_min, 1.0)
    plt.xlabel("CV F1 (mean, weighted)")
    plt.ylabel("Test F1 (weighted)")
    plt.title("Relation CV vs Test (F1 weighted)")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_cv_vs_test.png", dpi=150)
    plt.close()


def export_top_html(df: pd.DataFrame, outdir: Path, topn: int = 10) -> None:
    top_df = df.sort_values("Test_F1_weighted", ascending=False).head(topn)
    html_path = outdir / "top_results.html"
    top_df.to_html(html_path, index=False)


def export_full_tables_html(df: pd.DataFrame, outdir: Path) -> None:
    """Exporte le tableau complet en HTML (brut et trié par Test_F1_weighted)."""
    full_path = outdir / "full_results.html"
    sorted_path = outdir / "full_results_sorted.html"
    try:
        # Export brut
        df.to_html(full_path, index=False)

        # Export trié + explications sous le tableau
        sorted_df = df.sort_values("Test_F1_weighted", ascending=False)
        table_html = sorted_df.to_html(index=False)

        explanations = {
            "Pipeline": (
                "Chaîne de traitement utilisée (ex: XGBoost seul, XGBoost + PCA, XGBoost + LDA). "
                "Intérêt: indique quelles étapes amont (réduction de dimension) aident vraiment le modèle."
            ),
            "Config": (
                "Réglages du modèle (ex: nombre d'arbres, vitesse d'apprentissage, profondeur). "
                "Intérêt: permet de comparer les variantes et choisir le compromis performance/complexité."
            ),
            "CV_F1_mean": (
                "Moyenne du F1 (pondéré) durant la validation croisée, calculée sur les données d'entraînement. "
                "Intérêt: estime la performance attendue en général sans toucher au test (limite le surapprentissage)."
            ),
            "CV_F1_std": (
                "Écart-type du F1 en CV. Plus petit = plus stable entre les folds. "
                "Intérêt: évalue la robustesse; une faible variabilité est souvent préférable."
            ),
            "Test_Accuracy": (
                "Part des prédictions correctes sur le jeu de test. "
                "Intérêt: simple à comprendre, mais peut sous-pondérer les classes rares."
            ),
            "Test_F1_macro": (
                "Moyenne du F1 par classe sans pondération. Chaque classe compte autant. "
                "Intérêt: utile si l'équilibre entre classes est important (évite qu'une classe majoritaire domine)."
            ),
            "Test_F1_weighted": (
                "Moyenne du F1 pondérée par la fréquence de chaque classe. "
                "Intérêt: reflète la performance globale en tenant compte des classes fréquentes (score principal pour le tri)."
            ),
        }

        doc_items = "".join(
            f"<li><b>{col}</b>: {desc}</li>" for col, desc in explanations.items()
            if col in df.columns
        )

        html = (
            "<html><head><meta charset='utf-8'><title>Global Results (Sorted)</title>"
            "<style>body{font-family:Arial, sans-serif; margin:24px;} table{border-collapse:collapse;}"
            "table, th, td{border:1px solid #ddd; padding:6px;} th{background:#f5f5f5;}"
            "h2{margin-top:24px;} p{margin:0 0 6px 0; color:#444;} </style></head><body>"
            "<h1>Global Results (sorted by Test_F1_weighted desc)</h1>"
            f"{table_html}"
            "<h2>Explications des colonnes</h2>"
            f"<ul>{doc_items}</ul>"
            "</body></html>"
        )

        sorted_path.write_text(html, encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Export HTML complet échoué: {e}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    topn = int(args.topn)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    ensure_outdir(outdir)

    # Lecture des résultats
    df = pd.read_csv(csv_path)
    # Casting prudent pour éviter des surprises
    for c in ["CV_F1_mean", "CV_F1_std", "Test_Accuracy", "Test_F1_macro", "Test_F1_weighted"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Thème
    sns.set_theme(style="whitegrid")

    # Figures
    barplot_topn(df, outdir, topn=topn)
    barplot_mean_by_pipeline(df, outdir)
    scatter_cv_vs_test(df, outdir)

    # Export HTML top N
    export_top_html(df, outdir, topn=topn)
    # Export HTML complet (brut et trié)
    export_full_tables_html(df, outdir)

    # Résumé console
    print("\nTop résultats (triés par Test_F1_weighted):")
    print(df.sort_values("Test_F1_weighted", ascending=False).head(topn))
    print(f"\nFigures et tableaux HTML exportés dans: {outdir}")


if __name__ == "__main__":
    main()
