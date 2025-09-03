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
import shutil

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


def generate_class_summary(class_results_path: Path, best_pipeline: str, best_config: str, outdir: Path) -> None:
    """Génère un récapitulatif HTML des performances par classe pour le meilleur modèle.
    
    Args:
        class_results_path: Chemin vers le fichier class_results.csv
        best_pipeline: Nom du meilleur pipeline (ex: "XGBoost + PCA")
        best_config: Nom de la meilleure configuration (ex: "Deep Trees")
        outdir: Dossier de sortie pour le fichier HTML
    """
    try:
        # Charger les résultats par classe
        class_df = pd.read_csv(class_results_path)
        
        # Filtrer pour le meilleur modèle
        best_df = class_df[
            (class_df['Pipeline'] == best_pipeline) & 
            (class_df['Config'] == best_config)
        ].copy()
        
        if best_df.empty:
            print(f"[WARN] Aucune donnée trouvée pour {best_pipeline} - {best_config}")
            return
            
        # Calculer la variance des prédictions
        # On utilise le coefficient de variation (écart-type/moyenne) pour normaliser
        # et permettre la comparaison entre classes
        metrics = ['Precision', 'Recall', 'F1_score']
        for metric in metrics:
            best_df[f'{metric}_cv'] = (best_df[metric].std() / best_df[metric].mean()) * 100
        
        # Trier par F1_score décroissant
        best_df = best_df.sort_values('F1_score', ascending=False)
        
        # Formater les valeurs pour l'affichage
        format_style = {
            'F1_score': '{:.1%}'.format,
            'Precision': '{:.1%}'.format,
            'Recall': '{:.1%}'.format,
            'Support': '{:,}'.format,
            'F1_score_cv': '{:.1f}%'.format,
            'Precision_cv': '{:.1f}%'.format,
            'Recall_cv': '{:.1f}%'.format
        }
        
        # Générer le HTML
        html = """
        <html>
        <head>
            <meta charset='utf-8'>
            <title>Détail par Classe - {pipeline} ({config})</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 24px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white;
                    position: sticky;
                    top: 0;
                }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e6f7ff; }}
                .good {{ color: #27ae60; }}
                .medium {{ color: #f39c12; }}
                .bad {{ color: #e74c3c; }}
                .summary {{ 
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Détail des performances par Classe</h1>
            <div class='summary'>
                <h2>{pipeline} - {config}</h2>
                <p>F1 Score moyen: <b>{mean_f1:.1%}</b> (médiane: {median_f1:.1%})</p>
                <p>Variation (CV) entre classes: <b>{cv_f1:.1f}%</b> (écart-type/moyenne)</p>
                <p>Nombre de classes: {num_classes}</p>
            </div>
            
            <h2>Métriques détaillées</h2>
            <div style='overflow-x:auto;'>
                {table}
            </div>
            
            <h2>Explications des colonnes</h2>
            <ul>
                <li><b>Classe</b>: Nom de la catégorie prédite (maladie/espèce).</li>
                <li><b>F1_score</b>: Moyenne harmonique entre précision et rappel (entre 0 et 1). Intérêt: équilibre performance globale par classe; plus élevé = mieux.</li>
                <li><b>Precision</b>: Parmi les images prédites dans cette classe, part de celles qui sont correctes. Intérêt: faible précision = trop de faux positifs.</li>
                <li><b>Recall</b>: Parmi les images réellement de cette classe, part correctement retrouvée. Intérêt: faible rappel = trop de faux négatifs.</li>
                <li><b>Support</b>: Nombre d'images de cette classe dans le test. Intérêt: aide à relativiser les scores (petit support = score plus instable).</li>
                <li><b>F1_score_cv</b>, <b>Precision_cv</b>, <b>Recall_cv</b>: Coefficient de variation (%) des métriques entre classes (écart-type/moyenne × 100). Intérêt: plus petit = résultats plus homogènes; grand = disparités entre classes.</li>
            </ul>
            <p>Les lignes sont triées par F1_score décroissant pour mettre en avant les classes les plus performantes.</p>
        </body>
        </html>
        """.format(
            pipeline=best_pipeline,
            config=best_config,
            mean_f1=best_df['F1_score'].mean(),
            median_f1=best_df['F1_score'].median(),
            cv_f1=(best_df['F1_score'].std() / best_df['F1_score'].mean()) * 100,
            num_classes=len(best_df),
            table=best_df[['Classe', 'F1_score', 'Precision', 'Recall', 'Support', 
                         'F1_score_cv', 'Precision_cv', 'Recall_cv']]
                     .to_html(classes='dataframe', index=False, float_format='{:.1%}'.format,
                             formatters=format_style)
        )
        
        # Écrire le fichier
        output_path = outdir / f"class_summary_{best_pipeline.lower().replace(' ', '_')}_{best_config.lower().replace(' ', '_')}.html"
        output_path.write_text(html, encoding='utf-8')
        print(f"Récapitulatif par classe généré: {output_path}")
        
    except Exception as e:
        print(f"[ERREUR] Impossible de générer le récapitulatif par classe: {e}")


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
        
        # Générer le récapitulatif par classe pour le meilleur modèle
        if not sorted_df.empty:
            best_run = sorted_df.iloc[0]
            class_results_path = outdir.parent / "class_results.csv"
            if class_results_path.exists():
                generate_class_summary(
                    class_results_path=class_results_path,
                    best_pipeline=best_run['Pipeline'],
                    best_config=best_run['Config'],
                    outdir=outdir
                )
            else:
                print(f"[INFO] Fichier des résultats par classe non trouvé: {class_results_path}")
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

    # Export matrices de confusion du meilleur modèle si disponibles
    try:
        if not df.empty:
            best = df.sort_values("Test_F1_weighted", ascending=False).iloc[0]
            pipeline = str(best.get("Pipeline", "")).strip()
            config = str(best.get("Config", "")).strip()
            if pipeline and config:
                parent_dir = outdir.parent  # e.g., results/models/xgboost
                title_prefix = f"{pipeline} ({config})"
                safe = title_prefix.replace(" ", "_")
                src_counts = parent_dir / f"confusion_matrix_{safe}.png"
                src_norm = parent_dir / f"confusion_matrix_normalized_{safe}.png"

                dst_counts = outdir / "best_confusion_counts.png"
                dst_norm = outdir / "best_confusion_normalized.png"

                copied_any = False
                if src_counts.exists():
                    shutil.copy2(src_counts, dst_counts)
                    copied_any = True
                if src_norm.exists():
                    shutil.copy2(src_norm, dst_norm)
                    copied_any = True

                # Générer une petite page HTML qui embarque les images si elles existent
                html_parts = [
                    "<html><head><meta charset='utf-8'><title>Matrices de confusion - Meilleur modèle</title>",
                    "<style>body{font-family:Arial,sans-serif;margin:24px} h1,h2{color:#2c3e50} img{max-width:100%;height:auto;border:1px solid #ddd;margin:12px 0}</style>",
                    "</head><body>",
                    f"<h1>Matrices de confusion - {title_prefix}</h1>",
                ]
                if dst_counts.exists():
                    html_parts += [
                        "<h2>Confusion (comptes)</h2>",
                        "<p>Nombre de prédictions par paire (vraie classe × classe prédite). Diagonale = bonnes prédictions.</p>",
                        f"<img src='best_confusion_counts.png' alt='Confusion brute - {title_prefix}'>",
                    ]
                if dst_norm.exists():
                    html_parts += [
                        "<h2>Confusion normalisée (%)</h2>",
                        "<p>Chaque ligne est normalisée (rappel par classe). Valeurs en %: plus la diagonale est proche de 100%, mieux c'est.</p>",
                        f"<img src='best_confusion_normalized.png' alt='Confusion normalisée - {title_prefix}'>",
                    ]
                if not copied_any:
                    html_parts += [
                        "<p style='color:#a00'>Aucune matrice trouvée. Lancez le script d'entraînement pour les générer: <code>python models/xgboost_plantvillage.py --quick</code></p>",
                    ]
                html_parts += ["</body></html>"]
                (outdir / "best_confusion.html").write_text("".join(html_parts), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Export des matrices de confusion ignoré: {e}")

    # Résumé console
    print("\nTop résultats (triés par Test_F1_weighted):")
    print(df.sort_values("Test_F1_weighted", ascending=False).head(topn))
    print(f"\nFigures et tableaux HTML exportés dans: {outdir}")


if __name__ == "__main__":
    main()
