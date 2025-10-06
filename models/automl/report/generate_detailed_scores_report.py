import pandas as pd
import os
from pathlib import Path

def load_class_results(target_dir):
    """Charge les résultats par classe pour une cible donnée."""
    filepath = Path(target_dir) / 'class_results.csv'
    if not filepath.exists():
        print(f"Fichier non trouvé : {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    # Nettoyer les noms de colonnes
    df.columns = [col.strip() for col in df.columns]
    return df

def create_scores_report(target_name):
    """Crée un rapport détaillé des scores pour une cible."""
    base_dir = Path('/workspaces/datasciencetest_reco_plante/results/models/xgboost')
    target_dir = base_dir / f'nom_{target_name.lower()}'
    
    # Charger les données
    df = load_class_results(target_dir)
    if df is None or df.empty:
        return None
    
    # Créer un identifiant unique pour chaque configuration de pipeline
    df['Pipeline_Config'] = df['Pipeline'] + ' (' + df['Config'] + ')'
    
    # Sélectionner et réorganiser les colonnes
    df = df[['Pipeline_Config', 'Classe', 'Precision', 'Recall', 'F1_score', 'Support']]
    
    # Trier par F1_score décroissant
    df = df.sort_values(['Pipeline_Config', 'F1_score'], ascending=[True, False])
    
    # Formater les valeurs numériques
    for col in ['Precision', 'Recall', 'F1_score']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    # Générer le HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scores détaillés - {target_name}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .table {{ font-size: 14px; }}
            .table thead th {{ 
                background-color: #3498db; 
                color: white; 
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .pipeline-header {{
                background-color: #f2f9ff;
                font-weight: bold;
            }}
            .table-container {{
                max-height: 600px;
                overflow-y: auto;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Scores détaillés - Classification des {target_name}</h1>
            <p class="lead">Précision, Rappel et F1-score par classe et par configuration de pipeline</p>
            
            <div class="table-container">
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>Pipeline (Config)</th>
                            <th>Classe</th>
                            <th>Précision</th>
                            <th>Rappel</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Ajouter les lignes du tableau
    current_pipeline = None
    for _, row in df.iterrows():
        if row['Pipeline_Config'] != current_pipeline:
            current_pipeline = row['Pipeline_Config']
            html += f"""
                    <tr class="pipeline-header">
                        <td colspan="6">{current_pipeline}</td>
                    </tr>
            """
        
        html += f"""
                    <tr>
                        <td></td>
                        <td>{row['Classe']}</td>
                        <td>{row['Precision']}</td>
                        <td>{row['Recall']}</td>
                        <td><strong>{row['F1_score']}</strong></td>
                        <td>{int(float(row['Support']))}</td>
                    </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Sauvegarder le rapport
    output_file = target_dir / f'detailed_scores_{target_name.lower()}.html'
    with open(output_file, 'w') as f:
        f.write(html)
    
    return output_file

def create_index_report():
    """Crée une page d'index pour accéder à tous les rapports."""
    base_dir = Path('/workspaces/datasciencetest_reco_plante/results/models/xgboost')
    output_file = base_dir / 'index_scores_detailles.html'
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapports des Scores Détailés</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 40px; background-color: #f8f9fa; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .card-header {{ background-color: #3498db; color: white; }}
            .card-body {{ background-color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-5">Rapports des Scores Détailés</h1>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Classification des Maladies</h3>
                        </div>
                        <div class="card-body text-center">
                            <p>Scores détaillés pour chaque maladie et chaque configuration de pipeline</p>
                            <a href="nom_maladie/detailed_scores_maladies.html" class="btn btn-primary">Voir le rapport</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Classification des Espèces</h3>
                        </div>
                        <div class="card-body text-center">
                            <p>Scores détaillés pour chaque espèce et chaque configuration de pipeline</p>
                            <a href="nom_plante/detailed_scores_especes.html" class="btn btn-primary">Voir le rapport</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    return output_file

def main():
    # Créer les rapports pour chaque cible
    targets = ['Maladies', 'Especes']
    report_paths = []
    
    for target in targets:
        print(f"Traitement des données pour les {target}...")
        report_path = create_scores_report(target)
        if report_path:
            report_paths.append(report_path)
    
    # Créer la page d'index
    index_path = create_index_report()
    report_paths.append(index_path)
    
    # Afficher les liens vers les rapports
    print("\nRapports générés :")
    for path in report_paths:
        print(f"- {path}")
    
    print("\nAccès direct :")
    print(f"- Page d'accueil: http://localhost:8000/index_scores_detailles.html")
    for target in targets:
        print(f"- Rapport {target}: http://localhost:8000/nom_{target.lower()}/detailed_scores_{target.lower()}.html")

if __name__ == "__main__":
    main()
